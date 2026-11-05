"""Training state, config preservation, and atomic checkpoints.

Resume uses the saved recipe plus explicit Hydra overrides. Frozen encoders are
omitted from checkpoints and restored from pretrained weights. A mid-epoch save records
how many batches of its epoch ran, and resume trains only the remaining ones, which needs
no sampler cursor because the sampler draws with replacement.
Checkpoints hold only tensors and plain Python values, so they load with torch.load(weights_only=True).
Checkpoint format: model, optimizer, scheduler (if any), training_state, config (resolved), training_config (unresolved), overrides, epoch, train_global_iter
"""

import os
import re
import torch
import torch.distributed as dist
from dataclasses import dataclass, asdict, replace
from typing import Optional, Dict, Any
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging

from tapestry.models.model_loading import frozen_encoder_prefixes, load_checkpoint_cpu
from tapestry.utils.legacy_checkpoint import upgrade_legacy

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """What a resumed run needs besides the weights and the optimizer."""
    epoch: int = 0
    epoch_batches: int = 0   # batches of `epoch` already trained; nonzero only in a mid-epoch latest.pt
    train_global_iter: int = 0
    val_global_iter: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_val_loss: float = float('inf')
    best_val_action_future_frames_loss: float = float('inf')
    wandb_run_id: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        return cls(**data)

    def update(self, **kwargs) -> 'TrainingState':
        return replace(self, **kwargs)


def _config_nodes(data, path):
    """Walk unresolved config nodes, keeping slash-containing dictionary keys intact."""
    yield path, data
    if isinstance(data, dict):
        for key, value in data.items():
            yield from _config_nodes(value, (*path, str(key)))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from _config_nodes(value, (*path, str(i)))


class ConfigManager:
    """The launch config, or on resume the checkpoint's recipe with this launch's command-line overrides on top."""

    def __init__(self, base_config: DictConfig):
        if base_config['init_from'] is not None and (base_config['resume'] or base_config['checkpoint'] is not None):
            raise ValueError("init_from loads weights only; do not combine it with resume=true or checkpoint=...")
        self._base_config = base_config
        self._final_config = None
        self._original_overrides = []
        self._config_groups = set()

    def capture_overrides(self) -> None:
        """Record this launch's command-line overrides and config groups from Hydra."""
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        self._original_overrides = list(hydra_cfg.overrides.task)
        self._config_groups = set(hydra_cfg.runtime.choices)

    def merge_with_checkpoint(self, checkpoint_config: Dict[str, Any]) -> DictConfig:
        """Resume the saved recipe, applying only this invocation's explicit Hydra overrides."""
        from hydra.core.override_parser.overrides_parser import OverridesParser

        cfg = OmegaConf.create(checkpoint_config)
        saved = OmegaConf.to_container(cfg, resolve=True)
        recipe = OmegaConf.to_container(cfg, resolve=False)
        overrides = OverridesParser.create().parse_overrides(self._original_overrides)
        for override in overrides:
            if override.is_sweep_override():
                raise ValueError("Resume expects one concrete value per override")
            key = override.key_or_group
            if override.package is not None or key in self._config_groups or (
                '.' not in key and key not in self._base_config and key not in cfg
            ):
                raise ValueError(f"Config-group overrides are not supported during resume: {override.input_line}; use init_from")
            if override.is_delete():
                parts = key.split('.')
                parent = OmegaConf.select(cfg, '.'.join(parts[:-1])) if len(parts) > 1 else cfg
                leaf = parts[len(parts) - 1]
                if override.value() is not None and parent[leaf] != override.value():
                    raise ValueError(f"Delete override does not match the saved value: {override.input_line}")
                del parent[leaf]
            else:
                value = override.value()
                if override.is_add() and OmegaConf.select(cfg, key) is not None and not isinstance(value, (dict, list)):
                    raise ValueError(f"Append override already exists in the saved recipe: {key}; use ++ to replace it")
                # Parsed values retain dictionary deltas and resolve references in the saved recipe.
                OmegaConf.update(cfg, key, value, merge=True, force_add=True)
            logger.info(f"Applied resume override: {override.input_line}")

        # These describe this invocation, not how the original run was initialized.
        for key in ('resume', 'checkpoint', 'init_from'):
            cfg[key] = self._base_config[key]
        self._validate_resume(cfg, saved, recipe)
        self._final_config = cfg
        return cfg

    def _validate_resume(self, cfg, saved, recipe):
        current = OmegaConf.to_container(cfg, resolve=True)
        old_nodes = dict(_config_nodes(saved, ()))
        new_nodes = dict(_config_nodes(current, ()))
        recipe_nodes = dict(_config_nodes(recipe, ()))

        # Restoring AdamW/scheduler state would otherwise silently undo these changes.
        keys = ('learning_rate', 'weight_decay', 'pretrained_learning_rate_multiplier', 'pretrained_decoder_key')
        for key in set(saved['training']) | set(current['training']):
            path = ('training', key)
            if key in keys or key.startswith('lr_scheduler'):
                if path not in old_nodes or path not in new_nodes or old_nodes[path] != new_nodes[path]:
                    raise ValueError(f"Cannot change training.{key} while restoring optimizer/scheduler state; use init_from")

        # Older checkpoints lost their aliases. Refuse a change that would leave one stale.
        for path, value in _config_nodes(OmegaConf.to_container(self._base_config, resolve=False), ()):
            if not isinstance(value, str) or path not in recipe_nodes:
                continue
            old = recipe_nodes[path]
            if isinstance(old, str) and '${' in old:
                continue  # This checkpoint retained the dependency itself.
            for ref in re.findall(r'\$\{([A-Za-z_][\w./]*)\}', value):
                if ref == 'experiment_name':
                    continue  # A deliberate new logging name does not change model/data semantics.
                source = tuple(ref.split('.'))
                if source in old_nodes and (source not in new_nodes or old_nodes[source] != new_nodes[source]):
                    if path in new_nodes and old_nodes[path] == new_nodes[path]:
                        raise ValueError(
                            f"Checkpoint has no saved interpolation for {'.'.join(path)} after changing {ref}; "
                            "override the dependent field explicitly or use init_from"
                        )

    @property
    def config(self) -> DictConfig:
        """Hydra has already applied overrides to a fresh run's base config."""
        if self._final_config is None:
            self._final_config = self._base_config.copy()
        return self._final_config

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """The config entries of a checkpoint."""
        return {
            'config': OmegaConf.to_container(self.config, resolve=True),
            'training_config': OmegaConf.to_container(self.config, resolve=False),
            'overrides': self._original_overrides
        }


class CheckpointManager:
    """Checkpoint directory selection, atomic saves and strict restores."""

    def __init__(self, experiment_name: str, checkpoint_dir: Optional[str], resume: bool):
        self.experiment_name = experiment_name
        self.resume = resume
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None   # None: picked on first access

    @property
    def checkpoint_dir(self) -> Path:
        """Rank 0 picks the directory and broadcasts it, so after distributed init the first access must happen on every rank."""
        if self._checkpoint_dir is None:
            distributed = dist.is_available() and dist.is_initialized()
            picked = [str(self._pick_dir()) if not distributed or dist.get_rank() == 0 else None]
            if distributed:
                dist.broadcast_object_list(picked, src=0)
            self._checkpoint_dir = Path(picked[0])
        return self._checkpoint_dir

    def _pick_dir(self) -> Path:
        """Fresh runs take the first free <name>[_n]; resume takes the highest-numbered directory that find_latest can resume from."""
        base = Path("checkpoints")
        if not self.resume:
            n = 0
            while (path := base / (self.experiment_name if n == 0 else f"{self.experiment_name}_{n}")).exists():
                n += 1
            path.mkdir(parents=True)
            return path
        dirs = {0: base / self.experiment_name}
        for path in base.glob(f"{self.experiment_name}_*"):
            suffix = path.name[len(self.experiment_name) + 1:]
            if suffix.isdigit():
                dirs[int(suffix)] = path
        with_ckpt = [n for n, path in dirs.items() if (path / "latest.pt").exists() or any(path.glob("epoch_*.pt"))]
        if not with_ckpt:
            raise FileNotFoundError(f"resume=true but no checkpoint found for experiment '{self.experiment_name}' in {base}/")
        return dirs[max(with_ckpt)]

    def save(self, model, optimizer, scheduler, state, config_data, additional_metrics, filename) -> Path:
        """Atomic write of <checkpoint_dir>/<filename>. scheduler and additional_metrics may be None."""
        path = self.checkpoint_dir / filename
        # Frozen encoders come from pretrained weights, so omit them to save disk space and I/O.
        frozen = tuple(frozen_encoder_prefixes(model))
        model_state = {k: v for k, v in self._clean_state_dict(model.state_dict()).items() if not k.startswith(frozen)}
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer.state_dict(),
            'training_state': state.to_dict(),
            **config_data,
            # Read by scripts/prepare_release_checkpoint.py.
            'epoch': state.epoch,
            'train_global_iter': state.train_global_iter,
        }
        if additional_metrics is not None:
            checkpoint.update(additional_metrics)
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()

        # Atomic write: a kill mid-save must not corrupt an existing checkpoint.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load(self, path, model, optimizer, scheduler) -> TrainingState:
        """Restore weights/optimizer/scheduler strictly except for omitted frozen encoders; use init_from for weights only."""
        checkpoint = upgrade_legacy(torch.load(path, map_location="cpu", weights_only=True))
        result = model.load_state_dict(self._clean_state_dict(checkpoint["model"]), strict=False)
        frozen = tuple(frozen_encoder_prefixes(model))
        missing = [k for k in result.missing_keys if not k.startswith(frozen)]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"{path} does not match the model: missing={missing}, unexpected={list(result.unexpected_keys)}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        return TrainingState.from_dict(checkpoint["training_state"])

    def load_config_only(self, path: Path) -> Dict[str, Any]:
        """The recipe (unresolved config) of a trusted training checkpoint."""
        return load_checkpoint_cpu(path)['training_config']

    def find_latest(self) -> Path:
        """latest.pt, else the highest epoch_<n>.pt."""
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists():
            return latest
        epochs = {int(p.stem[6:]): p for p in self.checkpoint_dir.glob("epoch_*.pt") if p.stem[6:].isdigit()}
        if not epochs:
            raise FileNotFoundError(f"no latest.pt or epoch_<n>.pt in {self.checkpoint_dir}")
        return epochs[max(epochs)]

    @staticmethod
    def _clean_state_dict(state_dict):
        """Strip the DDP (`module.`) and torch.compile (`_orig_mod.`) prefixes."""
        return {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state_dict.items()}


class ExperimentTracker:
    """Glue between the config (ConfigManager), the checkpoints (CheckpointManager) and the TrainingState of one run."""

    def __init__(self, config: DictConfig):
        self.experiment_name = config['experiment_name']
        self.is_resume = config['resume']
        self.checkpoint_path = config['checkpoint']
        self.config_manager = ConfigManager(config)
        self.config_manager.capture_overrides()
        self.checkpoint_manager = CheckpointManager(self.experiment_name, None, self.is_resume)
        self.state = TrainingState()
        if self.is_resume or self.checkpoint_path is not None:   # the model must be built from the checkpoint's recipe
            path = Path(self.checkpoint_path) if self.checkpoint_path is not None else self.checkpoint_manager.find_latest()
            self.checkpoint_path = str(path)
            self.config_manager.merge_with_checkpoint(self.checkpoint_manager.load_config_only(path))
            saved_name = self.config_manager.config['experiment_name']
            if saved_name != self.experiment_name:
                raise ValueError(
                    f"{path} was saved by experiment '{saved_name}' but this run is '{self.experiment_name}'; "
                    "pass experiment_name=... explicitly (the saved name to continue it, a new name to fork it)"
                )
            logger.info(f"Loaded checkpoint config from {path}")

    def setup_training(self, model, optimizer, scheduler) -> TrainingState:
        """On resume: restore weights, optimizer, scheduler and state from the checkpoint picked in __init__."""
        if self.checkpoint_path is not None:
            self.state = self.checkpoint_manager.load(Path(self.checkpoint_path), model, optimizer, scheduler)
            logger.info(f"Resumed from epoch {self.state.epoch}")
        return self.state

    def get_merged_config(self) -> DictConfig:
        """The launch config, or on resume the checkpoint's recipe with this launch's overrides on top."""
        return self.config_manager.config

    def save(self, model, optimizer, scheduler, additional_metrics, filename) -> Path:
        return self.checkpoint_manager.save(model, optimizer, scheduler, self.state,
                                            self.config_manager.get_checkpoint_data(), additional_metrics, filename)

    def update_state(self, **kwargs) -> None:
        self.state = self.state.update(**kwargs)

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_manager.checkpoint_dir


# Parameters whose shape depends on the action dimension. They are re-initialized (not loaded)
# when init_from points at a checkpoint trained with a different action space.
ACTION_DIM_DEPENDENT = ("action_input_proj.act_proj",)


def _canonical_key(key: str) -> str:
    """Strip PEFT wrapping so LoRA and full-finetune checkpoints share one key space."""
    return key.replace("base_model.model.", "").replace("base_layer.", "")


def load_init_weights(model: torch.nn.Module, path: str) -> Dict[str, int]:
    """
    Initialize `model` from the weights of a checkpoint (no optimizer/epoch state).

    Works across LoRA <-> full-finetune wrappings and across action spaces:
      - keys are matched after stripping `base_model.model.` / `base_layer.` (PEFT wrapping)
      - action-dim dependent heads (ACTION_DIM_DEPENDENT) with a different shape stay freshly
        initialized; any other shape mismatch is an error
      - missing keys are only allowed for frozen encoders (never stored) and fresh LoRA adapters
      - a checkpoint with LoRA adapters cannot initialize a model built without use_lora
    """
    ck = load_checkpoint_cpu(path)
    ck_sd = CheckpointManager._clean_state_dict(ck["model"])

    canon_to_model = {}
    for k in model.state_dict():
        c = _canonical_key(k)
        if c in canon_to_model:
            raise RuntimeError(f"Ambiguous model keys {canon_to_model[c]} / {k} after PEFT stripping")
        canon_to_model[c] = k
    model_has_lora = any("lora_" in k for k in canon_to_model.values())

    model_sd = model.state_dict()
    mapped, n_fresh_heads = {}, 0
    for k, v in ck_sd.items():
        c = _canonical_key(k)
        if c not in canon_to_model:
            if "lora_" in k and not model_has_lora:
                raise RuntimeError(
                    f"{path} contains LoRA adapters but the model was built without use_lora; "
                    "set decoders.cogvideox.use_lora=true or use a merged checkpoint"
                )
            raise RuntimeError(f"Unexpected key in init_from checkpoint: {k}")
        mk = canon_to_model[c]
        if v.shape != model_sd[mk].shape:
            if any(tag in c for tag in ACTION_DIM_DEPENDENT):
                n_fresh_heads += 1
                continue
            raise RuntimeError(f"Shape mismatch for {k}: checkpoint {tuple(v.shape)} vs model {tuple(model_sd[mk].shape)}")
        mapped[mk] = v

    result = model.load_state_dict(mapped, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {result.unexpected_keys}")
    frozen_prefixes = frozen_encoder_prefixes(model)
    n_frozen = n_fresh_lora = 0
    for k in result.missing_keys:
        if any(k.startswith(p) for p in frozen_prefixes):
            n_frozen += 1
        elif "lora_" in k:
            n_fresh_lora += 1
        elif any(tag in _canonical_key(k) for tag in ACTION_DIM_DEPENDENT):
            continue  # counted in n_fresh_heads
        else:
            raise RuntimeError(f"Key missing from init_from checkpoint: {k}")
    return {
        "loaded": len(mapped),
        "fresh_action_heads": n_fresh_heads,
        "fresh_lora": n_fresh_lora,
        "frozen_encoder_skipped": n_frozen,
    }
