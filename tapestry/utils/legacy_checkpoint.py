"""Upgrade of training checkpoints written before the 2026-09-27 release pass, applied in memory when they load.

Those checkpoints hold training_state.additional_state (always empty), five parameters that no forward pass read (DEAD)
and config keys that the current code no longer reads because it implements one fixed value for each. The dead
parameters are dropped from the weights and the optimizer. A removed config key with any other value raises, since the
current code cannot continue that run, and so does a learning-rate scheduler, which the older code stepped once per GPU.
The older model also replaced each decoder's max_text_seq_length with encoders.text.max_seq_len when it was built, so
the upgraded recipe links the two as the current recipes do. Checkpoints from the 2026-08-22 code also lack
training_config, so their resolved config serves as the recipe. That code paired some jointly denoised samples with
another row's targets in batches that mixed action and video-only data, so these runs continue with the corrected
pairing. Checkpoints from July 2026 pickle OmegaConf objects and do not load with torch.load(weights_only=True).
"""
import copy
import logging
import re

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Removed keys and the value the current code implements. '*' matches every entry of a mapping.
REMOVED = {
    "model.d_model": 4096,
    "noise_scheduler": "cogvideox_ddim",
    "encoders.image": None,
    "encoders.video": None,
    "encoders.proprioception": None,
    "encoders.text.frozen": True,
    "encoders.text.return_all_tokens": True,
    "encoders.text.size": "xxl",
    "decoders.*.action_only_layers": 0,
    "decoders.*.action_tokenizer": "patch_shared",
    "decoders.*.attention_head_dim": 64,
    "decoders.*.bypass_text_projection": True,
    "decoders.*.dropout_first_frame_prob": 0.0,
    "decoders.*.dual_conditioning": False,
    "decoders.*.enable_encoder_hidden_states_grad": False,
    "decoders.*.frame_size": 16,
    "decoders.*.in_channels": 16,
    "decoders.*.joint_num_register_tokens": 0,
    "decoders.*.latent_dim": 4096,
    "decoders.*.num_attention_heads": 30,
    "decoders.*.num_frames": 7,
    "decoders.*.num_layers": 30,
    "decoders.*.out_channels": 16,
    "decoders.*.patch_size": 2,
    "decoders.*.prepend_first_frame_latents": True,
    "decoders.*.proj_out_action_head": "linear",
    "decoders.*.registers_for_action_only": False,
    "decoders.*.text_condition_dim": 4096,
    "decoders.*.text_embed_dim": 4096,
    "decoders.*.torch_dtype": "float16",
    "decoders.*.use_action_conditioning": False,
    "decoders.*.use_cogvideox_vae": True,
    "decoders.*.use_rotary_positional_embeddings": False,
    "decoders.*.use_safetensors": True,
    "decoders.*.use_text_conditioning": False,
    "decoders.*.use_view_embeddings": True,
    "decoders.*.vae_model_name": "THUDM/CogVideoX-2b",
    "sequence.num_timesteps": 0,
    "sequence.suffix_order": [],
    "sequence.timestep_order": [],
    "training.device": "cuda",
    "training.log_interval": 100,
    "training.use_deterministic_sampler": False,
    "training.view_dropout_prob": 0.0,
    "training.dataset.keys_to_load": ["images", "actions", "text", "proprioception"],
    "training.dataset.use_cached_index_map": False,
    "training.dataset.datasets.*.prefer_mp4": True,
    "training.dataset.datasets.*.use_cached_index_map": False,
    "training.dataset.datasets.*.video_diffusion_view": "joint",
    "training.dataset.val_datasets.*.prefer_mp4": True,
    "training.dataset.val_datasets.*.use_cached_index_map": False,
    "training.dataset.val_datasets.*.video_diffusion_view": "joint",
    "training.objectives.*.noise_scheduler": "cogvideox_ddim",
    "training.objectives.*.num_inference_steps": 50,
    "training.objectives.*.reduction": "mean",
    "training.objectives.*.timestep_sampler": "uniform",
    "training.objectives.*.timestep_sampler_cfg": None,
    "training.objectives.*.training_method": "diffusion",
    "training.objectives.*.type": "diffusion",
    "training.wandb.num_video_samples": 2,
    "training.wandb.video_sampling_method": "cogvideox_ddim",
}
# Removed copies of another config value, which they must equal.
ALIASES = {
    "training.dataset.image_size": "model.image_size",
    "training.dataset.true_horizon": "training.true_horizon",
    "training.objectives.*.action_horizon": "training.true_horizon",
}
# Run names and output directories; the current code derives them from experiment_name.
DROPPED = ("training.dataset.name", "training.wandb.name", "training.wandb.tags", "logging.save_dir", "logging.log_dir")
# Decoder arguments that the older recipes left at these constructor defaults.
ADDED = {"max_views": 3, "text_cond_dropout_prob": 0.0, "action_denoising_dropout": 0.0, "joint_asymmetric_action_attn": False}
# Parameters of the older model that no forward pass read. The current model does not have them.
DEAD = re.compile(r"(^|\.)(view_mask_token|proj_out_action\.(weight|bias)|action_text_proj\.(weight|bias))$")


def _matches(cfg, path):
    """(parent, key) of every node at a dotted path."""
    head, _, rest = path.partition(".")
    for key in (list(cfg) if head == "*" else [head] if head in cfg else []):
        if not rest:
            yield cfg, key
        elif isinstance(cfg[key], dict):
            yield from _matches(cfg[key], rest)


def _get(cfg, path):
    for key in path.split("."):
        cfg = cfg[key]
    return cfg


def _check(cfg):
    """Raise unless every removed key of the resolved config holds the value the current code implements."""
    for path, want in REMOVED.items():
        for parent, key in _matches(cfg, path):
            if parent[key] != want:
                raise ValueError(f"Checkpoint has {path}={parent[key]!r}; the current code implements only {want!r}")
    for path, source in ALIASES.items():
        want = _get(cfg, source)
        for parent, key in _matches(cfg, path):
            if parent[key] != want:
                raise ValueError(f"Checkpoint has {path}={parent[key]!r} but {source}={want!r}")
    if cfg["training"]["lr_scheduler"] is not None:
        raise ValueError(f"Checkpoint has training.lr_scheduler={cfg['training']['lr_scheduler']!r}; the older code "
                         "stepped it once per GPU at the end of each epoch, and the current code steps it once")


def _strip(cfg):
    """Drop the removed keys, add the decoder defaults and drop proprioception, which no encoder read."""
    for path in (*REMOVED, *ALIASES, *DROPPED):
        for parent, key in list(_matches(cfg, path)):
            del parent[key]
    for decoder in cfg["decoders"].values():
        for key, value in ADDED.items():
            decoder.setdefault(key, value)
    for split in ("datasets", "val_datasets"):
        for entry in cfg["training"]["dataset"][split].values():
            entry["active_modalities"] = [m for m in entry["active_modalities"] if m != "proprioception"]


def _drop_dead(ck):
    """Drop the dead parameters from the weights and the AdamW parameter ids. torch.optim pairs the saved ids with the
    optimizer's parameters by position, so the AdamW state of every other parameter stays with it."""
    sd, opt = ck["model"], ck["optimizer"]
    # The older model's parameter order: its own parameters, its frozen encoders (omitted from the file), its decoders.
    # A tensor saved under two names (patch_embed.proj and its action alias) is one parameter.
    names, seen = [], set()
    for k, v in sd.items():
        if v.data_ptr() not in seen:
            seen.add(v.data_ptr())
            names.append(k)
    n_frozen = sum(len(g["params"]) for g in opt["param_groups"]) - len(names)
    order = [k for k in names if "." not in k] + [None] * n_frozen + [k for k in names if "." in k]
    for i, s in opt["state"].items():
        if order[i] is None or s["exp_avg"].shape != sd[order[i]].shape:
            raise ValueError(f"The AdamW state of parameter {i} does not match the saved weights")
        if DEAD.search(order[i]):
            raise ValueError(f"{order[i]} has AdamW state, but the current model has no such parameter")
    dead = {i for i, k in enumerate(order) if k is not None and DEAD.search(k)}
    for group in opt["param_groups"]:
        group["params"] = [i for i in group["params"] if i not in dead]
    for k in [k for k in sd if DEAD.search(k)]:
        del sd[k]


def _align_text_len(ck):
    """Link each decoder's max_text_seq_length to encoders.text.max_seq_len, which the older model used in its place."""
    text_len = ck["config"]["encoders"]["text"]["max_seq_len"]
    for cfg, value in ((ck["config"], text_len), (ck["training_config"], "${encoders.text.max_seq_len}")):
        for decoder in cfg["decoders"].values():
            decoder["max_text_seq_length"] = value


def upgrade_legacy(ck):
    """ck in the current format, upgraded in place if it is an older training checkpoint (see the module docstring)."""
    if "training_state" not in ck or "additional_state" not in ck["training_state"]:
        return ck
    if ck["training_state"].pop("additional_state"):
        raise ValueError("Checkpoint has a non-empty training_state.additional_state, which the current code cannot restore")
    _drop_dead(ck)
    if "training_config" not in ck:
        ck["training_config"] = copy.deepcopy(ck["config"])
    _check(ck["config"])
    for cfg in (ck["config"], ck["training_config"]):
        _strip(cfg)
    _align_text_len(ck)
    if OmegaConf.to_container(OmegaConf.create(ck["training_config"]), resolve=True) != ck["config"]:
        raise RuntimeError("The upgraded recipe does not resolve to the upgraded config")
    logger.info("Upgraded a training checkpoint written before 2026-09-27 to the current format")
    return ck
