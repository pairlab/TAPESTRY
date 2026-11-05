"""Training checkpoints written before the 2026-09-27 release pass upgrade to the current format when they load."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tapestry.models.model_loading import load_checkpoint_cpu
from tapestry.utils.legacy_checkpoint import ADDED, ALIASES, DROPPED, REMOVED, upgrade_legacy
from tapestry.utils.training_manager import ConfigManager, ExperimentTracker, TrainingState, load_init_weights

CFG = str(Path(__file__).parent.parent / "cfg")
RECIPES = ("train_libero", "overfit_libero", "finetune", "train_tapestry")


class Model(torch.nn.Module):
    """Tapestry's parameter layout in miniature: a frozen encoder, which checkpoints omit, and a decoder that saves its
    input projection under two names. dead=True adds the five parameters of the older model that no forward pass read."""
    def __init__(self, dead):
        super().__init__()
        if dead:
            self.view_mask_token = torch.nn.Parameter(torch.randn(8))
        text = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.LayerNorm(3))
        self.encoders = torch.nn.ModuleDict({"text": text.requires_grad_(False)})
        dec = torch.nn.Module()
        dec.latent_view_mask = torch.nn.Parameter(torch.randn(10))
        dec.patch_embed = torch.nn.Module()
        dec.patch_embed.proj = torch.nn.Linear(4, 5)
        dec.action_input_proj = torch.nn.Module()
        dec.action_input_proj.proj = dec.patch_embed.proj
        dec.action_input_proj.act_proj = torch.nn.Linear(6, 5, bias=False)
        if dead:
            dec.proj_out_action = torch.nn.Linear(5, 7)
        dec.norm_out = torch.nn.LayerNorm(5)
        if dead:
            dec.action_text_proj = torch.nn.Linear(3, 9)
        dec.proj_out = torch.nn.Linear(5, 4)
        dec.vae = torch.nn.Linear(4, 4).requires_grad_(False)
        self.decoders = torch.nn.ModuleDict({"cogvideox": dec})


def random_grads(seed):
    """Gradients of the trained parameters by name. norm_out.bias gets none, like one parameter of the real model."""
    gen = torch.Generator().manual_seed(seed)
    return {name: torch.randn(p.shape, generator=gen) for name, p in Model(dead=False).named_parameters()
            if p.requires_grad and not name.endswith("norm_out.bias")}


def step(model, opt, grads):
    params = dict(model.named_parameters())
    for name, grad in grads.items():
        params[name].grad = grad.clone()
    opt.step()
    opt.zero_grad()


def trained_old():
    """The older model after one step of its AdamW optimizer, which held every parameter."""
    model = Model(dead=True)
    opt = torch.optim.AdamW(model.parameters())
    step(model, opt, random_grads(0))
    return model, opt


def compose_recipe(name, overrides):
    with initialize_config_dir(config_dir=CFG, version_base=None):
        return compose(config_name=name, overrides=overrides)


def resolved(cfg):
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)


def put(cfg, path, value):
    """Set the value at a dotted path, creating missing keys; '*' sets it in every entry of a mapping."""
    head, _, rest = path.partition(".")
    for key in (list(cfg) if head == "*" else [head]):
        if rest:
            put(cfg.setdefault(key, {}), rest, value)
        else:
            cfg[key] = deepcopy(value)


def legacy(recipe):
    """A current recipe as the older code saved it."""
    old = deepcopy(recipe)
    for path, value in REMOVED.items():
        put(old, path, value)
    for path, source in ALIASES.items():
        put(old, path, "${" + source + "}")
    for path in DROPPED:
        put(old, path, "old_run")
    for decoder in old["decoders"].values():
        for key, value in ADDED.items():
            if decoder[key] == value:
                del decoder[key]
        decoder["max_text_seq_length"] = 226   # what the older recipes said; the older model used the T5 length
    for split in ("datasets", "val_datasets"):
        for entry in old["training"]["dataset"][split].values():
            entry["active_modalities"] = [*entry["active_modalities"], "proprioception"]
    return old


def legacy_checkpoint(recipe, with_training_config, model, opt):
    """An older checkpoint of model and opt; the 2026-08-22 code saved no training_config."""
    old = legacy(recipe)
    ck = {"model": {k: v for k, v in model.state_dict().items() if not k.startswith("encoders.")},
          "optimizer": opt.state_dict(), "overrides": [], "config": resolved(old),
          "training_state": {**TrainingState(epoch=3).to_dict(), "additional_state": {}}}
    if with_training_config:
        ck["training_config"] = old
    return ck


@pytest.mark.parametrize("name", RECIPES)
def test_legacy_recipe_upgrades_to_the_current_recipe(name):
    recipe = OmegaConf.to_container(compose_recipe(name, []), resolve=False)
    ck = upgrade_legacy(legacy_checkpoint(recipe, True, *trained_old()))
    assert ck["training_config"] == recipe
    assert ck["config"] == resolved(recipe)
    assert TrainingState.from_dict(ck["training_state"]) == TrainingState(epoch=3)


@pytest.mark.parametrize("name", RECIPES)
def test_resolved_only_checkpoint_uses_its_config_as_the_recipe(name):
    recipe = OmegaConf.to_container(compose_recipe(name, []), resolve=False)
    ck = upgrade_legacy(legacy_checkpoint(recipe, False, *trained_old()))
    assert resolved(ck["training_config"]) == ck["config"] == resolved(recipe)


@pytest.mark.parametrize("path, value, match", [
    ("decoders.cogvideox.use_rotary_positional_embeddings", True, "implements only False"),
    ("training.objectives.video/future_frames.timestep_sampler", "logit_normal", "implements only 'uniform'"),
    ("training.dataset.image_size", 64, r"but model\.image_size="),
    ("training.view_dropout_prob", 0.3, "implements only 0.0"),
    ("training.lr_scheduler", "cosine_annealing", "once per GPU"),
])
def test_legacy_value_the_current_code_cannot_reproduce_raises(path, value, match):
    ck = legacy_checkpoint(OmegaConf.to_container(compose_recipe("train_libero", []), resolve=False), True, *trained_old())
    put(ck["config"], path, value)
    put(ck["training_config"], path, value)
    with pytest.raises(ValueError, match=match):
        upgrade_legacy(ck)


def test_non_empty_additional_state_raises():
    ck = legacy_checkpoint(OmegaConf.to_container(compose_recipe("train_libero", []), resolve=False), True, *trained_old())
    ck["training_state"]["additional_state"] = {"ema": 1}
    with pytest.raises(ValueError, match="additional_state"):
        upgrade_legacy(ck)


def test_current_and_release_checkpoints_are_left_alone():
    cfg = OmegaConf.to_container(compose_recipe("train_libero", []), resolve=False)
    current = {"config": resolved(cfg), "training_config": cfg, "training_state": TrainingState(epoch=3).to_dict()}
    release = {"config": resolved(cfg), "model": {}}
    for ck in (current, release):
        before = deepcopy(ck)
        assert upgrade_legacy(ck) is ck and ck == before


@pytest.mark.parametrize("name, shape, match", [
    ("view_mask_token", (8,), "view_mask_token has AdamW state"),
    ("encoders.text.0.weight", (3, 2), "does not match the saved weights"),
    ("decoders.cogvideox.patch_embed.proj.weight", (5, 6), "does not match the saved weights"),
])
def test_adamw_state_the_older_model_cannot_have_raises(name, shape, match):
    """A dead parameter, a frozen encoder weight and the [5, 4] patch_embed.proj.weight."""
    old, opt = trained_old()
    ck = legacy_checkpoint(OmegaConf.to_container(compose_recipe("train_libero", []), resolve=False), True, old, opt)
    index = [n for n, _ in old.named_parameters()].index(name)
    ck["optimizer"]["state"][index] = {"step": torch.tensor(1.0), "exp_avg": torch.zeros(shape), "exp_avg_sq": torch.zeros(shape)}
    with pytest.raises(ValueError, match=match):
        upgrade_legacy(ck)


@pytest.mark.parametrize("with_training_config", [True, False])
def test_legacy_checkpoint_resumes_where_it_stopped(tmp_path, monkeypatch, with_training_config):
    old, old_opt = trained_old()
    recipe = OmegaConf.to_container(compose_recipe("train_libero", []), resolve=False)
    path = tmp_path / "latest.pt"
    torch.save(legacy_checkpoint(recipe, with_training_config, old, old_opt), path)
    overrides = ["resume=true", f"checkpoint={path}"]
    monkeypatch.setattr(ConfigManager, "capture_overrides",
                        lambda manager: setattr(manager, "_original_overrides", overrides))
    tracker = ExperimentTracker(compose_recipe("train_libero", overrides))
    fresh = compose_recipe("train_libero", overrides)
    merged = tracker.get_merged_config()
    assert OmegaConf.to_container(merged, resolve=True) == OmegaConf.to_container(fresh, resolve=True)
    if with_training_config:
        assert OmegaConf.to_container(merged, resolve=False) == OmegaConf.to_container(fresh, resolve=False)

    new = Model(dead=False)
    new_opt = torch.optim.AdamW(new.parameters())
    assert tracker.setup_training(new, new_opt, None) == TrainingState(epoch=3)
    old_params = dict(old.named_parameters())
    for name, p in new.named_parameters():
        old_p = old_params[name]
        assert (p in new_opt.state) == (old_p in old_opt.state), name
        if p in new_opt.state:
            s, old_s = new_opt.state[p], old_opt.state[old_p]
            assert s.keys() == old_s.keys() and all(torch.equal(s[k], old_s[k]) for k in s), name
    # The next step matches the run that never stopped. Frozen encoders come from pretrained weights.
    grads = random_grads(1)
    step(old, old_opt, grads)
    step(new, new_opt, grads)
    for name, p in new.named_parameters():
        assert name.startswith("encoders.") or torch.equal(p, old_params[name]), name
    assert load_checkpoint_cpu(path)["config"] == resolved(recipe)


def test_legacy_checkpoint_initializes_a_model(tmp_path):
    old, opt = trained_old()
    path = tmp_path / "epoch_7.pt"
    torch.save(legacy_checkpoint(OmegaConf.to_container(compose_recipe("train_libero", []), resolve=False), True, old, opt), path)
    new = Model(dead=False)
    load_init_weights(new, str(path))
    old_params = dict(old.named_parameters())
    for name, p in new.named_parameters():
        assert name.startswith("encoders.") or torch.equal(p, old_params[name]), name
