from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tapestry.utils.training_manager import CheckpointManager, ConfigManager, ExperimentTracker, TrainingState


CFG = str(Path(__file__).parent.parent / "cfg")


class Tiny(torch.nn.Linear):
    """A model with the `encoders` container that checkpointing expects."""
    def __init__(self):
        super().__init__(2, 1)
        self.encoders = torch.nn.ModuleDict()


def recipe(overrides):
    with initialize_config_dir(config_dir=CFG, version_base=None):
        return compose(config_name="train_libero", overrides=overrides)


def test_fresh_config_is_already_composed_by_hydra():
    overrides = ["~training.dataset.datasets.video_only", "+training.extra={x:1,y:[2,3]}"]
    cfg = recipe(overrides)
    manager = ConfigManager(cfg)
    manager._original_overrides = overrides
    assert manager.config == cfg
    assert "video_only" not in manager.config.training.dataset.datasets
    assert "+training" not in manager.config


def test_resume_preserves_saved_recipe_and_honors_explicit_overrides():
    saved = recipe([
        "training.batch_size=8", "~training.dataset.datasets.video_only",
        "training.dataset.datasets.action_video.dataset_names=[libero_object_t0]",
    ])
    saved = OmegaConf.to_container(saved, resolve=True)
    before = deepcopy(saved)
    overrides = ["resume=true", "training.num_epochs=12", "~training.dataset.val_datasets.video_only_val",
                 "+training.extra={x:1,y:[2,3]}"]
    manager = ConfigManager(recipe(overrides))
    manager._original_overrides = overrides
    cfg = manager.merge_with_checkpoint(saved)
    assert cfg.training.batch_size == 8
    assert cfg.training.dataset.datasets.action_video.dataset_names == ["libero_object_t0"]
    assert "video_only" not in cfg.training.dataset.datasets
    assert "video_only_val" not in cfg.training.dataset.val_datasets
    assert cfg.training.num_epochs == 12
    assert cfg.training.extra == {"x": 1, "y": [2, 3]}
    assert cfg.resume is True
    assert saved == before


def test_resume_replaces_explicit_nested_mapping_instead_of_keeping_stale_children():
    saved = recipe([])
    overrides = ["training.dataset.datasets.action_video.dataset_names=[libero_object]"]
    manager = ConfigManager(recipe(overrides))
    manager._original_overrides = overrides
    cfg = manager.merge_with_checkpoint(OmegaConf.to_container(saved, resolve=True))
    assert cfg.training.dataset.datasets.action_video.dataset_names == ["libero_object"]


def resumed(saved, overrides):
    manager = ConfigManager(recipe(overrides))
    manager._original_overrides = overrides
    return manager.merge_with_checkpoint(saved)


def test_resume_resets_saved_invocation_controls():
    saved = recipe(['init_from=/initial.pt'])
    saved.resume = True
    saved.checkpoint = '/stale.pt'
    cfg = resumed(OmegaConf.to_container(saved, resolve=True), ['resume=true'])
    assert cfg.resume is True
    assert cfg.checkpoint is None
    assert cfg.init_from is None


@pytest.mark.parametrize('control', ['resume=true', 'checkpoint=/resume.pt'])
def test_conflicting_init_and_resume_fail_before_checkpoint_read(monkeypatch, control):
    cfg = recipe(['init_from=/initial.pt', control])
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('loaded checkpoint before validation'))
    with pytest.raises(ValueError, match='do not combine'):
        ExperimentTracker(cfg)


def test_resume_merges_explicit_dictionary_delta_only():
    saved = recipe([])
    saved.training.objectives['action/future_frames'].predict_future_frames = True
    cfg = resumed(OmegaConf.to_container(saved, resolve=True), [
        'resume=true', 'training.objectives={action/future_frames:{weight:2.5}}',
    ])
    assert cfg.training.objectives['action/future_frames'].weight == 2.5
    assert cfg.training.objectives['action/future_frames'].predict_future_frames is True
    assert cfg.training.objectives['video/future_frames'] == saved.training.objectives['video/future_frames']


def test_resume_interpolates_explicit_value_against_saved_recipe():
    saved = recipe(['training.learning_rate=.003'])
    cfg = resumed(OmegaConf.to_container(saved, resolve=True), [
        'resume=true', '+training.copied_lr=${training.learning_rate}',
    ])
    assert cfg.training.learning_rate == .003
    assert cfg.training.copied_lr == .003
    assert OmegaConf.to_container(cfg, resolve=False)['training']['copied_lr'] == '${training.learning_rate}'


@pytest.mark.parametrize('override', ['data=hdf5_demos', 'data@training.dataset=hdf5_demos'])
def test_resume_rejects_group_overrides(override):
    manager = ConfigManager(recipe(['resume=true']))
    manager._original_overrides = [override]
    with pytest.raises(ValueError, match='Config-group overrides'):
        manager.merge_with_checkpoint(OmegaConf.to_container(recipe([]), resolve=True))


def test_append_cannot_silently_overwrite_a_saved_only_key():
    saved = recipe(['+training.custom=4'])
    with pytest.raises(ValueError, match='Append override already exists'):
        resumed(OmegaConf.to_container(saved, resolve=True), ['resume=true', '+training.custom=5'])
    cfg = resumed(OmegaConf.to_container(saved, resolve=True), ['resume=true', '++training.custom=5'])
    assert cfg.training.custom == 5


def test_unresolved_dictconfig_input_retains_links():
    saved = recipe(['training.dataset.root_dir=/old'])
    cfg = resumed(saved, ['resume=true', 'training.dataset.root_dir=/new'])
    assert cfg.training.dataset.datasets.action_video.root_dir == '/new'


def test_resume_rejects_group_even_if_name_matches_a_config_key():
    manager = ConfigManager(recipe(['resume=true']))
    manager._config_groups = {'training'}
    manager._original_overrides = ['training=another_group']
    with pytest.raises(ValueError, match='Config-group overrides'):
        manager.merge_with_checkpoint(OmegaConf.to_container(recipe([]), resolve=True))


@pytest.mark.parametrize('override', [
    'training.dataset.root_dir=/new', 'model.action_dim=14', 'training.true_horizon=8',
    'cond_cameraviews=[agentview]', 'model.image_size=64',
    'training={dataset:{root_dir:/new}}',
])
def test_legacy_resolved_checkpoint_rejects_stale_dependent_values(override):
    saved = OmegaConf.to_container(recipe(['training.dataset.root_dir=/old']), resolve=True)
    with pytest.raises(ValueError, match='no saved interpolation'):
        resumed(saved, ['resume=true', override])


def test_legacy_resolved_checkpoint_accepts_explicit_dependent_overrides():
    saved = OmegaConf.to_container(recipe(['training.dataset.root_dir=/old']), resolve=True)
    cfg = resumed(saved, [
        'resume=true', 'training.dataset.root_dir=/new',
        'training.dataset.datasets.video_only.root_dir=/new',
        'training.dataset.datasets.action_video.root_dir=/new',
        'training.dataset.val_datasets.video_only_val.root_dir=/new',
        'training.dataset.val_datasets.action_video_val.root_dir=/new',
    ])
    assert cfg.training.dataset.root_dir == '/new'
    assert cfg.training.dataset.datasets.action_video.root_dir == '/new'


def test_unresolved_checkpoint_preserves_dependency_links(tmp_path):
    original = ConfigManager(recipe(['training.dataset.root_dir=/old', 'init_from=/initial.pt']))
    data = original.get_checkpoint_data()
    assert data['config']['training']['dataset']['datasets']['action_video']['root_dir'] == '/old'
    assert data['training_config']['training']['dataset']['datasets']['action_video']['root_dir'] == '${training.dataset.root_dir}'
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoints = CheckpointManager('test', str(tmp_path), False)
    path = checkpoints.save(model, optimizer, None, TrainingState(epoch=5), data, None, "latest.pt")
    saved = checkpoints.load_config_only(path)
    assert saved == data['training_config']
    cfg = resumed(saved, ['resume=true', 'training.dataset.root_dir=/new'])
    assert cfg.training.dataset.datasets.action_video.root_dir == '/new'
    assert cfg.training.dataset.val_datasets.action_video_val.root_dir == '/new'
    assert cfg.init_from is None
    cfg.training.dataset.root_dir = '/again'
    assert cfg.training.dataset.datasets.action_video.root_dir == '/again'


@pytest.mark.parametrize('override', [
    'training.learning_rate=1e-5', 'training.weight_decay=.01',
    'training.pretrained_learning_rate_multiplier=.1', 'training.lr_scheduler=cosine_annealing',
    'training={learning_rate:1e-5}',
])
def test_resume_rejects_optimizer_hyperparameter_changes(override):
    saved = OmegaConf.to_container(recipe([]), resolve=True)
    with pytest.raises(ValueError, match='restoring optimizer/scheduler state'):
        resumed(saved, ['resume=true', override])


def test_unchanged_optimizer_config_preserves_scheduled_lr_on_restore(tmp_path, monkeypatch):
    saved = recipe(['training.lr_scheduler=cosine_annealing'])
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=saved.training.learning_rate,
                            weight_decay=saved.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    opt.step()
    scheduler.step()
    previous_lr = opt.param_groups[0]['lr']
    data = ConfigManager(saved).get_checkpoint_data()
    checkpoints = CheckpointManager('test', str(tmp_path), False)
    path = checkpoints.save(model, opt, scheduler, TrainingState(epoch=1), data, None, "latest.pt")
    overrides = ['resume=true', f'checkpoint={path}', 'training.num_epochs=60']

    def capture(manager):
        manager._original_overrides = overrides

    monkeypatch.setattr(ConfigManager, 'capture_overrides', capture)
    tracker = ExperimentTracker(recipe(overrides))
    cfg = tracker.get_merged_config()
    target = Tiny()
    new_opt = torch.optim.AdamW(target.parameters(), lr=cfg.training.learning_rate,
                               weight_decay=cfg.training.weight_decay)
    new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=60)
    state = tracker.setup_training(target, new_opt, new_scheduler)
    assert state.epoch == 1
    assert cfg.training.num_epochs == 60
    assert new_opt.param_groups[0]['lr'] == previous_lr
    assert new_opt.param_groups[0]['weight_decay'] == saved.training.weight_decay
    assert new_scheduler.state_dict() == scheduler.state_dict()


def test_pace_resolved_checkpoint_continuation_keeps_task_recipe():
    saved = recipe([
        'training.batch_size=16', 'training.num_epochs=5', 'experiment_name=object_t0_corrected',
        'training.dataset.datasets.action_video.dataset_names=[libero_object_t0]',
        'training.dataset.datasets.video_only.batch_weight=0', 'training.wandb.use_wandb=false',
    ])
    cfg = resumed(OmegaConf.to_container(saved, resolve=True), [
        'resume=true', 'checkpoint=/old/epoch_5.pt', 'experiment_name=object_t0_continue',
        'training.num_epochs=10',
    ])
    assert cfg.training.batch_size == 16
    assert cfg.training.dataset.datasets.action_video.dataset_names == ['libero_object_t0']
    assert cfg.training.dataset.datasets.video_only.batch_weight == 0
    assert cfg.training.num_epochs == 10
    assert cfg.checkpoint == '/old/epoch_5.pt'
    assert cfg.experiment_name == 'object_t0_continue'
    assert cfg.init_from is None
    assert not cfg.training.wandb.use_wandb


def checkpoint(tmp_path):
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    manager = CheckpointManager("test", str(tmp_path), False)
    cfg = {"config": {"training": {"batch_size": 8}}, "training_config": {"training": {"batch_size": 8}}, "overrides": []}
    path = manager.save(model, optimizer, None, TrainingState(epoch=3, train_global_iter=25), cfg, None, "latest.pt")
    return manager, path, model, optimizer


def test_checkpoint_roundtrip_restores_weights_optimizer_and_state(tmp_path):
    manager, path, model, optimizer = checkpoint(tmp_path)
    target = Tiny()
    opt = torch.optim.AdamW(target.parameters())
    state = manager.load(path, target, opt, None)
    cfg = manager.load_config_only(path)
    assert state.epoch == 3
    assert state.train_global_iter == 25
    assert cfg["training"]["batch_size"] == 8
    for x, y in zip(model.parameters(), target.parameters()):
        torch.testing.assert_close(x, y)
    assert len(opt.state) == len(optimizer.state) == 2


def test_resume_raises_on_incompatible_optimizer(tmp_path):
    manager, path, model, _ = checkpoint(tmp_path)
    extra = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([*model.parameters(), extra])
    with pytest.raises(ValueError, match="parameter group"):
        manager.load(path, model, optimizer, None)


def test_resume_raises_when_requested_training_state_is_missing(tmp_path):
    manager, path, model, optimizer = checkpoint(tmp_path)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with pytest.raises(KeyError, match="scheduler"):
        manager.load(path, model, optimizer, scheduler)


def test_corrupt_checkpoint_fails_without_deleting_user_file(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a checkpoint")
    manager = CheckpointManager("test", str(tmp_path), True)
    with pytest.raises(Exception):
        manager.load_config_only(path)
    assert path.read_bytes() == b"not a checkpoint"


@pytest.mark.parametrize('name_override', [[], ['experiment_name=fork']])
def test_explicit_checkpoint_from_another_experiment(tmp_path, monkeypatch, name_override):
    saved = recipe(['experiment_name=other'])
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters())
    data = ConfigManager(saved).get_checkpoint_data()
    path = CheckpointManager('other', str(tmp_path), False).save(model, opt, None, TrainingState(epoch=1), data, None, "latest.pt")
    overrides = ['resume=true', f'checkpoint={path}', *name_override]
    monkeypatch.setattr(ConfigManager, 'capture_overrides',
                        lambda manager: setattr(manager, '_original_overrides', overrides))
    if name_override:
        tracker = ExperimentTracker(recipe(overrides))
        assert tracker.experiment_name == tracker.get_merged_config().experiment_name == 'fork'
    else:
        with pytest.raises(ValueError, match="saved by experiment 'other'"):
            ExperimentTracker(recipe(overrides))


def test_training_state_rejects_unknown_fields():
    with pytest.raises(TypeError):
        TrainingState().update(not_a_field=1)
    with pytest.raises(TypeError):
        TrainingState.from_dict({"epoch": 1, "not_a_field": 2})


def test_fresh_run_directories_auto_increment_and_resume_picks_the_latest_with_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = CheckpointManager("exp", None, False).checkpoint_dir
    second = CheckpointManager("exp", None, False).checkpoint_dir
    assert (first, second) == (Path("checkpoints/exp"), Path("checkpoints/exp_1"))
    with pytest.raises(FileNotFoundError, match="no checkpoint found"):
        CheckpointManager("exp", None, True).checkpoint_dir
    (first / "epoch_2.pt").touch()
    (first / "epoch_10.pt").touch()
    (second / "best_checkpoint.pt").touch()
    manager = CheckpointManager("exp", None, True)
    assert manager.checkpoint_dir == first          # exp_1 exists but holds nothing to resume from
    assert manager.find_latest() == first / "epoch_10.pt"
    (first / "latest.pt").touch()
    assert manager.find_latest() == first / "latest.pt"


def test_find_latest_never_falls_back_to_other_files(tmp_path):
    (tmp_path / "best_checkpoint.pt").touch()
    with pytest.raises(FileNotFoundError, match="no latest.pt"):
        CheckpointManager("exp", str(tmp_path), True).find_latest()


def test_load_rejects_weight_mismatch(tmp_path):
    manager, path, model, optimizer = checkpoint(tmp_path)
    other = torch.nn.Sequential(Tiny())
    other.encoders = torch.nn.ModuleDict()
    with pytest.raises(RuntimeError, match="does not match the model"):
        manager.load(path, other, torch.optim.AdamW(other.parameters()), None)
