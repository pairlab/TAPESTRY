"""wandb run resolution and mid-epoch continuation on resume, without constructing the video model."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import train
from tapestry.utils.training_manager import TrainingState

ROOT = Path(__file__).resolve().parents[1]


def resolve(monkeypatch, is_resume, experiment_name, **state):
    """train.resolve_wandb_run against a checkpoint of experiment 'overfit' whose training state has `state`."""
    loads = []
    payload = {'training_state': TrainingState(epoch=5, **state).to_dict(), 'config': {'experiment_name': 'overfit'}}

    def load(path, **kwargs):
        loads.append(path)
        return payload

    monkeypatch.setattr(train.torch, 'load', load)
    tracker = SimpleNamespace(is_resume=is_resume, checkpoint_path='checkpoint.pt')
    return train.resolve_wandb_run(tracker, experiment_name), loads


def main_def():
    return next(n for n in ast.parse(ROOT.joinpath('train.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')


def test_fresh_run_does_not_read_a_checkpoint(monkeypatch):
    assert resolve(monkeypatch, False, 'overfit') == ((None, 'overfit'), [])


def test_resume_requires_saved_run_identity(monkeypatch):
    with pytest.raises(ValueError, match='missing wandb_run_id'):
        resolve(monkeypatch, True, 'overfit')


def test_resume_continues_the_saved_run(monkeypatch):
    run, loads = resolve(monkeypatch, True, 'overfit', wandb_run_id='abc123', wandb_run_name='overfit')
    assert run == ('abc123', 'overfit')
    assert loads == ['checkpoint.pt']


def test_new_experiment_name_forks_the_run(monkeypatch):
    run, _ = resolve(monkeypatch, True, 'overfit_fork', wandb_run_id='abc123', wandb_run_name='overfit')
    assert run == (None, 'overfit_fork')


def test_disabled_logging_does_not_gate_training_resume():
    """The wandb run is resolved only when wandb is enabled, so resuming without wandb never needs a saved run id."""
    gated = [n for n in ast.walk(main_def()) if isinstance(n, ast.If) and 'use_wandb' in ast.unparse(n.test)]
    assert len(gated) == 1 and 'resolve_wandb_run' in ast.unparse(gated[0])
    assert ast.unparse(main_def()).count('resolve_wandb_run') == 1


def test_hydra_keeps_the_launch_directory():
    (decorator,) = main_def().decorator_list
    assert ast.unparse(decorator) == "hydra.main(config_path='cfg', version_base=None)"
    for path in ('train.py', 'tapestry/utils/training_manager.py'):
        assert 'get_original_cwd' not in ROOT.joinpath(path).read_text()


class Samples(Dataset):
    """12 samples that count how many were loaded."""
    def __init__(self):
        self.loaded = 0

    def __len__(self):
        return 12

    def __getitem__(self, i):
        self.loaded += 1
        return {"x": torch.ones(1)}


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs, loss_manager, targets):
        return {"x": inputs["x"] * self.w}


class Losses:
    target_keys = ["x"]

    def __call__(self, outputs, targets, noise):
        loss = outputs["x"].mean()
        return {"x": loss, "total": loss}

    def pop_per_timestep_mse_stats(self):
        return {}


class Tracker:
    """Records the training state of every checkpoint it saves."""
    def __init__(self, state):
        self.state, self.saved = state, []

    def update_state(self, **kwargs):
        self.state = self.state.update(**kwargs)

    def save(self, model, optimizer, scheduler, metrics, name):
        self.saved.append((name, self.state))
        return name


def test_mid_epoch_checkpoint_continues_after_its_batches():
    accelerator = Accelerator(cpu=True)
    ds = Samples()
    sampler = WeightedRandomSampler(torch.ones(12), 12, replacement=True)
    loader = accelerator.prepare(DataLoader(ds, batch_size=2, sampler=sampler))   # 6 batches
    model = Scale()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def run(start, train_global_iter, tracker):
        return train.train_epoch(model, opt, loader, Losses(), accelerator, 2, start, train_global_iter, ["x"], ["x"],
                                 log_interval=100, tracker=tracker, scheduler=None, step_ckpt_interval=2)[1]

    full = Tracker(TrainingState())
    assert run(0, 10, full) == 16
    assert [(s.epoch, s.epoch_batches, s.train_global_iter) for _, s in full.saved] == [(2, 2, 12), (2, 4, 14), (2, 6, 16)]
    ds.loaded = 0
    resumed = Tracker(TrainingState())
    assert run(4, 14, resumed) == 16   # from the latest.pt written after batch 4, the epoch ends where the full one did
    assert [(s.epoch, s.epoch_batches, s.train_global_iter) for _, s in resumed.saved] == [(2, 6, 16)]
    assert ds.loaded == 4   # only the 2 remaining batches of 2 samples were loaded


def test_epoch_checkpoint_starts_the_next_epoch_at_its_first_batch():
    tracker = Tracker(TrainingState(epoch=2, epoch_batches=4, train_global_iter=14))
    train.save_epoch(tracker, None, None, None, 2, 16, 3, {"total": 1.0}, {"total": 1.0}, save_numbered=False)
    assert [(n, s.epoch, s.epoch_batches) for n, s in tracker.saved] == [("best_checkpoint.pt", 3, 0), ("latest.pt", 3, 0)]
