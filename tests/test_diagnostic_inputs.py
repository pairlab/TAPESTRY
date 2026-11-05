"""CPU input checks for the experimental HDF5 and synthetic diagnostics."""
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.benchmark_tapestry_fullpath import load_cfg as load_full_cfg
from scripts.benchmark_tapestry_latency import load_cfg as load_latency_cfg
from scripts.eval_determinism_at_t import load_obs_from_hdf5
from scripts.eval_teacher_forced import load_chunk
from tapestry.utils.action_preprocess import to_relative_rot6d_eepose20


@pytest.mark.parametrize("load_cfg", [load_full_cfg, load_latency_cfg])
def test_benchmark_composes_inherited_config(load_cfg):
    cfg = load_cfg(Path(__file__).resolve().parents[1] / "cfg/finetune.yaml")
    assert cfg["training"]["objectives"]["video/future_frames"]["weight"] == 0.1
    assert cfg["encoders"]["text"]["max_seq_len"] == 96
    assert list(cfg["training"]["dataset"]["datasets"]) == ["demos"]


def _demo(path):
    with h5py.File(path, "w") as f:
        f["observations/images/custom_front"] = np.full((4, 8, 8, 3), 64, dtype=np.uint8)
        f["observations/images/custom_wrist"] = np.full((4, 8, 8, 3), 192, dtype=np.uint8)
        f["actions/libero"] = np.arange(28, dtype=np.float32).reshape(4, 7) / 30


def test_hdf5_diagnostics_use_requested_views(tmp_path):
    path = tmp_path / "demo.hdf5"
    _demo(path)
    views = ["custom_wrist", "custom_front"]
    video = load_obs_from_hdf5(path, 0, 8, "cpu", views)
    assert video.shape == (1, 2, 1, 3, 8, 8)
    torch.testing.assert_close(video[0, 0], torch.full((1, 3, 8, 8), 192 / 255))
    chunk = load_chunk(path, 0, None, 4, 7, 8, "cpu", views, "libero", False)
    assert chunk["gt_norm"].shape == (1, 4, 7)
    torch.testing.assert_close(chunk["video"], video)
    with pytest.raises(KeyError):
        load_obs_from_hdf5(path, 0, 8, "cpu", ["missing"])
    with pytest.raises(ValueError, match="action chunk"):
        load_chunk(path, 0, None, 4, 20, 8, "cpu", views, "libero", False)


def test_teacher_forced_relative_rotations_match_training(tmp_path):
    path = tmp_path / "demo.hdf5"
    _demo(path)
    actions = np.zeros((4, 20), dtype=np.float32)
    for t in range(4):
        angle = (t + 1) * 0.2
        rot = np.array([np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0])
        actions[t, 3:9] = rot
        actions[t, 13:19] = rot
    with h5py.File(path, "a") as f:
        f["actions/eepose_6drot"] = actions
    chunk = load_chunk(path, 0, None, 4, 20, 8, "cpu", ["custom_front"], "eepose_6drot", True)
    np.testing.assert_allclose(chunk["gt_norm"][0].numpy(), to_relative_rot6d_eepose20(actions))
