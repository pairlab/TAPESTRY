"""Core imports and the ImageIO backend do not require Decord."""
import os
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from tapestry.loaders.video_only_dataset import VideoOnlyDataset, _decord_reader


def test_core_imports_without_decord():
    root = Path(__file__).resolve().parents[1]
    code = """import sys
sys.modules['decord'] = None
import train
from tapestry.models.policy import TapestryPolicy
from tapestry.loaders.video_only_dataset import VideoOnlyDataset
print('CORE_IMPORTS_OK')
"""
    env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES='')
    p = subprocess.run([sys.executable, '-c', code], cwd='/tmp', env=env,
                       capture_output=True, text=True, check=True)
    assert 'CORE_IMPORTS_OK' in p.stdout


def test_missing_selected_decord_raises(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'decord', None)
    with pytest.raises(ModuleNotFoundError, match='decord'):
        _decord_reader(tmp_path / 'video.mp4')


def test_imageio_reads_requested_frame_order_without_decord(monkeypatch, tmp_path):
    path = tmp_path / 'video.mp4'
    frames = [np.full((16, 16, 3), i * 50, dtype=np.uint8) for i in range(4)]
    imageio.mimwrite(path, frames, fps=4)
    expected = imageio.mimread(path)
    ds = object.__new__(VideoOnlyDataset)
    ds.mp4_decoder = 'imageio'
    ids = [3, 1, 1, 0]
    monkeypatch.setitem(sys.modules, 'decord', None)
    actual = ds._load_local_mp4_at_indices(path, ids)
    torch.testing.assert_close(actual, torch.from_numpy(np.stack([expected[i] for i in ids])), rtol=0, atol=0)


def test_dataset_construction_requires_selected_decord(monkeypatch):
    ds = object.__new__(VideoOnlyDataset)
    ds.dataset_name, ds.translate_paths = 'clips', False
    monkeypatch.setitem(sys.modules, 'decord', None)
    ds.mp4_decoder = 'imageio'
    ds._validate_decoder_config()
    ds.mp4_decoder = 'decord'
    with pytest.raises(ModuleNotFoundError, match='decord'):
        ds._validate_decoder_config()
