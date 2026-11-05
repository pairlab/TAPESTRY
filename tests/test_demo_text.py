from pathlib import Path
import sys

import h5py
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.add_text_to_hdf5 import main


def test_existing_label_requires_explicit_overwrite_before_any_writes(tmp_path, monkeypatch):
    first, second = tmp_path / 'a.hdf5', tmp_path / 'b.hdf5'
    with h5py.File(first, 'w'):
        pass
    with h5py.File(second, 'w') as f:
        f.create_dataset('text', data='original', dtype=h5py.string_dtype())
    args = ['add_text_to_hdf5.py', '--demo-dir', str(tmp_path), '--text', 'replacement']
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(FileExistsError, match='--overwrite'):
        main()
    with h5py.File(first, 'r') as f:
        assert 'text' not in f
    with h5py.File(second, 'r') as f:
        assert f['text'].asstr()[()] == 'original'
    monkeypatch.setattr(sys, 'argv', args + ['--overwrite'])
    main()
    for p in (first, second):
        with h5py.File(p, 'r') as f:
            assert f['text'].asstr()[()] == 'replacement'
