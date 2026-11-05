"""
HDF5DemoDataset: robot demonstrations stored as one HDF5 file per demo.

HDF5 layout (per demo file, T = number of timesteps):
    actions/<action_space>:          (T, D)       float  - D = ACTION_SPACE_CONFIGS[action_space]['total_dim']
    observations/images/<camera>:    (T, H, W, 3) uint8  - one dataset per camera view, any view names
    text:                            scalar string       - task description (scripts/add_text_to_hdf5.py)

Any action space registered in tapestry/utils/action_normalizer.py:ACTION_SPACE_CONFIGS can be used;
normalizer stats come from `scripts/compute_normalizer_stats.py`.
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from tapestry.loaders.base_dataset import BaseDataset
from tapestry.utils.action_normalizer import ACTION_SPACE_CONFIGS, ActionNormalizer
from tapestry.utils.action_preprocess import to_relative_rot6d_eepose20

MOTION_THRESH = 1e-6


class HDF5DemoDataset(BaseDataset):
    """Chunks of `true_horizon` consecutive timesteps from HDF5 demos (see module docstring)."""

    def __init__(
        self,
        root_dir: str,
        demo_dir: str,                      # the demos are the *.hdf5 files directly inside root_dir/demo_dir
        active_modalities: List[str],
        data_percentage: float,
        action_space: str,
        cond_cameraviews: List[str],
        img_shape: Tuple[int, int],
        true_horizon: int,
        pred_horizon: int,
        is_validation: bool,
        normalize_actions: bool,
        normalizer_stats_path: Optional[str],
        relative_rotation_actions: bool,    # rot6d rotations relative to the chunk's first frame (eepose_6drot only)
        filter_no_motion: bool,             # drop chunks whose position dims never move
        skip_first_frame: bool,             # never start a chunk at t=0 (homing artifacts)
        corrupt_zero_blocks: Optional[List[List[int]]],  # drop chunks where any of these dim groups is all-zero
        aug_cfg: Optional[Dict],            # null: no augmentation (see BaseDataset.apply_augmentation)
    ):
        super().__init__(active_modalities, data_percentage, cond_cameraviews, img_shape, true_horizon, pred_horizon, is_validation)
        self.demo_dir = os.path.expanduser(os.path.join(root_dir, demo_dir))
        self.action_space = action_space
        self.action_dim = ACTION_SPACE_CONFIGS[action_space]["total_dim"]
        self.pos_dims = list(ACTION_SPACE_CONFIGS[action_space]["normalize_dims"])
        self.filter_no_motion = filter_no_motion
        self.skip_first_frame = skip_first_frame
        self.corrupt_zero_blocks = [list(b) for b in corrupt_zero_blocks] if corrupt_zero_blocks else []

        self.aug_cfg = aug_cfg

        self.relative_rotation_actions = relative_rotation_actions
        if relative_rotation_actions and action_space != "eepose_6drot":
            raise ValueError(f"relative_rotation_actions requires action_space='eepose_6drot', got '{action_space}'")

        self.normalizer = None
        if normalize_actions and "action" in self.active_modalities:
            if normalizer_stats_path is None:
                raise ValueError("normalize_actions=True requires normalizer_stats_path")
            self.normalizer = ActionNormalizer.from_stats_file(os.path.expanduser(normalizer_stats_path))
            if self.normalizer.action_space != action_space:
                raise ValueError(f"Normalizer action_space ({self.normalizer.action_space}) != dataset ({action_space})")
            if self.normalizer.horizon != true_horizon:
                raise ValueError(f"Normalizer horizon ({self.normalizer.horizon}) != true_horizon ({true_horizon})")

        self.index_map = self.create_index_map()

    def _chunk_validity(self, f: h5py.File, demo_file: str) -> Tuple[int, np.ndarray, np.ndarray]:
        """Per-timestep flags: valid (no corrupt zero block) and has_motion[t] (frame t -> t+1 moved)."""
        if "actions" in self.keys_to_load:
            key = f"actions/{self.action_space}"
            if key not in f:
                raise KeyError(f"{demo_file}: missing '{key}' (available: {list(f['actions'].keys()) if 'actions' in f else 'no actions group'})")
            actions = f[key][:].astype(np.float32)  # [T, D]
            if actions.shape[1] != self.action_dim:
                raise ValueError(f"{demo_file}: '{key}' has dim {actions.shape[1]}, expected {self.action_dim} for '{self.action_space}'")
            n = actions.shape[0]
            valid = np.ones(n, dtype=bool)
            for block in self.corrupt_zero_blocks:
                valid &= ~np.all(actions[:, block] == 0.0, axis=1)
            if self.filter_no_motion:
                has_motion = np.linalg.norm(np.diff(actions[:, self.pos_dims], axis=0), axis=1) > MOTION_THRESH  # [T-1]
            else:
                has_motion = np.ones(max(n - 1, 0), dtype=bool)
            return n, valid, has_motion
        key = f"observations/images/{self.cond_cameraviews[0]}"
        if key not in f:
            raise KeyError(f"{demo_file}: missing '{key}'")
        n = f[key].shape[0]
        return n, np.ones(n, dtype=bool), np.ones(max(n - 1, 0), dtype=bool)

    def create_index_map(self) -> List[Dict]:
        files = sorted(glob.glob(os.path.join(self.demo_dir, "*.hdf5")))
        if not files:
            raise FileNotFoundError(f"No *.hdf5 files in {self.demo_dir}")
        selected = self.split(files)
        split = "validation" if self.is_validation else "training"
        print(f"HDF5DemoDataset ({split}): {len(selected)}/{len(files)} demos from {self.demo_dir}")

        index_map = []
        n_skipped = {"first_frame": 0, "corrupt_zero": 0, "no_motion": 0}
        for demo_file in tqdm(selected, desc="Indexing demos"):
            with h5py.File(demo_file, "r") as f:
                n, valid, has_motion = self._chunk_validity(f, demo_file)
            for start_t in range(n):
                if self.skip_first_frame and start_t == 0:
                    n_skipped["first_frame"] += 1
                    continue
                end_t = min(start_t + self.true_horizon, n)
                if not np.all(valid[start_t:end_t]):
                    n_skipped["corrupt_zero"] += 1
                    continue
                motion = has_motion[start_t:end_t - 1]
                if len(motion) > 0 and not np.any(motion):
                    n_skipped["no_motion"] += 1
                    continue
                index_map.append({"demo_file": demo_file, "start_t": start_t, "end_t": end_t})
        print(f"  {len(index_map)} chunks (skipped: {n_skipped})")
        if not index_map:
            raise ValueError(f"No chunks left in {self.demo_dir} (skipped: {n_skipped})")
        return index_map

    def load_data(self, idx_dict: Dict) -> Dict:
        demo_file, start_t, end_t = idx_dict["demo_file"], idx_dict["start_t"], idx_dict["end_t"]
        data = {}
        with h5py.File(demo_file, "r") as f:
            if "actions" in self.keys_to_load:
                data["actions"] = f[f"actions/{self.action_space}"][start_t:end_t].astype(np.float32)  # [T, D]
            if "images" in self.keys_to_load:
                # Contiguous read then numpy subsample: strided HDF5 reads are ~100x slower.
                data["images"] = np.stack([
                    f[f"observations/images/{view}"][start_t:end_t][::self.skip_every] for view in self.cond_cameraviews
                ], axis=0).astype(np.float32)  # [V, T, H, W, C]
            if "text" in self.keys_to_load:
                if "text" not in f:
                    raise KeyError(f"{demo_file}: no 'text' dataset; add one with scripts/add_text_to_hdf5.py")
                text = f["text"][()]
                data["text"] = text.decode("utf-8") if isinstance(text, bytes) else str(text)
        return data

    def process_data(self, data: Dict) -> Dict[str, Any]:
        out = {"active_modalities": list(self.active_modalities)}

        if "actions" in data:
            actions = data["actions"]  # [T, D]
            if actions.shape[0] < self.true_horizon:  # pad by repeating the last action (delta 0 after normalization)
                pad = np.repeat(actions[actions.shape[0] - 1:], self.true_horizon - actions.shape[0], axis=0)
                actions = np.concatenate([actions, pad], axis=0)
            if self.relative_rotation_actions:
                actions = to_relative_rot6d_eepose20(actions)
            if self.normalizer is not None:
                actions = self.normalizer.normalize(actions)
            out["action/future_frames"] = torch.from_numpy(actions).float()  # [T, D]

        if "images" in data:
            video = self.to_video(data["images"])  # [V, T, C, H, W]
            if self.aug_cfg is not None:
                video = self.apply_augmentation(video, self.aug_cfg)
            out["video/history_frames"] = video[:, 0:1]  # [V, 1, C, H, W]
            out["video/future_frames"] = video  # [V, T, C, H, W]

        if "text" in data:
            out["text/embeddings"] = data["text"]  # raw string, encoded by the model's text encoder

        return out
