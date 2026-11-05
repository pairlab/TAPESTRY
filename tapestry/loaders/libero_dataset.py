"""
LiberoDataset: LIBERO demonstrations in the layout of the official datasets, one HDF5 file per task.

    <root_dir>/libero_demos/<suite>/<task>_demo.hdf5
        data/demo_<n>/actions:             (T, 7)       float  - end-effector deltas and gripper, used as stored
        data/demo_<n>/obs/<view>_rgb:      (T, H, W, 3) uint8  - one per camera view (agentview, eye_in_hand), stored upside down

Sample keys, shared by every dataset class (V camera views, T = pred_horizon frames):
    video/history_frames   [V, 1, C, H, W]      the first frame, in [0, 1]
    video/future_frames    [V, T, C, H, W]      all frames, including the first
    action/future_frames   [true_horizon, D]
    text/embeddings        str                  the task instruction, encoded by the model's text encoder
    active_modalities      List[str]

custom_collate batches the samples of all dataset classes.
"""

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from tapestry.loaders.base_dataset import BaseDataset

VIEW_KEYS = ("video/history_frames", "video/future_frames")


def _pad_views(values: List[torch.Tensor], key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad [V_i, T, C, H, W] videos to the largest view count. Returns [B, V_max, T, C, H, W] and a [B, V_max] mask of real views."""
    shape = values[0].shape[1:]
    for i, v in enumerate(values):
        if v.shape[1:] != shape:
            raise ValueError(f"'{key}': sample {i} has shape {tuple(v.shape)}, expected (V, {', '.join(map(str, shape))})")
    padded = torch.zeros((len(values), max(v.shape[0] for v in values)) + shape, dtype=values[0].dtype)
    view_mask = torch.zeros(padded.shape[:2], dtype=torch.bool)
    for i, v in enumerate(values):
        padded[i, :v.shape[0]] = v
        view_mask[i, :v.shape[0]] = True
    return padded, view_mask


def custom_collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch samples whose keys and camera view counts may differ (datasets with different modalities share a batch).

    Tensors are stacked to [B, ...]. A key missing from some samples is zero-filled there and gets a [B] bool `<key>_mask`
    of the samples that have it. Videos with different view counts are zero-padded to [B, V_max, ...] and get a
    [B, V_max] bool `video/view_mask`. Non-tensor values become lists of length B, with None where missing."""
    B = len(batch_list)
    batch = {}
    for key in dict.fromkeys(k for sample in batch_list for k in sample):
        present = [i for i, sample in enumerate(batch_list) if key in sample]
        values = [batch_list[i][key] for i in present]
        if not isinstance(values[0], torch.Tensor):
            batch[key] = [sample[key] if key in sample else None for sample in batch_list]
            continue
        if key in VIEW_KEYS and len({v.shape[0] for v in values}) > 1:
            if len(values) != B:
                raise ValueError(f"cannot pad the views of '{key}': only samples {present} of {B} have it")
            batch[key], batch["video/view_mask"] = _pad_views(values, key)
            continue
        shapes = {tuple(v.shape) for v in values}
        if len(shapes) > 1:
            raise ValueError(f"cannot collate '{key}': samples have shapes {sorted(shapes)}")
        if len(values) < B:  # zeros where the key is missing, so that batch[key][i] still belongs to sample i
            mask = torch.zeros(B, dtype=torch.bool)
            mask[present] = True
            batch[f"{key}_mask"] = mask
            values = [sample[key] if key in sample else torch.zeros_like(values[0]) for sample in batch_list]
        batch[key] = torch.stack(values)
    return batch


def grab_language_from_filename(x):
    """Task instruction from a LIBERO task file name. Copied from libero.libero.benchmark, which names the evaluation tasks,
    so that training and evaluation prompts match."""
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


class LiberoDataset(BaseDataset):
    """Chunks of `true_horizon` consecutive timesteps from LIBERO demos (see module docstring)."""

    def __init__(
        self,
        root_dir: str,                  # contains libero_demos/<suite>/*.hdf5
        dataset_names: List[str],       # suites, e.g. ["libero_90", "libero_10"]
        active_modalities: List[str],
        data_percentage: float,         # share of every task's demos: the first ones for training, the last ones for validation
        cond_cameraviews: List[str],
        img_shape: Tuple[int, int],
        true_horizon: int,
        pred_horizon: int,
        is_validation: bool,
    ):
        super().__init__(active_modalities, data_percentage, cond_cameraviews, img_shape, true_horizon, pred_horizon, is_validation)
        self.root_dir = os.path.expanduser(root_dir)
        self.dataset_names = list(dataset_names)
        self.index_map = self.create_index_map()

    def create_index_map(self) -> List[Dict]:
        needed = ["actions"] + [f"obs/{view}_rgb" for view in self.cond_cameraviews]
        index_map = []
        for name in self.dataset_names:
            task_files = sorted(glob.glob(os.path.join(self.root_dir, "libero_demos", name, "*.hdf5")))
            if not task_files:
                raise FileNotFoundError(f"no demo files for '{name}' in {self.root_dir}/libero_demos/{name}")
            for task_file in tqdm(task_files, desc=f"Indexing {name}"):
                with h5py.File(task_file, "r") as f:
                    demos = sorted(f["data"], key=lambda demo: int(demo.split("_")[1]))  # demo_0, demo_1, ...
                    for demo in self.split(demos):
                        missing = [key for key in needed if key not in f[f"data/{demo}"]]
                        if missing:
                            raise KeyError(f"{task_file}: {demo} has no {missing}")
                        n = f[f"data/{demo}/actions"].shape[0]
                        index_map += [
                            {"task_file": task_file, "demo_key": f"data/{demo}", "start_t": t, "end_t": min(t + self.true_horizon, n)}
                            for t in range(n)
                        ]
        split = "validation" if self.is_validation else "training"
        print(f"LiberoDataset ({split}): {len(index_map)} chunks from {self.dataset_names}, modalities {self.active_modalities}")
        return index_map

    def load_data(self, idx_dict: Dict) -> Dict:
        task_file, start_t, end_t = idx_dict["task_file"], idx_dict["start_t"], idx_dict["end_t"]
        data = {}
        if "text" in self.keys_to_load:
            data["text"] = grab_language_from_filename(Path(task_file).stem.replace("_demo", ".bddl"))
        with h5py.File(task_file, "r") as f:
            demo = f[idx_dict["demo_key"]]
            if "actions" in self.keys_to_load:
                data["actions"] = demo["actions"][start_t:end_t].astype(np.float32)  # [T, D]
            if "images" in self.keys_to_load:
                data["images"] = np.stack([
                    demo[f"obs/{view}_rgb"][start_t:end_t][::self.skip_every] for view in self.cond_cameraviews
                ], axis=0).astype(np.float32)  # [V, T, H, W, C]
        return data

    def process_data(self, data: Dict) -> Dict[str, Any]:
        out = {"active_modalities": list(self.active_modalities)}
        if "actions" in data:
            actions = data["actions"]  # [T, D]
            actions = np.pad(actions, ((0, self.true_horizon - actions.shape[0]), (0, 0)))  # zero deltas (no movement) after the demo's end
            out["action/future_frames"] = torch.from_numpy(actions).float()
        if "images" in data:
            video = self.to_video(data["images"][:, :, ::-1])  # [V, T, C, H, W], upright
            out["video/history_frames"] = video[:, 0:1]
            out["video/future_frames"] = video
        if "text" in data:
            out["text/embeddings"] = data["text"]
        return out
