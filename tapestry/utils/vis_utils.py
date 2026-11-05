"""Inspect training batches: print every key's type and shape, or save each sample's frames as a PNG."""
import os
from typing import Any, Dict, List

import numpy as np
import torch
from einops import rearrange
from matplotlib.figure import Figure


def print_batch(batch: Dict[str, Any], name: str):
    """Print every key of `batch` with its type, plus shape, dtype and range for tensors and arrays, length for lists."""
    print(f"----- {name} -----")
    for key, value in sorted(batch.items()):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            print(f"{key}: {type(value).__name__} {tuple(value.shape)} {value.dtype} [{value.min()}, {value.max()}]")
        elif isinstance(value, list):
            print(f"{key}: list of {len(value)}")
        else:
            print(f"{key}: {value!r}")


def save_batch_frames(batch: Dict[str, Any], keys: List[str], out_dir: str, num_samples: int):
    """Save <out_dir>/sample_<i>.png for the first num_samples samples of a collated batch, titled with the sample's
    instruction. Each key in `keys` (e.g. video/history_frames, video/future_frames) gets a panel with the camera views
    stacked top to bottom and time running left to right."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_samples):
        fig = Figure(figsize=(16, 4 * len(keys)))
        axes = fig.subplots(len(keys), 1, squeeze=False)
        for ax, key in zip(axes[:, 0], keys):
            video = batch[key][i].detach().float().cpu()  # [V, T, C, H, W] in [0, 1]
            ax.imshow(rearrange(video, "v t c h w -> (v h) (t w) c").numpy())
            ax.set_title(key)
            ax.axis("off")
        fig.suptitle(batch["text/embeddings"][i])
        fig.savefig(os.path.join(out_dir, f"sample_{i}.png"), bbox_inches="tight")
