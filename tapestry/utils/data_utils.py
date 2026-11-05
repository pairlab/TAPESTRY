from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from pytorch_msssim import ssim
from einops import rearrange


class WeightedConcatDataset(Dataset):
    """Concatenation of datasets; create_sampler() draws each dataset in proportion to its weight."""

    def __init__(self, datasets: List[Dataset], weights: List[float]):
        self.datasets = datasets
        self.weights = np.array(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()

        self.dataset_sizes = [len(d) for d in datasets]
        if 0 in self.dataset_sizes:
            raise ValueError(f"empty dataset(s) at positions {[i for i, n in enumerate(self.dataset_sizes) if n == 0]}")
        self.cumulative_sizes = np.cumsum(self.dataset_sizes)
        self._sample_weights = self._compute_sample_weights()

    def _compute_sample_weights(self) -> np.ndarray:
        sample_weights = []
        for dataset, weight in zip(self.datasets, self.weights):
            sample_weights.extend([weight / len(dataset)] * len(dataset))
        return np.array(sample_weights)

    def __len__(self) -> int:
        return sum(self.dataset_sizes)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def create_sampler(self) -> WeightedRandomSampler:
        """One epoch draws len(self) samples with replacement; each dataset gets its weight's share of them."""
        return WeightedRandomSampler(self._sample_weights, len(self), replacement=True)


def create_dataset_from_config(dataset_config, dataset_classes):
    """Build the dataset named by `_class`. Every other config key except batch_weight is a constructor argument."""
    params = {k: v for k, v in dataset_config.items() if k not in ("_class", "batch_weight")}
    return dataset_classes[dataset_config["_class"]](**params)


def compute_video_metrics(pred, gt, lpips_fn):
    """Mean per-frame PSNR, SSIM and LPIPS (lpips_fn: an lpips.LPIPS network) between videos [N, T, C, H, W] in [0, 1]."""
    pred = rearrange(pred.float().clamp(0, 1), "n t c h w -> (n t) c h w")
    gt = rearrange(gt.float().clamp(0, 1), "n t c h w -> (n t) c h w").to(pred.device)
    mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3))   # [N*T]
    psnr = (10 * torch.log10(1.0 / mse)).mean()
    return {
        "psnr": psnr.item(),
        "ssim": ssim(pred, gt, data_range=1.0, size_average=True).item(),
        "lpips": lpips_fn(pred * 2 - 1, gt * 2 - 1).mean().item(),   # LPIPS expects [-1, 1]
    }
