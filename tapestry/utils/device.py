"""Default torch devices: CUDA when available, else an Intel GPU (XPU), and the CPU only where the code allows it."""
import os

import torch


def default_device() -> torch.device:
    """CUDA when available, else XPU (Intel GPU), else the CPU. A device passed explicitly always takes precedence."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def gpu_device() -> torch.device:
    """CUDA when available, else XPU (Intel GPU). For code that needs a GPU: raises instead of falling back to the CPU,
    so a process pinned to a missing or unusable GPU stops at once."""
    device = default_device()
    if device.type == "cpu":
        raise RuntimeError(f"no usable CUDA or XPU device (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                           f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')})")
    return device
