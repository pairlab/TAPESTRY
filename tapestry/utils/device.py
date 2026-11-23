"""Default torch device: CUDA when available, else an Intel GPU (XPU), else the CPU."""
import torch


def default_device() -> torch.device:
    """CUDA when available, else XPU (Intel GPU), else the CPU. A device passed explicitly always takes precedence."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
