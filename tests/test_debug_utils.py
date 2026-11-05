"""grad_norms groups gradients by parameter name and skips parameters without one; the batch inspectors run on a
collated batch."""
import pytest
import torch

from tapestry.utils.monitor_utils import grad_norms
from tapestry.utils.vis_utils import print_batch, save_batch_frames


def test_grad_norms():
    model = torch.nn.ModuleDict({"a": torch.nn.Linear(3, 2), "b": torch.nn.Linear(3, 2)})
    model["a"](torch.ones(1, 3)).sum().backward()  # b receives no gradient
    expected = torch.cat([model["a"].weight.grad.flatten(), model["a"].bias.grad]).norm().item()
    norms = grad_norms(model, depth=1)
    assert norms == pytest.approx({"a": expected})
    assert grad_norms(model, depth=2) == pytest.approx({"a.weight": model["a"].weight.grad.norm().item(),
                                                        "a.bias": model["a"].bias.grad.norm().item()})


def test_batch_inspectors(tmp_path, capsys):
    batch = {
        "video/history_frames": torch.rand(2, 2, 1, 3, 8, 8),  # [B, V, 1, C, H, W]
        "video/future_frames": torch.rand(2, 2, 4, 3, 8, 8),  # [B, V, T, C, H, W]
        "text/embeddings": ["pick up the bowl", "open the drawer"],
    }
    print_batch(batch, "train")
    out = capsys.readouterr().out
    assert "video/future_frames: Tensor (2, 2, 4, 3, 8, 8) torch.float32" in out and "text/embeddings: list of 2" in out
    save_batch_frames(batch, ["video/history_frames", "video/future_frames"], str(tmp_path), num_samples=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sample_0.png", "sample_1.png"]
