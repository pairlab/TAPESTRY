"""Gradient norms per parameter group, to check which parts of a model the loss reaches.

Call between accelerator.backward(loss) and optimizer.step(), e.g. in train.py:

    norms = grad_norms(accelerator.unwrap_model(model), depth=4)
    wandb.log({f"grad_norm/{k}": v for k, v in norms.items()}, commit=False)
"""
from collections import defaultdict
from typing import Dict

import torch


def grad_norms(model: torch.nn.Module, depth: int) -> Dict[str, float]:
    """L2 gradient norm of each group of parameters whose names share their first `depth` components.

    With depth=4, 'decoders.cogvideox.cogvideox.transformer_blocks.0.attn1.to_q.weight' counts toward
    'decoders.cogvideox.cogvideox.transformer_blocks'. Parameters without a gradient are left out, as in
    torch.nn.utils.clip_grad_norm_, so a group missing from the result received no gradient.
    """
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if p.grad is not None:
            groups[".".join(name.split(".")[:depth])].append(p.grad.detach().float().norm())
    return {k: torch.stack(v).norm().item() for k, v in groups.items()}
