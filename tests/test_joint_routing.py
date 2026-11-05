"""Joint/separate diffusion must preserve sparse sample alignment and gradients."""

import pytest
import torch
from einops import rearrange
from torch import nn

from tapestry.loaders.libero_dataset import custom_collate
from tapestry.losses import ModalityLossManager
from tapestry.models.base import Tapestry, SequenceInfo


class _Decoder(nn.Module):
    joint_diffusion = True

    def __init__(self, prob):
        super().__init__()
        self.joint_diffusion_mix_prob = prob
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.2))
            for name in ("action", "video", "joint_action", "joint_video")
        })
        self.joint_inputs = []
        self.separate_inputs = {"action": [], "video": []}

    def _output(self, kind, clean, cond, joint):
        # Stand in for VAE encoding: video output shape differs from RGB targets.
        latent = clean[:, 0] if kind == "video" else clean  # [B, T, C, H, W] or [B, T, A]
        weight = self.weights[f"joint_{kind}" if joint else kind]
        ids = cond[:, 0, 0]  # [B]
        shape = "b -> b 1 1" if kind == "action" else "b -> b 1 1 1 1"
        pred = weight * latent.clone() + 0.1 * rearrange(ids, shape)
        out = {
            kind: pred,
            f"{kind}_noise": torch.zeros_like(latent),
            f"{kind}_timesteps": torch.full((len(latent),), 999 if joint else 499),
            f"{kind}_latent_target": latent.clone(),
        }
        if kind == "video":
            out["video_num_views"] = 1
        return out

    def forward_diffusion_training_joint(self, cond, video, actions, video_loss, action_loss, batch):
        self.joint_inputs.append((cond.clone(), actions.clone(), video.clone()))
        torch.testing.assert_close(batch["sample_id"], cond[:, 0, 0])
        return {
            **self._output("action", actions, cond, True),
            **self._output("video", video, cond, True),
        }

    def forward_diffusion_training(self, cond, target, loss, batch):
        kind = "action" if target.ndim == 3 else "video"
        self.separate_inputs[kind].append((cond.clone(), target.clone()))
        torch.testing.assert_close(batch["sample_id"], cond[:, 0, 0])
        return self._output(kind, target, cond, False)


def _model(prob):
    model = Tapestry.__new__(Tapestry)
    nn.Module.__init__(model)
    model.sequence_config = {"loss_mappings": {
        kind: {"decoder": "d", "tokens": "text", "target": f"{kind}/future_frames"}
        for kind in ("action", "video")
    }}
    model.decoders = nn.ModuleDict({"d": _Decoder(prob)})
    losses = ModalityLossManager({
        "action/future_frames": {"weight": 0.3, "predict_future_frames": False},
        "video/future_frames": {"weight": 0.7, "predict_future_frames": False},
    })
    return model, losses


@pytest.mark.parametrize("actions,videos,prob", [
    ([False, True, False, True], [True] * 4, 1.0),
    ([False, True, False, True], [True] * 4, 0.5),
    ([False, True, False, True], [True] * 4, 0.0),
    ([True] * 4, [True] * 4, 1.0),
    ([True] * 4, [True] * 4, 0.5),
    ([True] * 4, [True] * 4, 0.0),
    ([False] * 4, [True] * 4, 1.0),
    ([True] * 4, [False] * 4, 1.0),
    ([False, True, True, True], [True, False, True, True], 1.0),
    ([False, True, True, True], [True, False, True, True], 0.5),
    ([False, True, False, True], [True] * 4, 0.1),
    ([False, True, False, True], [True] * 4, 0.9),
    ([True] * 4, [True] * 4, 0.1),
    ([True] * 4, [True] * 4, 0.9),
])
def test_joint_routing_targets_and_gradients(monkeypatch, actions, videos, prob):
    rows = []
    for i, (has_action, has_video) in enumerate(zip(actions, videos)):
        row = {
            "text/embeddings": torch.tensor([[float(i + 1)]]), "active_modalities": ["text"],
            "sample_id": torch.tensor(float(i + 1)),
        }
        if has_action:
            row["action/future_frames"] = torch.full((2, 3), float(10 + i))  # [T, A]
            row["active_modalities"].append("action")
        if has_video:
            row["video/future_frames"] = torch.full((1, 2, 1, 2, 2), float(20 + i))  # [V, T, C, H, W]
            row["active_modalities"].append("video")
        rows.append(row)
    batch = custom_collate(rows)
    model, losses = _model(prob)
    dec = model.decoders["d"]
    both = [i for i in range(len(rows)) if actions[i] and videos[i]]
    draws = torch.linspace(0.25, 0.75, len(both))
    monkeypatch.setattr(torch, "rand", lambda n, device: draws.to(device))
    joint_ids = [i for i, draw in zip(both, draws) if draw < prob]
    seq = SequenceInfo()
    seq.register_tokens("text", 1)
    out = model._forward_with_loss_mappings(seq, batch["text/embeddings"], batch, losses)

    assert len(dec.joint_inputs) == bool(joint_ids)
    if joint_ids:
        cond, act, vid = dec.joint_inputs[0]
        torch.testing.assert_close(cond, torch.stack([rows[i]["text/embeddings"] for i in joint_ids]))
        torch.testing.assert_close(act, torch.stack([rows[i]["action/future_frames"] for i in joint_ids]))
        torch.testing.assert_close(vid, torch.stack([rows[i]["video/future_frames"] for i in joint_ids]))

    ref_loss = torch.zeros(())
    for kind in ("action", "video"):
        key = f"{kind}/future_frames"
        ids = [i for i, row in enumerate(rows) if key in row]
        separate_ids = [i for i in ids if i not in joint_ids]
        assert len(dec.separate_inputs[kind]) == bool(separate_ids)
        if separate_ids:
            cond, target = dec.separate_inputs[kind][0]
            torch.testing.assert_close(cond, torch.stack([rows[i]["text/embeddings"] for i in separate_ids]))
            torch.testing.assert_close(target, torch.stack([rows[i][key] for i in separate_ids]))
        if not ids:
            assert key not in out
            continue
        clean = torch.stack([rows[i][key] for i in ids])
        if kind == "video":
            clean = clean[:, 0]
            assert out["video_num_views"] == 1
        joint = torch.tensor([i in joint_ids for i in ids])
        ts = torch.where(joint, 999, 499)
        torch.testing.assert_close(out[f"{kind}_latent_target"], clean)
        torch.testing.assert_close(out[f"{kind}_timesteps"], ts)
        torch.testing.assert_close(out[f"{kind}_noise"], torch.zeros_like(clean))
        shape = "b -> b 1 1" if kind == "action" else "b -> b 1 1 1 1"
        weight = torch.stack([dec.weights[f"joint_{kind}" if i in joint_ids else kind] for i in ids])
        cond_ids = torch.tensor([float(i + 1) for i in ids])
        pred = rearrange(weight, shape) * clean + 0.1 * rearrange(cond_ids, shape)
        torch.testing.assert_close(out[key], pred)
        loss = losses.losses[key]
        velocity = loss.noise_scheduler.get_velocity(clean, torch.zeros_like(clean), ts)
        ref_loss = ref_loss + loss.weight * (pred - velocity).square().mean()

    noise = {key: value for key, value in out.items() if key.endswith("_noise")}
    actual_loss = losses(out, batch, noise)["total"]
    torch.testing.assert_close(actual_loss, ref_loss)
    params = list(dec.parameters())
    actual_grads = torch.autograd.grad(actual_loss, params, allow_unused=True)
    ref_grads = torch.autograd.grad(ref_loss, params, allow_unused=True)
    for param, actual, expected in zip(params, actual_grads, ref_grads):
        # Overwritten separate rows can retain a graph with zero gradient.
        actual = torch.zeros_like(param) if actual is None else actual
        expected = torch.zeros_like(param) if expected is None else expected
        torch.testing.assert_close(actual, expected)
    assert any(grad is not None and grad.abs() > 0 for grad in actual_grads)


def test_joint_selection_is_per_sample_bernoulli():
    torch.manual_seed(7)
    n = 2048
    model, losses = _model(0.5)
    ids = torch.arange(float(n))
    tokens = rearrange(ids, "b -> b 1 1")
    seq = SequenceInfo()
    seq.register_tokens("text", 1)
    batch = {
        "text/embeddings": tokens, "sample_id": ids,
        "action/future_frames": rearrange(ids, "b -> b 1 1"),
        "video/future_frames": rearrange(ids, "b -> b 1 1 1 1 1"),
    }
    model._run_joint_diffusion(seq, tokens, batch, losses)
    selected = model.decoders["d"].joint_inputs[0][0][:, 0, 0].long()
    assert 0.45 < len(selected) / n < 0.55
    mask = torch.zeros(n, dtype=torch.bool)
    mask[selected] = True
    # The mixture is per sample, not one random choice for the whole batch.
    assert torch.count_nonzero(mask[1:] != mask[:n - 1]) > n // 3
