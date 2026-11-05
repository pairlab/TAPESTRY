"""DiffusionLoss: uniform timestep sampling, strict objective options, multi-view loss masking."""

import pytest
import torch

from tapestry.losses import DiffusionLoss, ModalityLossManager


def test_timesteps_are_uniform_over_the_training_schedule():
    loss = DiffusionLoss(1.0, True)
    torch.manual_seed(0)
    t = loss.sample_timesteps(8192, torch.device("cpu"))  # [B]
    torch.manual_seed(0)
    torch.testing.assert_close(t, torch.randint(0, 1000, (8192,)))
    assert t.dtype == torch.long and t.min() >= 0 and t.max() < 1000
    assert abs(t.float().mean().item() - 499.5) < 15


@pytest.mark.parametrize("config", [
    {"weight": 1.0, "predict_future_frames": True, "noise_scheduler": "ddpm"},
    {"weight": 1.0, "predict_future_frames": True, "num_inference_steps": 50},
    {"weight": 1.0},
])
def test_objective_options_are_exactly_weight_and_predict_future_frames(config):
    DiffusionLoss.from_config({"weight": 1.0, "predict_future_frames": True})
    with pytest.raises(ValueError, match="predict_future_frames"):
        DiffusionLoss.from_config(config)


def test_padded_views_do_not_contribute_to_the_video_loss():
    key = "video/future_frames"
    losses = ModalityLossManager({key: {"weight": 0.5, "predict_future_frames": True}})
    B, V, C, T, H, W = 2, 2, 4, 3, 2, 2
    torch.manual_seed(0)
    latent, noise = torch.randn(B, C, T, H, V * W), torch.randn(B, C, T, H, V * W)
    ts = torch.tensor([100, 900])
    targets = {
        "active_modalities": [["video"]] * B,
        key: torch.zeros(B, V, T, 3, 8, 8),
        "video/view_mask": torch.tensor([[True, True], [True, False]]),  # sample 1 has one camera
    }

    def run(pred):
        out = {key: pred, "video_latent_target": latent, "video_timesteps": ts, "video_num_views": V}
        return losses(out, targets, {"video_noise": noise})["total"]

    pred = torch.randn(B, C, T, H, V * W, requires_grad=True)
    total = run(pred)
    total.backward()
    assert pred.grad[1, ..., W:].abs().max() == 0  # padded view of sample 1
    assert pred.grad[1, ..., :W].abs().min() > 0

    velocity = losses.losses[key].noise_scheduler.get_velocity(latent, noise, ts)
    err = (pred - velocity).detach() ** 2
    real = torch.cat([err[0].flatten(), err[1, ..., :W].flatten()])
    torch.testing.assert_close(total.detach(), 0.5 * real.mean())

    changed = pred.detach().clone()
    changed[1, ..., W:] += 10.0
    torch.testing.assert_close(run(changed), total.detach())


def test_video_loss_without_view_mask_covers_every_view():
    # custom_collate adds 'video/view_mask' only when the samples have different numbers of views
    key = "video/future_frames"
    losses = ModalityLossManager({key: {"weight": 0.5, "predict_future_frames": True}})
    B, V, C, T, H, W = 2, 2, 4, 3, 2, 2
    torch.manual_seed(0)
    latent, noise, pred = (torch.randn(B, C, T, H, V * W) for _ in range(3))
    ts = torch.tensor([100, 900])
    targets = {"active_modalities": [["video"]] * B, key: torch.zeros(B, V, T, 3, 8, 8)}
    out = {key: pred, "video_latent_target": latent, "video_timesteps": ts, "video_num_views": V}
    total = losses(out, targets, {"video_noise": noise})["total"]
    velocity = losses.losses[key].noise_scheduler.get_velocity(latent, noise, ts)
    torch.testing.assert_close(total, 0.5 * ((pred - velocity) ** 2).mean())
