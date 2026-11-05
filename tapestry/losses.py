"""Diffusion objectives: v-prediction MSE on video latents and action chunks."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import CogVideoXDDIMScheduler
from einops import rearrange


class DiffusionLoss(nn.Module):
    """Weighted v-prediction MSE under the CogVideoX-2b noise schedule, with uniformly sampled timesteps."""

    # Diffusion timesteps at which to bucket per-sample MSE for monitoring.
    T_BUCKETS: Tuple[int, ...] = (0, 100, 250, 499, 750, 999)

    def __init__(self, weight: float, predict_future_frames: bool):
        super().__init__()
        self.weight = weight
        self.predict_future_frames = predict_future_frames  # video: condition on the clean first frame
        # Same settings as the pretrained CogVideoX-2b scheduler. Also used for sampling at inference.
        self.noise_scheduler = CogVideoXDDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="v_prediction",
            clip_sample_range=1.0,
            sample_max_value=1.0,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
            snr_shift_scale=3.0,
        )
        # Per-timestep MSE buckets for monitoring, allocated on the first batch's device.
        self._t_mse_sums: Optional[torch.Tensor] = None
        self._t_mse_counts: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DiffusionLoss":
        if set(config) != {"weight", "predict_future_frames"}:
            raise ValueError(f"objective options are exactly weight and predict_future_frames, got {sorted(config)}")
        return cls(config["weight"], config["predict_future_frames"])

    def sample_timesteps(self, bsz: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()  # [B]

    def add_noise(self, x_start, noise, timesteps):
        return self.noise_scheduler.add_noise(x_start, noise, timesteps)

    def _update_t_mse_stats(self, timesteps: torch.Tensor, per_sample_mse: torch.Tensor) -> None:
        """Add each sample's MSE [B] to the bucket nearest to its timestep [B]."""
        device = per_sample_mse.device
        if self._t_mse_sums is None or self._t_mse_sums.device != device:
            self._t_mse_sums = torch.zeros(len(self.T_BUCKETS), device=device, dtype=torch.float64)
            self._t_mse_counts = torch.zeros(len(self.T_BUCKETS), device=device, dtype=torch.long)
        buckets = torch.tensor(self.T_BUCKETS, device=device, dtype=timesteps.dtype)
        dist = (rearrange(timesteps, "b -> b 1") - rearrange(buckets, "n -> 1 n")).abs()  # [B, N]
        bucket_idx = dist.argmin(dim=1)  # [B]
        self._t_mse_sums.scatter_add_(0, bucket_idx, per_sample_mse.detach().to(self._t_mse_sums.dtype))
        self._t_mse_counts.scatter_add_(0, bucket_idx, torch.ones_like(bucket_idx))

    def pop_t_mse_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (sums, counts) clones and reset buffers in place. None if no stats yet."""
        if self._t_mse_sums is None:
            return None
        sums = self._t_mse_sums.clone()
        counts = self._t_mse_counts.clone()
        self._t_mse_sums.zero_()
        self._t_mse_counts.zero_()
        return sums, counts

    def forward(self, prediction, latent, noise, timesteps, view_mask, num_views):
        """Weighted MSE between `prediction` and the velocity target of (`latent`, `noise`, `timesteps` [B]).

        Multi-view video arrives with the views concatenated along width, [B, C, T, H, V*W]. `view_mask` [B, V]
        (True = real camera) then keeps the padded views out of the loss. Pass None for both otherwise.
        """
        target = self.noise_scheduler.get_velocity(latent, noise, timesteps)
        if view_mask is None:
            diff = (prediction - target) ** 2
            loss = diff.mean()
            per_sample_mse = rearrange(diff.detach(), "b ... -> b (...)").mean(dim=1)  # [B]
        else:
            B, C, T, H, VW = prediction.shape
            W = VW // num_views
            pred_views = rearrange(prediction, "b c t h (v w) -> b v c t h w", v=num_views, w=W)
            target_views = rearrange(target, "b c t h (v w) -> b v c t h w", v=num_views, w=W)
            mask = rearrange(view_mask.float(), "b v -> b v 1 1 1 1")
            masked_mse = (pred_views - target_views) ** 2 * mask  # [B, V, C, T, H, W]
            loss = masked_mse.sum() / (mask.sum() * C * T * H * W)
            n_real = view_mask.float().sum(dim=1).clamp(min=1)  # [B]
            per_sample_mse = masked_mse.detach().sum(dim=(1, 2, 3, 4, 5)) / (n_real * (C * T * H * W))
        self._update_t_mse_stats(timesteps, per_sample_mse)
        return self.weight * loss


class ModalityLossManager(nn.Module):
    """One DiffusionLoss per entry of `training.objectives`, keyed by target (e.g. 'video/future_frames')."""

    def __init__(self, config: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.target_keys = list(config)
        self.losses = nn.ModuleDict({key: DiffusionLoss.from_config(cfg) for key, cfg in config.items()})

    def forward(
        self, predictions: Dict[str, Any], targets: Dict[str, Any], noise_dict: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Weighted loss for every target the model produced a prediction for, plus their sum under 'total'.

        predictions: model outputs. '<target>' comes with '<modality>_latent_target' and '<modality>_timesteps',
            multi-view video also with 'video_num_views'.
        targets: the batch. Supplies 'active_modalities', the per-sample '<target>_mask' and, when some samples
            have fewer camera views than others, 'video/view_mask'.
        noise_dict: {'<modality>_noise': the noise that was added to the latent target}
        """
        active = {m for sample in targets["active_modalities"] for m in sample}  # active for any sample of the batch
        losses = {}
        for key in self.target_keys:
            modality = key.split("/")[0]
            if modality not in active or key not in predictions or key not in targets:
                continue
            pred = predictions[key]
            name = key.replace("/", "_")  # 'video/future_frames' -> 'video_future_frames' -> 'video_noise', ...
            view_mask, num_views = None, None
            if modality == "video" and "video/view_mask" in targets:  # custom_collate adds it only when view counts differ
                view_mask, num_views = targets["video/view_mask"], predictions["video_num_views"]  # [B, V], V
                if view_mask.shape[0] != pred.shape[0]:  # predictions only cover the samples that have this target
                    view_mask = view_mask[targets[f"{key}_mask"]]
            losses[key] = self.losses[key](
                pred,
                predictions[name.replace("future_frames", "latent_target")],
                noise_dict[name.replace("future_frames", "noise")],
                predictions[name.replace("future_frames", "timesteps")],
                view_mask,
                num_views,
            )
        losses["total"] = sum(losses.values()) if losses else torch.tensor(0.0, requires_grad=True)
        return losses

    def pop_per_timestep_mse_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Pop the per-timestep MSE buckets of every objective that has seen a batch since the last pop."""
        stats = {key: loss.pop_t_mse_stats() for key, loss in self.losses.items()}
        return {key: s for key, s in stats.items() if s is not None}
