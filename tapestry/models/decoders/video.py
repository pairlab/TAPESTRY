"""CogVideoX as the joint video and action denoiser of TAPESTRY."""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed, get_1d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

from tapestry.video_diffusion.cogvideox_transformer_3d import CogVideoXTransformer3DModel, ActionPatchProj


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    vae_scale_factor_spatial: int,
    num_frames: int,
    transformer_config,
    device: Optional[torch.device],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build video RoPE tables [T*H/P*W/P, head_dim] matching diffusers."""

    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    p = transformer_config.patch_size
    p_t = transformer_config.patch_size_t

    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + p_t - 1) // p_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
            device=device,
        )

    return freqs_cos, freqs_sin


def _action_temporal_rotary_pos_embed(
    transformer_config,
    action_horizon: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE for action tokens at temporal-only positions.

    Mirrors the 2B sincos behavior where actions reshape to (T_act, 1, 1) image patches
    and pick up pos emb varying only along time. Spatial dims rotate with cos=1, sin=0
    (identity), temporal dim rotates with the integer time index.
    """
    embed_dim = transformer_config.attention_head_dim
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    grid_t = torch.arange(action_horizon, device=device, dtype=torch.float32)
    grid_zero = torch.zeros(1, device=device, dtype=torch.float32)

    t_cos, t_sin = get_1d_rotary_pos_embed(dim_t, grid_t, theta=10000, use_real=True)  # [T_act, dim_t]
    h_cos, h_sin = get_1d_rotary_pos_embed(dim_h, grid_zero, theta=10000, use_real=True)  # [1, dim_h]
    w_cos, w_sin = get_1d_rotary_pos_embed(dim_w, grid_zero, theta=10000, use_real=True)  # [1, dim_w]

    h_cos, h_sin, w_cos, w_sin = (repeat(x, "1 d -> t d", t=action_horizon) for x in (h_cos, h_sin, w_cos, w_sin))

    cos = torch.cat([t_cos, h_cos, w_cos], dim=1)  # [T_act, head_dim]
    sin = torch.cat([t_sin, h_sin, w_sin], dim=1)
    return cos, sin


def _identity_rotary_pos_embed(
    head_dim: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identity rotation: cos=1, sin=0 → apply_rotary_emb is a no-op."""
    cos = torch.ones(seq_len, head_dim, device=device, dtype=torch.float32)
    sin = torch.zeros(seq_len, head_dim, device=device, dtype=torch.float32)
    return cos, sin


class CogVideoXDecoder(nn.Module):
    """Denoise video latents [B, C, T, H, W] and actions [B, T, A] with CogVideoX."""

    def __init__(
        self,
        pretrained_model_name: str,
        use_pretrained: bool,
        freeze_pretrained_weights: bool,
        use_lora: bool,
        lora_config: Optional[Dict[str, Any]],
        gradient_checkpointing: bool,
        action_dim: int,
        action_horizon: int,
        max_text_seq_length: int,
        max_views: int,
        use_vae: bool,
        vae_scale_factor: float,
        vae_scale_factor_spatial: int,
        vae_scale_factor_temporal: int,
        text_cond_dropout_prob: float,
        action_denoising_dropout: float,
        joint_diffusion: bool,
        joint_coupled_timesteps: bool,
        joint_diffusion_mix_prob: float,
        joint_asymmetric_action_attn: bool,
        from_checkpoint: bool,
    ):
        super().__init__()
        if not 0.0 <= joint_diffusion_mix_prob <= 1.0:
            raise ValueError(f"joint_diffusion_mix_prob must be in [0, 1], got {joint_diffusion_mix_prob}")
        if not joint_diffusion and joint_diffusion_mix_prob != 1.0:
            raise ValueError("joint_diffusion_mix_prob != 1.0 requires joint_diffusion=true")
        if joint_asymmetric_action_attn and not joint_diffusion:
            raise ValueError("joint_asymmetric_action_attn requires joint_diffusion=true")

        self.action_dim, self.action_horizon, self.max_views = action_dim, action_horizon, max_views
        self.vae_scale_factor = vae_scale_factor
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.joint_diffusion = joint_diffusion
        self.joint_coupled_timesteps = joint_coupled_timesteps
        self.joint_diffusion_mix_prob = joint_diffusion_mix_prob
        self.joint_asymmetric_action_attn = joint_asymmetric_action_attn

        # Architecture comes from the HF transformer config, not the training YAML.
        over = {"max_text_seq_length": max_text_seq_length, "action_denoising_dropout": action_denoising_dropout}
        if use_pretrained:
            core, info = CogVideoXTransformer3DModel.from_pretrained(
                pretrained_model_name, subfolder="transformer", use_safetensors=True, output_loading_info=True,
                torch_dtype=torch.float16 if from_checkpoint else torch.float32,
                low_cpu_mem_usage=from_checkpoint, **over,
            )
            bad = {k: v for k, v in info.items() if v}
            if bad:
                raise RuntimeError(f"CogVideoX weights did not load cleanly: {bad}")
        else:
            config = CogVideoXTransformer3DModel.load_config(pretrained_model_name, subfolder="transformer")
            core = CogVideoXTransformer3DModel.from_config(config, **over)

        # Attach after loading: diffusers disables torch.nn.init inside from_pretrained.
        core.action_input_proj = ActionPatchProj(action_dim, core.patch_embed)
        self.cogvideox = get_peft_model(core, LoraConfig(**lora_config)) if use_lora else core
        if freeze_pretrained_weights:
            for name, p in self.cogvideox.named_parameters():
                p.requires_grad = "lora_" in name or name.endswith("action_input_proj.act_proj.weight")
        else:
            # PEFT freezes the base weights; full fine-tuning must re-enable them.
            self.cogvideox.requires_grad_(True)
        if gradient_checkpointing:
            core.enable_gradient_checkpointing()
        if use_vae:
            self.vae = AutoencoderKLCogVideoX.from_pretrained(
                pretrained_model_name, subfolder="vae",
            ).to(core.dtype).requires_grad_(False)

        C = core.config.in_channels
        self.latent_view_mask = nn.Parameter(torch.zeros(1, C, 1, 1, 1))  # replaces padded views
        self.view_embeddings = nn.Parameter(torch.randn(max_views, C, 1, 1, 1) * 0.01)  # identifies each view
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"CogVideoXDecoder: {trainable:,} trainable / {total:,} total parameters")

    def build_encoder_hidden_states(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        """T5 embeddings [B, L, D] cast for the transformer, with per-sample text dropout (zeros) in training for CFG."""
        device = next(self.cogvideox.parameters()).device
        enc = cond_tokens.to(device=device, dtype=self.cogvideox.dtype)
        if enc.shape[1] != self.cogvideox.config.max_text_seq_length:
            raise ValueError(
                f"expected T5 embeddings padded to {self.cogvideox.config.max_text_seq_length} tokens, got {enc.shape[1]}"
            )
        if self.training and self.text_cond_dropout_prob > 0.0:
            drop = torch.rand(enc.shape[0], device=device) < self.text_cond_dropout_prob  # [B]
            enc = enc * rearrange(~drop, "b -> b 1 1").to(enc.dtype)
        return enc

    def _build_prepended_tokens(self, batch, device, dtype):
        """Build first-frame VAE patch tokens [B, N, D]."""
        if batch is None:
            raise ValueError("First-frame conditioning requires batch to be provided")

        if "video/history_frames" in batch:
            frames = batch["video/history_frames"]
        elif "video/future_frames" in batch:
            frames = batch["video/future_frames"]
        else:
            raise ValueError(
                "First-frame conditioning requires 'video/history_frames' or "
                "'video/future_frames' in batch. "
                f"Available keys: {list(batch.keys())}"
            )

        if frames.dim() == 6:  # [B, V, T, C, H, W]: all views, concatenated along the width
            ff_lat = self.encode_multiview_to_latents(
                frames,
                view_mask=batch.get("video/view_mask"),
                get_single_frame=0,
            )
        elif frames.dim() == 5:  # [B, T, C, H, W]
            ff_lat = self.get_latents_from_video(frames, get_single_frame=0)
        else:
            raise ValueError(
                f"Expected video frames with shape [B, V, T, C, H, W] or [B, T, C, H, W], got {tuple(frames.shape)}"
            )

        # ff_lat: [B, C, T_lat, H_lat, W_lat]
        ff_lat = ff_lat.to(device=device, dtype=self.cogvideox.dtype)

        ff_lat = rearrange(ff_lat, "b c t h w -> b t c h w")
        bsz = ff_lat.shape[0]
        max_text_len = self.cogvideox.config.max_text_seq_length
        text_dim = self.cogvideox.config.text_embed_dim
        dummy_text = torch.zeros(bsz, max_text_len, text_dim, device=device, dtype=self.cogvideox.dtype)

        ff_patch = self.cogvideox.patch_embed(dummy_text, ff_lat)
        ff_tok = ff_patch[:, max_text_len:]
        # ff_tok: [B, N, inner_dim]
        if ff_tok.dim() != 3:
            raise ValueError(f"Expected first-frame tokens [B, N, D], got {tuple(ff_tok.shape)}")

        return ff_tok.to(device=device, dtype=dtype)

    def _unpatchify_video_tokens(
        self,
        tokens: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Unpatchify video tokens into latent video frames.

        Args:
            tokens: [B, N, P] token tensor from proj_out
            num_frames: Temporal length (latent frames)
            height: Latent height
            width: Latent width
        Returns:
            Latent video [B, C, T, H, W]
        """
        p = int(self.cogvideox.config.patch_size)
        p_t = self.cogvideox.config.patch_size_t
        if height % p != 0 or width % p != 0:
            raise ValueError(f"height/width must be divisible by patch_size (got {height}x{width} vs {p})")

        h_p = height // p
        w_p = width // p
        bsz, n_tok, p_dim = tokens.shape

        if p_t is None:
            exp_n = int(num_frames) * int(h_p) * int(w_p)
            if n_tok != exp_n:
                raise ValueError(f"Expected {exp_n} video tokens, got {n_tok}")
            p_area = int(p) * int(p)
            if p_dim % p_area != 0:
                raise ValueError(f"token dim {p_dim} not divisible by patch area {p_area}")
            c_out = p_dim // p_area
            return rearrange(
                tokens,
                "b (t h w) (c p1 p2) -> b c t (h p1) (w p2)",
                t=int(num_frames),
                h=int(h_p),
                w=int(w_p),
                c=int(c_out),
                p1=int(p),
                p2=int(p),
            )

        if int(p_t) <= 0:
            raise ValueError(f"patch_size_t must be positive, got {p_t}")
        t_lat = (int(num_frames) + int(p_t) - 1) // int(p_t)
        exp_n = int(t_lat) * int(h_p) * int(w_p)
        if n_tok != exp_n:
            raise ValueError(f"Expected {exp_n} video tokens, got {n_tok}")
        p_area = int(p_t) * int(p) * int(p)
        if p_dim % p_area != 0:
            raise ValueError(f"token dim {p_dim} not divisible by patch area {p_area}")
        c_out = p_dim // p_area
        return rearrange(
            tokens,
            "b (t h w) (c pt p1 p2) -> b c (t pt) (h p1) (w p2)",
            t=int(t_lat),
            h=int(h_p),
            w=int(w_p),
            c=int(c_out),
            pt=int(p_t),
            p1=int(p),
            p2=int(p),
        )

    def _prepare_video_latents(
        self,
        target_video: torch.Tensor,
        diffusion_loss,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool, int, Optional[torch.Tensor], bool]:
        """Prepare video targets [B, C, T, H, W], optional first-frame latents, and view metadata."""
        if target_video.dim() not in (5, 6):
            raise ValueError(
                f"Expected target_video with shape [B, T, C, H, W] or [B, V, T, C, H, W], got {tuple(target_video.shape)}"
            )

        use_joint_mv = target_video.dim() == 6  # all views are diffused together, concatenated along the width
        num_views = target_video.shape[1] if use_joint_mv else 1
        view_mask = batch.get("video/view_mask") if use_joint_mv else None

        use_first_frame_cond = diffusion_loss.predict_future_frames

        if use_joint_mv:
            first_frame = None
            if use_first_frame_cond:
                first_frame = self.encode_multiview_to_latents(
                    target_video,
                    view_mask=view_mask,
                    get_single_frame=0,
                ).to(self.cogvideox.dtype)
            future_frames = self.encode_multiview_to_latents(
                target_video,
                view_mask=view_mask,
                get_single_frame=None,
            ).to(self.cogvideox.dtype)
        else:
            first_frame = None
            if use_first_frame_cond:
                first_frame = self.get_latents_from_video(
                    target_video, get_single_frame=0
                ).to(self.cogvideox.dtype)
            future_frames = self.get_latents_from_video(
                target_video, get_single_frame=None
            ).to(self.cogvideox.dtype)

        return future_frames, first_frame, use_first_frame_cond, num_views, view_mask, use_joint_mv

    def get_latents_from_video(self, frames: torch.Tensor, get_single_frame: Optional[int]):
        """VAE-encode frames [B, T, C, H, W] in [0, 1] into latents [B, C_lat, T_lat, H_lat, W_lat].

        get_single_frame: index of one frame to encode on its own. It is repeated vae_scale_factor_temporal
        times so that it fills exactly one latent frame. None encodes the whole clip.
        """
        if frames.dim() != 5:
            raise ValueError(f"get_latents_from_video expects [B, T, C, H, W], got {tuple(frames.shape)}")
        if get_single_frame is not None:
            if not 0 <= get_single_frame < frames.shape[1]:
                raise ValueError(f"get_single_frame index {get_single_frame} out of range for T={frames.shape[1]}")
            frames = repeat(frames[:, get_single_frame:get_single_frame + 1], "b 1 c h w -> b t c h w", t=self.vae_scale_factor_temporal)

        x = rearrange(frames * 2.0 - 1.0, "b t c h w -> b c t h w").to(self.cogvideox.dtype)   # the VAE takes [-1, 1]
        with torch.no_grad():
            return self.vae.encode(x).latent_dist.sample() * self.vae_scale_factor

    def encode_multiview_to_latents(self, video: torch.Tensor, view_mask: Optional[torch.Tensor], get_single_frame: Optional[int]) -> torch.Tensor:
        """VAE-encode [B, V, T, C, H, W] into width-concatenated latents [B, C_lat, T_lat, H_lat, V*W_lat].

        Every view gets its learned view embedding. Views with view_mask False (padding) are replaced by the learned mask latent.
        """
        B, V = video.shape[:2]
        if V > self.max_views:
            raise ValueError(f"Expected at most {self.max_views} views, got {V}")
        flat = self.get_latents_from_video(rearrange(video, "b v t c h w -> (b v) t c h w"), get_single_frame)   # one VAE call for all views
        stacked = rearrange(flat, "(b v) c t h w -> b v c t h w", b=B, v=V)
        for i in range(V):   # per-view addition keeps the backward reduction order bitwise
            stacked[:, i] = stacked[:, i] + self.view_embeddings[i]
        if view_mask is not None:
            stacked = torch.where(rearrange(view_mask, "b v -> b v 1 1 1 1"), stacked, rearrange(self.latent_view_mask, "1 c 1 1 1 -> 1 1 c 1 1 1"))
        return rearrange(stacked, "b v c t h w -> b c t h (v w)")

    def forward(self, noisy_latents, timesteps, conditioning):
        """Predict video noise/velocity [B, C, T, H, W] from T5 conditioning."""
        batch_size, channels, num_frames, height, width = noisy_latents.shape
        device = noisy_latents.device

        # Convert latents to CogVideoX format: [B, T, C, H, W]
        hidden_states = rearrange(noisy_latents, "b c t h w -> b t c h w")

        # Build T5 conditioning for the transformer.
        encoder_hidden_states = self.build_encoder_hidden_states(conditioning)

        # Prepare rotary embeddings if needed
        image_rotary_emb = None
        if self.cogvideox.config.use_rotary_positional_embeddings:
            image_rotary_emb = prepare_rotary_positional_embeddings(
                height=height * self.vae_scale_factor_spatial,  # Convert back to pixel space
                width=width * self.vae_scale_factor_spatial,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                num_frames=num_frames,
                device=device,
                transformer_config=self.cogvideox.config,
            )

        # Forward through CogVideoX
        hidden_states = hidden_states.to(self.cogvideox.dtype)
        timesteps = timesteps.to(self.cogvideox.dtype)

        model_output = self.cogvideox(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        # Convert output back to [B, C, T, H, W] format to match expected output
        model_output = rearrange(model_output, "b t c h w -> b c t h w")

        return {'video': model_output}

    def forward_diffusion_training(
        self,
        conditioning: torch.Tensor,
        target_video: torch.Tensor,
        diffusion_loss,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """Noise video latents or actions and return predictions with diffusion targets."""
        device = target_video.device

        # Detect if we're processing actions (3D) or videos (5D/6D)
        if target_video.dim() == 3:  # [B, T, action_dim]
            # Action mode: target_video is actually actions [B, T, action_dim]
            return self._forward_diffusion_training_actions(
                conditioning, target_video, diffusion_loss, batch=batch
            )

        future_frames, first_frame, use_first_frame_cond, num_views, _, use_joint_mv = self._prepare_video_latents(
            target_video, diffusion_loss, batch
        )
        effective_batch_size = future_frames.shape[0]

        # Sample diffusion timesteps using the configured training sampler.
        timesteps = diffusion_loss.sample_timesteps(effective_batch_size, device)

        # Generate noise only for future frames
        noise_future = torch.randn_like(future_frames)

        # Add noise only to future frames
        noisy_future_frames = diffusion_loss.add_noise(future_frames, noise_future, timesteps)

        # Build input latents based on conditioning mode
        if use_first_frame_cond:
            # Standard mode: Temporal-concatenate first frame with noisy future frames
            noisy_latent_video = torch.cat([first_frame, noisy_future_frames], dim=2)
        else:
            # Pure generation: No first-frame conditioning
            noisy_latent_video = noisy_future_frames

        # Forward through CogVideoX decoder
        decoder_output = self(
            noisy_latent_video,
            timesteps,
            conditioning,
        )['video']

        # Strip the prepended first frame from the prediction.
        if not use_first_frame_cond:
            video_output = decoder_output
        else:
            t_start = decoder_output.shape[2] - future_frames.shape[2]
            video_output = decoder_output[:, :, t_start:]

        result = {
            'video': video_output,
            'video_noise': noise_future,
            'video_timesteps': timesteps,
            'video_latent_target': future_frames
        }

        # Include num_views for joint multi-view loss masking
        if use_joint_mv:
            result['video_num_views'] = num_views

        return result

    def _run_joint_transformer(
        self,
        vid_lat: torch.Tensor,
        act_noisy: torch.Tensor,
        t_vid: torch.Tensor,
        t_act: torch.Tensor,
        enc_h: torch.Tensor,
        prep_tok: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denoise video [B, C, T, H, W] and actions [B, T_act, A] with neutral prepended tokens [B, N, D]."""
        if enc_h is None or not isinstance(enc_h, torch.Tensor):
            raise ValueError("joint_diffusion requires encoder_hidden_states tensor")
        if t_vid.shape != t_act.shape:
            raise ValueError(
                f"t_vid and t_act must have identical shapes, got {tuple(t_vid.shape)} and {tuple(t_act.shape)}"
            )

        cog = self.cogvideox
        use_rope = cog.config.use_rotary_positional_embeddings
        if cog.ofs_embedding is not None:
            raise ValueError("joint_diffusion does not support ofs_embedding without explicit ofs input")

        # vid_lat: [B, C, T, H, W] -> [B, T, C, H, W]
        vid_in = rearrange(vid_lat, "b c t h w -> b t c h w")
        vid_in = vid_in.to(cog.dtype)
        act_in = act_noisy.to(cog.dtype)
        t_vid_in = t_vid.to(cog.dtype)
        t_act_in = t_act.to(cog.dtype)

        # Time conditioning:
        # - t_vid_tok / t_act_tok: [B, d_tok] for modality-specific token bias
        # - emb: [B, d_t] block modulation input (never averaged across modalities)
        t_vid_tok = cog.time_proj(t_vid_in).to(dtype=vid_in.dtype)
        t_act_tok = cog.time_proj(t_act_in).to(dtype=vid_in.dtype)
        emb = cog.time_embedding(t_vid_tok, None)
        if not torch.equal(t_vid, t_act):
            emb = torch.zeros_like(emb)

        # Patch embeddings for video and actions (text tokens reused from video path)
        vid_comb = cog.patch_embed(enc_h, vid_in)
        vid_comb = cog.embedding_dropout(vid_comb)
        txt_len = enc_h.shape[1]
        enc_tok = vid_comb[:, :txt_len]
        vid_tok = vid_comb[:, txt_len:]

        act_img = cog.action_input_proj(act_in)  # [B, T_act, C, P, P]
        act_comb = cog.patch_embed(enc_h, act_img)
        act_comb = cog.embedding_dropout(act_comb)
        act_tok = act_comb[:, txt_len:]
        if act_tok.shape[1] != act_in.shape[1]:
            raise ValueError(f"Expected {act_in.shape[1]} action tokens, got {act_tok.shape[1]}")

        # Modality-specific timestep token bias:
        # - vid_tok: [B, N_v, d_tok] gets t_vid_tok
        # - act_tok: [B, N_a, d_tok] gets t_act_tok
        # - prep_tok is appended after this and intentionally remains timestep-neutral
        v_bias = repeat(t_vid_tok, "b d -> b n d", n=vid_tok.shape[1])
        a_bias = repeat(t_act_tok, "b d -> b n d", n=act_tok.shape[1])
        vid_tok = vid_tok + v_bias
        act_tok = act_tok + a_bias

        if prep_tok.dim() != 3:
            raise ValueError(f"prepended tokens must have shape [B, N, D], got {tuple(prep_tok.shape)}")
        prep_tok = prep_tok.to(dtype=act_tok.dtype, device=act_tok.device)
        act_tok = torch.cat([prep_tok, act_tok], dim=1)
        prep_len = prep_tok.shape[1]

        vid_len = vid_tok.shape[1]
        act_len = act_in.shape[1]
        act_tok_len = act_tok.shape[1]

        hid = torch.cat([vid_tok, act_tok], dim=1)

        # Video tokens cannot attend to action tokens in [text, vid, act].
        attn_kwargs: Dict[str, Any] = {}
        if self.joint_asymmetric_action_attn:
            txt_len = enc_tok.shape[1]
            total = txt_len + vid_len + act_tok_len
            # mask[i, j] = True means token i CAN attend to token j
            mask = torch.ones(1, 1, total, total, dtype=torch.bool, device=hid.device)
            vid_start = txt_len
            vid_end = txt_len + vid_len
            act_start = txt_len + vid_len
            # Block video rows from attending to action columns
            mask[:, :, vid_start:vid_end, act_start:] = False
            attn_kwargs["attention_mask"] = mask

        # Video uses 3D RoPE, actions temporal-only RoPE, and prepended tokens identity RoPE.
        image_rotary_emb = None
        if use_rope:
            _, _, t_lat, h_lat, w_lat = vid_lat.shape
            v_cos, v_sin = prepare_rotary_positional_embeddings(
                height=h_lat * self.vae_scale_factor_spatial,
                width=w_lat * self.vae_scale_factor_spatial,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                num_frames=t_lat,
                device=hid.device,
                transformer_config=cog.config,
            )
            head_dim = v_cos.shape[1]
            cos_parts = [v_cos]
            sin_parts = [v_sin]
            p_cos, p_sin = _identity_rotary_pos_embed(head_dim, prep_len, device=hid.device)
            cos_parts.append(p_cos)
            sin_parts.append(p_sin)
            a_cos, a_sin = _action_temporal_rotary_pos_embed(
                transformer_config=cog.config,
                action_horizon=act_len,
                device=hid.device,
            )
            cos_parts.append(a_cos)
            sin_parts.append(a_sin)
            image_rotary_emb = (torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0))
            attn_kwargs["rope_seq_length"] = vid_len + act_tok_len

        attn_kwargs_pass = attn_kwargs if attn_kwargs else None

        # Transformer blocks
        for block in cog.transformer_blocks:
            if torch.is_grad_enabled() and cog.gradient_checkpointing:
                hid, enc_tok = cog._gradient_checkpointing_func(
                    block,
                    hid,
                    enc_tok,
                    emb,
                    image_rotary_emb,
                    attn_kwargs_pass,
                )
            else:
                hid, enc_tok = block(
                    hidden_states=hid,
                    encoder_hidden_states=enc_tok,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attn_kwargs_pass,
                )

        hid = cog.norm_final(hid)
        hid = cog.norm_out(hid, temb=emb)

        vid_h = hid[:, :vid_len]
        act_h_full = hid[:, vid_len : vid_len + act_tok_len]
        act_h = act_h_full[:, prep_len:]
        if act_h.shape[1] != act_len:
            raise ValueError(f"Expected {act_len} action tokens, got {act_h.shape[1]}")

        vid_proj = cog.proj_out(vid_h)
        bsz, _, t_lat, h_lat, w_lat = vid_lat.shape
        vid_out = self._unpatchify_video_tokens(vid_proj, t_lat, h_lat, w_lat)

        act_proj = cog.proj_out(act_h)
        if act_proj.shape[2] != cog.action_input_proj.patch_dim:
            raise ValueError("proj_out output dim must equal the action patch_dim")
        act_out = cog.action_input_proj.patch_to_actions(act_proj)

        return vid_out, act_out

    def forward_diffusion_training_joint(
        self,
        conditioning: torch.Tensor,
        target_video: torch.Tensor,
        target_actions: torch.Tensor,
        video_loss,
        action_loss,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """Train video and actions in one DiT pass with shared or modality-specific timesteps."""
        if not self.joint_diffusion:
            raise ValueError("forward_diffusion_training_joint requires joint_diffusion=true")

        if target_video.dim() == 3:
            raise ValueError("target_video must be video frames, not action tensor")
        if target_actions.dim() != 3:
            raise ValueError(f"target_actions must have shape [B, T, A], got {tuple(target_actions.shape)}")

        device = target_video.device

        future_frames, first_frame, use_first_frame_cond, num_views, _, use_joint_mv = self._prepare_video_latents(
            target_video, video_loss, batch
        )
        bsz = future_frames.shape[0]

        t_vid = video_loss.sample_timesteps(bsz, device)  # [B]
        t_act = t_vid if self.joint_coupled_timesteps else action_loss.sample_timesteps(bsz, device)  # [B]

        noise_vid = torch.randn_like(future_frames)
        noise_act = torch.randn_like(target_actions)

        noisy_vid = video_loss.add_noise(future_frames, noise_vid, t_vid)
        noisy_act = action_loss.add_noise(target_actions, noise_act, t_act)

        if use_first_frame_cond:
            noisy_lat = torch.cat([first_frame, noisy_vid], dim=2)
        else:
            noisy_lat = noisy_vid

        # Conditioning tokens
        enc_h = self.build_encoder_hidden_states(conditioning)

        prep_tok = self._build_prepended_tokens(batch, device, self.cogvideox.dtype)

        vid_out, act_out = self._run_joint_transformer(
            noisy_lat,
            noisy_act,
            t_vid,
            t_act,
            enc_h,
            prep_tok,
        )

        if not use_first_frame_cond:
            vid_pred = vid_out
        else:
            t_keep = int(future_frames.shape[2])
            t_start = int(vid_out.shape[2] - t_keep)
            vid_pred = vid_out[:, :, t_start:]

        result = {
            "video": vid_pred,
            "video_noise": noise_vid,
            "video_timesteps": t_vid,
            "video_latent_target": future_frames,
            "action": act_out,
            "action_noise": noise_act,
            "action_timesteps": t_act,
            "action_latent_target": target_actions,
        }
        if use_joint_mv:
            result["video_num_views"] = num_views

        return result

    def _forward_diffusion_training_actions(
        self,
        conditioning: torch.Tensor,
        target_actions: torch.Tensor,
        diffusion_loss,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """Noise actions [B, T, A] and predict their diffusion targets without VAE encoding."""
        device = target_actions.device
        B, T, action_dim = target_actions.shape

        # Sample diffusion timesteps using the configured training sampler.
        timesteps = diffusion_loss.sample_timesteps(B, device)

        # Generate noise for actions
        noise = torch.randn_like(target_actions)

        # Add noise to actions
        noisy_actions = diffusion_loss.add_noise(target_actions, noise, timesteps)

        # Build encoder_hidden_states from the conditioning tokens (text only)
        encoder_hidden_states = self.build_encoder_hidden_states(conditioning)

        # Build first-frame latent tokens for action conditioning.
        prepended_tokens = self._build_prepended_tokens(batch, device, self.cogvideox.dtype)
        # prepended_tokens: [B, num_tokens, inner_dim]

        # Forward through CogVideoX with decode_actions=True
        hidden_states = noisy_actions  # [B, T, action_dim]
        hidden_states = hidden_states.to(self.cogvideox.dtype)
        timesteps_dtype = timesteps.to(self.cogvideox.dtype)

        model_output = self.cogvideox(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps_dtype,
            image_rotary_emb=None,
            return_dict=False,
            decode_actions=True,
            prepended_hidden_states=prepended_tokens,
        )[0]

        result = {
            'action': model_output,
            'action_noise': noise,
            'action_timesteps': timesteps,
            'action_latent_target': target_actions
        }

        return result

    def decode_actions_with_diffusion(self, conditioning, noise_scheduler, batch, num_inference_steps):
        """Denoise actions [B, action_horizon, action_dim] with first-frame conditioning."""
        device = conditioning.device
        batch_size = conditioning.shape[0]

        # Initialize with random noise in action space
        action_horizon = self.action_horizon
        action_dim = self.action_dim
        actions = torch.randn(batch_size, action_horizon, action_dim, device=device)

        # Build encoder_hidden_states for conditioning (text only)
        encoder_hidden_states = self.build_encoder_hidden_states(conditioning).to(device)

        # Build first-frame latent tokens for action conditioning.
        prepended_tokens = self._build_prepended_tokens(batch, device, self.cogvideox.dtype)
        # prepended_tokens: [B, num_tokens, inner_dim]

        # Iterative denoising
        noise_scheduler.set_timesteps(num_inference_steps, device=device)

        for t in noise_scheduler.timesteps:
            # Expand timestep to batch dimension
            t_value = t.item() if isinstance(t, torch.Tensor) else t
            timesteps = torch.full((batch_size,), t_value, device=device, dtype=torch.long)

            # Keep scheduler inputs in the model dtype.
            actions = actions.to(dtype=self.cogvideox.dtype, device=device)
            timesteps_dtype = timesteps.to(dtype=self.cogvideox.dtype, device=device)

            # Predict noise using action decoding mode
            # First-frame tokens are prepended after action patch embedding.
            noise_pred = self.cogvideox(
                hidden_states=actions,  # [B, T, action_dim]
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps_dtype,
                image_rotary_emb=None,
                return_dict=False,
                decode_actions=True,
                prepended_hidden_states=prepended_tokens,  # [B, num_tokens, inner_dim]
            )[0]
            # noise_pred: [B, T, action_dim] - prepended tokens already discarded inside cogvideox

            # Denoise step
            actions = noise_scheduler.step(noise_pred, t, actions).prev_sample.to(device)

        return actions

    def decode_actions_with_joint_diffusion(self, conditioning, noise_scheduler, batch, num_inference_steps, predict_future_frames):
        """Jointly denoise video and actions, returning actions [B, action_horizon, action_dim]."""
        if "video/history_frames" not in batch:
            raise KeyError("decode_actions_with_joint_diffusion requires 'video/history_frames' in batch")

        device = conditioning.device
        bsz = conditioning.shape[0]

        # Build encoder_hidden_states for conditioning
        enc_h = self.build_encoder_hidden_states(conditioning).to(device)

        # Build first-frame latent tokens for action conditioning.
        prep_tok = self._build_prepended_tokens(batch, device, self.cogvideox.dtype)

        # Build first-frame-based video template for latent shape.
        # For joint multi-view checkpoints, preserve all views to match training path.
        frames = batch["video/history_frames"]
        if frames.dim() not in (5, 6):
            raise ValueError(
                f"Expected video/history_frames with shape [B, V, T, C, H, W] or [B, T, C, H, W], got {tuple(frames.shape)}"
            )

        use_joint_mv = frames.dim() == 6
        view_mask = batch.get("video/view_mask") if use_joint_mv else None

        first_frame_lat = None
        if use_joint_mv:
            rep = repeat(frames[:, :, :1], "b v 1 c h w -> b v t c h w", t=self.action_horizon)
            future_latents = self.encode_multiview_to_latents(
                rep,
                view_mask=view_mask,
                get_single_frame=None,
            ).to(device).to(self.cogvideox.dtype)

            if predict_future_frames:
                first_frame_lat = self.encode_multiview_to_latents(
                    frames,
                    view_mask=view_mask,
                    get_single_frame=0,
                ).to(device).to(self.cogvideox.dtype)
        else:
            rep = repeat(frames[:, :1], "b 1 c h w -> b t c h w", t=self.action_horizon)
            future_latents = self.get_latents_from_video(rep, get_single_frame=None).to(device)
            future_latents = future_latents.to(self.cogvideox.dtype)

            if predict_future_frames:
                first_frame_lat = self.get_latents_from_video(frames[:, :1], get_single_frame=0).to(device)
                first_frame_lat = first_frame_lat.to(self.cogvideox.dtype)

        # Initialize from noise
        act = torch.randn(bsz, self.action_horizon, self.action_dim, device=device)
        vid = torch.randn_like(future_latents)

        noise_scheduler.set_timesteps(num_inference_steps, device=device)

        for t in noise_scheduler.timesteps:
            timesteps = torch.full((bsz,), t.item(), device=device, dtype=torch.long)

            act = act.to(dtype=self.cogvideox.dtype, device=device)

            # Keep the same first-frame conditioning structure used during training.
            if predict_future_frames:
                vid_in = torch.cat([first_frame_lat, vid], dim=2)
            else:
                vid_in = vid

            vid_pred, act_pred = self._run_joint_transformer(
                vid_in, act, timesteps, timesteps, enc_h, prep_tok
            )

            if not predict_future_frames:
                vid_out = vid_pred
            else:
                t_keep = vid.shape[2]
                t_start = vid_pred.shape[2] - t_keep
                vid_out = vid_pred[:, :, t_start:]

            # Action step
            act = noise_scheduler.step(act_pred, t, act).prev_sample.to(device)
            # Video step
            vid = noise_scheduler.step(vid_out, t, vid).prev_sample.to(device)

        return act

    @torch.no_grad()
    def decode_latents(self, latents):
        """VAE-decode video latents [B, T, C, h, w] into frames [B, T, C, H, W] in [0, 1]."""
        latents = rearrange(latents, "b t c h w -> b c t h w")
        frames = self.vae.decode(1 / self.vae_scale_factor * latents).sample   # [B, C, T, H, W] in [-1, 1]
        frames = rearrange(frames, "b c t h w -> b t c h w")
        return ((frames + 1.0) / 2.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def generate_video(self, video_conditioning, first_frame_latents, height, width, num_frames, scheduler,
                       num_inference_steps, guidance_scale):
        """Sample frames [B, T, C, H, W] in [0, 1], optionally conditioned on clean first-frame latents and CFG."""
        dtype, device = self.cogvideox.dtype, video_conditioning.device
        cond = self.build_encoder_hidden_states(video_conditioning)   # [B, L, D]
        B = cond.shape[0]
        use_cfg = guidance_scale > 1.0
        if use_cfg:
            neg = torch.zeros_like(cond)
            cond = torch.cat([neg, cond], dim=0)   # [2B, L, D]

        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        h, w = height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial
        C = self.cogvideox.config.in_channels
        latents = torch.randn((B, T_lat, C, h, w), device=device, dtype=dtype) * scheduler.init_noise_sigma

        first, T_in = None, T_lat   # T_in: frames the transformer sees
        if first_frame_latents is not None:
            first = rearrange(first_frame_latents, "b c t h w -> b t c h w")   # [B, 1, C, h, w]
            if use_cfg:
                first = torch.cat([first] * 2)
            T_in = T_lat + first.shape[1]

        rope = None
        if self.cogvideox.config.use_rotary_positional_embeddings:
            rope = prepare_rotary_positional_embeddings(
                height=height, width=width, vae_scale_factor_spatial=self.vae_scale_factor_spatial, num_frames=T_in,
                transformer_config=self.cogvideox.config, device=device,
            )

        scheduler.set_timesteps(num_inference_steps, device=device)
        for t in scheduler.timesteps:
            x = torch.cat([latents] * 2) if use_cfg else latents
            x = scheduler.scale_model_input(x, t)
            if first is not None:
                x = torch.cat([first, x], dim=1)
            noise_pred = self.cogvideox(
                hidden_states=x, encoder_hidden_states=cond, timestep=t.expand(x.shape[0]).to(dtype),
                image_rotary_emb=rope, return_dict=False,
            )[0].to(dtype)
            noise_pred = noise_pred[:, T_in - T_lat:]   # drop the prediction for the prepended first frame
            if use_cfg:
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (text - uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents.to(dtype)
        return self.decode_latents(latents)

    def generate_multiview_video(self, video_conditioning, first_frame_latents, num_views, height, width, num_frames,
                                 scheduler, num_inference_steps, guidance_scale):
        """Sample frames [B, V, T, C, H, W] with optional first-frame latents [B, C, 1, h, V*w]."""
        video = self.generate_video(video_conditioning, first_frame_latents, height, width * num_views, num_frames,
                                    scheduler, num_inference_steps, guidance_scale)
        return rearrange(video, "b t c h (v w) -> b v t c h w", v=num_views)
