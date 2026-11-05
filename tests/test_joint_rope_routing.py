"""Test joint video [B, C, T, H, W] and action [B, T_act, A] RoPE routing with tiny CPU models."""
import json
from pathlib import Path

import pytest
import torch

from tapestry.models.decoders.video import CogVideoXDecoder
from tapestry.video_diffusion.cogvideox_transformer_3d import (
    CogVideoXJointAttnProcessor2_0,
)


def _build_decoder(use_rope: bool, device: torch.device, root: Path) -> CogVideoXDecoder:
    """Build a tiny decoder for video [B, 4, T, H, W] and actions [B, T, 4]."""
    config = {
        "_class_name": "CogVideoXTransformer3DModel",
        "num_attention_heads": 2, "attention_head_dim": 32,
        "in_channels": 4, "out_channels": 4, "num_layers": 1,
        "sample_height": 8, "sample_width": 8, "sample_frames": 2,
        "patch_size": 2, "max_text_seq_length": 4, "text_embed_dim": 64,
        "use_rotary_positional_embeddings": use_rope,
    }
    (root / "transformer").mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(json.dumps(config))
    return CogVideoXDecoder(
        pretrained_model_name=str(root), use_pretrained=False,
        freeze_pretrained_weights=False, use_lora=False, lora_config=None,
        gradient_checkpointing=False, action_dim=4, action_horizon=4,
        max_text_seq_length=4, max_views=1, use_vae=False,
        vae_scale_factor=0.7, vae_scale_factor_spatial=8, vae_scale_factor_temporal=4,
        text_cond_dropout_prob=0.0, action_denoising_dropout=0.0,
        joint_diffusion=True, joint_coupled_timesteps=True, joint_diffusion_mix_prob=1.0,
        joint_asymmetric_action_attn=True, from_checkpoint=False,
    ).to(device)


def _run_joint_step(use_rope: bool, device: torch.device, root: Path):
    dec = _build_decoder(use_rope=use_rope, device=device, root=root)
    cog = dec.cogvideox

    # tiny shapes
    bsz = 1
    in_c = cog.config.in_channels
    t_lat = 1
    h_lat = 4
    w_lat = 4
    act_t = 4
    txt_t = cog.config.max_text_seq_length
    inner_dim = cog.inner_dim

    vid_lat = torch.randn(bsz, in_c, t_lat, h_lat, w_lat, device=device)
    act_noisy = torch.randn(bsz, act_t, dec.action_dim, device=device)
    t_vid = torch.tensor([100], device=device)
    t_act = torch.tensor([100], device=device)
    enc_h = torch.randn(bsz, txt_t, inner_dim, device=device)
    prep_tok = torch.randn(bsz, 2, inner_dim, device=device)

    vid_out, act_out = dec._run_joint_transformer(vid_lat, act_noisy, t_vid, t_act, enc_h, prep_tok)
    loss = vid_out.float().pow(2).mean() + act_out.float().pow(2).mean()
    loss.backward()
    return float(loss.item())


def test_processor_rope_seq_length_truncates():
    """Direct test: rope_seq_length controls the RoPE application range."""
    proc = CogVideoXJointAttnProcessor2_0()
    # apply_rotary_emb requires (cos, sin) tensors of shape [seq, head_dim]
    head_dim = 8
    text_t = 2
    rope_t = 3
    extra_t = 2  # action / register tokens that must NOT be rotated
    total_post_text = rope_t + extra_t

    # Build a fake Attention-like object that performs identity QKV and
    # records the query slice it received after RoPE.
    class _FakeAttn:
        is_cross_attention = False
        norm_q = None
        norm_k = None

        def __init__(self):
            self.heads = 1
            self._q_after_rope = None

        # to_q / to_k / to_v: identity in feature dim
        def to_q(self, x):
            return x

        def to_k(self, x):
            return x

        def to_v(self, x):
            return x

        def prepare_attention_mask(self, mask, sequence_length, batch_size):
            return mask

        @property
        def to_out(self):
            return [lambda x: x, lambda x: x]

    attn = _FakeAttn()
    bsz = 1
    feat = head_dim  # head_dim==inner_dim for 1-head test
    enc = torch.zeros(bsz, text_t, feat)
    hid = torch.zeros(bsz, total_post_text, feat)
    # set the post-text slice so we can detect rotation
    hid[..., :] = 1.0

    cos = torch.ones(rope_t, head_dim)  # cos=1 means identity if sin=0 -> no rotation visible
    sin = torch.zeros(rope_t, head_dim)
    rotary_emb = (cos, sin)

    # With identity rotation and rope_seq_length=rope_t, the full output
    # equals input regardless. We verify it doesn't crash with a too-short
    # rotary that would otherwise be applied to extra_t tokens.
    out, _ = proc(
        attn=attn,
        hidden_states=hid,
        encoder_hidden_states=enc,
        attention_mask=None,
        image_rotary_emb=rotary_emb,
        rope_seq_length=rope_t,
    )
    assert out.shape == (bsz, total_post_text, feat)
    # Without rope_seq_length, the same call would attempt to rotate
    # total_post_text=5 tokens with a length-3 cos/sin -> raises.
    with pytest.raises(RuntimeError, match="must match the size"):
        proc(
            attn=attn,
            hidden_states=hid,
            encoder_hidden_states=enc,
            attention_mask=None,
            image_rotary_emb=rotary_emb,
            rope_seq_length=None,
        )


def test_joint_no_rope_path_runs(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = _run_joint_step(use_rope=False, device=device, root=tmp_path)
    assert loss == loss  # not NaN


def test_joint_with_rope_path_runs(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = _run_joint_step(use_rope=True, device=device, root=tmp_path)
    assert loss == loss


def test_action_temporal_rope_structure():
    """Action RoPE: temporal dims rotate with t, spatial dims are identity.

    Mirrors the 2B sincos behavior where actions reshape to (T_act, 1, 1) image
    patches: spatial pos collapses to a single index (identity rotation), only
    temporal index varies.
    """
    from tapestry.models.decoders.video import (
        _action_temporal_rotary_pos_embed,
        _identity_rotary_pos_embed,
    )

    class _Cfg:
        attention_head_dim = 64

    action_horizon = 6
    cos, sin = _action_temporal_rotary_pos_embed(_Cfg(), action_horizon=action_horizon, device=torch.device("cpu"))
    head_dim = _Cfg().attention_head_dim
    dim_t = head_dim // 4
    dim_h = head_dim // 8 * 3
    dim_w = head_dim // 8 * 3
    assert cos.shape == (action_horizon, dim_t + dim_h + dim_w)
    assert dim_t + dim_h + dim_w == head_dim

    # Spatial dims (h and w slices) must be identity for every time step.
    spat_cos = cos[:, dim_t:]
    spat_sin = sin[:, dim_t:]
    assert torch.allclose(spat_cos, torch.ones_like(spat_cos))
    assert torch.allclose(spat_sin, torch.zeros_like(spat_sin))

    # Temporal dim at t=0 should be identity (cos=1, sin=0); at t>0 it must differ.
    assert torch.allclose(cos[0, :dim_t], torch.ones(dim_t))
    assert torch.allclose(sin[0, :dim_t], torch.zeros(dim_t))
    assert not torch.allclose(cos[1, :dim_t], cos[0, :dim_t])
    # Distinct time positions produce distinct rotations.
    for t in range(1, action_horizon):
        assert not torch.allclose(cos[t, :dim_t], cos[0, :dim_t])

    # Identity RoPE is exactly the no-rotation table.
    i_cos, i_sin = _identity_rotary_pos_embed(head_dim, 3, device=torch.device("cpu"))
    assert torch.allclose(i_cos, torch.ones(3, head_dim))
    assert torch.allclose(i_sin, torch.zeros(3, head_dim))
