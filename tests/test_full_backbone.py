"""PEFT must not override the requested full-backbone training mode."""
import json

import torch

from tapestry.models.decoders import video


def build(frozen, root):
    """Build a tiny LoRA decoder for actions [B, T, 7]."""
    config = {
        "_class_name": "CogVideoXTransformer3DModel",
        "num_attention_heads": 2, "attention_head_dim": 32,
        "in_channels": 4, "out_channels": 4, "num_layers": 1,
        "sample_height": 4, "sample_width": 4, "sample_frames": 8,
        "patch_size": 2, "max_text_seq_length": 4, "text_embed_dim": 64,
        "use_rotary_positional_embeddings": False,
    }
    (root / "transformer").mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(json.dumps(config))
    torch.manual_seed(42)
    return video.CogVideoXDecoder(
        pretrained_model_name=str(root), use_pretrained=False,
        freeze_pretrained_weights=frozen, use_lora=True,
        lora_config={"r": 2, "lora_alpha": 4, "target_modules": ["to_k", "to_q", "to_v", "to_out.0"], "init_lora_weights": True},
        gradient_checkpointing=False, action_dim=7, action_horizon=4,
        max_text_seq_length=4, max_views=3, use_vae=False,
        vae_scale_factor=1.15258426, vae_scale_factor_spatial=8, vae_scale_factor_temporal=4,
        text_cond_dropout_prob=0.0, action_denoising_dropout=0.0,
        joint_diffusion=True, joint_coupled_timesteps=True, joint_diffusion_mix_prob=0.5,
        joint_asymmetric_action_attn=False, from_checkpoint=False,
    )


def output(dec):
    torch.manual_seed(17)
    return dec.cogvideox(hidden_states=torch.randn(2,4,7),encoder_hidden_states=torch.randn(2,4,64),
                        timestep=torch.full((2,),999),decode_actions=True,return_dict=False)[0]


def test_peft_respects_backbone_freezing_without_changing_weights_or_forward(tmp_path):
    frozen, full = build(True, tmp_path), build(False, tmp_path)
    assert frozen.state_dict().keys() == full.state_dict().keys()
    for key, val in frozen.state_dict().items():
        torch.testing.assert_close(val, full.state_dict()[key], rtol=0, atol=0)
    assert all(p.requires_grad for p in full.cogvideox.parameters())
    base = [(n,p) for n,p in frozen.cogvideox.named_parameters() if '.transformer_blocks.' in n and 'lora_' not in n]
    assert base and all(not p.requires_grad for _,p in base)
    assert any(p.requires_grad for n,p in frozen.cogvideox.named_parameters() if 'lora_' in n)
    frozen.eval(); full.eval()
    with torch.no_grad():
        torch.testing.assert_close(output(frozen), output(full), rtol=0, atol=0)
    output(full).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in full.cogvideox.named_parameters() if '.transformer_blocks.' in n and 'lora_' not in n)
