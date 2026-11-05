"""Tests for joint diffusion timestep coupling semantics."""

import torch
from einops import repeat

from tapestry.models.decoders.video import CogVideoXDecoder


class _FakeCfg:
    def __init__(self):
        self.use_rotary_positional_embeddings = False


class _FakeActionInputProj:
    def __init__(self, d):
        self.patch_dim = d

    def __call__(self, x):
        return x

    def patch_to_actions(self, x):
        return x


class _FakePatchEmbed:
    def __call__(self, enc_h, x):
        if x.dim() not in (3, 5):
            raise ValueError(f"Unsupported token source rank: {x.dim()}")
        bsz = int(x.shape[0])
        n_tok = int(x.shape[1])
        d_tok = int(enc_h.shape[2])
        enc = enc_h.to(device=x.device, dtype=x.dtype)
        tok = torch.zeros((bsz, n_tok, d_tok), device=x.device, dtype=x.dtype)
        return torch.cat([enc, tok], dim=1)


class _RecBlock:
    def __init__(self):
        self.calls = 0
        self.last_hidden = None
        self.last_temb = None

    def __call__(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, attention_kwargs):
        self.calls += 1
        self.last_hidden = hidden_states.detach().clone()
        self.last_temb = temb.detach().clone()
        return hidden_states, encoder_hidden_states


class _FakeCog:
    def __init__(self, d):
        self.d = d
        self.config = _FakeCfg()
        self.dtype = torch.float32
        self.ofs_embedding = None
        self.time_proj = self._time_proj
        self.time_embedding = self._time_embedding
        self.patch_embed = _FakePatchEmbed()
        self.embedding_dropout = self._identity
        self.action_input_proj = _FakeActionInputProj(d)
        self.transformer_blocks = [_RecBlock()]
        self.gradient_checkpointing = False
        self.norm_final = self._identity
        self.norm_out = self._norm_out
        self.proj_out = self._identity

    def _identity(self, x):
        return x

    def _norm_out(self, x, temb):
        return x

    def _time_proj(self, t):
        if t.ndim != 1:
            raise ValueError(f"Expected timestep tensor [B], got {tuple(t.shape)}")
        return repeat(t.float(), "b -> b d", d=self.d)

    def _time_embedding(self, t_emb, _timestep_cond):
        return t_emb


class _FakeLoss:
    def __init__(self, t):
        self._t = t
        self.sample_calls = 0
        self.add_noise_calls = []

    def sample_timesteps(self, bsz, device):
        self.sample_calls += 1
        if int(bsz) != int(self._t.shape[0]):
            raise ValueError(f"Expected bsz={int(self._t.shape[0])}, got {int(bsz)}")
        return self._t.to(device)

    def add_noise(self, x, noise, timesteps):
        self.add_noise_calls.append(timesteps.detach().clone())
        return x


def _mk_dec(d):
    dec = object.__new__(CogVideoXDecoder)
    dec.joint_diffusion = True
    dec.joint_coupled_timesteps = False
    dec.joint_asymmetric_action_attn = False
    dec.vae_scale_factor_spatial = 8
    dec.cogvideox = _FakeCog(d)
    dec._unpatchify_video_tokens = lambda tok, t_lat, h_lat, w_lat: tok.new_zeros(
        (int(tok.shape[0]), 1, int(t_lat), int(h_lat), int(w_lat))
    )
    return dec


def test_joint_run_uses_modality_specific_token_bias_and_keeps_prep_neutral():
    d = 4
    bsz = 2
    vid_t = 2
    act_t = 3
    txt_t = 2
    prep_t = 2
    dec = _mk_dec(d)

    vid_lat = torch.zeros((bsz, 1, vid_t, 1, 1), dtype=torch.float32)
    act_noisy = torch.zeros((bsz, act_t, d), dtype=torch.float32)
    t_vid = torch.tensor([1, 2], dtype=torch.long)
    t_act = torch.tensor([5, 6], dtype=torch.long)
    enc_h = torch.zeros((bsz, txt_t, d), dtype=torch.float32)
    prep_tok = torch.full((bsz, prep_t, d), 7.0, dtype=torch.float32)

    _ = dec._run_joint_transformer(vid_lat, act_noisy, t_vid, t_act, enc_h, prep_tok)

    rec = dec.cogvideox.transformer_blocks[0]
    assert rec.calls == 1
    hid = rec.last_hidden

    v_exp = repeat(t_vid.float(), "b -> b n d", n=vid_t, d=d)
    a_exp = repeat(t_act.float(), "b -> b n d", n=act_t, d=d)

    assert torch.allclose(hid[:, 0:vid_t, :], v_exp)
    assert torch.allclose(hid[:, vid_t : vid_t + prep_t, :], prep_tok)
    assert torch.allclose(hid[:, vid_t + prep_t : vid_t + prep_t + act_t, :], a_exp)
    assert torch.allclose(rec.last_temb, torch.zeros_like(rec.last_temb))


def test_joint_run_coupled_uses_shared_block_timestep():
    d = 4
    bsz = 2
    vid_t = 2
    act_t = 3
    txt_t = 2
    dec = _mk_dec(d)

    vid_lat = torch.zeros((bsz, 1, vid_t, 1, 1), dtype=torch.float32)
    act_noisy = torch.zeros((bsz, act_t, d), dtype=torch.float32)
    t = torch.tensor([3, 4], dtype=torch.long)
    enc_h = torch.zeros((bsz, txt_t, d), dtype=torch.float32)

    prep_tok = torch.zeros((bsz, 2, d), dtype=torch.float32)
    _ = dec._run_joint_transformer(vid_lat, act_noisy, t, t, enc_h, prep_tok)

    rec = dec.cogvideox.transformer_blocks[0]
    t_exp = repeat(t.float(), "b -> b d", d=d)
    assert torch.allclose(rec.last_temb, t_exp)


def test_joint_forward_decoupled_uses_independent_timesteps_single_pass():
    d = 4
    bsz = 2
    dec = _mk_dec(d)
    dec.joint_coupled_timesteps = False

    future = torch.zeros((bsz, 1, 2, 1, 1), dtype=torch.float32)
    first = torch.zeros((bsz, 1, 1, 1, 1), dtype=torch.float32)
    dec._prepare_video_latents = lambda target_video, video_loss, batch: (future, first, False, 1, None, False)
    dec.build_encoder_hidden_states = lambda conditioning: torch.zeros(
        (int(conditioning.shape[0]), 2, int(conditioning.shape[2])),
        dtype=conditioning.dtype,
        device=conditioning.device,
    )
    dec._build_prepended_tokens = lambda batch, device, dtype: None

    calls = {}

    def _joint_stub(vid_lat, act_noisy, t_vid, t_act, enc_h, prep_tok):
        calls["n"] = int(calls.get("n", 0)) + 1
        calls["t_vid"] = t_vid.detach().clone()
        calls["t_act"] = t_act.detach().clone()
        return torch.zeros_like(vid_lat), torch.zeros_like(act_noisy)

    dec._run_joint_transformer = _joint_stub

    v_loss = _FakeLoss(torch.tensor([11, 22], dtype=torch.long))
    a_loss = _FakeLoss(torch.tensor([33, 44], dtype=torch.long))
    cond = torch.zeros((bsz, 3, d), dtype=torch.float32)
    target_video = torch.zeros((bsz, 2, 3, 4, 4), dtype=torch.float32)
    target_actions = torch.zeros((bsz, 5, d), dtype=torch.float32)

    out = dec.forward_diffusion_training_joint(
        cond,
        target_video,
        target_actions,
        v_loss,
        a_loss,
        batch={},
    )

    assert int(calls["n"]) == 1
    assert torch.equal(calls["t_vid"], torch.tensor([11, 22], dtype=torch.long))
    assert torch.equal(calls["t_act"], torch.tensor([33, 44], dtype=torch.long))
    assert torch.equal(out["video_timesteps"], torch.tensor([11, 22], dtype=torch.long))
    assert torch.equal(out["action_timesteps"], torch.tensor([33, 44], dtype=torch.long))
    assert torch.equal(v_loss.add_noise_calls[0], torch.tensor([11, 22], dtype=torch.long))
    assert torch.equal(a_loss.add_noise_calls[0], torch.tensor([33, 44], dtype=torch.long))


def test_joint_forward_coupled_uses_shared_timestep_single_pass():
    d = 4
    bsz = 2
    dec = _mk_dec(d)
    dec.joint_coupled_timesteps = True

    future = torch.zeros((bsz, 1, 2, 1, 1), dtype=torch.float32)
    first = torch.zeros((bsz, 1, 1, 1, 1), dtype=torch.float32)
    dec._prepare_video_latents = lambda target_video, video_loss, batch: (future, first, False, 1, None, False)
    dec.build_encoder_hidden_states = lambda conditioning: torch.zeros(
        (int(conditioning.shape[0]), 2, int(conditioning.shape[2])),
        dtype=conditioning.dtype,
        device=conditioning.device,
    )
    dec._build_prepended_tokens = lambda batch, device, dtype: None

    calls = {}

    def _joint_stub(vid_lat, act_noisy, t_vid, t_act, enc_h, prep_tok):
        calls["n"] = int(calls.get("n", 0)) + 1
        calls["t_vid"] = t_vid.detach().clone()
        calls["t_act"] = t_act.detach().clone()
        return torch.zeros_like(vid_lat), torch.zeros_like(act_noisy)

    dec._run_joint_transformer = _joint_stub

    v_loss = _FakeLoss(torch.tensor([11, 22], dtype=torch.long))
    a_loss = _FakeLoss(torch.tensor([33, 44], dtype=torch.long))
    cond = torch.zeros((bsz, 3, d), dtype=torch.float32)
    target_video = torch.zeros((bsz, 2, 3, 4, 4), dtype=torch.float32)
    target_actions = torch.zeros((bsz, 5, d), dtype=torch.float32)

    out = dec.forward_diffusion_training_joint(
        cond,
        target_video,
        target_actions,
        v_loss,
        a_loss,
        batch={},
    )

    assert int(calls["n"]) == 1
    assert torch.equal(calls["t_vid"], torch.tensor([11, 22], dtype=torch.long))
    assert torch.equal(calls["t_act"], torch.tensor([11, 22], dtype=torch.long))
    assert torch.equal(out["video_timesteps"], torch.tensor([11, 22], dtype=torch.long))
    assert torch.equal(out["action_timesteps"], torch.tensor([11, 22], dtype=torch.long))
    assert v_loss.sample_calls == 1
    assert a_loss.sample_calls == 0
