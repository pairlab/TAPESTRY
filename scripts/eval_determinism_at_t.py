#!/usr/bin/env python3
"""
Determinism-at-T diagnostic for action diffusion.

For a fixed conditioning context c and a sweep of timesteps t, draws N noise
seeds x_t ~ N(0, I) and runs a single forward pass v_theta(x_t, t, c) for each.
Outputs:
  - raw v_theta tensor [num_t, N, horizon, action_dim]
  - per-(t, dim) variance across N seeds (averaged over horizon)
  - per-t total variance (mean over horizon and dim)
  - reference: variance of noise input x_t across N seeds

The two interpretations we want to distinguish are:
  (A) v_theta is nearly invariant in x_T at large t -> 1-step output approximates
      the conditional mean E[x_0 | c]. No mode selection.
  (B) v_theta varies meaningfully with x_T -> the noise input drives mode
      selection; the policy is a learned conditional sampler.

Var(v_theta) / Var(x_T) collapsing toward 0 supports (A).
A finite ratio that scales with action variance supports (B).

Usage:
  python scripts/eval_determinism_at_t.py \\
      --checkpoint <checkpoint>.pt \\
      --hdf5 <val_demo>.hdf5 \\
      --task "Pick up the corn and put it in the bowl" \\
      --frame-index 0 \\
      --num-seeds 64 \\
      --timesteps 999,899,799,699,599,499,399,299,199,99 \\
      --output-dir results/determinism_at_t
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from tapestry.eval_utils import action_noise_scheduler
from tapestry.models.model_loading import load_unified_model, encode_task_text_with_caching
from tapestry.models.unified_model_wrapper import UnifiedModelWrapper


def load_text_emb_from_file(path: str) -> torch.Tensor:
    """Load a precomputed task embedding payload."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        emb = payload
    elif isinstance(payload, dict):
        if "embedding" not in payload:
            raise KeyError(f"Embedding payload at {path} missing 'embedding' key.")
        emb = payload["embedding"]
    else:
        raise TypeError(f"Unsupported embedding payload type {type(payload)} at {path}.")
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)
    elif emb.dim() != 3:
        raise ValueError(f"Loaded text emb must be [L,D] or [1,L,D], got {tuple(emb.shape)}")
    if emb.shape[0] != 1:
        raise ValueError(f"Loaded text emb batch dim must be 1, got {emb.shape[0]}")
    return emb


def load_obs_from_hdf5(hdf5_path, frame_index, img_size, device, views):
    with h5py.File(hdf5_path, "r") as f:
        imgs = np.stack([f[f"observations/images/{v}"][frame_index] for v in views])  # [V, H, W, C]
        if imgs.dtype != np.uint8:
            raise TypeError(f"{hdf5_path}: expected uint8 RGB images, got {imgs.dtype}")

    video = rearrange(torch.from_numpy(imgs).float() / 255, "v h w c -> v c h w")
    if video.shape[2:4] != (img_size, img_size):
        video = F.interpolate(video, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return rearrange(video, "v c h w -> 1 v 1 c h w").to(device)


def expand_batch(t, n):
    """Tile a batch tensor n times along dim 0."""
    return repeat(t, "b ... -> (n b) ...", n=n)


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Encode (or load) the task text BEFORE building the main model, so T5-xxl
    # is freed before CogVideoX is loaded and the two never share GPU memory.
    if args.task_embedding_path is not None:
        print(f"Loading precomputed task embedding: {args.task_embedding_path}")
        text_emb = load_text_emb_from_file(args.task_embedding_path).to(device)
    else:
        print(f"Encoding task via checkpoint encoder (cache_dir={args.text_embedding_cache_dir})")
        text_emb = encode_task_text_with_caching(
            checkpoint_path=args.checkpoint,
            text=args.task,
            cache_dir=args.text_embedding_cache_dir,
            use_cache=args.cache_text_embeddings,
            device=device,
            t5_override=args.t5_override,
        )
    if text_emb.dim() == 2:
        text_emb = text_emb.unsqueeze(0)

    print(f"Loading model from {args.checkpoint} (skip_text_encoder=True)...")
    wrapper = UnifiedModelWrapper(load_unified_model(args.checkpoint, device, skip_text_encoder=True, t5_override=None))
    scheduler = action_noise_scheduler(wrapper.model.cfg)
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float64)
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError("This diagnostic implements v-prediction only")
    cfg = wrapper.model.cfg
    img_size = cfg["model"]["image_size"]
    if args.img_size is not None and args.img_size != img_size:
        raise ValueError("--img-size must match the checkpoint")

    decoder_name = wrapper.action_decoder_name
    action_decoder = wrapper.model.decoders[decoder_name]
    action_horizon = action_decoder.action_horizon
    action_dim = action_decoder.action_dim
    cog_dtype = action_decoder.cogvideox.dtype
    print(f"Action decoder: horizon={action_horizon}, dim={action_dim}, dtype={cog_dtype}")

    print(f"Text embedding shape: {tuple(text_emb.shape)}, dtype={text_emb.dtype}")

    print(f"Loading observation from {args.hdf5} @ frame {args.frame_index}")
    video = load_obs_from_hdf5(args.hdf5, args.frame_index, img_size, device, cfg["cond_cameraviews"])

    print("Building conditioning dict (single-batch)...")
    cond_dict = wrapper.get_conditioning_dict(video, text_emb)
    h = cond_dict["h"]               # [1, S, D]
    batch = cond_dict["batch"]

    enc_h = action_decoder.build_encoder_hidden_states(h).to(device=device, dtype=cog_dtype)
    prepended = action_decoder._build_prepended_tokens(batch, device, cog_dtype)

    enc_h_n = expand_batch(enc_h, args.num_seeds)
    prep_n = expand_batch(prepended, args.num_seeds)
    print(f"Replicated conditioning to N={args.num_seeds}: enc_h={tuple(enc_h_n.shape)}")

    timesteps_to_test = [int(t) for t in args.timesteps.split(",")]
    print(f"Timesteps to evaluate: {timesteps_to_test}")

    # Storage
    n_t = len(timesteps_to_test)
    all_v = np.zeros((n_t, args.num_seeds, action_horizon, action_dim), dtype=np.float32)
    all_x_t = np.zeros((n_t, args.num_seeds, action_horizon, action_dim), dtype=np.float32)
    all_x0_hat = np.zeros((n_t, args.num_seeds, action_horizon, action_dim), dtype=np.float32)

    g = torch.Generator(device=device).manual_seed(args.seed)

    for ti, t_int in enumerate(timesteps_to_test):
        # Sample N independent noise tensors, one per seed
        x_t = torch.randn(
            args.num_seeds, action_horizon, action_dim,
            generator=g, device=device, dtype=torch.float32,
        )
        x_t_for_model = x_t.to(dtype=cog_dtype)
        timesteps = torch.full((args.num_seeds,), t_int, device=device, dtype=cog_dtype)

        v_pred = action_decoder.cogvideox(
            hidden_states=x_t_for_model,
            encoder_hidden_states=enc_h_n,
            timestep=timesteps,
            image_rotary_emb=None,
            return_dict=False,
            decode_actions=True,
            prepended_hidden_states=prep_n,
        )[0]                                                        # [N, H, D]

        # x0_hat = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v_theta
        a = alphas_cumprod[t_int]
        x0_hat = (a.sqrt() * x_t.to(torch.float64) -
                  (1.0 - a).sqrt() * v_pred.to(torch.float64)).to(torch.float32)

        all_v[ti] = v_pred.detach().to(torch.float32).cpu().numpy()
        all_x_t[ti] = x_t.detach().cpu().numpy()
        all_x0_hat[ti] = x0_hat.detach().cpu().numpy()

        # Logging diagnostics
        v_var_per_dim = all_v[ti].var(axis=0).mean(axis=0)          # [D]
        x_var_per_dim = all_x_t[ti].var(axis=0).mean(axis=0)        # [D]
        x0_var_per_dim = all_x0_hat[ti].var(axis=0).mean(axis=0)    # [D]
        x0_mean_per_dim = all_x0_hat[ti].mean(axis=0).mean(axis=0)
        print(
            f"  t={t_int:4d}  alpha_t={float(a):.3e}  "
            f"|Var(v)|_mean={v_var_per_dim.mean():.4e}  "
            f"|Var(x_t)|_mean={x_var_per_dim.mean():.4e}  "
            f"|Var(x0_hat)|_mean={x0_var_per_dim.mean():.4e}  "
            f"|x0_hat|={np.abs(x0_mean_per_dim).mean():.3f}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "determinism_at_t.npz",
        timesteps=np.array(timesteps_to_test, dtype=np.int32),
        alphas_cumprod=alphas_cumprod[timesteps_to_test].cpu().numpy(),
        v_pred=all_v,
        x_t=all_x_t,
        x0_hat=all_x0_hat,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "task_embedding_path": str(args.task_embedding_path) if args.task_embedding_path else None,
        "hdf5": str(args.hdf5),
        "frame_index": args.frame_index,
        "num_seeds": args.num_seeds,
        "timesteps": timesteps_to_test,
        "per_t": [
            {
                "t": int(t),
                "alpha_t": float(alphas_cumprod[t]),
                "var_v_mean": float(all_v[i].var(axis=0).mean()),
                "var_x_t_mean": float(all_x_t[i].var(axis=0).mean()),
                "var_x0_hat_mean": float(all_x0_hat[i].var(axis=0).mean()),
                "var_ratio_v_over_x_t": float(all_v[i].var(axis=0).mean() / all_x_t[i].var(axis=0).mean()),
            }
            for i, t in enumerate(timesteps_to_test)
        ],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out / 'determinism_at_t.npz'}")
    print(f"Saved: {out / 'summary.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--task", default=None,
                   help="Task text. Required unless --task-embedding-path is provided.")
    p.add_argument("--task-embedding-path", default=None,
                   help="Optional path to a precomputed T5 embedding .pt; bypasses loading T5.")
    p.add_argument("--text-embedding-cache-dir", default=os.path.expanduser("~/.cache/tapestry_text_embeddings"),
                   help="Directory for cached T5 embeddings keyed by encoder identity + text.")
    p.add_argument("--cache-text-embeddings", action=argparse.BooleanOptionalAction, default=True,
                   help="If set (default), cache encoded text embeddings to disk for reuse.")
    p.add_argument("--t5-override", default=None, help="T5 encoder to load instead of the checkpoint's, e.g. THUDM/CogVideoX-2b for its smaller fp16 copy")
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--num-seeds", type=int, default=64)
    p.add_argument("--timesteps", default="999,899,799,699,599,499,399,299,199,99")
    p.add_argument("--img-size", type=int, default=None, help="Optional check; must match the checkpoint's image size")
    p.add_argument("--output-dir", default="results/determinism_at_t")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if (args.task is None) == (args.task_embedding_path is None):
        p.error("Provide exactly one of --task or --task-embedding-path.")
    return args


if __name__ == "__main__":
    run(parse_args())
