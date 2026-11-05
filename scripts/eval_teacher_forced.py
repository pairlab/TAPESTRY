#!/usr/bin/env python3
"""
Teacher-forced jitter sweep for action diffusion.

Tests whether the per-step jitter growth observed during iterative DDIM
inference is caused by self-feedback compounding (exposure bias) or by the
model itself producing high-frequency artifacts at every t.

Procedure
---------
For each ground-truth action chunk x_0 from the val set:
  for t in {999, 899, ..., 99}:
    sample fresh epsilon, build x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eps
    one forward: v_theta(x_t, t, c)
    decode x_hat_0 = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v_theta
    log:
      - x_hat_0 itself
      - jitter(x_hat_0)  (HF norm: l2 of finite-difference second derivative)
      - jitter(x_0)      (reference)
      - error(x_hat_0 - x_0)

If teacher-forced jitter at every t is ~ jitter(x_0) (low) but iterative jitter
grows with step count, exposure bias / iterate compounding is the cause.
If teacher-forced jitter is also high at low t, the model itself is the source.

Usage
-----
  python scripts/eval_teacher_forced.py \\
      --checkpoint <checkpoint>.pt \\
      --normalizer-stats data/my_robot/normalizer_stats.json \\
      --hdf5-glob 'data/my_robot/*.hdf5' \\
      --task "<task description>" \\
      --num-chunks 50 \\
      --timesteps 999,899,799,699,599,499,399,299,199,99 \\
      --output-dir results/teacher_forced
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from tapestry.utils.action_normalizer import ActionNormalizer
from tapestry.utils.action_preprocess import to_relative_rot6d_eepose20
from tapestry.eval_utils import action_noise_scheduler
from tapestry.models.model_loading import load_checkpoint_cpu, load_unified_model, encode_task_text_with_caching
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


def jitter_norm(x):
    """Per-chunk HF jitter: ||a_{t+1} - 2 a_t + a_{t-1}||_2 averaged over (T-2, D)."""
    # x: [..., T, D]
    x = np.asarray(x, dtype=np.float64)
    second_diff = x[..., 2:, :] - 2.0 * x[..., 1:-1, :] + x[..., :-2, :]
    return np.sqrt((second_diff ** 2).mean(axis=(-2, -1)))


def load_chunk(hdf5_path, t_index, normalizer, action_horizon, action_dim, img_size, device,
               views, action_space, relative_rotation_actions):
    with h5py.File(hdf5_path, "r") as f:
        gt = f[f"actions/{action_space}"][t_index:t_index + action_horizon].astype(np.float32)  # [T, D]
        if gt.shape != (action_horizon, action_dim):
            raise ValueError(f"{hdf5_path}: action chunk {gt.shape}, expected {(action_horizon, action_dim)}")
        imgs = np.stack([f[f"observations/images/{v}"][t_index] for v in views])  # [V, H, W, C]
        if imgs.dtype != np.uint8:
            raise TypeError(f"{hdf5_path}: expected uint8 RGB images, got {imgs.dtype}")

    if relative_rotation_actions:
        gt = to_relative_rot6d_eepose20(gt)
    gt_norm = normalizer.normalize(gt) if normalizer is not None else gt
    video = rearrange(torch.from_numpy(imgs).float() / 255, "v h w c -> v c h w")
    if video.shape[2:4] != (img_size, img_size):
        video = F.interpolate(video, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return {
        "video": rearrange(video, "v c h w -> 1 v 1 c h w").to(device),
        "gt_norm": rearrange(torch.from_numpy(gt_norm).float(), "t d -> 1 t d").to(device),
    }


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    ck = load_checkpoint_cpu(args.checkpoint)
    cfg = ck["config"]
    datasets = [ds for ds in cfg["training"]["dataset"]["datasets"].values() if "action" in ds["active_modalities"]]
    if not datasets or any(ds["_class"] != "HDF5DemoDataset" for ds in datasets):
        raise ValueError("This experimental diagnostic supports HDF5DemoDataset checkpoints, not LIBERO demo files")
    specs = {(ds["action_space"], ds["relative_rotation_actions"], ds["normalize_actions"]) for ds in datasets}
    if len(specs) != 1:
        raise ValueError(f"action datasets disagree on preprocessing: {specs}")
    ((action_space, relative_rotation_actions, normalize_actions),) = specs
    normalizer = ActionNormalizer.from_stats_file(args.normalizer_stats) if args.normalizer_stats else None
    if normalizer is None and "normalizer" in ck and ck["normalizer"] is not None:
        normalizer = ActionNormalizer.from_dict(ck["normalizer"])
    if normalize_actions != (normalizer is not None):
        raise ValueError("normalizer must match training; provide --normalizer-stats or an exported release checkpoint")
    del ck

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

    decoder_name = wrapper.action_decoder_name
    action_decoder = wrapper.model.decoders[decoder_name]
    action_horizon = action_decoder.action_horizon
    action_dim = action_decoder.action_dim
    cog_dtype = action_decoder.cogvideox.dtype
    if normalizer is not None and (normalizer.action_space != action_space or normalizer.horizon != action_horizon):
        raise ValueError("normalizer action space and horizon must match the checkpoint")
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError("This diagnostic implements v-prediction only")
    img_size = cfg["model"]["image_size"]
    if args.img_size is not None and args.img_size != img_size:
        raise ValueError("--img-size must match the checkpoint")

    print(f"Text embedding shape: {tuple(text_emb.shape)}, dtype={text_emb.dtype}")

    files = sorted(glob.glob(os.path.expanduser(args.hdf5_glob)))
    if not files:
        raise FileNotFoundError(f"no hdf5 files match: {args.hdf5_glob}")
    print(f"Found {len(files)} hdf5 files")

    timesteps_to_test = [int(t) for t in args.timesteps.split(",")]
    n_t = len(timesteps_to_test)

    chunks_collected = 0
    all_x0_hat = []      # list of [n_t, T, D]
    all_x_t = []
    all_v_pred = []
    all_gt = []          # [T, D] each
    chunk_meta = []

    while chunks_collected < args.num_chunks:
        f_path = rng.choice(files)
        with h5py.File(f_path, "r") as f:
            total = f[f"actions/{action_space}"].shape[0]
        max_start = total - action_horizon
        if max_start < 0:
            raise ValueError(f"{f_path}: {total} actions is shorter than horizon {action_horizon}")
        t_idx = rng.randint(0, max_start)
        chunk = load_chunk(
            f_path, t_idx, normalizer, action_horizon, action_dim, img_size, device,
            cfg["cond_cameraviews"], action_space, relative_rotation_actions,
        )

        cond_dict = wrapper.get_conditioning_dict(chunk["video"], text_emb)
        h = cond_dict["h"]
        batch = cond_dict["batch"]
        enc_h = action_decoder.build_encoder_hidden_states(h).to(device=device, dtype=cog_dtype)
        prepended = action_decoder._build_prepended_tokens(batch, device, cog_dtype)

        x0 = chunk["gt_norm"].to(torch.float64)                                  # [1, T, D]

        chunk_x0_hat = np.zeros((n_t, action_horizon, action_dim), dtype=np.float32)
        chunk_x_t = np.zeros((n_t, action_horizon, action_dim), dtype=np.float32)
        chunk_v_pred = np.zeros((n_t, action_horizon, action_dim), dtype=np.float32)

        for ti, t_int in enumerate(timesteps_to_test):
            a = alphas_cumprod[t_int]
            eps = torch.randn_like(x0)
            x_t = a.sqrt() * x0 + (1.0 - a).sqrt() * eps                         # [1, T, D]
            x_t_for_model = x_t.to(dtype=cog_dtype)
            timesteps = torch.full((1,), t_int, device=device, dtype=cog_dtype)

            v_pred = action_decoder.cogvideox(
                hidden_states=x_t_for_model,
                encoder_hidden_states=enc_h,
                timestep=timesteps,
                image_rotary_emb=None,
                return_dict=False,
                decode_actions=True,
                prepended_hidden_states=prepended,
            )[0]

            x0_hat = (a.sqrt() * x_t -
                      (1.0 - a).sqrt() * v_pred.to(torch.float64))               # [1, T, D]

            chunk_x0_hat[ti] = x0_hat[0].to(torch.float32).cpu().numpy()
            chunk_x_t[ti] = x_t[0].to(torch.float32).cpu().numpy()
            chunk_v_pred[ti] = v_pred[0].to(torch.float32).cpu().numpy()

        all_x0_hat.append(chunk_x0_hat)
        all_x_t.append(chunk_x_t)
        all_v_pred.append(chunk_v_pred)
        all_gt.append(chunk["gt_norm"][0].to(torch.float32).cpu().numpy())
        chunk_meta.append({"hdf5": f_path, "t_index": t_idx})
        chunks_collected += 1
        if chunks_collected % 5 == 0 or chunks_collected == args.num_chunks:
            jit_x0 = jitter_norm(np.array(all_x0_hat))                           # [chunks, n_t]
            jit_gt = jitter_norm(np.array(all_gt))                               # [chunks]
            print(
                f"chunk {chunks_collected}/{args.num_chunks}  "
                f"jit(x0_hat).mean per-t = {[f'{j:.3f}' for j in jit_x0.mean(0).tolist()]}  "
                f"jit(GT).mean = {jit_gt.mean():.3f}"
            )

    all_x0_hat = np.array(all_x0_hat)              # [N, n_t, T, D]
    all_x_t = np.array(all_x_t)
    all_v_pred = np.array(all_v_pred)
    all_gt = np.array(all_gt)                      # [N, T, D]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "teacher_forced.npz",
        timesteps=np.array(timesteps_to_test, dtype=np.int32),
        alphas_cumprod=alphas_cumprod[timesteps_to_test].cpu().numpy(),
        x0_hat=all_x0_hat,
        x_t=all_x_t,
        v_pred=all_v_pred,
        gt=all_gt,
    )

    # Aggregate diagnostics
    jit_x0 = jitter_norm(all_x0_hat)               # [N, n_t]
    jit_gt = jitter_norm(all_gt)                   # [N]
    err = np.linalg.norm(rearrange(all_x0_hat - all_gt[:, None], "n k t d -> n k (t d)"), axis=2)  # [N, n_t]

    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "task_embedding_path": str(args.task_embedding_path) if args.task_embedding_path else None,
        "hdf5_glob": args.hdf5_glob,
        "num_chunks": int(chunks_collected),
        "timesteps": timesteps_to_test,
        "per_t": [
            {
                "t": int(t),
                "alpha_t": float(alphas_cumprod[t]),
                "jit_x0_hat_mean": float(jit_x0[:, ti].mean()),
                "jit_x0_hat_std": float(jit_x0[:, ti].std()),
                "err_mean": float(err[:, ti].mean()),
            }
            for ti, t in enumerate(timesteps_to_test)
        ],
        "jit_gt_mean": float(jit_gt.mean()),
        "jit_gt_std": float(jit_gt.std()),
        "chunks": chunk_meta,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out / 'teacher_forced.npz'}")
    print(f"Saved: {out / 'summary.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalizer-stats", default=None)
    p.add_argument("--hdf5-glob", required=True,
                   help="Glob over val HDF5 files, e.g. data/my_robot/*.hdf5")
    p.add_argument("--task", default=None,
                   help="Task text. Required unless --task-embedding-path is provided.")
    p.add_argument("--task-embedding-path", default=None,
                   help="Optional path to a precomputed T5 embedding .pt; bypasses loading T5.")
    p.add_argument("--text-embedding-cache-dir", default=os.path.expanduser("~/.cache/tapestry_text_embeddings"),
                   help="Directory for cached T5 embeddings keyed by encoder identity + text.")
    p.add_argument("--cache-text-embeddings", action=argparse.BooleanOptionalAction, default=True,
                   help="If set (default), cache encoded text embeddings to disk for reuse.")
    p.add_argument("--t5-override", default=None, help="T5 encoder to load instead of the checkpoint's, e.g. THUDM/CogVideoX-2b for its smaller fp16 copy")
    p.add_argument("--num-chunks", type=int, default=50)
    p.add_argument("--timesteps", default="999,899,799,699,599,499,399,299,199,99")
    p.add_argument("--img-size", type=int, default=None, help="Optional check; must match the checkpoint's image size")
    p.add_argument("--output-dir", default="results/teacher_forced")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if (args.task is None) == (args.task_embedding_path is None):
        p.error("Provide exactly one of --task or --task-embedding-path.")
    return args


if __name__ == "__main__":
    run(parse_args())
