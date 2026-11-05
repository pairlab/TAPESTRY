"""Benchmark end-to-end policy latency with synthetic inputs.

End to end, the GPU work per action chunk is:

    cond_dict = model.get_conditioning_dict(video, text)  # encode the observation
    actions   = model.forward(cond_dict, scheduler, num_inference_steps=...)

This script builds the unified model from a training cfg (random weights), feeds
synthetic random tensors of the right shape, and times exactly that pair of calls.

Random weights are sufficient for latency: shapes determine timing, not values.

Usage:
  python scripts/benchmark_tapestry_fullpath.py \
      --cfg cfg/train_tapestry.yaml \
      --n-inf 1 --n-views 3 --warmup 5 --measure 15
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tapestry.eval_utils import action_noise_scheduler
from tapestry.models.base import Tapestry
from tapestry.models.unified_model_wrapper import UnifiedModelWrapper


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cfg(path: Path) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(path.resolve().parent), version_base="1.3"):
        cfg = compose(config_name=path.stem)
    cfg_d = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_d, dict):
        raise TypeError(f"Expected dict from {path}")
    return cfg_d


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=Path, required=True)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-inf", type=int, default=1)
    p.add_argument("--n-views", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=15)
    p.add_argument("--inference-mode", type=str, default="action_only", choices=["joint", "action_only"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dev = torch.device("cuda:0")
    torch.manual_seed(7)

    cfg = load_cfg(args.cfg)
    cfg["decoders"]["cogvideox"]["use_pretrained"] = False
    del cfg["encoders"]["text"]  # synthetic cached embeddings; T5 is excluded from timing

    # For action_only inference, the wrapper still needs joint_diffusion=True at
    # construction (mix_prob=0.5 means inference uses action-only path).
    # joint_diffusion_mix_prob is checked in forward; we set it to match the cfg.
    log("Building unified model from cfg (random weights)…")
    model = Tapestry(cfg, from_checkpoint=False).to(dev).eval()
    wrapper = UnifiedModelWrapper(model)
    wrapper = wrapper.to(dev).eval()

    # If joint_diffusion is true but we want action_only inference, set mix_prob=0.5
    # (forward dispatches to action_only when mix_prob != 1.0)
    dec = wrapper.model.decoders[wrapper.action_decoder_name]
    if args.inference_mode == "joint" and not dec.joint_diffusion:
        raise ValueError("joint inference requires a joint_diffusion decoder")
    dec.joint_diffusion_mix_prob = 1.0 if args.inference_mode == "joint" else 0.5

    sch = action_noise_scheduler(cfg)
    sch.set_timesteps(args.n_inf, device=dev)

    img_size = int(cfg["model"]["image_size"])
    # The decoder consumes text embeddings padded to its max_text_seq_length.
    # In deployment the precomputed text_emb is already this length.
    max_seq_len = int(cfg["decoders"]["cogvideox"]["max_text_seq_length"])
    text_dim = dec.cogvideox.config.text_embed_dim

    n_envs = args.n_envs
    v = args.n_views
    log(f"  shapes: video=({n_envs}, {v}, 1, 3, {img_size}, {img_size}) text=({n_envs}, {max_seq_len}, {text_dim})")

    video = torch.rand((n_envs, v, 1, 3, img_size, img_size), device=dev, dtype=torch.float32)
    text = torch.randn((n_envs, max_seq_len, text_dim), device=dev, dtype=torch.float32)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    lats: list[float] = []
    cond_lats: list[float] = []
    fwd_lats: list[float] = []
    n_total = args.warmup + args.measure
    log(f"Timing: get_conditioning_dict + forward (n_inf={args.n_inf}, mode={args.inference_mode})")
    for i in range(n_total):
        kind = "warmup" if i < args.warmup else "measure"
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()

        torch.cuda.synchronize(dev)
        tA = time.perf_counter()
        cond = wrapper.get_conditioning_dict(video, text)
        torch.cuda.synchronize(dev)
        tB = time.perf_counter()
        wrapper.forward(cond, sch, num_inference_steps=args.n_inf)
        torch.cuda.synchronize(dev)
        tC = time.perf_counter()

        total_ms = (tC - t0) * 1000.0
        cond_ms = (tB - tA) * 1000.0
        fwd_ms = (tC - tB) * 1000.0
        log(f"    iter {i+1}/{n_total} ({kind}) total={total_ms:.2f}ms (cond={cond_ms:.2f}, fwd={fwd_ms:.2f})")
        if i >= args.warmup:
            lats.append(total_ms)
            cond_lats.append(cond_ms)
            fwd_lats.append(fwd_ms)

    pk_alloc_mb = float(torch.cuda.max_memory_allocated(dev)) / 1024.0 / 1024.0

    avg = statistics.mean(lats)
    med = statistics.median(lats)
    stdev = statistics.stdev(lats) if len(lats) > 1 else 0.0
    p95 = max(lats) if len(lats) <= 4 else statistics.quantiles(lats, n=20)[18]
    cond_avg = statistics.mean(cond_lats)
    fwd_avg = statistics.mean(fwd_lats)

    out = {
        "n_envs": n_envs,
        "n_views": v,
        "image_size": img_size,
        "max_seq_len": max_seq_len,
        "num_inference_steps": args.n_inf,
        "inference_mode": args.inference_mode,
        "device": torch.cuda.get_device_name(dev),
        "torch_version": torch.__version__,
        "latency_ms_avg": avg,
        "latency_ms_median": med,
        "latency_ms_stdev": stdev,
        "latency_ms_p95": p95,
        "latency_ms_min": min(lats),
        "latency_ms_max": max(lats),
        "cond_dict_ms_avg": cond_avg,
        "forward_ms_avg": fwd_avg,
        "peak_alloc_mb": pk_alloc_mb,
        "samples_ms": lats,
    }

    log("")
    log("===== RESULTS (full inference-server window) =====")
    log(f"device                : {out['device']}")
    log(f"latency_ms_avg        : {avg:.2f}  (cond={cond_avg:.2f}, fwd={fwd_avg:.2f})")
    log(f"latency_ms_median     : {med:.2f}")
    log(f"latency_ms_p95        : {p95:.2f}")
    log(f"latency_ms_min/max    : {min(lats):.2f} / {max(lats):.2f}")
    log(f"peak_alloc_mb         : {pk_alloc_mb:.1f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
