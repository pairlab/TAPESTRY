"""Measure inference latency of TAPESTRY (joint diffusion) decoder.

Usage:
  python scripts/benchmark_tapestry_latency.py \
      --cfg cfg/train_tapestry.yaml \
      --n-envs 1 \
      --warmup 3 \
      --measure 10

Architecture (image_size, true_horizon, joint_diffusion, etc.) is read from --cfg.
Inputs are random tensors; VAE encode is patched to skip image preprocessing.
No checkpoint is loaded: latency is determined by architecture and input shapes,
not weight values.
"""
import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from einops import repeat
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tapestry.eval_utils import action_noise_scheduler
from tapestry.models.decoders.video import CogVideoXDecoder


@dataclass
class Case:
    name: str
    image_size: int
    pred_horizon: int
    max_seq_len: int
    joint_diffusion: bool  # construction-time setting (cogvideox decoder)
    inference_mode: str = "joint"  # "joint" -> decode_actions_with_joint_diffusion, "action_only" -> decode_actions_with_diffusion
    n_views: int = 1
    use_real_vae: bool = False


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with initialize_config_dir(config_dir=str(path.resolve().parent), version_base="1.3"):
        cfg = compose(config_name=path.stem)
    cfg_d = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_d, dict):
        raise TypeError(f"Expected dict config from {path}, got {type(cfg_d)}")
    return cfg_d


def patch_fake_latents(dec: CogVideoXDecoder) -> None:
    sp_s = int(dec.vae_scale_factor_spatial)
    sp_t = int(dec.vae_scale_factor_temporal)
    c_lat = int(dec.cogvideox.config.in_channels)

    def fake_get_latents_from_video(self, frames: torch.Tensor, get_single_frame: int | None) -> torch.Tensor:
        if frames.dim() != 5:
            raise ValueError(f"fake_get_latents_from_video expects [B, T, C, H, W], got {tuple(frames.shape)}")
        bsz, t_src, _, h_px, w_px = frames.shape
        if get_single_frame is not None and not 0 <= get_single_frame < t_src:
            raise ValueError(f"get_single_frame={get_single_frame} out of range for T={t_src}")
        if h_px % sp_s != 0 or w_px % sp_s != 0:
            raise ValueError(f"image size must be divisible by vae_scale_factor_spatial={sp_s}, got {h_px}x{w_px}")
        t_in = t_src if get_single_frame is None else sp_t
        t_lat = math.ceil(t_in / sp_t)
        return torch.randn((bsz, c_lat, t_lat, h_px // sp_s, w_px // sp_s), device=frames.device, dtype=self.cogvideox.dtype)

    dec.get_latents_from_video = MethodType(fake_get_latents_from_video, dec)


def mk_dec(cfg: dict[str, Any], c: Case, dev: torch.device, dtype: torch.dtype | None) -> CogVideoXDecoder:
    """Build the HF-config decoder for actions [B, T, A] and synthetic or real video latents."""
    dec_cfg = dict(cfg["decoders"]["cogvideox"])
    dec_cfg.pop("_class")
    sp_s = dec_cfg["vae_scale_factor_spatial"]
    lat_sz = c.image_size // sp_s
    if lat_sz * sp_s != c.image_size:
        raise ValueError(f"image_size={c.image_size} must be divisible by vae_scale_factor_spatial={sp_s}")

    dec_cfg["use_pretrained"] = False
    dec_cfg["use_vae"] = c.use_real_vae
    dec_cfg["use_lora"] = False
    dec_cfg["action_horizon"] = c.pred_horizon
    dec_cfg["max_text_seq_length"] = c.max_seq_len
    dec_cfg["joint_diffusion"] = c.joint_diffusion

    dec = CogVideoXDecoder(**dec_cfg, from_checkpoint=False).to(device=dev, dtype=dtype).eval()
    if not c.use_real_vae:
        patch_fake_latents(dec)
    return dec


def mk_inputs(c: Case, dec: CogVideoXDecoder, n_envs: int, dev: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    frame = torch.rand((n_envs, 3, c.image_size, c.image_size), device=dev, dtype=torch.float32)
    hist = repeat(frame, "b c h w -> b v t c h w", v=c.n_views, t=1)
    cond = torch.randn(
        (n_envs, c.max_seq_len, dec.cogvideox.config.text_embed_dim),
        device=dev,
        dtype=dec.cogvideox.dtype,
    )
    batch = {"video/history_frames": hist}
    if c.n_views > 1:
        batch["video/view_mask"] = torch.ones((n_envs, c.n_views), device=dev, dtype=torch.bool)
    return cond, batch


@torch.inference_mode()
def run(c: Case, cfg: dict[str, Any], n_envs: int, n_inf: int, warmup: int, measure: int, dev: torch.device, dtype: torch.dtype | None) -> dict[str, Any]:
    log(f"  building decoder + scheduler (joint_diffusion={c.joint_diffusion}, n_inf={n_inf}, n_envs={n_envs})")
    dec = mk_dec(cfg, c, dev, dtype)
    sch = action_noise_scheduler(cfg)
    cond, batch = mk_inputs(c, dec, n_envs, dev)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device=dev)
    torch.cuda.reset_peak_memory_stats(device=dev)

    lats: list[float] = []
    n_total = warmup + measure
    for i in range(n_total):
        kind = "warmup" if i < warmup else "measure"
        torch.cuda.synchronize(device=dev)
        t0 = time.perf_counter()
        if c.inference_mode == "joint":
            out = dec.decode_actions_with_joint_diffusion(
                cond,
                sch,
                batch=batch,
                num_inference_steps=n_inf,
                predict_future_frames=True,
            )
        elif c.inference_mode == "action_only":
            out = dec.decode_actions_with_diffusion(
                cond,
                sch,
                batch=batch,
                num_inference_steps=n_inf,
            )
        else:
            raise ValueError(f"Unknown inference_mode: {c.inference_mode}")
        torch.cuda.synchronize(device=dev)
        dt = (time.perf_counter() - t0) * 1000.0
        log(f"    iter {i+1}/{n_total} ({kind}) {dt:.2f} ms")
        if i >= warmup:
            lats.append(dt)

    pk_alloc_mb = float(torch.cuda.max_memory_allocated(device=dev)) / (1024.0 * 1024.0)
    pk_reserved_mb = float(torch.cuda.max_memory_reserved(device=dev)) / (1024.0 * 1024.0)

    avg = statistics.mean(lats)
    med = statistics.median(lats)
    stdev = statistics.stdev(lats) if len(lats) > 1 else 0.0
    p95 = max(lats) if len(lats) <= 4 else statistics.quantiles(lats, n=20)[18]
    out = {
        **asdict(c),
        "n_envs": n_envs,
        "num_inference_steps": n_inf,
        "device": torch.cuda.get_device_name(dev),
        "torch_version": torch.__version__,
        "latency_ms_avg": avg,
        "latency_ms_median": med,
        "latency_ms_stdev": stdev,
        "latency_ms_p95": p95,
        "latency_ms_min": min(lats),
        "latency_ms_max": max(lats),
        "denoise_step_ms_avg": avg / float(n_inf),
        "peak_alloc_mb": pk_alloc_mb,
        "peak_reserved_mb": pk_reserved_mb,
        "samples_ms": lats,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=Path, required=True, help="Train config (yaml)")
    p.add_argument("--n-envs", type=int, default=1, help="Batch size (n parallel envs)")
    p.add_argument("--n-inf", type=int, default=1, help="Number of inference (denoising) steps")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--measure", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dtype", type=str, default=None, help="Override decoder dtype (e.g. float16, bfloat16)")
    p.add_argument("--inference-mode", type=str, default="action_only", choices=["joint", "action_only"],
                   help="Which decoder method to call for inference. 'action_only' uses decode_actions_with_diffusion (skips video latent denoising)")
    p.add_argument("--max-seq-len", type=int, default=None, help="Override text max_seq_len")
    p.add_argument("--n-views", type=int, default=1, help="Number of camera views (deployment uses 3)")
    p.add_argument("--real-vae", action="store_true", help="Load actual CogVideoX-2b VAE (else patched random latents)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    dev = torch.device("cuda:0")
    torch.manual_seed(7)

    cfg = load_cfg(args.cfg)

    joint_diff = bool(cfg["decoders"]["cogvideox"]["joint_diffusion"])
    max_seq_len = int(cfg["encoders"]["text"]["max_seq_len"])
    if args.max_seq_len is not None:
        max_seq_len = int(args.max_seq_len)

    base = Case(
        name="base",
        image_size=int(cfg["model"]["image_size"]),
        pred_horizon=int(cfg["training"]["true_horizon"]),
        max_seq_len=max_seq_len,
        joint_diffusion=joint_diff,
        inference_mode=args.inference_mode,
        n_views=int(args.n_views),
        use_real_vae=bool(args.real_vae),
    )

    dtype = None if args.dtype is None else {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
    }[args.dtype]

    log(f"Device: {torch.cuda.get_device_name(dev)}  torch={torch.__version__}")
    log(f"cfg={args.cfg}")
    log(f"case: {base}")

    row = run(base, cfg, args.n_envs, args.n_inf, args.warmup, args.measure, dev, dtype)
    log("")
    log("===== RESULTS =====")
    log(f"device                : {row['device']}")
    log(f"n_envs                : {row['n_envs']}")
    log(f"num_inference_steps   : {row['num_inference_steps']}")
    log(f"latency_ms_avg        : {row['latency_ms_avg']:.2f}")
    log(f"latency_ms_median     : {row['latency_ms_median']:.2f}")
    log(f"latency_ms_p95        : {row['latency_ms_p95']:.2f}")
    log(f"latency_ms_min/max    : {row['latency_ms_min']:.2f} / {row['latency_ms_max']:.2f}")
    log(f"latency_ms_stdev      : {row['latency_ms_stdev']:.2f}")
    log(f"denoise_step_ms_avg   : {row['denoise_step_ms_avg']:.2f}")
    log(f"peak_alloc_mb         : {row['peak_alloc_mb']:.1f}")
    log(f"peak_reserved_mb      : {row['peak_reserved_mb']:.1f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        log(f"Wrote results to: {args.out}")


if __name__ == "__main__":
    main()
