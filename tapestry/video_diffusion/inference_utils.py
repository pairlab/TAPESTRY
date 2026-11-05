"""wandb logging of generated videos."""
import torch
import wandb


def to_wandb_video(video, caption):
    """video: [T, C, H, W] in [0, 1]."""
    frames = (video.float().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return wandb.Video(frames, fps=8, format="mp4", caption=caption)


def log_videos_to_wandb(generated, target, names, metrics, epoch, prefix, guidance_scale):
    """Log generated and target videos ([N, T, C, H, W] in [0, 1], one per name) with the metrics of the generated ones."""
    log = {"epoch": epoch}
    for i, name in enumerate(names):
        log[f"{prefix}/generated_{name}"] = to_wandb_video(generated[i], f"Generated_{name} (guidance {guidance_scale})")
        log[f"{prefix}/target_{name}"] = to_wandb_video(target[i], f"Target_{name}")
    for k, v in metrics.items():
        log[f"metrics/{prefix}/{k}_guidance_{guidance_scale}"] = v
    wandb.log(log)
