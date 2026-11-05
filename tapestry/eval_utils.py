"""LIBERO evaluation helpers shared by train.py (in-training eval) and eval_libero_parallel.py.

This module must stay importable without LIBERO/MuJoCo: the parent process of the parallel eval imports it.
"""
import gc
import random

import numpy as np
import torch
import wandb
from einops import rearrange

SUITES = ("libero_10", "libero_90", "libero_spatial", "libero_goal", "libero_object")


def num_tasks(suite):
    if suite not in SUITES:
        raise ValueError(f"unknown LIBERO suite {suite!r}; choose from {SUITES}")
    return 90 if suite == "libero_90" else 10


def select_tasks(suites, task_ids):
    """Select the same explicit task IDs within each suite, or all tasks when task_ids is None."""
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("suites must be nonempty and unique")
    tasks = []
    for suite in suites:
        count = num_tasks(suite)
        ids = list(range(count)) if task_ids is None else list(task_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("task_ids must be nonempty and unique")
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < count for i in ids):
            raise ValueError(f"task_ids for {suite} must be integers in [0, {count})")
        tasks.extend((suite, i) for i in ids)
    return tasks


def seed_eval(seed):
    """Reset policy and environment RNGs for a reproducible task, independently of worker assignment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def action_noise_scheduler(cfg):
    """Fresh instance of the scheduler the action diffusion objective was trained with (used for inference)."""
    from tapestry.losses import ModalityLossManager

    key = "action/future_frames"
    return ModalityLossManager({key: cfg["training"]["objectives"][key]}).losses[key].noise_scheduler


def log_evaluation_results(results, epoch):
    """Print LIBERO results (from eval_libero / run_parallel_libero_eval) and log them, with rollout videos, to the active wandb run if there is one."""
    summary = {k: v for k, v in results.items() if k != "videos"}
    print(f"LIBERO evaluation results: {summary}")
    if wandb.run is None:
        return
    log = {"epoch": epoch, "eval/overall_success_rate": results["overall_success_rate"]}
    for suite, suite_results in summary.items():
        if not isinstance(suite_results, dict):
            continue
        log[f"eval/{suite}_success_rate"] = suite_results["suite_success_rate"]
        for task_no, task_results in suite_results.items():
            if isinstance(task_results, dict):
                log[f"eval/{suite}_task_{task_no}_success_rate"] = task_results["success_rate"]
    if "videos" in results:
        for suite, suite_videos in results["videos"].items():
            for task_no, video in suite_videos.items():
                video_tchw = rearrange(video, "t h w c -> t c h w")
                log[f"eval/{suite}_task_{task_no}_video"] = wandb.Video(video_tchw, fps=15, format="mp4")
    wandb.log(log)


def run_direct_libero_evaluation(model, cfg):
    """In-training LIBERO eval of the (possibly accelerate-wrapped) model, settings from cfg["evaluation"];
    image size, camera views, action horizon and action dim come from the training config itself."""
    from tapestry.eval_libero import eval_libero
    from tapestry.models.unified_model_wrapper import UnifiedModelWrapper

    eval_cfg = cfg["evaluation"]
    decoder = cfg["sequence"]["loss_mappings"]["action"]["decoder"]
    core = model.module if hasattr(model, "module") else model
    policy = UnifiedModelWrapper(core)
    results = eval_libero(
        policy,
        action_noise_scheduler(cfg),
        suites=eval_cfg["suites"],
        img_size=cfg["model"]["image_size"],
        n_envs=eval_cfg["n_envs"],
        max_steps=eval_cfg["max_steps"],
        cond_cameraviews=list(cfg["training"]["dataset"]["cond_cameraviews"]),
        temporal_agg=eval_cfg["temporal_agg"],
        action_horizon=cfg["training"]["true_horizon"],
        action_dim=cfg["decoders"][decoder]["action_dim"],
        save_video=eval_cfg["save_rollout_video"],
        execution_horizon=eval_cfg["execution_horizon"],
        num_inference_steps=eval_cfg["num_inference_steps"],
    )
    del policy
    torch.cuda.empty_cache()
    gc.collect()
    core.train()
    return results
