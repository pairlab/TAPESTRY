"""Evaluate a checkpoint on LIBERO with one worker process per GPU.

    python eval_libero_parallel.py checkpoint=checkpoints/tapestry_libero_taskgen/latest.pt
    python eval_libero_parallel.py checkpoint=... 'suites=[libero_10,libero_spatial]' n_rollouts_per_task=50 num_workers=4

Architecture-bound settings (image size, camera views, action horizon and dim) come from the checkpoint;
everything else is in cfg/eval_libero_parallel.yaml.
"""
import json

import hydra
import wandb
from omegaconf import DictConfig

from tapestry.eval_utils import log_evaluation_results
from tapestry.models.model_loading import load_checkpoint_cpu
from tapestry.utils.parallel_eval import run_parallel_libero_eval


@hydra.main(version_base=None, config_path="cfg", config_name="eval_libero_parallel")
def main(cfg: DictConfig):
    if cfg.checkpoint is None:
        raise ValueError("pass checkpoint=/path/to/checkpoint.pt")
    if cfg.save_video and not cfg.use_wandb:
        raise ValueError("save_video logs rollout videos to wandb: set use_wandb=true or save_video=false")
    model_cfg = load_checkpoint_cpu(cfg.checkpoint)["config"]
    decoder = model_cfg["sequence"]["loss_mappings"]["action"]["decoder"]
    eval_kwargs = dict(
        checkpoint_path=cfg.checkpoint,
        t5_override=cfg.t5_override,
        suites=list(cfg.suites),
        task_ids=None if cfg.task_ids is None else list(cfg.task_ids),
        seed=cfg.seed,
        img_size=model_cfg["model"]["image_size"],
        n_envs=cfg.n_envs,
        n_rollouts_per_task=cfg.n_envs if cfg.n_rollouts_per_task is None else cfg.n_rollouts_per_task,
        max_steps=cfg.max_steps,
        cond_cameraviews=list(model_cfg["training"]["dataset"]["cond_cameraviews"]),
        temporal_agg=cfg.temporal_agg,
        action_horizon=model_cfg["training"]["true_horizon"],
        action_dim=model_cfg["decoders"][decoder]["action_dim"],
        save_video=cfg.save_video,
        execution_horizon=cfg.execution_horizon,
        num_inference_steps=cfg.num_inference_steps,
        num_workers=cfg.num_workers,
    )
    print(f"LIBERO eval settings: {eval_kwargs}")
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.experiment_name, config=eval_kwargs)

    results = run_parallel_libero_eval(**eval_kwargs, results_file=cfg.results_file)
    log_evaluation_results(results, epoch=0)
    if cfg.results_file is not None:
        report = {k: v for k, v in results.items() if k != "videos"}
        report["eval_settings"] = eval_kwargs
        with open(cfg.results_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Results written to {cfg.results_file}")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
