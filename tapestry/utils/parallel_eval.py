"""Multi-GPU LIBERO evaluation: one worker process per GPU, each loads the checkpoint once and evaluates whole tasks.

The parent process never imports LIBERO/MuJoCo; workers import them lazily.
"""
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from tapestry.eval_utils import action_noise_scheduler, seed_eval, select_tasks


class LiberoEvalWorker:
    def __init__(self, checkpoint_path, t5_override, img_size, n_envs, n_rollouts_per_task, max_steps, cond_cameraviews,
                 temporal_agg, action_horizon, action_dim, save_video, execution_horizon, num_inference_steps, seed):
        from tapestry.models.unified_model_wrapper import UnifiedModelWrapper

        self.policy = UnifiedModelWrapper.from_checkpoint(checkpoint_path, "cuda", t5_override)
        self.noise_scheduler = action_noise_scheduler(self.policy.model.cfg)
        self.img_size = img_size
        self.n_envs = n_envs
        self.n_rollouts_per_task = n_rollouts_per_task
        self.max_steps = max_steps
        self.cond_cameraviews = cond_cameraviews
        self.temporal_agg = temporal_agg
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.save_video = save_video
        self.execution_horizon = execution_horizon
        self.num_inference_steps = num_inference_steps
        self.seed = seed

    def evaluate_task(self, suite, task_no):
        """Evaluate distinct canonical states 0..N-1 so the trial count is never inflated by repeated states."""
        from libero.libero import benchmark, get_libero_path
        from tapestry.eval_libero import rollout
        from tapestry.utils.libero_utils import build_libero_env

        task = benchmark.get_benchmark_dict()[suite]().get_task(task_no)
        init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        n_states = len(torch.load(init_path, weights_only=False))
        if self.n_rollouts_per_task > n_states:
            raise ValueError(f"requested {self.n_rollouts_per_task} distinct trials, {suite} task {task_no} has {n_states} states")
        seed_eval(self.seed)
        episodes, video = [], None
        for start in range(0, self.n_rollouts_per_task, self.n_envs):
            ids = list(range(start, min(start + self.n_envs, self.n_rollouts_per_task)))
            env, task_desc = build_libero_env(suite, task_no, self.img_size, len(ids), ids)
            try:
                env.seed([self.seed + i for i in ids])
                frames, dones, done_steps = rollout(
                    env, self.policy, task_desc, self.noise_scheduler, self.max_steps, self.cond_cameraviews,
                    self.temporal_agg, self.action_horizon, self.action_dim, self.img_size,
                    self.num_inference_steps, self.execution_horizon,
                )
            finally:
                env.close()
            if video is None and self.save_video:
                video = frames[0, : done_steps[0]]
            for i, success, steps in zip(ids, dones, done_steps, strict=True):
                episodes.append(dict(init_state_id=i, env_seed=self.seed + i, success=bool(success), steps=int(steps)))
        successes = sum(ep["success"] for ep in episodes)
        n = len(episodes)
        print(f"{suite} task {task_no} ({task_desc}): {successes}/{n} successes")
        return dict(suite=suite, task_no=task_no, success_rate=successes / n, successes=successes,
                    num_rollouts=n, canonical_init_states=n_states, seed=self.seed, episodes=episodes, video=video)


_worker = None  # this process's LiberoEvalWorker, built by _init_worker


def _init_worker(gpus, worker_args):
    """Pin this worker process to its own GPU before CUDA and MuJoCo start, then load the checkpoint."""
    global _worker
    gpu = gpus.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu  # robosuite's EGL renderer must use the same device
    _worker = LiberoEvalWorker(*worker_args)


def _evaluate_task(suite, task_no):
    return _worker.evaluate_task(suite, task_no)


def visible_gpus():
    """Physical GPU IDs from CUDA_VISIBLE_DEVICES, or every GPU when it is unset."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [gpu for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu]
    return [str(i) for i in range(torch.cuda.device_count())]


def run_parallel_libero_eval(checkpoint_path, t5_override, suites, img_size, n_envs, n_rollouts_per_task, max_steps,
                             cond_cameraviews, temporal_agg, action_horizon, action_dim, save_video, execution_horizon,
                             num_inference_steps, num_workers, task_ids, seed, results_file):
    """Evaluate selected tasks across GPU workers, with distinct canonical states and a per-task RNG reset.

    task_ids=None selects all tasks and num_workers=None uses every visible GPU. Per-task results include every
    state ID, seed, outcome and control-step count. results_file (or None) is rewritten after every finished task,
    so a crash keeps the tasks completed so far.
    """
    tasks = select_tasks(suites, task_ids)
    if min(n_envs, n_rollouts_per_task, max_steps, num_inference_steps) <= 0:
        raise ValueError("environment/trial counts and step budgets must be positive")
    if not 1 <= execution_horizon <= action_horizon:
        raise ValueError("execution_horizon must be in [1, action_horizon]")
    if num_workers is not None and num_workers <= 0:
        raise ValueError("num_workers must be positive")
    gpus = visible_gpus()
    if not gpus:
        raise RuntimeError("no visible GPUs; check nvidia-smi and CUDA_VISIBLE_DEVICES")
    if num_workers is None:
        num_workers = len(gpus)
    if num_workers > len(gpus):
        raise ValueError(f"num_workers={num_workers}, but each worker needs its own GPU and {len(gpus)} are visible")
    print(f"Starting {num_workers} LIBERO eval workers on GPUs {gpus[:num_workers]}")
    # Spawned workers start CUDA in a fresh interpreter. Unlike multiprocessing.Pool workers they are not daemons,
    # so they can start the subprocesses of vectorized LIBERO environments.
    ctx = multiprocessing.get_context("spawn")
    free_gpus = ctx.SimpleQueue()
    for gpu in gpus[:num_workers]:
        free_gpus.put(gpu)
    worker_args = (checkpoint_path, t5_override, img_size, n_envs, n_rollouts_per_task, max_steps, cond_cameraviews,
                   temporal_agg, action_horizon, action_dim, save_video, execution_horizon, num_inference_steps, seed)

    results = {suite: {} for suite in suites}
    videos = {suite: {} for suite in suites}
    with ProcessPoolExecutor(num_workers, mp_context=ctx, initializer=_init_worker, initargs=(free_gpus, worker_args)) as pool:
        futures = [pool.submit(_evaluate_task, suite, task_no) for suite, task_no in tasks]
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="LIBERO tasks"):
                r = future.result()  # re-raises worker exceptions
                results[r["suite"]][r["task_no"]] = {k: v for k, v in r.items() if k not in {"suite", "task_no", "video"}}
                if r["video"] is not None:
                    videos[r["suite"]][r["task_no"]] = r["video"]
                if results_file is not None:
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
        except BaseException:
            # These are the pool's workers. Stopping them ends the tasks they are running, which the pool would
            # otherwise finish before raising.
            for worker in multiprocessing.active_children():
                worker.terminate()
            raise
    for suite in suites:
        results[suite]["suite_success_rate"] = float(np.mean([r["success_rate"] for r in results[suite].values()]))
    results["overall_success_rate"] = float(np.mean([results[suite]["suite_success_rate"] for suite in suites]))
    if save_video:
        results["videos"] = videos
    return results
