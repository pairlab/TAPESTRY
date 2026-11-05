"""Vectorized LIBERO environments for policy rollouts."""
import faulthandler
import multiprocessing
import os
import random
from functools import partial

import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from .wrappers import (
    LiberoImageUpsideDownWrapper,
    LiberoResetWrapper,
    LiberoSuccessWrapper,
    StackDummyVectorEnv,
    StackSubprocVectorEnv,
)

# Required for vectorized envs
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

CAMERAS = ["agentview", "robot0_eye_in_hand"]


def build_libero_env(task_suite, task_no, img_size, n_envs, init_state_ids):
    """n_envs copies of one LIBERO task, one per canonical initial state in init_state_ids (None: a random sample).

    Returns (env, task.language). env.reset()/env.step() return observation dicts with
    'agentview_image' and 'robot0_eye_in_hand_image' [n_envs, img_size, img_size, 3] uint8 (upright).
    """
    task = benchmark.get_benchmark_dict()[task_suite]().get_task(task_no)
    env_args = dict(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_heights=img_size,
        camera_widths=img_size,
        ignore_done=True,
        camera_depths=False,
        camera_segmentations=None,
        camera_names=CAMERAS,
    )
    init_file = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    # LIBERO's bare torch.load fails on torch>=2.6 (weights_only defaults to True).
    init_states = torch.load(init_file, weights_only=False)
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    ids = random.sample(range(len(init_states)), n_envs) if init_state_ids is None else list(init_state_ids)
    if len(ids) != n_envs or len(set(ids)) != n_envs:
        raise ValueError("init_state_ids must contain one unique ID per environment")
    if any(i < 0 or i >= len(init_states) for i in ids):
        raise ValueError(f"init_state_ids must be in [0, {len(init_states)})")

    def env_func(init_state_no):
        faulthandler.enable()  # runs in the worker: a native MuJoCo/EGL crash prints a traceback, not just EOFError in the parent
        env = OffScreenRenderEnv(**env_args)
        env = LiberoResetWrapper(env, init_states, init_state_no)
        env = LiberoImageUpsideDownWrapper(env, [f"{c}_image" for c in CAMERAS])
        env = LiberoSuccessWrapper(env)
        env.seed(init_state_no)
        return env

    if n_envs == 1:
        env = StackDummyVectorEnv([partial(env_func, ids[0])])
    else:
        env = StackSubprocVectorEnv([partial(env_func, i) for i in ids])
    return env, task.language
