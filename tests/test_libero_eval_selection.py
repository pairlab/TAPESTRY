import json
import multiprocessing
import os
import random
import sys
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tapestry import eval_libero
from tapestry.eval_utils import seed_eval, select_tasks
from tapestry.utils import parallel_eval


def test_task_selection_all_and_subset():
    assert select_tasks(["libero_object"], None) == [("libero_object", i) for i in range(10)]
    assert select_tasks(["libero_object", "libero_goal"], [1, 4]) == [
        ("libero_object", 1), ("libero_object", 4), ("libero_goal", 1), ("libero_goal", 4),
    ]
    assert select_tasks(["libero_90"], [89]) == [("libero_90", 89)]


@pytest.mark.parametrize("suites,ids", [
    ([], None), (["libero_object", "libero_object"], None), (["unknown"], None),
    (["libero_object"], []), (["libero_object"], [0, 0]), (["libero_object"], [10]),
    (["libero_object"], [-1]), (["libero_object"], [False]), (["libero_object"], [1.5]),
])
def test_invalid_task_selection_fails(suites, ids):
    with pytest.raises(ValueError):
        select_tasks(suites, ids)


def test_seed_eval_repeats_all_rngs():
    seed_eval(17)
    first = random.random(), np.random.rand(), torch.rand(3)
    seed_eval(17)
    second = random.random(), np.random.rand(), torch.rand(3)
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def fake_worker(monkeypatch, n_rollouts, n_envs):
    # Exercise the real worker method without GPUs or MuJoCo.
    worker = object.__new__(parallel_eval.LiberoEvalWorker)
    worker.policy = object()
    worker.noise_scheduler = object()
    worker.img_size = 2
    worker.n_envs = n_envs
    worker.n_rollouts_per_task = n_rollouts
    worker.max_steps = 10
    worker.cond_cameraviews = ["agentview"]
    worker.temporal_agg = False
    worker.action_horizon = 4
    worker.action_dim = 7
    worker.save_video = False
    worker.execution_horizon = 2
    worker.num_inference_steps = 1
    worker.seed = 17

    task = SimpleNamespace(problem_folder="libero_object", init_states_file="task.init")
    libero = ModuleType("libero.libero")
    libero.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {
        "libero_object": lambda: SimpleNamespace(get_task=lambda i: task),
    })
    libero.get_libero_path = lambda key: "/unused"
    monkeypatch.setitem(sys.modules, "libero.libero", libero)
    monkeypatch.setattr(torch, "load", lambda path, weights_only: np.zeros((50, 110)))
    builds, rng, closed = [], [], []

    class Env:
        def __init__(self, ids):
            self.ids = ids

        def seed(self, seeds):
            assert seeds == [17 + i for i in self.ids]

        def close(self):
            closed.append(self.ids)

    def build(suite, task_no, img_size, count, init_state_ids):
        assert count == len(init_state_ids)
        builds.append(init_state_ids)
        return Env(init_state_ids), "task"

    def rollout(env, *args):
        rng.append((random.random(), np.random.rand(), torch.rand(1).item()))
        ids = np.asarray(env.ids)
        return np.zeros((len(ids), 10, 2, 2, 3), dtype=np.uint8), ids % 2 == 0, np.full(len(ids), 10)

    util = ModuleType("tapestry.utils.libero_utils")
    util.build_libero_env = build
    monkeypatch.setitem(sys.modules, "tapestry.utils.libero_utils", util)
    monkeypatch.setattr(eval_libero, "rollout", rollout)
    return worker, builds, rng, closed


def test_worker_uses_50_distinct_states_with_partial_last_batch(monkeypatch):
    worker, builds, rng, closed = fake_worker(monkeypatch, 50, 7)
    result = worker.evaluate_task("libero_object", 0)
    assert [i for batch in builds for i in batch] == list(range(50))
    assert [len(batch) for batch in builds] == [7] * 7 + [1]
    assert closed == builds
    assert result["successes"] == 25
    assert result["num_rollouts"] == 50
    assert result["success_rate"] == 0.5
    assert result["canonical_init_states"] == 50
    assert [ep["init_state_id"] for ep in result["episodes"]] == list(range(50))
    assert [ep["env_seed"] for ep in result["episodes"]] == list(range(17, 67))
    first_rng = rng.copy()
    rng.clear()
    result2 = worker.evaluate_task("libero_object", 0)
    assert result2 == result
    assert rng == first_rng  # independent of any task previously evaluated by this worker


def test_worker_never_repeats_states_to_reach_large_trial_count(monkeypatch):
    worker, builds, _, _ = fake_worker(monkeypatch, 51, 7)
    with pytest.raises(ValueError, match="51 distinct trials"):
        worker.evaluate_task("libero_object", 0)
    assert builds == []


def test_worker_rejects_missing_episode_outcomes(monkeypatch):
    worker, _, _, closed = fake_worker(monkeypatch, 10, 5)

    def incomplete(env, *args):
        return np.zeros((5, 10, 2, 2, 3), dtype=np.uint8), np.zeros(4, dtype=bool), np.full(4, 10)

    monkeypatch.setattr(eval_libero, "rollout", incomplete)
    with pytest.raises(ValueError, match="shorter"):
        worker.evaluate_task("libero_object", 0)
    assert closed == [list(range(5))]


class InlinePool:
    """ProcessPoolExecutor stand-in that starts one worker and runs every task in this process."""

    def __init__(self, max_workers, mp_context, initializer, initargs):
        initializer(*initargs)

    def submit(self, fn, *args):
        future = Future()
        future.set_result(fn(*args))
        return future

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_visible_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert parallel_eval.visible_gpus() == ["2", "5"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert parallel_eval.visible_gpus() == []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert parallel_eval.visible_gpus() == ["0", "1", "2"]


def run_eval(checkpoint, num_workers, results_file):
    return parallel_eval.run_parallel_libero_eval(
        checkpoint, None, ["libero_object"], 2, 5, 10, 10, ["agentview"], False,
        4, 7, False, 2, 1, num_workers, [1, 4], 17, results_file,
    )


def test_selected_suite_aggregate_uses_only_requested_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,6")
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)  # restored after the worker sets it
    monkeypatch.setattr(parallel_eval, "_worker", None)
    monkeypatch.setattr(parallel_eval, "ProcessPoolExecutor", InlinePool)
    started = []

    class Worker:
        def __init__(self, *args):
            assert len(args) == 14
            started.append((os.environ["CUDA_VISIBLE_DEVICES"], os.environ["MUJOCO_EGL_DEVICE_ID"]))

        def evaluate_task(self, suite, task_no):
            return dict(suite=suite, task_no=task_no, success_rate=task_no / 10,
                        successes=task_no, num_rollouts=10, canonical_init_states=50,
                        seed=17, episodes=[], video=None)

    monkeypatch.setattr(parallel_eval, "LiberoEvalWorker", Worker)
    results_file = tmp_path / "partial.json"
    result = run_eval("unused.pt", None, str(results_file))
    assert started == [("3", "3")]  # the worker is pinned to the first visible GPU
    assert result["libero_object"]["suite_success_rate"] == 0.25
    assert result["overall_success_rate"] == 0.25
    assert set(result["libero_object"]) == {1, 4, "suite_success_rate"}
    partial = json.loads(results_file.read_text())  # written after every finished task
    assert set(partial["libero_object"]) == {"1", "4"} and partial["libero_object"]["4"]["successes"] == 4


@pytest.mark.parametrize("gpus,num_workers,error", [
    ("", None, "no visible GPUs"), ("0", 2, "each worker needs its own GPU"), ("0", 0, "must be positive"),
])
def test_worker_count_must_fit_the_visible_gpus(monkeypatch, gpus, num_workers, error):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpus)
    with pytest.raises((ValueError, RuntimeError), match=error):
        run_eval("unused.pt", num_workers, None)


def test_worker_start_failure_stops_the_eval(monkeypatch, tmp_path):
    # A real spawned worker that cannot load its checkpoint breaks the pool instead of hanging the eval.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(BrokenProcessPool):
        run_eval(str(tmp_path / "missing.pt"), None, None)
    assert multiprocessing.active_children() == []
