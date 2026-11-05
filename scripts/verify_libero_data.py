"""Replay stored LIBERO actions from recorded simulator states (not a policy evaluation).

    python scripts/verify_libero_data.py --demo-dir /path/to/libero_demos --suite libero_object --task 0 --demos 10 --output replay.json

Reports actual simulator success, image orientation/alignment, and controller settings.
The HDF5 rewards/dones are not used as evidence of success.
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from einops import rearrange
from libero.libero import benchmark, get_libero_path

from tapestry.eval_libero import obs_to_video
from tapestry.utils.libero_utils import build_libero_env
from tapestry.utils.libero_utils.wrappers import merge_dict


def replay(env, demo):
    """Restore only the first state, then execute every recorded action open-loop."""
    raw = env.workers[0].env  # n_envs == 1: the wrapped simulator lives in this process
    env.reset()
    obs = merge_dict([raw.regenerate_obs_from_state(demo["states"][0])])
    video = obs_to_video(obs, ["agentview", "eye_in_hand"], "cpu")
    saved = np.stack([demo["obs/agentview_rgb"][0], demo["obs/eye_in_hand_rgb"][0]])
    upright = rearrange(torch.from_numpy(saved[:, ::-1].copy()), "v h w c -> v c h w") / 255.0
    flipped = rearrange(torch.from_numpy(saved.copy()), "v h w c -> v c h w") / 255.0
    before = float((video[0, :, 0] - upright).abs().mean())
    wrong_flip = float((video[0, :, 0] - flipped).abs().mean())
    successes = []
    after = None
    for step, action in enumerate(demo["actions"][:]):
        obs, _, done, _ = env.step(action[np.newaxis])
        success = bool(raw.check_success())
        if bool(done[0]) != success:
            raise RuntimeError("rollout done flag disagrees with simulator success")
        if success:
            successes.append(step)
        if step == 0:
            video = obs_to_video(obs, ["agentview", "eye_in_hand"], "cpu")
            after = float((video[0, :, 0] - upright).abs().mean())
    joints = demo["obs/joint_states"][:]
    states = demo["states"][:, 1:8]
    return dict(success=bool(successes), first_success_step=successes[0] if successes else None,
                num_actions=len(demo["actions"]), image_mae_before_action=before,
                image_mae_after_action=after, image_mae_wrong_orientation=wrong_flip,
                joint_mae_same_index=float(np.abs(joints[:-1] - states[:-1]).mean()),
                joint_mae_next_state=float(np.abs(joints[:-1] - states[1:]).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo-dir", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--demos", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.demos <= 0:
        raise ValueError("--demos must be positive")
    task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task)
    path = os.path.join(args.demo_dir, args.suite, f"{task.name}_demo.hdf5")
    init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    states = torch.load(init_path, weights_only=False)
    results = dict(kind="demo_action_replay_not_policy", suite=args.suite, task=args.task,
                   task_name=task.name, canonical_init_states=len(states), demos={})
    with h5py.File(path, "r") as f:
        demos = f["data"]
        if args.demos > len(demos):
            raise ValueError(f"requested {args.demos} demos, file contains {len(demos)}")
        results["image_convention"] = demos.attrs["macros_image_convention"]
        results["recorded_controller"] = json.loads(demos.attrs["env_args"])["env_kwargs"]["controller_configs"]
        img_size = demos["demo_0/obs/agentview_rgb"].shape[1]
        env, _ = build_libero_env(args.suite, args.task, img_size, 1, None)
        try:
            controller = env.workers[0].env.robots[0].controller
            recorded = results["recorded_controller"]
            for key in ("input_min", "input_max", "output_min", "output_max"):
                np.testing.assert_allclose(getattr(controller, key), recorded[key])
            if controller.use_delta != recorded["control_delta"]:
                raise RuntimeError("evaluation controller delta/absolute mode differs from demonstrations")
            results["controller_matches"] = True
            for i in range(args.demos):
                result = replay(env, demos[f"demo_{i}"])
                results["demos"][i] = result
                print(f"demo {i}: {json.dumps(result)}", flush=True)
        finally:
            env.close()
    results["successes"] = sum(d["success"] for d in results["demos"].values())
    results["num_demos"] = args.demos
    results["success_rate"] = results["successes"] / args.demos
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Demo replay: {results['successes']}/{args.demos} actual simulator successes; NOT policy success", flush=True)


if __name__ == "__main__":
    main()
