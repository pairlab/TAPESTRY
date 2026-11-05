"""The minimal closed-loop deployment of a release checkpoint, on a LIBERO task. Copy this loop for your own robot:
grab the current camera images, call policy.act, execute the first execution_horizon actions, repeat.

    python scripts/prepare_release_checkpoint.py --checkpoint checkpoints/tapestry_libero_taskgen/latest.pt --output my_libero.pt --verify
    python scripts/policy_example.py --checkpoint my_libero.pt --suite libero_10 --task 0
"""
import argparse

import numpy as np

from tapestry.models.policy import TapestryPolicy
from tapestry.utils.libero_utils import build_libero_env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--execution-horizon", type=int, default=8, help="actions executed per chunk before re-planning")
    ap.add_argument("--num-inference-steps", type=int, default=1)
    ap.add_argument("--t5-override", default=None, help="T5 encoder to load instead of the checkpoint's, e.g. THUDM/CogVideoX-2b for its smaller fp16 copy")
    ap.add_argument("--text-cache-dir", default=None, help="keep T5 off the GPU and cache instruction embeddings in this directory")
    args = ap.parse_args()

    policy = TapestryPolicy(args.checkpoint, "cuda", args.num_inference_steps, args.t5_override, args.text_cache_dir)
    env, task = build_libero_env(args.suite, args.task, img_size=policy.image_size, n_envs=1, init_state_ids=None)
    print(f"task: {task}; cameras: {policy.cameras}; chunk: {policy.action_horizon} x {policy.action_dim}")
    obs = env.reset()
    chunk, idx = None, 0
    try:
        for step in range(args.max_steps):
            if chunk is None or idx >= args.execution_horizon:
                images = [obs["agentview_image"][0], obs["robot0_eye_in_hand_image"][0]]  # policy.cameras order
                chunk, idx = policy.act(images, task), 0
            obs, reward, done, info = env.step(chunk[idx][np.newaxis])  # vector env of 1
            idx += 1
            if done[0]:
                print(f"success at step {step + 1}")
                break
        else:
            print(f"no success within {args.max_steps} steps")
    finally:
        env.close()


if __name__ == "__main__":
    main()
