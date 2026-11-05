"""LIBERO rollouts for a UnifiedModelWrapper policy.

Used by train.py's in-training evaluation (eval_utils.run_direct_libero_evaluation) and by the workers of
eval_libero_parallel.py. LIBERO/MuJoCo is imported only when building environments.
"""
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from tapestry.eval_utils import num_tasks


def obs_to_video(obs, cond_cameraviews, device):
    """LIBERO observation dict -> the policy's video input [B, V, 1, C, H, W] in [0, 1]."""
    imgs = []
    for view in cond_cameraviews:
        key = f"{view}_image" if f"{view}_image" in obs else f"robot0_{view}_image"
        if key not in obs:
            raise KeyError(f"camera view {view!r} not among observation keys {sorted(obs)}")
        imgs.append(torch.from_numpy(obs[key]))
    return rearrange(torch.stack(imgs, dim=1), "b v h w c -> b v 1 c h w").to(device=device, dtype=torch.float32) / 255.0


@torch.no_grad()
def rollout(env, policy, task_desc, noise_scheduler, max_steps, cond_cameraviews, temporal_agg, action_horizon,
            action_dim, img_size, num_inference_steps, execution_horizon):
    """One episode in each of env's n_envs environments.

    The policy predicts a chunk of action_horizon actions; a new chunk is requested every execution_horizon steps.
    With temporal_agg the executed action is the exponentially weighted average (ACT-style) of every chunk that
    covers the current step, otherwise actions are read off the latest chunk.

    Returns frames [n_envs, max_steps, img_size, img_size, 3] uint8 (agentview), dones [n_envs] bool and
    done_steps [n_envs]: frames up to and including the first success (max_steps if the episode never succeeds).
    """
    if not 1 <= execution_horizon <= action_horizon:
        raise ValueError(f"execution_horizon must be in [1, action_horizon={action_horizon}], got {execution_horizon}")
    n_envs = env.env_num
    device = next(policy.parameters()).device
    policy.eval()
    text = policy.get_task_emb([task_desc] * n_envs)

    obs = env.reset()
    dones = np.zeros(n_envs, dtype=bool)
    done_steps = np.full(n_envs, max_steps, dtype=np.int64)
    frames = np.zeros((n_envs, max_steps, img_size, img_size, 3), dtype=np.uint8)
    if temporal_agg:
        all_time_actions = torch.zeros((n_envs, max_steps, action_horizon, action_dim), device=device)
        decay_weights = torch.exp(-0.01 * torch.arange(action_horizon, device=device, dtype=torch.float32))

    chunk = None  # [n_envs, action_horizon, action_dim]: latest predicted chunk
    chunk_idx = 0  # next action of the chunk to execute
    for step in tqdm(range(max_steps), desc=f"rollout: {task_desc}", leave=False):
        if chunk is None or chunk_idx >= execution_horizon:
            cond = policy.get_conditioning_dict(obs_to_video(obs, cond_cameraviews, device), text)
            chunk = policy(cond, noise_scheduler, num_inference_steps=num_inference_steps).float()
            if chunk.shape != (n_envs, action_horizon, action_dim):
                raise RuntimeError(
                    f"policy returned actions of shape {tuple(chunk.shape)}, expected {(n_envs, action_horizon, action_dim)}"
                )
            chunk_idx = 0
            if temporal_agg:
                all_time_actions[:, step] = chunk

        if temporal_agg:
            start = max(0, step - action_horizon + 1)
            start = ((start + execution_horizon - 1) // execution_horizon) * execution_horizon
            ts = torch.arange(start, step + 1, execution_horizon, device=device)  # actual model calls only
            preds = all_time_actions[:, ts, step - ts]  # [n_envs, len(ts), action_dim]
            w = decay_weights[: len(ts)]
            w = w / w.sum()
            action = (preds * rearrange(w, "n -> 1 n 1")).sum(dim=1)
        else:
            action = chunk[:, chunk_idx]
        chunk_idx += 1

        obs, reward, done, info = env.step(action.cpu().numpy())
        done = np.asarray(done, dtype=bool)
        done_steps[done & ~dones] = step + 1
        dones |= done
        frames[:, step] = obs["agentview_image"]
        if dones.all():
            break
    return frames, dones, done_steps


def eval_libero(policy, noise_scheduler, suites, img_size, n_envs, max_steps, cond_cameraviews, temporal_agg,
                action_horizon, action_dim, save_video, execution_horizon, num_inference_steps):
    """Every task of every suite, one batch of n_envs episodes per task, serially on the policy's GPU.

    Returns {suite: {task_no: {'success_rate'}, 'suite_success_rate'}, 'overall_success_rate',
    'videos': {suite: {task_no: [T, H, W, 3] uint8}} (if save_video)}.
    """
    from tapestry.utils.libero_utils import build_libero_env

    results = {suite: {} for suite in suites}
    videos = {suite: {} for suite in suites}
    for suite in suites:
        for task_no in range(num_tasks(suite)):
            env, task_desc = build_libero_env(suite, task_no, img_size, n_envs, None)
            try:
                frames, dones, done_steps = rollout(
                    env, policy, task_desc, noise_scheduler, max_steps, cond_cameraviews, temporal_agg,
                    action_horizon, action_dim, img_size, num_inference_steps, execution_horizon,
                )
            finally:
                env.close()
            results[suite][task_no] = {"success_rate": float(dones.mean())}
            videos[suite][task_no] = frames[0, : done_steps[0]]
            print(f"{suite} task {task_no} ({task_desc}): {dones.sum()}/{n_envs} successes")
        results[suite]["suite_success_rate"] = float(np.mean([results[suite][t]["success_rate"] for t in range(num_tasks(suite))]))
    results["overall_success_rate"] = float(np.mean([results[suite]["suite_success_rate"] for suite in suites]))
    if save_video:
        results["videos"] = videos
    return results
