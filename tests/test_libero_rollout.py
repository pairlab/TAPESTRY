from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tapestry.eval_libero import obs_to_video, rollout
from tapestry.loaders.libero_dataset import LiberoDataset


class ChunkPolicy(torch.nn.Module):
    def __init__(self, chunks):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = SimpleNamespace(encoders={})
        self.chunks = chunks
        self.calls = 0

    def get_task_emb(self, text):
        return torch.zeros((len(text), 1, 1))

    def get_conditioning_dict(self, video, text):
        assert video.shape == (1, 2, 1, 3, 2, 2)
        return {}

    def forward(self, cond, scheduler, num_inference_steps):
        chunk = self.chunks[self.calls]
        self.calls += 1
        return chunk


class RecordingEnv:
    def __init__(self, success_steps):
        self.env_num = 1
        self.success_steps = success_steps
        self.actions = []

    def reset(self):
        return {
            "agentview_image": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        }

    def step(self, action):
        step = len(self.actions)
        self.actions.append(action)
        return self.reset(), np.zeros(1), np.array([step in self.success_steps]), {}


def run(chunks, temporal_agg, steps, success_steps):
    env = RecordingEnv(success_steps)
    policy = ChunkPolicy(chunks)
    frames, dones, done_steps = rollout(
        env, policy, "task", None, steps, ["agentview", "eye_in_hand"], temporal_agg,
        4, 7, 2, 1, 2,
    )
    return env, policy, frames, dones, done_steps


def test_temporal_aggregation_counts_each_prediction_once():
    chunks = [torch.full((1, 4, 7), x) for x in (1.0, 3.0, 5.0)]
    env, policy, _, dones, _ = run(chunks, True, 6, [])
    w = np.exp(-0.01)
    expected = [1.0, 1.0, (1 + 3 * w) / (1 + w), (1 + 3 * w) / (1 + w),
                (3 + 5 * w) / (1 + w), (3 + 5 * w) / (1 + w)]
    np.testing.assert_allclose(np.array(env.actions)[:, 0, 0], expected, rtol=1e-6)
    assert policy.calls == 3
    assert not dones.any()


def test_zero_action_is_valid_prediction_not_padding():
    chunks = [torch.ones((1, 4, 7)), torch.zeros((1, 4, 7))]
    env, _, _, _, _ = run(chunks, True, 4, [])
    np.testing.assert_allclose(env.actions[2], 1 / (1 + np.exp(-0.01)), rtol=1e-6)


def test_no_aggregation_executes_latest_chunk_in_order():
    chunks = [torch.arange(4).view(1, 4, 1).expand(1, 4, 7).float() + x for x in (0, 10, 20)]
    env, policy, _, _, _ = run(chunks, False, 6, [])
    np.testing.assert_array_equal(np.array(env.actions)[:, 0, 0], [0, 1, 10, 11, 20, 21])
    assert policy.calls == 3


def test_rollout_reports_first_actual_success():
    env, _, frames, dones, done_steps = run([torch.ones((1, 4, 7))], False, 4, [1])
    assert dones.tolist() == [True]
    assert done_steps.tolist() == [2]
    assert len(env.actions) == 2
    assert frames.shape == (1, 4, 2, 2, 3)


def test_obs_to_video_preserves_camera_pixels():
    obs = RecordingEnv([]).reset()
    obs["agentview_image"][:] = 255
    obs["robot0_eye_in_hand_image"][:] = 64
    video = obs_to_video(obs, ["agentview", "eye_in_hand"], "cpu")
    assert video.shape == (1, 2, 1, 3, 2, 2)
    torch.testing.assert_close(video[:, 0], torch.ones((1, 1, 3, 2, 2)))
    torch.testing.assert_close(video[:, 1], torch.full((1, 1, 3, 2, 2), 64 / 255))
    with pytest.raises(KeyError):
        obs_to_video(obs, ["missing"], "cpu")


def test_dataset_and_eval_agree_on_upright_camera_tensor():
    # No HDF5 or simulator needed: exercise production dataset processing directly.
    ds = object.__new__(LiberoDataset)
    ds.active_modalities = ["video", "action", "text"]
    ds.true_horizon = 4
    ds.pred_horizon = 4
    ds.img_shape = (2, 2)
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (2, 4, 2, 2, 3), dtype=np.uint8)
    actions = rng.standard_normal((4, 7)).astype(np.float32)
    batch = ds.process_data({"images": imgs.astype(np.float32), "actions": actions.copy(), "text": "task"})
    obs = {
        "agentview_image": imgs[0:1, 0, ::-1].copy(),
        "robot0_eye_in_hand_image": imgs[1:2, 0, ::-1].copy(),
    }
    video = obs_to_video(obs, ["agentview", "eye_in_hand"], "cpu")
    torch.testing.assert_close(video[0], batch["video/history_frames"], rtol=0, atol=0)
    np.testing.assert_array_equal(batch["action/future_frames"].numpy(), actions)
