"""Wrappers around LIBERO's OffScreenRenderEnv and its vector environments."""
import numpy as np
from libero.libero.envs.venv import DummyVectorEnv, SubprocVectorEnv
from robosuite.wrappers import Wrapper


class LiberoResetWrapper(Wrapper):
    """Reset to the task's canonical initial states, starting at init_state_no and cycling through all of them."""

    def __init__(self, env, init_states, init_state_no):
        super().__init__(env)
        self.init_states = init_states
        self.reset_times = init_state_no

    def reset(self):
        self.env.reset()
        print(f"Resetting with init state {self.reset_times}")
        obs = self.env.set_init_state(self.init_states[self.reset_times])

        dummy = np.zeros(7)
        dummy[6] = -1.0  # open gripper
        for _ in range(5):  # let the physics settle
            obs, _, _, _ = self.env.step(dummy)

        self.reset_times = (self.reset_times + 1) % len(self.init_states)
        return obs

    def seed(self, seed):
        self.env.seed(seed)


class LiberoImageUpsideDownWrapper(Wrapper):
    """MuJoCo renders bottom-up: flip the camera images so they are upright."""

    def __init__(self, env, image_keys):
        super().__init__(env)
        self.image_keys = image_keys

    def flip_image(self, obs):
        for k in self.image_keys:
            obs[k] = obs[k][::-1]
        return obs

    def reset(self):
        return self.flip_image(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.flip_image(obs), reward, done, info

    def regenerate_obs_from_state(self, mujoco_state):
        return self.flip_image(self.env.regenerate_obs_from_state(mujoco_state))


class LiberoSuccessWrapper(Wrapper):
    """done = the task's success predicate."""

    def step(self, action):
        obs, reward, _, info = self.env.step(action)
        return obs, reward, self.env.check_success(), info


def merge_dict(dicts):
    """n_envs dicts of arrays -> one dict of [n_envs, ...] arrays."""
    return {k: np.stack([d[k] for d in dicts], axis=0) for k in dicts[0].keys()}


class StackedObs:
    """Vector env mixin: reset/step return one observation dict of [n_envs, ...] arrays."""

    def reset(self):
        return merge_dict(super().reset())

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return merge_dict(obs), reward, done, merge_dict(info)


class StackDummyVectorEnv(StackedObs, DummyVectorEnv):
    pass


class StackSubprocVectorEnv(StackedObs, SubprocVectorEnv):
    pass
