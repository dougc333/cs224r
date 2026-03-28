from collections import deque, OrderedDict
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_env import StepType, specs

try:
    import cv2
except ImportError:
    cv2 = None

from metaworld import ALL_V3_ENVIRONMENTS


class MetaWorldEnv:
    def __init__(self, name="hammer-v3", action_repeat=2, duration=50):
        render_params = {
            "elevation": -22.5,
            "azimuth": 15,
            "distance": 0.75,
            "lookat": np.array([-0.15, 0.60, 0.25]),
        }

        env_cls = ALL_V3_ENVIRONMENTS[name]

        # Meta-World v3 is Gymnasium-like. Prefer rgb_array rendering.
        try:
            self._env = env_cls(render_mode="rgb_array")
        except TypeError:
            self._env = env_cls()

        self._env.max_path_length = np.inf
        if hasattr(self._env, "_freeze_rand_vec"):
            self._env._freeze_rand_vec = False
        if hasattr(self._env, "_partially_observable"):
            self._env._partially_observable = False
        if hasattr(self._env, "_set_task_called"):
            self._env._set_task_called = True

        self.hand_init_pose = self._env.hand_init_pos.copy()
        self.hand_init_pose = np.array([0.1, 0.5, 0.30])

        self.action_repeat = action_repeat
        self.duration = duration
        self._step = None

        # No mujoco_py viewer in Colab/new mujoco.
        self.viewer = None
        self.render_params = render_params

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    @property
    def observation_space(self):
        return self._env.observation_space

    def set_viewer_params(self, params):
        # No-op for Colab/offscreen rendering.
        return

    def _unwrap_reset(self, out):
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, info
        return out, {}

    def _unwrap_step(self, out):
        # Gymnasium: obs, reward, terminated, truncated, info
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
            return obs, reward, done, info

        # Old Gym: obs, reward, done, info
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, done, info

        raise ValueError(f"Unexpected step output format: {type(out)} / {out}")

    def step(self, action):
        reward = 0.0
        done = False
        info = {}

        for _ in range(self.action_repeat):
            out = self._env.step(action)
            state, rew, done, info = self._unwrap_step(out)
            state = state.astype(self._env.observation_space.dtype)
            reward += rew
            if done:
                break

        #reward = 1.0 * info.get("success", 0.0)
        # Debug: print raw env info occasionally
        if self._step is not None and self._step < 100:
            print("debug info:", info)

        reward = np.float32(1.0 * info.get("success", 0.0))
        self._step += 1
        if self._step >= self.duration:
            done = True

        return state, reward, done, info

    def reset(self):
        self._env.hand_init_pos = self.hand_init_pose + 0.03 * np.random.normal(size=3)
        out = self._env.reset()
        _, _ = self._unwrap_reset(out)

        for _ in range(10):
            step_out = self._env.step(np.zeros(self.action_space.shape, dtype=self.action_space.dtype))
            state, _, done, _ = self._unwrap_step(step_out)
            state = state.astype(self._env.observation_space.dtype)
            if done:
                break

        self._step = 0
        return state

    def render(self, mode="rgb_array", width=84, height=84):
        img = None

        # Preferred Gymnasium/Meta-World v3 render path
        try:
            img = self._env.render()
        except TypeError:
            try:
                img = self._env.render(mode="rgb_array")
            except TypeError:
                img = None

        if img is None:
            raise RuntimeError("Meta-World env.render() returned None")

        img = np.asarray(img)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if cv2 is not None and (img.shape[0] != height or img.shape[1] != width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        return img


class GymWrapper:
    def __init__(self, env, act_key="action"):
        self._env = env
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def observation_spec(self):
        return dm_env.specs.Array(
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype,
            name="observation",
        )

    def action_spec(self):
        return dm_env.specs.BoundedArray(
            shape=self._env.action_space.shape,
            minimum=self._env.action_space.low,
            maximum=self._env.action_space.high,
            dtype=self._env.action_space.dtype,
            name="action",
        )

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return dm_env._environment.TimeStep(
            step_type=StepType.LAST if done else StepType.MID,
            reward=reward,
            discount=1.0,
            observation=obs,
        )

    def reset(self):
        obs = self._env.reset()
        return dm_env._environment.TimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=obs,
        )


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make():
    env = MetaWorldEnv()
    env = GymWrapper(env)
    env = ActionDTypeWrapper(env, np.float32)
    env = ExtendedTimeStepWrapper(env)
    return env