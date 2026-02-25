

#!/usr/bin/env python3
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class CubePickupCameraEnv(gym.Env):
    def __init__(self, xml_path: str, camera_name="cam", w=128, h=128):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)

        self.camera_name = camera_name
        self.w, self.h = w, h

        act_dim = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        # Pixels: HxWx3 uint8
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

        self.t = 0
        self.max_steps = 200

    def _rgb(self):
        self.renderer.update_scene(self.data, camera=self.camera_name)
        img = self.renderer.render()
        return img  # uint8 HxWx3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self._rgb(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)
        self.t += 1

        # TODO: reward shaping same as state env (needs ee/cube ids)
        reward = 0.0
        terminated = False
        truncated = self.t >= self.max_steps
        return self._rgb(), float(reward), terminated, truncated, {}

    def close(self):
        self.renderer.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--camera", default="cam")
    args = ap.parse_args()
    env = CubePickupCameraEnv(args.xml, camera_name=args.camera)
    obs, _ = env.reset()
    print("obs:", obs.shape, obs.dtype)

if __name__ == "__main__":
    main()

