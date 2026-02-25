#!/usr/bin/env python3
import os, argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import torch
import torch.nn as nn

# -----------------------
# MuJoCo pickup env (state obs)
# -----------------------
class CubePickupStateEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, xml_path: str, ee_site: str, cube_body: str, render_w=256, render_h=256):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site_id < 0:
            raise ValueError(f"ee_site not found: {ee_site}")

        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_body)
        if self.cube_body_id < 0:
            raise ValueError(f"cube body not found: {cube_body}")

        # Offscreen renderer (optional, for debugging)
        self.renderer = mujoco.Renderer(self.model, height=render_h, width=render_w)

        # Action: joint controls (ctrl)
        act_dim = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        # Observation: qpos + qvel + ee_pos(3) + cube_pos(3) + cube_quat(4)
        obs_dim = self.model.nq + self.model.nv + 3 + 3 + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.t = 0
        self.max_steps = 200

        # success criteria
        self.lift_height = 0.10  # meters above table (tune)
        self.table_z = None      # will set at reset from cube initial

    def _ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _cube_pos(self):
        return self.data.xpos[self.cube_body_id].copy()

    def _cube_quat(self):
        return self.data.xquat[self.cube_body_id].copy()

    def _obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ee = self._ee_pos()
        cpos = self._cube_pos()
        cquat = self._cube_quat()
        return np.concatenate([qpos, qvel, ee, cpos, cquat], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0

        # Randomize cube XY a bit (domain randomization)
        # NOTE: you may need to find the cube joint qpos address if cube is a free body
        # For a free joint, its qpos is 7 values: pos(3) + quat(4). You can locate by name too.
        # Here we just run forward and read initial table height as baseline.
        mujoco.mj_forward(self.model, self.data)
        self.table_z = float(self._cube_pos()[2])

        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)

        # Scale to ctrlrange if present
        if self.model.actuator_ctrlrange is not None and self.model.actuator_ctrlrange.shape[0] == self.model.nu:
            lo = self.model.actuator_ctrlrange[:, 0]
            hi = self.model.actuator_ctrlrange[:, 1]
            ctrl = lo + (action + 1.0) * 0.5 * (hi - lo)
            self.data.ctrl[:] = ctrl
        else:
            self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data)
        self.t += 1

        ee = self._ee_pos()
        cpos = self._cube_pos()
        dist = np.linalg.norm(ee - cpos)

        # Dense shaping reward
        r_reach = -dist
        r_lift = max(0.0, float(cpos[2] - self.table_z))
        success = (cpos[2] - self.table_z) > self.lift_height

        reward = 1.0 * r_reach + 5.0 * r_lift + (10.0 if success else 0.0)

        terminated = bool(success)
        truncated = bool(self.t >= self.max_steps)

        return self._obs(), float(reward), terminated, truncated, {"dist": dist, "lift": r_lift}

    def render(self):
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


# -----------------------
# Minimal PPO-ish policy (simple supervised stub-friendly)
# (For real training: use your PPO loop or SB3/CleanRL)
# -----------------------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.net(obs)
        return self.mu(h), self.v(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--ee-site", required=True)
    ap.add_argument("--cube-body", required=True)
    ap.add_argument("--out", default="policy_state.onnx")
    args = ap.parse_args()

    env = CubePickupStateEnv(args.xml, args.ee_site, args.cube_body)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # TODO: plug in your PPO trainer here
    # For now we just export an untrained network to show the pipeline.
    policy = MLPPolicy(obs_dim, act_dim).eval()

    # Export to ONNX for browser
    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)
    torch.onnx.export(
        policy,
        dummy,
        args.out,
        input_names=["obs"],
        output_names=["mu", "value"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "mu": {0: "batch"}, "value": {0: "batch"}},
    )
    print("wrote:", args.out)
    env.close()

if __name__ == "__main__":
    main()
