#!/usr/bin/env python3
import argparse
import math
import os
import time
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Headless MuJoCo for Colab
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class Frac1000AndEntropyCallback(BaseCallback):
    def __init__(self, horizon=1000, entropy_batch=2048, window=200, verbose=0):
        super().__init__(verbose)
        self.horizon = horizon
        self.entropy_batch = entropy_batch
        self.window = window

        self.ep_lens = deque(maxlen=window)
        self.ep_rews = deque(maxlen=window)

        self.was_time_limit = deque(maxlen=window)
        self.was_terminated = deque(maxlen=window)

        self.last_frac_1000 = 0.0
        self.last_frac_time_limit = 0.0
        self.last_frac_terminated = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode", None)
            if ep is None:
                continue

            l = float(ep.get("l", 0.0))
            r = float(ep.get("r", 0.0))
            self.ep_lens.append(l)
            self.ep_rews.append(r)

            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = not truncated

            self.was_time_limit.append(truncated)
            self.was_terminated.append(terminated)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.ep_lens) > 0:
            lens = np.asarray(self.ep_lens, dtype=np.float32)
            rews = np.asarray(self.ep_rews, dtype=np.float32)

            frac_1000 = float(np.mean(lens >= self.horizon))
            self.last_frac_1000 = frac_1000

            tl = np.asarray(self.was_time_limit, dtype=np.float32) if len(self.was_time_limit) else np.zeros((0,), dtype=np.float32)
            term = np.asarray(self.was_terminated, dtype=np.float32) if len(self.was_terminated) else np.zeros((0,), dtype=np.float32)

            frac_time_limit = float(tl.mean()) if tl.size else 0.0
            frac_terminated = float(term.mean()) if term.size else 0.0

            self.last_frac_time_limit = frac_time_limit
            self.last_frac_terminated = frac_terminated

            self.logger.record("custom/ep_len_mean_window", float(lens.mean()))
            self.logger.record("custom/ep_rew_mean_window", float(rews.mean()))
            self.logger.record("custom/frac_time_limit", frac_time_limit)
            self.logger.record("custom/frac_terminated", frac_terminated)
            self.logger.record("custom/ep_len_frac_1000", frac_1000)

        obs = self.model.rollout_buffer.observations
        obs = obs.reshape(-1, obs.shape[-1])

        if obs.shape[0] > self.entropy_batch:
            idx = np.random.choice(obs.shape[0], self.entropy_batch, replace=False)
            obs = obs[idx]

        obs_t = th.as_tensor(obs, device=self.model.device)
        with th.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            ent = dist.distribution.entropy()
            if ent.ndim == 2:
                ent = ent.sum(dim=-1)
            ent_mean = float(ent.mean().item())

        self.logger.record("custom/policy_entropy_mean", ent_mean)


class EarlyStopOnFrac1000(BaseCallback):
    def __init__(self, frac_cb: Frac1000AndEntropyCallback, threshold=0.8, patience=3, verbose=1):
        super().__init__(verbose)
        self.frac_cb = frac_cb
        self.threshold = threshold
        self.patience = patience
        self._streak = 0

    def _on_step(self) -> bool:
        if self._streak >= self.patience:
            if self.verbose:
                print(f"[early-stop] frac_1000={self.frac_cb.last_frac_1000:.2f} hit for {self.patience} rollouts.")
            return False
        return True

    def _on_rollout_end(self) -> None:
        frac = float(self.frac_cb.last_frac_1000)
        if frac >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0

        self.logger.record("custom/earlystop_streak", float(self._streak))


class GoodGaitCheckpoint(BaseCallback):
    def __init__(
        self,
        frac_cb,
        save_dir: str,
        start: float = 0.50,
        stop: float = 0.80,
        step: float = 0.05,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.frac_cb = frac_cb
        self.save_dir = save_dir
        self.start = float(start)
        self.stop = float(stop)
        self.step = float(step)

        self._saved_buckets = set()

        os.makedirs(save_dir, exist_ok=True)

        self._start_i = int(round(self.start * 100))
        self._stop_i = int(round(self.stop * 100))
        self._step_i = int(round(self.step * 100))

        if self._step_i <= 0:
            raise ValueError("step must be > 0")

    def _on_step(self) -> bool:
        return True

    def _bucket_index(self, frac: float):
        frac_i = int(math.floor(frac * 100 + 1e-9))
        if frac_i < self._start_i:
            return None
        k = (frac_i - self._start_i) // self._step_i
        bucket_i = self._start_i + k * self._step_i
        if bucket_i > self._stop_i:
            return None
        return int(k)

    def _bucket_tag(self, k: int) -> str:
        bucket_i = self._start_i + k * self._step_i
        return f"{bucket_i:02d}"

    def _on_rollout_end(self) -> None:
        frac = float(getattr(self.frac_cb, "last_frac_1000", 0.0))

        k = self._bucket_index(frac)
        self.logger.record("custom/good_gait_bucket_hit", -1.0 if k is None else float(k))

        if k is None:
            self.logger.record("custom/good_gait_saved", 0.0)
            return

        if k in self._saved_buckets:
            self.logger.record("custom/good_gait_saved", 0.0)
            return

        tag = self._bucket_tag(k)
        path = os.path.join(self.save_dir, f"check_{tag}_t{self.num_timesteps}_frac{frac:.3f}.zip")
        self.model.save(path)
        self._saved_buckets.add(k)

        self.logger.record("custom/good_gait_saved", 1.0)
        if self.verbose:
            print(f"[good-gait] saved bucket {tag} at frac_1000={frac:.3f}: {path}")


class VideoToMp4Callback(BaseCallback):
    def __init__(self, eval_env, render_freq: int, out_dir: str, fps: int = 30, max_frames: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.fps = fps
        self.max_frames = max_frames
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq != 0:
            return True

        screens = []
        obs = self.eval_env.reset()
        dones = np.zeros((self.eval_env.num_envs,), dtype=bool)

        while not dones[0] and len(screens) < self.max_frames:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _rewards, dones, _infos = self.eval_env.step(action)
            screen = self.eval_env.get_images()[0]
            screens.append(screen)

        if len(screens) == 0:
            return True

        try:
            import imageio.v2 as imageio
        except Exception as exc:
            print(f"[video] imageio not available, skipping mp4 save: {exc}")
            return True

        video_path = os.path.join(self.out_dir, f"rollout_t{self.num_timesteps}.mp4")
        with imageio.get_writer(video_path, fps=self.fps) as writer:
            for frame in screens:
                writer.append_data(frame)

        print(f"[video] saved {video_path}")
        return True


class SweepStatsCallback(BaseCallback):
    def __init__(self, print_every_rollouts=1, verbose=0):
        super().__init__(verbose)
        self.print_every_rollouts = print_every_rollouts
        self._rollouts = 0
        self.ep_lens = deque(maxlen=200)
        self.ep_rews = deque(maxlen=200)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.ep_lens.append(ep["l"])
                self.ep_rews.append(ep["r"])
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self._rollouts % self.print_every_rollouts != 0:
            return

        if len(self.ep_lens) == 0:
            print(f"[t={self.num_timesteps}] no completed episodes yet")
            return

        lens = np.array(self.ep_lens, dtype=np.float32)
        rews = np.array(self.ep_rews, dtype=np.float32)
        frac_1000 = float(np.mean(lens >= 1000))

        print(
            f"[t={self.num_timesteps}] "
            f"ep_len mean={lens.mean():.1f} "
            f"max={lens.max():.0f} "
            f"p90={np.percentile(lens, 90):.0f} "
            f"frac_1000={frac_1000:.2f} | "
            f"ep_rew mean={rews.mean():.1f} "
            f"max={rews.max():.1f}"
        )


class LogAllLegsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.joint_names = [
            "FL_hip", "FL_ankle", "FR_hip", "FR_ankle",
            "BL_hip", "BL_ankle", "BR_hip", "BR_ankle"
        ]

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            mean_actions = np.mean(np.abs(actions), axis=0)
            for i, name in enumerate(self.joint_names):
                self.logger.record(f"joints/{name}", mean_actions[i])
        return True


@dataclass
class AntTrainConfig:
    env_id: str = "Ant-v4"
    n_training_envs: int = 8
    n_eval_envs: int = 4
    total_timesteps: int = 4_000_000
    log_dir: str = "./logs/ant_run"
    render_freq: int = 20_000
    device: str = "auto"


class AntPPOTrainer:
    def __init__(self, cfg: AntTrainConfig):
        self.cfg = cfg
        self.start_time = time.time()

        self.log_dir = os.path.abspath(cfg.log_dir)
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        self.train_env = None
        self.eval_env = None
        self.model = None

    def _make_envs(self):
        self.train_env = make_vec_env(
            self.cfg.env_id,
            n_envs=self.cfg.n_training_envs,
            vec_env_cls=SubprocVecEnv,
        )
        self.train_env = VecMonitor(self.train_env, filename=os.path.join(self.log_dir, "train_monitor.csv"))
        self.train_env = VecNormalize(self.train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        self.eval_env = make_vec_env(
            self.cfg.env_id,
            n_envs=self.cfg.n_eval_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"render_mode": "rgb_array"},
        )
        self.eval_env = VecMonitor(self.eval_env, filename=os.path.join(self.log_dir, "eval_monitor.csv"))
        self.eval_env = VecNormalize(self.eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.training = False
        self.eval_env.norm_reward = False

    def _make_model(self):
        device = self.cfg.device
        if device == "auto":
            device = "cuda" if th.cuda.is_available() else "cpu"

        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.04,
            verbose=1,
            tensorboard_log=self.log_dir,
            device=device,
        )

    def train(self):
        self._make_envs()
        self._make_model()

        frac_entropy_cb = Frac1000AndEntropyCallback(horizon=1000, entropy_batch=2048)
        earlystop_cb = EarlyStopOnFrac1000(frac_entropy_cb, threshold=0.8, patience=3)
        goodgait_cb = GoodGaitCheckpoint(
            frac_cb=frac_entropy_cb,
            save_dir=self.ckpt_dir,
            start=0.50,
            stop=0.85,
            step=0.05,
            verbose=1,
        )

        callbacks = [
            LogAllLegsCallback(),
            SweepStatsCallback(),
            VideoToMp4Callback(self.eval_env, render_freq=self.cfg.render_freq, out_dir=self.video_dir),
            frac_entropy_cb,
            earlystop_cb,
            goodgait_cb,
        ]

        self.model.learn(total_timesteps=self.cfg.total_timesteps, callback=callbacks, log_interval=1)
        self.eval_env.close()
        self.train_env.close()

        elapsed = (time.time() - self.start_time) / 3600.0
        print(f"elapsed:{elapsed:.3f} hrs")
        print(f"logs: {self.log_dir}")
        print(f"videos: {self.video_dir}")
        print(f"checkpoints: {self.ckpt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Ant-v4 with logging and mp4 videos.")
    parser.add_argument("--log-dir", type=str, default=None, help="Base log directory (default: ./logs/ant_run_<timestamp>)")
    parser.add_argument("--timesteps", type=int, default=4_000_000, help="Total training timesteps")
    parser.add_argument("--train-envs", type=int, default=8, help="Number of training envs")
    parser.add_argument("--eval-envs", type=int, default=4, help="Number of eval envs")
    parser.add_argument("--render-freq", type=int, default=20_000, help="Steps between mp4 rollouts")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.log_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("./logs", f"ant_run_{stamp}")
    else:
        log_dir = args.log_dir

    cfg = AntTrainConfig(
        n_training_envs=args.train_envs,
        n_eval_envs=args.eval_envs,
        total_timesteps=args.timesteps,
        log_dir=log_dir,
        render_freq=args.render_freq,
        device=args.device,
    )

    trainer = AntPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
