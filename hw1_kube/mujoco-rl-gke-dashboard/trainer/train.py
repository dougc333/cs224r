import os
import time
import traceback
from dataclasses import dataclass

import gymnasium as gym
from prometheus_client import start_http_server, Gauge, Counter, Info

from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

try:
    from google.cloud import storage
except Exception:
    storage = None

os.environ.setdefault("MUJOCO_GL", "osmesa")

SUPPORTED_ENVS = [
    "Ant-v5",
    "Humanoid-v5",
    "HalfCheetah-v5",
    "Walker2d-v5",
    "Hopper-v5",
    "Swimmer-v5",
    "Reacher-v5",
    "Pusher-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
]

@dataclass
class Config:
    env_id: str = os.getenv("ENV_ID", "Ant-v5")
    algo: str = os.getenv("ALGO", "PPO")
    run_id: str = os.getenv("RUN_ID", "local-run")
    total_timesteps: int = int(os.getenv("TOTAL_TIMESTEPS", "1000000"))
    n_envs: int = int(os.getenv("N_ENVS", "8"))
    seed: int = int(os.getenv("SEED", "42"))
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    output_dir: str = os.getenv("OUTPUT_DIR", "/outputs")
    output_bucket: str = os.getenv("OUTPUT_BUCKET", "")
    record_video: bool = os.getenv("RECORD_VIDEO", "false").lower() == "true"
    video_steps: int = int(os.getenv("VIDEO_STEPS", "1000"))

cfg = Config()

run_info = Info("rl_run_info", "RL run metadata")
training_steps = Gauge("rl_training_steps", "Total RL training timesteps", ["run_id", "env", "algo"])
episode_return = Gauge("rl_episode_return", "Latest episode return", ["run_id", "env", "algo"])
episode_return_mean = Gauge("rl_episode_return_mean", "Mean episode return", ["run_id", "env", "algo"])
episode_return_min = Gauge("rl_episode_return_min", "Min episode return in buffer", ["run_id", "env", "algo"])
episode_return_max = Gauge("rl_episode_return_max", "Max episode return in buffer", ["run_id", "env", "algo"])
episode_length = Gauge("rl_episode_length", "Latest episode length", ["run_id", "env", "algo"])
episode_length_mean = Gauge("rl_episode_length_mean", "Mean episode length", ["run_id", "env", "algo"])
episode_length_min = Gauge("rl_episode_length_min", "Min episode length in buffer", ["run_id", "env", "algo"])
episode_length_max = Gauge("rl_episode_length_max", "Max episode length in buffer", ["run_id", "env", "algo"])
train_loss = Gauge("rl_train_loss", "Training loss if reported by SB3 logger", ["run_id", "env", "algo", "name"])
rollout_metric = Gauge("rl_rollout_metric", "Rollout metric from SB3 logger", ["run_id", "env", "algo", "name"])
training_updates = Counter("rl_training_updates_total", "Callback step calls", ["run_id", "env", "algo"])
run_status = Gauge("rl_run_status", "0 starting, 1 running, 2 succeeded, -1 failed", ["run_id", "env", "algo"])

LABELS = (cfg.run_id, cfg.env_id, cfg.algo)

class PrometheusCallback(BaseCallback):
    def _on_step(self) -> bool:
        training_steps.labels(*LABELS).set(self.num_timesteps)
        training_updates.labels(*LABELS).inc()

        if len(self.model.ep_info_buffer) > 0:
            returns = [float(ep["r"]) for ep in self.model.ep_info_buffer]
            lengths = [float(ep["l"]) for ep in self.model.ep_info_buffer]
            episode_return.labels(*LABELS).set(returns[-1])
            episode_return_mean.labels(*LABELS).set(sum(returns) / len(returns))
            episode_return_min.labels(*LABELS).set(min(returns))
            episode_return_max.labels(*LABELS).set(max(returns))
            episode_length.labels(*LABELS).set(lengths[-1])
            episode_length_mean.labels(*LABELS).set(sum(lengths) / len(lengths))
            episode_length_min.labels(*LABELS).set(min(lengths))
            episode_length_max.labels(*LABELS).set(max(lengths))

        # SB3 logger contains names like train/loss, train/policy_gradient_loss, rollout/ep_rew_mean.
        # Not every algo emits every metric at every step.
        name_to_value = getattr(self.model.logger, "name_to_value", {})
        for k, v in list(name_to_value.items()):
            try:
                val = float(v)
            except Exception:
                continue
            safe = k.replace("/", "_")
            if k.startswith("train/"):
                train_loss.labels(*LABELS, safe).set(val)
            elif k.startswith("rollout/"):
                rollout_metric.labels(*LABELS, safe).set(val)
        return True


def make_env(env_id: str, rank: int, seed: int):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return _init


def upload_file(path: str, bucket_name: str, blob_name: str):
    if not bucket_name or storage is None:
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(path)
    print(f"Uploaded gs://{bucket_name}/{blob_name}")


def maybe_record_video(model, env_id, out_path):
    if not cfg.record_video:
        return
    import imageio
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=cfg.seed + 999)
    frames = []
    total_reward = 0.0
    for _ in range(cfg.video_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    imageio.mimsave(out_path, frames, fps=30)
    print(f"Saved video {out_path}, return={total_reward}")


def main():
    if cfg.env_id not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported ENV_ID={cfg.env_id}. Use one of {SUPPORTED_ENVS}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    print("Config:", cfg)
    run_info.info({
        "run_id": cfg.run_id,
        "env": cfg.env_id,
        "algo": cfg.algo,
        "total_timesteps": str(cfg.total_timesteps),
        "n_envs": str(cfg.n_envs),
        "seed": str(cfg.seed),
    })
    start_http_server(cfg.prometheus_port)
    run_status.labels(*LABELS).set(1)

    vec_cls = SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env(cfg.env_id, i, cfg.seed) for i in range(cfg.n_envs)])
    env = VecMonitor(env)

    algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}
    Algo = algo_map[cfg.algo.upper()]

    if cfg.algo.upper() == "PPO":
        model = Algo(
            "MlpPolicy", env,
            learning_rate=float(os.getenv("LEARNING_RATE", "3e-4")),
            n_steps=int(os.getenv("N_STEPS", "2048")),
            batch_size=int(os.getenv("BATCH_SIZE", "256")),
            n_epochs=int(os.getenv("N_EPOCHS", "10")),
            gamma=float(os.getenv("GAMMA", "0.99")),
            gae_lambda=float(os.getenv("GAE_LAMBDA", "0.95")),
            clip_range=float(os.getenv("CLIP_RANGE", "0.2")),
            verbose=1,
        )
    elif cfg.algo.upper() == "A2C":
        model = Algo("MlpPolicy", env, learning_rate=float(os.getenv("LEARNING_RATE", "7e-4")), verbose=1)
    else:
        # Off-policy algos do not use VecEnv as heavily; still works for simple runs.
        model = Algo(
            "MlpPolicy", env,
            learning_rate=float(os.getenv("LEARNING_RATE", "3e-4")),
            buffer_size=int(os.getenv("BUFFER_SIZE", "1000000")),
            batch_size=int(os.getenv("BATCH_SIZE", "256")),
            gamma=float(os.getenv("GAMMA", "0.99")),
            verbose=1,
        )

    callback = PrometheusCallback()
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)

    model_path = os.path.join(cfg.output_dir, f"{cfg.run_id}.zip")
    model.save(model_path)
    upload_file(model_path, cfg.output_bucket, f"models/{cfg.run_id}.zip")

    video_path = os.path.join(cfg.output_dir, f"{cfg.run_id}.mp4")
    maybe_record_video(model, cfg.env_id, video_path)
    if os.path.exists(video_path):
        upload_file(video_path, cfg.output_bucket, f"videos/{cfg.run_id}.mp4")

    run_status.labels(*LABELS).set(2)
    print("Training complete; metrics endpoint will stay up for 60s.")
    time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        run_status.labels(*LABELS).set(-1)
        traceback.print_exc()
        time.sleep(30)
        raise
