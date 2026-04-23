#!/usr/bin/env python3
"""Generate BabyAI demonstrations with the hand-coded bot.

This script targets the original `mila-iqia/babyai` package used by the ICLR
paper. It creates an environment, asks `babyai.bot.Bot` for actions, steps the
environment, and stores successful trajectories in a gzip-compressed pickle.

Example:
    python generate_bot_demos.py \
        --env BabyAI-GoToRedBall-v0 \
        --episodes 1000 \
        --output demos.pkl.gz
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
import time
from pathlib import Path
from typing import Any


Episode = dict[str, Any]


def import_babyai_runtime():
    """Import BabyAI's original runtime with a helpful error if unavailable."""
    local_babyai = Path(__file__).resolve().parent / "original_babyai_iclr19"
    if local_babyai.exists():
        sys.path.insert(0, str(local_babyai))

    try:
        import gym  # type: ignore
        import babyai  # noqa: F401  # type: ignore
        from babyai.bot import Bot  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Could not import the original BabyAI runtime.\n\n"
            "Install it in an environment that has the ICLR-era BabyAI package, "
            "for example:\n"
            "  git clone https://github.com/mila-iqia/babyai.git\n"
            "  cd babyai\n"
            "  pip install -e .\n\n"
            "You also need the compatible gym-minigrid dependency used by that "
            "BabyAI version."
        ) from exc

    return gym, Bot


def reset_env(env, seed: int | None = None):
    """Handle both old Gym and newer Gymnasium-style reset signatures."""
    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        reset_result = env.reset()

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _info = reset_result
        return obs
    return reset_result


def step_env(env, action: int):
    """Normalize old Gym and Gymnasium step signatures."""
    step_result = env.step(action)

    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        return obs, reward, done, info

    obs, reward, done, info = step_result
    return obs, reward, done, info


def mission_text(env) -> str:
    """Read the human-readable mission string from common BabyAI env variants."""
    if hasattr(env, "mission"):
        return env.mission
    if hasattr(env, "instrs") and hasattr(env.instrs, "surface"):
        return env.instrs.surface(env)
    return ""


def rollout_bot_episode(env, Bot, seed: int | None, max_steps: int) -> Episode | None:
    """Run one bot-controlled episode and return it if successful."""
    obs = reset_env(env, seed)
    bot = Bot(env)

    observations: list[Any] = [obs]
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []

    last_action = None
    for _ in range(max_steps):
        action = bot.replan(last_action)
        obs, reward, done, _info = step_env(env, action)

        actions.append(int(action))
        rewards.append(float(reward))
        dones.append(bool(done))
        observations.append(obs)

        last_action = action
        if done:
            if reward > 0:
                return {
                    "mission": mission_text(env),
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                }
            return None

    return None


def generate_demos(
    env_name: str,
    episodes: int,
    output_path: str,
    seed: int,
    max_steps: int,
) -> None:
    gym, Bot = import_babyai_runtime()
    try:
        env = gym.make(env_name, disable_env_checker=True)
    except TypeError:
        env = gym.make(env_name)
    bot_env = getattr(env, "unwrapped", env)

    demos: list[Episode] = []
    attempts = 0
    start_time = time.time()
    last_log_time = start_time

    while len(demos) < episodes:
        episode_seed = seed + attempts
        attempts += 1

        episode = rollout_bot_episode(bot_env, Bot, episode_seed, max_steps)
        if episode is None:
            continue

        demos.append(episode)
        if len(demos) % 100 == 0:
            now = time.time()
            elapsed = now - start_time
            interval = now - last_log_time
            last_log_time = now
            rate = 100 / interval if interval > 0 else float("inf")
            print(
                f"collected {len(demos)}/{episodes} successful demos "
                f"| elapsed {elapsed:.1f}s | last100 {interval:.1f}s "
                f"| rate {rate:.2f} demos/s",
                flush=True,
            )

    with gzip.open(output_path, "wb") as f:
        pickle.dump(
            {
                "env": env_name,
                "seed": seed,
                "attempts": attempts,
                "episodes": demos,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"wrote {len(demos)} demos to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BabyAI-GoToRedBall-v0")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", default="demos.pkl.gz")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_demos(
        env_name=args.env,
        episodes=args.episodes,
        output_path=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
