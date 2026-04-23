#!/usr/bin/env python3
"""Generate BabyAI demonstrations with the hand-coded bot.

The output is a gzip-compressed pickle with one entry per successful episode:

    {
        "mission": str,
        "observations": [obs_0, obs_1, ..., obs_T],
        "actions": [a_0, ..., a_{T-1}],
        "rewards": [r_0, ..., r_{T-1}],
        "dones": [d_0, ..., d_{T-1}],
    }

`observations` intentionally includes the initial observation and the final
post-step observation. `convert_bot_demos_to_babyai.py` uses `observations[:-1]`
so each saved action is paired with the observation seen before taking it.
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


def add_local_babyai_to_path() -> None:
    """Prefer local patched BabyAI checkouts over globally installed packages."""
    root = Path(__file__).resolve().parent
    for dirname in ("original_babyai_iclr19", "babyai_iclr19", "original_babyai"):
        candidate = root / dirname
        if candidate.exists() and (candidate / "babyai").is_dir():
            sys.path.insert(0, str(candidate))
            return


def import_babyai_runtime():
    add_local_babyai_to_path()
    try:
        import gym  # type: ignore
        import babyai  # noqa: F401  # type: ignore
        from babyai.bot import Bot  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Could not import BabyAI/Gym runtime. Run the setup first, or make "
            "sure a patched BabyAI checkout exists at original_babyai_iclr19/ "
            "or babyai_iclr19/."
        ) from exc
    return gym, Bot


def make_env(gym, env_name: str):
    try:
        env = gym.make(env_name, disable_env_checker=True)
    except TypeError:
        env = gym.make(env_name)
    return getattr(env, "unwrapped", env)


def reset_env(env, seed: int | None = None):
    """Reset legacy BabyAI envs deterministically when possible."""
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)

    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        reset_result = env.reset()

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _info = reset_result
        return obs
    return reset_result


def step_env(env, action: int):
    step_result = env.step(action)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated or truncated, info
    obs, reward, done, info = step_result
    return obs, reward, done, info


def mission_text(env, obs: Any) -> str:
    if hasattr(env, "mission"):
        return env.mission
    if isinstance(obs, dict) and "mission" in obs:
        return obs["mission"]
    return ""


def rollout_bot_episode(env, Bot, seed: int, max_steps: int) -> Episode | None:
    obs = reset_env(env, seed)
    bot = Bot(env)

    observations: list[Any] = [obs]
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []

    for _ in range(max_steps):
        # When the bot controls the env directly, omitting action_taken means
        # "assume the previous suggested action was taken", which is exactly
        # how BabyAI's original BotAgent uses it.
        action = int(bot.replan())
        obs, reward, done, _info = step_env(env, action)

        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        observations.append(obs)

        if done:
            if reward > 0:
                return {
                    "mission": mission_text(env, observations[0]),
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
    max_attempts: int | None = None,
) -> None:
    gym, Bot = import_babyai_runtime()
    env = make_env(gym, env_name)

    demos: list[Episode] = []
    attempts = 0
    start_time = time.time()
    last_log_time = start_time

    while len(demos) < episodes:
        if max_attempts is not None and attempts >= max_attempts:
            raise RuntimeError(
                f"Stopped after {attempts} attempts with only "
                f"{len(demos)}/{episodes} successful demos."
            )

        episode = rollout_bot_episode(env, Bot, seed + attempts, max_steps)
        attempts += 1
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
                f"| rate {rate:.2f} demos/s | attempts {attempts}",
                flush=True,
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb") as f:
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

    print(f"wrote {len(demos)} demos to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_demos(
        env_name=args.env,
        episodes=args.episodes,
        output_path=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
