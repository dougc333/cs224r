#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
import time
import argparse, gzip, pickle, sys
from pathlib import Path

def import_runtime():
    sys.path.insert(0, str(Path.cwd() / "babyai_iclr19"))
    import gym, babyai
    from babyai.bot import Bot
    return gym, Bot

def reset_env(env, seed=None):
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    return env.reset()

def rollout(env, Bot, seed, max_steps):
    obs = reset_env(env, seed)
    bot = Bot(env)
    observations, actions, rewards, dones = [obs], [], [], []
    for _ in range(max_steps):
        action = bot.replan()
        obs, reward, done, _ = env.step(action)
        actions.append(int(action)); rewards.append(float(reward)); dones.append(bool(done)); observations.append(obs)
        if done:
            if reward > 0:
                return {"mission": env.mission, "observations": observations, "actions": actions, "rewards": rewards, "dones": dones}
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



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True); p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=1); p.add_argument("--max-steps", type=int, default=4096)
    args = p.parse_args()
    gym, Bot = import_runtime()
    env = gym.make(args.env, disable_env_checker=True).unwrapped
    demos, attempts = [], 0
    while len(demos) < args.episodes:
        ep = rollout(env, Bot, args.seed + attempts, args.max_steps)
        attempts += 1
        if ep is not None:
            demos.append(ep)
            if len(demos) % 100 == 0:
                print(f"collected {len(demos)}/{args.episodes}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"env": args.env, "seed": args.seed, "attempts": attempts, "episodes": demos}, gzip.open(args.output, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(demos)} demos to {args.output}")


if __name__ == "__main__":
    main()
