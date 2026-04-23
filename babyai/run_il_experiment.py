#!/usr/bin/env python3
"""Run a BabyAI imitation-learning experiment from generated bot demos.

This wraps three steps:
1. Convert generated `.pkl.gz` bot rollouts to BabyAI's native demo format.
2. Optionally generate/convert a validation rollout file.
3. Launch the original BabyAI `scripts/train_il.py` trainer.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from convert_bot_demos_to_babyai import convert_file
from generate_bot_demos import generate_demos


ROOT = Path(__file__).resolve().parent
BABYAI_SRC = ROOT / "original_babyai_iclr19"
STORAGE = ROOT / "babyai_storage"


def env_id(level: str) -> str:
    return level if level.startswith("BabyAI-") else f"BabyAI-{level}-v0"


def native_demo_name(level: str, episodes: int) -> str:
    safe = level.replace("BabyAI-", "").replace("-v0", "").lower()
    return f"{safe}_{episodes}"


def run(args: argparse.Namespace) -> None:
    STORAGE.joinpath("demos").mkdir(parents=True, exist_ok=True)

    level_env = env_id(args.level)
    demo_name = args.demo_name or native_demo_name(args.level, args.episodes)
    train_native = STORAGE / "demos" / f"{demo_name}.pkl"
    valid_native = STORAGE / "demos" / f"{demo_name}_valid.pkl"

    convert_file(Path(args.train_input), train_native, args.episodes)

    if args.valid_input:
        convert_file(Path(args.valid_input), valid_native, args.val_episodes)
    else:
        valid_generated = ROOT / "generated_valid" / f"{demo_name}_valid_{args.val_episodes}.pkl.gz"
        valid_generated.parent.mkdir(exist_ok=True)
        generate_demos(
            env_name=level_env,
            episodes=args.val_episodes,
            output_path=str(valid_generated),
            seed=args.val_seed,
            max_steps=args.max_steps,
        )
        convert_file(valid_generated, valid_native, args.val_episodes)

    child_env = os.environ.copy()
    child_env["BABYAI_STORAGE"] = str(STORAGE)
    child_env["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    child_env["PYTHONPATH"] = str(BABYAI_SRC)

    cmd = [
        sys.executable,
        str(BABYAI_SRC / "scripts" / "train_il.py"),
        "--env",
        level_env,
        "--demos",
        demo_name,
        "--episodes",
        str(args.episodes),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--epoch-length",
        str(args.epoch_length),
        "--batch-size",
        str(args.batch_size),
        "--val-episodes",
        str(args.val_episodes),
        "--log-interval",
        str(args.log_interval),
        "--val-interval",
        str(args.val_interval),
        "--patience",
        str(args.patience),
        "--save-interval",
        "0",
    ]
    subprocess.run(cmd, check=True, env=child_env, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="GoToObjMaze")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--valid-input", default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--val-episodes", type=int, default=512)
    parser.add_argument("--val-seed", type=int, default=1_000_000_000)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--demo-name", default=None)
    parser.add_argument("--model", default="il_experiment")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--epoch-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
