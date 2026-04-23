#!/usr/bin/env python3
"""Run BabyAI imitation-learning baselines for selected ICLR19 levels.

The paper-style baseline trains each level independently from bot demos. This
script orchestrates demo generation, validation demo generation, conversion to
BabyAI's native format, and `train_il.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_LEVELS = [
    "GoToObjMaze",
    "GoTo",
    "Pickup",
    "Open",
    "PutNext",
    "Synth",
    "BossLevel",
]

PAPER_1M_SUCCESS = {
    "GoToObjMaze": "99.9%",
    "GoTo": "99.4%",
    "Pickup": "99.0%",
    "Open": "100.0%",
    "PutNext": "98.8%",
    "Synth": "97.3%",
    "BossLevel": "77.0%",
}


def env_id(level: str) -> str:
    return level if level.startswith("BabyAI-") else f"BabyAI-{level}-v0"


def generated_path(level: str, episodes: int) -> Path:
    return ROOT / "demos_iclr19_3x3" / f"{level}_{episodes}.pkl.gz"


def valid_path(level: str, val_episodes: int) -> Path:
    return ROOT / "demos_iclr19_3x3" / f"{level}_valid_{val_episodes}.pkl.gz"


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_generated(level: str, episodes: int, max_steps: int, dry_run: bool) -> Path:
    path = generated_path(level, episodes)
    if path.exists():
        print(f"using existing train demos: {path}", flush=True)
        return path

    cmd = [
        sys.executable,
        str(ROOT / "generate_bot_demos.py"),
        "--env",
        env_id(level),
        "--episodes",
        str(episodes),
        "--output",
        str(path),
        "--seed",
        "1",
        "--max-steps",
        str(max_steps),
    ]
    run_cmd(cmd, dry_run)
    return path


def ensure_valid(level: str, val_episodes: int, max_steps: int, dry_run: bool) -> Path:
    path = valid_path(level, val_episodes)
    if path.exists():
        print(f"using existing valid demos: {path}", flush=True)
        return path

    cmd = [
        sys.executable,
        str(ROOT / "generate_bot_demos.py"),
        "--env",
        env_id(level),
        "--episodes",
        str(val_episodes),
        "--output",
        str(path),
        "--seed",
        "1000000000",
        "--max-steps",
        str(max_steps),
    ]
    run_cmd(cmd, dry_run)
    return path


def run_level(level: str, args: argparse.Namespace) -> None:
    train_input = ensure_generated(level, args.episodes, args.max_steps, args.dry_run)
    valid_input = ensure_valid(level, args.val_episodes, args.max_steps, args.dry_run)
    model_name = f"{level.lower()}_il_{args.episodes // 1000}k_e{args.epochs}"

    cmd = [
        sys.executable,
        str(ROOT / "run_il_experiment.py"),
        "--level",
        level,
        "--train-input",
        str(train_input),
        "--valid-input",
        str(valid_input),
        "--episodes",
        str(args.episodes),
        "--val-episodes",
        str(args.val_episodes),
        "--model",
        model_name,
        "--epochs",
        str(args.epochs),
        "--epoch-length",
        str(args.epoch_length),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--log-interval",
        str(args.log_interval),
        "--val-interval",
        str(args.val_interval),
    ]
    run_cmd(cmd, args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--val-episodes", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--epoch-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paper-baseline",
        action="store_true",
        help="Use the paper's 1M-demo baseline scale. This is a very long run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paper_baseline:
        args.episodes = 1_000_000
        args.val_episodes = 512
        args.epochs = 20
        args.epoch_length = 1_000_000
        args.batch_size = 128
        args.patience = 100

    print("Selected levels and paper 1M-demo success references:", flush=True)
    for level in args.levels:
        print(f"  {level}: {PAPER_1M_SUCCESS.get(level, 'n/a')}", flush=True)

    for level in args.levels:
        print(f"\n=== {level} ===", flush=True)
        run_level(level, args)


if __name__ == "__main__":
    main()
