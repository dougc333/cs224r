#!/usr/bin/env python3
"""Run BabyAI demo generation/training on Colab with Ray CPU workers.

Use this after `python colab_babyai_baseline.py --setup`.

Recommended Colab CPU workflow:
    !pip install -q "ray[default]"
    !python ray_colab_babyai.py --generate-only --episodes 10000 --val-episodes 512 --num-workers 4
    !python ray_colab_babyai.py --train-only --episodes 10000 --epochs 20 --epoch-length 10000 --num-workers 2

Notes:
    - Demo generation is CPU-only and parallelizes well.
    - Training is also forced onto CPU here. Run fewer concurrent train workers
      than demo workers because each trainer is heavier.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd()
BABYAI_SRC = ROOT / "babyai_iclr19"
STORAGE = ROOT / "babyai_storage"
DEMOS = ROOT / "demos_iclr19"

DEFAULT_LEVELS = [
    "GoToObjMaze",
    "GoTo",
    "Pickup",
    "Open",
    "PutNext",
    "Synth",
    "BossLevel",
]


def env_id(level: str) -> str:
    return level if level.startswith("BabyAI-") else f"BabyAI-{level}-v0"


def train_path(level: str, episodes: int) -> Path:
    return DEMOS / f"{level}_{episodes}.pkl.gz"


def valid_path(level: str, val_episodes: int) -> Path:
    return DEMOS / f"{level}_valid_{val_episodes}.pkl.gz"


def demo_name(level: str, episodes: int) -> str:
    return f"{level.lower()}_{episodes}"


def model_name(level: str, episodes: int, epochs: int) -> str:
    return f"{level.lower()}_cpu_ray_{episodes // 1000}k_e{epochs}"


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BABYAI_SRC)
    env["BABYAI_STORAGE"] = str(STORAGE)
    env["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    # Force CPU. This avoids tiny CUDA kernels / CPU-GPU synchronization overhead
    # dominating these small BabyAI models on Colab.
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=child_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return proc.stdout


def generate_level(level: str, episodes: int, val_episodes: int, max_steps: int) -> str:
    DEMOS.mkdir(exist_ok=True)
    outputs = [f"=== generate {level} ==="]

    train = train_path(level, episodes)
    if train.exists():
        outputs.append(f"using existing {train}")
    else:
        outputs.append(
            run([
                sys.executable,
                "generate_bot_demos.py",
                "--env",
                env_id(level),
                "--episodes",
                str(episodes),
                "--output",
                str(train),
                "--seed",
                "1",
                "--max-steps",
                str(max_steps),
            ])
        )

    valid = valid_path(level, val_episodes)
    if valid.exists():
        outputs.append(f"using existing {valid}")
    else:
        outputs.append(
            run([
                sys.executable,
                "generate_bot_demos.py",
                "--env",
                env_id(level),
                "--episodes",
                str(val_episodes),
                "--output",
                str(valid),
                "--seed",
                "1000000000",
                "--max-steps",
                str(max_steps),
            ])
        )

    return "\n".join(outputs)


def train_level(
    level: str,
    episodes: int,
    val_episodes: int,
    epochs: int,
    epoch_length: int,
    batch_size: int,
    patience: int,
) -> str:
    STORAGE.joinpath("demos").mkdir(parents=True, exist_ok=True)
    outputs = [f"=== train {level} ==="]

    name = demo_name(level, episodes)
    train_native = STORAGE / "demos" / f"{name}.pkl"
    valid_native = STORAGE / "demos" / f"{name}_valid.pkl"

    outputs.append(
        run([
            sys.executable,
            "convert_bot_demos_to_babyai.py",
            "--input",
            str(train_path(level, episodes)),
            "--output",
            str(train_native),
            "--limit",
            str(episodes),
        ])
    )
    outputs.append(
        run([
            sys.executable,
            "convert_bot_demos_to_babyai.py",
            "--input",
            str(valid_path(level, val_episodes)),
            "--output",
            str(valid_native),
            "--limit",
            str(val_episodes),
        ])
    )

    model = model_name(level, episodes, epochs)
    outputs.append(
        run([
            sys.executable,
            str(BABYAI_SRC / "scripts" / "train_il.py"),
            "--env",
            env_id(level),
            "--demos",
            name,
            "--episodes",
            str(episodes),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--epoch-length",
            str(epoch_length),
            "--batch-size",
            str(batch_size),
            "--val-episodes",
            str(val_episodes),
            "--log-interval",
            "1",
            "--val-interval",
            "1",
            "--patience",
            str(patience),
            "--save-interval",
            "0",
        ])
    )

    return "\n".join(outputs)


def summarize(levels: list[str], episodes: int, epochs: int) -> None:
    print("\nSummary")
    for level in levels:
        log_path = STORAGE / "logs" / model_name(level, episodes, epochs) / "log.csv"
        if not log_path.exists():
            print(f"{level:12s} no log")
            continue
        rows = list(csv.DictReader(log_path.open()))
        if not rows:
            print(f"{level:12s} empty log")
            continue
        best = max(rows, key=lambda r: float(r["validation_success_rate"]))
        last = rows[-1]
        print(
            f"{level:12s} "
            f"best_success={float(best['validation_success_rate']):.3f} "
            f"last_success={float(last['validation_success_rate']):.3f} "
            f"last_valid_acc={float(last['validation_accuracy']):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--val-episodes", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch-length", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import ray
    except ImportError as exc:
        raise SystemExit(
            "Ray is not installed. In Colab run: !pip install -q 'ray[default]'"
        ) from exc

    ray.init(num_cpus=args.num_workers, ignore_reinit_error=True, include_dashboard=False)

    gen_task = ray.remote(num_cpus=1)(generate_level)
    train_task = ray.remote(num_cpus=1)(train_level)

    if not args.train_only:
        refs = [
            gen_task.remote(level, args.episodes, args.val_episodes, args.max_steps)
            for level in args.levels
        ]
        for ref in refs:
            print(ray.get(ref), flush=True)

    if not args.generate_only:
        refs = [
            train_task.remote(
                level,
                args.episodes,
                args.val_episodes,
                args.epochs,
                args.epoch_length,
                args.batch_size,
                args.patience,
            )
            for level in args.levels
        ]
        for ref in refs:
            print(ray.get(ref), flush=True)

    summarize(args.levels, args.episodes, args.epochs)


if __name__ == "__main__":
    main()
