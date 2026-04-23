#!/usr/bin/env python3
"""Ray-friendly BabyAI Colab runner with separate demo/train commands.

Use after:
    python colab_babyai_baseline.py --setup
    pip install -q "ray[default]"

Examples:
    # One level only, then stop.
    python ray_colab_babyai.py run-level --level GoToObjMaze

    # Parallel demo generation across default levels.
    python ray_colab_babyai.py demo --all-default-levels --num-workers 4

    # Train one level only.
    python ray_colab_babyai.py train --level GoTo

    # Parallel CPU training across explicitly selected levels.
    python ray_colab_babyai.py train --levels GoToObjMaze GoTo --num-workers 2 --cpus-per-train 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
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

TASK_TIMINGS: list[tuple[str, str, float]] = []


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def timed(label: str, fn):
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    return result, elapsed


def record_timing(level: str, task: str, elapsed: float) -> None:
    TASK_TIMINGS.append((level, task, elapsed))


def env_id(level: str) -> str:
    return level if level.startswith("BabyAI-") else f"BabyAI-{level}-v0"


def train_path(level: str, episodes: int) -> Path:
    return DEMOS / f"{level}_{episodes}.pkl.gz"


def valid_path(level: str, val_episodes: int) -> Path:
    return DEMOS / f"{level}_valid_{val_episodes}.pkl.gz"


def shard_path(level: str, split: str, total: int, shard_index: int) -> Path:
    return DEMOS / "shards" / f"{level}_{split}_{total}_shard{shard_index}.pkl.gz"


def demo_name(level: str, episodes: int) -> str:
    return f"{level.lower()}_{episodes}"


def model_name(level: str, episodes: int, epochs: int) -> str:
    return f"{level.lower()}_cpu_ray_{episodes // 1000}k_e{epochs}"


def count_generated_episodes(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import gzip

        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        return len(data.get("episodes", []))
    except Exception:
        return None


def count_native_demos(path: Path) -> int | None:
    if not path.exists():
        return None


def merge_generated_shards(shard_paths: list[Path], output_path: Path, expected_total: int, env_name: str, seed: int, force: bool) -> str:
    output_count = count_generated_episodes(output_path)
    if not force and output_count is not None and output_count >= expected_total:
        return f"using existing merged {output_path} ({output_count} demos)"

    episodes = []
    attempts = 0
    for path in shard_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing shard {path}")
        import gzip

        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        shard_episodes = data.get("episodes", [])
        episodes.extend(shard_episodes)
        attempts += int(data.get("attempts", len(shard_episodes)))

    if len(episodes) < expected_total:
        raise RuntimeError(f"Only {len(episodes)} demos available, expected {expected_total}")

    episodes = episodes[:expected_total]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import gzip

    with gzip.open(output_path, "wb") as f:
        pickle.dump(
            {
                "env": env_name,
                "seed": seed,
                "attempts": attempts,
                "episodes": episodes,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return f"merged {len(episodes)} demos into {output_path}"
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
        return len(data)
    except Exception:
        return None


def completed_epochs(model: str) -> int:
    status_path = STORAGE / "logs" / model / "status.json"
    if status_path.exists():
        try:
            return int(json.load(status_path.open()).get("i", 0))
        except Exception:
            pass

    log_path = STORAGE / "logs" / model / "log.csv"
    if not log_path.exists():
        return 0
    try:
        rows = list(csv.DictReader(log_path.open()))
        if not rows:
            return 0
        return max(int(row["update"]) for row in rows if row.get("update"))
    except Exception:
        return 0


def model_checkpoint_exists(model: str) -> bool:
    return (STORAGE / "models" / model / "model.pt").exists()


def selected_levels(args: argparse.Namespace) -> list[str]:
    if getattr(args, "all_default_levels", False):
        return DEFAULT_LEVELS
    if getattr(args, "levels", None):
        return args.levels
    return [args.level]


def child_env(force_cpu: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BABYAI_SRC)
    env["BABYAI_STORAGE"] = str(STORAGE)
    env["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run(cmd: list[str], force_cpu: bool = True) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=child_env(force_cpu=force_cpu),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return proc.stdout


def generate_level(level: str, episodes: int, val_episodes: int, max_steps: int, force: bool) -> str:
    task_start = time.time()
    DEMOS.mkdir(exist_ok=True)
    outputs = [f"=== generate {level} ==="]

    train = train_path(level, episodes)
    train_count = count_generated_episodes(train)
    if not force and train_count is not None and train_count >= episodes:
        outputs.append(f"using existing {train} ({train_count} demos)")
    else:
        if train.exists():
            outputs.append(f"regenerating incomplete/unreadable {train} (count={train_count})")
        out, elapsed = timed("generate_train", lambda: run([
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
            ]))
        outputs.append(out)
        outputs.append(f"train demo generation elapsed: {format_duration(elapsed)}")

    valid = valid_path(level, val_episodes)
    valid_count = count_generated_episodes(valid)
    if not force and valid_count is not None and valid_count >= val_episodes:
        outputs.append(f"using existing {valid} ({valid_count} demos)")
    else:
        if valid.exists():
            outputs.append(f"regenerating incomplete/unreadable {valid} (count={valid_count})")
        out, elapsed = timed("generate_valid", lambda: run([
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
            ]))
        outputs.append(out)
        outputs.append(f"valid demo generation elapsed: {format_duration(elapsed)}")

    total_elapsed = time.time() - task_start
    outputs.append(f"generate {level} total elapsed: {format_duration(total_elapsed)}")
    return "\n".join(outputs)


def generate_shard(
    level: str,
    split: str,
    shard_index: int,
    shard_episodes: int,
    total: int,
    seed: int,
    max_steps: int,
    force: bool,
) -> str:
    path = shard_path(level, split, total, shard_index)
    existing_count = count_generated_episodes(path)
    if not force and existing_count is not None and existing_count >= shard_episodes:
        return f"using existing {path} ({existing_count} demos)"

    path.parent.mkdir(parents=True, exist_ok=True)
    out, elapsed = timed("generate_shard", lambda: run([
        sys.executable,
        "generate_bot_demos.py",
        "--env",
        env_id(level),
        "--episodes",
        str(shard_episodes),
        "--output",
        str(path),
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
    ]))
    return "\n".join([
        f"=== generate shard {level} {split} #{shard_index} ===",
        out,
        f"shard elapsed: {format_duration(elapsed)}",
    ])


def train_level(
    level: str,
    episodes: int,
    val_episodes: int,
    epochs: int,
    epoch_length: int,
    batch_size: int,
    patience: int,
    force_cpu: bool,
    force: bool,
) -> str:
    task_start = time.time()
    STORAGE.joinpath("demos").mkdir(parents=True, exist_ok=True)
    outputs = [f"=== train {level} ==="]

    name = demo_name(level, episodes)
    model = model_name(level, episodes, epochs)
    done_epochs = completed_epochs(model)
    if not force and done_epochs >= epochs and model_checkpoint_exists(model):
        return (
            f"=== train {level} ===\n"
            f"skipping completed model {model}: {done_epochs}/{epochs} epochs"
        )

    train_native = STORAGE / "demos" / f"{name}.pkl"
    valid_native = STORAGE / "demos" / f"{name}_valid.pkl"

    train_native_count = count_native_demos(train_native)
    if not force and train_native_count is not None and train_native_count >= episodes:
        outputs.append(f"using existing {train_native} ({train_native_count} demos)")
    else:
        out, elapsed = timed("convert_train", lambda: run([
                sys.executable,
                "convert_bot_demos_to_babyai.py",
                "--input",
                str(train_path(level, episodes)),
                "--output",
                str(train_native),
                "--limit",
                str(episodes),
            ], force_cpu=force_cpu))
        outputs.append(out)
        outputs.append(f"train conversion elapsed: {format_duration(elapsed)}")

    valid_native_count = count_native_demos(valid_native)
    if not force and valid_native_count is not None and valid_native_count >= val_episodes:
        outputs.append(f"using existing {valid_native} ({valid_native_count} demos)")
    else:
        out, elapsed = timed("convert_valid", lambda: run([
                sys.executable,
                "convert_bot_demos_to_babyai.py",
                "--input",
                str(valid_path(level, val_episodes)),
                "--output",
                str(valid_native),
                "--limit",
                str(val_episodes),
            ], force_cpu=force_cpu))
        outputs.append(out)
        outputs.append(f"valid conversion elapsed: {format_duration(elapsed)}")

    if done_epochs > 0:
        outputs.append(f"resuming {model} from epoch {done_epochs}/{epochs}")
    out, elapsed = timed("train", lambda: run([
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
        ], force_cpu=force_cpu))
    outputs.append(out)
    outputs.append(f"training elapsed: {format_duration(elapsed)}")

    total_elapsed = time.time() - task_start
    outputs.append(f"train {level} total elapsed: {format_duration(total_elapsed)}")
    return "\n".join(outputs)


def summarize(levels: list[str], episodes: int, epochs: int, timings: list[tuple[str, str, float]] | None = None) -> None:
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
    if timings:
        print("\nElapsed Times")
        for level, task, elapsed in timings:
            print(f"{level:12s} {task:12s} {format_duration(elapsed)}")


def init_ray(num_workers: int):
    try:
        import ray
    except ImportError as exc:
        raise SystemExit(
            "Ray is not installed. In Colab run: !pip install -q 'ray[default]'"
        ) from exc
    ray.init(num_cpus=num_workers, ignore_reinit_error=True, include_dashboard=False)
    return ray


def run_demo_command(args: argparse.Namespace) -> None:
    levels = selected_levels(args)
    command_start = time.time()
    ray = init_ray(args.num_workers)
    gen_task = ray.remote(num_cpus=1)(generate_level)
    refs = [
        gen_task.remote(level, args.episodes, args.val_episodes, args.max_steps, args.force)
        for level in levels
    ]
    timings: list[tuple[str, str, float]] = []
    for level, ref in zip(levels, refs):
        start = time.time()
        print(ray.get(ref), flush=True)
        elapsed = time.time() - start
        timings.append((level, "demo_wait", elapsed))
    timings.append(("ALL", "demo_total", time.time() - command_start))
    summarize(levels, args.episodes, args.epochs, timings)


def shard_sizes(total: int, shards: int) -> list[int]:
    base, remainder = divmod(total, shards)
    return [base + (1 if i < remainder else 0) for i in range(shards)]


def run_demo_sharded_command(args: argparse.Namespace) -> None:
    level = args.level
    command_start = time.time()
    ray = init_ray(args.num_workers)
    shard_task = ray.remote(num_cpus=1)(generate_shard)

    train_sizes = shard_sizes(args.episodes, args.shards)
    valid_sizes = shard_sizes(args.val_episodes, args.valid_shards or args.shards)

    refs = []
    train_shards = []
    seed_offset = 1
    for index, size in enumerate(train_sizes):
        path = shard_path(level, "train", args.episodes, index)
        train_shards.append(path)
        refs.append(
            shard_task.remote(
                level,
                "train",
                index,
                size,
                args.episodes,
                seed_offset,
                args.max_steps,
                args.force,
            )
        )
        seed_offset += size

    valid_shards = []
    seed_offset = 1_000_000_000
    for index, size in enumerate(valid_sizes):
        path = shard_path(level, "valid", args.val_episodes, index)
        valid_shards.append(path)
        refs.append(
            shard_task.remote(
                level,
                "valid",
                index,
                size,
                args.val_episodes,
                seed_offset,
                args.max_steps,
                args.force,
            )
        )
        seed_offset += size

    for ref in refs:
        print(ray.get(ref), flush=True)

    out, elapsed = timed(
        "merge_train",
        lambda: merge_generated_shards(
            train_shards,
            train_path(level, args.episodes),
            args.episodes,
            env_id(level),
            1,
            args.force,
        ),
    )
    print(out, flush=True)
    train_merge_elapsed = elapsed

    out, elapsed = timed(
        "merge_valid",
        lambda: merge_generated_shards(
            valid_shards,
            valid_path(level, args.val_episodes),
            args.val_episodes,
            env_id(level),
            1_000_000_000,
            args.force,
        ),
    )
    print(out, flush=True)

    timings = [
        (level, "merge_train", train_merge_elapsed),
        (level, "merge_valid", elapsed),
        ("ALL", "demo_sharded", time.time() - command_start),
    ]
    summarize([level], args.episodes, args.epochs, timings)


def run_train_command(args: argparse.Namespace) -> None:
    levels = selected_levels(args)
    command_start = time.time()
    ray = init_ray(args.num_workers)
    train_task = ray.remote(num_cpus=args.cpus_per_train)(train_level)
    refs = [
        train_task.remote(
            level,
            args.episodes,
            args.val_episodes,
            args.epochs,
            args.epoch_length,
            args.batch_size,
            args.patience,
            not args.use_gpu,
            args.force,
        )
        for level in levels
    ]
    timings: list[tuple[str, str, float]] = []
    for level, ref in zip(levels, refs):
        start = time.time()
        print(ray.get(ref), flush=True)
        elapsed = time.time() - start
        timings.append((level, "train_wait", elapsed))
    timings.append(("ALL", "train_total", time.time() - command_start))
    summarize(levels, args.episodes, args.epochs, timings)


def run_one_level_command(args: argparse.Namespace) -> None:
    level = args.level
    command_start = time.time()
    timings: list[tuple[str, str, float]] = []
    out, elapsed = timed(
        "generate_level",
        lambda: generate_level(level, args.episodes, args.val_episodes, args.max_steps, args.force),
    )
    print(out, flush=True)
    timings.append((level, "demo", elapsed))
    out, elapsed = timed(
        "train_level",
        lambda: train_level(
            level,
            args.episodes,
            args.val_episodes,
            args.epochs,
            args.epoch_length,
            args.batch_size,
            args.patience,
            not args.use_gpu,
            args.force,
        ),
    )
    print(
        out,
        flush=True,
    )
    timings.append((level, "train", elapsed))
    timings.append(("ALL", "run_total", time.time() - command_start))
    summarize([level], args.episodes, args.epochs, timings)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--level", default=DEFAULT_LEVELS[0])
    parser.add_argument("--levels", nargs="+", default=None)
    parser.add_argument("--all-default-levels", action="store_true")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--val-episodes", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch-length", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--force", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate train/valid demos only.")
    add_common_args(demo)
    demo.add_argument("--num-workers", type=int, default=4)

    demo_sharded = subparsers.add_parser("demo-sharded", help="Generate one level's demos across multiple Ray workers.")
    add_common_args(demo_sharded)
    demo_sharded.add_argument("--num-workers", type=int, default=4)
    demo_sharded.add_argument("--shards", type=int, default=4)
    demo_sharded.add_argument("--valid-shards", type=int, default=None)

    train = subparsers.add_parser("train", help="Train from existing demos only.")
    add_common_args(train)
    train.add_argument("--num-workers", type=int, default=1)
    train.add_argument("--cpus-per-train", type=int, default=2)
    train.add_argument("--use-gpu", action="store_true")

    run_level = subparsers.add_parser("run-level", help="Generate and train exactly one level, then stop.")
    add_common_args(run_level)
    run_level.add_argument("--use-gpu", action="store_true")

    summary = subparsers.add_parser("summary", help="Print summaries from existing logs.")
    add_common_args(summary)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "demo":
        run_demo_command(args)
    elif args.command == "demo-sharded":
        run_demo_sharded_command(args)
    elif args.command == "train":
        run_train_command(args)
    elif args.command == "run-level":
        run_one_level_command(args)
    elif args.command == "summary":
        summarize(selected_levels(args), args.episodes, args.epochs)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
