#!/usr/bin/env python3
"""Convert generated bot rollouts into BabyAI's native demo pickle format.

Input format is produced by `generate_bot_demos.py`.
Output format matches original BabyAI:
    (mission, blosc.pack_array(np.array(images)), directions, actions)
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import blosc
import numpy as np


def add_local_babyai_to_path() -> None:
    local_babyai = Path(__file__).resolve().parent / "original_babyai_iclr19"
    if local_babyai.exists():
        sys.path.insert(0, str(local_babyai))


def load_generated(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def convert_episode(episode: dict) -> tuple:
    observations = episode["observations"]
    actions = episode["actions"]
    if len(observations) != len(actions) + 1:
        raise ValueError(
            "Expected observations to include the initial observation plus one "
            "post-step observation per action."
        )

    pre_action_observations = observations[:-1]
    mission = episode["mission"] or pre_action_observations[0]["mission"]
    images = np.array([obs["image"] for obs in pre_action_observations])
    directions = [int(obs["direction"]) for obs in pre_action_observations]
    return mission, blosc.pack_array(images), directions, [int(a) for a in actions]


def convert_file(input_path: Path, output_path: Path, limit: int | None) -> None:
    add_local_babyai_to_path()
    from babyai import utils

    generated = load_generated(input_path)
    episodes = generated["episodes"]
    if limit is not None:
        episodes = episodes[:limit]

    demos = [convert_episode(episode) for episode in episodes]
    utils.save_demos(demos, str(output_path))
    print(f"wrote {len(demos)} BabyAI demos to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_file(Path(args.input), Path(args.output), args.limit)


if __name__ == "__main__":
    main()
