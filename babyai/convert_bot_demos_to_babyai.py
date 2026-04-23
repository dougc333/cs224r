#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, pickle, sys
from pathlib import Path
import blosc, numpy as np


def add_local_babyai_to_path() -> Path:
    root = Path(__file__).resolve().parent
    for dirname in ("original_babyai_iclr19", "babyai_iclr19", "original_babyai"):
        candidate = root / dirname
        if candidate.exists() and (candidate / "babyai").is_dir():
            sys.path.insert(0, str(candidate))
            return candidate
    raise SystemExit(
        "Could not locate a local BabyAI checkout with a babyai package. "
        "Expected one of: original_babyai_iclr19/, babyai_iclr19/, original_babyai/."
    )

def main():
    add_local_babyai_to_path()
    from babyai import utils
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    data = pickle.load(gzip.open(args.input, "rb"))
    episodes = data["episodes"][:args.limit]
    demos = []
    for ep in episodes:
        obss = ep["observations"][:-1]
        mission = ep["mission"] or obss[0]["mission"]
        images = np.array([obs["image"] for obs in obss])
        directions = [int(obs["direction"]) for obs in obss]
        actions = [int(a) for a in ep["actions"]]
        demos.append((mission, blosc.pack_array(images), directions, actions))
    utils.save_demos(demos, args.output)
    print(f"wrote {len(demos)} demos to {args.output}")
if __name__ == "__main__":
    main()
