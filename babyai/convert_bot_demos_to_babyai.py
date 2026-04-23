#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, pickle, sys
from pathlib import Path
import blosc, numpy as np

def main():
    sys.path.insert(0, str(Path.cwd() / "babyai_iclr19"))
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
