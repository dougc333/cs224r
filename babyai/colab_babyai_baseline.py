#!/usr/bin/env python3
"""Colab-compatible BabyAI baseline runner.

This script bootstraps the legacy ICLR19 BabyAI code on Colab, applies the
small compatibility patches needed for modern Python/PyTorch/Gym, and runs
demo generation plus imitation-learning training.

Typical Colab usage:
    !python colab_babyai_baseline.py --setup
    !python colab_babyai_baseline.py --levels GoToObjMaze GoTo --episodes 10000 --epochs 20 --epoch-length 10000 --batch-size 128
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd()
BABYAI_REPO = ROOT / "babyai_iclr19"
STORAGE = ROOT / "babyai_storage"
DEMOS = ROOT / "demos_iclr19"


LEVELS = [
    "GoToObjMaze",
    "GoTo",
    "Pickup",
    "Open",
    "PutNext",
    "Synth",
    "BossLevel",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd or ROOT)


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        return
    path.write_text(text.replace(old, new))


def setup() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "gym==0.26.2", "gym-minigrid==1.0.3", "numpy<2", "blosc"])

    if not BABYAI_REPO.exists():
        run(["git", "clone", "https://github.com/mila-iqia/babyai.git", str(BABYAI_REPO)])
    run(["git", "fetch", "origin"], cwd=BABYAI_REPO)
    run(["git", "checkout", "2152dd6479d8725f52ae8c48495bd14621389cbe"], cwd=BABYAI_REPO)

    # Avoid importing training utilities when torch is unavailable during package import.
    init_path = BABYAI_REPO / "babyai" / "__init__.py"
    replace(
        init_path,
        "from . import utils\n",
        "try:\n"
        "    from . import utils\n"
        "except ModuleNotFoundError as exc:\n"
        "    if exc.name != 'torch':\n"
        "        raise\n"
        "    utils = None\n",
    )

    # Modern numpy RNG compatibility.
    minigrid_path = BABYAI_REPO / "babyai" / "minigrid" / "minigrid.py"
    replace(minigrid_path, "return self.np_random.randint(low, high)", "return int(self.np_random.integers(low, high)) if hasattr(self.np_random, 'integers') else self.np_random.randint(low, high)")
    replace(minigrid_path, "return (self.np_random.randint(0, 2) == 0)", "return (self._rand_int(0, 2) == 0)")
    replace(minigrid_path, "self.np_random.randint(xLow, xHigh),\n            self.np_random.randint(yLow, yHigh)", "self._rand_int(xLow, xHigh),\n            self._rand_int(yLow, yHigh)")
    replace(minigrid_path, "dtype=np.bool", "dtype=bool")
    replace(BABYAI_REPO / "babyai" / "bot.py", "dtype=np.bool", "dtype=bool")
    replace(BABYAI_REPO / "babyai" / "oldbot.py", "dtype=np.bool", "dtype=bool")

    # Disable modern Gym checker for old-style envs.
    replace(BABYAI_REPO / "babyai" / "imitation.py", "gym.make(item)", "gym.make(item, disable_env_checker=True)")
    replace(BABYAI_REPO / "babyai" / "imitation.py", "gym.make(self.args.env)", "gym.make(self.args.env, disable_env_checker=True)")
    replace(BABYAI_REPO / "babyai" / "evaluate.py", "gym.make(env_name)", "gym.make(env_name, disable_env_checker=True)")

    # PyTorch 2.6+ loads full model pickles with weights_only=True by default; old BabyAI saves full modules.
    replace(
        BABYAI_REPO / "babyai" / "utils" / "model.py",
        "model = torch.load(path)",
        "try:\n            model = torch.load(path, weights_only=False)\n        except TypeError:\n            model = torch.load(path)",
    )

    write_file(ROOT / "generate_bot_demos.py", GENERATE_BOT_DEMOS)
    write_file(ROOT / "convert_bot_demos_to_babyai.py", CONVERT_DEMOS)
    print("Setup complete. If Colab runtime is GPU-enabled, torch.cuda.is_available() should be True.")


def env_id(level: str) -> str:
    return level if level.startswith("BabyAI-") else f"BabyAI-{level}-v0"


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BABYAI_REPO)
    env["BABYAI_STORAGE"] = str(STORAGE)
    env["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    return env


def run_with_env(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=child_env())


def ensure_demos(level: str, episodes: int, val_episodes: int, max_steps: int) -> tuple[Path, Path]:
    DEMOS.mkdir(exist_ok=True)
    train = DEMOS / f"{level}_{episodes}.pkl.gz"
    valid = DEMOS / f"{level}_valid_{val_episodes}.pkl.gz"

    if not train.exists():
        run_with_env([sys.executable, "generate_bot_demos.py", "--env", env_id(level), "--episodes", str(episodes), "--output", str(train), "--seed", "1", "--max-steps", str(max_steps)])
    if not valid.exists():
        run_with_env([sys.executable, "generate_bot_demos.py", "--env", env_id(level), "--episodes", str(val_episodes), "--output", str(valid), "--seed", "1000000000", "--max-steps", str(max_steps)])
    return train, valid


def train_level(level: str, args: argparse.Namespace) -> None:
    train_gz, valid_gz = ensure_demos(level, args.episodes, args.val_episodes, args.max_steps)
    demo_name = f"{level.lower()}_{args.episodes}"
    STORAGE.joinpath("demos").mkdir(parents=True, exist_ok=True)

    run_with_env([sys.executable, "convert_bot_demos_to_babyai.py", "--input", str(train_gz), "--output", str(STORAGE / "demos" / f"{demo_name}.pkl"), "--limit", str(args.episodes)])
    run_with_env([sys.executable, "convert_bot_demos_to_babyai.py", "--input", str(valid_gz), "--output", str(STORAGE / "demos" / f"{demo_name}_valid.pkl"), "--limit", str(args.val_episodes)])

    model = f"{level.lower()}_il_{args.episodes // 1000}k_e{args.epochs}"
    run_with_env([
        sys.executable,
        str(BABYAI_REPO / "scripts" / "train_il.py"),
        "--env",
        env_id(level),
        "--demos",
        demo_name,
        "--episodes",
        str(args.episodes),
        "--model",
        model,
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
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--levels", nargs="+", default=LEVELS)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--val-episodes", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epoch-length", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--paper-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.setup:
        setup()
        return

    if args.paper_baseline:
        args.episodes = 1_000_000
        args.val_episodes = 512
        args.epochs = 20
        args.epoch_length = 1_000_000
        args.batch_size = 128
        args.patience = 100

    run([
        sys.executable,
        "-c",
        "import torch; print('cuda', torch.cuda.is_available()); "
        "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')",
    ])
    for level in args.levels:
        print(f"\n=== {level} ===", flush=True)
        train_level(level, args)


GENERATE_BOT_DEMOS = r'''#!/usr/bin/env python3
from __future__ import annotations
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
'''


CONVERT_DEMOS = r'''#!/usr/bin/env python3
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
'''


if __name__ == "__main__":
    main()
