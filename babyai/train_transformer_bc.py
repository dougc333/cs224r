#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import blosc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent


def find_babyai_src() -> Path:
    for dirname in ("original_babyai_iclr19", "babyai_iclr19", "original_babyai"):
        candidate = ROOT / dirname
        if candidate.exists() and (candidate / "babyai").is_dir():
            return candidate
    raise SystemExit(
        "Could not locate a local BabyAI checkout. Expected one of "
        "original_babyai_iclr19/, babyai_iclr19/, or original_babyai/."
    )


BABYAI_SRC = find_babyai_src()
if str(BABYAI_SRC) not in os.sys.path:
    os.sys.path.insert(0, str(BABYAI_SRC))

import babyai.utils as babyai_utils  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str):
    import gym

    try:
        env = gym.make(env_name, disable_env_checker=True)
    except TypeError:
        env = gym.make(env_name)
    return getattr(env, "unwrapped", env)


def reset_env(env, seed: int | None = None):
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def step_env(env, action: int):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result


def tokenize_mission(text: str) -> list[str]:
    return re.findall(r"([a-z]+)", text.lower())


@dataclass
class StepExample:
    image: np.ndarray
    direction: int
    mission_tokens: list[int]
    action: int


class MissionVocab:
    def __init__(self):
        self.stoi = {"<pad>": 0, "<cls>": 1}

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi["<cls>"]]
        for token in tokenize_mission(text):
            if token not in self.stoi:
                self.stoi[token] = len(self.stoi)
            ids.append(self.stoi[token])
        return ids

    @property
    def size(self) -> int:
        return len(self.stoi)


def load_step_examples(
    demos_name: str,
    limit_episodes: int | None = None,
    vocab: MissionVocab | None = None,
) -> tuple[list[StepExample], MissionVocab]:
    demos_path = Path(babyai_utils.get_demos_path(demos_name, None, None, valid=False))
    demos = babyai_utils.load_demos(str(demos_path))
    if limit_episodes is not None:
        demos = demos[:limit_episodes]

    vocab = vocab or MissionVocab()
    examples: list[StepExample] = []
    for mission, packed_images, directions, actions in demos:
        images = blosc.unpack_array(packed_images)
        token_ids = vocab.encode(mission)
        for image, direction, action in zip(images, directions, actions):
            examples.append(
                StepExample(
                    image=image.astype(np.int64),
                    direction=int(direction),
                    mission_tokens=token_ids,
                    action=int(action),
                )
            )
    return examples, vocab


class StepDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[StepExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> StepExample:
        return self.examples[index]


def collate_batch(batch: list[StepExample]) -> dict[str, torch.Tensor]:
    max_len = max(len(item.mission_tokens) for item in batch)
    mission = torch.zeros(len(batch), max_len, dtype=torch.long)
    mission_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    image = torch.tensor(np.stack([item.image for item in batch]), dtype=torch.long)
    direction = torch.tensor([item.direction for item in batch], dtype=torch.long)
    action = torch.tensor([item.action for item in batch], dtype=torch.long)

    for row, item in enumerate(batch):
        ids = torch.tensor(item.mission_tokens, dtype=torch.long)
        mission[row, : len(ids)] = ids
        mission_mask[row, : len(ids)] = True

    return {
        "mission": mission,
        "mission_mask": mission_mask,
        "image": image,
        "direction": direction,
        "action": action,
    }


class BabyAITransformerPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        action_dim: int = 7,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.dir_embed = nn.Embedding(4, d_model)

        self.obj_embed = nn.Embedding(16, d_model)
        self.color_embed = nn.Embedding(16, d_model)
        self.state_embed = nn.Embedding(16, d_model)
        self.pos_embed = nn.Embedding(49, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(
        self,
        mission: torch.Tensor,
        mission_mask: torch.Tensor,
        image: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = mission.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        mission_tokens = self.word_embed(mission)

        grid = image.reshape(batch_size, 49, 3)
        pos_ids = torch.arange(49, device=image.device).unsqueeze(0).expand(batch_size, -1)
        grid_tokens = (
            self.obj_embed(grid[:, :, 0])
            + self.color_embed(grid[:, :, 1])
            + self.state_embed(grid[:, :, 2])
            + self.pos_embed(pos_ids)
        )

        direction_token = self.dir_embed(direction).unsqueeze(1)
        tokens = torch.cat([cls, mission_tokens, direction_token, grid_tokens], dim=1)

        cls_mask = torch.ones(batch_size, 1, device=mission.device, dtype=torch.bool)
        direction_mask = torch.ones(batch_size, 1, device=mission.device, dtype=torch.bool)
        grid_mask = torch.ones(batch_size, 49, device=mission.device, dtype=torch.bool)
        padding_mask = ~torch.cat([cls_mask, mission_mask, direction_mask, grid_mask], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        cls_out = encoded[:, 0]
        return self.head(cls_out)


def evaluate_action_accuracy(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["mission"], batch["mission_mask"], batch["image"], batch["direction"])
            loss = F.cross_entropy(logits, batch["action"])
            preds = logits.argmax(dim=1)
            total += batch["action"].numel()
            correct += (preds == batch["action"]).sum().item()
            loss_sum += float(loss.item()) * batch["action"].numel()
    return correct / max(total, 1), loss_sum / max(total, 1)


def evaluate_success(
    model,
    env_name: str,
    vocab: MissionVocab,
    episodes: int,
    seed: int,
    device: torch.device,
    max_steps: int = 4096,
) -> tuple[float, float]:
    env = make_env(env_name)
    successes = 0
    returns = []
    try:
        model.eval()
        for episode_id in range(episodes):
            obs = reset_env(env, seed=seed + episode_id)
            total_reward = 0.0
            for _ in range(max_steps):
                mission_ids = vocab.encode(obs["mission"])
                batch = collate_batch([
                    StepExample(
                        image=np.asarray(obs["image"], dtype=np.int64),
                        direction=int(obs["direction"]),
                        mission_tokens=mission_ids,
                        action=0,
                    )
                ])
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    logits = model(batch["mission"], batch["mission_mask"], batch["image"], batch["direction"])
                action = int(logits.argmax(dim=1).item())
                obs, reward, done, _ = step_env(env, action)
                total_reward += float(reward)
                if done:
                    if reward > 0:
                        successes += 1
                    break
            returns.append(total_reward)
    finally:
        if hasattr(env, "close"):
            env.close()
    return successes / max(episodes, 1), float(np.mean(returns)) if returns else 0.0


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_examples, vocab = load_step_examples(args.demos, limit_episodes=args.episodes)
    valid_examples, _ = load_step_examples(
        f"{args.demos}_valid",
        limit_episodes=args.val_episodes,
        vocab=vocab,
    )

    train_loader = torch.utils.data.DataLoader(
        StepDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    valid_loader = torch.utils.data.DataLoader(
        StepDataset(valid_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = BabyAITransformerPolicy(
        vocab_size=vocab.size,
        action_dim=args.action_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = ROOT / "transformer_runs" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    best_success = -1.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["mission"], batch["mission_mask"], batch["image"], batch["direction"])
            loss = F.cross_entropy(logits, batch["action"])
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            batch_total = batch["action"].numel()
            running_loss += float(loss.item()) * batch_total
            running_correct += (preds == batch["action"]).sum().item()
            running_total += batch_total

            if batch_index % args.log_every == 0:
                print(
                    f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                    f"loss={running_loss / max(running_total, 1):.4f} "
                    f"acc={running_correct / max(running_total, 1):.4f}",
                    flush=True,
                )

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        valid_acc, valid_loss = evaluate_action_accuracy(model, valid_loader, device)
        success_rate, mean_return = evaluate_success(
            model,
            env_name=args.env,
            vocab=vocab,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            device=device,
            max_steps=args.max_steps,
        )

        elapsed = int(time.time() - start)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} "
            f"success={success_rate:.4f} return={mean_return:.4f} "
            f"elapsed={elapsed}s",
            flush=True,
        )

        latest_path = output_dir / "latest.pt"
        payload = {
            "model_state": model.state_dict(),
            "vocab": vocab.stoi,
            "args": vars(args),
            "epoch": epoch,
            "metrics": {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "success_rate": success_rate,
                "mean_return": mean_return,
            },
        }
        torch.save(payload, latest_path)
        if success_rate > best_success:
            best_success = success_rate
            torch.save(payload, output_dir / "best.pt")
            print(f"saved best checkpoint to {output_dir / 'best.pt'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BabyAI-Pickup-v0")
    parser.add_argument("--demos", default="pickup_10000")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--val-episodes", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=int(1e9))
    parser.add_argument("--model", default="pickup_transformer_10k")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
