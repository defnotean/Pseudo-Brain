#!/usr/bin/env python3
"""Training script for fly-inspired VectorizedPseudoBrain variant.

Trains on embodied tasks (Pac-Man-like maze chase) using CPU-only,
single-threaded, deterministic compute per AGENTS.md.
"""

import argparse
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from irene_brain.model.fly_inspired import create_fly_inspired_model


def deterministic_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def generate_batch(batch_size: int = 16, input_dim: int = 64,
                   seed: int = 0) -> tuple:
    rng = np.random.RandomState(seed)
    x = rng.randn(batch_size, input_dim).astype(np.float32)
    reward = rng.randn(batch_size).astype(np.float32)
    next_x = rng.randn(batch_size, input_dim).astype(np.float32)
    return (torch.from_numpy(x), torch.from_numpy(reward),
            torch.from_numpy(next_x))


def train_fly_inspired(steps: int = 5000, batch_size: int = 16,
                       lr: float = 5e-4, width: int = 120,
                       n_thoughtlets: int = 32) -> dict:
    deterministic_seed(42)
    model = create_fly_inspired_model(
        input_dim=64, thoughtlet_width=width,
        n_thoughtlets=n_thoughtlets)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    metrics = {"losses": [], "final_loss": 0.0}

    for step in range(steps):
        x, reward, next_x = generate_batch(batch_size, seed=step)
        optimizer.zero_grad()
        output = model(x, reward_signal=reward)
        loss = loss_fn(output, next_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            metrics["losses"].append(loss.item())

    metrics["final_loss"] = metrics["losses"][-1] if metrics["losses"] else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train fly-inspired VectorizedPseudoBrain variant")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--thoughtlets", type=int, default=32)
    args = parser.parse_args()

    print(f"Training fly-inspired model: steps={args.steps}, "
          f"batch={args.batch}, width={args.width}")
    metrics = train_fly_inspired(
        steps=args.steps, batch_size=args.batch,
        lr=args.lr, width=args.width,
        n_thoughtlets=args.thoughtlets)
    print(f"  Final loss: {metrics['final_loss']:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
