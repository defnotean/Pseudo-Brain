"""Sequence-based training script for KeysDoors Memory Benchmark.

Trains Reactive, GRU, and Thoughtlet models on the KeysDoors demonstration corpus.
Supports multi-seed runs, scheduled sampling, gradient clipping, and validation checks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from memory_benchmark.dataset import KeysDoorsSequenceDataset, collate_sequences
from memory_benchmark.models import make_model, N_FRAMES, ACTION_CLASSES

def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            if torch_directml.device_count() > 1:
                return torch_directml.device(1) # default to 9070 XT
            elif torch_directml.device_count() > 0:
                return torch_directml.device(0)
        except ImportError:
            pass
        return torch.device("cpu")
    if device_str.startswith("dml:"):
        import torch_directml
        idx = int(device_str.split(":")[1])
        return torch_directml.device(idx)
    return torch.device(device_str)


@torch.no_grad()
def evaluate_split(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int = 40) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        if batches >= max_batches:
            break
        frames = batch["frames"].to(device)         # (B, T, N_FRAMES, 3, 16, 16)
        actions = batch["actions"].to(device)       # (B, T)
        prev_actions = batch["prev_actions"].to(device)
        B, T = frames.shape[:2]

        h = None
        curr_prev_act = prev_actions[:, 0]

        for t in range(T):
            f_t = frames[:, t]
            logits, h = model(f_t, curr_prev_act, h)
            loss = criterion(logits, actions[:, t])
            total_loss += loss.item() * B

            preds = logits.argmax(dim=-1)
            total_correct += (preds == actions[:, t]).sum().item()
            total_tokens += B

            curr_prev_act = actions[:, t] # teacher-forced for eval validation

        batches += 1

    return {
        "val_loss": total_loss / max(total_tokens, 1),
        "val_acc": total_correct / max(total_tokens, 1),
    }


def train_model(
    model_type: str,
    corpus_dir: str | Path,
    seed: int = 42,
    total_steps: int = 3000,
    batch_size: int = 16,
    lr: float = 5e-4,
    device_str: str = "cpu",
    output_dir: Optional[str | Path] = None,
) -> Tuple[nn.Module, Dict]:
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = resolve_device(device_str)
    print(f"Training {model_type} (seed={seed}) on device {device} for {total_steps} steps...")

    corpus_path = Path(corpus_dir)
    train_ds = KeysDoorsSequenceDataset(corpus_path / "train", seq_len=32)
    dev_ds = KeysDoorsSequenceDataset(corpus_path / "dev", seq_len=32)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_sequences,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )

    model = make_model(model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    history = []

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        frames = batch["frames"].to(device)          # (B, T, N_FRAMES, 3, 16, 16)
        actions = batch["actions"].to(device)        # (B, T)
        prev_actions = batch["prev_actions"].to(device)

        B, T = frames.shape[:2]

        # Scheduled sampling rate: 0.0 -> 0.6 over first 50% steps
        ss_rate = min(0.6, (step / (0.5 * total_steps)) * 0.6)

        optimizer.zero_grad()
        seq_loss = 0.0
        h = None
        curr_prev_act = prev_actions[:, 0]

        for t in range(T):
            f_t = frames[:, t]
            logits, h = model(f_t, curr_prev_act, h)

            loss = criterion(logits, actions[:, t])
            seq_loss = seq_loss + loss

            # Scheduled sampling for next step's prev_action input
            if np.random.rand() < ss_rate:
                curr_prev_act = logits.detach().argmax(dim=-1)
            else:
                curr_prev_act = actions[:, t]

        seq_loss = seq_loss / T
        seq_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        if step % 250 == 0 or step == total_steps:
            val_metrics = evaluate_split(model, dev_loader, device)
            elapsed = time.time() - t0
            print(
                f"Step {step:4d}/{total_steps:4d} | "
                f"Train Loss: {seq_loss.item():.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.3f} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            history.append({
                "step": step,
                "train_loss": float(seq_loss.item()),
                "val_loss": float(val_metrics["val_loss"]),
                "val_acc": float(val_metrics["val_acc"]),
                "elapsed_s": elapsed,
            })

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ckpt_file = out_path / f"{model_type}_seed_{seed}.pt"
        torch.save({
            "model_type": model_type,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "history": history,
            "total_steps": total_steps,
        }, ckpt_file)
        print(f"Saved checkpoint to {ckpt_file}")

    return model, {"history": history, "device": str(device)}


def main():
    parser = argparse.ArgumentParser(description="Train memory benchmark model")
    parser.add_argument("--model", type=str, required=True, choices=["reactive", "gru", "thoughtlet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--corpus_dir", type=str, default="brain/datasets/keys_doors_corpus_v1")
    parser.add_argument("--output_dir", type=str, default="brain/runs/memory_benchmark/checkpoints")

    args = parser.parse_args()
    train_model(
        model_type=args.model,
        corpus_dir=args.corpus_dir,
        seed=args.seed,
        total_steps=args.steps,
        batch_size=args.batch_size,
        device_str=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
