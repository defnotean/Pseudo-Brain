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
def evaluate_split(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int = 15) -> Dict[str, float]:
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
    device_str: str = "auto",
    output_dir: Optional[str | Path] = None,
    save_every: int = 250,
    resume: bool = False,
    resume_from: Optional[str | Path] = None,
    telemetry: Optional[Any] = None,
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

    out_path = Path(output_dir) if output_dir else Path("brain/runs/memory_benchmark/checkpoints")
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = out_path / f"{model_type}_seed_{seed}.pt"
    latest_file = out_path / f"{model_type}_seed_{seed}_latest.pt"
    best_file = out_path / f"{model_type}_seed_{seed}_best.pt"

    best_val_loss = float("inf")
    best_state_dict = None

    step = 0
    history = []

    target_resume = None
    if resume_from:
        target_resume = Path(resume_from)
    elif resume:
        if latest_file.exists():
            target_resume = latest_file
        elif ckpt_file.exists():
            target_resume = ckpt_file

    if target_resume and target_resume.exists():
        ckpt_data = torch.load(target_resume, map_location="cpu")
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.to(device)
        if "optimizer_state_dict" in ckpt_data and ckpt_data["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        step = ckpt_data.get("step", 0)
        history = ckpt_data.get("history", [])
        print(f"  [{model_type.upper()}] Successfully resumed at step {step}/{total_steps}")
        if step >= total_steps:
            print(f"  [{model_type.upper()}] Training already complete ({step} >= {total_steps} steps).")
            return model, {"history": history, "device": str(device)}

    start_step = step
    t0 = time.time()
    train_iter = iter(train_loader)

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

        if model_type == "cgp_thoughtlet" and hasattr(model, "forward_sequence"):
            logits, m_logits, z_hat, all_z = model.forward_sequence(
                frames, prev_actions, actions, ss_rate=ss_rate
            )
            act_loss = criterion(logits.reshape(B * T, -1), actions.reshape(B * T))

            true_hk = batch["has_key"].to(device).float()
            true_do = batch.get("door_open", torch.zeros_like(true_hk)).to(device).float()
            true_m = torch.stack([true_hk, true_do], dim=-1)

            milestone_loss = F.binary_cross_entropy_with_logits(m_logits, true_m)
            if z_hat.size(1) > 1:
                latent_loss = F.mse_loss(z_hat[:, :-1], all_z[:, 1:].detach())
            else:
                latent_loss = torch.tensor(0.0, device=device)

            seq_loss = act_loss + 0.5 * milestone_loss + 0.1 * latent_loss
        else:
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

        if telemetry is not None:
            telemetry.log_step(
                step=step,
                act_loss=seq_loss.item(),
            )

        if step % 500 == 0 or step == total_steps:
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

            # Save periodic latest checkpoint
            checkpoint_state = {
                "model_type": model_type,
                "seed": seed,
                "step": step,
                "total_steps": total_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "device": str(device),
            }
            torch.save(checkpoint_state, latest_file)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(checkpoint_state, best_file)
                print(f"  * New best validation checkpoint saved (loss: {best_val_loss:.4f})")

            if step == total_steps:
                if best_state_dict is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
                    print(f"  * Restored best validation weights (loss: {best_val_loss:.4f}) for evaluation")
                    checkpoint_state["model_state_dict"] = model.state_dict()
                torch.save(checkpoint_state, ckpt_file)
                print(f"Saved final checkpoint to {ckpt_file}")

    return model, {"history": history, "device": str(device)}


def main():
    parser = argparse.ArgumentParser(description="Train memory benchmark model")
    parser.add_argument("--model", type=str, required=True, choices=["reactive", "gru", "thoughtlet", "cgp_thoughtlet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--corpus_dir", type=str, default="brain/datasets/keys_doors_corpus_v1")
    parser.add_argument("--output_dir", type=str, default="brain/runs/memory_benchmark/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()
    train_model(
        model_type=args.model,
        corpus_dir=args.corpus_dir,
        seed=args.seed,
        total_steps=args.steps,
        batch_size=args.batch_size,
        device_str=args.device,
        output_dir=args.output_dir,
        save_every=args.save_every,
        resume=args.resume,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
