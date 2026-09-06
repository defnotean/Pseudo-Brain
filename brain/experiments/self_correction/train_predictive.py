"""Training pipeline for Predictive Recurrent Models with Surprise Feedback.

Jointly trains:
1. Primary action loss (CrossEntropy with scheduled sampling)
2. Auxiliary future-latent prediction loss: MSE(z_hat_{t+1}, z_{t+1})

The prediction head is trained with full gradients into the recurrent representation,
ensuring foresight is grounded in actual environmental dynamics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from memory_benchmark.dataset import KeysDoorsSequenceDataset, collate_sequences
from self_correction.predictive_models import (
    PredictiveGRUModel,
    PredictiveThoughtletModel,
)


def train_predictive_model(
    model_type: str,
    corpus_dir: str | Path,
    seed: int = 42,
    total_steps: int = 2000,
    batch_size: int = 16,
    lr: float = 5e-4,
    pred_weight: float = 0.5,
    use_plasticity: bool = False,
    device_str: str = "cpu",
    output_dir: Optional[str | Path] = None,
) -> Tuple[nn.Module, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    print(f"Training Predictive {model_type.upper()} (plasticity={use_plasticity}, seed={seed}) for {total_steps} steps...")

    corpus_path = Path(corpus_dir)
    train_ds = KeysDoorsSequenceDataset(corpus_path / "train", seq_len=32)
    dev_ds = KeysDoorsSequenceDataset(corpus_path / "dev", seq_len=32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_sequences)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

    if model_type == "gru":
        model = PredictiveGRUModel(use_plasticity=use_plasticity).to(device)
    elif model_type == "thoughtlet":
        model = PredictiveThoughtletModel(use_plasticity=use_plasticity).to(device)
    else:
        raise ValueError(f"Unknown predictive model type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    act_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    pred_criterion = nn.MSELoss()

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
        frames = batch["frames"].to(device)         # [B, T, N_FRAMES, 3, 16, 16]
        actions = batch["actions"].to(device)       # [B, T]
        prev_actions = batch["prev_actions"].to(device)

        B, T = frames.shape[:2]
        ss_rate = min(0.6, (step / (0.5 * total_steps)) * 0.6)

        # 1. Pre-encode all observation latents: [B, T, LATENT_DIM]
        # Reshape to batch encode
        all_frames_flat = frames.reshape(B * T, 4, 3, 16, 16)
        all_z = model.encode_observation(all_frames_flat).reshape(B, T, -1)

        total_act_loss = 0.0
        total_pred_loss = 0.0

        h = None
        P_t = None
        curr_prev_act = prev_actions[:, 0]
        surprise = torch.zeros(B, 1, device=device)

        for t in range(T):
            z_t = all_z[:, t]
            logits, h, e_t, P_t = model.forward_step(z_t, curr_prev_act, surprise, h, P_t)

            loss_act = act_criterion(logits, actions[:, t])
            total_act_loss = total_act_loss + loss_act

            # Predict next latent
            if t < T - 1:
                z_hat_next = model.predict_next_latent(h, actions[:, t])
                z_next_true = all_z[:, t + 1].detach()

                loss_pred = pred_criterion(z_hat_next, z_next_true)
                total_pred_loss = total_pred_loss + loss_pred

                # Surprise for next step
                with torch.no_grad():
                    surprise = torch.norm(z_hat_next.detach() - z_next_true, dim=-1, keepdim=True)
            else:
                surprise = torch.zeros(B, 1, device=device)

            if np.random.rand() < ss_rate:
                curr_prev_act = logits.detach().argmax(dim=-1)
            else:
                curr_prev_act = actions[:, t]

        total_act_loss = total_act_loss / T
        total_pred_loss = total_pred_loss / max(T - 1, 1)

        loss = total_act_loss + pred_weight * total_pred_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        if step % 250 == 0 or step == total_steps:
            elapsed = time.time() - t0
            print(
                f"Step {step:4d}/{total_steps:4d} | "
                f"Act Loss: {total_act_loss.item():.4f} | "
                f"Pred Loss: {total_pred_loss.item():.4f} | "
                f"Mean Surprise: {surprise.mean().item():.3f} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            history.append({
                "step": step,
                "act_loss": float(total_act_loss.item()),
                "pred_loss": float(total_pred_loss.item()),
                "mean_surprise": float(surprise.mean().item()),
                "elapsed_s": elapsed,
            })

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        plast_tag = "_plastic" if use_plasticity else ""
        ckpt_file = out_path / f"predictive_{model_type}{plast_tag}_seed_{seed}.pt"
        torch.save({
            "model_type": model_type,
            "seed": seed,
            "use_plasticity": use_plasticity,
            "model_state_dict": model.state_dict(),
            "history": history,
            "total_steps": total_steps,
        }, ckpt_file)
        print(f"Saved checkpoint to {ckpt_file}")

    return model, {"history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["gru", "thoughtlet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--plasticity", action="store_true")
    parser.add_argument("--corpus_dir", type=str, default="brain/datasets/keys_doors_corpus_v1")
    parser.add_argument("--output_dir", type=str, default="brain/runs/self_correction/checkpoints")

    args = parser.parse_args()
    train_predictive_model(
        model_type=args.model,
        corpus_dir=args.corpus_dir,
        seed=args.seed,
        total_steps=args.steps,
        use_plasticity=args.plasticity,
        output_dir=args.output_dir,
    )
