"""Training pipeline for Predictive Recurrent Models with Surprise Feedback.

Jointly trains:
1. Primary action loss (CrossEntropy with scheduled sampling)
2. Auxiliary future-latent prediction loss: MSE(z_hat_{t+1}, z_{t+1})

Features:
- Automatic device resolution (CUDA on Colab / DGX, CPU fallback)
- Resumable checkpoints with optimizer state and step count
- Unattended training with ETA and periodic disk persistence
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.device import resolve_device, get_hardware_summary
from memory_benchmark.dataset import KeysDoorsSequenceDataset, collate_sequences
from self_correction.models import (
    PredictiveGRUModel,
    PredictiveThoughtletModel,
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def train_predictive_model(
    model_type: str,
    corpus_dir: str | Path,
    seed: int = 42,
    total_steps: int = 2000,
    batch_size: int = 16,
    lr: float = 5e-4,
    pred_weight: float = 0.5,
    use_plasticity: bool = False,
    device_str: str = "auto",
    output_dir: Optional[str | Path] = None,
    save_every: int = 250,
    resume: bool = False,
    resume_from: Optional[str | Path] = None,
    telemetry: Optional[Any] = None,
) -> Tuple[nn.Module, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_str)

    hw = get_hardware_summary()
    dev_name = hw["gpu_name"] if hw["cuda_available"] else "CPU"
    print(f"[PREDICTIVE_{model_type.upper()}] Starting training on {dev_name} (plasticity={use_plasticity}, seed={seed}, steps={total_steps})...")

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

    out_path = Path(output_dir) if output_dir else Path("brain/runs/self_correction/checkpoints")
    out_path.mkdir(parents=True, exist_ok=True)

    plast_tag = "_plastic" if use_plasticity else ""
    ckpt_file = out_path / f"predictive_{model_type}{plast_tag}_seed_{seed}.pt"
    latest_file = out_path / f"predictive_{model_type}{plast_tag}_seed_{seed}_latest.pt"

    step = 0
    history = []

    # Handle resumption
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
        print(f"  [PREDICTIVE_{model_type.upper()}] Successfully resumed at step {step}/{total_steps}")
        if step >= total_steps:
            print(f"  [PREDICTIVE_{model_type.upper()}] Training already complete ({step} >= {total_steps} steps).")
            return model, {"history": history}

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
        frames = batch["frames"].to(device)         # [B, T, N_FRAMES, 3, 16, 16]
        actions = batch["actions"].to(device)       # [B, T]
        prev_actions = batch["prev_actions"].to(device)

        B, T = frames.shape[:2]
        ss_rate = min(0.6, (step / max(0.5 * total_steps, 1)) * 0.6)

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
            step_out = model.forward_step(z_t, curr_prev_act, surprise, h, P_t)
            logits, h, e_t, P_t = step_out[0], step_out[1], step_out[2], step_out[3]

            loss_act = act_criterion(logits, actions[:, t])
            total_act_loss = total_act_loss + loss_act

            # Predict next latent
            if t < T - 1:
                z_hat_next = model.predict_next_latent(h, actions[:, t])
                z_next_true = all_z[:, t + 1].detach()

                loss_pred = pred_criterion(z_hat_next, z_next_true)
                total_pred_loss = total_pred_loss + loss_pred

                with torch.no_grad():
                    surprise = model.compute_surprise(z_hat_next.detach(), z_next_true)
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

        if telemetry is not None:
            telemetry.log_step(
                step=step,
                act_loss=total_act_loss.item(),
                pred_loss=total_pred_loss.item(),
                surprise=surprise.mean().item(),
            )

        if step % save_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            steps_done = step - start_step
            rate = steps_done / max(elapsed, 1e-5)
            eta_s = (total_steps - step) / max(rate, 1e-5)
            eta_m, eta_sec = int(eta_s // 60), int(eta_s % 60)

            print(
                f"  [PREDICTIVE_{model_type.upper()}] Step {step:4d}/{total_steps:4d} | "
                f"Act Loss: {total_act_loss.item():.4f} | "
                f"Pred Loss: {total_pred_loss.item():.4f} | "
                f"Mean Surprise: {surprise.mean().item():.3f} | "
                f"Elapsed: {elapsed:.1f}s | "
                f"ETA: {eta_m}m{eta_sec:02d}s"
            )
            history.append({
                "step": step,
                "act_loss": float(total_act_loss.item()),
                "pred_loss": float(total_pred_loss.item()),
                "mean_surprise": float(surprise.mean().item()),
                "elapsed_s": elapsed,
            })

            checkpoint_state = {
                "model_type": model_type,
                "seed": seed,
                "step": step,
                "total_steps": total_steps,
                "use_plasticity": use_plasticity,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "git_commit": get_git_commit(),
                "timestamp": time.time(),
                "device": str(device),
            }
            torch.save(checkpoint_state, latest_file)
            if step == total_steps:
                torch.save(checkpoint_state, ckpt_file)
                print(f"  [PREDICTIVE_{model_type.upper()}] Saved final checkpoint to {ckpt_file}")

    return model, {"history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["gru", "thoughtlet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--plasticity", action="store_true")
    parser.add_argument("--corpus_dir", type=str, default="brain/datasets/keys_doors_corpus_v1")
    parser.add_argument("--output_dir", type=str, default="brain/runs/self_correction/checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()
    train_predictive_model(
        model_type=args.model,
        corpus_dir=args.corpus_dir,
        seed=args.seed,
        total_steps=args.steps,
        use_plasticity=args.plasticity,
        output_dir=args.output_dir,
        device_str=args.device,
        save_every=args.save_every,
        resume=args.resume,
        resume_from=args.resume_from,
    )
