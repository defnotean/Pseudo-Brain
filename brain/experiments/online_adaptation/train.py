"""Training pipeline for Rapid Online Adaptation benchmark models.

Jointly trains:
1. Primary action loss (CrossEntropy with scheduled sampling)
2. Auxiliary future-latent prediction loss: MSE(z_hat_{t+1}, z_{t+1})

Features:
- Automatic device resolution (CUDA on Colab / DGX, CPU fallback)
- Resumable checkpoints with optimizer state, scheduler, and step count
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
from online_adaptation.dataset import HiddenRuleSequenceDataset, collate_sequences
from online_adaptation.models import (
    ReactiveModel,
    PredictiveGRUModel,
    PredictiveThoughtletModel,
    PredictiveCGPThoughtletModel,
    ACTION_CLASSES,
)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def train_model(
    model_type: str,
    corpus_dir: str | Path,
    seed: int = 42,
    total_steps: int = 1500,
    total_samples: Optional[int] = None,
    batch_size: int = 16,
    grad_accum_steps: int = 1,
    lr: float = 5e-4,
    pred_weight: float = 0.5,
    device_str: str = "auto",
    output_dir: Optional[str | Path] = None,
    save_every: int = 250,
    resume: bool = False,
    resume_from: Optional[str | Path] = None,
    telemetry: Optional[Any] = None,
    compile_model: bool = False,
    use_amp: bool = False,
    use_legacy_forward: bool = False,
) -> Tuple[nn.Module, Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_str)

    grad_accum_steps = max(1, grad_accum_steps)
    if total_samples is not None:
        total_steps = max(1, total_samples // batch_size)

    effective_batch_size = batch_size * grad_accum_steps
    total_optimizer_updates = max(1, total_steps // grad_accum_steps)

    hw = get_hardware_summary()
    dev_name = hw["gpu_name"] if hw["cuda_available"] else "CPU"
    print(
        f"[{model_type.upper()}] Starting training on {dev_name} (device={device}, seed={seed}, "
        f"batch_size={batch_size}, accum={grad_accum_steps}, eff_batch={effective_batch_size}, "
        f"steps={total_steps}, updates={total_optimizer_updates})..."
    )

    corpus_path = Path(corpus_dir)
    train_ds = HiddenRuleSequenceDataset(corpus_path / "train", seq_len=32)
    dev_ds = HiddenRuleSequenceDataset(corpus_path / "dev", seq_len=32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_sequences)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

    if model_type == "reactive":
        model = ReactiveModel().to(device)
    elif model_type == "gru":
        model = PredictiveGRUModel(use_plasticity=False).to(device)
    elif model_type == "plastic_gru":
        model = PredictiveGRUModel(use_plasticity=True).to(device)
    elif model_type == "thoughtlet":
        model = PredictiveThoughtletModel(use_plasticity=False).to(device)
    elif model_type == "plastic_thoughtlet":
        model = PredictiveThoughtletModel(use_plasticity=True).to(device)
    elif model_type == "cgp_thoughtlet":
        model = PredictiveCGPThoughtletModel().to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    act_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    pred_criterion = nn.MSELoss()

    out_path = Path(output_dir) if output_dir else Path("brain/runs/online_adaptation/checkpoints")
    out_path.mkdir(parents=True, exist_ok=True)

    ckpt_file = out_path / f"{model_type}_seed_{seed}.pt"
    latest_file = out_path / f"{model_type}_seed_{seed}_latest.pt"

    step = 0
    history = []

    # Handle checkpoint resumption
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
            return model, {"history": history}

    if compile_model and device.type == "cuda" and not use_legacy_forward:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"  [{model_type.upper()}] torch.compile(model, mode='reduce-overhead') activated.")
        except Exception as e:
            print(f"  [{model_type.upper()}] Warning: torch.compile failed: {e}. Running uncompiled.")

    start_step = step
    t0 = time.time()
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        frames = batch["frames"].to(device)        # [B, T, 4, 3, 16, 16]
        actions = batch["actions"].to(device)      # [B, T]
        prev_actions = batch["prev_actions"].to(device)
        rewards = batch["rewards"].to(device) if "rewards" in batch else None

        B, T = frames.shape[:2]

        if model_type == "reactive":
            frames_flat = frames.reshape(B * T, 4, 3, 16, 16)
            actions_flat = actions.reshape(B * T)

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(frames_flat)
                    loss = act_criterion(logits, actions_flat)
            else:
                logits = model(frames_flat)
                loss = act_criterion(logits, actions_flat)

            loss_unscaled = loss.item()
            loss = loss / grad_accum_steps
            loss.backward()

            act_loss_val = loss_unscaled
            pred_loss_val = 0.0
            surprise_val = 0.0
            surprise_var = 0.0
            surprise_max = 0.0
            gate_val = 0.0

        elif use_legacy_forward:
            # Unrolled eager loop (Cell A / B original pipeline)
            all_frames_flat = frames.reshape(B * T, 4, 3, 16, 16)
            all_z = model.encode_observation(all_frames_flat).reshape(B, T, -1)
            h = None
            P_t = None
            curr_prev_act = prev_actions[:, 0]
            surprise = torch.zeros(B, 1, device=device)
            loss_act = torch.tensor(0.0, device=device)
            loss_pred = torch.tensor(0.0, device=device)
            surp_list = []
            gate_list = []

            for t in range(T):
                z_t = all_z[:, t]
                if model_type == "gru":
                    logits, h, e_t, P_t, _, gate = model.forward_step(
                        z_t, curr_prev_act, surprise, h=h, P_t=P_t, return_gate=True
                    )
                else:
                    logits, h, e_t, P_t, _, gate = model.forward_step(
                        z_t, curr_prev_act, surprise, thoughts=h, P_t=P_t, return_gate=True
                    )
                if gate is not None:
                    gate_list.append(gate)

                loss_act = loss_act + act_criterion(logits, actions[:, t])

                if t < T - 1:
                    z_hat = model.predict_next_latent(h, actions[:, t])
                    z_next_true = all_z[:, t + 1].detach()
                    loss_pred = loss_pred + pred_criterion(z_hat, z_next_true)
                    surprise = torch.norm(z_hat.detach() - z_next_true, dim=-1, keepdim=True)
                    surp_list.append(surprise)

                curr_prev_act = actions[:, t]

            loss_act = loss_act / T
            loss_pred = loss_pred / max(T - 1, 1)
            loss = loss_act + pred_weight * loss_pred
            loss_unscaled = loss.item()
            loss = loss / grad_accum_steps
            loss.backward()

            act_loss_val = loss_act.item()
            pred_loss_val = loss_pred.item()
            if surp_list:
                s_cat = torch.cat(surp_list, dim=0)
                surprise_val = float(s_cat.mean().item())
                surprise_var = float(s_cat.var().item()) if s_cat.numel() > 1 else 0.0
                surprise_max = float(s_cat.max().item())
            else:
                surprise_val = surprise_var = surprise_max = 0.0
            gate_val = float(torch.cat(gate_list, dim=0).mean().item()) if gate_list else 0.0

        else:
            # Accelerated forward_sequence with diagnostics
            all_frames_flat = frames.reshape(B * T, 4, 3, 16, 16)
            ss_rate = min(0.5, (step / max(0.5 * total_steps, 1)) * 0.5)

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    all_z = model.encode_observation(all_frames_flat).reshape(B, T, -1)
                    res = model.forward_sequence(
                        all_z, prev_actions, actions, rewards=rewards, ss_rate=ss_rate, return_diagnostics=True
                    )
                    if len(res) == 5:
                        all_logits, all_z_hat, all_r_hat, surprise, diag = res
                        loss_rew = pred_criterion(all_r_hat, rewards[:, 1:]) if (all_r_hat.numel() > 0 and rewards is not None) else torch.tensor(0.0, device=device)
                    else:
                        all_logits, all_z_hat, surprise, diag = res
                        loss_rew = torch.tensor(0.0, device=device)

                    loss_act = act_criterion(all_logits.reshape(B * T, -1), actions.reshape(B * T))
                    if all_z_hat.shape[1] > 0:
                        loss_pred = pred_criterion(all_z_hat, all_z[:, 1:].detach())
                    else:
                        loss_pred = torch.tensor(0.0, device=device)
                    loss = loss_act + pred_weight * loss_pred + 0.5 * loss_rew
            else:
                all_z = model.encode_observation(all_frames_flat).reshape(B, T, -1)
                res = model.forward_sequence(
                    all_z, prev_actions, actions, rewards=rewards, ss_rate=ss_rate, return_diagnostics=True
                )
                if len(res) == 5:
                    all_logits, all_z_hat, all_r_hat, surprise, diag = res
                    loss_rew = pred_criterion(all_r_hat, rewards[:, 1:]) if (all_r_hat.numel() > 0 and rewards is not None) else torch.tensor(0.0, device=device)
                else:
                    all_logits, all_z_hat, surprise, diag = res
                    loss_rew = torch.tensor(0.0, device=device)

                loss_act = act_criterion(all_logits.reshape(B * T, -1), actions.reshape(B * T))
                if all_z_hat.shape[1] > 0:
                    loss_pred = pred_criterion(all_z_hat, all_z[:, 1:].detach())
                else:
                    loss_pred = torch.tensor(0.0, device=device)
                loss = loss_act + pred_weight * loss_pred + 0.5 * loss_rew

            loss_unscaled = loss.item()
            loss = loss / grad_accum_steps
            loss.backward()

            act_loss_val = loss_act.item()
            pred_loss_val = loss_pred.item()
            all_surps = diag.get("all_surprises", surprise)
            if all_surps.numel() > 0:
                surprise_val = float(all_surps.mean().item())
                surprise_var = float(all_surps.var().item()) if all_surps.numel() > 1 else 0.0
                surprise_max = float(all_surps.max().item())
            else:
                surprise_val = surprise_var = surprise_max = 0.0

            all_gates = diag.get("all_gates", None)
            if all_gates is not None and all_gates.numel() > 0:
                gate_val = float(all_gates.mean().item())
            else:
                gate_val = 0.0

        grad_norm_val = 0.0
        update_norm_val = 0.0
        is_accum_boundary = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == total_steps)

        if is_accum_boundary:
            grad_params = [p for p in model.parameters() if p.grad is not None]
            if grad_params:
                grad_norm_val = float(torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in grad_params]), 2).item())
                old_params = [p.detach().clone() for p in grad_params]
            else:
                grad_norm_val = 0.0
                old_params = []

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if old_params:
                update_norm_val = float(torch.norm(torch.stack([torch.norm(p.detach() - old_p, 2) for p, old_p in zip(grad_params, old_params)]), 2).item())
            optimizer.zero_grad()

        step += 1

        if telemetry is not None:
            telemetry.log_step(
                step=step,
                act_loss=act_loss_val,
                pred_loss=pred_loss_val,
                surprise=surprise_val,
            )

        # Periodic logging and checkpoint persistence
        if step % save_every == 0 or step == total_steps:
            elapsed = time.time() - t0
            steps_done = step - start_step
            rate = steps_done / max(elapsed, 1e-5)
            eta_s = (total_steps - step) / max(rate, 1e-5)
            eta_m, eta_sec = int(eta_s // 60), int(eta_s % 60)

            print(
                f"  [{model_type.upper()}] Step {step:4d}/{total_steps:4d} | "
                f"Act Loss: {act_loss_val:.4f} | "
                f"Pred Loss: {pred_loss_val:.4f} | "
                f"Surprise Mean: {surprise_val:.3f} | Var: {surprise_var:.3f} | "
                f"Gate: {gate_val:.3f} | Grad: {grad_norm_val:.3f} | "
                f"Elapsed: {elapsed:.1f}s | "
                f"ETA: {eta_m}m{eta_sec:02d}s"
            )
            history.append({
                "step": step,
                "act_loss": float(act_loss_val),
                "pred_loss": float(pred_loss_val),
                "surprise_mean": float(surprise_val),
                "surprise_var": float(surprise_var),
                "surprise_max": float(surprise_max),
                "gate_mean": float(gate_val),
                "grad_norm": float(grad_norm_val),
                "update_norm": float(update_norm_val),
                "elapsed_s": elapsed,
            })

            # Save latest checkpoint for resumption
            raw_model = getattr(model, "_orig_mod", model)
            checkpoint_state = {
                "model_type": model_type,
                "seed": seed,
                "step": step,
                "total_steps": total_steps,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "git_commit": get_git_commit(),
                "timestamp": time.time(),
                "device": str(device),
                "lr": lr,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": effective_batch_size,
            }
            torch.save(checkpoint_state, latest_file)
            if step == total_steps:
                torch.save(checkpoint_state, ckpt_file)
                print(f"  [{model_type.upper()}] Final checkpoint saved to {ckpt_file}")

    summary = {
        "history": history,
        "mean_grad_norm": float(np.mean([h["grad_norm"] for h in history if h.get("grad_norm", 0.0) > 0.0])) if history else 0.0,
        "mean_update_norm": float(np.mean([h["update_norm"] for h in history if h.get("update_norm", 0.0) > 0.0])) if history else 0.0,
        "mean_surprise_var": float(np.mean([h["surprise_var"] for h in history])) if history else 0.0,
        "mean_surprise": float(np.mean([h["surprise_mean"] for h in history])) if history else 0.0,
        "mean_gate": float(np.mean([h["gate_mean"] for h in history])) if history else 0.0,
        "final_loss": float(history[-1]["act_loss"] + pred_weight * history[-1]["pred_loss"]) if history else 0.0,
    }
    return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["reactive", "gru", "thoughtlet", "plastic_thoughtlet"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--total_samples", type=int, default=None, help="Fixed total samples (normalizes steps = total_samples // batch_size)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--corpus_dir", type=str, default="brain/datasets/hidden_rule_corpus_v1")
    parser.add_argument("--output_dir", type=str, default="brain/runs/online_adaptation/checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--use_amp", action="store_true", help="Enable bfloat16 automatic mixed precision")
    parser.add_argument("--legacy_forward", action="store_true", help="Use unrolled eager step loop (original pipeline)")

    args = parser.parse_args()
    train_model(
        model_type=args.model,
        corpus_dir=args.corpus_dir,
        seed=args.seed,
        total_steps=args.steps,
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=args.output_dir,
        device_str=args.device,
        save_every=args.save_every,
        resume=args.resume,
        resume_from=args.resume_from,
        compile_model=args.compile,
        use_amp=args.use_amp,
        use_legacy_forward=args.legacy_forward,
    )

