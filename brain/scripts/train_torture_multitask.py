"""Phase 2.6 torture-suite baseline part 2: multi-task training + evaluation.

Trains a model on a mixture of the 25 torture tasks (supervised decision
targets), then evaluates per-task accuracy. This is the canonical baseline:
Core changes must beat THESE numbers, not the fresh-init zeros.

Identical recipe for pb_k32 and gru: 4000 steps, batch of 8 episodes drawn
round-robin from all tasks, cross-entropy on the decision-frame action.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_mt"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402
sys.path.insert(0, _here)
from run_torture_suite import build_model, score_model  # noqa: E402


def episode_tensor(ep, device):
    frames = torch.from_numpy(
        np.stack(ep.frames).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    return frames


@torch.no_grad()
def rollout(model, frames, device):
    """Run frames, return action logits AT the decision frame and final state."""
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    state = None
    logits = None
    for i in range(frames.shape[0]):
        out, state = model(frames[i:i+1], ctrl, dt, state=state)
        logits = out.action_logits
    return logits


def train_multitask(model, device, steps: int, lr: float, seed: int,
                    tasks, episodes_per_task: int = 24):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    # Pre-generate episode bank (CPU) for determinism.
    bank = []
    for task_fn in tasks:
        for i in range(episodes_per_task):
            rng = np.random.default_rng(7000 + hash(task_fn.__name__) % 100000 + i)
            ep = task_fn(rng)
            bank.append((task_fn.__name__, episode_tensor(ep, "cpu"), ep.label,
                         ep.decision_frame_index))
    print(f"episode bank: {len(bank)} episodes across {len(tasks)} tasks")

    for step in range(steps):
        name, frames, label, dec_idx = bank[step % len(bank)]
        frames = frames.to(device)
        state = None
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        for i in range(dec_idx + 1):
            out, state = model(frames[i:i+1], ctrl, dt, state=state)
        logits = out.proposals.action_logits.mean(dim=1)  # [1,5] slot-averaged
        target = torch.tensor([label], device=device)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{steps} loss={loss.item():.4f}")
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pb_k32", "gru"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--save-checkpoint", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.seed, device, None)
    print(f"=== MULTITASK TRAINING {args.model} steps={args.steps} on {device} ===")
    train_multitask(model, device, args.steps, args.lr, args.seed, TASKS)

    print("=== POST-TRAINING EVALUATION ===")
    results = score_model(model, device, args.episodes, 20260822)
    total_lift = float(np.mean([v["lift"] for v in results.values()]))
    n_above = sum(1 for v in results.values() if v["lift"] > 0.02)
    payload = {"model": args.model, "steps": args.steps, "seed": args.seed,
               "episodes_per_task": args.episodes,
               "per_task": results,
               "mean_lift_over_chance": round(total_lift, 4),
               "tasks_above_chance_2pct": n_above}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    for name, v in results.items():
        flag = " +" if v["lift"] > 0.02 else "  "
        print(f"{name:28s} {v['accuracy']:7.3f} {v['chance']:7.3f} {v['lift']:+7.3f}{flag}")
    print(f"\nMEAN LIFT: {total_lift:+.4f}  ({n_above} tasks >+2%)")
    if args.save_checkpoint:
        torch.save({"state_dict": model.state_dict(), "meta": {"model": args.model}},
                   args.save_checkpoint)
        print("checkpoint ->", args.save_checkpoint)


if __name__ == "__main__":
    main()
