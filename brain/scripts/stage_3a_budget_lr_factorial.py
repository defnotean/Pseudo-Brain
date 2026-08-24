"""Stage 3a: training-budget x LR-schedule factorial (preregistered).

Frozen Core V1 W=120 K=32 C=3. One 24k-step run per (schedule, seed);
evaluated at 6k/12k/24k checkpoints. Arms: FIXED lr=5e-4 vs SCHEDULED
(warmup 500 -> cosine to 5e-5). Deterministic mode, pinned banks,
full provenance per run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "src"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "esc_3a", os.path.join(_here, "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_3a"] = esc
_spec.loader.exec_module(esc)

from run_provenance import (build_bank_pinned, bank_digest, param_digest,
                            apply_deterministic_mode, provenance)
from irene_brain.evaluation.torture_suite import TASKS

SEEDS = [42, 142, 242, 342]
TOTAL_STEPS = 24000
CHECKPOINTS = [6000, 12000, 24000]


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0) \
        .permute(0, 3, 1, 2).to(device)


def build_order(bank, B=16):
    by_len = {}
    for idx, ep in enumerate(bank):
        by_len.setdefault(ep.decision_frame_index + 1, []).append(idx)
    order = []
    for ln in sorted(by_len):
        members = by_len[ln]
        for k in range(0, len(members), B):
            group = members[k:k + B]
            while len(group) < B:
                group.append(members[(k + len(group)) % len(members)])
            order.append((ln, group))
    return order


def lr_at(schedule, step, base_lr=5e-4, warmup=500, total=TOTAL_STEPS,
          floor=5e-5):
    if schedule == "FIXED":
        return base_lr
    if step < warmup:
        return base_lr * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + (base_lr - floor) * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def eval_full(model, device, episodes=30, base_seed=20260822):
    """Returns mean lift, above-chance+2% count, per-faculty lifts."""
    model.eval()
    lifts = []
    for fn in TASKS:
        ok = 0
        chance = float(getattr(fn(np.random.default_rng(0)), "chance", 0.0))
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            act = 0
            for j in range(ep.decision_frame_index + 1):
                out, state = model(ft[j:j + 1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
            ok += int(act == ep.label)
        lifts.append(ok / episodes - chance)
    model.train()
    lifts_a = np.array(lifts)
    return {
        "mean_lift": round(float(lifts_a.mean()), 4),
        "above_2pct": int((lifts_a > 0.02).sum()),
        "per_faculty": [round(float(x), 4) for x in lifts],
    }


def run_cell(schedule, seed, device, bdig):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)
    model = esc.VectorizedPseudoBrain(
        thoughtlets=32, width=120, heads=4, cycles=3, init_seed=seed).to(device)
    init_dig = param_digest(model)

    # Use FIXED bank seed (42) for ALL runs - pinned banks are the whole point
    bank = build_bank_pinned(42, TASKS)
    assert bank_digest(bank) == bdig, "bank digest mismatch"
    order = build_order(bank)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()

    rec = {"schedule": schedule, "seed": seed,
           "provenance": provenance(model=model, train_seed=seed,
                                    eval_seed=20260822, bank_digest=bdig,
                                    deterministic=True)}
    rec["provenance"]["init_param_digest"] = init_dig
    rec["provenance"]["bank_policy"] = "pinned_constant_across_seeds_seed42"
    rec["provenance"]["bank_seed"] = 42

    checkpoints = {}
    loss_curve = []
    grad_norms = {}
    step = 0
    t0 = time.perf_counter()
    while step < TOTAL_STEPS:
        for ln, group in order:
            if step >= TOTAL_STEPS:
                break
            eps = [bank[i] for i in group]
            ftb = torch.stack([
                torch.stack([frames_tensor(e, device)[j] for j in range(ln)])
                for e in eps])
            ctrl = torch.zeros(len(eps), 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], ctrl, dt, state=state)
            logits = out_j.proposals.action_logits
            tgt = torch.tensor([e.label for e in eps], dtype=torch.long,
                               device=device).unsqueeze(1).expand(-1, logits.shape[1])
            loss = F.cross_entropy(logits.reshape(-1, 5), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            if step in (100, 1000, 6000, 12000, 20000):
                gsq = sum(float(p.grad.to(torch.float64).pow(2).sum())
                          for p in model.parameters() if p.grad is not None)
                grad_norms[str(step)] = round(gsq ** 0.5, 4)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_at(schedule, step)
            opt.step()
            if step % 500 == 0:
                loss_curve.append({"step": step, "loss": round(float(loss.item()), 5)})
            step += 1
            if step in CHECKPOINTS:
                res = eval_full(model, device)
                checkpoints[str(step)] = res
                print(f"[{schedule} s{seed}] ckpt {step}: lift={res['mean_lift']} "
                      f"above={res['above_2pct']}")
    wall = time.perf_counter() - t0
    rec["checkpoints"] = checkpoints
    rec["loss_curve"] = loss_curve
    rec["grad_norms"] = grad_norms
    rec["wall_s"] = round(wall, 1)
    # AUC over checkpoint lifts (trapezoid, steps as x)
    xs = [float(c) for c in CHECKPOINTS]
    ys = [checkpoints[str(c)]["mean_lift"] for c in CHECKPOINTS]
    rec["lift_auc"] = round(float(np.trapz(ys, xs) / (xs[-1] - xs[0])), 5)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    apply_deterministic_mode()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    print("=== STAGE 3a: BUDGET x SCHEDULE FACTORIAL ===")
    print(f"Device: {device} | Seeds: {seeds}")

    ref_bank = build_bank_pinned(42, TASKS)
    bdig = bank_digest(ref_bank)
    print("bank digest:", bdig)

    payload = {"bank_digest": bdig, "cells": {}}
    for schedule in ("FIXED", "SCHEDULED"):
        runs = []
        for seed in seeds:
            r = run_cell(schedule, seed, device, bdig)
            runs.append(r)
        payload["cells"][schedule] = runs
        for ck in CHECKPOINTS:
            lifts = [r["checkpoints"][str(ck)]["mean_lift"] for r in runs if str(ck) in r["checkpoints"]]
            aboves = [r["checkpoints"][str(ck)]["above_2pct"] for r in runs if str(ck) in r["checkpoints"]]
            if lifts:
                print(f"{schedule} @{ck}: lift={np.mean(lifts):+.4f} "
                      f"±{np.std(lifts):.4f} above={sum(aboves)}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
