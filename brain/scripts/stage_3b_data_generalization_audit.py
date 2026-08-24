"""Stage 3b: Data / Generalization Audit (preregistered).

Frozen Core V1 W=120 K=32 C=3. FIXED lr=5e-4. 6k steps. 4 seeds.
ARM A: finite pinned bank (repeating). ARM B: non-repeating deterministic stream.
Compare: training loss collapse, held-out torture lift, train/eval gap.
"""
from __future__ import annotations

import argparse
import json
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
TOTAL_STEPS = 6000


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


@torch.no_grad()
def eval_full(model, device, episodes=30, base_seed=20260822):
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


def run_arm_a(schedule, seed, device, bdig):
    """ARM A: finite pinned bank (repeating)"""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)
    model = esc.VectorizedPseudoBrain(
        thoughtlets=32, width=120, heads=4, cycles=3, init_seed=seed).to(device)
    init_dig = param_digest(model)

    bank = build_bank_pinned(42, TASKS)  # fixed bank seed
    assert bank_digest(bank) == bdig, "bank digest mismatch"
    order = build_order(bank)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()

    rec = {"arm": "A_finite_bank", "schedule": schedule, "seed": seed,
           "provenance": provenance(model=model, train_seed=seed,
                                    eval_seed=20260822, bank_digest=bdig,
                                    deterministic=True)}
    rec["provenance"]["init_param_digest"] = init_dig

    loss_curve = []
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                loss_curve.append({"step": step, "loss": round(float(loss.item()), 5)})
            step += 1
    wall = time.perf_counter() - t0
    res = eval_full(model, device)
    rec["final_eval"] = res
    rec["final_loss"] = loss_curve[-1]["loss"] if loss_curve else None
    rec["loss_curve"] = loss_curve
    rec["wall_s"] = round(wall, 1)
    return rec


def run_arm_b(schedule, seed, device):
    """ARM B: non-repeating deterministic stream"""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)
    model = esc.VectorizedPseudoBrain(
        thoughtlets=32, width=120, heads=4, cycles=3, init_seed=seed).to(device)
    init_dig = param_digest(model)

    # Non-repeating stream: fresh episodes every step
    # Seeded deterministically from (seed, step)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()

    rec = {"arm": "B_nonrepeating_stream", "schedule": schedule, "seed": seed,
           "provenance": provenance(model=model, train_seed=seed,
                                    eval_seed=20260822, bank_digest=None,
                                    deterministic=True)}
    rec["provenance"]["init_param_digest"] = init_dig
    rec["provenance"]["stream_seeding"] = "base_seed + step"

    loss_curve = []
    step = 0
    t0 = time.perf_counter()
    while step < TOTAL_STEPS:
        # Generate fresh batch for this step (deterministic from seed+step)
        rng = np.random.default_rng(seed + step * 10007 + 12345)
        eps = [fn(rng) for fn in TASKS]  # one episode per task
        # Group by length
        by_len = {}
        for e in eps:
            by_len.setdefault(e.decision_frame_index + 1, []).append(e)
        # Process each length group
        for ln, group_eps in by_len.items():
            B = len(group_eps)
            ftb = torch.stack([
                torch.stack([frames_tensor(e, device)[j] for j in range(ln)])
                for e in group_eps])
            ctrl = torch.zeros(B, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], ctrl, dt, state=state)
            logits = out_j.proposals.action_logits
            tgt = torch.tensor([e.label for e in group_eps], dtype=torch.long,
                               device=device).unsqueeze(1).expand(-1, logits.shape[1])
            loss = F.cross_entropy(logits.reshape(-1, 5), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % 100 == 0:
            loss_curve.append({"step": step, "loss": round(float(loss.item()), 5)})
        step += 1
    wall = time.perf_counter() - t0
    res = eval_full(model, device)
    rec["final_eval"] = res
    rec["final_loss"] = loss_curve[-1]["loss"] if loss_curve else None
    rec["loss_curve"] = loss_curve
    rec["wall_s"] = round(wall, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    apply_deterministic_mode()
    device = torch.device("cuda:0")
    print("=== STAGE 3b: DATA / GENERALIZATION AUDIT ===")

    ref_bank = build_bank_pinned(42, TASKS)
    bdig = bank_digest(ref_bank)
    print("bank digest (ARM A):", bdig)

    payload = {"bank_digest_arm_a": bdig, "results": {}}
    for arm_name, arm_fn in [("A_finite_bank", run_arm_a), ("B_nonrepeating_stream", run_arm_b)]:
        runs = []
        for seed in SEEDS:
            if arm_name == "A_finite_bank":
                r = arm_fn("FIXED", seed, device, bdig)
            else:
                r = arm_fn("FIXED", seed, device)
            runs.append(r)
            print(f"[{arm_name} s{seed}] loss={r['final_loss']:.4f} lift={r['final_eval']['mean_lift']:+.4f} above={r['final_eval']['above_2pct']}")
        payload["results"][arm_name] = runs

        # Aggregate
        lifts = [r["final_eval"]["mean_lift"] for r in runs]
        losses = [r["final_loss"] for r in runs]
        aboves = [r["final_eval"]["above_2pct"] for r in runs]
        print(f"  {arm_name} mean: loss={np.mean(losses):.4f} lift={np.mean(lifts):+.4f} above={sum(aboves)}")

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()