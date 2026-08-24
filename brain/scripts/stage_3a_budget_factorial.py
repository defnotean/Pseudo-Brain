"""Stage 3a: training-budget x LR-schedule factorial (preregistered).

Core V1 frozen (W=120 K=32 C=3). Cells: {FIXED, COSINE} x {6k, 12k, 24k}.
FIXED: one 24k trajectory per seed, evaluated at 6k/12k/24k checkpoints
(prefix property). COSINE: horizon is part of the cell -> separate run per
budget. Deterministic mode; pinned banks; full provenance per prereg
2026-08-23-stage3a-budget-factorial-prereg.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import zlib

_here = os.path.dirname(os.path.abspath(__file__))
_cand_esc = [
    os.path.join(_here, "dgx_phase2_definitive_escalation.py"),
    os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"),
]
_esc_path = next((p for p in _cand_esc if os.path.exists(p)), _cand_esc[0])
_spec = importlib.util.spec_from_file_location("esc_3a", _esc_path)
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_3a"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402

sys.path.insert(0, _here)
from run_provenance import provenance, apply_deterministic_mode  # noqa: E402

SEEDS = [42, 142, 242]
BANK_SEED = 42


def task_offset(name: str) -> int:
    return zlib.crc32(name.encode()) % 99991


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0) \
        .permute(0, 3, 1, 2).to(device)


def build_bank(seed, eps_per_task=24):
    bank = []
    for fn in TASKS:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + task_offset(fn.__name__) + i)
            bank.append(fn(r))
    return bank


def bank_digest(bank) -> str:
    h = hashlib.sha256()
    for ep in bank:
        h.update(str(ep.label).encode())
        h.update(str(ep.decision_frame_index).encode())
        h.update(np.asarray(np.stack(ep.frames), dtype=np.uint8).tobytes())
    return h.hexdigest()[:16]


# ---- locked eval (identical to prior stages) ----

@torch.no_grad()
def eval_lift(model, device, episodes=30, base_seed=20260822):
    model.eval()
    lifts, acts = [], []
    for fn in TASKS:
        ok = 0
        chance = float(getattr(fn(np.random.default_rng(0)), "chance", 0.0))
        counts = [0] * 5
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
            counts[act] += 1
            ok += int(act == ep.label)
        acts.append(counts)
        lifts.append(ok / episodes - chance)
    model.train()
    return float(np.mean(lifts)), lifts, acts


@torch.no_grad()
def heldout_ce(model, device, episodes=10, base_seed=20260822 + 77777):
    """Held-out eval CE loss on unseen episodes (never trained on)."""
    model.eval()
    ces = []
    for fn in TASKS:
        for i in range(episodes):
            r = np.random.default_rng(base_seed + hash_free(fn.__name__) + i)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            logits = None
            for j in range(ep.decision_frame_index + 1):
                out, state = model(ft[j:j + 1], ctrl, dt, state=state)
                logits = out.proposals.action_logits
            tgt = torch.tensor([ep.label], device=device)
            k_slot = int(torch.argmax(
                out.proposals.branch_probability[0].sum(dim=1)).item())
            slot_logits = logits[0, k_slot]
            ces.append(float(F.cross_entropy(
                slot_logits.unsqueeze(0), tgt).item()))
    model.train()
    return round(float(np.mean(ces)), 4)


def hash_free(name: str) -> int:
    return zlib.crc32(name.encode()) % 99991


def make_order(bank, B=16):
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


PROBE_STEPS = {3000, 6000, 9000, 12000, 18000, 24000}
CHECKPOINTS = [6000, 12000, 24000]


def train_fixed_with_checkpoints(model, device, steps_total, seed,
                                 train_bank_digest):
    """Fixed-LR single trajectory; evaluate at each checkpoint."""
    rec = {"provenance": provenance(model=model, train_seed=seed,
                                    eval_seed=20260822,
                                    bank_digest=train_bank_digest,
                                    deterministic=True)}
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = None
    return _run_trajectory(model, opt, sched, device, seed, steps_total, rec)


def train_cosine(model, device, steps_horizon, seed, train_bank_digest):
    rec = {"provenance": provenance(model=model, train_seed=seed,
                                    eval_seed=20260822,
                                    bank_digest=train_bank_digest,
                                    deterministic=True)}
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_horizon)
    return _run_trajectory(model, opt, sched, device, seed, steps_horizon, rec)


def _run_trajectory(model, opt, sched, device, seed, steps_total, rec):
    bank = build_bank(BANK_SEED)
    order = make_order(bank)
    model.train()
    losses, grad_norms = {}, {}
    step = 0
    ckpts = {}
    t0 = time.perf_counter()
    while step < steps_total:
        for ln, group in order:
            if step >= steps_total:
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
            gsq = sum(float(p.grad.to(torch.float64).pow(2).sum())
                      for p in model.parameters() if p.grad is not None)
            grad_norms[step] = round(gsq ** 0.5, 4)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            if step in PROBE_STEPS or step == 0:
                losses[step] = round(float(loss.item()), 6)
            if (step + 1) in CHECKPOINTS and (step + 1) <= steps_total:
                lift, per_task, acts = eval_lift(model, device)
                ho = heldout_ce(model, device)
                ckpts[step + 1] = {
                    "lift": round(lift, 4),
                    "above_2pct": int(sum(1 for x in per_task if x > 0.02)),
                    "per_task": [round(x, 4) for x in per_task],
                    "heldout_ce": ho,
                    "action_dist_mean": [
                        round(c / sum(a), 3) for a in acts for c in a][:5],
                    "wall_s": round(time.perf_counter() - t0, 1),
                }
                print(f"  s{seed} @{step+1}: lift={lift:+.4f} "
                      f"above={ckpts[step+1]['above_2pct']} hoCE={ho}")
            step += 1
    rec["losses"] = losses
    rec["grad_norms_sampled"] = {str(k): v for k, v in
                                 list(grad_norms.items())[::500]}
    rec["checkpoints"] = ckpts
    auc_steps = sorted(int(k) for k in ckpts)
    if auc_steps:
        xs = np.array([s / steps_total for s in auc_steps])
        ys = np.array([ckpts[s]["lift"] for s in auc_steps])
        rec["lift_curve_auc"] = round(float(np.trapz(ys, xs)) * -1.0, 4)
    rec["final_param_l2"] = round(float(sum(
        p.detach().to(torch.float64).pow(2).sum()
        for p in model.parameters())) ** 0.5, 6)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    apply_deterministic_mode()
    device = torch.device("cuda:0")
    bank = build_bank(BANK_SEED)
    bdig = bank_digest(bank)
    print("=== STAGE 3a: BUDGET x SCHEDULE FACTORIAL ===")
    print("bank digest:", bdig)
    payload = {"bank_digest": bdig}

    # FIXED arm: one 24k trajectory per seed; cells are its checkpoints
    fixed_runs = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.cuda.manual_seed_all(seed)
        model = esc.VectorizedPseudoBrain(
            thoughtlets=32, width=120, heads=4, cycles=3,
            init_seed=seed).to(device)
        print(f"[FIXED] s{seed} -> 24k")
        fixed_runs[str(seed)] = train_fixed_with_checkpoints(
            model, device, 24000, seed, bdig)
        torch.cuda.empty_cache()
    payload["FIXED"] = fixed_runs

    # COSINE arm: horizon part of the cell -> separate runs
    cosine_runs = {}
    for horizon in (6000, 12000, 24000):
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.cuda.manual_seed_all(seed)
            model = esc.VectorizedPseudoBrain(
                thoughtlets=32, width=120, heads=4, cycles=3,
                init_seed=seed).to(device)
            print(f"[COSINE h={horizon}] s{seed}")
            key = f"h{horizon}_s{seed}"
            cosine_runs[key] = train_cosine(model, device, horizon, seed, bdig)
            # keep only the declared-horizon checkpoint as the cell result
            cosine_runs[key]["cell_checkpoint"] = \
                cosine_runs[key]["checkpoints"].get(str(horizon)) or \
                cosine_runs[key]["checkpoints"].get(horizon)
            torch.cuda.empty_cache()
    payload["COSINE"] = cosine_runs

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
