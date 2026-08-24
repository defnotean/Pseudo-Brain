"""Stage V2.0: Core V2 deployed-loss baseline vs Core V1 reference (preregistered).

Prereg: brain/docs/preregistrations/2026-08-24-core-v2-deployed-loss-baseline-prereg.md
Frozen gates: alpha Δ>=+0.05 & σ<=0.06 | beta |Δ|<=0.02 | gamma Δ<=-0.03 | else AMBIGUOUS.
Arms: V2-A (CONFIG_A_DECISION_ONLY), V2-C (CONFIG_C_FULL). Deployed decision loss only.
Bank/eval protocol identical to Stage 3b (bank b3bb5fc33fd5f605, eval seed 20260822+i*7919).
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

from run_provenance import (build_bank_pinned, bank_digest, param_digest,
                            apply_deterministic_mode, provenance)
from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import (
    CoreV2Model,
    CONFIG_A_DECISION_ONLY, CONFIG_C_FULL,
)
from irene_brain.v2.losses import deployed_decision_loss

SEEDS = [42, 142, 242, 342]
TOTAL_STEPS = 6000
REFERENCE_R = -0.2130  # Core V1 FIXED @6k, same bank, n=4


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
def eval_full_v2(model, device, episodes=30, base_seed=20260822):
    """Identical protocol to Stage 3b eval_full: argmax on DEPLOYED action_dist.

    Returns mean_lift plus non-gated diagnostics (slot agreement, mass entropy).
    """
    model.eval()
    lifts = []
    slot_agree = []
    mass_ent = []
    for fn in TASKS:
        ok = 0
        chance = float(getattr(fn(np.random.default_rng(0)), "chance", 0.0))
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            state = model.init_state(1, device)
            act = 0
            for j in range(ep.decision_frame_index + 1):
                out, state = model(ft[j:j + 1], state)
                act = int(torch.argmax(out.decision.action_dist[0]).item())
            ok += int(act == ep.label)

            # diagnostics at decision frame
            dec = out.decision
            slot_argmax = dec.thought_action_probs[0].argmax(dim=-1)  # [K]
            slot_agree.append(float((slot_argmax == act).float().mean().item()))
            p = dec.mass_per_action[0].clamp_min(0)
            if p.sum() > 0:
                p = p / p.sum()
                mass_ent.append(float(-(p * p.clamp_min(1e-9).log()).sum().item()))
        lifts.append(ok / episodes - chance)
    model.train()
    lifts_a = np.array(lifts)
    return {
        "mean_lift": round(float(lifts_a.mean()), 4),
        "above_2pct": int((lifts_a > 0.02).sum()),
        "per_faculty": [round(float(x), 4) for x in lifts],
        "slot_agreement": round(float(np.mean(slot_agree)), 4) if slot_agree else None,
        "mass_entropy": round(float(np.mean(mass_ent)), 4) if mass_ent else None,
    }


def run_v2_arm(flags, arm_name, seed, device, bdig):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)
    model = CoreV2Model(flags=flags).to(device)
    init_dig = param_digest(model)

    bank = build_bank_pinned(42, TASKS)  # fixed bank seed, same as all Stage 3 runs
    assert bank_digest(bank) == bdig, "bank digest mismatch"
    order = build_order(bank)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()

    rec = {"arm": arm_name, "seed": seed,
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
            labels = torch.tensor([e.label for e in eps], dtype=torch.long,
                                  device=device)
            # Fresh trial state per sequence batch; single supervised pass to
            # the decision frame (identical exposure to V1's harness).
            state = model.init_state(len(eps), device)
            out_j = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], state)
            loss = deployed_decision_loss(out_j, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                loss_curve.append({"step": step, "loss": round(float(loss.item()), 5)})
            step += 1
    wall = time.perf_counter() - t0
    res = eval_full_v2(model, device)
    rec["final_eval"] = res
    rec["final_loss"] = loss_curve[-1]["loss"] if loss_curve else None
    rec["loss_curve"] = loss_curve
    rec["wall_s"] = round(wall, 1)
    return rec


def classify(mean_lift, sigma):
    delta = mean_lift - REFERENCE_R
    if delta >= 0.05 and sigma <= 0.06:
        return "ALPHA_contract_fix_works", round(delta, 4)
    if abs(delta) <= 0.02:
        return "BETA_parity_intake_bottleneck", round(delta, 4)
    if delta <= -0.03:
        return "GAMMA_regression_stop", round(delta, 4)
    return "AMBIGUOUS", round(delta, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override seed list (for smoke testing only)")
    args = ap.parse_args()

    apply_deterministic_mode()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bank_check = build_bank_pinned(42, TASKS)
    bdig = bank_digest(bank_check)
    assert bdig == "b3bb5fc33fd5f605", f"bank digest drift: {bdig}"

    seeds = args.seeds if args.seeds else SEEDS
    results = {"prereg": "2026-08-24-core-v2-deployed-loss-baseline",
               "reference_R": REFERENCE_R, "bank_digest": bdig,
               "seeds": seeds, "arms": {}}

    for flags, name in [(CONFIG_A_DECISION_ONLY, "V2A_decision_only"),
                        (CONFIG_C_FULL, "V2C_full")]:
        recs = []
        for s in seeds:
            print(f"[{name}] seed {s} ...", flush=True)
            recs.append(run_v2_arm(flags, name, s, device, bdig))
        lifts = [r["final_eval"]["mean_lift"] for r in recs]
        ml, sig = float(np.mean(lifts)), float(np.std(lifts))
        cls, delta = classify(ml, sig)
        results["arms"][name] = {
            "runs": recs, "mean_lift": round(ml, 4),
            "sigma": round(sig, 4), "delta_vs_V1": delta,
            "classification": cls,
        }
        print(f"[{name}] mean={ml:.4f} σ={sig:.4f} Δ={delta:+.4f} -> {cls}", flush=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE -> {args.output}")


if __name__ == "__main__":
    main()
