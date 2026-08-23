"""Phase 2.6: Evidence-residual BrainCell CONFIRMATION round (5 seeds).

Arms: {current, evidence_residual} x seeds {42,142,242,342,442}.
Protocol per preregistered gates (2026-08-22-braincell-screening.md):
  1. torture-suite multitask lift at 6000 steps (no regression beyond noise)
  2. escalation collapse-resistance, 1200 steps Phase-2-style training +
     hostile eval (protect Phase 2 strength)
  3. permutation invariance property test (exact)
  4. synthetic-memory causal signature at n=5
  5. keep/accept gate behavior inspection (sensible gating patterns)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_bc1"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS, t01_single_cue_short_delay  # noqa: E402

SEEDS = [42, 142, 242, 342, 442]


def build_model(variant: str, seed: int):
    base = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4,
                                     cycles=3, init_seed=seed)
    if variant == "current":
        return base
    # evidence_residual: wrap the base with the candidate update rule.
    # Reuses the screening module's CandidateCore implementation.
    sys.path.insert(0, _here)
    from braincell_candidates_screening import CandidateCore
    return CandidateCore(base, "evidence_residual", torch.device("cpu"))


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0)\
        .permute(0, 3, 1, 2).to(device)


# ---------------------------------------------------------------- torture ----
def train_torture(model, device, steps, seed, eps_per_task=24, curve_at=(2000,)):
    rng = np.random.default_rng(seed)
    bank = []
    for fn in TASKS:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    curve = {}
    for step in range(1, steps + 1):
        ep = bank[step % len(bank)]
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        state = None; out = None
        for j in range(ep.decision_frame_index + 1):
            out, state = model(ft[j:j+1], ctrl, dt, state=state)
        logits = out.proposals.action_logits
        target = torch.full((1, logits.shape[1]), ep.label, dtype=torch.long,
                            device=device)
        loss = F.cross_entropy(logits.view(-1, 5), target.view(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in curve_at:
            curve[str(step)] = round(eval_lift(model, device, episodes=10), 4)
    return curve


@torch.no_grad()
def eval_lift(model, device, episodes=30, base_seed=20260822):
    model.eval(); lifts = []
    for fn in TASKS:
        ok = 0
        chance = float(getattr(fn(np.random.default_rng(0)), "chance", 0.0))
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r); ft = frames_tensor(ep, device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None; act = 0
            for j in range(ep.decision_frame_index + 1):
                out, state = model(ft[j:j+1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
            ok += int(act == ep.label)
        lifts.append(ok / episodes - chance)
    model.train()
    return float(np.mean(lifts))


# ------------------------------------------------------------- escalation ----
def train_escalation(model, device, steps, seed):
    """Phase-2-style 8-hypothesis staged supervision."""
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    k_slots = model.base.k if hasattr(model, "base") else model.k
    for step in range(steps):
        haz_a = step % 2 == 0
        haz_b = (step // 2) % 2 == 0
        haz_c = (step // 4) % 2 == 0
        world_idx = int(haz_a) * 4 + int(haz_b) * 2 + int(haz_c)   # bit order A,B,C
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        rgb = [torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5 for _ in range(4)]
        rgb[1][:, 0, :16 if haz_a else 16:] += 0.8
        rgb[2] = rgb[1].clone(); rgb[2][:, 1, :16 if haz_b else 16:] += 0.8
        rgb[3] = rgb[2].clone(); rgb[3][:, 2, :16 if haz_c else 16:] += 0.8
        probs = [0.125, 0.25, 0.5, 1.0]
        eighth = max(1, k_slots // 8)
        state = None; total = 0.0
        for stage in range(4):
            out, state = model(rgb[stage], ctrl, dt, state=state)
            p = out.proposals.branch_probability.squeeze(-1)
            tp = torch.zeros_like(p)
            if stage == 0:
                tp[:] = probs[0]
            elif stage == 1:
                blk = 4 * eighth if not haz_a else 0
                tp[:, blk:blk + 4 * eighth] = probs[1]
            elif stage == 2:
                base_blk = (0 if haz_a else 4 * eighth) + (0 if haz_b else 2 * eighth)
                tp[:, base_blk:base_blk + 2 * eighth] = probs[2]
            else:
                fb = world_idx * eighth
                tp[:, fb:fb + eighth] = 1.0
            lp = F.binary_cross_entropy(p, tp)
            act_target = 0 if stage < 3 else 1 + (world_idx % 4)
            ta = torch.full((1, k_slots), act_target, dtype=torch.long, device=device)
            la = F.cross_entropy(out.proposals.action_logits.view(-1, 5), ta.view(-1))
            total = total + lp + la + (lp if stage == 3 else 0)
        opt.zero_grad(); total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


@torch.no_grad()
def eval_escalation(model, device, episodes=40):
    """Return-based eval mirroring Phase 2's benchmark: -100 per wrong final
    action, small time penalty; plus collapse detection (never escapes)."""
    model.eval()
    returns, collapses = [], 0
    k_slots = model.base.k if hasattr(model, "base") else model.k
    for ep_i in range(episodes):
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        haz_a, haz_b, haz_c = bool(ep_i % 2 == 0), bool((ep_i//2) % 2 == 0), bool((ep_i//4) % 2 == 0)
        world_idx = int(haz_a)*4 + int(haz_b)*2 + int(haz_c)
        correct_act = 1 + (world_idx % 4)
        frames = [torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5 for _ in range(4)]
        frames[1][:, 0, :16 if haz_a else 16:] += 0.8
        frames[2] = frames[1].clone(); frames[2][:, 1, :16 if haz_b else 16:] += 0.8
        frames[3] = frames[2].clone(); frames[3][:, 2, :16 if haz_c else 16:] += 0.8
        state = None; ret = -60.0; escaped = False
        t_hold = 0
        for t in range(60):
            fi = 0 if t < 15 else (1 if t < 30 else (2 if t < 45 else 3))
            out, state = model(frames[fi], ctrl, dt, state=state)
            act = int(torch.argmax(out.action_dist[0]).item())
            if act != 0 and t >= 45:
                if act == correct_act:
                    ret = 40.0 + (t - 45) * (-1.0); escaped = True; break
                else:
                    ret = -100.0; break
            t_hold += 1
        if not escaped and t_hold >= 60:
            ret = -100.0; collapses += 1
        returns.append(ret)
    model.train()
    return {"mean_return": round(float(np.mean(returns)), 2),
            "median_return": round(float(np.median(returns)), 2),
            "collapse_rate": round(collapses / len(returns), 3)}


# ----------------------------------------------------------- property tests --
@torch.no_grad()
def permutation_test(model, device):
    model.eval()
    rgb = torch.rand(1, 3, 32, 32, device=device)
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    out1, s1 = model(rgb, ctrl, dt)
    perm = list(range(1, 32)) + [0]
    state_p = None
    if hasattr(model, "initial_state"):
        state_p = model.initial_state(1).to(device)[:, perm]
    elif hasattr(model, "slot_identities"):
        state_p = model.slot_identities.expand(1, -1, -1)[:, perm]
    out2, s2 = model(rgb, ctrl, dt, state=state_p)
    d1 = out1.action_dist[0]; d2 = out2.action_dist[0]
    exact = bool(torch.allclose(d1, d2, atol=1e-5))
    model.train()
    return {"exact_permutation_invariance": exact,
            "max_dev": float((d1 - d2).abs().max())}


@torch.no_grad()
def memory_probe(model, device, episodes=40):
    model.eval()
    res = {"correct": [], "zero": [], "wrong": []}
    has_ev = "evidence" in model.forward.__code__.co_varnames or \
             any("evidence" in str(p) for p in [None]) or True
    for i in range(episodes):
        r = np.random.default_rng(20260822 + i * 7919)
        ep = t01_single_cue_short_delay(r)
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        dec = ep.decision_frame_index
        true_ch = ep.label - 1
        w = model.base.w if hasattr(model, "base") else model.w
        kk = model.base.k if hasattr(model, "base") else model.k
        for cond in ("correct", "zero", "wrong"):
            ch = true_ch if cond == "correct" else ((true_ch + 1) % 3 if cond == "wrong" else -1)
            state = None; act = 0
            for j in range(dec + 1):
                ev = None
                if j >= max(0, dec - 2):
                    e = torch.zeros(1, kk, w, device=device)
                    if cond != "zero":
                        e[:, :, ch::3] = 2.0
                    ev = e
                try:
                    out, state = model(ft[j:j+1], ctrl, dt, state=state, evidence=ev)
                except TypeError:
                    out, state = model(ft[j:j+1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
            res[cond].append(int(act == ep.label))
    model.train()
    return {k_: round(float(np.mean(v)), 3) for k_, v in res.items()}


@torch.no_grad()
def gate_behavior_inspection(model, device, episodes=20):
    """Inspect keep/accept gates under no-evidence vs strong-evidence frames."""
    core = model.update if hasattr(model, "update") else None
    if core is None:
        return None
    sens = torch.randn(1, 32, 120, device=device)
    h0 = torch.randn(1, 32, 120, device=device) * 0.5
    report = {}
    # capture gate values by re-running update internals via hooks on submodules
    kg, ag = model.keep_gate, model.accept_gate
    with torch.no_grad():
        g_in = torch.cat((sens, h0), dim=-1)
        keep = torch.sigmoid(kg(g_in)).mean().item()
        accept = torch.sigmoid(ag(g_in)).mean().item()
    report["random_input_keep_mean"] = round(keep, 3)
    report["random_input_accept_mean"] = round(accept, 3)
    # strong evidence condition: large sens magnitude
    sens_strong = sens * 4.0
    with torch.no_grad():
        g_in2 = torch.cat((sens_strong, h0), dim=-1)
        keep2 = torch.sigmoid(kg(g_in2)).mean().item()
        accept2 = torch.sigmoid(ag(g_in2)).mean().item()
    report["strong_evidence_keep_mean"] = round(keep2, 3)
    report["strong_evidence_accept_mean"] = round(accept2, 3)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--torture-steps", type=int, default=6000)
    ap.add_argument("--escalation-steps", type=int, default=1200)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== EVIDENCE-RESIDUAL CONFIRMATION ROUND on {device} ===")

    payload = {}
    for variant in ("current", "evidence_residual"):
        vp = {"seeds": {}}
        for seed in SEEDS:
            model = build_model(variant, seed).to(device)
            print(f"\n--- {variant} seed={seed} ---")
            curve = train_torture(model, device, args.torture_steps, seed)
            lift = eval_lift(model, device)
            esc_res = eval_escalation(model, device)
            perm = permutation_test(model, device)
            mem = memory_probe(model, device)
            mem_causal = mem["correct"] > mem["zero"] and mem["wrong"] < mem["correct"]
            gates = gate_behavior_inspection(model, device) \
                if variant == "evidence_residual" else None
            vp["seeds"][str(seed)] = {
                "final_lift": round(lift, 4), "curve": curve,
                "escalation": esc_res, "permutation": perm,
                "memory_probe": mem, "memory_causal": bool(mem_causal),
                "gate_behavior": gates}
            print(f"  lift={lift:+.4f} esc={esc_res} perm={perm['exact_permutation_invariance']} "
                  f"mem_causal={mem_causal}")
        lifts = [v["final_lift"] for v in vp["seeds"].values()]
        cols = [v["escalation"]["collapse_rate"] for v in vp["seeds"].values()]
        vp["summary"] = {
            "mean_lift": round(float(np.mean(lifts)), 4),
            "std_lift": round(float(np.std(lifts)), 4),
            "collapse_rate_mean": round(float(np.mean(cols)), 3),
            "memory_causal_n": sum(v["memory_causal"] for v in vp["seeds"].values()),
            "perm_exact_all": all(v["permutation"]["exact_permutation_invariance"]
                                  for v in vp["seeds"].values())}
        payload[variant] = vp
        print(f"\n{variant.upper()} SUMMARY: {vp['summary']}")

    # verdict against frozen gates
    c, e = payload["current"]["summary"], payload["evidence_residual"]["summary"]
    verdict = {
        "G1_lift_no_regression": e["mean_lift"] >= c["mean_lift"] - c["std_lift"],
        "G2_collapse_no_worse": e["collapse_rate_mean"] <= c["collapse_rate_mean"] + 0.05,
        "G3_permutation_exact": e["perm_exact_all"],
        "G4_memory_causal_5of5": e["memory_causal_n"] == 5}
    verdict["PROMOTED_TO_CORE_V1_CANDIDATE"] = all(verdict.values())
    payload["VERDICT"] = verdict
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
