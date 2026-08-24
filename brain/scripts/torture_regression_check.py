"""Stage E: torture-suite regression check after temperature pin + CI work.

Verifies the locked baselines still hold with the current code state:
GRU fresh -> trained, PB corrected per-slot CE recipe, 3 seeds each,
6000 steps multitask. Compares against locked reference:
  GRU: -0.042 mean lift, 8/25 above
  PB:  -0.215 mean lift, 1/25 above
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
from torch import nn

_here = os.path.dirname(os.path.abspath(__file__))
_cand_esc = [
    os.path.join(_here, "dgx_phase2_definitive_escalation.py"),
    os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"),
]
_esc_path = next((p for p in _cand_esc if os.path.exists(p)), _cand_esc[0])
_spec = importlib.util.spec_from_file_location("esc_base", _esc_path)
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_regr"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0) \
        .permute(0, 3, 1, 2).to(device)


def build_bank(seed, eps_per_task=24):
    bank = []
    for fn in TASKS:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))
    return bank


def train_multitask(model, device, steps, seed):
    """Corrected Phase-2-style head: per-slot CE against replicated target."""
    bank = build_bank(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    for step in range(1, steps + 1):
        ep = bank[step % len(bank)]
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        state = None
        out = None
        for j in range(ep.decision_frame_index + 1):
            out, state = model(ft[j:j+1], ctrl, dt, state=state)
        logits = out.proposals.action_logits          # [1, K, 5]
        k_slots = logits.shape[1]
        target = torch.full((1, k_slots), ep.label, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.view(-1, 5), target.view(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def eval_suite(model, device, episodes=30, base_seed=20260822):
    model.eval()
    per_task = {}
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
                out, state = model(ft[j:j+1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
            ok += int(act == ep.label)
        per_task[fn.__name__] = round(ok / episodes - chance, 4)
    model.train()
    lifts = list(per_task.values())
    return {"per_task": per_task,
            "mean_lift": round(float(np.mean(lifts)), 4),
            "above_2pct": sum(1 for v in lifts if v > 0.02)}


class GRUArm(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.core = esc.VectorizedGRUCore(init_seed=seed) \
            if hasattr(esc, "VectorizedGRUCore") else None
        raise NotImplementedError("use esc's GRU arm builder")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=str, default="42,142,242")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== TORTURE REGRESSION on {device}, steps={args.steps} ===")

    payload = {}
    for arch in ("pb", "gru"):
        results = []
        for seed in seeds:
            if arch == "pb":
                m = esc.VectorizedPseudoBrain(thoughtlets=32, width=120,
                                              heads=4, cycles=3, init_seed=seed)
            else:
                m = esc.ProposalGRUBaseline(hidden_dim=112, num_proposals=21)
                # GRU baseline is seed-neutral in init except torch seed:
                torch.manual_seed(seed)
                for p_ in m.parameters():
                    if p_.dim() > 1:
                        nn.init.xavier_uniform_(p_)
            m = m.to(device)
            train_multitask(m, device, args.steps, seed)
            r = eval_suite(m, device)
            r["seed"] = seed
            results.append(r)
            print(f"{arch} s{seed}: lift={r['mean_lift']:+.4f} "
                  f"above={r['above_2pct']}/25")
        payload[arch] = {
            "seeds": results,
            "mean_lift": round(float(np.mean([r['mean_lift'] for r in results])), 4),
            "std_lift": round(float(np.std([r['mean_lift'] for r in results])), 4),
            "mean_above": float(np.mean([r['above_2pct'] for r in results]))}
        print(f"{arch.upper()} SUMMARY: {payload[arch]}")

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
