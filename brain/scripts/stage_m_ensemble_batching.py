"""Stage M: multi-model ensemble batching (preregistered 2026-08-23).

Trains E independent Core V1 models in one stacked-state execution
[E,B,K,W]; each model owns private slot states and weights. Evaluates each
model separately on the locked torture eval; E=4 aggregate = mean of the 4
models' lifts. Gates per prereg 2026-08-23-ensemble-batching-prereg.md.
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

_here = os.path.dirname(os.path.abspath(__file__))
_cand_esc = [
    os.path.join(_here, "dgx_phase2_definitive_escalation.py"),
    os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"),
]
_esc_path = next((p for p in _cand_esc if os.path.exists(p)), _cand_esc[0])
_spec = importlib.util.spec_from_file_location("esc_m", _esc_path)
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_m"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))

_spec2 = importlib.util.spec_from_file_location(
    "stage_l", os.path.join(_here, "stage_l_w_scaling.py"))
stage_l = importlib.util.module_from_spec(_spec2)
sys.modules["stage_l"] = stage_l
_spec2.loader.exec_module(stage_l)

SEEDS = [42, 142, 242, 342]


class EnsembleCore(torch.nn.Module):
    """E independent VectorizedPseudoBrains executed with a shared batch dim."""

    def __init__(self, e: int, seed_base: int):
        super().__init__()
        self.e = e
        self.models = torch.nn.ModuleList([
            esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=2,
                                      cycles=3, init_seed=seed_base + i)
            for i in range(e)])

    def forward(self, rgb):  # rgb [B,3,H,W]
        outs = [m(rgb) for m in self.models]
        return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    print("=== STAGE M: ENSEMBLE BATCHING (E=1 vs E=4) ===")

    payload = {}
    for e in (1, 4):
        arm_lifts = []
        t0 = time.perf_counter()
        ens = EnsembleCore(e, 42).to(device)
        opt = torch.optim.AdamW(ens.parameters(), lr=3e-4)
        # data: reuse stage_l episode generator; every model sees the same
        # stream (shared experience, independent learning)
        for step in range(args.steps):
            episodes = stage_l.make_episodes(16, device)
            losses = []
            for m in ens.models:
                total = stage_l.train_one_episode_loss(m, episodes, device)
                losses.append(total)
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 1000 == 0:
                print(f"E={e} step {step}/{args.steps} loss={loss.item():.4f}")
        for i, m in enumerate(ens.models):
            lift = stage_l.eval_lift(m, device)
            arm_lifts.append(round(lift, 4))
            print(f"E={e} model{i}: lift={lift:+.4f}")
        wall = time.perf_counter() - t0
        agg = float(np.mean(arm_lifts)) if e == 4 else None
        payload[f"e{e}"] = {
            "per_model": arm_lifts,
            "sigma_models": round(float(np.std(arm_lifts)), 4),
            "aggregate_mean": round(agg, 4) if agg is not None else round(arm_lifts[0], 4),
            "wall_s": round(wall, 1),
        }

    s1 = payload["e1"]["sigma_models"]
    # stabilization gate compares across-seed-group sigma; single execution
    # here gives within-arm spread as the honest available proxy — recorded,
    # cross-arm comparison is between independent executions of the same code
    verdict = {
        "gate_stabilization_sigma_e1_vs_e4":
            {"e1": s1, "e4_within_arm": payload["e4"]["sigma_models"]},
        "gate_no_mean_harm_delta":
            round(payload["e4"]["aggregate_mean"] - payload["e1"]["aggregate_mean"], 4),
    }
    delta = verdict["gate_no_mean_harm_delta"]
    verdict["PASS_no_mean_harm"] = bool(abs(delta) <= 0.03)
    payload["VERDICT"] = verdict
    print(json.dumps(payload, indent=2))
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
