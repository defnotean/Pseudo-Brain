"""Stage L2: W=480 variance disambiguation — more data or unstable?

Preregistered inline (before run):
- Arms: W480 x {6000 steps (repeat), 12000 steps} x 5 seeds {42,142,242,342,442}
- Question: does doubling training budget shrink seed variance (data-hungry)
  or leave it high (unstable)?
- Metric: torture mean lift per seed; sigma comparison across budgets.
- Interpretation gates (frozen): sigma_12k < sigma_6k/2 -> data-hungry;
  sigma_12k >= sigma_6k/2 and mean not improved -> instability.
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

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_l2"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))

SEEDS = [42, 142, 242, 342, 442]

# reuse approved engine + eval from stage_l module
_spec2 = importlib.util.spec_from_file_location(
    "stage_l", os.path.join(_here, "stage_l_w_scaling.py"))
stage_l = importlib.util.module_from_spec(_spec2)
sys.modules["stage_l"] = stage_l
_spec2.loader.exec_module(stage_l)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    print("=== STAGE L2: W480 VARIANCE DISAMBIGUATION ===")

    payload = {}
    for budget in (6000, 12000):
        arm = {"seeds": {}}
        t0 = time.perf_counter()
        for seed in SEEDS:
            model = esc.VectorizedPseudoBrain(
                thoughtlets=32, width=480, heads=4, cycles=3,
                init_seed=seed).to(device)
            stage_l.train_batched(model, device, budget, seed)
            lift = stage_l.eval_lift(model, device)
            arm["seeds"][str(seed)] = round(lift, 4)
            print(f"W480 {budget} steps s{seed}: lift={lift:+.4f}")
        lifts = list(arm["seeds"].values())
        arm["mean"] = round(float(np.mean(lifts)), 4)
        arm["sigma"] = round(float(np.std(lifts)), 4)
        arm["wall_s"] = round(time.perf_counter() - t0, 1)
        payload[f"steps_{budget}"] = arm
        print(f"{budget}-step SUMMARY: mean={arm['mean']} sigma={arm['sigma']}")

    s6 = payload["steps_6000"]["sigma"]
    s12 = payload["steps_12000"]["sigma"]
    m6 = payload["steps_6000"]["mean"]
    m12 = payload["steps_12000"]["mean"]
    if s12 < s6 / 2:
        verdict = "DATA-HUNGRY (variance halves with 2x budget)"
    elif m12 < m6:
        verdict = "MIXED (variance persists but mean improves)"
    else:
        verdict = "UNSTABLE (high variance persists without mean gain)"
    payload["VERDICT"] = {"sigma_6k": s6, "sigma_12k": s12,
                          "mean_6k": m6, "mean_12k": m12,
                          "classification": verdict}
    print(json.dumps(payload["VERDICT"], indent=2))
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
