"""De-risk: verify PB21K MIX control geometry + collect_fresh_live_tape sig."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

import numpy as np
import inspect

# --- PB21K evidence geometry ---
P = BRAIN / "runs/v21k-diagnostics/2026-08-25-pb21k-nz-dualcal-v1.evidence.npz"
z = np.load(P, allow_pickle=False)
print("PB21K keys:", sorted(z.keys()))
for k in ("cal_nz_raw_logits", "cell_cal_index", "cal_targets", "cal_actions",
          "mix_proposed_scales", "mix_proposed_biases", "mix_per_action_accepted"):
    if k in z:
        print(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")
    else:
        print(f"  {k}: MISSING")

# --- collect_fresh_live_tape signature ---
import v21i_strict_live_representation_probe_v1 as strict
sig = inspect.signature(strict.collect_fresh_live_tape)
print("\ncollect_fresh_live_tape params:", list(sig.parameters))
print("has collect_fresh_live_tape:", hasattr(strict, "collect_fresh_live_tape"))

# --- partition contracts ---
import v21m_fresh_bal_cal_first_qualification_v1 as pb21m
print("\npartition_contracts:", sorted(pb21m.partition_contracts().keys()))
