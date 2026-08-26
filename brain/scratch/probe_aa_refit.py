"""De-risk: reproduce sealed PB21M AA cross-fit params with the canonical solver.

Confirms that irene_brain.v2.hazard_calibration._fit_one_action, run on the
correct fold rows, reproduces the sealed crossfit_scales/biases byte-for-byte.
If so, the audit's solver control C1 is sound.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from irene_brain.v2 import hazard_calibration as hc  # noqa: E402

EVIDENCE = BRAIN / "runs/v21m-qualification/2026-08-25-pb21m-fresh-bal-cal-first-v1.cal-evidence.npz"
L2 = 1.0e-6
MIN_SCALE = 1.0e-4
MAX_ITER = 100
TOL = 1.0e-10

z = np.load(EVIDENCE, allow_pickle=False)
scorer = 0
cell = 0
cohort = int(z["cell_fit_index"][cell])

# fold A rows = cal_targets[cohort][0:3072]; fold B = [3072:6144]
# crossfit entry j evaluates fold evfold[j]. entry0: eval1 fit0. entry1: eval0 fit1.
evfold = z["crossfit_eval_fold_index"]
sealed = z["crossfit_scales"]
raw = z["cal_raw_logits"]
targets = z["cal_targets"][cohort]

worst = 0.0
for entry in range(2):
    fit_fold = int(z["crossfit_fit_fold_index"][entry])
    for act in range(5):
        l = torch.from_numpy(raw[scorer, cell, fit_fold][:, act])
        fit_rows = targets[0:3072] if fit_fold == 0 else targets[3072:6144]
        t = torch.from_numpy(fit_rows[:, act])
        s, b = hc._fit_one_action(
            l, t, l2_regularization=L2, minimum_scale=MIN_SCALE,
            max_iterations=MAX_ITER, tolerance=TOL, fit_mode="per_action_affine",
        )
        d = max(abs(s - float(sealed[scorer, cell, entry, act])),
                abs(b - float(z["crossfit_biases"][scorer, cell, entry, act])))
        worst = max(worst, d)

print(f"scorer{scorer} cell{cell}: worst |param diff| over 2 entries x 5 actions = {worst:.3e}")
print("solver control C1 viable (byte-exact AA refit):", worst < 1e-9)
