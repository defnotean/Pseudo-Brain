"""L9b — outcome-history signal diagnostic (READ-ONLY, no publish).

Question: does the factual hazard target have *temporal* structure — is the
current factual hazard better predicted by lagged hazard events (within the
same episode) than by the marginal hazard rate? This decides whether an
"outcome-history features" hazard-path input (a representation-level
direction orthogonal to the L9-refuted prior-action *window-length* axis) is
motivated enough to spend a probe/trial on.

If P(h_t | h_{t-k}) ≈ P(h_t) for all k (no lift), the hazard target is
approximately exchangeable over ticks and outcome-history features carry no
ranking signal on this partition -> that direction is refuted cheaply.

Read-only on the sealed PB21O evidence (no partition, no publish, no
frozen-state change). Row order within an episode = tick order; 12 roots
per episode (sequence_length 16 - burn-in 4).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

BRAIN = Path(__file__).resolve().parents[1]
p = sorted((BRAIN / "runs" / "pb21o-isotonic-hazard-cal").glob(
    "*evidence.npz"))[0]
z = np.load(p)
targets = z["targets"].astype(np.float64)   # [3072, 5]
applied = z["applied"].astype(np.int64)
h = np.array([targets[r, applied[r]] for r in range(len(applied))],
             dtype=np.float64)
rows_per_ep = 12
n = len(h)
tick = np.arange(n) % rows_per_ep
ep = np.arange(n) // rows_per_ep
marg = float(h.mean())

print("=" * 66)
print("Outcome-history signal diagnostic (READ-ONLY; PB21O evidence)")
print("=" * 66)
print(f"factual marginal hazard rate = {marg:.4f}  (n={n})")

print("\n--- lagged factual-hazard conditioning (within-episode) ---")
for lag in (1, 2, 3, 4, 6, 8, 10):
    pp_hit = pp_den = pn_hit = pn_den = 0
    for r in range(n):
        if tick[r] < lag:
            continue
        pred = r - lag
        if ep[pred] != ep[r]:
            continue
        if h[pred] == 1.0:
            pp_den += 1
            pp_hit += h[r]
        else:
            pn_den += 1
            pn_hit += h[r]
    pp = pp_hit / pp_den if pp_den else float("nan")
    pn = pn_hit / pn_den if pn_den else float("nan")
    print(f"  lag {lag:2d}: P(h|h_prev=1)={pp:.4f} (n={pp_den:4d})   "
          f"P(h|h_prev=0)={pn:.4f} (n={pn_den:4d})   "
          f"lift+={pp/marg:.3f}x  lift-={pn/marg:.3f}x")

print("\n--- any-hazard-in-last-K-ticks (within episode) as predictor ---")
for K in (1, 2, 3, 4, 6):
    hit = den = 0
    for r in range(n):
        if tick[r] < K:
            continue
        if h[r - K:r].any():
            den += 1
            hit += h[r]
    print(f"  K={K}: P(h_t=1 | any hazard in last {K}) = "
          f"{hit/den:.4f} (n={den}) vs marginal {marg:.4f} "
          f"({(hit/den)/marg:.3f}x)" if den else f"  K={K}: no pairs")

print("\nGUIDE: lift substantially != 1 on the binding direction =>")
print("outcome-history features are a motivated representation-level")
print("candidate (worth a fresh probe/trial). lift ~1 everywhere =>")
print("the hazard target is ~exchangeable over ticks; that direction is")
print("refuted cheaply.")
