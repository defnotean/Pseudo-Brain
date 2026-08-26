"""Hazard information-ceiling diagnostic on the published PB21O evidence.

READ-ONLY analysis on sealed arrays (no new partition, no publish, no
frozen-state change). Purpose: decompose the factual-domain miscalibration
that every post-processing trial (PB21N affine, PB21O isotonic) failed to
fix, to decide whether a representation-level trial is worth authoring.

Three questions:
  Q1. RANKING: factual AUC of each hazard logit table, per action +
      aggregate. AUC is rank-based and invariant to ANY monotone remap —
      it is the ceiling every calibrator can ever reach.
  Q2. LEVEL: the best factual ECE a per-action IN-SAMPLE isotonic (PAV)
      fit achieves on the same table (no cross-fit penalty). If even
      in-sample PAV cannot reach the 0.05 factual ECE gate, the gap is
      not cross-fit variance.
  Q3. REPRESENTATION: can a linear readout of the FROZEN upstream beliefs
      predict the factual hazard better (AUC) than the hazard path's own
      logit? If yes -> representation sufficient, hazard path under-fit.
      If no -> the frozen representation itself lacks the signal; only a
      representation-level change can move the gate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

spec = importlib.util.spec_from_file_location(
    "pb21o", BRAIN / "scripts" / "v21o_isotonic_hazard_calibration_v1.py")
pb21o = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21o
spec.loader.exec_module(pb21o)

from irene_brain.evaluation.v21_qualification_metrics import (  # noqa: E402
    binary_probability_metrics as bpm,
)

NPZ = (BRAIN / "runs" / "pb21o-isotonic-hazard-cal" /
       "2026-08-25-pb21o-isotonic-hazard-calibration-v1.evidence.npz")
z = np.load(NPZ)
targets = z["targets"]
applied = z["applied"]
beliefs = z["beliefs"]
fold_index = z["fold_index"]
base_raw = z["base_raw_logits"]
oof_prior_raw = z["oof_prior_raw"]
ACTION_COUNT = 5

f0 = fold_index == 0
f1 = fold_index == 1


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Deterministic rank-sum AUC with tie-averaged ranks."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1.0]
    neg = s[y == 0.0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="stable")
    vals = combined[order]
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    i = 0
    while i < vals.size:
        j = i
        while j + 1 < vals.size and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = ranks[: pos.size].sum()
    return float(
        (rank_sum_pos - pos.size * (pos.size + 1) / 2.0)
        / (pos.size * neg.size))


def combined_eval(tab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack the two OOF slots' eval halves into one (score, target) pair
    per action via per-row applied-action selection."""
    scores = []
    ys = []
    for f in range(2):
        rows = (fold_index == f)
        s = tab[f][rows]                      # [1536, 5]
        t = targets[rows]                     # [1536, 5]
        a = applied[rows]                     # [1536]
        idx = a                               # factual column per row
        scores.append(s[np.arange(rows.sum()), idx])
        ys.append(t[np.arange(rows.sum()), idx])
    return np.concatenate(scores), np.concatenate(ys)


print("=" * 72)
print("PB21O published-evidence diagnostic (READ-ONLY)")
print("=" * 72)

print("\n--- Factual positive rates (applied action) ---")
for a in range(ACTION_COUNT):
    mask = applied == a
    n = int(mask.sum())
    pos = int(targets[mask, a].sum())
    print(f"  action {a}: n_factual={n:5d}  positives={pos:4d}  "
          f"rate={pos / n:.3f}")

print("\n--- Q1: factual AUC (rank ceiling for ANY monotone calibrator) ---")
for name, tab in (("frozen base_raw", np.stack([base_raw, base_raw])),
                  ("prior-conditioned OOF", oof_prior_raw)):
    s_all, y_all = combined_eval(tab)
    agg = auc(y_all, s_all)
    line = []
    for a in range(ACTION_COUNT):
        s_a, y_a = s_all[applied == a], y_all[applied == a]
        line.append(f"a{a}={auc(y_a, s_a):.3f}")
    print(f"  {name:24s} agg AUC={agg:.4f}   " + "  ".join(line))

print("\n--- Q2: in-sample per-action PAV factual ECE (no cross-fit) ---")
for f in range(2):
    rows = (fold_index == f)
    per = []
    s_cat, t_cat = [], []
    for a in range(ACTION_COUNT):
        fa = applied[rows] == a
        x = np.ascontiguousarray(oof_prior_raw[f][rows][fa, a])
        t = np.ascontiguousarray(targets[rows][fa, a])
        nv, np_, _ = pb21o.pav_fit(x, t)
        p_in = pb21o.pav_evaluate(x, nv, np_)
        per.append(bpm(t, p_in, ece_bins=10).as_dict()["ece_equal_mass"])
        s_cat.append(p_in)
        t_cat.append(t)
    agg_ece = bpm(np.concatenate(t_cat), np.concatenate(s_cat),
                  ece_bins=10).as_dict()["ece_equal_mass"]
    print(f"  fold{f}: " + "  ".join(f"a{a}={v:.4f}" for a, v in enumerate(per))
          + f"   agg ECE={agg_ece:.4f}")
print("  (trial cross-fit PAV factual ECE for reference: fold0 0.0842,")
print("   fold1 0.0625; in-sample numbers above are an optimistic bound)")

print("\n--- Q3: linear readout of frozen beliefs vs hazard logit AUC ---")
try:
    from sklearn.linear_model import LogisticRegression
    have_sk = True
except Exception:
    have_sk = False
if not have_sk:
    print("  sklearn unavailable; skipping.")
else:
    for f in range(2):
        rows = (fold_index == f)
        for a in range(ACTION_COUNT):
            fa = applied[rows] == a
            if int(fa.sum()) < 40:
                continue
            X = beliefs[rows][fa]
            y = targets[rows][fa, a]
            lr = LogisticRegression(max_iter=500, random_state=0,
                                    n_jobs=1).fit(X, y)
            p_lr = lr.predict_proba(X)[:, 1]
            s_haz = oof_prior_raw[f][rows][fa, a]
            print(f"  fold{f} a{a}: hazard AUC={auc(y, s_haz):.4f}   "
                  f"linear-belief AUC={auc(y, p_lr):.4f}")

print("\n--- Guide ---")
print("  Q1 AUC low (~0.55-0.70) -> ranking weak; no calibrator fixes it;")
print("    representation-level trial is the only remaining route.")
print("  Q1 AUC high (~0.80+) but Q2 in-sample PAV >> 0.05 ECE -> logit")
print("    mapping pathology; a hazard-path training change, not post-hoc.")
print("  Q3 linear-belief AUC >> hazard AUC -> representation sufficient;")
print("    the hazard path is under-fit / poorly conditioned.")
