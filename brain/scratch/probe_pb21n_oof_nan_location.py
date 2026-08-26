"""Diagnose WHERE the NaN/garbage sits in the published PB21N OOF tables.

For each OOF table, check finiteness SEPARATELY on:
  - eval rows   (fold_index == f)  <- the rows the gates actually read
  - complement  (fold_index != f)  <- expected to be uninitialized np.empty

This tells us whether the gate values were computed on valid data or on NaN.
Read-only; no model run.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "runs" / "pb21n-hazard-prior-action"
prefix = "2026-08-25-pb21n-prior-action-hazard-conditioning"
z = np.load(d / (prefix + "-v1.evidence.npz"), allow_pickle=False)

fold_index = z["fold_index"]
print("fold_index counts: fold0=%d fold1=%d" % (
    int((fold_index == 0).sum()), int((fold_index == 1).sum())))
print("episode_ordinal: min=%d max=%d unique=%d" % (
    int(z["episode_ordinal"].min()), int(z["episode_ordinal"].max()),
    len(np.unique(z["episode_ordinal"]))))

for name in ["oof_H_prior_raw", "oof_H_retrain_raw",
             "oof_H_prior_calibrated", "oof_H_retrain_calibrated",
             "oof_H_base_calibrated"]:
    tab = z[name]
    print("\n=== %s shape=%s ===" % (name, tab.shape))
    for f in range(2):
        ev_rows = fold_index == f
        comp = fold_index != f
        ev_block = tab[f][ev_rows]
        comp_block = tab[f][comp]
        ev_fin = bool(np.isfinite(ev_block).all())
        comp_fin = bool(np.isfinite(comp_block).all())
        ev_min = float(ev_block.min()) if ev_fin else float("nan")
        ev_max = float(ev_block.max()) if ev_fin else float("nan")
        print("  slot[%d]: eval_rows finite=%s (n=%d) min=%.4g max=%.4g | "
              "complement finite=%s (n=%d)" % (
                  f, ev_fin, int(ev_rows.sum()), ev_min, ev_max,
                  comp_fin, int(comp.sum())))
    # also check the OTHER slots: slot[f] complement vs slot[1-f] eval
    # i.e. is slot[0][fold1-rows] garbage while slot[1][fold1-rows] valid?
    for f in range(2):
        other = 1 - f
        cross = tab[f][fold_index == other]
        print("  slot[%d] rows-belonging-to-fold%d: finite=%s (n=%d)" % (
            f, other, bool(np.isfinite(cross).all()),
            int((fold_index == other).sum())))
