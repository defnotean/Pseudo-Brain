"""Read-only probe: does a FROZEN BLOCK-CAP on PAV fix the cross-fit
overfit that sank PB21O?

Recomputes, from the published PB21O OOF logits (sealed arrays, no
publish, no frozen-state change), a per-action monotone calibrator that is
PAV pooled down to a fixed block cap K (deterministic greedy: repeatedly
merge the adjacent block-pair whose merge gives the smallest resulting
within-block SSE). Sweeps K as a DESIGN probe only (not a trial sweep; the
frozen trial will pick ONE K). Reports cross-fit factual ECE (the G1
metric) and all-action control, vs the affine (PB21N) and full-PAV
(PB21O) reference points.

K=1 is a constant (degenerate), K=2 ~ affine-like, K=n is full PAV.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
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
targets = z["targets"]; applied = z["applied"]
fold_index = z["fold_index"]; oof_prior_raw = z["oof_prior_raw"]
AA = 5


@dataclass
class Block:
    start: int
    count: float
    sum_y: float
    sum_y2: float
    values: np.ndarray = field(repr=False)  # representative x per member

    @property
    def mean(self):
        return self.sum_y / self.count

    def sse(self):
        return self.sum_y2 - (self.sum_y ** 2) / self.count


def pav_blocks(x: np.ndarray, y: np.ndarray) -> list[Block]:
    """Pav_fit + reconstruct per-member x for SSE bookkeeping."""
    n = x.shape[0]
    order = np.lexsort((np.arange(n), x))
    sv = x[order]; sy = y[order]
    # PAV means
    cnt = np.ones(n); mean = sy.copy()
    stack: list[tuple[int, float, float]] = []
    for i in range(n):
        stack.append((i, 1.0, sy[i]))
        while len(stack) >= 2:
            a = stack[-2]; b = stack[-1]
            mm = (a[2] * a[1] + b[2] * b[1]) / (a[1] + b[1])
            if a[2] <= b[2]:
                break
            stack.pop(); stack.pop()
            stack.append((a[0], a[1] + b[1], mm))
    blocks: list[Block] = []
    for (s, c, m) in stack:
        e = int(s + c)
        blocks.append(Block(start=s, count=c, sum_y=m * c,
                            sum_y2=float(sy[s:e].sum() ** 2),
                            values=sv[s:e]))
    # fix sum_y2 properly (we only stored (sum)^2); recompute from members
    for blk in blocks:
        e = int(blk.start + blk.count)
        blk.sum_y2 = float((sy[blk.start:e] ** 2).sum())
        blk.sum_y = float(sy[blk.start:e].sum())
    return blocks


def pool_to_k(blocks: list[Block], k: int) -> list[Block]:
    if len(blocks) <= k:
        return blocks
    blocks = list(blocks)
    while len(blocks) > k:
        best = None
        for i in range(len(blocks) - 1):
            A = blocks[i]; B = blocks[i + 1]
            c = A.count + B.count
            sy = A.sum_y + B.sum_y; sy2 = A.sum_y2 + B.sum_y2
            merged_sse = sy2 - (sy ** 2) / c
            if best is None or merged_sse < best[0]:
                best = (merged_sse, i)
        i = best[1]
        A = blocks[i]; B = blocks[i + 1]
        c = A.count + B.count; sy = A.sum_y + B.sum_y; sy2 = A.sum_y2 + B.sum_y2
        merged = Block(start=A.start, count=c, sum_y=sy, sum_y2=sy2,
                       values=np.concatenate([A.values, B.values]))
        blocks[i:i + 2] = [merged]
    return blocks


def eval_blocks(blocks: list[Block], x: np.ndarray) -> np.ndarray:
    starts = np.array([b.start for b in blocks], dtype=np.int64)
    probs = np.array([b.mean for b in blocks])
    # x is the logit; map via sorted-value membership. We stored per-block
    # value arrays (sorted sv). For evaluation, map each x to the block whose
    # value-range contains it (right-continuous on the node starts).
    node_vals = np.array([b.values[0] for b in blocks])
    idx = np.clip(np.searchsorted(node_vals, x, side="right") - 1,
                  0, len(probs) - 1)
    return probs[idx]


def crossfit_cal(raw, K, kind):
    """Return OOF prob table [2,3072,5]."""
    out = np.zeros_like(raw)
    for f in range(2):
        fit = fold_index == (1 - f); ev = fold_index == f
        for a in range(AA):
            x = np.ascontiguousarray(raw[1 - f][fit, a])
            t = np.ascontiguousarray(targets[fit, a])
            if kind == "pav":
                blocks = pav_blocks(x, t)
            elif kind == "capped":
                blocks = pool_to_k(pav_blocks(x, t), K)
            elif kind == "affine":
                # closed-form per-action affine on (logit, y)
                A = np.vstack([x, np.ones_like(x)]).T
                coef, *_ = np.linalg.lstsq(A, t, rcond=None)
                m, b = float(coef[0]), float(coef[1])
                out[f][ev, a] = np.clip(m * raw[f][ev, a] + b, 1e-6, 1 - 1e-6)
                continue
            out[f][ev, a] = eval_blocks(blocks, np.ascontiguousarray(
                raw[f][ev, a]))
    return out


def factual_ece_table(tab):
    res = []
    for f in range(2):
        rows = fold_index == f
        s = tab[f][rows]; t = targets[rows]; a = applied[rows]
        sel = np.arange(rows.sum())
        p = s[sel, a]; y = t[sel, a]
        res.append(bpm(y, p, ece_bins=10).as_dict()["ece_equal_mass"])
    return res


def all_action_bias(tab):
    res = []
    for f in range(2):
        rows = fold_index == f
        p = tab[f][rows].reshape(-1); y = targets[rows].reshape(-1)
        res.append(bpm(y, p, ece_bins=10).as_dict()["calibration_bias"])
    return res


print("=" * 68)
print("BLOCK-CAP PAV cross-fit probe (READ-ONLY; sealed PB21O OOF logits)")
print("=" * 68)
for label, K, kind in (
    ("affine (PB21N)", None, "affine"),
    ("capped K=2", 2, "capped"),
    ("capped K=4", 4, "capped"),
    ("capped K=6", 6, "capped"),
    ("capped K=8", 8, "capped"),
    ("capped K=16", 16, "capped"),
    ("full PAV (PB21O)", None, "pav"),
):
    tab = crossfit_cal(oof_prior_raw, K, kind)
    fe = factual_ece_table(tab)
    ab = all_action_bias(tab)
    print(f"  {label:18s} factual ECE f0={fe[0]:.4f} f1={fe[1]:.4f} "
          f"  all-act bias f0={ab[0]:+.4f} f1={ab[1]:+.4f}")
print("\n  G1 gate: factual ECE <= 0.05 BOTH folds.")
