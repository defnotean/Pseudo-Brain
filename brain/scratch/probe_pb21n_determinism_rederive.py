"""Confirm whether PB21N trial #1 determinism=false is an artifact of
comparing uninitialized OOF complement rows.

Re-derives crossfit_arms TWICE from the saved evidence arrays (read-only;
does NOT re-publish anything) and compares:
  - whole array        (reproduces the published determinism=false)
  - eval rows only     (fold f's own rows) for both folds, both arms

If eval-rows match bit-exactly across the two re-derivations, the
published determinism flag was a false negative from the uninitialized
complement, and the trial's gate values are reproducible.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]
os_ = __import__("os")
os_.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

spec = importlib.util.spec_from_file_location(
    "pb21n", BRAIN / "scripts" / "v21n_prior_action_hazard_conditioning_v1.py")
pb21n = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21n
spec.loader.exec_module(pb21n)

d = BRAIN / "runs" / "pb21n-hazard-prior-action"
prefix = "2026-08-25-pb21n-prior-action-hazard-conditioning"
z = np.load(d / (prefix + "-v1.evidence.npz"), allow_pickle=False)

ev = pb21n.CALEvidence(
    beliefs=z["beliefs"].astype(np.float64),
    targets=z["targets"].astype(np.float64),
    applied=z["applied"].astype(np.int64),
    prior_applied=z["prior_applied"].astype(np.int64),
    episode_ordinal=z["episode_ordinal"].astype(np.int64),
    root_ids=z["root_ids"],
    base_raw_logits=z["base_raw_logits"].astype(np.float64),
    fold_index=z["fold_index"].astype(np.int64),
    episode_permutation_sha256=str(z["episode_permutation_sha256"].tobytes().decode("latin1")),
    partition_manifest_sha256=str(z["partition_manifest_sha256"].tobytes().decode("latin1")),
    rows=int(z["fold_index"].shape[0]),
)

model, _, _ = pb21n.load_parent()
outcome = model.world_model.outcome_model

r1, p1, _ = pb21n.crossfit_arms(outcome, ev)
r2, p2, _ = pb21n.crossfit_arms(outcome, ev)

fold = ev.fold_index


def cmp(name, A, B):
    whole = bool(np.array_equal(A, B))
    lines = []
    for f in range(2):
        m = fold == f
        eq = bool(np.array_equal(A[f][m], B[f][m]))
        lines.append("    fold%d eval rows bit-equal=%s" % (f, eq))
    print("%-12s whole-array equal=%s" % (name, whole))
    print("\n".join(lines))
    return whole


w_retrain = cmp("retrain", r1, r2)
w_prior = cmp("prior", p1, p2)

print("\n=== conclusion ===")
print("re-derivation reproducible (eval rows) for retrain:",
      all(bool(np.array_equal(r1[f][fold == f], r2[f][fold == f])) for f in range(2)))
print("re-derivation reproducible (eval rows) for prior:",
      all(bool(np.array_equal(p1[f][fold == f], p2[f][fold == f])) for f in range(2)))
