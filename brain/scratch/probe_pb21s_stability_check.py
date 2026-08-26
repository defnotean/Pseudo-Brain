"""PB21S-SIG stability check: is the rec-minus-base factual AUC gain robust
to the OOF fold seed? (Read-only; reuses the PB21S-SIG probe module.)

The single-seed run (seed_offset=0 -> TRIAL_SEED+100*fold) gave rec 0.6846
vs base 0.6499 (+0.0347). This sweeps a second seed offset to check the
gain holds across independent OOF permutations (not a fluke).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

spec = importlib.util.spec_from_file_location(
    "pb21s", BRAIN / "scratch" / "probe_pb21s_recurrent_history_signal.py")
pb21s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21s
spec.loader.exec_module(pb21s)


def main() -> None:
    model, _r, _p = pb21s.pb21n.load_parent()
    parent_outcome = model.world_model.outcome_model
    ev = pb21s.collect_ep_evidence(model)
    print(f"[i] evidence {ev['rows']} roots; seed check across offsets")

    for seed_offset in (0, 555, 999):
        base = pb21s.crossfit_arm(parent_outcome, ev, False, seed_offset)
        rec = pb21s.crossfit_arm(parent_outcome, ev, True, seed_offset)
        sb, yb = pb21s.factual_select(base, ev)
        sr, yr = pb21s.factual_select(rec, ev)
        ab = pb21s.rank_auc(yb, sb)
        ar = pb21s.rank_auc(yr, sr)
        per = [f"a{i}:{pb21s.rank_auc(yb[ev['applied']==i], sb[ev['applied']==i]):.3f}"
               f"->{pb21s.rank_auc(yr[ev['applied']==i], sr[ev['applied']==i]):.3f}"
               for i in range(5)]
        print(f"  seed_offset={seed_offset:5d}: base={ab:.4f} rec={ar:.4f} "
              f"gain={ar-ab:+.4f}   [" + "  ".join(per) + "]")


if __name__ == "__main__":
    main()
