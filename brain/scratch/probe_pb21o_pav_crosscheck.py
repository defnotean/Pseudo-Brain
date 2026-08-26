"""Cross-check the hand-rolled PAV against sklearn's IsotonicRegression.

The trial uses the hand-rolled PAV; this is a test-only cross-check to
confirm the two agree (same monotone piecewise-constant mapping up to the
tie-handling convention). Run as a probe, not part of the trial path.
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

from sklearn.isotonic import IsotonicRegression

rng = np.random.default_rng(0)
max_abs_diff = 0.0
for trial in range(5):
    n = int(rng.integers(50, 1537))
    x = rng.standard_normal(n)
    # make targets correlated-ish with x but with noise
    t = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * x))).astype(float)
    node_values, node_probs, steps = pb21o.pav_fit(x, t)
    my_eval = pb21o.pav_evaluate(x, node_values, node_probs)
    sk = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, t)
    sk_eval = sk.predict(x)
    diff = float(np.max(np.abs(my_eval - sk_eval)))
    max_abs_diff = max(max_abs_diff, diff)
    # monotonicity check on our mapping
    assert np.all(np.diff(node_probs) >= -1e-12), "PAV not monotone"
print(f"max |mine - sklearn| across 5 random fits = {max_abs_diff:.3e}")
print("PAV cross-check vs sklearn:", "OK" if max_abs_diff < 1e-9 else "DIVERGED")
