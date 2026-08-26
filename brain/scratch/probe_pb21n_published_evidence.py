"""Inspect the published PB21N trial #1 evidence (diagnostic, read-only).

Confirms the published evidence.npz holds a real, populated replay (not
uninitialized OOF tables) and reports key statistics, WITHOUT re-running
the model. This is what the published determinism=false was computed on.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "runs" / "pb21n-hazard-prior-action"
prefix = "2026-08-25-pb21n-prior-action-hazard-conditioning"
ev = d / (prefix + "-v1.evidence.npz")
print("evidence path:", ev, "exists:", ev.exists())

z = np.load(ev, allow_pickle=False)
print("keys:", sorted(z.keys()))
for k in sorted(z.keys()):
    a = z[k]
    try:
        info = "shape=%s dtype=%s" % (a.shape, a.dtype)
        if a.dtype.kind in "fc":
            info += " finite=%s" % bool(np.isfinite(a).all())
            info += " min=%.4g max=%.4g" % (float(a.min()), float(a.max()))
    except Exception as e:
        info = "%s (%s)" % (a.dtype, e)
    print("  %s: %s" % (k, info))
