"""Extract the full gate detail from the published PB21N trial #1 result JSON.

Read-only. Prints G1 per-fold factual bias/ECE + per-action, G2 per-fold
point+LCB, G3 per-fold control deltas + worst AUC drop, G5 observed range /
permuted p95 + skipped strata, G6 per-fold point+LCB, and the AA calibration
provenance. This is what the run report is built from.
"""
from __future__ import annotations
import json
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "runs" / "pb21n-hazard-prior-action"
prefix = "2026-08-25-pb21n-prior-action-hazard-conditioning"
r = json.loads((d / (prefix + "-v1.json")).read_text(encoding="utf-8"))

print("status:", r["status"], " passed:", r["passed"])
print("wall_seconds:", r["wall_seconds"])
print("determinism:", r["determinism"])
print("init_identity:", r.get("init_identity"))

g = r["gates"]
for k in ["G1_factual_absolute", "G2_paired_vs_base",
          "G3_all_action_controls", "G4_parent_preservation",
          "G5_shuffle_specificity", "G6_mechanism_isolation"]:
    print("\n## " + k, "-> passed:", g[k]["passed"])
    for key, val in g[k].items():
        if key == "passed":
            continue
        s = json.dumps(val, sort_keys=True, default=str)
        print(" ", key, "=", s[:1400])

print("\n## oof_logit_sha256")
for k, v in r["oof_logit_sha256"].items():
    print(" ", k, "=", v)
print("\n## partition")
print(json.dumps(r["partition"], sort_keys=True, indent=1))
print("\n## training_provenance (fold/prior/steps)")
for arm in ("retrain", "prior"):
    for rec in r["training_provenance"][arm]:
        print(" ", arm, "fold=%s steps=%s passes=%s params=%s" % (
            rec["fold"], rec["optimizer_steps"], rec["passes"],
            rec["trainable_parameter_count"]))
