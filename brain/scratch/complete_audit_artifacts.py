"""Deterministic artifact completion for the sealed PB21M post-hoc audit.

The runner's _publish wrote result.json + attempt.json (create-only). The
prereg execution contract also names registration.json + evidence.npz. This
completes those two WITHOUT modifying the sealed result/attempt:

  1. Load the sealed parent evidence (runs the OOF reconstruction self-check).
  2. Deterministically recompute the focal MIX35 (w=0.35) OOF logits.
  3. Gate: write evidence.npz ONLY if its SHA-256 byte-matches the sealed
     attempt.json focal_oof_logits_sha256 (determinism proof, not a re-run).
  4. Write registration.json from the frozen module constants.
  5. Record artifact-completion provenance with all digests.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

import numpy as np  # noqa: E402
import v21m_posthoc_mechanism_audit_v1 as m  # noqa: E402

EVID = BRAIN_ROOT / "runs" / "pb21m-posthoc-audit"
RUN = m.RUN_ID

arrays = m.load_parent_evidence()
focal = m.arm_a(arrays, m.MIX35_FOCAL_WEIGHT)
oof_logits = np.ascontiguousarray(focal["oof_logits"])
oof_prob = np.ascontiguousarray(focal["oof_probabilities"])
h_logits = m._sha256_bytes(oof_logits.tobytes())

att = json.loads((EVID / f"{RUN}.attempt.json").read_text())
recorded = att["focal_oof_logits_sha256"]
match = (h_logits == recorded)
print("focal OOF logits recompute sha256 :", h_logits)
print("sealed attempt focal_oof sha256   :", recorded)
print("determinism", "OK" if match else "MISMATCH")
if not match:
    print("STOP: determinism red flag; evidence.npz NOT written.")
    sys.exit(1)

ev_path = EVID / f"{RUN}.evidence.npz"
if ev_path.exists():
    print("evidence.npz already exists; leaving untouched.")
else:
    np.savez(
        ev_path,
        focal_oof_logits=oof_logits,
        focal_oof_probabilities=oof_prob,
        focal_oof_logits_sha256=h_logits.encode("ascii"),
        focal_weight=np.float64(m.MIX35_FOCAL_WEIGHT),
    )
    print("wrote evidence.npz  shape", oof_logits.shape)

reg = {
    "schema_version": 1,
    "run_id": m.RUN_ID,
    "mode": m.MODE,
    "classification": m.CLASSIFICATION,
    "prereg": "brain/docs/preregistrations/"
              "2026-08-25-pb21m-posthoc-mechanism-audit-v1.md",
    "parent_sha": m.PARENT_SHA,
    "parent_paths": {k: str(v) for k, v in m.PARENT_PATHS.items()},
    "solver": {"L2": m.L2, "MIN_SCALE": m.MIN_SCALE,
               "MIN_EXAMPLES": m.MIN_EXAMPLES, "MIN_CLASS": m.MIN_CLASS,
               "MAX_ITER": m.MAX_ITER, "TOL": m.TOL},
    "arm_a": {"focal_weight": m.MIX35_FOCAL_WEIGHT,
              "context_weights": list(m.MIX35_CONTEXT_WEIGHTS),
              "gates": {"factual_bias_limit": m.FACTUAL_AGGREGATE_BIAS_LIMIT,
                        "factual_ece_limit": m.FACTUAL_AGGREGATE_ECE_LIMIT,
                        "all_action_worse_margin": m.ALL_ACTION_CONTROL_WORSE_MARGIN,
                        "factual_auc_drop_limit": m.FACTORIAL_AUC_DROP_LIMIT}},
    "arm_b": {"b2_cells_required": m.B2_CELLS_REQUIRED,
              "b2_stratum_bias": m.B2_STRATUM_BIAS,
              "b2_stratum_ece": m.B2_STRATUM_ECE,
              "b3_permutations": m.B3_PERMUTATIONS,
              "b3_shuffle_seed": m.B3_SHUFFLE_SEED,
              "min_stratum_rows": m.MIN_STRATUM_ROWS},
    "one_sided_alpha": m.ONE_SIDED_ALPHA,
    "quantile_method": m.QUANTILE_METHOD,
    "focal_oof_logits_sha256": h_logits,
}
reg_path = EVID / f"{RUN}.registration.json"
if reg_path.exists():
    print("registration.json already exists; leaving untouched.")
else:
    reg_payload = json.dumps(reg, sort_keys=True, allow_nan=False,
                             separators=(",", ":"))
    reg_path.write_text(reg_payload + "\n", encoding="utf-8")
    print("wrote registration.json")

ev_bytes = ev_path.read_bytes()
reg_bytes = reg_path.read_bytes()
comp = {
    "note": "Artifact completion. Runner _publish wrote result.json + attempt.json "
            "only; registration.json and evidence.npz (prereg execution-contract "
            "artifacts) were completed post-hoc by deterministic recomputation. "
            "focal_oof_logits recompute SHA-256 verified byte-equal to attempt.json "
            "focal_oof_logits_sha256 before evidence.npz was written. result.json "
            "and attempt.json were NOT modified.",
    "focal_oof_logits_sha256_verified": bool(match),
    "evidence_npz_sha256": hashlib.sha256(ev_bytes).hexdigest(),
    "registration_sha256": hashlib.sha256(reg_bytes).hexdigest(),
}
comp_path = EVID / f"{RUN}.artifact-completion.json"
comp_path.write_text(json.dumps(comp, sort_keys=True, indent=1) + "\n",
                     encoding="utf-8")
print("wrote artifact-completion.json")
print("completion done")
