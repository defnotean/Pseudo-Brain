"""Extract PB21O trial #1 gate detail + provenance (read-only).

Glob the run id rather than embedding the long literal.
"""
from __future__ import annotations
import json
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]
EVID = BRAIN / "runs" / "pb21o-isotonic-hazard-cal"
res_files = [p for p in sorted(EVID.glob("*.json"))
             if not p.name.endswith(
                 (".attempt.json", ".registration.json"))]
assert res_files, f"no result json in {EVID}"
RES = res_files[0]
R = RES.name[:-len(".json")]
doc = json.loads(RES.read_text(encoding="utf-8"))
g = doc["gates"]

print("RUN_ID:", R)
print("status:", doc.get("status"), " passed:", doc.get("passed"))
print("determinism:", doc.get("determinism"))
print()

def _cell(container, key):
    """Fetch a per-fold cell whether the container is a dict (fold<->key)
    or a list (fold index)."""
    if isinstance(container, dict):
        return container[key]
    return container[int(str(key))]


for fold in (0, 1):
    t = _cell(g["G1_factual_absolute"]["tables"], str(fold))
    print(f"G1 fold{fold}: bias={t['factual_bias']:.6f} "
          f"ece={t['factual_ece']:.6f} bce={t['factual_bce']:.6f} "
          f"brier={t['factual_brier']:.6f} passed={t['passed']}")
print()

for f in ("fold0", "fold1"):
    c = _cell(g["G2_paired_vs_base"]["cells"], f)
    print(f"G2 {f}: BCE pt={c['point']['BCE']:+.6f} "
          f"LCB={c['LCB_97.5']['BCE']:+.6f} | Brier pt="
          f"{c['point']['Brier']:+.6f} LCB={c['LCB_97.5']['Brier']:+.6f} "
          f"passed={c['passed']}")
print()

g3 = g["G3_all_action_controls"]
g3_list = g3.get("tables") or g3.get("cells") or g3.get("folds") or []
for i, t in enumerate(g3_list):
    print(f"G3 fold{i}:", json.dumps(t, default=str))
print()

g5 = g["G5_function_class_isolation"]
print("G5:", json.dumps(g5, default=str))
print()

print("G4:", json.dumps(g["G4_parent_preservation"], default=str))
print()
prov = doc.get("provenance", {})
print("provenance keys:", sorted(prov.keys()))
for k in ("evidence", "partition", "oof_logit_sha256",
          "iso_cal_provenance", "aa_cal_provenance"):
    if k in doc:
        print(k, ":", json.dumps(doc[k], default=str)[:500])

