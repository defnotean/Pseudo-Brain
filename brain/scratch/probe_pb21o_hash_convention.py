"""Verify the PB21O result_sha256 convention (read-only diagnostic).

The runner hashes the compact JSON payload (no trailing newline) but
write_text on Windows emits CRLF, so file bytes = payload + CRLF. Test
which transformation of the on-disk file reproduces the recorded hash.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]
d = BRAIN / "runs" / "pb21o-isotonic-hazard-cal"
res = [p for p in sorted(d.glob("*.json"))
       if not p.name.endswith((".attempt.json", ".registration.json"))][0]
att = json.loads(list(d.glob("*.attempt.json"))[0].read_text(encoding="utf-8"))
rec = att["result_sha256"]
b = res.read_bytes()
cands = {
    "raw": b,
    "strip_crlf": b[:-2] if b.endswith(b"\r\n") else b,
    "strip_lf": b[:-1] if b.endswith(b"\n") else b,
}
# also: re-serialize the parsed doc with the runner's exact params
doc = json.loads(b.decode("utf-8"))
reser = json.dumps(doc, sort_keys=True, allow_nan=False,
                   separators=(",", ":")).encode("utf-8")
cands["reserialize"] = reser
for name, data in cands.items():
    print(f"{name:12s} {len(data):7d}  "
          f"{hashlib.sha256(data).hexdigest()[:16]}  "
          f"{'MATCH' if hashlib.sha256(data).hexdigest() == rec else ''}")
print("recorded     ", rec[:16])
