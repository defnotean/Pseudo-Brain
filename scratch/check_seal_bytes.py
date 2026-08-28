import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path("..").resolve()

def walk(o):
    if isinstance(o, dict):
        if "files" in o and isinstance(o.get("files"), dict) and "sha256" in o:
            yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)

d = json.loads(
    (REPO / "brain/runs/v21i-development/2026-08-24-seed42-v3.json").read_text(encoding="utf-8")
)
for b in walk(d):
    recorded = b["files"]
    break

targets = [
    "brain/src/irene_brain/evaluation/temporal_persistence_diagnostics.py",
    "brain/src/irene_brain/memory/episodic_v0.py",
    "brain/src/irene_brain/model/actuator.py",
    "brain/src/irene_brain/model/baselines.py",
    "brain/src/irene_brain/model/brain_cell.py",
    "brain/src/irene_brain/model/thought_mediated_actuator.py",
    "brain/src/irene_brain/model/topological_goal.py",
    "brain/src/irene_brain/model/torch_model.py",
]

for rel in targets:
    want = recorded[rel]
    blob = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-p", f":{rel}"],
        capture_output=True,
    )
    if blob.returncode != 0:
        print(rel, "NOT IN INDEX")
        continue
    lb = blob.stdout  # index bytes (as stored)
    h_lf = hashlib.sha256(lb).hexdigest()
    h_crlf = hashlib.sha256(lb.replace(b"\n", b"\r\n")).hexdigest()
    q = REPO / rel
    h_wt = hashlib.sha256(q.read_bytes()).hexdigest() if q.exists() else None
    print(rel)
    print("   dict:", want[:28])
    print("   index-blob:", h_lf[:28], ("MATCH" if h_lf == want else ""))
    print("   crlf-of-blob:", h_crlf[:28], ("MATCH" if h_crlf == want else ""))
    print("   worktree-now:", (h_wt or "?")[:28])
    has_cr_in_blob = b"\r" in lb
    print("   blob contains CR bytes:", has_cr_in_blob)
