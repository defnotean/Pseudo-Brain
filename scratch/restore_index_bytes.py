import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path("..").resolve()

# list tracked files (POSIX paths)
r = subprocess.run(
    ["git", "-C", str(REPO), "ls-files", "-z"],
    capture_output=True, check=True,
)
paths = [p.decode("utf-8") for p in r.stdout.split(b"\x00") if p]
print("tracked files:", len(paths))

restored = 0
drifted = []
for rel in paths:
    q = REPO / rel
    blob = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-p", f":{rel}"],
        capture_output=True,
    )
    if blob.returncode != 0:
        continue
    want = blob.stdout
    cur = q.read_bytes() if q.exists() else None
    if cur == want:
        continue
    q.write_bytes(want)
    restored += 1
    if hashlib.sha256(want).hexdigest() != hashlib.sha256(cur or b"").hexdigest():
        drifted.append(rel)

print("restored files:", len(restored))
print("files whose content hash changed:", len(drifted))
for rel in drifted[:20]:
    print("   ", rel)
