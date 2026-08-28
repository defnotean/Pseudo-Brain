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

# targets: every recorded file, to find seal-time bytes
targets = {rel: want for rel, want in recorded.items()}

# current disk match check + variants
need = {}
for rel, want in targets.items():
    q = REPO / rel
    if q.exists():
        cur = q.read_bytes()
        if hashlib.sha256(cur).hexdigest() == want:
            continue
        if hashlib.sha256(cur.replace(b"\n", b"\r\n")).hexdigest() == want:
            q.write_bytes(cur.replace(b"\n", b"\r\n"))
            print("REWROTE CRLF:", rel)
            continue
    # blob variants
    blob = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-p", f":{rel}"],
        capture_output=True,
    )
    if blob.returncode == 0:
        for form in (blob.stdout, blob.stdout.replace(b"\n", b"\r\n")):
            if hashlib.sha256(form).hexdigest() == want:
                (REPO / rel).write_bytes(form)
                print("RESTORED from index:", rel)
                break
        else:
            need[rel] = want

if need:
    print("still missing:", len(need))
    # enumerate all git objects (loose + packed), compute sha256 of content
    r = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "--batch-all-objects", "--batch-check"],
        capture_output=True, text=True,
    )
    entries = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            entries.append((parts[0], int(parts[2])))
    print("blobs in object db:", len(entries))
    hits = {}
    # stream contents in batches
    batch = 32
    for i in range(0, len(entries), batch):
        chunk = entries[i:i + batch]
        proc = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "--batch"],
            input="\n".join(h for h, _ in chunk).encode(),
            capture_output=True,
        )
        # parse output: "<oid> <type> <size>\n" then content
        out = proc.stdout
        pos = 0
        for oid, size in chunk:
            hdr = out[pos:].split(b"\n", 1)
            try:
                meta = hdr[0].decode()
            except Exception:
                break
            if " " not in meta:
                break
            a, b, c = meta.split(" ", 2)
            if a == oid and b == "blob":
                csize = int(c)
                content = out[pos + len(hdr[0]) + 1: pos + len(hdr[0]) + 1 + csize]
                h = hashlib.sha256(content).hexdigest()
                for rel, want in need.items():
                    if h == want:
                        hits[rel] = oid
                pos += len(hdr[0]) + 1 + csize + 1
            else:
                pos += out[pos:].find(b"\n", 0) + 1
                pos2 = out.find(b"\n", pos)
                pos = pos2 + 1 if pos2 != -1 else len(out)
        if len(hits) == len(need):
            break
    print("object-db hits:", hits)
    for rel, oid in hits.items():
        content = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-p", oid],
            capture_output=True,
        ).stdout
        (REPO / rel).write_bytes(content)
        print("WROTE seal-time bytes:", rel)
