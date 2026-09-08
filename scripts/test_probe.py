import os
from pathlib import Path

print("=== PROBE RUNTIME ===")
print("Files in /content:", os.listdir("/content"))
p = Path("/content/Pseudo-Brain")
if p.exists():
    print("Files in /content/Pseudo-Brain:", os.listdir(p))
    b = p / "brain"
    if b.exists():
        print("Files in /content/Pseudo-Brain/brain:", os.listdir(b))
        d = b / "data"
        if d.exists():
            print("Files in data:", os.listdir(d))
            hf = d / "huggingface"
            if hf.exists():
                print("Files in hf:", os.listdir(hf))
