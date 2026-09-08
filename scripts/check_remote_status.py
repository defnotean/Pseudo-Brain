import os
import subprocess
from pathlib import Path
import torch

print("=== CHECKING REMOTE A100 STATUS ===")
print("Files in /content:")
for f in sorted(os.listdir("/content")):
    p = Path("/content") / f
    sz = p.stat().st_size / (1024*1024) if p.is_file() else 0
    print(f"  {f} {'(File, ' + f'{sz:.2f} MB)' if p.is_file() else '(Dir)'}")

ckpt_path = Path("/content/tier2_conversational_champion.pt")
if ckpt_path.exists():
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        print(f"\nCheckpoint details:")
        print(f"  Keys: {list(ckpt.keys())}")
        print(f"  Step: {ckpt.get('step')}")
        print(f"  Stage: {ckpt.get('stage')}")
        print(f"  Params: {ckpt.get('parameters')}")
        print(f"  Config: {ckpt.get('config')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("Checkpoint does not exist.")

# Check running processes
res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
print("\nActive Python processes:")
for line in res.stdout.splitlines():
    if "python" in line and not "check_remote_status" in line:
        print(f"  {line}")
