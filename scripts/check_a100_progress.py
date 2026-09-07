import subprocess
import os
from pathlib import Path

log_path = Path("/content/train.log")
if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"=== REMOTE TRAINING LOG (Showing last 35 of {len(lines)} lines) ===")
    for l in lines[-35:]:
        print(l)
else:
    print("train.log does not exist yet.")

res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
running = any("colab_train_a100" in line for line in res.stdout.splitlines())
print(f"\n[DAEMON STATUS] Trainer Process Running: {running}")

ckpt = Path("/content/tier2_conversational_champion.pt")
if ckpt.exists():
    print(f"[CHECKPOINT] Found champion checkpoint: {ckpt.stat().st_size / (1024*1024):.2f} MB")
