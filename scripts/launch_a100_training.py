import subprocess
import os

print("=== LAUNCHING DAEMONIZED TRAINING ON CLOUD A100 ===")
with open("/content/train.log", "w") as log_file:
    p = subprocess.Popen(
        ["python3", "-u", "/content/colab_train_a100.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"Training daemon successfully spawned! PID = {p.pid}")
