import subprocess
res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for line in res.stdout.splitlines():
    if "python" in line or "train" in line or "pt" in line:
        print(line)
