"""Parallel GPU+CPU training launcher — saturates 16 CPU cores + RX 9070 XT (dml:1).

dml:0 = AMD Radeon(TM) Graphics (iGPU) — NOT used.
dml:1 = AMD Radeon RX 9070 XT (dGPU) — all GPU jobs go here.
Runs up to MAX_PARALLEL concurrent jobs for ~3-4x wall-clock speedup.
Reactive 4 seeds already done — trains remaining gru/thoughtlet jobs.
"""
import subprocess, sys, time, os
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[2]

# Remaining jobs: gru and thoughtlet (reactive already done with fix)
# GPU jobs -> dml:1 (9070 XT) with big batch to amortize DML launch overhead
jobs = [
    # (model, seed, device, batch, cpu_threads)
    ("gru", 42, "dml:1", 64, None),
    ("gru", 142, "dml:1", 64, None),
    ("gru", 242, "cpu", 32, "8"),
    ("gru", 342, "cpu", 32, "8"),
    ("thoughtlet", 42, "dml:1", 64, None),
    ("thoughtlet", 142, "dml:1", 64, None),
    ("thoughtlet", 242, "cpu", 32, "8"),
    ("thoughtlet", 342, "cpu", 32, "8"),
]

MAX_PARALLEL = 3  # 1x GPU (9070 XT) + 2x CPU to avoid oversubscription
STEPS = 3000

def run_job(model, seed, device, batch, threads):
    cmd = [
        sys.executable, str(BRAIN_ROOT / "experiments" / "gru_vs_thoughtlet" / "train.py"),
        "--models", model,
        "--seeds", str(seed),
        "--steps", str(STEPS),
        "--batch-size", str(batch),
        "--seq-len", "32",
        "--eval-every", "500",
        "--device", device,
    ]
    env = dict(os.environ)
    if threads:
        env["TORCH_NUM_THREADS"] = threads
    log = BRAIN_ROOT / "runs" / "gru_vs_thoughtlet" / f"parallel_{model}_{seed}_{device.replace(':','_')}.log"
    print(f"START {model} seed={seed} device={device} batch={batch} -> {log.name}", flush=True)
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(BRAIN_ROOT.parent), env=env)
    return p, f, log, (model, seed, device)

pending = jobs[:]
running = []
start = time.time()
while pending or running:
    while pending and len(running) < MAX_PARALLEL:
        job = pending.pop(0)
        p, f, log, meta = run_job(*job)
        running.append((p, f, log, meta))

    for item in running[:]:
        p, f, log, meta = item
        ret = p.poll()
        if ret is not None:
            f.close()
            elapsed = time.time() - start
            status = "OK" if ret == 0 else f"FAIL {ret}"
            print(f"DONE {meta[0]} seed={meta[1]} {status} {elapsed:.0f}s log={log.name}", flush=True)
            running.remove(item)
            if ret != 0:
                try:
                    print(open(log).read()[-2000:])
                except Exception:
                    pass
    if pending or running:
        time.sleep(2)

print(f"\nALL PARALLEL DONE {time.time()-start:.0f}s")
