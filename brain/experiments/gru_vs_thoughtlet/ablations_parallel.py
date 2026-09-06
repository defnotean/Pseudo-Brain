"""Parallel ablation campaign: unified recipe, 3 seeds each.
1x 9070 XT (dml:1, big models) + 2x CPU. k32 arm = existing base thoughtlet checkpoints.
"""
import subprocess, sys, time, os
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[2]

groups = [
    (["k1", "k2"], "dml:1", None, 64),
    (["k4", "k8", "k16", "no_attention"], "cpu", "8", 32),
    (["no_persistence", "independent_slots"], "cpu", "8", 32),
]
STEPS = 3000
SEEDS = "42 142 242"

def run_group(variants, device, threads, batch):
    cmd = [sys.executable, str(BRAIN_ROOT / "experiments" / "gru_vs_thoughtlet" / "ablations.py"),
           "--variants"] + variants + ["--steps", str(STEPS), "--seeds"] + SEEDS.split() + [
           "--device", device, "--batch-size", str(batch)]
    env = dict(os.environ)
    if threads: env["TORCH_NUM_THREADS"] = threads
    log = BRAIN_ROOT / "runs" / "gru_vs_thoughtlet" / "ablations" / f"campaign_{'_'.join(variants)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {variants} device={device} batch={batch} -> {log.name}", flush=True)
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(BRAIN_ROOT.parent), env=env)
    return p, f, log, variants

running = [run_group(v, d, t, b) for v, d, t, b in groups]
start = time.time()
while running:
    for item in running[:]:
        p, f, log, variants = item
        ret = p.poll()
        if ret is not None:
            f.close()
            print(f"DONE {variants} {'OK' if ret==0 else f'FAIL {ret}'} {time.time()-start:.0f}s", flush=True)
            running.remove(item)
    if running: time.sleep(5)
print(f"ALL ABLATIONS DONE {time.time()-start:.0f}s")
