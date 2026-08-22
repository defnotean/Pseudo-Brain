"""Phase 2.6 scaling sweep: latency/params/memory across K and W on GB10.

Pure measurement — no training, no architecture changes. Uses the canonical
VectorizedPseudoBrain from the Phase 2 campaign (batched [B,K,W] hot path).
"""
import time
import json
import numpy as np
import torch

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "esc_base", "/workspace/repo/dgx_phase2_definitive_escalation.py")
esc = importlib.util.module_from_spec(_spec)
import sys
sys.modules["esc_sweep"] = esc
_spec.loader.exec_module(esc)

device = torch.device("cuda:0")
RESULTS = []

def bench_k_w(k: int, w: int, batch: int = 1, iters: int = 200, warmup: int = 50):
    try:
        model = esc.VectorizedPseudoBrain(
            thoughtlets=k, width=w, heads=4, cycles=3, init_seed=42).to(device)
    except Exception as e:
        print(f"K={k} W={w}: BUILD FAILED {e}")
        return
    params = sum(p.numel() for p in model.parameters())
    model.eval()
    rgb = torch.rand(batch, 3, 32, 32, device=device)
    ctrl = torch.zeros(batch, 307, device=device)
    dt = torch.tensor([0.016667] * batch, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            out = model(rgb, ctrl, dt, state=None)
        torch.cuda.synchronize()
        times = []
        mem_before = torch.cuda.memory_allocated()
        for _ in range(iters):
            t0 = time.perf_counter()
            model(rgb, ctrl, dt, state=None)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
    t = np.array(times[5:])
    row = {"k": k, "w": w, "batch": batch, "params": params,
           "p50_ms": round(float(np.percentile(t, 50)), 3),
           "p95_ms": round(float(np.percentile(t, 95)), 3),
           "mean_ms": round(float(t.mean()), 3),
           "peak_alloc_mb": round(peak_mem_mb, 1)}
    RESULTS.append(row)
    print(f"K={k:4d} W={w:4d} params={params:>12,} p50={row['p50_ms']:8.3f}ms "
          f"p95={row['p95_ms']:8.3f}ms peak_mem={row['peak_alloc_mb']:8.1f}MB")

print("=== PHASE 2.6 SCALING SWEEP (GB10, batch=1, 32x32 input, C=3) ===")
for k in (32, 64, 128, 256):
    for w in (120, 240, 480):
        bench_k_w(k, w)

# GRU reference at each comparable size
gru = esc.ProposalGRUBaseline(hidden_dim=112, num_proposals=21).to(device)
params = sum(p.numel() for p in gru.parameters())
gru.eval()
rgb = torch.rand(1, 3, 32, 32, device=device)
ctrl = torch.zeros(1, 307, device=device)
dt = torch.tensor([0.016667], device=device)
with torch.no_grad():
    for _ in range(50): gru(rgb, ctrl, dt, state=None)
    torch.cuda.synchronize()
    times = []
    for _ in range(200):
        t0 = time.perf_counter(); gru(rgb, ctrl, dt, state=None); torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)*1000)
t = np.array(times[5:])
print(f"\nGRU ref: params={params:,} p50={np.percentile(t,50):.3f}ms p95={np.percentile(t,95):.3f}ms")

with open("/workspace/out/scaling_sweep.json", "w") as f:
    json.dump({"rows": RESULTS,
               "gru": {"params": params, "p50_ms": float(np.percentile(t, 50))}}, f, indent=2)
print("\nDONE -> /workspace/out/scaling_sweep.json")
