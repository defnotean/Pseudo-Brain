# Phase 2.7 Stage H: training pipeline profile — baseline (2026-08-22)

**Setup:** frozen Core V1, canonical multitask trainer, K=32 W=120 C=3,
GB10 (DGX Spark), 300 steps measured after sync points.

## Baseline [MEASURED]

| Metric | Value |
|---|---|
| Wall time | 24.0 s / 300 steps |
| Per train-step | 79.96 ms |
| Frames per train-step (recurrent) | 27.0 mean |
| Raw forward frames/sec | 337.6 |
| Breakdown | forward 45.9% · backward 51.8% · optimizer 0.8% · loss 0.9% · data-prep 0.4% |

## Reading

1. Compute is genuinely GPU-bound: forward+backward = 97.7%. Data prep and
   optimizer are negligible. Python-loop overhead in env access ≈ 0.
2. **The dominant cost is recurrent serialization**: ~27 sequential
   frame-forwards per train step, each a tiny batch-1 kernel launch sequence.
   This is the classic underutilization pattern — the GPU is idle most of each
   frame's latency.
3. Implication for Stage I/J: biggest wins come from (a) batching independent
   seeds/models into one process (M-dim batches amortize kernel launches),
   and (b) possibly CUDA graphs to remove launch overhead. Environment
   vectorization is NOT the bottleneck here.

## Optimization targets (Stage I), frozen-science constraint

- T1: batch M independent models → [M,B,K,W] execution. Expected: near-linear
  speedup until GPU saturates (tiny model ⇒ likely large headroom).
- T2: CUDA graph capture of the train step to cut launch overhead.
- T3: bf16 autocast where numerically safe (verify equivalence).
- T4: fused AdamW (foreach) — cheap win.
