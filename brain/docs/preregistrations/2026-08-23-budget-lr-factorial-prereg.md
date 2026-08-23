# Preregistration: Stage 3a — training-budget × LR-schedule factorial (frozen before run)

**Date:** 2026-08-23. **Question:** can Core V1 (frozen W=120 K=32) convert
additional training compute into torture-suite lift, and does the optimizer
regime gate that conversion?

## Design (clean factorial)

- Architecture frozen: W=120 K=32 C=3. Pinned banks (`crc32`, digest
  verified). Strict deterministic mode. 4 seeds {42,142,242,342}.
- **One training run to 24k steps per (schedule, seed)**; evaluate the SAME
  run at checkpoints 6k / 12k / 24k (checkpoint eval is optimizer-state-free,
  so scientifically equivalent to independent runs — recorded as such).
- Arms (the ONLY two):
  - **FIXED**: AdamW lr=5e-4 constant (the canonical recipe).
  - **SCHEDULED**: linear warmup 500 steps → cosine decay to 5e-5 over 24k.
    The schedule is an explicitly declared cell, not a side effect.
- No other recipe changes. Same bank, batch engine, loss, grad clip 1.0,
  weight decay 1e-4.

## Metrics per cell

mean torture lift; count above chance+2%; per-faculty lifts; training loss
(probe steps); seed σ; learning-curve AUC (trapezoid over checkpoint lifts);
gradient-norm probes; wall time. Provenance block per run (bank digest, init
digest, seeds, det flags, torch/CUDA).

## Predeclared outcome classes

- **A budget-limited:** 24k ≫ 6k (Δ ≥ 0.06 mean) in BOTH schedules, σ stays
  ≤ 0.05 → rerun W-curve at the adequate budget before any W conclusion.
- **B schedule-bound:** SCHEDULED−FIXED ≥ 0.06 at 24k → optimizer regime is
  the bottleneck; stability-compatible optimizer becomes the Phase 2.7
  problem; no size increase.
- **C plateau:** |24k − 6k| ≤ 0.03 in both arms → current core is NOT simply
  undertrained; capability plateau in objective/dynamics; parameter scaling
  pointless; pivot to learning-signal analysis.
- **D fragile:** mean improves ≥ 0.06 but σ(24k) > 0.08 → training-stability
  workstream before any scaling.

Ambiguous mixes: report all cells, adopt the most conservative classification,
no tuning loops.

## Cost

4 seeds × 2 schedules × 24k steps ≈ 8 runs ≈ 8–10 h on GB10 (batched engine,
deterministic mode). Checkpoints shared across budgets by design.
