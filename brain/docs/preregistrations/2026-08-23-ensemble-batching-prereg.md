# Preregistration: multi-model ensemble batching — stabilization + throughput (frozen before run)

**STATUS: DEFERRED 2026-08-23** — superseded by the determinism audit after
the salted-hash root-cause finding
(`brain/docs/runs/2026-08-23-salted-hash-root-cause.md`). The W-axis
variance this experiment targeted was confounded by per-process episode
banks; stabilization claims would be uninterpretable until banks are pinned.
May be re-preregistered after the corrected stability measurement.

> **SUPERSEDED / VOID before run (2026-08-23):** owner redirected priorities
> after the Stage L2 UNSTABLE finding — same-seed cross-run divergence means
> execution nondeterminism is uncontrolled, so any behavioral comparison
> (including ensembling) would be confounded. Replaced by the determinism +
> scaling-stability audit (`2026-08-23-determinism-audit.md`, audit-only).
> This prereg was never executed.

**Date:** 2026-08-23. **Type:** engine/mechanism hybrid — the batching itself
is engine infrastructure (equivalence already proven at episode level), but
the *ensembling claim* is behavioral and gated below.

## Question

Does training E independent Core V1 models simultaneously (stacked-state
batching) and evaluating their slot-averaged aggregate reduce evaluation
variance relative to single models, without degrading mean lift?

## Design

- Frozen Core V1, W=120 K=32 C=3 (the frozen config, NOT W480).
- Engine: extend episode-batched execution to a model dimension [E,B,K,W];
  each model keeps private states; identical recipe/seed handling as
  Stage J/K equivalence run.
- Arms: E ∈ {1, 4} × seeds {42,142,242,342} (E=1 replicates solo baseline;
  E=4 trains four models per arm in one batched execution).
- Aggregate metric for E=4: mean of the 4 models' torture lifts (post-hoc
  ensemble; no cross-model communication — that would be an architecture
  change and is out of scope).

## Gates (frozen)

1. STABILIZATION PASS: σ(aggregate over seed-groups) for E=4 arms < σ(E=1)
   by ≥ 30% relative.
2. NO MEAN HARM: mean lift(E=4 aggregate) within noise of mean lift(E=1)
   (|Δ| ≤ 0.03).
3. THROUGHPUT NOTE [descriptive]: wall-clock per model must not exceed 2×
   solo baseline (batched efficiency check, informational only).

## Predeclared outcomes

- Both gates pass → record ensembling as approved evaluation protocol for
  future scaling runs (engine-level adoption, no core change).
- Gate 2 fails → ensembling rejected as evaluation protocol; instability is
  correlated across seeds (informative for the UNSTABLE finding's mechanism).
- Either gate ambiguous → archive numbers, no adoption, move on.
