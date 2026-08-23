# Phase 2.7 Stage L2: W=480 variance disambiguation — UNSTABLE (2026-08-23)

**Prereg (inline, frozen before run):** arms W480 × {6000, 12000 steps} ×
5 seeds {42,142,242,342,442}; gates sigma_12k < sigma_6k/2 → DATA-HUNGRY;
sigma_12k ≥ sigma_6k/2 without mean gain → UNSTABLE.
Data: Spark `runs/l2_variance_results.json`.

## Results [MEASURED]

| Budget | Mean lift | Seed σ | Per-seed |
|---|---|---|---|
| 6000 | −0.274 | ±0.073 | −0.290 / −0.246 / −0.149 / −0.343 / −0.343 |
| 12000 | −0.243 | ±0.102 | −0.119 / −0.283 / −0.339 / −0.123 / −0.350 |

**Verdict: UNSTABLE** — doubling the training budget did not shrink seed
variance (it grew slightly); mean gain is small relative to spread.

## Additional finding: fixed-seed nondeterminism

Same seeds, same width, same trainer function (`stage_l.train_batched`),
different processes:
- Stage L W480@6k: s42 −0.338, s142 −0.009, s242 −0.170
- Stage L2 W480@6k: s42 −0.290, s142 −0.246, s242 −0.149

Identical nominal configs produced materially different lifts per seed,
consistent with non-deterministic GPU kernel accumulation amplified by the
wide model's optimization landscape. This is independent corroboration that
W=480 under this recipe is unstable rather than merely data-hungry.
[MEASURED — cross-script comparison]

## Consequences

1. Core V1 stays W=120 (frozen). Width scaling does NOT currently deliver
   reliable capability: mean improvement exists but cannot be banked when
   single-run outcomes range from near-chance to worst-in-phase.
2. Scaling-curve conclusion: K-axis flat-and-safe, W-axis noisy-and-unstable
   at light budgets. Any future width push needs stability work first
   (deterministic kernels, LR/warmup tuning, or ensemble averaging) — none of
   which belongs in the frozen core.
3. The best single run ever observed (s142@L: lift −0.009) remains recorded
   as evidence the capacity exists but is not reliably reachable.
