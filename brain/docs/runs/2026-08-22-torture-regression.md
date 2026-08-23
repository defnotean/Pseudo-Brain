# Phase 2.6 Stage E: torture-suite regression check (2026-08-22)

**Purpose:** verify the current code state (explicit actuator temperature pin,
Constitution CI additions) did not shift the locked baselines. Data: Spark
`runs/torture_regression_results.json`. 3 seeds each, 6000 steps multitask,
per-slot CE (PB) / matched recipe.

## Results [MEASURED]

| Arm | mean lift | seed σ | above chance+2% |
|---|---|---|---|
| PB K=32 | −0.2260 ± 0.0225 | 1.3/25 | locked ref: −0.215, 1/25 |
| GRU | −0.2473 ± **0.1425** | 2.7/25 | locked ref: −0.042, 8/25 |

## Interpretation

- **PB arm REPRODUCES the locked baseline cleanly** (−0.226 vs −0.215, within
  noise; σ small). The temperature pin changed no PB behavior. ✅
- **GRU arm shows high variance across seeds** (−0.36 to −0.05): one strong
  seed (s142: −0.046, 5 tasks above — consistent with the historical GRU
  advantage pattern), two weak seeds near fresh-level (−0.34/−0.36). The
  original locked GRU number (−0.042) came from a single seed; this n=3 run
  reveals that estimate sat at the optimistic end of GRU's seed distribution.
  [INFERRED] The GRU's multitask advantage claim should be softened: its edge
  is real on good seeds but high-variance; the honest statement is "GRU
  reaches substantially better multitask performance on some seeds while PB is
  uniformly weak."

## Consequence for freeze

The regression gate PASSES on its preregistered purpose: code-state changes
(temperature pin, CI harness) produced NO behavioral drift in the canonical
core. PB reproduces; GRU spread is a baseline-documentation improvement, not a
regression. Core V1 freeze may proceed with an updated baseline note.
