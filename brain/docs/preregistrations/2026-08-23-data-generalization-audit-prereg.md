# Preregistration: Stage 3b — Data / Generalization Audit (2026-08-23)

**Question:** Is Core V1's torture-suite ceiling caused by finite-bank
memorization / data diversity, or by the learning mechanism itself?

## Design (minimal causal test)

- Architecture frozen: Core V1 W=120 K=32 C=3. Fixed optimizer: AdamW
  lr=5e-4 (FIXED schedule from 3a). 6k steps (fast, already enough to
  see loss collapse in 3a). Deterministic mode. 4 seeds {42,142,242,342}.
  Full provenance per run.

- **ARM A (control):** Current finite pinned bank (420 episodes from 14
  tasks × 30 eps, repeating). Bank digest `b3bb5fc33fd5f605`.

- **ARM B (intervention):** Deterministic **non-repeating** training
  stream. Same 14 procedural tasks, but **fresh episodes every step** —
  procedurally generated on the fly with no repetition within a run.
  Seeding: `stream_seed = base_seed + step` ensures exact reproducibility.
  No fixed bank digest (by design).

- Both arms use identical episode construction functions from
  `torture_suite.TASKS`. ARM B calls `fn(rng)` with a fresh RNG per step.
  ARM A samples from the pre-built pinned bank.

- Evaluation: identical locked torture eval (30 eps/task, eval_seed=20260822)
  at 6k only (single checkpoint). We compare ARM A vs ARM B on:
  1. Final training loss (does ARM B stop trivial collapse?)
  2. Mean torture lift (does held-out capability improve?)
  3. Train/eval gap (does it shrink?)

## Predeclared outcome classes

- **Outcome DATA:** ARM B mean lift ≥ ARM A + 0.03 AND train/eval gap
  smaller → finite-bank memorization was the bottleneck. Proceed to
  larger-scale diverse-data training.
- **Outcome MECHANISM:** ARM B mean lift ≈ ARM A (|Δ| ≤ 0.02) AND
  training loss still collapses → data diversity not the primary cause.
  Move aggressively to Stage 3c (evidence-intake / learning-objective
  mechanisms).
- **Ambiguous:** Report all cells, adopt most conservative classification.

## Cost

4 seeds × 2 arms × 6k steps = 8 runs ≈ 2–3 h on GB10 (batched engine).
Much cheaper than 3a; high information per compute.

## Notes

ARM B is still fully deterministic/reproducible — the RNG stream is
seeded from step number, so the entire non-repeating sequence is
fixed per run seed. This maintains our provenance discipline while
testing the causal variable (experience diversity).