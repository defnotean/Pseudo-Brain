# Preregistration: Episodic Memory v1 acceptance criteria (frozen before run)

**Date frozen:** 2026-08-22, before any v1 training/eval. **Architecture changes
allowed in v1:** longer training, task-focused curriculum, write-gate sparsity
pressure, retrieval instrumentation, injection-gain sweep. **Not allowed:** store
redesign beyond gate temperature, BrainCell changes, dynamic-K, new heads.

## Training protocol (fixed)

- Tasks: t17 + t18 only for first 12k steps, then +fillers for 4k (16k total).
- Controls trained identically: PB no-memory same steps; param-matched widened
  no-memory control.
- Seeds fixed: model init 42, episode seeds 20260822+i*7919.

## Six-condition battery (unchanged from v0) + generalization set

| # | Condition | Requirement |
|---|---|---|
| 1 | normal | t17 ≥ 0.50 AND t18 ≥ 0.65 |
| 2 | store disabled | substantial drop vs normal: Δ ≥ 0.15 on t17 |
| 3 | present-but-unread | ≈ disabled (within 0.05 of condition 2) |
| 4 | correct injected | ≥ normal − 0.10 (restores/improves) |
| 5 | donor injected | ≤ normal − 0.15 on t17 (predictably misleads) |
| 6 | stale present | t18 within 0.10 of its normal value |

Generalization (all must hold):
- G1 unseen key/value combos (fresh seeds 50000+): ≥ 0.8 × seen accuracy
- G2 longer delay (2× training delay on t17): ≥ 0.7 × seen accuracy
- G3 distractor memories present: within 0.10 of clean retrieval
- G4 overwrite: after A→X then A→Y, answers follow Y

## Internal metrics (reported, diagnostic)

- write rate (target: <90% committed, i.e. gate is selective)
- retrieval precision / recall against ground-truth keys
- injection-gain sweep {0, 0.25, 0.5, 1.0, 2.0}: correct-helps/donor-hurts must
  be monotone-ish in gain; gain 0 ≈ disabled

## Cost report (required)

added params, store bytes, latency delta at eval, write/retrieval counts, GPU peak mem.

## Acceptance

ALL six conditions + all four generalization tests pass → Core V1 candidate.
Any failure → record and move to Dynamic-K/lifecycle per directive (no endless tuning).

## Failure conditions (predeclared)

If after focused curriculum + 16k steps the battery remains near v0 levels
(t17 < 0.30), memory v0/v1 records FAIL and the workstream moves on.
