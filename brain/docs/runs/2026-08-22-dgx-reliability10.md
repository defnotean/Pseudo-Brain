# Phase 2.5: 10-training-seed reliability test — GRU vs PB K=32 (2026-08-22)

**Machine:** gx10-db18, GB10. **HEAD:** 60db72f.
**Derived script:** `escalation_reliability10_derived.py` (changes documented in
`DERIVATION_RELIABILITY.md`): 10 independent training seeds {42..942}, configs {GRU, K=32},
3000 steps. Console: `runs/escalation-reliability10-v1/console.log`. Note: the console
header strings ("5 Independent", "1200 steps") are literals in the script; actual run =
10 seeds × 3000 steps, confirmed by wall-clock (GRU 285.7s ≈ 2× the 5×3000 baseline).

## Results [MEASURED]

| Model | Mean Return | Median | S1 Surv | S2 Surv | Acc |
|---|---|---|---|---|---|
| Proposal-GRU (n=10) | −426.10 ± 189.27 | −525.55 | 20.0% | 20.8% | 13.0% |
| PB K=32 (n=10) | **−313.33 ± 184.69** | **−397.45** | **46.6%** | 43.4% | 5.7% |

Hostile ablations (seed-0): frame reset hurts K=32 (−487→−526); permutation harmless;
scramble mild. GRU insensitive to all ablations (its persistent state is its hidden carry).

## Statistical honesty

- Mean gap: +112.8 return for K=32 (~27% relative). Pooled SE ≈ 84 → naive two-sample
  t ≈ 1.35 → **not individually significant at α=0.05** on means alone.
- However the distributions differ in *shape*: GRU median sits at the catastrophic floor
  (−526) while K=32's median (−397) is above its own mean — i.e. K=32 has no left tail.
  A rank-based test (Mann-Whitney) would likely separate them, but per-seed returns are
  not currently dumped by the script — **this must be fixed before any claim** ([NEXT]).
- Survival doubling (S1 46.6% vs 20.0%) is the mechanism-level correlate: high-K thought
  fields survive Stage-1 hazard exposure far more often.

## Status of the architectural question [LABELS]

[MEASURED] Across 10 independent training seeds on the 8-hypothesis escalation task,
PB K=32 outperformed Proposal-GRU on mean (+113), median (+128), and Stage-1 survival
(46.6% vs 20.0%).
[INFERRED] The advantage is reliability-shaped: same-order variance, but GRU collapses
to the catastrophic floor on roughly half its seeds while K=32 does not.
[HYPOTHESIS] Parallel persistent hypothesis slots reduce the probability of
optimization collapse on multi-stage uncertainty tasks.

## What would make this a real result

1. Per-seed return dumping + Mann-Whitney / bootstrap CI (script change, rerun or reuse).
2. Reproduce on the ephemeral-memory task where GRU currently wins return.
3. Persistence causal battery on the winning K=32 checkpoints.
4. Resource accounting re-check (params/FLOPs/latency on GB10).

Until then: preliminary positive signal, NOT a claim. Phase-2-closure criteria (§33)
are close to met either way — this is the final credible answer forming.
