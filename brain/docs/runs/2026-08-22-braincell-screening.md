# Phase 2.6 BrainCell screening — results (2026-08-22)

**Design:** 3 variants × 3 seeds (42/142/242) × 6000 steps multitask, identical
recipe. Diagnostics per model: synthetic-memory causal probe, evidence-revision,
learning curves, latency. Data: Spark `runs/bc_screening_results.json`.

## Screening table [MEASURED]

| Variant | Params | Mean lift | Seed σ | Tasks above | Mem causal? | Latency |
|---|---|---|---|---|---|---|
| A current | 824,667 | −0.2024 | ±0.1221 | 3.3/25 | **No** | ~0.88 ms |
| B gated_gru | 853,587 | −0.2833 | ±0.0599 | 1.3/25 | No | ~0.93 ms |
| C evidence_residual | 911,427 | −0.2140 | **±0.0171** | 0.7/25 | **YES (all seeds)** | ~0.92 ms |

## Key findings

1. **[MEASURED] Evidence-residual is the first architecture in this project that
   passes the synthetic-memory causal test.** Injected *correct* evidence improves
   decisions and *wrong* evidence misleads, consistently across all 3 seeds
   (best: s142 correct=0.250 vs zero=0.050 vs wrong=0.050). The current core FAILS
   this test — ideal externally supplied evidence does NOT reliably move its
   decisions. **This confirms the owner's routing hypothesis: the shared update
   rule itself is a memory-integration bottleneck**, independent of store quality.

2. **[MEASURED] gated_gru REJECTED:** strictly worse lift than current, no causal
   memory response. Extra write gating alone doesn't help.

3. **[MEASURED] evidence_residual has 7× lower seed variance than current**
   (±0.0171 vs ±0.1221). The keep/accept decomposition makes training far more
   predictable even where means are similar. Multitask mean lift is statistically
   indistinguishable from current at this budget; tasks-above is lower on average
   but driven by one strong current-seed (s142's 6-task outlier inside ±0.12 σ).

4. Learning curves are noisy for all variants at 6000 steps — no candidate yet
   fixes PB's light-budget multitask weakness. That remains open.

## Promotion decision (per screening strategy)

- **PROMOTE: evidence_residual** to the 5-seed confirmation round + escalation
  collapse-resistance test. Rationale: only candidate with causal memory intake;
  dramatically lower variance; lift parity; latency within noise of control.
- **REJECT: gated_gru** (obvious loser).
- **current** stays as control/reference arm in the confirmation round.

## Confirmation-round gates (pre-declared before running)

evidence_residual earns Core V1 candidacy iff, at 5 seeds:
- collapse rate on escalation ≤ current's (protect Phase 2 strength)
- permutation invariance exact (property test)
- memory causal signature persists at n=5
- multitask lift ≥ current mean − 1σ (no regression beyond noise)

## Routing hypothesis verdict so far

[MEASURED] Partially CONFIRMED: the shared update rule is a demonstrated
memory-intake bottleneck (arm A fails ideal-evidence probe; arm C passes).
[OPEN] Whether fixing intake also improves multitask learning — needs the
confirmation round at higher training budgets.
