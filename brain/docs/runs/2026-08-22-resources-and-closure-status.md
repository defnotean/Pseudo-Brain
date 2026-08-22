# Phase 2.5 closure inputs: resource accounting + status memo (2026-08-22)

## Resource table [MEASURED on GB10, batch=1, 32×32 input, torch 2.13.0+cu130]

| Model | Params | fwd p50 | fwd p95 |
|---|---|---|---|
| Proposal-GRU (H=112, 21 proposals) | 907,859 | 0.624 ms | 0.630 ms |
| PB K=1 (W=120) | 824,667 | 1.035 ms | 1.048 ms |
| PB K=32 (W=120) | **824,667** | **1.113 ms** | 1.125 ms |

- PB has ~10% *fewer* parameters than the GRU and runs at 1.11–1.13 ms — well inside
  the 16.67 ms / 60 Hz frame budget (≈6.7% of budget). K=1→K=32 costs only +0.08 ms.
- FLOPs not yet formally counted; params and measured latency are the primary matched
  resources here. The GRU is *faster* (0.62 vs 1.11 ms), so PB is not buying its result
  with extra serial depth per step beyond what latency shows.

## Session summary — what was established

1. [MEASURED] CPU preliminary "K=8 beats GRU" did not reproduce under proper DGX
   training-seed methodology.
2. [MEASURED] On 8-hypothesis multi-stage uncertainty (n=10 independent training seeds,
   3000 steps): PB K=32 vs GRU — catastrophic training collapse 1/10 vs 6/10
   (Fisher p=0.027); per-seed returns stochastically higher (permutation p=0.036);
   medians −422 vs −527; S1 survival 46.6% vs 20.0%.
3. [MEASURED] On ephemeral single-cue memory (n=10): no PB advantage (p=0.80, point
   estimate favors GRU). The reliability advantage is task-specific.
4. [MEASURED] Persistence causality replicates: frame-reset collapses PB returns toward
   floor while whole-slot permutation stays harmless.
5. [MEASURED] Resource accounting above.

[INFERRED] Where multiple mutually-incompatible latent hypotheses must be maintained and
separately updated over a long horizon, the parallel persistent thought field makes
*training* substantially more reliable than a parameter-matched monolithic GRU with
equivalent decoder capacity — while matching (not beating) best-case return and costing
~2× per-step latency inside budget.

[HYPOTHESIS] The mechanism is collapse-resistance during optimization: slots preserve
competing hypotheses through the phase where a monolithic state commits prematurely and
then cannot recover.

## Remaining before formal Phase 2 closure

1. Mechanistic battery (stale-state D∈{1,5,20}, prob/binding scrambles, VoI probes)
   on surviving escalation checkpoints — to test the HYPOTHESIS directly.
2. Optional third multi-latent-variable task for the task-specificity claim.
3. Owner review of whether the reliability result meets the §32 bar as registered.
