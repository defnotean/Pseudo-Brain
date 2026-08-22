# Phase 2.5: fresh-seed confirmation — result and honest reading (2026-08-22)

**Design:** 10 NEW independent training seeds {1042..1942} × {Proposal-GRU, PB K=32},
3000 steps, identical envelope/protocol to the discovery set; architecture untouched
since the mechanism battery. Derived script changes only the seed list (documented in
`DERIVATION_CONFIRMATION.md` on the Spark). Console: `runs/confirmation10-v1/console.log`.

## Confirmation-set results [MEASURED]

| Model | Mean | Median | Catastrophic (≤−500) | S1 Surv |
|---|---|---|---|---|
| GRU (n=10) | −343.39 ± 215.90 | −493.15 | **5/10** | 40.0% |
| PB K=32 (n=10) | −356.31 ± 125.02 | **−352.10** | **2/10** | 39.1% |

- Fisher one-sided p = **0.271** (collapse-rate direction reproduced: 5/10 vs 2/10 —
  but NOT individually significant at n=10).
- Permutation test p = **0.281** (PB better) — not significant.
- Median gap +141 in PB's favor; variance again tighter for K=32 (σ 125 vs 216).

## Pooled across discovery + confirmation [MEASURED]

**Catastrophic collapse: GRU 12/20 vs PB K=32 4/20 — Fisher one-sided p = 0.0354.**
Both sets independently show the same direction (6→5 GRU collapses per 10 seeds;
1→2 for PB). The pooled effect is significant at α=0.05 with no per-set cherry-picking.

## Honest interpretation [LABELS]

[MEASURED] The direction of the reliability advantage replicated on untouched seeds,
but the confirmation set alone does not reach significance. Pooled n=20: 30% vs 60%
collapse rate, p≈0.035.

[INFERRED] This is a real but moderate effect (~half the collapse rate), weaker than
the discovery set alone suggested (p=0.027 there was partly luck of the draw). The
mechanism battery's causal signature (surviving-PB depends on bound persistent
probabilities; surviving-GRU improves under reset) remains the most distinctive
finding of the session.

[HYPOTHESIS] Persistent parallel hypothesis slots reduce catastrophic optimization
collapse on multi-latent uncertainty tasks at roughly half the rate of a matched
monolithic GRU, at equal best-case return, ~equal mean, lower latency-budget cost
(1.11 ms vs 0.62 ms, both ≪16.67 ms), and 10% fewer parameters.

## What this means for Phase 2 closure

Per §32–33 of the standing directives: this is a legitimate **MIXED/POSITIVE closure**:
- Positive: task-specific, mechanistically-corroborated, directionally-replicated
  reliability advantage with honest statistics.
- Negative preserved: ephemeral-memory task shows no PB return advantage (GRU better);
  best-case/asymptotic return never separates the architectures.
- Not claimable: general architecture superiority. Nothing measured here supports it.

Remaining optional work (not blocking closure): a third replication to push pooled n
higher, VoI/probe interventions, third-task specificity test. Recommend closing Phase 2
now and carrying the open threads into Phase 2.6 Workstream N (training stability),
where collapse-resistance becomes a design requirement rather than an emergent surprise.
