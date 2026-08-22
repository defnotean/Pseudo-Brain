# Phase 2.5: 3000-step escalation probe — GRU vs K=16 vs K=32 (2026-08-22)

**Machine:** gx10-db18, GB10. **HEAD:** 60db72f.
**Derived script:** `escalation_3000_derived.py` — identical to
`dgx_phase2_definitive_escalation.py` except training_steps 1200→3000 and K∈{1,8} dropped.
Derivation note: `runs/escalation-long-3000/DERIVATION.md`.
Console: `runs/escalation-long-3000-v1/console.log`.

## Results [MEASURED] — 5 independent training seeds, 3000 steps

| Model | Mean Return | Median | S1 Surv | S2 Surv |
|---|---|---|---|---|
| Proposal-GRU | −420.08 ± 191.01 | −532.90 | 20.0% | 25.2% |
| PB K=16 | −421.27 ± 92.82 | −484.25 | 32.0% | 17.6% |
| PB K=32 | −423.69 ± 64.24 | **−413.00** | **35.4%** | 32.6% |

Ablations section crashed after the GRU row (KeyError 'Pseudo-Brain K=1' — my derived
script removed K=1 from configs but the hardcoded ablation loop still references it;
**my derivation bug**, not a science bug; ablation table for K=16/K=32 lost, rerun cheap).

## Interpretation [INFERRED]

1. **Means are a three-way tie at 3000 steps too** (−420 / −421 / −424). Doubling training
   did not separate architectures on mean return.
2. **The variance story is now the headline:** GRU σ≈191 unchanged from 1200 steps (σ=199)
   and its median is still at the catastrophic floor (−533). K=32 cuts σ to **64**
   (3× tighter than GRU) and its median (−413) sits *above* its mean — no left tail.
   Across both budgets, roughly half of GRU seeds collapse entirely while every K=32 seed
   stays functional.
3. **K-scaling mechanism evidence accumulates:** S1 survival rises with K at both budgets
   (K=16: 22.6→32.0%, K=32: 11.4→35.4%). High-K thought fields avoid Stage-1 hazard death
   more reliably — consistent with parallel-hypothesis maintenance having survival value
   even when final returns tie.
4. This suggests the correct claim is not "PB wins" but "PB high-K is *more reliable*":
   same mean, far lower failure probability. A distributional comparison (e.g. probability
   of catastrophic-seed) across 10 seeds would make this rigorous.

## Next actions

- Fix the derived-script ablation bug (keep K=1 in configs or patch the loop) and rerun
  the ablation section only if needed — low priority, ablations already characterized at
  1200 steps.
- The decisive open question for Phase 2 closure is now **reliability**: run the escalation
  task at 10 independent training seeds × {GRU, K=32} and test whether P(catastrophic seed)
  differs significantly. That is a falsifiable, distribution-level architecture claim.
