# Phase 2.5: ephemeral-memory campaign on DGX GB10 (2026-08-22)

**Machine:** gx10-db18, GB10, CUDA 13.0, torch 2.13.0+cu130 (container `177a406d7cb2`).
**HEAD:** 25b16e3 (scripts unmodified, md5-matched).
**Script:** `dgx_phase2_ephemeral_memory_campaign.py`. Console:
`~/projects/pseudo-brain/runs/phase25-ephemeral-gpu-v1/console.log`.
Design: cues flash 3 frames → vanish; final execution input 100% blank. 5 independent
training seeds {42,142,242,342,442} × {GRU, K=1/8/16/32}, 1500 steps; held-out eval,
delay=15; Normal vs frame-reset persistent state.

## Results [MEASURED]

| Model | Normal Return | Reset Return | Final Acc | S1/S2 Surv |
|---|---|---|---|---|
| Proposal-GRU | **−357.14 ± 177.02** | −515.92 ± 10.57 | 4.8% | 50.4% / 40.0% |
| PB K=1 | −512.85 ± 13.31 | −511.12 ± 11.29 | 2.2% | 0.0% / 0.0% |
| PB K=8 | −495.16 ± 37.91 | −522.22 ± 7.57 | 7.4% | 5.2% / 4.4% |
| PB K=16 | −493.50 ± 30.36 | −521.68 ± 7.04 | 11.2% | 4.8% / 5.0% |
| PB K=32 | −455.20 ± 87.93 | −390.52 ± 200.85 | **16.0%** | 9.6% / 7.8% |

## Interpretation

1. **GRU clearly wins return** (−357 vs best-PB −455). Not close relative to spread.
   [MEASURED] No PB architectural win here.
2. **The benchmark genuinely requires memory:** GRU reset collapses −357 → −516 (≈floor),
   proving reactive sensory cheating cannot explain GRU's score. The comparison in §13
   of the directive is satisfied.
3. **Puzzle — PB reset asymmetry is weak/inverted:** K=1/K=8/K=16 barely change under reset
   (they were already near floor), while K=32 reset (−390 ± 201) is nominally *better* than
   normal. With σ=200 on a seed-0 checkpoint this is noise, not signal. [INFERRED] The low-K
   PB variants appear to solve the task through something reset-insensitive (likely the
   bounded reflex path + partial cue re-detection), i.e. they may not be using persistent
   thought state much at all at this training budget.
4. **Accuracy ordering replicates from CPU:** PB accuracy rises monotonically with K
   (2.2 → 7.4 → 11.2 → 16.0%) while GRU sits at 4.8%. Same direction as the CPU
   preliminary (K=8 16% there). [INFERRED] High-K PB extracts decision information better
   even while its closed-loop return control is worse. Return vs accuracy dissociation is
   the interesting thread — PB "knows" more than it "does".
5. GRU's Stage survival (50%/40%) ≫ PB's (<10%) — PB variants fail early-stage hazard
   avoidance regardless of final-decision accuracy.

## Cross-campaign picture (both DGX runs today)

- Escalation task: no separation (all within noise); K=32 S2-survival outlier 45%.
- Ephemeral task: GRU wins return decisively; PB wins accuracy monotonically in K.
- Both campaigns: PB persistence ablations show weak causal use of thought state at
  1200–1500 step budgets, unlike earlier Gate-6-era probes where knockout was catastrophic.
  [HYPOTHESIS]: the vectorized core changed the balance — cheaper slots may be getting
  ignored by the aggregator in favor of the reflex path during early training.

## Next actions

- Longer-training probe (3000 steps) on GRU vs K=16/K=32 for the escalation task:
  does the GRU's heavy left tail (median −531 vs mean −421) stabilize, and does PB's
  tighter variance become an advantage with more optimization?
- Then decide: implement stale-state/probability-scramble battery as a standalone
  audited script before any further architecture claims.
