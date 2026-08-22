# Phase 2.5: pre-collapse trajectory analysis (discovery set, 2026-08-22)

**Data:** `runs/mechanism-discovery-v1/training_trajectories.jsonl` (20 runs × 30
telemetry points; cheap held-out probe every 100 steps, never used for training).
**Discovery-set replication confirmed first:** GRU 7/10 collapsed, PB K=32 3/10
collapsed — matches the reliability result with independent re-training.

## Per-seed trajectory summary [MEASURED]

GRU (7 collapsed): 5 of 7 collapsed seeds were **healthy at step ≤300 and collapsed
by step ~600–2000** (e.g. seed 42: probe return −62 at step 200 → final −535; seed
842: +0.5 at step 300 → −521.5). One seed (342) never got healthy at all.
Peak grad norms in collapsed GRU seeds reach 8.7–26.8.

PB K=32 (3 collapsed): the three PB collapses are **later and slower** (collapse_step
1500–2600 vs GRU's typical 600–1000), and two of them (142, 242) had already reached
probe returns of −51 / −294 before degrading. Surviving PB seeds include strong late
peaks (+9.5 @ 2200, −7.0 @ 1500). PB peak grad norms run lower overall (4.4–18.9).

Stage-0 probability mass (`p0_max`, `p0_committed_frac`) stays ≈0.125 / 0.00 in ALL
runs both architectures, early and late — **no premature probability concentration
is visible in this telemetry**. The collapse signature is behavioral (WAIT survival
loss → catastrophic action loop), not a stage-0 calibration failure.

## [INFERRED] mechanism-shaped observations

1. **GRU collapse is an early-window bifurcation.** By ~step 300 a GRU seed is either
   climbing or already doomed; five of seven doomed seeds had *reached near-ceiling*
   probes (−62 to +0.5) before losing it. That is consistent with a fragile
   monolithic state that learns the task then overwrites/compresses the needed
   hypothesis structure during continued optimization.
2. **PB fails differently:** when PB collapses, it does so later and from a lower
   peak — and 7/10 PB seeds keep improving through step 1400–2400 while all surviving-
   then-doomed GRU seeds degrade by ≤1100. High-K slots appear to preserve learned
   structure longer under the same optimizer pressure.
3. **Gradient-norm spikes co-occur with GRU collapse windows** (seed 842: 26.8 peak;
   seed 342: 18.0, never healthy) but this telemetry is too coarse (100-step bins)
   to establish causality. [HYPOTHESIS] worth instrumenting at finer granularity.

These are discovery-set observations intended to shape the mechanistic battery —
NOT yet claims. The battery must show a causal difference between surviving and
collapsed checkpoints under identical interventions.

## Next: hostile battery design (frozen before running)

For each preserved checkpoint — surviving-PB (742, 842, 442), collapsed-PB (142,
242, 542), surviving-GRU (442, 642, 242), collapsed-GRU (42, 142, 842) — run the
full held-out eval under: normal / reset-every-frame / stale D∈{1,5,20} /
probability scramble / binding scramble / permutation (PB only) / consequence
scramble. Compare degradation profiles between cell types. Prediction if the
collapse-resistance story is right: surviving-PB checkpoints show large binding/
stale sensitivity (hypotheses actively maintained); collapsed checkpoints of either
architecture look alike (reflex-only behavior).
