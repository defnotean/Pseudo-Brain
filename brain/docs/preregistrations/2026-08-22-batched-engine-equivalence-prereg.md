# Preregistration: batched training engine equivalence (frozen before run)

**Date:** 2026-08-22. **Purpose:** authorize the batched engine (11× candidate
throughput) as a replacement for the canonical solo trainer in Phase 2.7
campaigns, WITHOUT changing training science.

## Required engine fixes before the test

1. Loss = final decision frame ONLY (per episode), not averaged over frames.
2. Padded frames masked out of BOTH forward gradient contribution and loss
   (state must not update on padding; loss must ignore padded members).
3. Per-member optimizer state (each episode's model update independent).

Note on semantics: canonical trainer processes ONE episode's frames
sequentially with state carry and trains on the final frame. The batched
engine processes B episodes in parallel — each member's state carries across
its own frames. Members are independent; batching changes ONLY execution, not
the per-episode computation, once fixes 1–2 land.

## Equivalence test (frozen)

- Arms: canonical solo trainer vs corrected batched engine (B=16).
- 3 seeds {42,142,242}, 6000 steps canonical-equivalent episodes, identical
  data order, optimizer, LR.
- Metric: torture-suite mean lift per arm per seed (locked eval protocol).
- PASS iff: per-seed |lift_batched − lift_solo| ≤ 0.03 for ≥ 2/3 seeds AND
  mean |Δ| ≤ 0.02. (Tolerances reflect measured seed noise σ≈0.02-0.04.)
- Latency recorded for both; expected ~10× wall reduction at B=16.

## On PASS

Batched engine becomes the Phase 2.7 campaign trainer. Canonical solo trainer
remains in-repo as reference implementation and for debugging.

## On FAIL

Record divergence, diagnose (padding mask correctness, state-carry semantics),
one fix iteration allowed, re-run. If still failing, batched engine stays
quarantined as engineering-only.
