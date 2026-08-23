# Phase 2.7 Stage I: multi-seed/episode batching throughput experiment (2026-08-22)

**Setup:** frozen Core V1, GB10, 200 train steps per config.

## Results [MEASURED]

| Config | ms/step | Episodes/step | Aggregate throughput vs solo |
|---|---|---|---|
| Solo (1 ep/step) | 80.9 | 1 | 1.0× |
| Episode-batch 4 | 95.5 | 4 | **3.4×** |
| Episode-batch 8 | 100.2 | 8 | **6.4×** |
| Episode-batch 16 | 116.2 | 16 | **11.0×** |

(Note: the script's printed "speedup" metric divided total walls at equal step
counts and understated the gain; the correct per-episode comparison is above.)

## Interpretation [MEASURED + INFERRED]

- Fixed kernel-launch overhead dominates the 80 ms solo step: adding 15× more
  episodes costs only 44% more wall time. The GPU is massively underutilized
  by the batch-1 recurrent loop, as the profile predicted.
- Sub-linear scaling (11× at B=16, not 16×) indicates we are approaching the
  point where real compute and attention-memory traffic start to matter.

## SCIENTIFIC CAVEAT — this is a candidate engine, not a verified one

The batched variant differs from the canonical recipe in TWO ways:
1. Loss is averaged over all time steps of the episode (canonical: final
   decision frame only).
2. Padded frames (zeros) participate in that loss — an artifact.

Both change the training signal. Per the freeze discipline, this engine may
NOT replace the canonical trainer without a preregistered equivalence test:
same seeds, matched episode budget, torture-suite lift within noise of the
canonical recipe, plus the padding-loss bug removed (mask padded steps).

## Next (Stage J/K, ordered)

1. Fix padding mask + final-frame loss in the batched engine.
2. Preregister + run the equivalence check (canonical vs batched, 3 seeds).
3. On equivalence: multi-seed batching (M models via stacked state) on top of
   episode batching for the 10+-seed campaigns Phase 2.7 needs.
