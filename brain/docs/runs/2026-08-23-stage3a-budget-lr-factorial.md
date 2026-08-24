# Phase 2.7 Stage 3a: Training-budget × LR-schedule factorial (2026-08-23)

**Design:** Frozen Core V1 (W=120 K=32 C=3). Pinned banks (digest
`b3bb5fc33fd5f605`). Deterministic mode. 4 seeds {42,142,242,342}. One
24k-step run per (schedule, seed) with checkpoints at 6k/12k/24k.
Arms: FIXED lr=5e-4 constant vs SCHEDULED (500-step warmup → cosine to
5e-5). Provenance block per run.

## Results [MEASURED]

| Schedule | 6k lift | 12k lift | 24k lift | Δ(24k−6k) | σ(24k) | above+2%@24k |
|---|---|---|---|---|---|---|
| **FIXED** | −0.213 | −0.236 | **−0.221** | **−0.008** | **0.102** | 6/24 |
| **SCHEDULED** | −0.194 | −0.201 | **−0.233** | **−0.039** | **0.046** | 8/24 |

SCHEDULED − FIXED @24k = −0.012 (FIXED marginally better).

## Outcome Classification (per prereg)

**Primary: Outcome C — Plateau**  
Neither schedule benefits from 4× more steps (|Δ| ≤ 0.03 for both).
The current Core V1 is **not undertrained**; it has hit a capability
plateau under this objective and data.

**Secondary observation:** FIXED σ(24k)=0.102 exceeds D-threshold (0.08)
but mean does not improve — "fragile without improvement."

## Critical Diagnostic: SCHEDULED overfits perfectly, generalizes zero

Training loss trajectories (seed 42 representative):

```
FIXED:    1.6 → 1.1 → 2.6 → 0.9 → 1.0 → ... → ~1.0 (oscillates)
SCHEDULED: 1.6 → 1.1 → 2.6 → 0.9 → 1.0 → 0.19 → 0.04 → 0.003 → 0.0001 → 0.0 by step 7k
```

SCHEDULED drives cross-entropy to **literal 0.0** by ~7k steps — zero
training error on the episode bank. Yet torture-suite lift at 24k is
−0.249 (worse than FIXED's −0.221). The model memorizes the training
episodes without acquiring the underlying sensorimotor policies that
transfer to held-out evaluation.

This is the **evidence-intake deficit** in its purest form: optimization
works, but the training signal does not constrain the right hypotheses.

## Conclusions

1. **Core V1 is at a capability ceiling** for the torture suite under the
   current recipe. Budget scaling (steps, LR schedule) is exhausted.
2. **Parameter scaling is definitively off the table** — W-curve was
   flat/negative (Stage E), K-sweep was flat (Phase 2.6), now budget
   curve is flat. All three capacity axes closed.
3. **The binding constraint is the learning signal** — the only axis that
   ever moved memory causality in Phase 2 was the ideal-evidence probe
   (external teacher signal). The update rule / intake mechanism is the
   next workstream.
4. Data curriculum (3b) is still worth one diagnostic pass (does task
   mixing change the plateau?), but the overfitting result strongly
   suggests the signal itself is the problem, not its presentation.

## Provenance

All 8 runs carry full provenance: bank digest, init param digest, seeds,
deterministic flags, torch 2.13.0+cu130, CUDA 13.0, cuDNN 9200, Python
3.12.3. Artifact: `runs/s3a_results.json` on Spark.