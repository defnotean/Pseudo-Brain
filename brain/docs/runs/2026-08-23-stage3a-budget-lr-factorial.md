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

**Mechanical classification: Mixed / Ambiguous**  
Prereg Outcome C required |24k − 6k| ≤ 0.03 in **both** arms.  
FIXED: Δ = −0.008 (passes). SCHEDULED: Δ = −0.039 (**fails**, |Δ| > 0.03).  
SCHEDULED − FIXED @24k = −0.012 (not B). σ(24k) FIXED = 0.102 > 0.08 but no mean improvement (not classic D).

**Scientific read: consistent with a training-signal/generalization plateau rather than undertraining or LR-schedule limitation.**

| Test | Result |
|---|---|
| More steps improve FIXED? | No: −0.008 |
| More steps improve SCHEDULED? | No — it gets worse: −0.039 |
| Scheduled beats fixed at 24k? | No: −0.012 |
| Budget-limited? | No |
| Optimizer-gated? | No |
| Classic fragile improvement? | No |
| Evidence of training/eval decoupling? | **Very strong** |

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

Interestingly, SCHEDULED **reduces seed variance** (σ 0.102 → 0.046) while
failing to improve mean capability — it stabilizes convergence toward a
solution that still doesn't generalize.

## Conclusions

1. **Core V1 cannot convert additional optimization on the current fixed
   training distribution/objective into greater held-out torture-suite
   capability.** It is not primarily suffering from insufficient optimizer
   steps or the wrong LR schedule.
2. **The current training objective/data distribution permits near-perfect
   fitting without transferable cognitive capability.** The train/eval gap
   is the dominant phenomenon.
3. **Parameter scaling is not the next step** — W-curve was flat/negative
   (Stage E), K-sweep was flat (Phase 2.6), now budget curve shows no
   positive return. All three capacity axes (W, K, budget) are
   flat-to-negative under this recipe.
4. **The binding constraint is the learning signal / data / objective.**
   The only axis that ever moved memory causality in Phase 2 was the
   ideal-evidence probe (external teacher signal). The update rule /
   intake mechanism is the *ultimate* workstream — but first, the cheap
   causal question is whether the problem is the information we're
   teaching with (data/curriculum) or the mechanism that learns it.

## Next Workstream: Stage 3b — Data / Curriculum / Generalization Audit

Before modifying Core V1, the highest-information experiment attacks the
data/generalization hypothesis directly:

**Design:** Frozen Core V1 (W=120 K=32), frozen optimizer (FIXED lr=5e-4),
frozen budget (6k steps for speed), deterministic mode, pinned seeds.
Vary ONLY:

- **ARM A (control):** Current finite pinned bank (repeating 420 episodes).
- **ARM B:** Deterministically generated **non-repeating** training stream
  (same procedural tasks, fresh episodes every step — vastly more unique
  experiences). Both arms fully reproducible.

**Questions:**
- Does ARM B stop training loss from trivially collapsing?
- Does held-out torture lift improve?
- Does the train/eval gap shrink?

If yes → data diversity / memorization was the bottleneck.
If no (still fits training without transferring) → move aggressively to
Stage 3c: evidence-intake / learning-objective mechanisms.

This is cleaner than immediately modifying Core V1, and scientifically
identifies whether the ceiling is in the *experience* or the *learning
mechanism*.

## Provenance

All 8 runs carry full provenance: bank digest, init param digest, seeds,
deterministic flags, torch 2.13.0+cu130, CUDA 13.0, cuDNN 9200, Python
3.12.3. Artifact: `runs/s3a_results.json` on Spark.