# Phase 2.7 Stage 3b: Data / Generalization Audit (2026-08-23)

**Design:** Frozen Core V1 (W=120 K=32 C=3). FIXED lr=5e-4. 6k steps.
Deterministic mode. 4 seeds {42,142,242,342}. Full provenance per run.
ARM A = finite pinned bank (420 eps, repeating, digest `b3bb5fc33fd5f605`).
ARM B = deterministic non-repeating procedural stream (fresh episode every step,
seeding `base_seed + step`). Identical eval (30 eps/task, eval_seed=20260822).

## Results [MEASURED]

| Arm | Seed | Mean Lift | Final Loss | Wall (s) | Above+2% |
|-----|------|-----------|------------|----------|----------|
| A finite | 42 | −0.2073 | 1.0144 | 727s | 2 |
| A finite | 142 | −0.2207 | 1.0162 | 735s | 1 |
| A finite | 242 | −0.2193 | 1.0152 | 735s | 1 |
| A finite | 342 | −0.2047 | 1.0119 | 728s | 2 |
| B fresh | 42 | −0.2287 | 2.2161 | 9989s | 0 |
| B fresh | 142 | **−0.0460** | 2.1353 | 10112s | 5 |
| B fresh | 242 | −0.2540 | 2.0949 | 9953s | 1 |
| B fresh | 342 | −0.3553 | 0.9596 | 9849s | 1 |

**ARM A**: mean lift = **−0.2130** ± 0.0071; mean loss = 1.0144
**ARM B**: mean lift = **−0.2210** ± 0.1116; mean loss = 1.8515

Δ (B−A) = **−0.0080** (|Δ| = 0.008)
Fresh-stream wall time ≈ **13.6×** ARM A (9976s vs 732s)

## Frozen-Gate Classification [MEASURED]

Preregistered gates:
- **DATA**: ARM B mean lift ≥ ARM A + 0.03 AND train/eval gap smaller
- **MECHANISM**: |ARM B − ARM A| ≤ 0.02 AND ARM B training loss still collapses
- Otherwise: **AMBIGUOUS**

Check:
- DATA: Δ = −0.008 ≥ +0.03? **NO** → DATA rejected at this budget
- MECHANISM part 1: |Δ| = 0.008 ≤ 0.02? **YES**
- MECHANISM part 2: ARM B loss collapses? ARM B loss ≈ **1.85** (HIGHER than
  ARM A's ~1.01, not collapsed) → **NO**

→ **Classification: AMBIGUOUS (strict gate), strongly MECHANISM-leaning.**

The MECHANISM gate was explicitly constructed to require loss collapse as the
second condition. ARM B did NOT achieve loss collapse (it optimized
substantially less completely than ARM A). Therefore the frozen gate cannot
declare MECHANISM. Two explanations remain alive:

**(A)** Learning mechanism/objective is wrong (fresh data would not help even
with matched training fit).
**(B)** Fresh diverse training is useful in principle, but 6k updates are far
too few to learn the harder non-repeating distribution — ARM B was simply
undertrained relative to ARM A.

## What is established [MEASURED]

1. Fresh non-repeating experience did **not** show mean torture-lift improvement
   at 6k steps: finite −0.213, fresh −0.221, Δ = −0.008.
2. Fresh-stream training is much **less stable**: σ 0.0071 → 0.1116 (16×).
3. Fresh-stream optimization is **less complete**: finite loss ≈ 1.01,
   fresh loss ≈ 1.85.
4. Current implementation costs **~13.6×** more wall time for no measured gain.

## What cannot be claimed yet

- NOT "the bottleneck is the learning mechanism, not data diversity."
- ARM B did not fit its training distribution as well as ARM A, so the
  comparison is confounded by optimization completeness, not just held-out
  generalization.
- Seed 142 (−0.046) is a **strong relative outlier**, still negative under the
  metric — NOT a positive-lift run. With only 4 seeds it is undecided whether
  this is noise or a training basin the other seeds missed.

## Smallest discriminating follow-up (preregister before running)

Train ARM B to a **matched training-loss / exposure criterion** (e.g. continue
fresh-stream updates until loss ≈ 1.0, matching ARM A's fit), then compare
held-out lift:

- If fresh data reaches comparable training fit and **still** doesn't generalize
  → much stronger MECHANISM evidence (A).
- If fresh data at matched fit **outperforms** → data diversity mattered; 6k was
  simply undertrained for the harder stream (B).

This requires Training Engine V2 (the 13.6× cost makes matched-fit runs
prohibitive under the current single-threaded generator).

## Critical code-level observation (drives Stage 3c0)

While reviewing the training/eval paths, a potential **learning-contract defect**
stands out independent of the data question:

- The multitask trainer applies the same final action target to **every** K=32
  thoughtlet (replicated per-slot CE on `action_logits`).
- The deployed decision uses `argmax(out.action_dist)` — the **aggregated**
  output of consequence utilities, branch probabilities, and confidence, which
  are NOT on the supervised path.
- This may explain why Stage 3a training loss → 0.0 but held-out stayed flat:
  the per-slot classifiers can be driven to the target while the consequence
  heads (reward/hazard/confidence/branch_prob) receive little/no gradient,
  leaving the aggregator with untrained inputs.

This is a higher-priority hypothesis than data diversity alone. Stage 3c0
(Learning Contract Audit) must run **before** any expensive fresh-stream rerun.

## Provenance

All 8 runs carry full provenance: bank digest (ARM A), init param digest,
seeds, det flags, torch 2.13.0+cu130, CUDA 13.0, glibc. Artifact:
`runs/s3b_results.json` on Spark. ARM B carries `stream_seeding: base_seed+step`.
