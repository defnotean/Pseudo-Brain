# Phase 2.6: torture-suite baseline LOCKED (2026-08-22)

**Core:** tag `phase26-baseline-core` (84009fd). All numbers reproducible via
`brain/scripts/run_torture_suite.py` + `brain/scripts/train_torture_multitask.py`
(25 tasks × 40 episodes, fixed seeds). Data: Spark `runs/torture-baseline/`.

## Canonical baseline table [MEASURED]

| Configuration | Mean lift | Tasks >+2% | Wait@decision |
|---|---|---|---|
| PB K=32 fresh | −0.246 | 1/24 | 35–60% |
| GRU fresh | −0.333 | 2/24 | ~0% |
| PB K=32 escalation ckpt (742) zero-shot | −0.380 | 0/24 | 70–100% |
| **PB K=32 multitask-trained (4000 steps)** | **−0.245** | **0/25** | — |
| **GRU multitask-trained (4000 steps)** | **−0.042** | **8/25** | — |

Per-task highlights (trained): GRU solves t03 three-cues at 100% when fresh
(seed artifact) and reaches +9% lift on goal-hold, anytime-commit; near-chance on
most evidence tasks. PB trained stays ≈15–20 points below chance on most tasks —
its slot-averaged decision head has not learned the answer-token mapping under
this recipe.

## What this baseline establishes

1. **The suite discriminates.** Fresh vs trained differ hugely for GRU
   (−0.333 → −0.042), so the metric detects learning. The two architectures
   produce clearly different signatures (PB idles/wait-heavy; GRU commits).
2. **Both architectures are far from solving the suite** after a light recipe —
   exactly what makes it useful as a Core V1 improvement yardstick.
3. **PB's trained baseline is anomalously bad relative to its Phase 2 record.**
   [HYPOTHESIS] the slot-averaged `action_logits.mean(dim=1)` training head used
   here is not how Phase 2 tasks trained decisions (they used per-slot CE + the
   consequence aggregator). This is a *training-recipe* gap, not necessarily an
   architecture gap — but it must be flagged honestly in the baseline.

## Baseline lock decision

Locked as-is with the caveat above recorded: any Core V1 change must show
improvement on this exact protocol (same seeds/tasks/steps) AND the recipe
discrepancy for PB's decision head is itself a Phase 2.6 fix candidate
(audit item #1 follow-up).

## Next per approved order

Episodic memory v0 → causal-use proof → dynamic-K/lifecycle v0 → BrainCell
candidates → adaptive-gate wire-or-descope → constitution CI → freeze review.
