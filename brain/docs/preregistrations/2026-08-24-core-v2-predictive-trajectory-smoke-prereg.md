# Preregistration: Stage V2.1 predictive trajectory smoke

**Date frozen:** 2026-08-24, before any accelerator run  
**Status:** FROZEN after CPU-only integration; gates may not change after launch

## Question

Can the corrected Core V2 training path learn grounded one-tick consequences
from causal maze-chase trajectories without sacrificing the deployed turn
policy, and can the experiment distinguish predictive supervision from actual
prediction-error feedback?

This is a development smoke, not a gameplay or breakthrough claim. A pass
licenses a multi-seed confirmation and causal feedback intervention. It does
not establish useful planning, memory, adaptation, or human-speed play.

## Frozen dataset

- Generator: repository `MazeChaseBatchSource`, pixel/planner teacher only.
- Split sizes: 96 train / 32 validation / 32 unopened test sequences.
- Sequence length 16, burn-in 4, batch size 8.
- Split-local seed offset `8,388,608`.
- Five direct-chasing ghosts, ghost period 1; all other world knobs default.
- Manifest SHA-256:
  `7c7c12e29011f5bd5d7436fc5aa31c42ccf983d8abd876f18991af4926f3f72b`.
- Pre-launch coverage audit over optimized ticks: 63 train / 23 validation /
  19 test `caught` events. The test split remains unused by this smoke.
- Validation contains 180 genuine teacher direction changes in 384 optimized
  transitions and zero applied-control/requested-control mismatches.
- Cognition receives current pixels and the observation's previous physically
  applied control. The current transition's applied control conditions only
  the factual world-model prediction; teacher requested action, reward,
  `caught` hazard, and next pixels remain labels. This prevents action-label
  leakage into the deployed decision while aligning the predicted future with
  the action that actually caused it.
- At tick `t`, prediction-error input receives only reward/hazard from
  transition `t-1`; current-transition targets never enter cognition.

## Frozen architecture and training

- Seed 42 for all arms; deterministic PyTorch/CUDA mode.
- W=120, K=32 thoughtlets, C=3 cycles, five actions idle/W/A/S/D.
- Exact footprint: 730,484 total parameters, 695,420 trainable, and 35,064
  frozen EMA-target parameters.
- Direct mean deployed logits, normalized-mixture BrainCell, convex-gated
  belief, sigmoid probability hazard head.
- Same CONFIG-B predictive module inventory in every arm, including target
  encoder and prediction-error module.
- Four epochs = 48 optimizer steps per arm.
- AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Gradient clipping at norm 1 after each full trajectory batch.
- EMA target encoder updated exactly once after each optimizer step.
- Decision loss uses fixed-exposure normalization. Direction holds receive
  weight 0.1 and turns receive weight 1.0; weights are not renormalized inside
  each batch.
- Predictive weights: next latent 0.50, reward 0.25, hazard 0.25.
- Script: `brain/scripts/stage_v21_predictive_trajectory_smoke.py`.
- Each completed arm writes an atomic checkpoint containing model state,
  architecture config, feature flags, seed, and dataset manifest; its SHA-256
  is recorded in the result JSON before any closed-loop use.
- Frozen script SHA-256:
  `6a019f4155e5e0d1e5f02a3007644881f522e671663cca572c86e616d7b7838f`.
- Frozen trajectory-objective SHA-256:
  `144f596383d27e70b72b11a4be49ef5b2e0bd31158786488a855c3ee4ff39520`.

## Frozen arms

1. `decision_only_zero_pe`: all predictive loss weights zero; prediction-error
   signal explicitly zeroed. This retains the same trainable architecture.
2. `predictive_zero_pe`: all three grounded predictive losses active;
   prediction-error signal zeroed. This isolates auxiliary representation
   learning.
3. `predictive_normal_pe`: predictive losses active and normal next-tick
   prediction-error feedback. This tests the complete V2.1 path.

All arms must have the same initial parameter digest. Any mismatch is a smoke
failure.

## Frozen measurements

- Validation deployed action accuracy and turn-only accuracy.
- Next-latent MSE and cosine similarity against the frozen EMA encoder.
- Reward MSE/MAE.
- Hazard binary cross-entropy and Brier score.
- Validation turn count and hazard-positive count.
- Maximum absolute belief and thought values.
- Pre-clip gradient norm, parameters, runtime, device, seeds, versions, and
  dataset manifest.

Evaluation uses a fixed isolated RNG seed and restores the training RNG state,
so validation cannot alter the training-noise schedule.

## Frozen smoke gate

The smoke passes only if all conditions hold:

1. Every arm finishes, all values/parameters are finite, and all initial
   parameter digests match.
2. Validation has at least one turn and at least 10 hazard-positive examples.
3. Across both predictive arms, final validation next-latent, reward, and
   hazard losses are each at most 85% of their own initial values, and mean
   predicted hazard on caught transitions exceeds the non-caught mean by at
   least 0.02. This blocks a constant low-hazard predictor from passing.
4. `predictive_normal_pe` final turn accuracy is no more than 0.10 below the
   decision-only control.
5. Maximum absolute belief is at most `1.00001`; maximum absolute thought is
   at most `16.0`.

No test-set result is consulted for the gate.

## Outcomes

- **PASS:** predictive heads demonstrably learn under the causal timestamp
  contract without catastrophic deployed-policy regression. Next preregister
  multi-seed validation plus paired normal/zero/scrambled feedback causality.
- **FAIL — prediction:** one or more predictive losses misses the 15% reduction.
  Diagnose target scale, rare-event weighting, gradient shares, and slot
  assignment locally; do not extend training blindly.
- **FAIL — policy:** prediction learns but turn accuracy regresses by more than
  0.10. Measure gradient conflict and test a fixed auxiliary-weight schedule.
- **FAIL — numerical/coverage:** stop and repair the contract or dataset before
  any rerun.
