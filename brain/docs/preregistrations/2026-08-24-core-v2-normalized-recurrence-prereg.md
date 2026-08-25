# Preregistration: Stage V2.0c — normalized recurrent state

**Date frozen:** 2026-08-24, before V2.0c GPU execution  
**Status:** FROZEN  
**Predecessor:** V2.0b smoke failed V2-C numerics; its full run is forbidden.

## Hypothesis and research basis

Stage V2.0b proved that direct action evidence fixes V2-A's zero-gradient
decision contract, but exposed exponential state growth in V2-C. The failure
is structurally predicted by independently adding multiple gates and by an
unbounded belief residual.

Stage V2.0c makes two versioned changes:

1. BrainCell gates form one softmax-normalized convex competition among keep,
   accept/react, and revise proposals; the resulting state is normalized per
   sample and per cycle.
2. Belief uses GRU-style interpolation,
   `b_new = (1-g)*b_old + g*tanh(candidate)`, which keeps every coordinate in
   `[-1, 1]` from a zero initial state.

**Falsifiable hypothesis:** these changes will preserve finite gradients and
bounded belief/thought states for both arms across 400 updates, while retaining
the V2.0b direct policy's learning and non-constant held-out behavior.

Primary research basis:

- Ba, Kiros & Hinton, *Layer Normalization* (recurrent hidden-state
  stabilization): https://arxiv.org/abs/1607.06450
- Chung et al., *Empirical Evaluation of Gated Recurrent Neural Networks on
  Sequence Modeling*: https://arxiv.org/abs/1412.3555
- Chen, Pennington & Schoenholz, *Dynamical Isometry and a Mean Field Theory of
  RNNs*: https://proceedings.mlr.press/v80/chen18i.html
- Gu et al., *Improving the Gating Mechanism of Recurrent Neural Networks*:
  https://proceedings.mlr.press/v119/gu20a.html

## Frozen implementation identity

Base Git commit: `106c193`. Frozen working-tree overlay:

| File | SHA-256 |
|---|---|
| `brain/src/irene_brain/v2/config.py` | `e481a230f8a995b227793cf8805ddb86b3287e8441152284396d4b9abddc29df` |
| `brain/src/irene_brain/v2/aggregator.py` | `7670325f0bba1e6e1c47e464bc22087ccf55da86352d62d8a2b314d3fb4d2717` |
| `brain/src/irene_brain/v2/brain_cell.py` | `03cc6c86d3e41f4ef71ec3f89db0cb79a8609818b79e661f5395330a9da0c298` |
| `brain/src/irene_brain/v2/belief_updater.py` | `cae13b9c8da6d0bb33c732de68d0c21ba98bc15619f122b03cea72c345a76318` |
| `brain/scripts/stage_v20_core_v2_deployed_loss_baseline.py` | `ecda5a92204c11bccc339c8e09fb48095693c4bbb16b43518c17ab0da2ad966e` |
| `brain/scripts/stage_v20c_normalized_recurrence.py` | `4c9e5aaa234293803e53294d43805d7e4baf3e8b00ca623a23402d994950b8c6` |
| `brain/scripts/stage_v20c_feedback_ablation_diagnostic.py` | `79af59bd783bc166b5916670735f86eb572de08d0487f7cf6dac5c8a72f59a0e` |
| `brain/tests/test_core_v2_decision_contract.py` | `53729fd580df4cc19e8ef80467d31bc10d8099659866955cd421c31b65210c8d` |

Any byte change requires a new preregistration identifier.

## Local entry gate

Before Spark execution:

- all five deployed-decision contracts pass;
- a 25-tick full V2-C sequence keeps belief `<=1.000001`, thought magnitude
  `<=11`, gradients finite, and optimizer parameters finite for 12 updates;
- the 120-update full diagnostic completes finite; and
- constitution and play-safe regression suites pass.

## Frozen Spark smoke

| Item | Value |
|---|---|
| Script | `stage_v20c_normalized_recurrence.py --smoke` |
| Arms | V2-A normalized; V2-C normalized |
| Seed / updates | `42` / `400` each |
| Bank | seed 42, digest `b3bb5fc33fd5f605` |
| Optimizer | AdamW, lr `5e-4`, weight decay `1e-4`, clip `1.0` |
| Evaluation | 30 episodes/task, base seed `20260822`, deployed argmax |
| Runtime | deterministic, pinned image, no network, bounded container |

For **each** arm, all conditions must pass:

- no non-finite loss, gradient norm, or parameter;
- fixed-probe final loss `<1.45` and `<=90%` of its pre-training value;
- at least two distinct deployed actions across held-out evaluation;
- maximum belief magnitude `<=1.00001`;
- maximum thought magnitude `<=11.0`; and
- final JSON is atomically persisted and parses with strict finite numbers.

Failure stops the stage. The confirmatory run may not start.

## Frozen confirmatory run

Only after smoke PASS: both arms, 6,000 updates, seeds
`{42, 142, 242, 342}`, otherwise identical protocol. Per-arm outcome classes
against frozen Core V1 R=`-0.2130` remain:

- ALPHA: delta `>=+0.05` and sigma `<=0.06`;
- BETA: absolute delta `<=0.02`;
- GAMMA: delta `<=-0.03` (stop);
- otherwise AMBIGUOUS.

This stage tests trainability and held-out decision behavior only. It does not
establish grounded consequence semantics, memory benefit, real-time play, or
human-level game understanding.

