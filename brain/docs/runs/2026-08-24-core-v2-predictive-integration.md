# Core V2.1 predictive training and live-loop integration

**Date:** 2026-08-24  
**Status:** CPU integration passed; preregistered Spark smoke not yet launched

## Contract defects found and fixed

The pre-existing V2 predictive skeleton could not train its world model:

1. The config declared `next_weight`, while the loss looked up
   `next_latent_weight` and silently applied weight zero.
2. `StepOutput.pending_prediction` was detached before the same-tick auxiliary
   loss read it, so no world-model gradient existed.
3. The EMA target encoder existed but was never connected to Core V2.
4. Hazard was an unbounded scalar despite downstream probability semantics.
5. The first trajectory objective passed the current transition's applied
   action as `prev_action`, leaking the teacher action into cognition.
6. The world model conditioned on its own proposal while the target next frame
   was caused by the recorded physical action.

The corrected contract keeps a live same-tick prediction for supervision,
stores a detached recurrent copy, initializes and explicitly advances the EMA
target, provides a versioned sigmoid hazard path, feeds cognition only the
observation's previous control, and separately conditions the factual world
prediction on the action that caused the transition.

## Dataset audit

The first three-ghost/period-2 candidate had only 6/1/1 train/validation/test
hazard events and was rejected before registration. The selected five-ghost,
period-1 direct-chase split has 63/23/19 `caught` events and manifest
`7c7c12e29011f5bd5d7436fc5aa31c42ccf983d8abd876f18991af4926f3f72b`.
Validation has 180 turns in 384 optimized transitions and zero
applied/requested-control mismatches.

## CPU integration

The three-arm mini run completed one optimizer step per arm through causal
unroll, deploy-aligned decision loss, reward/hazard/next-latent losses,
backpropagation, norm clipping, optimizer update, EMA update, validation,
provenance, atomic JSON, and atomic checkpoint writing. All three checkpoint
hashes matched their files. The predictive-normal checkpoint reloaded into
`CoreV2MazePolicy` and produced a valid action.

The one-batch checkpoint then controlled a 32-tick closed-loop maze episode:

- 32/32 decisions submitted; zero rejected;
- zero opposite-direction conflicts or non-movement activations;
- 2 pellets and 3 catches, reward −28.

The behavior score is intentionally not evidence of competence: the checkpoint
had only one optimizer step. This run establishes mechanical closure from
pixels through recurrent state/action and back to changed pixels.

## Single-thread latency

Checkpoint: mini `predictive_normal_pe`  
Device: local CPU, one PyTorch thread, CUDA hidden  
Protocol: 20 warm-up + 200 streaming recurrent ticks, 32×32 RGB, batch 1

| metric | milliseconds |
|---|---:|
| mean | 2.2663 |
| p50 | 2.2376 |
| p95 | 2.4258 |
| p99 | 2.6017 |
| max | 2.7032 |

This is model inference only and is below the 16.67 ms 60 Hz frame interval.

The diagnostic was then extended to include RGB conversion, recurrent policy
state, exclusive idle/W/A/S/D decoding, control construction, and one
in-process simulated maze step. On a separate 20 warm-up + 200 measured-tick
run, the full in-process loop measured 2.5835 ms mean, 2.9284 ms p95,
3.2426 ms p99, and 3.5196 ms maximum. Policy act alone measured 2.4133 ms
mean; the simulated environment step measured 0.1701 ms mean.

Neither measurement is physical end-to-end latency: real screen capture,
serialization/network, OS scheduling, HID submission, game scheduling, and
visible display effect must all be measured separately.

## Verification

- 30 focused V2 contracts pass after the live adapter was added.
- The final full repository regression after the adapter and artifact verifier
  additions passed 779 tests with 2 expected skips in 100.370 seconds.
- The fail-closed artifact verifier accepts the trusted mini artifact and its
  tamper contract rejects a checkpoint whose bytes no longer match its record.
- Preregistration:
  `brain/docs/preregistrations/2026-08-24-core-v2-predictive-trajectory-smoke-prereg.md`.
