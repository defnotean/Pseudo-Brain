# Model-side maze_chase closed-loop integration pin (2026-08-17)

Status: infrastructure. Test-only change; no model checkpoint is measured.

## What was added

One test in `tests/test_closed_loop_play.py`:
`test_untrained_model_plays_maze_chase_without_rejections`. The untrained
smoke-scale thought-field model drives the canonical maze_chase slot
through `evaluate_closed_loop_play`'s `environment_factory` hook — two
episodes, 24 ticks each, every decision submitted and accepted.

## Why it matters

Phase 4's arcade proof needs a *trained model* playing maze_chase through
this evaluator. Until now only moving_shapes had exercised the model-side
path; every maze_chase row came from scripted policies. The pin retires
the integration risk early — world-generic observation encoding, control
decoding, and deadline mechanics all accept the maze slot — so the first
trained maze_chase evaluation will measure the model, not plumbing.

The full play-safe suite exits 0.
