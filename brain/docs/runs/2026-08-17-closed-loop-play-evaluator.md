# Closed-loop moving-shapes play evaluator (2026-08-17)

Status: implementation only. No checkpoint is qualified, so nothing here is a
"the model plays" claim; this change builds the instrument that will make that
claim measurable once a qualified checkpoint exists. It covers roadmap item 6
(ROADMAP_TO_PACMAN.md §6) and the evaluation half of §4 step 2.

## What was added

New module `brain/src/irene_brain/evaluation/closed_loop_play.py`:

- **`ClosedLoopPlayConfig`** — frozen knobs for one campaign: a non-empty,
  unique tuple of episode seeds (required), `max_ticks` (default 600),
  `hazard_count` (default 3), `tick_period_ns` (default 16,666,667 — one 60 Hz
  frame), `decision_interval_ns`, simulated `inference_latency_ns`,
  `submit_deadline_slack_ns`, and `expiry_slack_ns`. All fields are
  range-validated plain integers.
- **`decode_closed_loop_control`** — decodes the packed 296-entry
  `button_logits` vector through `BUTTON_TARGET_INDICES` exactly like the
  open-loop RCQ decode (activation strictly above a zero logit), and the
  eleven continuous channels. It emits a `GenericControl` plus audit stats:
  movement mask, opposite-direction conflict flag, non-movement key
  activations, mouse/gamepad counts, and continuous deadzone violations —
  the two action-path failure modes from the RCQ-v2 post-mortem stay visible
  in closed-loop evidence.
- **`run_closed_loop_episode`** — one deterministic episode. The model drives
  `MovingShapesEnv` through `runtime/continuous.py`'s `ContinuousDriver` on a
  `ManualClock` at 1 GHz tick frequency: poll latest observation, infer,
  advance the clock by the declared simulated inference latency (the world
  keeps running while the model "computes"), then submit a deadline-bound
  `ActionEnvelope`. Stale-frame rejections are counted by reason;
  `PostAdvanceRuntimeError` ends the episode. Brain state is threaded across
  decisions with `deterministic_eval_thought_noise`, matching the open-loop
  evaluator's call pattern.
- **`evaluate_closed_loop_play` / `ClosedLoopPlayReport`** — runs every
  registered seed on CPU in eval mode (training mode is restored afterward)
  and assembles a canonical evidence record with sorted keys, totals, and a
  SHA-256 over the canonical JSON, so closed-loop results can be cited and
  compared exactly like gate reports.

Everything is simulated and deterministic: no wall-clock latency is reported
as a physical measurement, no screen capture or HID output occurs, and the
model only ever sees canonical observations. The module is exploratory
evidence tooling, not a qualification component; Phase 0B physical latency
evidence remains a separate, explicitly armed path.

## Verification

New `brain/tests/test_closed_loop_play.py` (14 tests):

- Decode unit tests: strict-above-zero threshold, W/S opposite-conflict flag,
  movement-mask packing, deadzone counting at the exact ±0.05 boundary,
  non-movement key and mouse-button routing, length validation.
- Config validation: seeds must be a non-empty unique tuple; knob ranges.
- End-to-end on an untrained smoke model: episodes complete at `max_ticks`
  with finite outputs; a zero-latency run has zero rejections; an injected
  latency of five tick periods produces stale-frame rejections
  ("action does not target the newest observation") and dropped observations;
  two identical evaluations produce byte-identical canonical JSON and SHA-256;
  training mode is restored after evaluation.

Local gates: the new module passes standalone, the matched-baseline suite
(10 tests, including the live manifest digest `78ba9cfc…`) passes unchanged —
`closed_loop_play.py` is not a manifest implementation file — and the full
play-safe suite exits 0.

## What this unblocks

Once the RCQ-v3 ceremony (owner-run) produces a qualified checkpoint, the
first honest closed-loop measurement is: load the checkpoint, run
`evaluate_closed_loop_play` across a registered seed set at 60 Hz with the
measured Phase 0B inference latency, and compare closed-loop task outcomes
and deadline behavior against the open-loop gate rows. Remaining roadmap §6
gaps around this item are live-play latency evidence records (extending
`evaluation/latency_evidence.py` from benchmarks to play) and Phase 0B
harness integration.
