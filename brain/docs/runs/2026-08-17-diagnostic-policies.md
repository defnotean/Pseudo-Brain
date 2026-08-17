# Diagnostic policies for the closed-loop evaluator (2026-08-17)

Status: implementation only. These are PLAN.md §28 diagnostic reference rows
for the closed-loop moving-shapes evaluator, not qualified results; no model
checkpoint is measured against them yet.

## What was added

1. **Generic decision-source episode core** in
   `src/irene_brain/evaluation/closed_loop_play.py`. The episode loop (manual
   clock, `ContinuousDriver`, deadline-bound envelopes, rejection accounting)
   is now `_run_episode_core`, parameterized by a decision callback. The model
   path (`run_closed_loop_episode`) is a thin wrapper and its behavior is
   unchanged — the full pre-refactor test suite passes byte-identically. New
   `run_policy_closed_loop_episode` runs any non-model policy through the same
   mechanics, and `control_audit_stats` derives the decode-style audit fields
   from an already-formed `GenericControl`, so policy and model episode
   reports are column-comparable.
2. **`src/irene_brain/evaluation/diagnostic_policies.py`** — the four §28
   diagnostic decision sources behind one contract (`identity`,
   `uses_privileged_state`, `reset(episode_seed)`, `act(observation)`):
   - `diagnostic.noop.v1` (§28.1 floor): never presses anything.
   - `diagnostic.random_movement.v1` (§28.1 floor): a uniform W/A/S/D subset
     per decision from a per-episode SplitMix64 (the environment's own
     generator, reused so determinism semantics match the world).
   - `diagnostic.scripted_chase.v1` (§28.2): pixel-only greedy target chaser
     reading just the public render-contract colors — proof that a
     procedural teacher needs nothing a model would not see. When the player
     stands on the target the target pixel is occluded for one observation;
     the policy holds still for that tick.
   - `diagnostic.oracle_privileged.v1` (§28.15, diagnosis only): reads
     simulator internals, predicts hazard paths with the exact bounce rule,
     and takes the safe minimum-distance move using the environment's own
     swept-path collision test. Its outputs must never be routed into a model
     input or training target.
   `evaluate_diagnostic_policy_suite` runs a policy tuple over every
   registered seed and returns one canonical `ClosedLoopPlayReport` per
   policy (identities must be unique).

## Bug found and fixed during bring-up

The movement-key table carried over into the closed-loop evaluator had W and
S swapped (`22` is S, `26` is W in the HID usage table and in
`MovingShapesEnv._KEY_BITS`). The opposite-conflict pairs were unaffected
(both members of each pair were present), but the scripted chaser moved away
from targets vertically, which the new episode tests caught immediately.
`_MOVEMENT_KEYS` and the policy key constants now match the environment.

## Verification

New `tests/test_diagnostic_policies.py` (17 tests, torch-free — they also run
in the no-ML play-safe environment): policy contract and identity checks,
`control_audit_stats` equivalence with decode semantics, no-op floor
(60/60 decisions, mask 0 only, zero targets), scripted chaser collects from
pixels only, oracle collects with zero collisions on the pinned test seeds,
per-seed determinism and seed sensitivity for the random policy, suite
ordering (oracle strictly above the no-op floor), suite determinism
(identical SHA-256 across runs), and suite input validation. The full
play-safe suite exits 0, including the unchanged `test_closed_loop_play` and
`test_matched_baselines` modules.

Reference rows (exploratory, unregistered seeds 5/9/13, 600 ticks, 3 hazards,
zero latency, one 60 Hz decision per tick):

| Policy | Targets | Collisions | Reward |
|---|---|---|---|
| `diagnostic.noop.v1` | 0 | 0 | 0 |
| `diagnostic.random_movement.v1` | 4 | 48 | -44 |
| `diagnostic.scripted_chase.v1` | 237 | 51 | +186 |
| `diagnostic.oracle_privileged.v1` | 236 | 0 | +236 |

The spread is the expected diagnostic signature: the no-op floor is dead
flat, random movement occasionally stumbles onto targets while hitting
hazards, the pixel-only chaser solves approach but not avoidance, and the
oracle converts the same approach into a collision-free ceiling. Any future
qualified checkpoint's closed-loop rows now have a floor, a reference, and a
ceiling to sit between.
