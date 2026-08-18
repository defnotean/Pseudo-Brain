# Actuation-aware planner compensation (2026-08-17)

Status: exploratory evidence tooling. Diagnostic-policy evidence only; no
model checkpoint is measured.

## What was added

`ScriptedMazeChasePlannerPolicy`
(`evaluation/diagnostic_policies.py`) gains three actuation-awareness knobs
so the same pixel-only planner can run matched to a variant slot's published
mechanics:

- **`input_delay_ticks`** — the planner tracks its own submitted presses
  (its own outputs, never simulator state) as an exact model of the world's
  delay FIFO, simulates the forced prefix those queued presses determine,
  and aims each new press at the tick it will actually apply.
- **`player_period`** — the simulation gates player movement and input
  sampling to every Nth tick, like the world's speed curve.
- **`ghost_elroy`** — the simulation shortens the ghost period by one once
  the *visible* pellet count says at least half the pellets are eaten, the
  same derivation the world uses. Pellet pixels hidden under ghosts make the
  visible count an undercount, so the simulated speed-up can only arrive
  early — conservative, never optimistic.

The internals were reorganized around one `_step_sim` tick simulator shared
by the forced prefix, the path lookahead, and the survival fallback.
Default knobs (all-zero compensation) are byte-for-byte behavior-preserving:
the full 65-cell canonical matrix SHA-256 after the rewrite is still
`3ca6cc8cd371f3e2bbb2799bcd4536dc0b9e3721ff515c6273ec0052e4ff4793`.

Remaining blind spots, by construction: non-direct ghost rules are not
modeled, and when the forced prefix is already fatal the post-catch respawn
cell is hidden RNG, so the fallback past that point is approximate.

## What compensation recovers (seeds 5/9/13, 600 ticks)

| Slot | Plain planner | Compensated planner |
|---|---|---|
| `delayed_input` (2-tick FIFO) | **-147** reward, 52 catches | **+386** reward, 7 catches |
| `elroy` (endgame speed-up) | +244 reward, 20 catches | **+309** reward, 11 catches |
| `slow_player` (period 2) | +226 reward, 20 catches | +226 reward, 20 catches |

Read-out:

- The delayed-input failure was *pure actuation-latency mismatch*. With an
  exact FIFO model the planner recovers from its only negative slot to
  near-clear performance (+386 vs the canonical +426). This is the
  diagnostic proof that a play-capable model must plan against its own
  decision-to-actuation latency — the quantity
  `evaluation/play_latency.py` already measures — not just against the
  world.
- Elroy compensation removes 45% of the catches and lifts reward by 65;
  the residual gap to canonical is the hidden-pellet undercount making the
  simulated speed-up occasionally early.
- Player-period compensation is *outcome-neutral*: presses submitted on
  non-move ticks are discarded by the world, so the plain planner's
  every-tick player model never costs it on this slot. The identical row is
  the expected invariance, and the test suite pins it.

## Verification

`tests/test_diagnostic_policies.py` grows to 25 tests: constructor
validation for the new knobs, default-vs-explicit report equality on the
canonical slot, per-seed recovery assertions on the delayed and elroy
slots, and the slow-player outcome-neutrality invariance. The full
play-safe suite exits 0.
