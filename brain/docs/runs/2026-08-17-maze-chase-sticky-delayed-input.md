# maze_chase sticky/delayed input (2026-08-17)

Status: implementation only. Completes the registered variant axes for the
Phase 4 target world (ROADMAP §5 item 3), on top of the ghost pursuit rules
([2026-08-17-maze-chase-ghost-rules.md](2026-08-17-maze-chase-ghost-rules.md))
and speed curves
([2026-08-17-maze-chase-speed-curves.md](2026-08-17-maze-chase-speed-curves.md)).
No dataset generation, training, or registration uses these knobs yet.

## What was added

Two constructor knobs forming the world's input pipeline, both part of
snapshot identity:

- **`input_delay_ticks`** (default 0, range [0, 16]) — requested control
  masks enter a fixed-length FIFO; the mask that drives tick *t* is the one
  requested at *t − delay*. The queue zero-fills on reset and is snapshot
  state (one byte per queued mask).
- **`sticky_direction`** (default False) — a direction mask, once it leaves
  the delay queue nonzero, persists until another nonzero mask replaces it.
  Releasing every key no longer stops the player. Delay applies before
  stickiness: the queue sees raw requests, the sticky latch sees delayed
  ones.

`StepOutcome.applied_control` and the observation's `previous_control`
report the *effective* post-pipeline mask — what the environment actually
did — so teachers and evaluators see actuator truth, not request echo.

Defaults reproduce the historical behavior exactly (delay 0 passes the
queue straight through; sticky off ignores the latch), so the pinned
observation content hashes remain valid.

## Snapshot versioning

The snapshot header (magic `IBMC`) bumps to **version 4**, appending the
delay length, the sticky flag, and the sticky latch (one byte each), plus
`input_delay_ticks` queue bytes after the ghost table. Restore rejects
delay/sticky config mismatches, a sticky latch or queue byte with
unsupported key bits, and (via the existing length check) a truncated or
padded queue. The 1,000-step replay pin's final state hash was regenerated
(`cc1d5ef1…` → `3613c576…`); observation content hashes are unchanged.

## Verification

`tests/test_maze_chase.py` grows from 28 to 33 tests. New coverage: a
two-tick delay postpones a single press by exactly two steps with the
effective control surfacing in `applied_control`; sticky direction persists
through release (versus a non-sticky control env) and is replaced by the
next press; delay + sticky + every prior knob combined survive a snapshot
roundtrip with identical post-restore outcomes; restore rejects delay and
sticky mismatches and a tampered queue byte; constructor validation for
both knobs. The full play-safe suite exits 0.

## Variant axes: complete

All registered axes from ROADMAP §5 item 3 are now implemented on
`maze_chase`: maze layouts (seed + `extra_loops`), ghost AI rules, speed
curves, and sticky/delayed input. Visuals and control mappings remain
shared ladder surfaces. The cross-world matrix slot `world.maze_chase.v1`
still pins the default configuration; promoting variants into the matrix is
a separate, preregistered step.
