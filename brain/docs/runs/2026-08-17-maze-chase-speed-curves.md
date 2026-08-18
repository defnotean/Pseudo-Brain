# maze_chase speed curves (2026-08-17)

Status: implementation only. Adds the speed-curves variant axis from
ROADMAP §5 item 3 to the Phase 4 target world, on top of the ghost pursuit
rules ([2026-08-17-maze-chase-ghost-rules.md](2026-08-17-maze-chase-ghost-rules.md)).
No dataset generation, training, or registration uses these knobs yet.

## What was added

Two constructor knobs, both part of snapshot identity:

- **`player_period`** (default 1, range [1, 64]) — the player acts once
  every N ticks. Control input is *sampled on move ticks only*: a direction
  pressed and released between move ticks changes neither the position nor
  the facing. The curve slows the player's whole decision loop, not just
  its motion, which is what makes fast-ghost/slow-player variants a real
  reaction-time test.
- **`ghost_elroy`** (default False) — once at least half of the pellets
  have been eaten, ghosts move one tick faster (minimum period 1). The
  threshold derives from `pellets_remaining` and the regenerated maze
  (`_effective_ghost_period`), so the curve is pure snapshot state — no
  extra fields beyond the config flag. Named for the arcade convention of
  the chase speeding up as the board empties; the implementation is
  original.

Defaults reproduce the historical behavior exactly, so the pinned
observation content hashes and every earlier world result remain valid.

## Snapshot versioning

The snapshot header (magic `IBMC`) bumps to **version 3**, appending
`player_period` and the elroy flag (one byte each). Restore rejects a
period or flag mismatch against the constructed configuration. The
1,000-step replay pin's final state hash was regenerated
(`d08096a3…` → `cc1d5ef1…`); observation content hashes are unchanged.

## Verification

`tests/test_maze_chase.py` grows from 25 to 28 tests. New coverage:
player-period gating (movement on move ticks only, position and facing
frozen through gated-tick presses), the elroy threshold (full period before
half the pellets are eaten, one faster after, recomputed correctly after a
snapshot roundtrip, inert without the flag), restore rejection of period
and elroy mismatches, and constructor validation for both knobs. The full
play-safe suite exits 0.

## Remaining variant axes

Sticky/delayed input remains open on this world; visuals and control
mappings are shared ladder surfaces. The cross-world matrix slot
`world.maze_chase.v1` still pins the default configuration.
