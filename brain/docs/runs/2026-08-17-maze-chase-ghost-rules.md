# maze_chase ghost pursuit rules (2026-08-17)

Status: implementation only. Extends the Phase 4 target world
([2026-08-17-maze-chase-world.md](2026-08-17-maze-chase-world.md)) with the
ghost-AI-rules variant axis from ROADMAP §5 item 3. No dataset generation,
training, or registration uses these rules yet.

## What was added

`MazeChaseEnv` gains a `ghost_rule` constructor knob (default `"direct"`,
part of snapshot identity) selecting each ghost's pursuit target:

- **`direct`** — the historical behavior: exact BFS shortest path to the
  player's cell.
- **`ambush`** — targets the corridor cell `_AMBUSH_LEAD` (4) steps ahead of
  the player's last pressed direction. The raw lead cell is clamped to the
  grid; if it lands inside a wall, the nearest corridor cell by Manhattan
  distance is used (row-major scan order breaks ties). Facing tracks control
  *intent*, not achieved motion: a wall-refused press still turns the
  player, so the ambush rule reads what the player last asked for, not
  simulator state — the model can exploit feints.
- **`shy`** — chases the player while the BFS distance exceeds
  `_SHY_DISTANCE` (8) and retreats uphill (the neighbor that maximizes
  distance from the player, fixed direction tie order) while closer. A shy
  ghost provably never catches a stationary player.
- **`mixed`** — assigns direct/ambush/shy cyclically by ghost index, so one
  world instance carries all three pressures at once.

Ghost movement itself is unchanged: one cell per `ghost_period` ticks along
the exact BFS shortest path to that ghost's target, with the shared
swept-path same-time collision test.

## Snapshot versioning

The snapshot header (magic `IBMC`) bumps to **version 2**, appending the
player facing (two signed bytes) and the ghost-rule code (one byte).
Restore validates facing components in {-1, 0, 1} with a nonzero direction
and rejects a rule byte that is unknown or mismatched against the
constructed `ghost_rule`. The versioned-header discipline means old
version-1 snapshots are rejected by version mismatch, and the 1,000-step
replay pin's final state hash was regenerated
(`192d76ca…` → `d08096a3…`); both observation content hashes are unchanged
because rendering and the default `direct` rule are behavior-identical.

## Verification

`tests/test_maze_chase.py` grows from 18 to 25 tests. New coverage: a shy
ghost never catches a stationary player across five seeds over 300 steps at
full speed; ambush and mixed runs diverge from direct under identical seeds
and controls; facing tracks control intent through wall refusals; facing
and rule survive a snapshot roundtrip; restore rejects a ghost-rule
mismatch; constructor rejects unknown and non-string rules. The replay
exactness test was re-pinned against the version-2 header. The full
play-safe suite exits 0.

## Remaining variant axes

Speed curves and sticky/delayed input remain open on this world; visuals
and control mappings are shared ladder surfaces. The cross-world matrix
slot `world.maze_chase.v1` still pins the default `direct` configuration —
adding rule variants to the matrix is a separate, preregistered step.
