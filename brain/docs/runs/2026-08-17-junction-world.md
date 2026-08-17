# Junction world: maze-based junction choice on the environment ladder (2026-08-17)

Status: implementation only. No dataset generation, training, or registration
uses this world yet.

## What was added

**`src/irene_brain/environments/junction.py` — `JunctionEnv`**, the junction-
choice rung of PLAN.md §20 / ROADMAP §5, and the closest world so far to the
Pac-Man skill: deciding correctly at branch points while a chaser closes in.

- **World**: a perfect maze carved deterministically from the episode seed by
  a seeded iterative recursive backtracker (rooms at odd coordinates, SplitMix64
  with a domain separator). The player moves one corridor cell per tick;
  walls refuse the whole move (no sliding). Targets relocate on collection
  (+1). Chasers follow the exact BFS shortest path through the corridors
  every `chaser_period` ticks (fixed W/A/S/D tie-break order); contact under
  the shared swept-path same-time test emits `caught`, costs -1, and respawns
  the player on a sampled empty corridor cell. The target spawns at the
  BFS-farthest cell from the player; chasers spawn at corridor distance ≥ 10
  where the maze allows.
- **Rigor**: same pattern as moving shapes and pursuit — explicit integer
  PRNG, canonical checksummed version-1 snapshots (magic `IBJS`), atomic
  validated restore, configuration-mismatch rejection, `state_hash`. The maze
  is a pure function of the episode seed under the pinned snapshot version,
  so snapshots need no maze bytes: restore regenerates and validates against
  it.
- **Render contract**: same 16×16 one-pixel-per-cell frame with the shared
  public `PLAYER_RGB`/`TARGET_RGB` colors plus a public `WALL_RGB`, so
  pixel-only teachers and the model input path work unmodified.
- Constructor knobs: `chaser_count` (default 1, max 8), `chaser_period`
  (default 2), `tick_period_ns`, `max_ticks`.

## Verification

New `tests/test_junction.py` (18 tests, torch-free): maze connectivity and
seed determinism (BFS from the spawn reaches exactly the carved set for ten
seeds), protocol satisfaction, same-seed pixel/state identity, walls refuse
movement, everything spawns on corridors over ten seeds, a chaser catches a
stationary player via shortest-path pursuit (reward -1), catch respawns onto
an empty corridor, a BFS teacher collects targets (the maze is traversable
end to end under the step mechanics), 1,000-step replay exactness against
three pinned hashes (initial observation `6caf2fa5…`, final observation
`dd6a5fbf…`, final state `d3a1bea4…`), atomic restore on corruption,
configuration-mismatch rejection, terminal boundary, invalid control,
constructor and observation-field validation, and two closed-loop cross-world
tests (no-op gets caught; deterministic episodes). The full play-safe suite
exits 0.

## Ladder status

Implemented: moving shapes, pursuit/evasion, junction choice. Remaining
in-repo rungs: occlusion and keys/doors; then branch-DAG data generation at
scale on the Spark.
