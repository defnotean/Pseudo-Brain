# maze_chase: the original Pac-Man-like environment (2026-08-17)

Status: implementation only. Rights-clean by construction: no Namco code,
assets, names, or layouts; the maze is carved from the episode seed by the
repo's own generator and every behavior is defined in
`src/irene_brain/environments/maze_chase.py`. No dataset generation,
training, or registration uses this world yet.

## What was added

**`MazeChaseEnv`** — the Phase 4 target world (ROADMAP §5 item 3, PLAN.md
§20). A 16×16 maze full of pellets, one player, deterministic BFS
shortest-path ghosts:

- Every corridor cell except the spawn holds a pellet (+1 each, `pellet_eaten`).
- Ghosts (default 3, `ghost_period` 2) chase along exact BFS paths; contact
  under the shared swept-path same-time test emits `caught`, costs 10, and
  respawns the player at the spawn cell (or a sampled empty corridor cell if
  a ghost occupies it).
- Eating the last pellet emits `cleared`, adds +10, and terminates the
  episode — the first ladder world with a win condition.
- Render contract: public `PLAYER_RGB`, `PELLET_RGB`, `WALL_RGB`; ghosts
  red; same one-pixel-per-cell 16×16 frame and W/A/S/D surface as the whole
  ladder.
- Snapshot (magic `IBMC`, version 1) covers the pellet bitmap, ghost
  positions, and counters, with full pellet-accounting validation on
  restore (a pellet inside a wall, on the spawn cell, or an eaten/remaining
  total that does not match the maze is rejected).

## Design finding: perfect mazes are unplayable under pursuit

The first version reused the junction world's perfect-maze carver unchanged.
Bring-up with a greedy ghost-avoiding teacher showed zero of twenty seeds
were clearable: a perfect maze is a tree, so a chasing ghost is a hard wall
— once it sits between the player and the remaining pellets, every approach
crosses it, forever. Real chase mazes have loops for exactly this reason.
`maze_chase` therefore carves `_carve_maze_with_loops`: the shared perfect
maze plus `extra_loops` knocked walls (default 16, one independent cycle
each, own PRNG domain separator, pure function of the seed so snapshots
still regenerate by version rule; the knob is part of snapshot identity).
With 16 loops the simple teacher clears 8 of 10 surveyed seeds in a few
hundred steps; the default ghost pressure (3 ghosts at half speed) leaves
real escape planning to the model.

## Verification

New `tests/test_maze_chase.py` (18 tests, torch-free): protocol
satisfaction, same-seed identity, full pellet coverage minus the spawn,
pellet scoring and cell clearing, ghost catch and ghost-free respawn, a
ghost-avoiding greedy teacher clearing the maze and hitting the terminated
win condition (pinned seed 7), cycle counting (≥12 independent cycles at the
default 16 knocked walls), wall refusal, 1,000-step replay exactness against
three pinned hashes (initial observation `a4fffc12…`, final observation
`272cb411…`, final state `192d76ca…`), atomic restore on corruption,
count/period/loops/max-ticks mismatch rejection, tampered pellet-accounting
rejection, terminal boundary, invalid control, constructor and
observation-field validation, and two closed-loop tests through the
evaluator (no-op gets caught; episodes deterministic). The full play-safe
suite exits 0.

## What remains for this world

A scripted pellet-greedy teacher policy (the reference row), inclusion in
the cross-world matrix, the registered variant axes from ROADMAP §5 item 3
(ghost AI rules beyond direct BFS, speed curves, sticky/delayed input), and
branch-DAG lifetime data generation at scale on the Spark.
