# Maze-chase planner-teacher dataset (2026-08-17)

Status: infrastructure. The dataset is lazy and virtual — construction
performs no rollout work, and no sealed ranges exist for this family.

## What was added

**`data/maze_chase_dataset.py`** — a lazy, deterministic supervision
dataset for the maze_chase ladder world, mirroring the moving_shapes
dataset contract:

- `MazeChaseDatasetConfig` pins the full world knob set (ghost count,
  period, player period, extra loops, ghost rule, elroy, input delay,
  sticky direction) plus split/count/length/offset/discount. Every knob is
  part of the canonical manifest identity
  (`irene.maze_chase.planner_teacher.v1`), so each registered variant
  configuration is its own dataset identity.
- `MazeChaseSequenceDataset` materializes one episode per index on demand:
  the pixel-only lookahead planner
  (`diagnostic.scripted_maze_chase_planner.v1`) plays teacher, with its
  actuation-awareness knobs automatically matched to the world's
  `ghost_period` / `player_period` / `input_delay_ticks` / `ghost_elroy`
  settings, so demonstrations stay frontier-quality on every variant the
  planner can model. Non-direct ghost rules remain the planner's documented
  blind spot; those sequences are still valid demonstrations, just
  optimistic.
- The moving_shapes transition/sequence containers are reused unchanged
  (their invariants — boundary-frame continuity, terminal placement,
  zero-bootstrapped discounted value targets — are family-generic), and the
  same disjoint 62-bit split namespaces apply.

Sealed-range status: the RCQ-v2/v3 seals cover moving_shapes only. No
maze_chase TEST range is sealed, and the module docstring records that a
capability guard must be added before any future maze_chase TEST sealing.

Smoke behavior (train split, index 0, 48 ticks): 42 pellets eaten, 1 catch,
ends truncated at the sequence boundary with the terminal value target
exactly equal to the final reward (zero bootstrap). Materialization costs
~1.3 ms per planner decision on this machine, so a 128-tick sequence takes
~0.2 s — generation at scale remains accelerator-window work, as with the
other ladder datasets.

This closes the loop from evaluation to training data: the same planner
that maps the scripted frontier now labels the demonstrations a model can
be distilled from, through the same `Sequence[...]`/epoch interface the
trainer already consumes for moving_shapes.

## Verification

`tests/test_maze_chase_dataset.py` (11 tests, torch-free): config
validation fail-closed, manifest determinism and knob sensitivity, teacher
identity in the manifest, well-formed materialized sequences (terminal
placement, zero-bootstrapped final value), pellets eaten from pixels only,
per-index determinism, disjoint split namespaces, deterministic epoch
bijection, index validation, and variant-configuration materialization.
The full play-safe suite exits 0 (490 tests).
