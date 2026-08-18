# Maze-chase training batches and the distillation smoke (2026-08-17)

Status: infrastructure. Local CPU-only pipeline verification; no training
run, no accelerator, no sealed data.

## What was added

**`MazeChaseBatchConfig` / `MazeChaseBatchSource`**
(`training/batches.py`, exported from `training/__init__.py`) — the
split-namespaced batch source that feeds planner-teacher maze_chase
demonstrations to the generic trainer through the exact
`DeterministicBatchSource` protocol (`manifest_sha256`,
`batches_per_epoch`, `iter_batches`). Kept separate from the pinned
`DatasetConfig` (whose `kind` supports only moving_shapes), so no
registered moving_shapes configuration hash changes.

- Every maze_chase world knob rides in the batch-source manifest identity
  (`batch_source: "maze_chase_split_namespaces"`), so each variant
  configuration is a distinct, citable data identity end to end.
- Batch validity note, documented on the class: `TrajectoryBatch` requires
  equal-length sequences and fails closed otherwise. A maze_chase episode
  cannot terminate before every pellet is eaten — at least one tick per
  pellet — so sequences shorter than the maze's pellet count always
  truncate at the boundary and batch cleanly. The canonical slot clears in
  roughly 150–250 ticks, above the registered 128-tick training length.

## End-to-end smoke

`tests/test_maze_chase_training.py` (7 tests) closes the distillation loop
on CPU: config validation fail-closed, manifest determinism and knob
sensitivity, batch-source identity distinct from the moving_shapes source,
exact batch repeatability and batch-boundary resume, unshuffled validation
order — and one real bounded optimizer step: a smoke-scale
`IreneBrainModel` + `ThoughtFieldObjective` + `TorchTrainingSystem`
consumes a planner-teacher maze_chase batch and returns finite loss with
the probed parameter provably updated.

The chain env → frontier planner → lazy dataset → batch source → objective
→ optimizer step now executes entirely in-repo, so a future distillation
run on the DGX is a configuration exercise, not new machinery.

## Verification

The full play-safe suite exits 0 (497 tests).
