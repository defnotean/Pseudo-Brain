# Solver demonstration datasets and training batches (2026-08-17/18)

Status: infrastructure. Local CPU-only pipeline verification; no training
run, no accelerator, no sealed data.

## What was added

**`SolverDatasetConfig` / `SolverSequenceDataset`**
(`data/solver_dataset.py`, exported from `data/__init__.py`) — lazy,
mechanics-matched solver-teacher demonstrations for the four solver worlds
(keys_doors, junction, occlusion, pursuit), through the exact same sequence
interface the trainer already consumes. Generator identities
`irene.{world}.solver_teacher.v1`; manifest hash prefix
`IRMSOLVERDATASET\x01`; epoch shuffle identical to the maze_chase dataset
scheme.

- The world registry `_SOLVER_WORLDS` maps each world to its canonical
  matrix configuration, its frontier solver policy (lazy-imported from
  `evaluation/diagnostic_policies.py` to avoid the evaluation↔data import
  cycle), and the teacher identity string.
- `moving_shapes` is deliberately excluded: it belongs to the sealed RCQ
  family, so any future solver-style dataset there requires the guard and a
  fresh preregistration before TEST sealing. `maze_chase` is excluded
  because it already has its own planner-teacher dataset.
- Teachers are verified to score positive reward from pixels only on every
  world within 128 ticks (the registered training length).

**`SolverBatchConfig` / `SolverBatchSource`** (`training/batches.py`,
exported from `training/__init__.py`) — the split-namespaced batch source
that feeds solver demonstrations to the generic trainer through the
`DeterministicBatchSource` protocol (`manifest_sha256`,
`batches_per_epoch`, `iter_batches`). Kept separate from the pinned
`DatasetConfig` (whose `kind` supports only moving_shapes), so no
registered moving_shapes configuration hash changes.

- The world name rides in the batch-source manifest identity
  (`batch_source: "solver_split_namespaces"`), so each world is a
  distinct, citable data identity end to end.
- Batch validity note, documented on the class: `TrajectoryBatch` requires
  equal-length sequences and fails closed otherwise. All four solver
  worlds never terminate an episode (`terminated` is always `False`), so
  every sequence truncates at the length boundary and batches cleanly —
  even simpler than maze_chase, whose episodes end once every pellet is
  eaten.

## Verification

`tests/test_solver_dataset.py` (11 tests) and
`tests/test_solver_training.py` (8 tests) cover: config validation
fail-closed on both levels, manifest determinism and knob sensitivity
(including per-world identity), batch-source identity distinct from the
maze_chase source, exact batch repeatability and batch-boundary resume,
unshuffled validation order, split-namespace disjointness, per-world
well-formed materialization with boundary-frame continuity, teacher
scoring on every world — and one real bounded CPU optimizer step: a
smoke-scale `IreneBrainModel` + `ThoughtFieldObjective` +
`TorchTrainingSystem` consumes a solver-teacher keys_doors batch and
returns finite loss with the probed parameter provably updated.

The full play-safe suite exits 0 (543 tests).

The chain env → frontier solver → lazy dataset → batch source → objective
→ optimizer step now executes entirely in-repo for all four solver worlds,
so a future multi-world distillation run on the DGX is a configuration
exercise, not new machinery.
