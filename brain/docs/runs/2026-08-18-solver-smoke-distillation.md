# Solver smoke distillation probe: occlusion (2026-08-18)

Status: exploratory verification probe. Local CPU-only, single-threaded;
not a training run, not a qualified result, no sealed data.

## What was added

**`scripts/distill_solver_smoke.py`** — the generalized solver-world
counterpart of `scripts/distill_maze_chase_smoke.py`. It distills the
smoke-scale thought-field model on `SolverBatchSource` demonstrations for
any of the four solver worlds (`python brain/scripts/distill_solver_smoke.py
[world]`, default `occlusion`) for 256 optimizer steps and pins the same
four measurements as the maze probe: train loss (first vs last 32 steps),
validation action loss and movement exact-match before/after, and
closed-loop play (seeds 5/9, 240 ticks) before/after.

To keep play evaluation on exactly the dataset's canonical world
configurations, `data/solver_dataset.py` gains a public
`solver_environment_factory(world)` accessor returning the registered
`(max_ticks, tick_period_ns)` factory — play harnesses can no longer
drift from the teaching configuration. Exported from `data/__init__.py`;
covered by a new dataset test (12 total).

## Reference outcome (occlusion, bit-identical across two runs)

| measurement | before | after 256 steps |
| --- | --- | --- |
| train loss (first32 → last32) | 0.7623 | 0.4306 |
| validation action loss | 1.0133 | 0.4316 |
| validation movement exact-match | 0.0000 | 0.3065 |
| closed-loop reward / collisions | 0.0 / 0 | -17.0 / 17 |
| decisions rejected | 0 | 0 |

## All four solver worlds (2026-08-18, same protocol)

| world | train loss first32→last32 | val action loss | val exact-match | play reward/collisions |
| --- | --- | --- | --- | --- |
| keys_doors | 0.6136 → 0.4197 | 0.8469 → 0.4174 | 0.0000 → 0.0887 | 0/0 → 0/0 |
| junction | 0.4381 → 0.0783 | 0.8110 → 0.0549 | 0.0000 → **1.0000** | −4/4 → −3/4 |
| occlusion | 0.7623 → 0.4306 | 1.0133 → 0.4316 | 0.0000 → 0.3065 | 0/0 → −17/17 |
| pursuit | 0.8850 → 0.6064 | 1.0364 → 0.5426 | 0.0242 → 0.0242 | −24/24 → −23/24 |

The four worlds span the full imitability spectrum at smoke scale:
junction's deterministic corridor policy is learned **perfectly** in
validation (exact-match 1.0000) yet still gets caught in 240-tick play —
small errors compound past the 32-step training horizon; occlusion
imitates well but walks into unremembered hazards; keys_doors learns
measurable agreement without yet scoring; pursuit's lure-and-dodge
teacher barely distills at all (exact-match flat at 0.0242). These are
the pinned zero-points the DGX distillation campaign must move.

## Reading

Two findings, both pinned for the DGX campaign:

1. **The occlusion teacher is far more imitable than the maze planner at
   smoke scale.** Movement exact-match reached 0.3065 — 38× the maze
   probe's 0.0081 — because the memory policy's action distribution is
   dominated by directed, repeatable travel (quadrant sweeps, remembered
   target homing) rather than the planner's knife-edge lookahead
   decisions.
2. **Half-learned movement is worse than stillness.** The untrained
   model barely moves (0 reward, 0 collisions); the distilled model moves
   purposefully but walks into hazards it cannot yet remember — 17
   collisions. The teacher's zero-collision record comes entirely from
   persistent hazard beliefs, the exact skill the thought-state thesis
   says the architecture should hold. At smoke scale the gradient teaches
   *where to go* long before *what to avoid*; closing that gap is a
   scale question, not a machinery question.

The chain solver teacher → `SolverSequenceDataset` → `SolverBatchSource`
→ objective → optimizer step → closed-loop play now runs end to end on
CPU for all four solver worlds, deterministically.

## Verification

Probe output bit-identical across two consecutive runs (full stdout
diff). The full play-safe suite exits 0 (544 tests).
