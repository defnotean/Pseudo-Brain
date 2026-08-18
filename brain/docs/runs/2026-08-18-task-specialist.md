# B3: task specialist versus unchanged generalist, smoke machinery (2026-08-18)

Status: implementation. Local CPU-only verification; no training run, no
accelerator, no sealed data. Implements preregistered control **B3** from
[2026-08-17-remaining-baseline-controls-preregistration.md](2026-08-17-remaining-baseline-controls-preregistration.md)
under the owner's explicit 2026-08-18 go. Campaign-scale ladder
materialization remains DGX-window work; this slice uses the existing lazy
datasets.

## What was added

**`MixedWorldBatchConfig` / `MixedWorldBatchSource`**
(`training/batches.py`, exported from `training/__init__.py`) — the B3
generalist data identity: an equal-count round-robin of moving_shapes and
maze_chase through the same `DeterministicBatchSource` protocol. Kept
separate from the pinned `DatasetConfig` (whose `kind` supports only
moving_shapes), so no registered moving_shapes configuration hash changes.
`train.py` is untouched.

- Member configs must share split counts, sequence_length, burn_in_steps,
  seed_offset, and discount; per-world timing knobs may differ.
- Unshuffled order is moving_shapes[0], maze_chase[0], moving_shapes[1],
  maze_chase[1], … so early batches already see both worlds. Train epochs
  apply the single-world affine bijection over the combined index space,
  keyed by this source's manifest (`batch_source:
  "mixed_world_split_namespaces"`).
- Specialists consume the member sources (`MovingShapesBatchSource`,
  `MazeChaseBatchSource`) under the same knobs. That is the §29
  "unchanged generalist versus game-specific fine-tuning" split: one
  architecture, three training regimes.

B3 is a training-regime distinction, not a new factory in the slot-suite
architecture manifest.

## End-to-end smoke

`tests/test_task_specialist.py` (9 tests) closes the loop on CPU: mixed
config fail-closed, manifest determinism and knob sensitivity, mixed
identity distinct from either member, unshuffled validation round-robin,
exact batch repeatability and resume, both worlds present in a shuffled
train epoch, one real optimizer step on a mixed batch, and a 2+1-step
clone/fine-tune: specialists load a snapshot of the generalist, take one
specialist step, and the generalist weights stay bit-identical.

`scripts/compare_task_specialist_smoke.py` is the longer probe (schema 2 /
constant_after_warmup, 16 generalist steps on the mix, 8 fine-tune steps
per specialist). Row pinned at
`docs/runs/artifacts/task-specialist-smoke/irene.task_specialist.v1.json`.
Mixed-source SHA-256
`17c36a349972ef434c4655b30c9e80999374b2f8b0d285d097295061d626935c`.

| agent | moving_shapes action loss | maze_chase action loss |
| --- | ---: | ---: |
| unchanged generalist | 0.7151 | 0.5780 |
| moving_shapes specialist | 0.7551 | 0.6085 |
| maze_chase specialist | 0.7331 | **0.4902** |

Train loss: generalist 2.095 → 1.453; shapes specialist last 0.620;
maze specialist last 1.205. Movement exactness is 0.000 on every cell
(the same quiescence pull the constant-LR 64-step suite learned first).

Reading, all exploratory:

1. **The maze specialist beats the frozen generalist on maze_chase**
   (0.490 vs 0.578) and pays a transfer cost on moving_shapes. That is
   the capability-ceiling signature B3 was registered to measure.
2. **The moving_shapes specialist does not beat the generalist
   in-distribution at this scale** (0.755 vs 0.715) and overfits its four
   train sequences (last train 0.620). Open-field imitation is the
   noisier of the two worlds here; eight fine-tune steps are not a
   ceiling measurement.
3. **Campaign-scale B3 is now a configuration exercise**, not new
   machinery — the same status maze_chase distillation reached. A DGX
   B3 run still needs generated, registered datasets at campaign length;
   the first-matched architecture campaign remains blocked until a
   reference qualifies.

## Verification

9 new tests; play-safe gate green, 566 tests, one expected POSIX skip.
No architecture-manifest regeneration (`batches.py` is outside the
matched-baseline implementation set). No RCQ registration, pin, or Spark
job.
