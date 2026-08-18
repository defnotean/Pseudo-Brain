# Checkpoint comparison tool (2026-08-18)

Status: tooling. Local CPU-only verification; no training run, no
accelerator, no sealed data. Implements the strategic review's §3.4.3
checkpoint-comparison item.

## What was added

**`evaluation/checkpoint_compare.py`** — analysis-only comparison
primitives, explicitly *not* a resume path
(`training.checkpoint.load_checkpoint` remains the only verified resume
interface):

- `load_analysis_payload` — restricted `weights_only` deserialization
  with fail-closed payload shape checks and the file's SHA-256 recorded.
- `require_same_run_identity` — refuses to compare checkpoints whose
  config, data, or code digests differ.
- `objective_states` / `parameter_differences` — per-tensor relative-L2
  difference table, sorted by change magnitude ("which thoughtlets
  changed most"), with identical-tensor flags.
- `metric_delta_table` — mean metric values for two models over the same
  batches, with deltas.

**`scripts/compare_checkpoints.py`** — the CLI: rebuilds the model from
`--config`'s factory, restores both checkpoints, evaluates both on the
same validation batches, and prints/writes the JSON report (top parameter
changes + metric deltas). `--cpu` overrides a CUDA-configured recipe to
CPU/float32 so DGX-produced checkpoints can be analyzed locally.

## Verification

- `tests/test_checkpoint_compare.py` (3 tests): analysis roundtrip with
  the run-identity guard failing closed on digest mismatch; a perturbed
  `thought_attention.feed_forward.0.weight` ranks first in the difference
  table with all 222 other tensors flagged identical; metric-table key-set
  discipline.
- End-to-end demo on two real checkpoints (4-step intervals of a
  constant-LR smoke run): 198/223 tensors changed, actuator coordination
  norms move fastest, action loss 0.7320 → 0.6531 between the snapshots.
  Artifacts: `docs/runs/artifacts/checkpoint-compare-demo/`.
- The full play-safe suite exits 0 (551 tests).

## Debugging note worth recording

The demo initially appeared to show zero parameter change between
checkpoints four steps apart. The cause was not the tool: dgx-smoke's
legacy cosine schedule (schema_version 1) decays the learning rate to
exactly 0.0 at `max_optimizer_steps`, so steps past the configured
maximum compute gradients but apply nothing. A useful property to
remember when hand-driving `TorchTrainingSystem` beyond a recipe's
configured horizon.
