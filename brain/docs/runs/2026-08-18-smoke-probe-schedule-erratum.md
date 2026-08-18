# Erratum: smoke probes trained for 2 effective optimizer updates (2026-08-18)

Status: measurement erratum, corrected with pinned re-runs. Local
CPU-only, single-threaded; exploratory probes, not qualified results.

## What was wrong

`scripts/compare_baselines_smoke.py` and `scripts/transfer_gap_smoke.py`
built their programmatic `TrainingConfig` with `schema_version=1` and
`run.max_optimizer_steps=2`. Schema 1 forces the legacy cosine schedule
(`config.py`: "schema_version 1 supports only the legacy cosine
schedule"), and the cosine multiplier is exactly 0.0 from
`max_optimizer_steps` onward (`training/schedules.py`). Every numbered
"optimizer step" past step 1 ran the forward and backward passes with a
learning rate of exactly zero — **no parameters updated**. The same
gotcha had already been pinned for dgx-smoke in
[2026-08-18-checkpoint-comparison-tool.md](2026-08-18-checkpoint-comparison-tool.md);
the probe configs reproduced it in miniature.

Verified directly: multiplier 1.0 at step 0, 0.5 at step 1, 0.0 at every
step ≥ 2; and the transfer battery's last-16 train loss was bit-identical
(0.8337) at 128, 256, 512, and 1024 numbered steps.

## Impact

- **Affected:** every published smoke-comparison row (all eleven variants
  plus the B2 world-model actor) and every transfer-battery row
  (reference and B2 actor, including the frozen-regime liftoff attempts).
  What those rows actually measured: two effective optimizer updates from
  identical seeds, then frozen evaluation. Inter-variant comparisons
  within each table remain internally fair — all rows share the identical
  frozen condition — but the "64 steps" / "128 steps" labels are wrong,
  and any conclusion about learning dynamics from those tables is void.
- **Not affected:** the distillation probes
  (`distill_solver_smoke.py`, `distill_maze_chase_smoke.py`) set
  `max_optimizer_steps` to their real step counts, so their cosine
  schedule decays across the run and every numbered step trains. All
  campaign/DGX recipes are likewise unaffected.
- **Preserved:** all frozen-regime rows remain at their original artifact
  paths untouched; the corrected runs write to `constant-lr/`
  subdirectories.

## The fix

Both probes now build schema-2 configs with
`scheduler_kind="constant_after_warmup"` — the campaign baseline regime —
so every numbered step is a real optimizer update at constant LR.

## Corrected smoke comparison (64 REAL steps, seed 20260818)

Rows: `docs/runs/artifacts/baseline-smoke-compare/constant-lr/<variant>.json`.

| variant | params | train loss first→last | val action loss | val movement exact | val world loss | pairwise cos |
| --- | --- | --- | --- | --- | --- | --- |
| **world_model_actor (B2)** | 95,417 | 0.8676 → 0.5931 | 1.1124 → **0.5341** | 0.000 | **0.0325** | — |
| monolithic_gru.parameter_matched | 53,909 | 0.8702 → 0.6131 | 1.1124 → 0.5373 | 0.000 | 0.1682 | — |
| dense_routing | 47,258 | 0.8121 → 0.6175 | 1.1178 → 0.5375 | 0.000 | 0.1252 | 0.174 |
| **routed (reference)** | 47,258 | 0.8121 → 0.6175 | 1.1175 → 0.5375 | 0.000 | 0.1253 | 0.174 |
| isolated_slots | 47,258 | 0.8121 → 0.6174 | 1.1303 → 0.5380 | 0.000 | 0.1306 | 0.175 |
| recurrent_transformer | 39,689 | 0.8806 → 0.6227 | 1.0541 → 0.5401 | 0.000 | 0.1399 | — |
| reset_slots | 47,258 | 0.8091 → 0.6141 | 1.1138 → 0.5404 | 0.000 | 0.1383 | 0.158 |
| fixed_multi_horizon (B1) | 47,258 | 0.8196 → 0.6314 | 1.1138 → 0.5411 | 0.000 | 0.1848 | 0.419 |
| reactive | 47,258 | 0.8094 → 0.6152 | 0.9548 → 0.5411 | 0.000 | 0.1835 | 0.152 |
| independent_ensemble | 49,833 | 0.7987 → 0.6147 | 0.7798 → 0.5440 | 0.000 | 0.2405 | 0.151 |
| serial_depth | 58,123 | 0.8056 → 0.6174 | 0.8356 → 0.5442 | 0.000 | 0.1491 | 0.228 |
| monolithic_gru.same_width | 46,313 | 0.7942 → 0.6314 | 0.7218 → 0.5717 | 0.000 | 0.1453 | — |

## Corrected transfer battery (128 REAL steps)

Rows: `docs/runs/artifacts/transfer-gap-smoke/constant-lr/<variant>/<world>.json`.

| world | reference | **B2 actor** | no-op | reactive chaser |
| --- | --- | --- | --- | --- |
| moving_shapes (in-distribution) | 0 | 0 | 0 | +57 |
| pursuit | −31 | −25 | −24 | +30 |
| junction | −4 | −4 | −4 | −3 |
| occlusion | 0 | 0 | 0 | +1 |
| keys_doors | 0 | 0 | 0 | 0 |
| maze_chase | −2198 / 221 collisions | **−161 / 17 collisions** | −161 / 17 | −161 / 17 |

## Liftoff sweep (moving_shapes play, REAL steps)

| steps | reference | B2 actor |
| --- | --- | --- |
| 128 | 0 / 0 targets / 0 collisions (train 0.558) | 0 / 0 / 0 (train 0.575) |
| 256 | −16 / 0 / 16 (train 0.329) | 0 / 0 / 0 (train 0.442) |
| 512 | **+1 / 1 / 0** (train 0.213) | 0 / 0 / 0 (train 0.205) |
| 1024 | −27 / 1 / 28 (train 0.092) | **+1 / 1 / 0** (train 0.121) |

## Revised readings

1. **serial_depth's frozen-regime "lead" was an initialization artifact.**
   Its lower loss at init (0.8356) survived because nothing trained.
   Under real training the whole pack converges within ±0.007 of 0.54 and
   serial_depth lands mid-pack (0.5442). Frozen-regime movement-exact
   "wins" (0.208) were likewise init-output artifacts: real training
   drives every variant's movement exactness to 0.000 at 64 steps — the
   quiescence pull is the first thing constant-LR training learns.
2. **The reference still does not lead at smoke scale** — it ties
   dense_routing exactly and sits inside a ±0.007 pack. The review's
   RCQ-v3 smoke proceed-criterion remains unmet under the corrected
   regime.
3. **The B2 world-model actor leads the suite on action loss (0.5341)
   and its world loss reverses from worst to best by 4×** (1.0853 frozen
   → 0.0325 real; the pack sits at 0.125–0.24). The rollout objective
   was not slow to learn — it was never training.
4. **At 128 real steps the actor eliminates the catastrophic maze_chase
   failure mode**: −161/17 collisions, exactly the no-op floor, while the
   reference remains catastrophic at −2198/221. The frozen-regime hint
   (−1358 vs −3890) was real and understated.
5. **Liftoff is now measurable on CPU.** First above-floor
   in-distribution play: reference at 512 steps (non-monotonic — it
   blunders when it moves: −16 at 256, −27 at 1024 with visible
   overfitting, train loss 0.092 on 16 sequences), actor at 1024 steps
   with a never-below-floor profile (0, 0, 0, +1). Two play seeds make
   single cells noisy; the actor's no-blunder consistency across both the
   sweep and the six-world battery is the signal worth a DGX-scale
   falsification run.

## Verification

The corrected configs validate as schema 2 / constant_after_warmup; the
liftoff sweep's train losses now decrease monotonically with numbered
steps (routed 0.558 → 0.329 → 0.213 → 0.092), confirming every numbered
step trains. Frozen-regime artifacts are preserved at their original
paths. Play-safe suite green, 558 tests (script-only changes).
