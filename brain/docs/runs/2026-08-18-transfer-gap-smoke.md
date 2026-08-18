# Zero-shot transfer battery, smoke scale (2026-08-18)

> **ERRATUM (2026-08-18):** the battery's schema-1 config forced the
> legacy cosine schedule with `max_optimizer_steps=2`, so every numbered
> step past step 1 applied a learning rate of exactly 0.0 — the tables
> below (reference and the B2 addendum) measured **two effective
> optimizer updates**, not 128. The zero-points are preserved
> frozen-regime only. Corrected constant-LR re-runs — including the
> liftoff sweep the frozen regime made invisible — and revised readings:
> [2026-08-18-smoke-probe-schedule-erratum.md](2026-08-18-smoke-probe-schedule-erratum.md).

Status: exploratory verification probe. Local CPU-only, single-threaded;
not a training run and not a qualified result. Quantifies the transfer
gap named in the 2026-08-18 strategic review (§3.3.2) with pinned
numbers.

## Protocol

`scripts/transfer_gap_smoke.py` trains a smoke-scale reference
thought-field model on moving_shapes for 128 optimizer steps (campaign
window shape: sequence_length 8, burn-in 2; final train-loss window
2.13 → 0.84), then plays it **zero-shot, no fine-tuning** on all six
canonical ladder worlds (seeds 5/9, 240 ticks, hazard_count 3), alongside
three non-privileged diagnostic baselines evaluated under the identical
play configuration. Per-world JSON rows:
`docs/runs/artifacts/transfer-gap-smoke/<world>.json`.

## Results (reward / targets / collisions)

| world | trained model | no-op | random | reactive chaser |
| --- | --- | --- | --- | --- |
| moving_shapes (in-distribution) | 0 / 0 / 0 | 0 / 0 / 0 | −11 / 1 / 12 | +57 / 65 / 8 |
| pursuit | −29 / 2 / 31 | −24 / 0 / 24 | −21 / 2 / 23 | +30 / 55 / 25 |
| junction | −4 / 0 / 4 | −4 / 0 / 4 | −3 / 1 / 4 | −3 / 1 / 4 |
| occlusion | 0 / 0 / 0 | 0 / 0 / 0 | −11 / 1 / 12 | +1 / 1 / 0 |
| keys_doors | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| maze_chase | **−3890 / 0 / 391** | −161 / 0 / 17 | −171 / 0 / 23 | −161 / 0 / 17 |

## Reading

1. **Zero-shot transfer is at or below the no-op floor everywhere.** The
   model matches no-op exactly on four worlds, is slightly worse than
   no-op on pursuit, and is catastrophically worse on maze_chase (−3890
   vs −161): maze walls plus pellets make every tick costly for a policy
   whose movement was shaped by open-field statistics.
2. **Even in-distribution play is at the no-op floor after 128 steps** —
   consistent with the distillation probes: smoke-scale imitation teaches
   measurable validation agreement long before it teaches play-level
   competence. The transfer question is therefore not yet "how much
   transfers" but "how much scale precedes any transfer at all" — a DGX
   question, now with a pinned CPU zero-point table to beat.
3. The battery itself is the deliverable: one command produces the full
   six-world × four-agent table with the exact canonical world
   configurations, so every future checkpoint (smoke or DGX) can be
   dropped into the same comparison.

## Addendum: the B2 world-model actor runs the same battery (2026-08-18)

After B2 landed
([2026-08-18-world-model-actor.md](2026-08-18-world-model-actor.md)), the
battery gained `--variant world_model_actor`: the latent world-model
actor (monolithic trunk at smoke width 18) is trained through its
declared `LatentRolloutObjective` hook — the same resolution discipline
as train.py — under the identical 128-step protocol, with rows written to
`docs/runs/artifacts/transfer-gap-smoke/world_model_actor/<world>.json`
so the pinned reference rows stay untouched.

| world | reference | **B2 actor** | no-op | reactive chaser |
| --- | --- | --- | --- | --- |
| moving_shapes (in-distribution) | 0 | 0 | 0 | +57 |
| pursuit | −29 | −26 | −24 | +30 |
| junction | −4 | −4 | −4 | −3 |
| occlusion | 0 | 0 | 0 | +1 |
| keys_doors | 0 | 0 | 0 | 0 |
| maze_chase | −3890 | **−1358** | −161 | −161 |

Reading:

1. **The headline stands:** the actor's zero-shot transfer is also at or
   below the no-op floor everywhere — exactly no-op on four worlds,
   slightly below on pursuit. The transfer gap is not a direct-head
   artifact; a learned transition model trained on the same 128 steps
   does not close it either. Scale remains the open variable.
2. **One hint worth a campaign question:** the actor's maze_chase failure
   is markedly less catastrophic than the reference's (−1358 vs −3890,
   138 vs 391 collisions) at identical seeds and play configuration. One
   seed pair at smoke scale is not evidence of a transfer advantage — but
   "does the rollout objective buy hazard avoidance in wall worlds" is
   now a falsifiable DGX-scale comparison with both zero-points pinned.
3. The battery now compares recipe families, not just checkpoints: any
   future variant with a declared objective hook drops into the same
   table via `--variant`.

## Verification

Deterministic (fixed seeds, fixed batch order, simulated clock; zero
decision rejections in all 24 episodes). Play-safe suite unaffected
(script-only change).
