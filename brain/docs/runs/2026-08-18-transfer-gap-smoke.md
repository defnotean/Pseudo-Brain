# Zero-shot transfer battery, smoke scale (2026-08-18)

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

## Verification

Deterministic (fixed seeds, fixed batch order, simulated clock; zero
decision rejections in all 24 episodes). Play-safe suite unaffected
(script-only change).
