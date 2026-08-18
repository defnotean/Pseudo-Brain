# Maze-chase smoke distillation probe (2026-08-17)

Status: exploratory verification probe. CPU-only, single-threaded, bounded
at 256 optimizer steps. **Not a training run, not a qualified result, no
registered campaign.** Sealed ranges are untouched (train/validation splits
only).

## Question

The distillation chain (env → frontier planner → lazy dataset → batch
source → objective → optimizer) is proven mechanically. Does the smoke-scale
thought field actually *learn* from planner demonstrations, and does any of
it transfer to closed-loop play at this scale?

## Setup

Pinned in `brain/scripts/distill_maze_chase_smoke.py` (fully deterministic;
re-running reproduces every number below exactly):

- Model: smoke-scale thought field (width 16, 4 thoughtlets, 2 cognitive
  cycles, 1 block) — the same configuration the closed-loop integration pin
  uses.
- Data: `MazeChaseBatchSource`, canonical slot, 16 train / 4 validation
  sequences of 32 ticks, burn-in 1, planner teacher with matched knobs.
- Training: 256 optimizer steps, batch size 1, lr 1e-3, CPU float32.
- Evaluation: teacher-agreement on held-out validation sequences, plus
  closed-loop play on the canonical slot (seeds 5/9, 240 ticks), before and
  after.

## Results

| Measurement | Before | After |
|---|---|---|
| Train loss (first/last 32 steps) | 14.3908 | 11.0962 |
| Validation action loss | 1.0156 | **0.4600** |
| Validation movement exact-match | 0.0081 | 0.0081 |
| Closed-loop reward (seeds 5/9, 240 ticks) | -163 | -161 |
| Closed-loop catches | 18 | 17 |

## Read-out

- **The imitation gradient is real and generalizes**: validation action
  loss on unseen planner demonstrations drops 55% — the model is fitting
  the teacher's policy distribution, not memorizing train sequences.
- **No play-level transfer at smoke scale**: exact-movement match and
  closed-loop rows are unmoved. At width 16 with 256 CPU steps this is the
  expected floor, not a failure signal.
- Consequence for the campaign: the DGX distillation run should track
  *validation teacher-agreement* as its early learning curve and
  closed-loop play as the late one — this probe pins both instruments and
  their zero-points. Scaling beyond smoke scale is DGX work by rule; the
  probe deliberately stops here.

## Verification

The script is the artifact; every number above is its printed output. The
full play-safe suite exits 0.
