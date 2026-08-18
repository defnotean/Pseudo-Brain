# Maze-chase planner frontier: 16-seed robustness (2026-08-17)

Status: exploratory evidence tooling. Diagnostic-policy evidence only; no
model checkpoint is measured.

## Question

The scripted-frontier claim so far rested on the three canonical matrix
seeds (5/9/13). Does the pixel-only lookahead planner hold up across a
wider seed set, or were those mazes lucky?

## Setup

Canonical matrix slot (`world.maze_chase.v1`: 3 ghosts, period 2, 16 extra
loops), 16 fresh seeds (100–115), 600 ticks per episode, one 60 Hz decision
per tick, zero latency. The planner runs with default (canonical) knobs;
the greedy pellet teacher is the reference point.

## Result

| Policy | Total reward | Catches | Clears |
|---|---|---|---|
| `diagnostic.scripted_maze_chase_planner.v1` | **+1,875** | **52** | **15/16** |
| `diagnostic.scripted_pellet_teacher.v1` | -39,397 | 4,066 | — |

- Every one of the 16 planner episodes is net-positive (per-seed reward
  92–142), and 15 of 16 mazes are fully cleared inside 600 ticks.
  Ticks-to-clear range 213–373 (mean ≈ 267); the one non-clear (seed 112)
  still ended positive (+105, 1 catch).
- The teacher loses every episode catastrophically (≈ -2,462 reward and
  ≈ 254 catches per episode), confirming on 16 seeds what the canonical
  row showed on 3: reflexive greed is not a viable maze_chase strategy,
  and the frontier gap is structural, not seed luck.

The scripted frontier on the canonical slot is therefore: **near-always
clear, single-digit catches**. A model row can now be read against a
robust reference: matching the frontier means ≥ 15/16 clears and ≤ ~3
catches per episode on this seed set.

## Verification

Numbers produced by `run_policy_closed_loop_episode` (the canonical
closed-loop evaluator) for reward/catches, cross-checked against direct
environment counters for pellets/clears — the two channels agree exactly,
as they did on the canonical seeds. The full play-safe suite exits 0.
