# Pursuit world: first moving-shapes successor on the environment ladder (2026-08-17)

Status: implementation only. No dataset generation, training, or registration
uses this world yet; branch-DAG data generation at scale remains DGX work.

## What was added

1. **`src/irene_brain/environments/pursuit.py` — `PursuitEnv`**, the
   pursuit/evasion rung of PLAN.md §20 / ROADMAP §5. Same 16×16 one-pixel-per-
   cell render contract and W/A/S/D control surface as moving shapes, so
   every pixel-only teacher and the model input path work unmodified. The
   player collects relocating targets (+1) while deterministic greedy
   pursuers (larger axis first, horizontal on ties) close in every
   `pursuer_period` ticks; contact — tested with the same swept-path
   same-time intersection as moving shapes — emits `caught`, costs -1, and
   respawns the player on a sampled empty cell. Pursuers spawn at Manhattan
   distance ≥ 4 via deterministic rejection sampling. Full branchable-core
   rigor: explicit SplitMix64 state (reused from `moving_shapes`), canonical
   checksummed version-1 snapshots (magic `IBPS`), atomic validated restore
   with configuration-mismatch rejection, and `state_hash` over the canonical
   bytes. Constructor knobs: `pursuer_count` (default 2), `pursuer_period`
   (default 2 — pursuers at half player speed), `tick_period_ns`,
   `max_ticks`.
2. **World-generalized closed-loop evaluator.** `_run_episode_core` and both
   public runners (`run_closed_loop_episode`,
   `run_policy_closed_loop_episode`, plus `evaluate_closed_loop_play`) accept
   an optional `environment_factory`, defaulting to the config-driven
   moving-shapes world. The episode report's `collisions` column now counts
   contact under either event name (`collision` in moving shapes, `caught`
   in pursuit), keeping diagnostic and model rows column-comparable across
   worlds.

## Verification

New `tests/test_pursuit.py` (18 tests, torch-free): protocol satisfaction,
same-seed pixel/state identity, seed sensitivity, spawn-distance invariant
over 20 seeds, pursuers deterministically close on and catch a stationary
player (reward -1, `caught` event), `pursuer_period=2` measurably slows the
chase, catch respawns onto an empty cell, target collection scores and
relocates, a 1,000-step replay is exact against three pinned hashes (initial
observation `1e34b3e9…`, final observation `247b3faa…`, final state
`3cf6301d…`), atomic restore on corruption, restore rejects
count/period/max-ticks mismatches, terminal boundary, invalid control,
constructor and observation-field validation — and three cross-world
closed-loop tests proving the evaluator drives the pursuit world for both
the no-op (gets caught) and scripted-chaser (collects, deterministically)
policies. The full play-safe suite exits 0.

## What this unblocks

The environment ladder's second rung is playable by anything that satisfies
the policy contract or the model interface, with the same deadline/latency
mechanics and canonical evidence records as moving shapes. Remaining ladder
rungs: junction choice and occlusion worlds, then branch-DAG data generation
at scale on the Spark.
