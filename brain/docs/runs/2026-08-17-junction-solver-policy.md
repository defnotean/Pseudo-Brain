# Junction solver policy: the scripted pursuit-planning frontier (2026-08-17)

Status: exploratory evidence tooling. The solver row is a diagnostic
reference that maps the achievable scripted frontier on the junction
world; no model checkpoint is measured yet.

## What was added

**`ScriptedJunctionSolver`** (`evaluation/diagnostic_policies.py`,
identity `diagnostic.scripted_junction_solver.v1`) — a pixel-only
chaser-aware solver that joins the cross-world matrix as the seventh
non-privileged policy.

Junction isolates branch-point decisions under pursuit on a perfect maze:
the corridor path between any two cells is unique, the chaser walks the
exact BFS shortest path to the player every `chaser_period=2` ticks, and
contact costs one reward plus a hidden-RNG respawn. The solver reads
nothing but the canonical observation frame — walls, the player, the
yellow target, and the visible chaser-red cells, via the new
`_parse_junction_frame` helper — and simulates the world's published
mechanics before committing:

- It reconstructs the (unique) BFS shortest path to the visible target and
  simulates walking it one cell per tick while the chaser steps downhill
  on the post-move player field — reusing `MazeChaseEnv._step_downhill`,
  whose tie-break is byte-identical to the junction chaser rule — with the
  same swept-path contact test the simulator runs. It commits to the
  path's first step only if the whole walk survives.
- If the walk is caught, a one-tick survival fallback picks the move
  (four directions, then stay) that is not caught and maximizes the
  post-move BFS distance to the nearest chaser, ties broken by fixed scan
  order.

Activation gating keeps foreign rows honest: the solver activates only
when walls, a yellow target, and chaser red appear together — the
open-field worlds have no walls, maze_chase has pellets instead of a
target, and keys_doors has no red. Until then it holds still.

## Structural facts the frontier exposes

Two properties of the world fall out of its published mechanics and shape
what any policy — scripted or learned — can achieve:

1. **A guarded target is unreachable.** On a perfect maze the player →
   target path is unique, and a chaser sitting on that path stays on it
   while chasing (its path to the player always heads away from the
   target). The corridor is one cell wide, so passing is impossible:
   contact is eventually forced.
2. **The player outruns an unguarded chase.** The player moves every tick;
   the chaser moves every second tick. When the chaser is not on the
   target path, raw speed usually settles the question.

Consequence: when the target is guarded, kiting only delays the forced
catch; the catch itself re-rolls the geometry through the hidden-RNG
respawn. The solver never walks into the chaser on purpose — it kites and
accepts the cornered catch — so the frontier row below is a *no deliberate
deaths* reference. A reward-maximizing agent could in principle take
guarded-state catches immediately to re-roll faster; quantifying that
delta is future diagnostic work.

## Canonical junction frontier

Seeds 5/9/13, 600 ticks per episode, one half-speed chaser. The full
matrix is now 13 worlds × 7 policies = 91 cells, SHA-256
`311f2ebe84b650ea9c024e57b4cb16711b0e4336ebd408e4cfdbb02d5bd05e11`
(superseding `4f10467fbecf539…`, 78 cells; all pre-existing cells are
unchanged).

| Policy | Reward | Targets | Catches | Notes |
|---|---|---|---|---|
| noop | -18 | 0 | 18 | passive floor |
| random_movement | -15 | 1 | 16 | |
| scripted_chase | -16 | 3 | 19 | greedy Manhattan, defeated by walls |
| pellet_teacher | -18 | 0 | 18 | no pellet concept |
| maze_chase_planner | -18 | 0 | 18 | no pellet pixels, never plans |
| keys_doors_solver | -18 | 0 | 18 | no key/door pixels, never activates |
| **junction_solver** | **+10** | **22** | **12** | **10/5/7 targets per seed** |

Every prior policy scored negative on junction — the best (greedy chase)
managed 3 targets against 19 catches. The solver collects 22 targets
against 12 catches (4 per seed, all forced cornerings), the first
positive-reward row on this world. The scripted frontier is therefore
*unique-path planning plus pursuit timing*: the reference a model must
approach on junction, and the middle rung between keys_doors (static
ordered planning) and maze_chase (full evasion planning under pursuit).

## Verification

`tests/test_diagnostic_policies.py` gains `JunctionSolverTests` (6 tests):
contract flags and constructor validation, canonical-frame rejection,
per-seed determinism, aggregated head-to-head dominance over the greedy
chaser on seeds 5/9, target collection under pursuit with zero rejections,
and honest-zero stillness on moving_shapes, keys_doors, and maze_chase.
`tests/test_cross_world_matrix.py` updates the canonical policy list and
cell counts (78 → 91). The full play-safe suite exits 0 (512 tests).
