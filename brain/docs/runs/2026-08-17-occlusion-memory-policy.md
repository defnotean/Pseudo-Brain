# Occlusion memory policy: the scripted persistence frontier (2026-08-17)

Status: exploratory evidence tooling. The policy row is a diagnostic
reference that maps the achievable scripted frontier on the occlusion
world; no model checkpoint is measured yet.

## What was added

**`ScriptedOcclusionMemoryPolicy`** (`evaluation/diagnostic_policies.py`,
identity `diagnostic.scripted_occlusion_memory.v1`) — a pixel-only
episodic-memory policy that joins the cross-world matrix as the eighth
non-privileged policy.

The occlusion world renders only the Chebyshev-radius-4 neighborhood of
the player; everything else is fog. The world was designed as the direct
environment-level test of the persistent thought-state thesis: a reactive
policy is structurally blind there. This policy demonstrates the positive
side — how much a *memory* alone is worth, with zero learning — by
deriving two kinds of persistent state from pixels across time:

- **Target memory.** The target is static until collected, and only the
  player can collect it. So a remembered target cell stays valid until
  the player walks onto it. The policy keeps exactly one cell of target
  memory: refreshed whenever the yellow pixel is visible, cleared when the
  player arrives (collection relocates the target; the new one is unknown
  unless it happens to render inside the window).
- **Hazard trajectory hypotheses.** Hazards move deterministically (unit
  diagonal, boundary bounce). Each unexplained red pixel seeds four
  velocity hypotheses anchored at the sighting tick; a hypothesis whose
  projected cell is visible but shows no hazard is contradicted and
  dropped. The true hypothesis is confirmed on every window pass and
  survives indefinitely, so the policy reconstructs exact hazard
  trajectories from pixels alone — and avoids even hazards currently
  inside the fog. The avoidance set is the union of surviving hypotheses,
  conservative by construction because the true hazard is always in it.
  Projection uses a closed-form per-axis reflection
  (`2 * (GRID_SIZE - 1)` period), matching the world's bounce rule.

When no target is remembered, the policy sweeps four quadrant waypoints
((3,3), (11,3), (11,11), (3,11)) whose radius-4 windows tile the grid,
guaranteeing acquisition within one lap. Move selection is a one-tick
exact simulation over the nine king moves: fewest hypothesis crossings,
then Chebyshev distance to the goal, then Manhattan distance, then fixed
scan order.

Two implementation findings worth recording:

1. **The first version orbited forever on seed 13** (1 target in 600
   ticks). King-move Chebyshev descent has wide tie plateaus, and a fixed
   lexicographic tie-break made the policy slide sideways along the
   plateau instead of converging; grazing waypoint neighborhoods then
   advanced the sweep before the quadrant window was fully observed, and
   the (15,7) target stayed inside the resulting blind ring indefinitely.
   Exact-arrival waypoints plus the Manhattan secondary tie-break fixed
   it: 1 → 37 targets on seed 13.
2. Activation gating uses the fog pixel `(2, 3, 5)`, which no other
   ladder world renders — until fog appears the policy holds still, so
   its foreign matrix rows are honest zeros.

## Canonical occlusion frontier

Seeds 5/9/13, 600 ticks per episode, three bouncing hazards, view radius
4. The full matrix is now 13 worlds × 8 policies = 104 cells, SHA-256
`26a28941308431836b7fd8ef636b324fc8a379aa6476311b12f94d9e9a4d9c64`
(superseding `311f2ebe84b650ea…`, 91 cells; all pre-existing cells are
unchanged).

| Policy | Reward | Targets | Collisions | Notes |
|---|---|---|---|---|
| noop | 0 | 0 | 0 | passive floor (hazards never find a still player) |
| random_movement | -44 | 4 | 48 | walks into hazards |
| scripted_chase | +1 | 1 | 0 | holds still whenever the target is fogged |
| pellet_teacher | 0 | 0 | 0 | no pellet concept |
| maze_chase_planner | 0 | 0 | 0 | no pellet pixels, never plans |
| keys_doors_solver | 0 | 0 | 0 | no key/door pixels, never activates |
| junction_solver | 0 | 0 | 0 | no wall+target+chaser combination |
| **occlusion_memory** | **+125** | **125** | **0** | **43/45/37 per seed** |

The reactive frontier on this world is one lucky target; the memory
frontier is 125 targets with zero collisions — a 125× gap that quantifies
exactly what persistent state is worth on this world, with no learning
involved. This is the reference row for the thought-state thesis: any
model that cannot carry a target memory and hazard beliefs across fog
cannot leave the reactive floor, and a model that carries both perfectly
approaches +125.

## Verification

`tests/test_diagnostic_policies.py` gains `OcclusionMemoryPolicyTests`
(6 tests): contract flags and reset validation, canonical-frame
rejection, per-seed determinism, collection under fog with zero
collisions and zero rejections on seeds 5/9, aggregated dominance over
the reactive chaser, and honest-zero stillness on moving_shapes,
junction, and maze_chase. `tests/test_cross_world_matrix.py` updates the
canonical policy list and cell counts (91 → 104). The full play-safe
suite exits 0 (518 tests).
