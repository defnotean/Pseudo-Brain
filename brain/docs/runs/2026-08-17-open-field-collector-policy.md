# Open-field collector policy: both shared-signature frontiers (2026-08-17)

Status: exploratory evidence tooling. The policy row is a diagnostic
reference that maps the achievable scripted frontier on the two open-field
worlds; no model checkpoint is measured yet.

## What was added

**`ScriptedOpenFieldCollectorPolicy`** (`evaluation/diagnostic_policies.py`,
identity `diagnostic.scripted_open_field_collector.v1`) — a pixel-only
careful collector that joins the cross-world matrix as the ninth
non-privileged policy.

moving_shapes and pursuit share one pixel signature — no walls, no fog, a
yellow target, red movers — so a pixel-only policy cannot tell the two
worlds apart from one frame. What differs is the mover rule, and the
policy identifies it from observed motion (pixel-derivable behavioral
evidence, never simulator state):

- a full-set stay (red cells unchanged across a transition, counts
  included) happens every other tick in pursuit and practically never
  under diagonal bouncing → identifies pursuit;
- a red cell with no stay/orthogonal predecessor moved diagonally or
  jumped, which pursuit never does → identifies bouncing;
- transitions where a mover hides under the overlap pixel or a stack
  splits (the red count changes) carry no identity information and are
  skipped.

Until identification completes (one or two transitions) the policy avoids
the union of both rules' moves; afterward only the identified rule's: the
four diagonal bounce afters, or the pursuit pair {stay, greedy step toward
the candidate cell} — larger axis first, horizontal on ties, the world's
exact rule. Two more pixel-derived structures complete the policy:

- **Target memory.** The target renders *under* the movers, so a mover
  standing on the target cell hides it. The target is static until the
  player collects it, so one remembered cell keeps the goal alive while a
  camper hides it (the same disappearance-implies-state trick the
  keys_doors solver uses for key possession).
- **The lure.** A camping pursuer never leaves on its own — it always
  chases the player — so bare goal-seeking plus avoidance deadlocks in a
  hover (observed directly: seed 9 sat two cells from a camped target for
  570 ticks). Once the goal has been contested for twelve straight ticks
  with the player nearby, the goal switches to the farthest grid corner;
  the pursuers trail at half speed, and when no mover remains within two
  cells of the target the policy darts in with a safe margin.

Move selection is a one-tick exact simulation over the nine king moves:
fewest predicted collisions, then Chebyshev, then Manhattan, then fixed
scan order.

## Canonical open-field frontiers

Seeds 5/9/13, 600 ticks per episode. The full matrix is now 13 worlds × 9
policies = 117 cells, SHA-256
`e76c4dd3f16673cd5e60aa99653e43b7b6bb4880ec7a6c002b274a7362e31f91`
(superseding `26a2894130843183…`, 104 cells; all pre-existing cells are
unchanged).

| World | Policy | Reward | Targets | Contact |
|---|---|---|---|---|
| moving_shapes | scripted_chase | +186 | 237 | 51 collisions |
| moving_shapes | **open_field_collector** | **+236** | **236** | **0** |
| pursuit | scripted_chase | +134 | 219 | 85 catches |
| pursuit | **open_field_collector** | **+214** | **214** | **0** |

The collector gives up one target on moving_shapes and five on pursuit
relative to the greedy chaser and in exchange takes *zero* contact — the
avoidance premium is +50 reward on moving_shapes and +80 on pursuit. Both
open-field scripted frontiers are now pinned at *perfect safety with
near-greedy collection speed*, which is the reference row a model must
approach. The implementation record also documents three failure modes
that any learned policy must solve on these worlds: mover-rule
identification under a shared pixel signature, render-occlusion of the
target by a camper, and the hover deadlock that only a lure (a
deliberate temporary goal switch) escapes.

## Verification

`tests/test_diagnostic_policies.py` gains `OpenFieldCollectorTests`
(6 tests): contract flags and reset validation, canonical-frame rejection,
per-seed determinism, zero-contact collection on both worlds for seeds
5/9 with zero rejections, aggregated head-to-head dominance over the
greedy chaser on both worlds, and honest-zero stillness on occlusion,
junction, and maze_chase. `tests/test_cross_world_matrix.py` updates the
canonical policy list and cell counts (104 → 117). The full play-safe
suite exits 0 (524 tests).
