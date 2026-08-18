# Cross-world matrix: maze_chase variant slots (2026-08-17)

Status: exploratory evidence tooling. The matrix rows are diagnostic
references, not qualified results; no model checkpoint is measured yet.

## What was added

The cross-world matrix (`evaluation/cross_world_matrix.py`) now promotes
every registered maze_chase variant axis to its own world slot, appended
after the six ladder slots so all pre-existing cells stay byte-identical:

| Slot | Changed knob |
|---|---|
| `world.maze_chase.ambush.v1` | `ghost_rule="ambush"` |
| `world.maze_chase.shy.v1` | `ghost_rule="shy"` |
| `world.maze_chase.mixed.v1` | `ghost_rule="mixed"` |
| `world.maze_chase.elroy.v1` | `ghost_elroy=True` |
| `world.maze_chase.slow_player.v1` | `player_period=2` |
| `world.maze_chase.delayed_input.v1` | `input_delay_ticks=2` |
| `world.maze_chase.sticky.v1` | `sticky_direction=True` |

All other knobs stay canonical (3 ghosts, period 2, 16 extra loops). The
matrix is now 13 worlds × 5 policies = 65 cells.

## Canonical variant frontier

Seeds 5/9/13, 600 ticks per episode. Matrix SHA-256
`3ca6cc8cd371f3e2bbb2799bcd4536dc0b9e3721ff515c6273ec0052e4ff4793`
(superseding `53949071414fd54a…`, 30 cells; the original 30 cells are
unchanged — spot-checked: all five canonical maze_chase rows reproduce
exactly). Planner/teacher rows, reward (catches):

| World slot | pellet_teacher | maze_chase_planner |
|---|---|---|
| maze_chase (canonical) | -6,274 (665) | **+426 (3)** |
| ambush | +416 (4) | +415 (3) |
| shy | +446 (1) | +408 (0) |
| mixed | -9,028 (936) | **+395 (5)** |
| elroy | -8,853 (920) | **+244 (20)** |
| slow_player | -8,263 (831) | **+226 (20)** |
| delayed_input | -8,326 (838) | -147 (52) |
| sticky | -6,274 (665) | **+426 (3)** |

What the variants discriminate:

- **Ambush inverts the greedy lesson.** Ambush ghosts target four cells
  ahead of the player's control-intent facing, so a still player is nearly
  safe (no-op: -65 vs -590 canonical) and the greedy teacher's constant
  facing-changes dodge into empty lead cells by accident (+416). Rule
  identity, not reflexes, decides this slot.
- **Shy ghosts never catch a still player** (no-op: 0 catches; even random
  walks to +111). The planner's zero-catch row is the cleanest in the
  matrix.
- **Mixed rules restore full pressure** (no-op -598) and the teacher's
  worst row anywhere (-9,028, 936 catches): cycling pursuit rules punish
  single-rule reflexes. The planner — which simulates only direct pursuit —
  still clears (+395, 5 catches), because conservative direct-pursuit
  clearance distances mostly cover the other rules' approaches.
- **Elroy and slow_player degrade the planner gracefully** (+244/+226, 20
  catches each): its period-2, every-tick-player model goes stale exactly
  where the mechanics say it should, yet re-planning every tick keeps both
  rows strongly positive.
- **Delayed input is the planner's only negative slot** (-147, 52 catches):
  controls apply two ticks late, so a policy that assumes immediate
  actuation plans into ghosts. This is the honest signature of an
  actuation-latency model the planner does not have — and a preview of what
  the model's own inference latency will do without deadline-aware play.
- **Sticky input is free** for a re-planning policy: each tick's fresh press
  replaces the latch, so the planner row is byte-identical to canonical.

## Verification

`tests/test_cross_world_matrix.py` updates the canonical slot list and cell
counts (30 → 65). The full play-safe suite exits 0. The canonical 65-cell
matrix run completes in ~26 s on this machine.
