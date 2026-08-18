# Maze-chase planner policy: the scripted frontier (2026-08-17)

Status: exploratory evidence tooling. The planner row is a diagnostic
reference that maps the achievable scripted frontier on the canonical
maze_chase slot; no model checkpoint is measured yet.

## What was added

**`ScriptedMazeChasePlannerPolicy`** (`evaluation/diagnostic_policies.py`,
identity `diagnostic.scripted_maze_chase_planner.v1`) — a pixel-only
lookahead planner that joins the cross-world matrix as the fifth
non-privileged policy.

Where the pellet teacher is greedy (nearest pellet, one step of ghost
avoidance), the planner simulates the world's published mechanics ahead:

- It reads only the canonical observation frame — walls, pellets, the
  player, and the visible ghost-red cells — via the new shared
  `_parse_maze_chase_frame` helper (extracted from the pellet teacher with
  identical behavior; the teacher's error messages are unchanged).
- It reconstructs the shortest path to each of the six nearest visible
  pellets (`candidate_pellets=6`), then simulates walking each path end to
  end, capped at `horizon=24` ticks: ghosts step one cell toward the
  post-move player cell along the exact BFS shortest path every
  `ghost_period=2` ticks (one shared player-anchored field per move tick,
  exactly as the simulator computes it), and contact is checked with the
  same swept-path collision rule the simulator uses, for every ghost on
  every simulated tick.
- It commits to the first path it can walk without being caught and presses
  only that path's first step — re-planning from fresh pixels every tick.
- If no pellet path is safe, a one-tick survival fallback picks the move
  (four directions, then stay) that is not caught and maximizes the
  post-move BFS distance to the nearest ghost, ties broken by fixed scan
  order.

Honest pixel-vision caveats, documented in the class docstring: the
player/ghost overlap pixel briefly hides a ghost on the player's own cell,
and stacked ghosts render as one. Re-planning every tick bounds the damage.
The lookahead assumes the canonical slot's published mechanics (direct
pursuit, fixed period); on other ghost rules it degrades to optimistic
direct-pursuit guesses without ever reading simulator state.

## Canonical maze_chase frontier

Seeds 5/9/13, 600 ticks per episode, three half-speed direct ghosts,
16 extra loops, zero latency, one 60 Hz decision per tick. The full matrix
is now 6 worlds × 5 policies = 30 cells, SHA-256
`53949071414fd54a3e3129c3f463f46e31460107c407b86080c852ead9f14de6`
(superseding `627eb4df03d504d2…`, 24 cells; all pre-existing cells are
unchanged).

| Policy | Reward | Catches | Notes |
|---|---|---|---|
| noop | -590 | 62 | passive floor |
| random_movement | -728 | 91 | |
| scripted_chase | -590 | 62 | no pellet concept |
| pellet_teacher | -6,274 | 665 | 366 pellets, 1 clear |
| **maze_chase_planner** | **+426** | **3** | **426 pellets, 3/3 clears** |

(Teacher pellet count corrected from the earlier record's 376 to 366; the
reward and catch figures were and are exact, and 366 pellets is the only
count consistent with them.)

The planner is the first policy with positive maze_chase reward: it clears
all three canonical mazes inside 600 ticks and is caught only three times
across 1,800 ticks of play. The scripted frontier on the canonical slot is
therefore *full clears with rare catches* — this is the reference row a
model must approach before the arcade proof gate means anything, and it
confirms the earlier diagnosis: the binding skill is evasion planning under
pursuit, not pellet reflexes.

## Verification

`tests/test_diagnostic_policies.py` gains `MazeChasePlannerTests` (4 tests):
contract flags and constructor validation, canonical-frame rejection,
per-seed determinism, and a head-to-head assertion that the planner beats
the greedy teacher on both reward and catches over seeds 5/9.
`tests/test_cross_world_matrix.py` updates the canonical policy list and
cell counts (24 → 30). `tests/test_maze_chase.py` (33 tests) confirms the
teacher refactor is behavior-preserving. The full play-safe suite exits 0.
