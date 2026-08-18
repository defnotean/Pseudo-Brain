# Keys-doors solver policy: the scripted planning frontier (2026-08-17)

Status: exploratory evidence tooling. The solver row is a diagnostic
reference that maps the achievable scripted frontier on the keys_doors
world; no model checkpoint is measured yet.

## What was added

**`ScriptedKeysDoorsSolver`** (`evaluation/diagnostic_policies.py`,
identity `diagnostic.scripted_keys_doors_solver.v1`) — a pixel-only
key → door → target solver that joins the cross-world matrix as the sixth
non-privileged policy.

The keys_doors world isolates ordered planning plus one piece of
unrendered episodic memory: whether the player holds the key is never
painted. The solver derives that memory the only honest way a model could,
from pixel disappearance across frames:

- The new `_parse_keys_doors_frame` helper reads only the public render
  contract (`WALL_RGB` / `PLAYER_RGB` / `KEY_RGB` / `DOOR_RGB` /
  `TARGET_RGB`). The world renders the key only while uncollected and the
  door only while closed, so once both have been seen, their *absence* in
  later frames is the remembered state. No simulator state is ever read.
- Phase 1 (key seen, still visible): walk the BFS shortest path to the
  key, treating the closed door cell as blocked — the world places the key
  on the player side by construction.
- Phase 2 (key collected — seen before, now absent — door still closed):
  walk to the door cell; stepping into it while holding the key opens it.
- Phase 3 (door open — seen before, now absent): walk the BFS shortest
  path to the target; the open door cell renders as an ordinary corridor.

There are no movers in this world, so no lookahead is needed; the only
stochasticity is the target relocation, which is visible in the very next
frame and absorbed by re-planning every tick. On worlds without key or
door pixels the solver never activates and holds still, so its matrix rows
there are honest zeros — no other world renders the key cyan
`(90, 220, 230)` or the door orange `(170, 110, 40)`.

## Canonical keys_doors frontier

Seeds 5/9/13, 600 ticks per episode. The full matrix is now 13 worlds × 6
policies = 78 cells, SHA-256
`4f10467fbecf539aa8c3da531898e83d894e5facbbe475d4dae52a18ec190ba6`
(superseding `3ca6cc8cd371f3…`, 65 cells; all pre-existing cells are
unchanged).

| Policy | Reward | Targets | Notes |
|---|---|---|---|
| noop | 0 | 0 | passive floor |
| random_movement | 0 | 0 | bumps walls and the closed door |
| scripted_chase | 0 | 0 | greedy target pursuit, defeated by the maze |
| pellet_teacher | 0 | 0 | no pellet concept |
| maze_chase_planner | 0 | 0 | no pellet pixels, never plans |
| **keys_doors_solver** | **+36** | **36** | **11/12/13 per seed, zero stalls** |

The world had scored a flat zero for every prior policy: reactive target
pursuit cannot route around maze walls, and nothing else had a key/door
concept at all. The solver collects 36 targets across the three seeds (11, 12, and 13
collections per 600-tick episode) — after the first key → door → target
sequence, the door stays open and relocating targets keep scoring —
pinning the frontier at *ordered three-step planning plus cross-frame
possession memory*. This is the
reference row a model must approach on keys_doors, and it quantifies what
the world's design doc promised: the binding skill is remembering an
unrendered fact (key possession) long enough to act on it.

## Verification

`tests/test_diagnostic_policies.py` gains `KeysDoorsSolverTests` (5
tests): contract flags and reset validation, canonical-frame rejection,
per-seed determinism, target collection on seeds 5/9 with zero rejections,
and honest-zero stillness on moving_shapes. `tests/test_cross_world_matrix.py`
updates the canonical policy list and cell counts (65 → 78). The full
play-safe suite exits 0 (506 tests, 41 files).
