# Cross-world diagnostic matrix (2026-08-17)

Status: exploratory evidence tooling. The matrix rows are diagnostic
references, not qualified results; no model checkpoint is measured yet.

## What was added

**`src/irene_brain/evaluation/cross_world_matrix.py`**: one canonical
evidence record covering every (world, policy) pair of the completed in-repo
ladder. Five canonical world slots (`world.moving_shapes.v1`,
`world.pursuit.v1`, `world.junction.v1`, `world.occlusion.v1`,
`world.keys_doors.v1`, each with a frozen knob set) × the three
non-privileged §28 diagnostic policies, run through the closed-loop
evaluator with identical timing mechanics. Each cell is a full
`ClosedLoopPlayReport` with its own SHA-256; the matrix has its own
canonical JSON and SHA-256. Privileged policies are rejected — the oracle
binds moving-shapes internals only, and privileged rows on the wrong world
would be meaningless.

## Reference matrix

Seeds 5/9/13, 600 ticks per episode, zero latency, one 60 Hz decision per
tick. Matrix SHA-256 `4e5f9e2bb89e57f5…`.

| World | Policy | Targets | Collisions | Reward |
|---|---|---|---|---|
| moving_shapes | noop | 0 | 0 | 0 |
| moving_shapes | random | 4 | 48 | -44 |
| moving_shapes | scripted_chase | 237 | 51 | +186 |
| pursuit | noop | 0 | 89 | -89 |
| pursuit | random | 5 | 89 | -84 |
| pursuit | scripted_chase | 219 | 85 | +134 |
| junction | noop | 0 | 18 | -18 |
| junction | random | 1 | 16 | -15 |
| junction | scripted_chase | 3 | 19 | -16 |
| occlusion | noop | 0 | 0 | 0 |
| occlusion | random | 4 | 48 | -44 |
| occlusion | scripted_chase | 1 | 0 | +1 |
| keys_doors | noop | 0 | 0 | 0 |
| keys_doors | random | 0 | 0 | 0 |
| keys_doors | scripted_chase | 0 | 0 | 0 |

The matrix already discriminates the skills the ladder was built for:

- **Pursuit** punishes passivity (no-op is caught 89 times) and greedy
  chase without avoidance (scripted collects 219 but is caught 85).
- **Junction** defeats greedy pixel-chasing (3 targets): walls demand
  pathing, not reflexes.
- **Occlusion** blinds the reactive teacher (1 target vs 237 on the same
  mechanics unfogged): the target is invisible beyond the view radius, so
  only persistent internal state can find it.
- **Keys/doors** scores zero for every reactive policy: nothing short of the
  ordered key → door → target plan collects anything.

Every future qualified model row now has a floor and a skill-discriminating
reference on all five worlds, in one citable canonical record.

## Verification

New `tests/test_cross_world_matrix.py` (9 tests, torch-free): canonical slot
and policy order, full 15-cell coverage, byte-level determinism of the
matrix SHA-256, no-op floor flat on every world, scripted chaser scoring on
moving shapes, privileged-policy rejection, input validation, canonical JSON
round-trip, and cross-config cell rejection. The full play-safe suite exits
0.
