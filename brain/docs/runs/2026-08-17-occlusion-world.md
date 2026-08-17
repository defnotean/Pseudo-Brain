# Occlusion world: fog of war on moving-shapes mechanics (2026-08-17)

Status: implementation only. No dataset generation, training, or registration
uses this world yet.

## What was added

**`src/irene_brain/environments/occlusion.py` — `OcclusionEnv`**, the
occlusion rung of PLAN.md §20 / ROADMAP §5. The mechanics are exactly the
moving-shapes world (bouncing hazards, relocating targets, one-cell W/A/S/D
movement, swept-path same-time contact), but the rendered frame reveals only
the Chebyshev `view_radius` neighborhood around the player (default 4,
max 8). Everything else — including hazards that were visible a tick ago —
is fog (`FOG_RGB`, public render contract). Visibility never affects
mechanics: an unseen hazard still collides, which is the point. A purely
reactive policy is structurally blind in this world, so it is the direct
environment-level test of the persistent thought-state thesis (the
`reactive.v1` / `reset_slots.v1` baselines vs the reference, at the world
level).

Snapshot identity includes the view radius (magic `IBOS`, version 1, with a
zero-pinned reserved byte), so a checkpoint taken under one observability
regime cannot silently restore into another. All other rigor matches the
ladder: explicit SplitMix64 state, atomic validated restore,
configuration-mismatch rejection, `state_hash`.

## Verification

New `tests/test_occlusion.py` (13 tests, torch-free): protocol satisfaction,
same-seed pixel/state identity, exact fog geometry (every cell beyond the
radius is fog, every cell within it is not), hidden hazards stay invisible
yet still move and collide, the fog window follows the player, 1,000-step
replay exactness against three pinned hashes (initial observation
`639c56e9…`, final observation `1b6a0f39…`, final state `f4e9944b…`),
atomic restore on corruption, view-radius/count/max-ticks mismatch
rejection, terminal boundary, invalid control, constructor and
observation-field validation, and a deterministic closed-loop episode
through the evaluator's environment factory. The full play-safe suite exits
0.

## Ladder status

Implemented: moving shapes, pursuit/evasion, junction choice, occlusion.
Remaining in-repo rung: keys/doors; then branch-DAG data generation at scale
on the Spark.
