# Keys/doors world: multi-step planning on the environment ladder (2026-08-17)

Status: implementation only. No dataset generation, training, or registration
uses this world yet.

## What was added

**`src/irene_brain/environments/keys_doors.py` — `KeysDoorsEnv`**, the
keys/doors rung of PLAN.md §20 / ROADMAP §5, completing the named in-repo
ladder. A perfect maze (the junction world's seeded carver, reused) with one
key, one door, and one target:

- The target spawns at the BFS-farthest cell from the player; the door sits
  at the midpoint of the unique player→target path (the maze is perfect, so
  blocking one path cell provably separates the target — verified at reset);
  the key spawns at the farthest cell reachable with the door closed.
- Walls and the closed door refuse movement. Walking onto the key collects it
  (`key_collected`); stepping into the door holding the key opens it
  permanently and moves through (`door_opened`); reaching the target scores
  +1 and relocates it (`target_collected`).
- Key possession is deliberately **not rendered**. The frame shows the key
  cell until pickup and the door until it opens; remembering that you hold
  the key is part of the task, alongside junction choice.
- Snapshot (magic `IBKS`, version 1) covers key/door positions and both
  flags, with consistency validation (an open door without the key, or a
  held key still placed on the map, is rejected) and atomic restore.

## Verification

New `tests/test_keys_doors.py` (15 tests, torch-free): protocol satisfaction,
same-seed identity, layout invariants across ten seeds (door separates
player/key side from target side), door refusal without the key (on a seed
where the approach path does not cross the key), the full
key → door → target sequence scoring through the step mechanics, the key
pixel disappearing after pickup, 1,000-step replay exactness against three
pinned hashes (initial observation `e1561ed2…`, final observation
`daed54d3…`, final state `cdcf9ec1…`), atomic restore on corruption,
key/door state surviving restore, tampered-flag rejection, terminal
boundary, invalid control, constructor and observation-field validation, and
a deterministic closed-loop episode through the evaluator. The full
play-safe suite exits 0.

## Ladder status

All four named in-repo successor worlds are implemented: pursuit/evasion,
junction choice, occlusion, keys/doors — all behind the shared render and
control contract and all playable through the closed-loop evaluator's
`environment_factory`. Next ladder work: branch-DAG data generation at scale
on the Spark, then external world adapters (XLand-MiniGrid first) and the
`maze_chase` Phase 4 environment.
