# Rights ledger

One entry per environment source, tracking the four records PLAN.md §20
requires: code license, asset/ROM rights, trajectory/annotation license,
and model-training/redistribution terms. No public-data ingestion begins
until a source's record here reads accepted. Entries are dated on acceptance
and never edited silently: corrections append a dated amendment line.

## In-repository worlds — accepted (original works)

All six in-repo worlds are original works authored inside this repository:
procedurally generated layouts from episode seeds, programmatically
rendered one-pixel-per-cell frames, no third-party code, assets, names, or
layouts. Their module docstrings carry the rights-clean-by-construction
statement, and their tests pin the exact render and state contracts.

| Source | Code license | Asset/ROM rights | Trajectory/annotation license | Training/redistribution terms | Accepted |
|---|---|---|---|---|---|
| moving shapes (`environments/moving_shapes.py`) | Project license; original in-repo code | Original: generated shapes and colors; no external assets | Original: all trajectories generated in-repo by registered policies | Unrestricted for project training and redistribution | 2026-08-17 |
| pursuit (`environments/pursuit.py`) | Same as above | Same as above | Same as above | Same as above | 2026-08-17 |
| junction (`environments/junction.py`) | Same as above | Original: seeded perfect-maze carver written in-repo | Same as above | Same as above | 2026-08-17 |
| occlusion (`environments/occlusion.py`) | Same as above | Original: fog-of-war rendering written in-repo | Same as above | Same as above | 2026-08-17 |
| keys/doors (`environments/keys_doors.py`) | Same as above | Original: key/door mechanics written in-repo | Same as above | Same as above | 2026-08-17 |
| maze_chase (`environments/maze_chase.py`) | Same as above | Original: chase-maze mechanics written in-repo; deliberately no Namco code, assets, names, characters, or layouts; ghost rules (direct/ambush/shy/mixed) are original implementations of generic pursuit concepts | Same as above | Same as above | 2026-08-17 |

Design note for maze_chase: the chase-arcade genre conventions it draws on
(maze, pellets, pursuing ghosts, an endgame speed-up) are unprotectable
mechanics; every expression of them here — maze generator, movement rules,
timings, colors, reward values, snapshot format — is original to this
repository. The `ghost_elroy` knob is named after the arcade convention as
a comment-level homage only; its behavior (one tick faster after half the
pellets, minimum period 1) is defined wholly in-repo.

## External sources — not reviewed, not ingested

No external environment is used for training, evaluation, or dataset
generation today. Each of the following remains unreviewed; ingestion is
blocked until its four records above are filled in and accepted here.

| Source | Status |
|---|---|
| XLand-MiniGrid | Not reviewed; not ingested |
| Craftax | Not reviewed; not ingested |
| Procgen | Not reviewed; not ingested |
| Craftium/Luanti (with original or cleared asset pack) | Not reviewed; not ingested |
| Minecraft (late external evaluation only) | Not reviewed; not ingested |
| ALE/Atari (separate code, ROM, recording, automation, and training-rights review) | Not reviewed; not ingested |
| Namco Pac-Man (rights-reviewed external validation only, never a training foundation) | Not reviewed; not ingested |

## Amendments

- 2026-08-17: ledger created with the six in-repo worlds accepted and all
  external sources marked unreviewed.
