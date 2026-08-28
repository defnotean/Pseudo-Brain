# T1-3: Expert behavior-cloning corpus generated (2026-08-28)

**Date:** 2026-08-28
**Status:** CLOSED — all five corpus-construction gates PASS at the frozen scale
**Preregistration:** `brain/docs/preregistrations/2026-08-28-pacman-corpus-v1.md` (T1-3, frozen)
**Mode:** local CPU only, single-thread, CUDA hidden, deterministic protocol
**Base HEAD at start:** `ad36f39`

## What was built

1. `brain/scripts/run_pacman_corpus_v1.py` — the T1-3 corpus generator:
   - runs the frozen pixel-only expert
     (`diagnostic.scripted_maze_chase_planner.v1`, family-matched
     `ghost_period`/`player_period`/`ghost_elroy` knobs,
     `candidate_pellets=6`, `horizon=24`, `input_delay_ticks=0`) on the
     registered **TRAIN** seeds only;
   - stores, per native tick, the three model-input fields
     (`frame` = `Observation.rgb` 16×16×3 uint8; `prev_action`;
     `target_action` ∈ {0..4} via `trajectory_objective.control_action_class`)
     plus evidence-only fields (reward, event flag, terminated, truncated);
   - writes a compact artifact tree under
     `brain/datasets/embodied-corpus-v1/` (gitignored) and a committed
     `manifest.json` with per-episode records + a corpus SHA-256;
   - the create-only acceptance report at
     `brain/runs/embodied-corpus-v1/2026-08-28-corpus-construction-v1.json`.

## Gate results [MEASURED] (frozen scale: 250 episodes = 50/family, wall 692.0 s)

| Gate | Result | Evidence |
|---|---|---|
| C determinism | PASS | two independent regenerations of the fixed pair (F1 seed 201331188, F3 seed 201339380) byte-match the frame stream + prev/target action streams + final state hash; stored arrays byte-match the live replay |
| D causal boundary (C8) | PASS | static audit: the model-input construction lines read only `obs.rgb` and the teacher's emitted control; zero privileged `MazeChaseEnv` attribute reads in the model-input path |
| E action-class validity | PASS | every `prev_action`/`target_action` in {0..4} across all 250 episodes |
| F scale | PASS | exactly 250 episodes; per-family seed sets equal `partition_seeds(f,"TRAIN")[:50]`, disjoint from DEV/CAL/TEST |
| G expert-competence floor | PASS | per-family mean attained-pellet fraction ≥ 0.50 for all 5; ≥4 of 5 families mean M_P ≥ 0.50 (see table) |

**Corpus size:** 250 episodes, **507,768 transitions** (full 60 Hz control
cadence, no temporal subsampling). Artifact tree 383 MB on disk (gitignored).
`corpus_sha256 = 59c41b6dc71f…` (create-only; the report refuses to overwrite).

## Expert competence, per family [MEASURED] (the teacher, not a claim)

| Family | mean M_P | mean pellet-fraction | mean survival-fraction |
|---|---:|---:|---:|
| F0 calm      | 0.758 | 0.991 | 0.524 |
| F1 standard  | 0.525 | 1.000 | 0.050 |
| F2 pressured | 0.525 | 1.000 | 0.051 |
| F3 fast      | 0.684 | 0.842 | 0.525 |
| F4 open-slow | 0.766 | 0.768 | 0.764 |

(Full per-episode values in the manifest / result JSON.) Reading: F1/F2 are
"clear the maze fast under 3–4 fast ghosts" (pellet-fraction 1.00, short
episodes); F3/F4 survive longer (F4's slow `player_period=2` + 40 loops
stresses sequencing; F3's elroy endgame-speed-up + mixed ghosts is the
hardest). All families clear ≥ 0.77 of their pellets on mean — a competent,
non-degenerate teacher. The N=8 dry-run that set the G floor is preserved in
`brain/scratch/` (reproducible).

## Findings recorded

- **The G floor is non-vacuous and passed comfortably.** Mean attained-pellet
  fraction ranged 0.77–1.00 (floor 0.50); mean M_P ranged 0.52–0.94 (≥4 of 5
  families ≥ 0.50, actually all 5 ≥ 0.50). [MEASURED]
- **Episode length is bimodal:** most episodes clear fast (~200–600 ticks);
  some run to the `max_ticks` cap (4000–8000) when the player survives but
  cannot clear (F3/F4). Both regimes are preserved in the corpus — the
  candidate sees both "clear it" and "survive it" dynamics. [MEASURED]
- **No corpus was used to tune anything.** The floor is a predeclared gate on
  the *teacher*; passing it licenses the corpus as a valid BC source, it does
  not license any candidate claim.

## What this does NOT do

- No model training, no checkpoint, no CPU-QUAL, no TEST opening.
- No DEV/CAL/TEST seed used (TRAIN only). No re-baselining of any sealed pin.
- The reactive baseline (T1-4) is a separate policy, trained on the same TRAIN
  partition under the candidate's budget class; it is not part of this corpus
  and is never weakened.

## Next (queue)

- **T1-4** — reactive baseline (non-weakened), trained on TRAIN under the
  candidate's budget class; the Q2/Q3 comparison arm.
- **T2-1** — integrate Core V2 W120/K32/C3 + `EmbodiedInterfaceV1` heads under
  the deployed decision loss on this corpus; multi-seed; then **T2-2**
  CPU-QUAL prereg.
