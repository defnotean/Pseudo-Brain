# T1-3 — Expert behavior-cloning corpus on TRAIN (preregistration)

**Date:** 2026-08-28
**Status:** FROZEN (design contract; run artifacts go to `brain/runs/embodied-corpus-v1/`)
**Mode:** local CPU only, single-thread where determinism needs it, CUDA hidden.
**Upstream contract:** `brain/docs/preregistrations/2026-08-27-pacman-harness-v1.md`
(Tier-1 harness; §1 families, §3 partitions, §5 determinism/C8) and
`brain/docs/preregistrations/2026-08-25-realtime-embodied-qualification-v3.md`
(§2 interface, §6 human metric `M_P`, §7 Q2).
**Candidate that will consume this corpus (T2-1):** Core V2 W120/K32/C3 +
`EmbodiedInterfaceV1` heads, trained under the deployed decision loss
(`losses.deployed_decision_loss` → aggregated 5-class `action_values`).

This preregistration authorizes generating a behavior-cloning (BC) corpus from
the frozen pixel-only expert planner on the registered **TRAIN** seeds only.
It does **not** authorize model training, checkpoint publication, CPU-QUAL
materialization, or opening DEV/CAL/TEST. The expert is the **corpus generator**,
not the candidate and not the Q2/Q3 reactive baseline (that is T1-4, never
weakened).

## 1. The expert (teacher) policy — frozen

`diagnostic.scripted_maze_chase_planner.v1`
(`brain/src/irene_brain/evaluation/diagnostic_policies.py`,
`ScriptedMazeChasePlannerPolicy`). It is **pixel-only**
(`uses_privileged_state = False`): it derives walls, pellets, the player, and
visible ghost cells from the rendered 16×16 frame and plans a ghost-aware
pellet path by simulating the world's published mechanics (BFS shortest paths,
swept-path contact rule, actuation-aware knobs). It never reads simulator
coordinates, pellets, reward, or hazard state.

**Family-matched knobs (frozen):** the planner's
`ghost_period` / `player_period` / `ghost_elroy` are set to each family's
registered values (so its internal world-simulation matches the family's
published mechanics). `candidate_pellets=6`, `horizon=24`,
`input_delay_ticks=0` (the harness families all run `input_delay_ticks=0`).

**Honesty caveat (recorded, not a gate):** the planner models all non-direct
ghost rules as optimistic direct-pursuit, and a post-catch respawn cell is
sampled from hidden RNG, so its plan past a catch is approximate. This is an
expert *approximation* of the true mechanics, preserved verbatim as the BC
target. Its competence is measured below; it is the teacher, not a claim.

## 2. Corpus geometry (frozen)

- **Seed source:** the registered **TRAIN** partition only, from the
  `EMBODIED_BLOCK` of the harness prereg §3. For family `f` and corpus index
  `i ∈ [0, EP)` the seed is `partition_seeds(f, "TRAIN")[i]`
  (i.e. `201326592 + f*4096 + 500 + i`, since TRAIN spans base+500..base+1023).
  No DEV/CAL/TEST seed is ever used. No seed is resampled.
- **Episodes per family:** `EP = 50` (frozen). 50 < 524 available TRAIN seeds
  per family, leaving 474 reserved for corpus expansion.
- **Total episodes:** 5 families × 50 = **250**.
- **Episode mechanics:** each episode runs on the family's `MazeChaseEnv`
  configuration (the harness `DifficultyFamily.env_kwargs()` plus the
  family's `max_ticks` cap) until `terminated` (cleared) or
  `truncated` (`tick >= max_ticks`). Catch is a respawn (not a terminal), so
  episodes run to the clear or the `max_ticks` cap.
- **Transitions:** every native tick of every episode is one corpus
  transition (no temporal subsampling — the candidate acts every tick in
  closed-loop, so the corpus preserves the full 60 Hz control cadence).

## 3. Per-transition schema (frozen)

The model input boundary is the ordinary rendered frame plus the previous
applied control (v3 §8.1 / constitution C8). Nothing privileged is stored in
the **model-input** fields; evidence fields are logged for analysis only and
never enter the model.

Model-input fields (what the candidate trains on):
- `frame`: the `Observation.rgb` `RgbFrame` at the tick, **before** the
  teacher's action is applied (16×16×3 uint8, 768 bytes). This is exactly the
  frame the candidate sees in closed-loop; training upsamples it to 32×32 with
  the same `nearest` interpolation the deployed encoder uses
  (`training.objective._rgb_tensor`), so the stored 16×16 is lossless for the
  model.
- `prev_action`: the teacher's applied action class on the **previous** tick
  (int 0..4 = idle/W/A/S/D), the control context the candidate conditions on.
  Tick 0's `prev_action = 0` (idle).
- `target_action`: the teacher's applied action class on **this** tick
  (int 0..4). This is the BC supervision label (5-class cross-entropy against
  the candidate's `action_values`).

Evidence fields (analysis only, never model input):
- `family` (0..4), `episode_seed` (int64), `tick` (int32),
- `reward` (float32), `events` (a small int-encoded flag: 0 none, 1
  pellet_eaten, 2 caught, 3 cleared — the terminal/clear event),
- `terminated` (bool), `truncated` (bool).

Storage: a compact binary (`.npz` / `.npy` memory-mapped) under
`brain/datasets/embodied-corpus-v1/` (**gitignored** — the corpus is an
artifact, not source). The committed record is the generator script, the
deterministic manifest, and the acceptance report.

## 4. Determinism protocol (mandatory, verbatim from the harness prereg §5)

```
torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE_CONFIG=:4096:8
cudnn.deterministic=True, benchmark=False
tf32 off both paths
seed banks via zlib.crc32(name) — NEVER builtin hash()
```
The simulator is integer-deterministic (SplitMix64, BFS, exact float rewards)
and the planner is a pure function of (rendered frame, its own issued-press
FIFO). Two regenerations of the full corpus MUST byte-match the frame stream,
the `prev_action`/`target_action` streams, and the per-episode
`final_state_hash`. The acceptance check (§6, gate C) verifies this on a
fixed two-episode pair.

## 5. Causal-integrity invariant (C8, binding)

The `frame` field is produced **only** from `Observation.rgb` (the rendered
16×16 pixels). No ghost coordinate, pellet count, reward, hazard event,
cleared/caught flag, maze map, task label, or environment token is written to
any model-input field. A static audit (§6, gate D) proves the generator reads
no privileged `MazeChaseEnv` attribute to build a model-input field; the
planner itself is already pixel-only and carries
`uses_privileged_state = False`.

## 6. Acceptance gates for THIS corpus build (run-local, no model involved)

A corpus build is accepted only if all hold on the full 250-episode run:

1. **C — Determinism:** two independent regenerations of a fixed 2-episode
   pair (one F1, one F3, clean) byte-match the frame stream +
   `prev_action`/`target_action` streams + final env state hash.
2. **D — Causal boundary (C8):** the static audit passes — no privileged
   `MazeChaseEnv` attribute is read to build a model-input field; every stored
   `frame` equals the `Observation.rgb` for that (seed, tick) byte-for-byte
   (checked on a sample).
3. **E — Action-class validity:** every `target_action` and `prev_action`
   is in {0..4}; the mapping is via
   `trajectory_objective.control_action_class` (exclusive W/A/S/D or idle).
4. **F — Scale:** exactly 250 episodes (50 per family), seed sets equal to
   `partition_seeds(f, "TRAIN")[:50]`, disjoint from DEV/CAL/TEST.
5. **G — Expert competence floor (non-degenerate teacher):** the corpus
   generator records per-family `M_P` (v3 §6:
   `0.5*attained_score/max + 0.5*survival_ticks/max_ticks`, clipped [0,1]) and
   per-family attained-pellet fraction. **Floor (frozen):** every family's
   mean attained-pellet fraction ≥ 0.50, and at least 4 of 5 families have
   mean `M_P` ≥ 0.50. Rationale: this proves the teacher actually plays and
   clears pellets rather than idling/dying; it is a floor on the *teacher*,
   not a claim about the candidate. The N=8 dry-run measured mean
   attained-pellet fraction 0.77–1.00 and mean `M_P` 0.52–0.77 per family,
   so the floor is comfortably non-vacuous. (A family below the floor is a
   recorded datum that the teacher is weak on that family — it does NOT
   license re-tuning the planner or the families; it is preserved verbatim.)

Gate pass is recorded in `brain/runs/embodied-corpus-v1/
2026-08-28-corpus-construction-v1.json` (create-only). On any gate failure,
the build is NOT accepted, the failure is recorded verbatim, and the next
iteration is a new probe — no silent re-tune of §1/§2/§3.

## 7. What this preregistration does NOT do

- No model training, no checkpoint, no CPU-QUAL, no TEST opening.
- No change to `MazeChaseEnv` internals or the planner (both consumed as-is).
- No re-baselining of any sealed pin. No DGX/Spark dispatch (local CPU only,
  per the 2026-08-26 decision record).
- The reactive baseline (T1-4) is a **separate** policy, trained on the same
  TRAIN partition under the candidate's budget class; it is never weakened and
  is not part of this corpus.

## 8. Claim boundary

Passing §6 permits only: "a deterministic, causally-clean, non-degenerate
behavior-cloning corpus was generated from the frozen pixel-only expert on the
registered TRAIN seeds." It does not permit any "the model learned to play"
or "the model beats a baseline" claim — those require T2-1 (integrated
training), T2-2 (CPU-QUAL prereg), and the Tier-3 battery.

## 9. Next (queue)

1. **T1-3** (this) — run the generator; record the manifest + acceptance.
2. **T1-4** — reactive baseline (non-weakened) on TRAIN.
3. **T2-1** — integrate Core V2 + `EmbodiedInterfaceV1` heads under the
   deployed decision loss on this corpus; multi-seed; then **T2-2** CPU-QUAL
   prereg.
