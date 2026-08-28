# Preregistration — Pac-Man-like v3 §4 clock/clock-queue harness (T1-1)

**Date:** 2026-08-27
**Status:** FROZEN (this file is the preregistration; run artifacts go to `brain/runs/embodied-harness-v1/`)
**Mode:** local CPU only, single-thread where determinism needs it, CUDA hidden.
**Scope:** Construction + acceptance contract for the Pac-Man-like v3 §4
no-pause 60 Hz harness layer, the five registered difficulty families, the
six registered perturbations, and the registered TRAIN/DEV/CAL/TEST seed
partitions. This is **construction** (deterministic simulator + harness
wrapper + baselines), NOT a qualification run: it authorizes no model
evaluation, no CPU-QUAL, and no TEST opening. The model-side battery is the
separate Tier-2/Tier-3 preregistrations.

## 0. What already exists (baseline, not rebuilt here)

- `environments/maze_chase.py` (`MazeChaseEnv`, SNAPSHOT_VERSION 4):
  deterministic 16×16 loop-carved maze, pellets (+1 each), clear = terminated
  win, BFS ghosts with rules `direct`/`ambush`/`shy`/`mixed`, `ghost_period`,
  `player_period`, `ghost_elroy`, `input_delay_ticks`, `sticky_direction`,
  checksummed snapshot/restore. Tick period `DEFAULT_TICK_PERIOD_NS = 16_666_667`
  (60 Hz). Original, rights-clean.
- `v2/embodied_interface.py` (`EmbodiedInterfaceV1`, identity
  `irene.brain.embodied_interface.v1`): frozen record schema + stateless
  adapter + supervised heads on the frozen Core V2; Q0 component prequal
  green 6/6 (`brain/runs/embodied-interface-v1-q0-prequal/`).
- `evaluation/closed_loop_play.py`: deterministic manual-clock closed-loop
  evaluator with declared simulated latency and `LatencyEvidence`.

**What is MISSING (this preregistration builds it):**
1. a §4-compliant **clock/queue harness** (cycle/frame/action IDs, monotonic
   receipt events, no-pause schedule, capacity-1 latest-complete-frame queue,
   deadline/miss/jitter accounting);
2. the **five difficulty families** as a frozen parameterization of
   `MazeChaseEnv`;
3. the **six registered perturbations** applied in a deterministic,
   evidence-logging wrapper;
4. the **registered seed partitions** (layout families × partitions);
5. the **expert/scripted behavior-cloning teacher** and the **reactive
   baseline** (never weakened).

## 1. Frozen difficulty families (five)

Each family is a frozen `MazeChaseEnv` constructor parameterization. A layout
family is defined by its (extra_loops, ghost_rule, ghost_period, ghost_count)
tuple; the episode seed then varies the actual maze carve within the family.
All families run `player_period=1`, `ghost_elroy=False`, `input_delay_ticks=0`,
`sticky_direction=False` (the clean baseline mechanics; delay/sticky belong to
the perturbation axes, not the difficulty axis).

| Family | Name | extra_loops | ghost_rule | ghost_period | ghost_count |
|---|---|---:|---|---:|---:|
| F0 | `open-calm` | 40 | shy | 4 | 2 |
| F1 | `standard` | 24 | mixed | 3 | 3 |
| F2 | `dense-fast` | 16 | direct | 2 | 3 |
| F3 | `tight-ambush` | 10 | ambush | 2 | 4 |
| F4 | `choke-elroy-off` | 6 | direct | 1 | 4 |

Rationale [HYPOTHESIS, preregistered]: F0–F4 form a monotone-difficulty
ladder (more loops + faster + more + meaner ghosts = harder). These are
frozen now and are NOT to be re-tuned against later results. If a family
turns out to be trivially clearable or unwinnable at corpus scale, that is a
recorded datum, not a reason to re-open the table.

## 2. Frozen six perturbations (P0–P5)

Applied by the harness wrapper to the clean per-family mechanics, on
registered episodes only:

| ID | Name | Deterministic implementation |
|---|---|---|
| P0 | palette/texture shift | add a fixed per-family 8-bit offset (frozen table §4) to every rendered pixel channel, clipped [0,255]; no geometry change |
| P1 | exactly 10% frame loss | drop the rendered frame every 10th frame id (deterministic stride; the queue keeps the latest *complete* frame — a dropped frame is logged as `frame_lost` and the previous complete frame is reused) |
| P2 | one-frame observation delay | enqueue every frame one native cycle late (a fixed 16,666,667 ns offset applied at enqueue) |
| P3 | game speed 90% | scale tick period to `round(16_666_667/0.90)=18_518_519` ns on BOTH the game and the controller schedule (both slow identically; no pause) |
| P4 | game speed 110% | scale tick period to `round(16_666_667/1.10)=15_151_515` ns on both (both fast identically) |
| P5 | altered enemy-policy | swap the ghost rule of the *clean* family to its registered alternate: F0 shy→direct, F1 mixed→direct, F2 direct→ambush, F3 ambush→direct, F4 direct→shy (frozen per-family, §2 table) |

P0 offset table (per family, applied to (r,g,b)): F0 (+10,-6,+4), F1 (-8,+12,+0),
F2 (+15,+15,-10), F3 (-12,-4,+18), F4 (+6,-14,-6). Frozen.

No perturbation may pause, slow asymmetrically, or inject privileged state.
Every perturbed tick is logged with its perturbation id, frame id, and cycle
id so Q3's paired analysis can bind it.

## 3. Clock / queue / no-pause contract (v3 §4, verbatim obligations)

The harness assigns:
- `cycle_id` to every native control cycle (monotonic from 0 at process start),
- `frame_id` to every rendered observation (monotonic; a dropped frame under
  P1 still consumes an id),
- `action_id` to every model output.

Receipt events (host `time.perf_counter_ns`, one process-wide monotonic clock):
`t_emit` (frame complete/readable), `t_enqueue` (enters obs queue), `t_dequeue`
(exact frame removed), `t_infer_start`, `t_infer_end`, `t_dispatch` (bound
action sent), `t_accept` (game-side hook acknowledges `action_id` for the
target native cycle). `epsilon=0` for all host-clock events (single clock
domain; no cross-domain affine fit needed for Pac-Man-like in-process play).

`L_plus = (t_accept - t_emit) + 0 + 0`.

Schedule: `s_n = t_0 + floor((n*1_000_000_000 + 30)/60)` ns; deadline
`d_n = s_{n+1}`; period `P_n = d_n - s_n`. At `s_n` the controller dequeues
the newest frame with `t_enqueue <= s_n` (capacity 1, latest-complete-frame;
overwrite/duplicate/stale/empty-dequeue all logged). A miss is
`t_accept > d_n`; a late action is retained and the preceding action held.
Jitter: `J_plus_n = |(t_accept_n - t_accept_{n-1}) - (s_n - s_{n-1})|`.
Nearest-rank quantiles over all eligible ticks. The game clock never pauses.

**Frozen timing gates (Pac-Man-like), copied from v3 §4.3 verbatim:**

| Gate | Limit |
|---|---:|
| Median `L_plus` | <= 8.00 ms |
| p95 `L_plus` | <= 13.00 ms |
| p99 `L_plus` | <= 16.00 ms |
| Deadline-miss rate | <= 0.1% |
| p99 `J_plus` | <= 2.00 ms |
| Consecutive misses | 0 |

## 4. Registered seed partitions

Layout family × partition, disjoint by seed. Base seed block
`EMB0 = 201_326_592 = 12 * 2^24` (a fresh 24-bit-aligned block; no overlap with
the sealed PB21* blocks up to `184_550_144 = 11*2^24+768`, the RCQ family, or
any range in `RCQ_V2_PROTOCOL.md`). Within the block, each partition gets a
contiguous, non-overlapping sub-range; the offset is a fixed function of the
family and partition index (no random selection).

| Partition | Sub-range size | Use |
|---|---:|---|
| TRAIN | 200,000 | behavior-clone corpus + integrated training |
| DEV | 50,000 | early-stop / selection (only) |
| CAL | 50,000 | threshold/adapter/quantization freeze; calibration |
| TEST | 400,000 | opened exactly once, after CAL freeze, under a separate create-only registration |

Within each partition, the 40 Q2/Q3 TEST layouts (8 per family, 5 families)
are the first 40 seeds of the TEST sub-range, in family order F0..F4. The 5
stochastic Q3 replicates per layout are the next 5 seeds after the 40
layout seeds. All other seeds in the partition are reserved for corpus
expansion and Q1 endurance (two independent 2-hour runs each draw a fresh
disjoint 2,000-seed slice from TRAIN and CAL respectively).

**Causal-integrity rule (C8, binding):** the model input on every tick is
exactly `ModelObservation` (frame, elapsed, age, previous control, dropped-frames
counter) — no future frame, no privileged coordinate/map/reward/hazard/task
label, no environment or task token. The harness MAY log privileged state for
evidence/Q2 scoring, but that log never enters the model.

## 5. Determinism protocol (mandatory)

Every run:
```
torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE_CONFIG=:4096:8
cudnn.deterministic=True, benchmark=False
tf32 off both paths
seed banks via zlib.crc32(name) — NEVER builtin hash()
```
The simulator is integer-deterministic (SplitMix64, BFS, exact float rewards);
two runs with the same seed+family+perturbation MUST byte-match the full
event log. A harness acceptance check (§7) verifies this on a fixed pair.

## 6. What "expert" and "reactive baseline" mean here (frozen)

- **Expert (teacher) policy** = the `diagnostic.scripted_maze_chase_planner`
  pixel-only lookahead planner (existing, published mechanics, actuation-aware
  knobs), run offline to emit (frame, prev_action, record) behavior-clone
  triples. It is the corpus generator; it is NOT the candidate and NOT the
  baseline the candidate must beat at Q2.
- **Reactive baseline** = a no-cross-step-state policy: on each tick, choose
  the locomotion that (a) avoids the closest visible ghost cell within
  Chebyshev distance 3 if one exists, else (b) moves toward the nearest
  visible pellet, else NOOP. No memory, no lookahead, no planner. Implemented
  as a pixel-only reader (it may only see the same `ModelObservation` the
  candidate sees). **Never weakened**: if it scores below random, that is a
  recorded datum, not a reason to lower it.

## 7. Acceptance gates for THIS construction (run-local, no model involved)

A harness build is accepted only if all of the following hold on a fixed
2,000-episode probe slice (seeds from TRAIN, all 5 families, all 6
perturbations + clean):

1. **Determinism:** two independent runs of the same slice byte-match the
   canonical event log (sha256 of the JSON-lines of cycle/frame/action ids +
   reward + events + all timestamps modulo the monotonic clock offset).
2. **No-pause:** zero `pause`/`slow`/`frame-advance`/`wait-for-inference`
   events in the log; the game clock advances every cycle.
3. **Schedule fidelity:** over the slice, deadline-miss rate and
   consecutive-miss count are exactly 0 for the *harness itself* (a
   no-op policy that just re-emits the neutral record); the L_plus/J_plus
   distribution is produced and reported (not gated here — the model gates
   are Tier 3).
4. **Causal boundary:** a static audit proves no field of the privileged
   event log (ghost coordinates, pellet map, reward, hazard, task id,
   family id) is present in any `ModelObservation` handed to a model hook.
5. **Perturbation isolation:** for each P1..P5, exactly the registered
   number of perturbation events occurs (P1: exactly 10% of frames; P2:
   exactly 1 delay; P3/P4: the two registered periods; P5: the registered
   rule swap; P0: the registered per-pixel offset) — no more, no less.
6. **Legal-action surface:** every emitted record decodes through
   `translate_record` to keys inside `ACTUATED_KEY_SET`; no record is
   rejected by `EmbodiedInterfaceRecord.from_dict`.

Gate pass is recorded in `brain/runs/embodied-harness-v1/
2026-08-27-harness-construction-v1.json` (create-only). On any gate failure,
the build is NOT accepted, the failure is recorded verbatim, and the next
iteration is a new probe — no silent re-tune of §1/§2/§3/§4.

## 8. What this preregistration does NOT do

- No model training, no CPU-QUAL, no TEST opening.
- No re-baselining of any sealed V2.1i / baseline-architecture pin.
- No change to `maze_chase.py` internals (it is consumed as-is; the harness
  is a wrapper). If a `maze_chase` bug is found during construction, it is
  recorded and fixed in a separately versioned, tested patch — not silently.
- No DGX/Spark dispatch (local CPU only, per the 2026-08-26 decision record).

## 9. Claim boundary

Passing §7 permits only: "the Pac-Man-like v3 §4 harness layer is
constructed, deterministic, no-pause, and causally clean on the local CPU
host." It does not permit any "the model plays" or "the model beats a
baseline" claim — those require the Tier-2/3 model-side preregistrations.
