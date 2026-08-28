# Pac-Man-like (maze_chase) Embodied Harness — v1 Preregistration

**Date:** 2026-08-27
**Status:** FROZEN (design contract; no run is authorized by this document)
**Scope:** the deterministic 60 Hz clock/queue harness that wraps the existing
`maze_chase` environment, the five difficulty families, the six registered
perturbations, and the registered TRAIN/DEV/CAL/TEST seed partitions — the
Tier-1 harness of the owner's 2026-08-26 autonomous-construction contract.
**Upstream contract:**
`brain/docs/preregistrations/2026-08-25-realtime-embodied-qualification-v3.md`
(§2 interface, §4 clock/queue/no-pause, §5 partitions, §7 Q2/Q3).
**Candidate:** Core V2 W120/K32/C3 + `EmbodiedInterfaceV1` heads
(`irene.brain.embodied_interface.v1`, Q0 component prequal green 6/6 rev2,
`brain/runs/embodied-interface-v1-q0-prequal/2026-08-26-…q0_component_prequalification_v1.json`).

This document is a **preregistration of the harness build contract**, not a
run. It freezes the environment's difficulty families, perturbations, seed
partitions, clock discipline, and the acceptance gates that the harness build
(T1-2) must pass before any training corpus (T1-3) or baseline (T1-4) is
generated. It does not open TEST, does not authorize training, and does not
alter any sealed registration or gate. All compute is local CPU
(`py -3.11`, CUDA hidden, single-thread where determinism requires it); the
DGX Spark is embargoed (decision 2026-08-26).

## 0. What exists vs. what this preregistration adds

The base environment is already implemented and test-covered:
`brain/src/irene_brain/environments/maze_chase.py` — an original, rights-clean
16×16 pellet maze carved from the episode seed, one player, deterministic BFS
ghosts under configurable pursuit rules, W/A/S/D control, 60 Hz
(`tick_period_ns = 16_666_667`), checksummed snapshot/restore (version 4), and
the registered variant axes (ghost rules, speed curves, sticky/delayed input).
Its render contract (`_render`) emits a 32×32 `RgbFrame`; privileged labels
(cleared/caught/pellets/coordinates) are returned **only** in `StepOutcome`,
never in `Observation`.

What is **not yet built** (this preregistration freezes it):
1. A no-pause 60 Hz **clock/queue harness** that assigns `cycle_id`,
   `frame_id`, `action_id` and enforces the v3 §4 schedule, deadline, jitter,
   and correspondence rules, with a capacity-one latest-complete-frame queue.
2. The **five difficulty families** (frozen config tuples over
   `MazeChaseEnv`'s constructor axes).
3. The **six registered perturbations** (v3 §7 Q3), mapped to
   `MazeChaseEnv` axes + a deterministic harness observation channel.
4. The **registered TRAIN/DEV/CAL/TEST seed partitions** by layout family.
5. The **expert/scripted policy** (behavior-cloning corpus source) and the
   **reactive baseline** (never weakened) — these are T1-3/T1-4 and are only
   specified here, built under this contract.

No part of this may reach the model input except through the ordinary
`ModelObservation` boundary (v3 §8.1 causal-integrity; constitution C8).

## 1. Difficulty families (frozen)

Five families, each a fixed `MazeChaseEnv` constructor tuple. The maze layout
is a pure function of the episode seed within a family; the family fixes only
the dynamics (ghost count/period/rule, loop count, player speed, lifetime).
All five use `tick_period_ns = 16_666_667` (native 60 Hz) and default
`input_delay_ticks = 0`, `sticky_direction = False` unless a family says
otherwise.

| ID | Name | ghost_count | ghost_period | ghost_rule | ghost_elroy | extra_loops | player_period | max_ticks |
|----|------|:---:|:---:|:----------|:---:|:---:|:---:|:---:|
| F0 | calm    | 1 | 3 | shy     | False | 24 | 1 | 4000 |
| F1 | standard| 3 | 2 | mixed   | False | 16 | 1 | 6000 |
| F2 | pressured| 4 | 2 | direct  | False | 12 | 1 | 6000 |
| F3 | fast    | 4 | 1 | mixed   | True  | 16 | 1 | 8000 |
| F4 | open-slow| 2 | 3 | ambush  | False | 40 | 2 | 8000 |

Rationale [HYPOTHESIS, frozen]: the axis ordering spans ghost pressure
(F0 low → F3 high), pursuit style (shy/mixed/direct/ambush), endgame speed-up
(elroy on F3 only), loop richness (few loops = harder evasion), and a
slow-player open-maze case (F4) that stresses memory/sequencing over reflex.
These are design choices for a coverage ladder, not measured difficulty
ranks; difficulty is only asserted later from measured baseline scores.

## 2. The six registered perturbations (frozen, v3 §7 Q3)

Applied on top of a family's base config, independently, to the same 40
TEST layouts (per Q3). Each is deterministic and verifiable at the harness.
The "clean" condition is the family base with no perturbation.

| ID | Name | Mechanism (frozen rule) |
|----|------|--------------------------|
| P1 | `palette_shift` | Remap the four render RGB constants (`PLAYER`, `PELLET`, `GHOST`, `WALL`) + the two background shades by a fixed, seed-derived byte offset (see §5 P1 rule). Geometry, dynamics, and pellet layout are byte-identical to clean; only pixel values change. |
| P2 | `frame_loss_10pct` | The observation channel drops exactly every 10th rendered frame: the reset frame (g=0) initializes the capacity-one queue and is always delivered; for g >= 1, frame `g` is **not delivered** iff `g mod 10 == 0` (steady-state loss is exactly 10%, deterministic, no randomness). When frame g is dropped the queue retains the prior delivered frame, which the controller dequeues at cycle g. |
| P3 | `obs_delay_1` | The observation delivered at control tick `n` is the frame rendered at tick `n-1` (exactly one-frame latency). Tick 0's frame is the reset frame; the agent's first decision sees the reset frame. |
| P4 | `speed_90pct` | Channel-level world speed at 0.9× the agent's pace: at every cycle `n` with `n mod 10 == 0` (including `n = 0`), the world is **held** for that cycle (no environment step; the controller's action for that cycle is not applied and the queue retains the prior frame; at `n = 0` the reset frame is delivered unchanged). Over every 100 agent cycles aligned to the 10-cycle grid the world advances exactly 90 ticks. |
| P5 | `speed_110pct` | Channel-level world speed at 1.1× the agent's pace: at every cycle `n` with `n mod 10 == 0` (including `n = 0`), the environment is stepped **an extra time** (the agent's action for that cycle is applied to two consecutive world ticks; the queue's latest-complete-frame semantics delivers the newest). Over every 100 agent cycles aligned to the 10-cycle grid the world advances exactly 110 ticks. |
| P6 | `enemy_policy_change` | Swap `ghost_rule` to the family's frozen alternate: F0→`direct`, F1→`direct`, F2→`shy`, F3→`ambush`, F4→`direct`. All other axes unchanged. |

Notes: P2/P3/P4/P5 act on the **harness observation/world channel**, not the
simulator, so they are the cleanest demonstration that the candidate is
timing- and corruption-robust without the simulator being altered. P1/P6 act
on the simulator render/config. Every perturbation is paired to its clean
twin on the **same seed/layout** (v3 §5 shared seed-paired draws).

## 3. Registered seed partitions (frozen)

All seeds are 64-bit integers. We reserve one fresh 2^24-aligned block,
disjoint from every sealed/retired range in the repo (verified against the
PB21* reservations which occupy up to `184550144`):

```
EMBODIED_BLOCK = [201326592, 201347072)   # 12 * 2^24, size 20480 = 5*4*1024
```

Within the block, for each family `f ∈ {0,1,2,3,4}` and partition
`p ∈ {TRAIN, DEV, CAL, TEST}` with partition index
`pi = {TRAIN:0, DEV:1, CAL:2, TEST:3}`:

```
base(f, p) = 201326592 + f * 4096 + pi * 1024
```

- **TEST:** seeds `[base+0, base+40)` → 40 distinct layouts per family,
  200 total (v3 §7 Q2: 40 unseen layouts × 5 families).
- **CAL:** seeds `[base+100, base+200)` → 100 per family (500 total).
- **DEV:** seeds `[base+400, base+500)` → 100 per family (500 total).
- **TRAIN:** seeds `[base+500, base+1024)` → 524 per family (2620 total),
  for the behavior-cloning corpus and integrated training (T1-3/T2-1).

Layouts are disjoint **by construction**: TEST seeds are a fresh range never
used for TRAIN/DEV/CAL; the same 40 TEST seeds are reused for Q3's five
stochastic replicates per layout and their clean pairing. No seed is ever
resampled across partitions. TEST is opened exactly once, after CAL and all
architectural/weight/interface/adapter/reset/quantization/runtime parameters
are frozen (v3 §5).

## 4. Clock, queue, and no-pause contract (frozen, verbatim v3 §4)

The harness assigns `cycle_id` (per native cycle), `frame_id` (per rendered
observation), and `action_id` (per output), each with a
`time.perf_counter_ns` host receipt. The scheduled start of cycle `n` is
exactly:

```
s_n = t0 + floor((n * 1_000_000_000 + 30) / 60)   nanoseconds
```

deadline `d_n = s_(n+1)`, integer-nanosecond period `P_n = d_n - s_n`. The
observation queue has **capacity one** with latest-complete-frame semantics
(overwrite/duplicate/stale/empty-dequeue events are logged). At `s_n` the
controller dequeues the newest frame with `t_enqueue <= s_n`. The conservative
authoritative latency is `L_plus = (t_accept - t_emit) + eps_accept + eps_emit`
(same clock domain → `eps = 0`). A miss is `t_accept > d_n`; a late action is
retained and the preceding action is held. Per-cycle jitter is
`J_plus_n = |(t_accept_n - t_accept_(n-1)) - (s_n - s_(n-1))|`.

**Frozen Pac-Man-like timing gates (v3 §4.3):** native 60 Hz; median `L_plus`
≤ 8.00 ms; p95 ≤ 13.00 ms; p99 ≤ 16.00 ms; deadline-miss rate ≤ 0.1%; p99
`J_plus` ≤ 2.00 ms; **zero consecutive misses**. The game clock **never
pauses, slows, frame-advances, or waits for inference.** (Q1 live endurance
and the scored Q2–Q3 timing battery are Tier-3; this preregistration only
freezes the harness that produces those measurements.)

## 5. Determinism and causal-integrity invariants (frozen)

- **Determinism protocol (mandatory):** `torch.use_deterministic_algorithms(True)`;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `cudnn.deterministic=True`,
  `benchmark=False`; TF32 off both paths; seed banks derived from
  `zlib.crc32(name)` — **never** the process-salted builtin `hash()`.
- **No privileged state (C8):** the model sees only the `ModelObservation`
  boundary — `frame_id`, `elapsed_ns`, `observation_age_ns`, the 32×32
  `RgbFrame`, `previous_control`, `dropped_frames`, and (optional) audio/text.
  It never receives coordinates, pellet counts, ghost positions, reward,
  hazard events, cleared/caught flags, the maze map, a task label, or an
  environment token. The harness must expose a leakage audit proving this
  (see §6 gates).
- **Reproducibility:** a full episode at a fixed (family, seed, perturbation)
  triple must byte-reproduce: every rendered frame, every `L_plus`/`J_plus`
  value, and the final `state_hash`. Snapshot version 4 already checksums the
  sim state; the harness must additionally hash the rendered-frame stream and
  the timing record.

## 6. Harness-build acceptance gates (frozen; T1-2 must pass all)

These are the T1-2 (deterministic sim + harness) acceptance gates. T1-2 may
only proceed to T1-3/T1-4 after every gate passes; on a preregistered failure,
close per the predeclared condition and record the negative.

- **H1 (determinism):** two independent runs of one fixed
  (family, seed, perturbation) episode produce byte-identical rendered-frame
  streams and timing records (frame SHA-256 stream match, `L_plus`/`J_plus`
  arrays exact). ≥3 seeds × ≥2 families × {clean, P2, P3} must match.
- **H2 (no-pause schedule):** the harness's `s_n` sequence exactly equals the
  §4 formula for the full episode length; assert no cycle start is delayed
  past its scheduled value by inference (i.e., inference never blocks the
  clock in the deterministic offline replay).
- **H3 (5 families valid):** each family, on 20 TEST seeds, produces a valid
  start (pellets > 0, no immediate ghost contact at spawn, player/ghosts in
  distinct corridor cells, maze connected — every pellet reachable by BFS).
  Record per-family mean start pellet count and farthest-ghost distance.
- **H4 (6 perturbations live + distinct):** each P1–P6, on 10 clean TEST
  seeds, (a) runs to completion without crash, and (b) produces an
  observation stream that differs from its clean twin (P1: pixel-diff > 0 with
  identical geometry hash; P2/P3: measurable channel change — dropped-frame
  count = exactly 10% for P2, +1-frame age for P3; P4/P5/P6: dynamics differ,
  e.g., different ghost step counts or pellet-clear time). P6 must change the
  ghost rule (verified by the alternate rule's behavior signature).
- **H5 (partition disjointness):** the TEST/CAL/DEV/TRAIN seed sets for every
  family have empty pairwise intersection, and no seed collides with any
  sealed/retired range in the repo (assert against the known reservation list).
- **H6 (leakage audit — C8):** an automated check proves no privileged field
  (coordinates, pellets, ghosts, reward, hazard, cleared/caught, map, task,
  env-token) is reachable from the `ModelObservation` the harness hands the
  model. Concretely: perturb a privileged internal (e.g., move a ghost one
  cell off the rendered path, or zero the pellet count) and assert the model
  observation is unchanged unless the rendered pixel actually changes.
- **H7 (legal action surface):** the adapter (`translate_record` from
  `EmbodiedInterfaceV1`) only ever actuates keys in the frozen
  `ACTUATED_KEY_SET` ∩ {W,A,S,D} for maze_chase; look/cursor/buttons/hotbar
  fields are ignored (the env physically lacks them). Illegal key sets raise.

## 7. What this preregistration does NOT authorize

- No training, no checkpoint publication, no CPU-QUAL materialization.
- No TEST open (TEST seeds are reserved but sealed until CAL + full identity
  freeze per v3 §5).
- No change to `MazeChaseEnv`'s snapshot version or render contract (that
  would be a new environment version requiring its own preregistration).
- The expert policy (T1-3) and reactive baseline (T1-4) are **specified** in
  §8 but **built and validated** under this contract as separate steps; they
  must not weaken each other (the reactive baseline is a real, non-privileged,
  pixel-reactive policy, never a weakened strawman).

## 8. Baselines specified (built in T1-3/T1-4)

- **Expert/scripted policy (BC corpus source):** a pixel-only planner that
  derives the maze geometry from rendered pixels (as the existing
  `diagnostic.scripted_maze_chase_planner.v1` does) and plans a
  ghost-aware pellet path. Its (observation, action) traces on TRAIN seeds
  form the behavior-cloning corpus for T2-1.
- **Reactive baseline (never weakened):** a fixed-depth, pixel-only reactive
  policy: a small CNN/MLP on the last 1–4 rendered frames that greedily
  maximizes immediate pellet gain while minimizing measured ghost proximity
  (BFS-free, pixel-derived). It is a genuine baseline, trained on the same
  TRAIN partition under the same budget class as the candidate, and is the
  Q2/Q3 comparison arm. It must not be weakened to make the candidate look
  better; its own competence is reported.

## 9. Next steps (queue)

1. **T1-2** — build the no-pause 60 Hz harness + §1 families + §2
   perturbations + §3 seed partitions + §6 H1–H7 gates; run H1–H7; record.
2. **T1-3** — expert/scripted policy → BC corpus on TRAIN.
3. **T1-4** — reactive baseline (non-weakened) on TRAIN.
4. **T2-1** — integrate Core V2 + `EmbodiedInterfaceV1` heads under the
   deployed decision loss; multi-seed; then **T2-2** CPU-QUAL prereg.

Every step above commits and pushes after its probe, per the owner contract.
