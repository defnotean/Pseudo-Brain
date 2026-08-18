# Roadmap: from the failed RCQ-v2 to a model that plays Pac-Man

Status: planning document, 2026-08-17. Nothing in this file launches training,
changes source, opens sealed data, or alters any registration. It maps the
verified current state to a concrete implementation sequence whose end point is
an original 60 Hz Pac-Man-like arcade proof ([PLAN.md](../PLAN.md) Phase 4)
and, behind it, the general agent phases 5–7. Every gate below inherits the
stopping rules already frozen in
[RCQ_V2_PROTOCOL.md](RCQ_V2_PROTOCOL.md) and PLAN.md sections 31 and 37.

## 0. Verified starting position

What exists today, with evidence:

- **Phase 0A core: done.** Deterministic moving-shapes world, canonical
  observation/control types, snapshot/restore/branch, leakage audits, timing
  contracts, 299-test CPU-only suite green (2026-08-17).
- **Model + trainer: implemented.** Shared-weight BrainCell thoughtlets
  (K=32, R=3, C=3, d=384), direct 307-query HID readout, anytime exits,
  staged trainer, schema-2/3 checkpoints, exact resume, frozen-stage
  invariance machinery — all smoke- and canary-proven on the DGX Spark.
- **Qualification machinery: done and battle-tested.** Target-blind
  registration, immutable releases, pins, frozen gates, once-only TEST
  chronology. RCQ-v2 exercised all of it end to end.
- **RCQ-v2 reference: terminally failed** at the step-1,536 entry gate
  (commit `2124e09`). Close miss: every action-volume check passed
  (1,422/1,536 exact, 0.976 recall); it failed on 9 opposite-direction
  conflicts (limit 7) and 875 continuous outputs outside the deadzone
  (limit 0). Diagnosis: commit `4ca88ff`,
  [runs/2026-08-17-rcq-v2-entry-gate-failure-diagnosis.md](runs/2026-08-17-rcq-v2-entry-gate-failure-diagnosis.md).
- **Phase 0B physical harness: not started** (deliberately, while the owner
  games on the workstation).
- **No architecture claim exists.** The matched-baseline campaign
  (BASELINE_PROTOCOL) has never run its decisive comparison.

## 1. What the failure bought us

The failed run is the most informative artifact in the project so far:

1. **The policy path works.** 0.926 movement exact vs 0.696 copy-previous on
   the open slice means genuine state-conditioned control, still improving at
   the stop. The architecture's action readout is not the bottleneck.
2. **The two failures are recipe-shaped, not capacity-shaped.** The
   continuous head has identically-zero targets and no loss term that
   penalizes the deadzone boundary; opposite-key co-activation (W/S) had no
   explicit penalty either. Both were visible in logger rows from step 512
   onward.
3. **The gate machinery did its job.** A model that "felt" good (0.93 exact)
   was correctly rejected on hard-zero quiescence. The frozen protocol
   prevented us from shipping a controller that twitches the mouse and
   occasionally presses W and S together — exactly the behaviors that would
   be unacceptable in a live Pac-Man loop.

## 2. RCQ-v3: the redesigned qualification (next scientific step)

A new, separately named qualification. The v2 registration, ranges, and
candidate are dead and stay dead. All choices below are **preregistration-time
decisions** — they must be frozen into the v3 config and registration before
any v3 training update, or they are worthless.

### 2.1 Design decisions to make and freeze

| # | Decision | Options (pick one, freeze it) |
|---|---|---|
| D1 | Continuous-head quiescence | (a) deadzone-aware loss: hinge penalty on `\|out\|` beyond 0.04 with margin; (b) post-hoc clamp at readout: squash outputs through `0.05 * tanh(x / 0.05)` so the deadzone is structurally unreachable; (c) both. Option (b) makes the zero-violation check structurally satisfiable; (a) alone relies on optimization. |
| D2 | Opposite-key conflicts | (a) mutual-exclusion penalty on W/S and A/D logit pairs; (b) represent movement as 9-way categorical (none + 4 + diagonals…) — changes the registered geometry; (c) keep multi-label but add conflict penalty only. Option (c) is the smallest recipe change consistent with the existing geometry. |
| D3 | Budget | Same 2,048 updates (the miss was small) or a re-justified budget. If changed, the change and its justification are preregistered. |
| D4 | Seed and slices | New seed or reuse 1702 — either is defensible; freeze the choice. Final TEST ranges: fresh unused ranges from the reserved 2^21 namespace family; v2's sealed/retired ranges stay untouched forever. |
| D5 | Thresholds | Keep v2's frozen thresholds unchanged unless D2(b) changes geometry. Lowering a threshold because v2 failed is forbidden. |

**Implementation status (2026-08-17):** D1(a), D1(b), and D2(c) are
implemented and config-gated with defaults off
([runs/2026-08-17-rcq-v3-recipe-options.md](runs/2026-08-17-rcq-v3-recipe-options.md)).
The squash bound is 0.046875 (exactly representable, strictly inside the 0.05
deadzone in float32 and bfloat16). D2(b) was rejected as a geometry change.
**All five decisions are now frozen** under the owner's standing delegation:
D1=(c) both, D2=(c) pair penalty 0.25, D3=same 2,048-update budget,
D4=seed 1702 reused with a fresh sealed TEST family at `[4194304, 4195840)`,
D5=thresholds and gate IDs unchanged. The frozen v3 reference config is
`configs/training/dgx-rcq-v3-reference.toml`, and the full qualification
machinery (evaluator lineage, dispatcher `rcq_v3_*` family, operator
wrappers) is implemented and locally verified. Record:
[runs/2026-08-17-rcq-v3-reference-v1-preregistration.md](runs/2026-08-17-rcq-v3-reference-v1-preregistration.md).
Still pending before any v3 training: the owner-run registration ceremony and
DGX sequence below.

### 2.2 RCQ-v3 execution checklist (mirrors the proven v2 sequence)

1. Implement D1/D2 in `irene_brain` behind the config, with unit tests:
   deadzone-squash or hinge loss exact-zero property tests; conflict-penalty
   gradient tests; regression: action geometry unchanged. **(Done.)**
2. Freeze source, regenerate the baseline-architecture manifest, run the
   play-safe suite, commit and push. **(Done; live digest `eb46988b…`.)**
3. Build the target-blind v3 registration at its own path
   (`registrations/rcq-v3-reference-v1.json`, qualification id
   `rcq_v3_reference_v1`), record its SHA-256 off-repo. The builder and
   wrapper exist; the create-once ceremony is owner-run.
4. DGX: preflight → immutable release sync → pretraining pin → smoke →
   staging canary (must include the invariance receipt) → pinned reference
   train → entry gate at 1,536 → value stage → completion gate at 2,048.
5. Only on pass: preclaim, independent review, final authorization, one-shot
   TEST. On any gate failure: same terminal discipline as v2; diagnose and
   decide whether the recipe family is still worth a v4.

**Exit criterion: one seed qualified under a frozen recipe.** This proves the
recipe can produce a reference policy. It does not yet prove the architecture
beats anything.

## 3. The architecture question (matched-baseline campaign)

RCQ-v3 passing qualifies a candidate; it does not validate the microthought
thesis. The next scientific layer is PLAN.md Phase 2 / BASELINE_PROTOCOL:

- Implement the remaining matched baselines in `model/baselines.py` (several
  already exist): equal-state wider monolith, equal-FLOP deeper serial,
  no-persistence and no-communication thought ablations, reactive CNN.
- Run the equal-parameter / equal-FLOP / equal-latency comparison on the
  preregistered multi-seed evaluation namespace (`[2097152, …)` reservation).
- Decision rule (already frozen in PLAN.md §31 Phase 2): ≥10% relative
  multi-horizon prediction gain and ≥10% IQM return gain at matched compute,
  or the thought-field thesis is redesigned once, then retired.

This campaign answers "is the architecture real?" — the question that decides
whether Pac-Man is built on thoughtlets or on a conventional recurrent world
model. Either answer keeps the Pac-Man roadmap alive; it changes what we
scale.

## 4. Closing the loop: from open-loop teacher forcing to live play

Everything so far is open-loop. Pac-Man is closed-loop. The gap is Phase 0B
plus a real-time evaluation stack:

1. **Phase 0B physical harness** (workstation, RTX 5070, explicit non-gaming
   window): DXGI/Windows Graphics Capture, GPU-texture path, virtual HID
   output, hardware kill switch, QPC instrumentation, 30–120 min soak. Gate:
   no-model capture-to-input p99 < 4 ms (PLAN.md §31 Phase 0).
2. **Closed-loop moving-shapes play:** the qualified v3 (or later) checkpoint
   drives the in-repo world in real time through `runtime/continuous.py`.
   The deterministic evaluator half is implemented
   (`evaluation/closed_loop_play.py`; see §6 item 6); what remains is running
   a qualified checkpoint through it and the physical Phase 0B latency path.
   Measure observe-to-submit latency, deadline misses, and closed-loop task
   success vs the open-loop gate numbers. This is the first honest "the model
   plays" claim, still in a toy world.
3. **Robustness battery** (already specced in PLAN.md §27, §30): dropped
   frames, 0–2 frame input delay, sticky actions, thermal soak.

## 5. The environment ladder to Pac-Man

PLAN.md §20 defines the ladder; the work items in this repo are:

1. **Richer in-repo worlds** (moving-shapes successors): pursuit/evasion,
   junction choice, occlusion, keys/doors — each with registered splits. These
   generate the lifetime + branch-DAG data the thought field is designed for.
   Pursuit/evasion is implemented (`environments/pursuit.py`, playable through
   the closed-loop evaluator via `environment_factory`;
   [runs/2026-08-17-pursuit-world.md](runs/2026-08-17-pursuit-world.md));
   junction choice is implemented (`environments/junction.py`, a seeded
   perfect-maze world with BFS shortest-path chasers;
   [runs/2026-08-17-junction-world.md](runs/2026-08-17-junction-world.md));
   occlusion is implemented (`environments/occlusion.py`, moving-shapes
   mechanics under a fog-of-war view radius;
   [runs/2026-08-17-occlusion-world.md](runs/2026-08-17-occlusion-world.md));
   keys/doors is implemented (`environments/keys_doors.py`, key → door →
   target planning on the maze;
   [runs/2026-08-17-keys-doors-world.md](runs/2026-08-17-keys-doors-world.md)).
   All four named successor worlds are done; what remains is branch-DAG data
   generation at scale and registered splits. The five worlds are covered by
   one canonical cross-world diagnostic matrix
   (`evaluation/cross_world_matrix.py`;
   [runs/2026-08-17-cross-world-diagnostic-matrix.md](runs/2026-08-17-cross-world-diagnostic-matrix.md))
   that already discriminates pursuit pressure, maze planning, occlusion
   memory, and key/door sequencing. The keys_doors scripted frontier is now
   mapped by `diagnostic.scripted_keys_doors_solver.v1`, a pixel-only
   key → door → target solver that derives the unrendered key-possession
   state from pixel disappearance across frames: it collects 36 targets
   over the three canonical seeds where every reactive policy scores a flat
   zero, and the matrix is now 13 worlds × 6 policies = 78 cells
   ([runs/2026-08-17-keys-doors-solver-policy.md](runs/2026-08-17-keys-doors-solver-policy.md)).
   The junction scripted frontier is likewise mapped by
   `diagnostic.scripted_junction_solver.v1`, a pixel-only chaser-aware
   solver that simulates the exact BFS pursuit before committing to the
   unique corridor path: it collects 22 targets against 12 forced catches
   (+10 reward) where the best reactive policy scored -16, and the matrix
   is now 13 worlds × 7 policies = 91 cells
   ([runs/2026-08-17-junction-solver-policy.md](runs/2026-08-17-junction-solver-policy.md)).
   The occlusion scripted frontier completes the ladder's skill mapping:
   `diagnostic.scripted_occlusion_memory.v1`, a pixel-only episodic-memory
   policy, derives target memory (static until collected) and exact hazard
   trajectory hypotheses (velocity beliefs pruned by contradiction) from
   pixels across time: it collects 125 targets with zero collisions where
   the reactive frontier is one lucky target — a 125× quantification of
   what persistent state is worth on this world — and the matrix is now
   13 worlds × 8 policies = 104 cells
   ([runs/2026-08-17-occlusion-memory-policy.md](runs/2026-08-17-occlusion-memory-policy.md)).
   The two open-field worlds complete the frontier sweep:
   `diagnostic.scripted_open_field_collector.v1` handles the shared
   moving_shapes/pursuit pixel signature by identifying the mover rule
   from observed motion (bounce vs pursuit), remembering the target while
   a camping mover hides it, and luring campers off the goal: it collects
   with zero contact on both worlds (+236 vs the chaser's +186 on
   moving_shapes, +214 vs +134 on pursuit), and the matrix is now
   13 worlds × 9 policies = 117 cells
   ([runs/2026-08-17-open-field-collector-policy.md](runs/2026-08-17-open-field-collector-policy.md)).
   The distillation loop is now wired for all four solver worlds:
   `SolverSequenceDataset` (`data/solver_dataset.py`,
   `irene.{world}.solver_teacher.v1`) serves lazy mechanics-matched solver
   demonstrations, and `SolverBatchSource` (`training/batches.py`) feeds
   them through the generic trainer protocol — verified end to end by a
   bounded CPU optimizer step on solver-labeled batches. `moving_shapes`
   is deliberately excluded (sealed RCQ family) and `maze_chase` keeps its
   own planner-teacher dataset
   ([runs/2026-08-17-solver-demonstration-datasets.md](runs/2026-08-17-solver-demonstration-datasets.md)).
   A bounded CPU distillation probe (`scripts/distill_solver_smoke.py`,
   256 steps, any solver world) pins the occlusion reference: the memory
   teacher is 38× more imitable at smoke scale than the maze planner
   (movement exact-match 0.3065 vs 0.0081), but half-learned movement
   plays worse than stillness (0/0 → −17 reward/17 collisions) — hazard
   avoidance, the memory-dependent half, is what DGX scale must close
   ([runs/2026-08-18-solver-smoke-distillation.md](runs/2026-08-18-solver-smoke-distillation.md)).
2. **External open worlds** (license-cleared): XLand-MiniGrid, Craftax,
   Procgen — adapters behind `environments/protocol.py`, lifetime recording,
   held-out generator families.
3. **The original Pac-Man-like environment** (Phase 4 target): new module,
   e.g. `environments/maze_chase.py` — original code and assets (no Namco
   IP), deterministic, branchable, 60 Hz, with the registered variant axes:
   maze layouts, ghost AI rules, speed curves, visuals, control mappings,
   sticky/delayed input. Rights-clean by construction. **Implemented** as
   `environments/maze_chase.py` (loop-carved seeded mazes, pellet clearing
   with a terminated win condition, BFS ghosts;
   [runs/2026-08-17-maze-chase-world.md](runs/2026-08-17-maze-chase-world.md)).
   All registered variant axes are now implemented: ghost AI rules
   (`direct`/`ambush`/`shy`/`mixed`, snapshot version 2;
   [runs/2026-08-17-maze-chase-ghost-rules.md](runs/2026-08-17-maze-chase-ghost-rules.md)),
   speed curves (`player_period` input-sampling gating and the
   `ghost_elroy` endgame speed-up, snapshot version 3;
   [runs/2026-08-17-maze-chase-speed-curves.md](runs/2026-08-17-maze-chase-speed-curves.md)),
   and sticky/delayed input (`input_delay_ticks` FIFO plus the
   `sticky_direction` latch, snapshot version 4;
   [runs/2026-08-17-maze-chase-sticky-delayed-input.md](runs/2026-08-17-maze-chase-sticky-delayed-input.md)).
   Visuals and control mappings remain shared ladder surfaces. The scripted pellet teacher and matrix inclusion
   are done: `diagnostic.scripted_pellet_teacher.v1` and
   `world.maze_chase.v1` joined the cross-world matrix, where the teacher's
   -6,274 canonical row shows reflexive greed losing badly to ghost
   pressure. The scripted frontier is now mapped by
   `diagnostic.scripted_maze_chase_planner.v1`, a pixel-only lookahead
   planner that simulates the published ghost mechanics before committing to
   a pellet path: it clears all three canonical mazes (+426 reward, 3
   catches, 3/3 clears), and the matrix is now 6 worlds × 5 policies
   ([runs/2026-08-17-maze-chase-planner-policy.md](runs/2026-08-17-maze-chase-planner-policy.md)).
   Every registered variant axis is also promoted to its own matrix world
   slot (13 worlds × 5 policies = 65 cells), which shows the planner
   clearing mixed/shy/ambush rules, degrading gracefully under elroy and
   slow-player speed curves, and going negative only under two-tick input
   delay — the actuation-latency signature
   ([runs/2026-08-17-maze-chase-variant-matrix-slots.md](runs/2026-08-17-maze-chase-variant-matrix-slots.md)).
   The planner then gained actuation-awareness knobs (delay-FIFO tracking of
   its own presses, player-period gating, elroy period derivation from
   visible pixels): compensation recovers the delayed-input slot from -147
   to +386 and nearly halves elroy catches, proving the failure was pure
   latency mismatch — the quantity a play-capable model must plan against
   ([runs/2026-08-17-planner-actuation-compensation.md](runs/2026-08-17-planner-actuation-compensation.md)).
4. **The arcade proof gate** (PLAN.md §31 Phase 4): kernel p99 ≤ 8 ms,
   end-to-end p99 ≤ 16.67 ms at frame skip 1, ≥10% score over the matched
   real-time recurrent/world-model baseline across ≥100 seeds, positive
   frozen-weight learning curve in held-out variants. Only then does "plays
   Pac-Man" mean something; actual Namco Pac-Man remains a rights-reviewed
   external validation, never a training foundation.

## 6. Consolidated implementation backlog (ordered)

Near-term slices, each independently committable and test-covered:

1. ~~**RCQ-v3 design doc + frozen decisions** (D1–D5 above, owner sign-off).~~
   Done 2026-08-17 under the owner's standing delegation; veto window closes
   at the registration ceremony.
2. ~~**D1/D2 implementation + tests** in model/training code (§2.2 item 1).~~
   Done.
3. **v3 registration ceremony and DGX sequence** (§2.2 items 3–5). Machinery
   ready; the ceremony itself is owner-run from a console-attached terminal.
4. **Remaining matched baselines + the multi-seed comparison harness**
   (`evaluation/multiseed_comparison.py` exists; extend to the full baseline
   suite of PLAN.md §28). In progress: the statistical harness is complete and
   the suite now covers ten variants — reference, isolated-slot, reset-slot
   (persistence removed), dense-routing (unrestricted communication),
   reactive (no cross-step state), serial-depth (untied twelve-block serial
   stack at matched block FLOPs), both monolithic GRU controls, the
   matched-cost independent ensemble (four untied members at width 352,
   0.72% under the reference budget), and the recurrent carry-token
   Transformer control (width 568, 0.22% under budget) — the last four
   inside the manifest's verified parameter-matched regime
   ([runs/2026-08-17-baseline-suite-persistence-and-density-ablations.md](runs/2026-08-17-baseline-suite-persistence-and-density-ablations.md),
   [runs/2026-08-17-matched-ensemble-baseline.md](runs/2026-08-17-matched-ensemble-baseline.md),
   [runs/2026-08-17-recurrent-transformer-baseline.md](runs/2026-08-17-recurrent-transformer-baseline.md)).
   Still missing from §28: fixed multi-horizon no-persistence heads,
   recurrent world-model actor, and task specialist. Scoping found all
   three blocked on shared-objective or campaign-shape changes, so they
   are preregistered design proposals (B1–B3) awaiting explicit owner
   sign-off rather than drop-in variants
   ([runs/2026-08-17-remaining-baseline-controls-preregistration.md](runs/2026-08-17-remaining-baseline-controls-preregistration.md)). The §28 diagnostic
   policies (no-op, random, scripted chaser, privileged oracle) are
   implemented for the closed-loop evaluator in
   `evaluation/diagnostic_policies.py`
   ([runs/2026-08-17-diagnostic-policies.md](runs/2026-08-17-diagnostic-policies.md)).
5. **Phase 0B capture/control harness** (runtime/capture, runtime/controls,
   watchdog, kill switch; behind explicit arming).
6. **Closed-loop evaluator**. Implemented for the in-repo world:
   `evaluation/closed_loop_play.py` drives moving-shapes deterministically
   through `runtime/continuous.py` on a manual clock with declared simulated
   inference latency, and emits canonical SHA-256 evidence records (task
   outcomes, deadline misses, stale-frame rejections, RCQ-v2 action-path
   failure modes) ([runs/2026-08-17-closed-loop-play-evaluator.md](runs/2026-08-17-closed-loop-play-evaluator.md)).
   Simulated play-latency evidence records are implemented:
   `evaluation/play_latency.py` converts per-decision manual-clock
   milestones into the fail-closed `LatencyEvidence` contract with an
   analytic, pre-pinnable decision schedule
   ([runs/2026-08-17-play-latency-evidence.md](runs/2026-08-17-play-latency-evidence.md)).
   Still missing: physical live-play records (blocked on the Phase 0B
   harness) and Phase 0B harness integration.
7. **Next in-repo worlds** (pursuit/junction/occlusion) + branch-DAG data
   generation at scale on the Spark.
8. **External world adapters** (XLand-MiniGrid first; smallest integration
   surface).
9. **`maze_chase` environment** + variant generator + rights ledger entry.
   Done: the environment, every registered variant axis (ghost rules, speed
   curves, sticky/delayed input), and the rights ledger
   ([RIGHTS_LEDGER.md](RIGHTS_LEDGER.md)) with all six in-repo worlds
   accepted as original works and every external source marked unreviewed.
   The planner-teacher dataset
   (`data/maze_chase_dataset.py`, `irene.maze_chase.planner_teacher.v1`)
   closes the loop to training: lazy, mechanics-matched demonstrations from
   the frontier planner through the same sequence interface the trainer
   already consumes
   ([runs/2026-08-17-maze-chase-planner-dataset.md](runs/2026-08-17-maze-chase-planner-dataset.md)),
   and `MazeChaseBatchSource` (`training/batches.py`) feeds those
   demonstrations through the generic trainer protocol — verified end to
   end by a bounded CPU optimizer step on planner-labeled batches
   ([runs/2026-08-17-maze-chase-training-batches.md](runs/2026-08-17-maze-chase-training-batches.md)).
   A bounded CPU distillation probe
   (`scripts/distill_maze_chase_smoke.py`, 256 steps) confirms the smoke
   model genuinely fits the planner's policy (validation action loss
   1.0156 → 0.4600 on unseen sequences) while play-level transfer stays at
   the smoke-scale floor — pinning the instruments and zero-points for the
   DGX distillation campaign
   ([runs/2026-08-17-maze-chase-smoke-distillation.md](runs/2026-08-17-maze-chase-smoke-distillation.md)).
10. **Arcade-scale training and the Phase 4 gate.**

## 7. Resources and constraints

- Workstation: CPU-only verification (current rule); RTX 5070 live loop only
  in an explicitly armed non-gaming window (Phase 0B).
- DGX Spark: all accelerator training through the pinned wrappers; the
  console-attached-terminal requirement for production actions is documented
  in DGX_SPARK_TRAINING.md. `irene-qwen38-heretic` may be restored now that
  the GPU is idle; training windows must stop it again.
- Timeline honesty: PLAN.md estimates Phase 2 at 8–12 weeks and Phase 4 at
  2–4 months *after* earlier gates, for reference. The v2→v3 loop is days of
  implementation plus ~2.5 hours of DGX time per attempt; the matched-baseline
  campaign is the first multi-week item.
- Every stage keeps the standing rules: no resume of failed frozen gates, no
  TEST construction before a durable claim, CPU-only local verification,
  continuous commit/push of operator-facing docs.

## 8. Stop conditions that still apply

PLAN.md §37 is unchanged: two failed thought-field designs at three seeds
each, or disappearance of arcade gains under frame skip/sticky
actions/physical capture, ends the thesis as an architecture claim. A failed
RCQ-v3 does not automatically end anything — but two failed recipe iterations
on the same failure modes would force a design-level rethink before a v4, not
another blind retry.
