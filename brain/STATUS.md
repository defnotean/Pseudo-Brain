# Implementation status

Updated: 2026-08-18

## Current campaign: play-gated maze-chase distill v1

Status: **play-gated maze-chase distill v1** is the live campaign
(`play_gated_maze_chase_distill_v1`). Play moved on the 32-step probe v2
and held on the 128-step probe at reward_sum **-150**, collisions **16**
(sticky D, ~10 implied pellets). Window-32 and episode-windows **failed**
idle no-op (9 pellets, mask 0 × 480). Named play decode
`exclusive_argmax_wasd_v1` **unstuck idle** on the same episode-windows
teacher and 32-step budget (`dgx-play-maze-chase-distill-exclusive-argmax-v1`,
`play-gate.json` SHA-256 `865927de…`, release `r20260818t172702z-e492020c6fca`):
histogram **S×478 + A×2**, **20 pellets**, 391 collisions, reward **−3890**,
`sticky_or_idle: false`. Val WASD predicted-positive stayed **0.0**
(metrics SHA identical to episode-windows). Campaign still failed (pellets
< 32). Exclusive-direction softmax `exclusive_wasd_softmax_v1` on the same
teacher and decode (`dgx-play-maze-chase-distill-exclusive-ce-v1`,
`play-gate.json` SHA-256 `17388344…`, release `r20260818t175216z-c804bcd101c1`)
**failed sticky S**: histogram **S×480**, 20 pellets, 391 collisions,
reward −3890. Val exclusive-argmax match **0.167** (= teacher S); teacher
logit gap **−0.445**. Action-only exclusive CE
(`dgx-play-maze-chase-distill-exclusive-ce-action-only-v1`) **failed idle
no-op**: mask 0 × 480, 9 pellets, 17 collisions, reward −161; val exclusive-argmax
match **0.0**; inactive logit max ≈ −4.81. Value-only exclusive CE
(`dgx-play-maze-chase-distill-exclusive-ce-value-only-v1`,
`play-gate.json` SHA-256 `7dc8959c…`, release `r20260818t181900z-85606667a9fa`)
**failed idle no-op**: mask 0 × 480, 9 pellets, 17 collisions, reward −161;
val exclusive-argmax match **0.0**; inactive logit max ≈ −4.82. Tiled 1:1
planner windows (`dgx-play-maze-chase-distill-tiled-windows-v1`,
`play-gate.json` SHA-256 `9fac7fd9…`, release `r20260818t184756z-49bac601d0ca`,
canonical config SHA-256 `2c126296c820830d037c0fc14053f1cbdbd05ec40d218ea2e84975cb02002b83`)
**failed sticky A**: histogram **A×476 + D×4**, 15 pellets, 18 collisions,
reward **−165**, `sticky_or_idle: false`. Val exclusive-argmax match **0.083**
(down from exclusive-CE 0.167). Full-episode tiled update
(`dgx-play-maze-chase-distill-episode-update-v1`,
`play-gate.json` SHA-256 `ba622c99…`, release
`r20260818t190103z-ddf0904b5d81`, canonical config SHA-256
`b9b7888c194f72d91b6464b6e3e99dc2e52103db35c9a4d441ca69b60ee80c40`)
**failed sticky D**: histogram **D×431 + A×49**, 10 pellets, 16 collisions,
reward **−150**. Val exclusive-argmax match **0.417** equals teacher D, not a
ranking gain. Do not scale accumulation-30. Multi-episode tiled tiles
(`dgx-play-maze-chase-distill-multi-episode-v1`,
`play-gate.json` SHA-256 `c6367334…`, release
`r20260818t192855z-6d85cc69dd69`, canonical config SHA-256
`4231288135f8465008a106f1133a8f1a9321f7ee0a8cf7aa76494a0788c72c54`)
**failed mixed W/A/D**: histogram **W×48 + A×200 + D×232**, 17 pellets,
17 collisions, reward **−153**. Val exclusive-argmax match **0.083**
equals teacher A. Do not scale 90-seq. Next GPU probe is turn-weighted
exclusive CE (`dgx-play-maze-chase-distill-turn-weighted-v1`), now running
on release `r20260818t195814z-872818a4fa68`. Named CPU
farm jobs finished planner seeds 132–147, tiled teacher mix, multi-episode
coverage, off-policy teacher labels, an episode-update thoughtlet dump,
and a 90-window majority audit (79 mixed / 10 pure, mean majority 0.614).
Campaign success is **pellets ≥ 32**
with a non-idle non-D-only histogram. Official 32-step `play-gate.json`
SHA-256
`df9d88f4…`; 128-step `6401a922…`; window-32 `2cdedaf9…`; episode-windows
`84331559…`; exclusive-argmax `865927de…`. Logger train loss is not the
gate. Workstation is orchestration; all compute is Spark. Record:
[docs/runs/2026-08-18-play-gated-maze-chase-distill.md](./docs/runs/2026-08-18-play-gated-maze-chase-distill.md).

Historical: live qualification `rcq_v2_reference_v2` is **terminally failed**. The
2,048-update reference stopped at optimizer step 1,536 when the frozen
`rcq_v2_development_v1` entry gate failed on the open development slice: nine
opposite-direction conflicts (limit seven) and 875 continuous outputs outside
the [-0.05, 0.05] deadzone (limit zero). Every action-volume check passed
(1,422/1,536 movement exact, 402/467 changed exact, 0.976 recall, 64 movement
false positives, zero off-support outputs). The canonical failed report
(`development-gate-step-00001536.json`, SHA-256 `6e1bcdec…`) makes the
step-1,536 checkpoint terminal: no resume, no tuning, no retry under this
registration. Final TEST was never opened and stays sealed. Any next attempt
needs a newly preregistered experiment. Record:
[docs/runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md](./docs/runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md).
Post-mortem failure mechanics (quiescence tail, W/S opposite conflicts):
[docs/runs/2026-08-17-rcq-v2-entry-gate-failure-diagnosis.md](./docs/runs/2026-08-17-rcq-v2-entry-gate-failure-diagnosis.md).
The v3 recipe options (deadzone hinge, structural squash, opposite-key pair
penalty) are implemented and config-gated with defaults off; the unregistered
review candidate is `configs/training/dgx-rcq-v3-reference-candidate.toml`.
Record: [docs/runs/2026-08-17-rcq-v3-recipe-options.md](./docs/runs/2026-08-17-rcq-v3-recipe-options.md).

RCQ-v3 preregistration machinery is complete and the D1–D5 decisions are
frozen under the owner's standing delegation: qualification
`rcq_v3_reference_v1`, frozen config
`configs/training/dgx-rcq-v3-reference.toml` (hinge 0.5 @ margin 0.04, pair
penalty 0.25, deadzone-tanh squash, seed 1702, 2,048 updates), fresh sealed
TEST family `[4194304, 4195840)`, unchanged gates and thresholds. The v3
evaluator lineage, trusted-dispatcher `rcq_v3_*` action family, and all ten
operator wrappers exist and are locally verified; the registration file
itself is deliberately not yet created. The 2026-08-18 constant-LR smoke
erratum left the proceed-criterion unmet, so the create-once ceremony stays
deferred — do not run `New-RcqV3Registration.ps1`. Record:
[docs/runs/2026-08-17-rcq-v3-reference-v1-preregistration.md](./docs/runs/2026-08-17-rcq-v3-reference-v1-preregistration.md).
Roadmap and pending owner decisions:
[docs/ROADMAP_TO_PACMAN.md](./docs/ROADMAP_TO_PACMAN.md).

RCQ-v2 asks whether the existing single model can learn a state-conditioned
W/A/S/D policy and a useful value estimate on synthetic moving-shapes, for one
seed, under a frozen recipe. A later pass would not be closed-loop gameplay or
architecture superiority. Read [CURRENT_WORK.md](../CURRENT_WORK.md) and
[docs/OPERATOR_GUIDE.md](./docs/OPERATOR_GUIDE.md) before [docs/RCQ_V2_PROTOCOL.md](./docs/RCQ_V2_PROTOCOL.md).

Live create-once registration:

- File: `registrations/rcq-v2-reference-v2.json`
- SHA-256: `6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`
- Qualification id: `rcq_v2_reference_v2`
- `sealed_test_examples_opened`: 0
- Record: [docs/runs/2026-08-16-rcq-v2-reference-v2-registration.md](./docs/runs/2026-08-16-rcq-v2-reference-v2-registration.md)

Historical v1 file `registrations/rcq-v2-reference-v1.json` remains
`33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0`. Do not
edit either JSON. If source or the RCQ training config changes after the live
pin, start a newly named qualification instead.

Two unused Spark pins (`cee1cb76…` then `04608d70…`) were discarded after
pin-bound smoke died in isolated tests that only fail on the immutable Linux
release: a mode-`444` evaluator copy, then an OS-dependent absolute checkpoint
path. CUDA backward passed both times. Record:
[docs/runs/2026-08-16-rcq-v2-unused-pretraining-pin-discard.md](./docs/runs/2026-08-16-rcq-v2-unused-pretraining-pin-discard.md).

The first live-v2 Spark pin (`2847e786…`, release
`r20260817t021020z-6f2359089eeb`) is unused: smoke died because a preservation
test required git-only `registrations/rcq-v2-reference-v1.json`. CUDA backward
passed. Record:
[docs/runs/2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md](./docs/runs/2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md).
Keep the live registration SHA. Do not rebuild it.

Live v2 pin `adf79ccc…` (release `r20260817t021531z-7a2967ebec60`) passed
smoke and the staging canary. The 2,048-update reference then failed the
frozen step-1,536 development entry gate (opposite-direction conflicts 9 > 7;
continuous outputs outside the deadzone 875 ≠ 0) and is terminal. Record:
[docs/runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md](./docs/runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md).
Spark disk cleanup of non-3.8 weights:
[docs/runs/2026-08-16-spark-non-3.8-model-cleanup.md](./docs/runs/2026-08-16-spark-non-3.8-model-cleanup.md).

The third historical-v1 pin (`dbcb6afc…`, release `r20260817t013858z-db1f586ef3a0`) passed
smoke and failed the staging canary. Diagnosis:
[docs/runs/2026-08-16-rcq-v2-canary-invariance-diagnosis.md](./docs/runs/2026-08-16-rcq-v2-canary-invariance-diagnosis.md).
Do not start the 2,048-update reference on that pin. The capture fix changed
`source_tree_sha256`; v1 JSON is preserved and live work uses
`registrations/rcq-v2-reference-v2.json`.

The matched-baseline architecture manifest at
`configs/baseline-architecture-manifest.json` was regenerated as an intentional
local freeze on 2026-08-16, twice on 2026-08-17 (RCQ-v3 recipe options, then
v3 smoke-factory squash parity), and again on 2026-08-17 when the baseline
suite gained the reset-slot, dense-routing, reactive, and serial-depth
ablations. Live digest
`3d7ff5bd338503a79ea43a4949d48c073615f78bc6fa553cc8169a14d688fba3` (previous
`5e0f2536…`, `eda3cf38…`, `78ba9cfc…`, `8a41131e…`, `f4e9b355…`, `eb46988b…`, `5decb402…`,
and `30d4c119…`; all are
explicit `new_comparison` identities). The
historical first-matched campaign pin `52bba6a9…` is unchanged and that
campaign stays blocked. Records:
[docs/runs/2026-08-16-baseline-architecture-manifest-freeze.md](./docs/runs/2026-08-16-baseline-architecture-manifest-freeze.md),
[docs/runs/2026-08-17-rcq-v3-recipe-options.md](./docs/runs/2026-08-17-rcq-v3-recipe-options.md),
[docs/runs/2026-08-17-rcq-v3-reference-v1-preregistration.md](./docs/runs/2026-08-17-rcq-v3-reference-v1-preregistration.md),
[docs/runs/2026-08-17-baseline-suite-persistence-and-density-ablations.md](./docs/runs/2026-08-17-baseline-suite-persistence-and-density-ablations.md).
The freeze does not open TEST and does not start DGX training.

The closed-loop moving-shapes play evaluator (roadmap §6 item 6) is
implemented in `src/irene_brain/evaluation/closed_loop_play.py`: a
deterministic, manual-clock harness that drives the in-repo world through
`runtime/continuous.py` with declared simulated inference latency and emits
canonical SHA-256 evidence records (task outcomes, deadline misses,
stale-frame rejections, and the RCQ-v2 action-path failure modes). It is
exploratory evidence tooling, not a qualification component; no checkpoint
is qualified to be measured by it yet. Record:
[docs/runs/2026-08-17-closed-loop-play-evaluator.md](./docs/runs/2026-08-17-closed-loop-play-evaluator.md).
The PLAN.md §28 diagnostic policies (no-op, random movement, pixel-only
scripted chaser, privileged-state oracle) are implemented for that evaluator
in `src/irene_brain/evaluation/diagnostic_policies.py`, giving every future
closed-loop model row a floor, reference, and ceiling. Record:
[docs/runs/2026-08-17-diagnostic-policies.md](./docs/runs/2026-08-17-diagnostic-policies.md).
The first moving-shapes successor world is implemented:
`src/irene_brain/environments/pursuit.py` (pursuit/evasion with deterministic
greedy chasers, canonical snapshots, and swept-path contact), and the
closed-loop evaluator now drives any branchable in-repo world through an
`environment_factory` parameter. Record:
[docs/runs/2026-08-17-pursuit-world.md](./docs/runs/2026-08-17-pursuit-world.md).
The junction-choice rung is also implemented:
`src/irene_brain/environments/junction.py` (a seeded perfect-maze world with
BFS shortest-path chasers and the same snapshot/render rigor). Record:
[docs/runs/2026-08-17-junction-world.md](./docs/runs/2026-08-17-junction-world.md).
The occlusion rung is also implemented:
`src/irene_brain/environments/occlusion.py` (moving-shapes mechanics under a
fog-of-war view radius; the direct world-level test of the persistent-state
thesis). Record:
[docs/runs/2026-08-17-occlusion-world.md](./docs/runs/2026-08-17-occlusion-world.md).
The keys/doors rung completes the named in-repo ladder:
`src/irene_brain/environments/keys_doors.py` (key → door → target planning
on the seeded maze, with key possession deliberately unrendered). Record:
[docs/runs/2026-08-17-keys-doors-world.md](./docs/runs/2026-08-17-keys-doors-world.md).
A canonical cross-world diagnostic matrix
(`src/irene_brain/evaluation/cross_world_matrix.py`) now covers all five
worlds × the three non-privileged diagnostic policies in one SHA-256-pinned
evidence record, and it already discriminates pursuit pressure, maze
planning, occlusion memory, and key/door sequencing. Record:
[docs/runs/2026-08-17-cross-world-diagnostic-matrix.md](./docs/runs/2026-08-17-cross-world-diagnostic-matrix.md).
The Phase 4 target world is implemented and rights-clean by construction:
`src/irene_brain/environments/maze_chase.py` — pellet clearing with a
terminated win condition, BFS ghosts, and loop-carved seeded mazes (a
perfect maze proved unplayable under pursuit: a chasing ghost in a tree is a
hard wall). Record:
[docs/runs/2026-08-17-maze-chase-world.md](./docs/runs/2026-08-17-maze-chase-world.md).
A pixel-only scripted pellet teacher (`diagnostic.scripted_pellet_teacher.v1`)
and `world.maze_chase.v1` joined the cross-world diagnostic matrix; the
teacher's canonical maze_chase row
(-6,274 reward: 366 pellets eaten, 665 catches) shows reflexive greed losing
badly and marks evasion planning as the model's required skill. A pixel-only
lookahead planner (`diagnostic.scripted_maze_chase_planner.v1`) now maps the
scripted frontier: simulating the published ghost mechanics over candidate
pellet paths, it clears all three canonical mazes (+426 reward, 3 catches,
3/3 clears), and the matrix is now 6 worlds × 5 policies (SHA-256
`53949071414fd54a…`). Record:
[docs/runs/2026-08-17-maze-chase-planner-policy.md](./docs/runs/2026-08-17-maze-chase-planner-policy.md).
The maze_chase ghost-AI-rules variant axis is now implemented: `ghost_rule`
selects direct chase, four-cell ambush off the player's control-intent
facing, shy far-chase/near-retreat, or a cyclical mix, under snapshot
version 2 (25 tests; play-safe gate green). Record:
[docs/runs/2026-08-17-maze-chase-ghost-rules.md](./docs/runs/2026-08-17-maze-chase-ghost-rules.md).
The speed-curve axis followed: `player_period` gates player action and
input sampling to every Nth tick, and `ghost_elroy` speeds the ghosts up by
one tick once half the pellets are eaten, under snapshot version 3
(28 tests; play-safe gate green). Record:
[docs/runs/2026-08-17-maze-chase-speed-curves.md](./docs/runs/2026-08-17-maze-chase-speed-curves.md).
The sticky/delayed-input axis completes the registered variant set:
`input_delay_ticks` runs controls through a fixed FIFO and
`sticky_direction` latches the last pressed direction until replaced, with
the effective mask reported as `applied_control`, under snapshot version 4
(33 tests; play-safe gate green). Record:
[docs/runs/2026-08-17-maze-chase-sticky-delayed-input.md](./docs/runs/2026-08-17-maze-chase-sticky-delayed-input.md).
Every registered variant axis is now also a matrix world slot (13 worlds ×
5 policies = 65 cells, SHA-256 `3ca6cc8cd371f3…`): the planner clears the
ambush/shy/mixed rule slots, degrades gracefully under elroy and
slow-player speed curves (+244/+226), and goes negative only under
two-tick input delay — the actuation-latency signature a deadline-aware
model must beat. Record:
[docs/runs/2026-08-17-maze-chase-variant-matrix-slots.md](./docs/runs/2026-08-17-maze-chase-variant-matrix-slots.md).
The planner then gained actuation-awareness knobs (delay-FIFO tracking of
its own presses, player-period gating, elroy period derivation from visible
pixels), byte-identical on the canonical slot (matrix SHA-256 unchanged):
compensation recovers the delayed-input slot from -147 to +386 and cuts
elroy catches from 20 to 11, while player-period compensation is pinned as
outcome-neutral. Record:
[docs/runs/2026-08-17-planner-actuation-compensation.md](./docs/runs/2026-08-17-planner-actuation-compensation.md).
The planner now also labels training data:
`src/irene_brain/data/maze_chase_dataset.py`
(`irene.maze_chase.planner_teacher.v1`) is a lazy, deterministic dataset
whose action targets come from the mechanics-matched frontier planner
through the same sequence interface the trainer already consumes — the
distillation path from scripted frontier to trained model is now in-repo
(11 tests; play-safe gate green, 490 tests). Record:
[docs/runs/2026-08-17-maze-chase-planner-dataset.md](./docs/runs/2026-08-17-maze-chase-planner-dataset.md).
`MazeChaseBatchSource` (`src/irene_brain/training/batches.py`) now feeds
those demonstrations through the generic trainer protocol, and a bounded
CPU smoke proves one real optimizer step on planner-labeled maze_chase
batches (finite loss, parameters updated) — a future DGX distillation run
is a configuration exercise, not new machinery (7 tests; play-safe gate
green, 497 tests). Record:
[docs/runs/2026-08-17-maze-chase-training-batches.md](./docs/runs/2026-08-17-maze-chase-training-batches.md).
A deterministic replay artifact renders the frontier for the eye:
`brain/scripts/render_maze_chase_replay.py` replays the canonical slot on
seed 5 with the pixel-only planner and writes
[docs/runs/artifacts/2026-08-17-maze-chase-planner-clear-seed5.gif](./docs/runs/artifacts/2026-08-17-maze-chase-planner-clear-seed5.gif)
— 142 pellets, zero catches, maze cleared at tick 208. A 16-seed
robustness study (seeds 100–115) hardens the frontier claim: the planner
clears 15/16 mazes with 52 total catches (+1,875 reward; every episode
positive) against the teacher's -39,397 and 4,066 catches. Record:
[docs/runs/2026-08-17-maze-chase-planner-robustness.md](./docs/runs/2026-08-17-maze-chase-planner-robustness.md).
The model-side closed-loop path is now pinned as world-generic: the
untrained smoke model plays the canonical maze_chase slot through
`evaluate_closed_loop_play` with every decision accepted, retiring the
Phase 4 integration risk before a trained checkpoint exists. Record:
[docs/runs/2026-08-17-model-maze-chase-integration-pin.md](./docs/runs/2026-08-17-model-maze-chase-integration-pin.md).
A bounded CPU distillation probe
(`brain/scripts/distill_maze_chase_smoke.py`, 256 optimizer steps, fully
deterministic) shows the smoke-scale thought field genuinely learning from
planner demonstrations — validation teacher-agreement action loss 1.0156 →
0.4600 on unseen sequences — with play-level transfer unmoved at this
scale, pinning both instruments' zero-points for the DGX campaign. Record:
[docs/runs/2026-08-17-maze-chase-smoke-distillation.md](./docs/runs/2026-08-17-maze-chase-smoke-distillation.md).
The keys_doors scripted frontier is now mapped:
`diagnostic.scripted_keys_doors_solver.v1` — a pixel-only key → door →
target solver that derives the unrendered key-possession state from pixel
disappearance across frames — collects 36 targets over the three canonical
seeds (11/12/13 per 600-tick episode) where every prior policy scored a
flat zero. The cross-world matrix is now 13 worlds × 6 policies = 78 cells
(SHA-256 `4f10467fbecf539…`, all pre-existing cells unchanged; play-safe
gate green, 506 tests). Record:
[docs/runs/2026-08-17-keys-doors-solver-policy.md](./docs/runs/2026-08-17-keys-doors-solver-policy.md).
The junction scripted frontier is now mapped:
`diagnostic.scripted_junction_solver.v1` — a pixel-only chaser-aware
solver that simulates the exact BFS pursuit before committing to the
unique corridor path — collects 22 targets against 12 forced catches (+10
reward) over the three canonical seeds, where the best prior policy
(greedy chase) scored -16 with 3 targets and 19 catches. The record also
pins two structural facts: a guarded target is unreachable on a perfect
maze, and the solver never takes a deliberate death. The cross-world
matrix is now 13 worlds × 7 policies = 91 cells (SHA-256
`311f2ebe84b650ea…`, all pre-existing cells unchanged; play-safe gate
green, 512 tests). Record:
[docs/runs/2026-08-17-junction-solver-policy.md](./docs/runs/2026-08-17-junction-solver-policy.md).
The occlusion scripted frontier completes the ladder's skill mapping:
`diagnostic.scripted_occlusion_memory.v1` — a pixel-only episodic-memory
policy that derives target memory (the target is static until collected)
and exact hazard trajectory hypotheses (four velocity beliefs per
sighting, pruned by contradiction when the projected cell renders empty)
from pixels across time — collects 125 targets with zero collisions over
the three canonical seeds, where the reactive frontier is one lucky
target: a 125× quantification of what persistent state is worth on the
world designed to test the thought-state thesis. The cross-world matrix
is now 13 worlds × 8 policies = 104 cells (SHA-256 `26a2894130843183…`,
all pre-existing cells unchanged; play-safe gate green, 518 tests).
Record:
[docs/runs/2026-08-17-occlusion-memory-policy.md](./docs/runs/2026-08-17-occlusion-memory-policy.md).
The two open-field worlds complete the frontier sweep:
`diagnostic.scripted_open_field_collector.v1` identifies the mover rule
from observed motion under the shared moving_shapes/pursuit pixel
signature (full-set stays identify pursuit, diagonal steps identify
bouncing), remembers the target while a camping mover hides it, and lures
campers off the goal — collecting with zero contact on both worlds (+236
vs the greedy chaser's +186 on moving_shapes, +214 vs +134 on pursuit).
The cross-world matrix is now 13 worlds × 9 policies = 117 cells (SHA-256
`e76c4dd3f16673cd…`, all pre-existing cells unchanged; play-safe gate
green, 524 tests). Record:
[docs/runs/2026-08-17-open-field-collector-policy.md](./docs/runs/2026-08-17-open-field-collector-policy.md).
The distillation loop is now wired for all four solver worlds:
`SolverSequenceDataset` (`data/solver_dataset.py`, generator identities
`irene.{world}.solver_teacher.v1`) serves lazy mechanics-matched solver
demonstrations for keys_doors, junction, occlusion, and pursuit, and
`SolverBatchSource` (`training/batches.py`) feeds them through the generic
trainer protocol — verified end to end by a bounded CPU optimizer step on
solver-labeled batches. `moving_shapes` is deliberately excluded (sealed
RCQ family) and `maze_chase` keeps its own planner-teacher dataset; no
registered configuration hash changes (19 tests; play-safe gate green,
543 tests). Record:
[docs/runs/2026-08-17-solver-demonstration-datasets.md](./docs/runs/2026-08-17-solver-demonstration-datasets.md).
A bounded CPU distillation probe (`scripts/distill_solver_smoke.py`, 256
steps, any solver world) pins the occlusion reference outcome,
bit-identical across two runs: train loss 0.7623 → 0.4306, validation
action loss 1.0133 → 0.4316, movement exact-match 0.0000 → 0.3065 (38×
the maze probe's 0.0081 — the memory teacher's directed travel is far
more imitable than the planner's knife-edge lookahead), but closed-loop
play goes 0 reward/0 collisions → −17/17: half-learned movement walks
into hazards the smoke model cannot yet remember. Hazard avoidance — the
persistent-belief half of the teacher's skill — is what DGX scale must
close. Play evaluation uses the new public
`solver_environment_factory(world)` accessor so harnesses cannot drift
from the canonical teaching configurations (12 dataset tests; play-safe
gate green, 544 tests). Record:
[docs/runs/2026-08-18-solver-smoke-distillation.md](./docs/runs/2026-08-18-solver-smoke-distillation.md).
The owner gave an explicit go for the B1–B3 preregistered baseline
controls (2026-08-18), and **B1 is implemented**: the shared objective
gained the preregistered multi-horizon world loss — window-derived
power-of-two offsets, exactly {1, 2, 4} on the registered length-8,
burn-in-2 campaign window with `world_weight` split evenly across
horizons, bit-identical single-horizon behavior on length-2 smoke
windows, and flexible min-over-slots per horizon for all existing
variants — plus the control variant
`irene.thought_field.fixed_multi_horizon.v1`: 32 slots statically
partitioned 11/11/10 across horizons, each group trained only on its own
offset, persistence removed exactly like reset_slots, at precisely the
reference parameter count (29,674,318 trainable) inside the regenerated
architecture manifest (digest `8d93eeca…`). B2 has since landed (below);
B3 (task specialist) stays blocked on ladder dataset generation (4 new
baseline tests; play-safe gate green, 547 tests). Record:
[docs/runs/2026-08-18-multi-horizon-world-loss.md](./docs/runs/2026-08-18-multi-horizon-world-loss.md).
Thought-collapse instrumentation now answers the review's "does collapse
actually happen at K=32?" question with per-step evidence: the shared
objective reports `thought_pairwise_cosine_mean` (alert threshold 0.7),
`thought_duplicate_pair_fraction` (near-identical register pairs), and
`movement_query_slot_entropy` (readout utilization) alongside the
existing rank/private-energy/effective-slot diagnostics, on every logged
step in training and evaluation, with exact additive identities for
single-latent controls. Manifest regenerated (digest `1d73c7ef…`) for the
objective source change (2 new tests; play-safe gate green, 548 tests).
Record:
[docs/runs/2026-08-18-thought-collapse-diagnostics.md](./docs/runs/2026-08-18-thought-collapse-diagnostics.md).
The strategic review's highest-leverage cheap item ran:
`scripts/compare_baselines_smoke.py` trained all eleven registered
variants at smoke scale (64 steps, campaign window shape, identical seeds
and batches) with per-variant JSON rows pinned under
`docs/runs/artifacts/baseline-smoke-compare/`. Three honest signals: no
thought collapse anywhere (pairwise cosine 0.44–0.58, duplicates exactly
zero); the reference does **not** lead at smoke scale — it ties its
parameter-exact ablations and trails serial_depth (action loss 0.6892,
movement exact 0.208) and the parameter-matched GRU (0.7029) — so the
review's smoke-scale proceed-criterion for RCQ-v3 is **not met**, making
the qualified matched-baseline comparison the highest-value DGX item; and
B1 behaves as designed (static partition pays the expected world-loss
price, 0.9786 vs 0.8770). Record:
[docs/runs/2026-08-18-baseline-smoke-comparison.md](./docs/runs/2026-08-18-baseline-smoke-comparison.md).
The transfer-gap battery now has pinned smoke-scale zero-points:
`scripts/transfer_gap_smoke.py` trains a reference smoke model on
moving_shapes (128 steps) and plays it zero-shot across all six canonical
ladder worlds against no-op/random/reactive baselines under one play
configuration. Result: transfer is at or below the no-op floor
everywhere — exactly no-op on four worlds, slightly worse on pursuit
(−29 vs −24), catastrophically worse on maze_chase (−3,890 vs −161) —
and even in-distribution play is still at the floor after 128 steps.
The open question is now "how much scale precedes any transfer," and the
battery is one command that any future checkpoint can be dropped into
(play-safe gate unaffected, script-only). Record:
[docs/runs/2026-08-18-transfer-gap-smoke.md](./docs/runs/2026-08-18-transfer-gap-smoke.md).
Checkpoint comparison tooling landed:
`evaluation/checkpoint_compare.py` (analysis-only primitives — restricted
loading, run-identity guard, per-tensor relative-L2 difference table,
shared-batch metric deltas; `load_checkpoint` remains the only resume
path) and `scripts/compare_checkpoints.py` (CLI with a `--cpu` override
for analyzing CUDA-configured checkpoints locally). Verified end to end
on two real smoke checkpoints (198/223 tensors changed, actuator
coordination norms move fastest). A debugging note is pinned: dgx-smoke's
schema-1 cosine schedule decays LR to exactly 0.0 at
`max_optimizer_steps`, so hand-driven steps past the configured horizon
apply nothing (3 tests; play-safe gate green, 551 tests). Record:
[docs/runs/2026-08-18-checkpoint-comparison-tool.md](./docs/runs/2026-08-18-checkpoint-comparison-tool.md).
**B2 is implemented**: the recurrent world-model actor control
`irene.world_model_actor.gru_latent.v1` pairs the parameter-matched
monolithic GRU trunk with a learned latent transition model — action
embedder (307→64), residual transition (460→192→396), bottleneck decoder
(396→64→396) — adding exactly 235,800 trainable parameters for 29,879,700
total, +0.69% over the reference and inside the preregistered 1% band by
exact allocated-parameter enumeration. Its defining rollout world loss
cannot be expressed by the shared slot-suite objective, so the variant
has its own recipe family: the per-step horizon world-loss computation
was refactored into an overridable method (behavior-preserving), and
`LatentRolloutObjective` (`training/world_model_objective.py`) rolls the
GRU latent k times through the transition model on recorded action
targets and decodes to the frozen sensor encoding of the frame k steps
ahead, per horizon {1, 2, 4}; every other objective term and diagnostic
is inherited unchanged. `train.py` resolves the model's fail-closed
`training_objective_class_path` hook; the recipe is
`configs/training/baseline-stagea-world-model-actor.toml`; and the family
is pinned by its own manifest (`configs/world-model-actor-manifest.json`,
digest `4e603895…`, previous `be571ba4…`, `885aff85…`, `3d2e3cc0…`, then `c207401f…`) with a fail-closed 1%-band gate, not the slot-suite
manifest (regenerated for turn-weighted exclusive WASD softmax, digest `3d7ff5bd…`). B3 (task
specialist) remains blocked on ladder dataset generation (7 new tests;
play-safe gate green, 558 tests). Record:
[docs/runs/2026-08-18-world-model-actor.md](./docs/runs/2026-08-18-world-model-actor.md).
The smoke comparison probe now covers the B2 actor as its twelfth
variant: `scripts/compare_baselines_smoke.py` resolves each variant's
declared `training_objective_class_path` hook (the same discipline as
train.py) instead of hardwiring the shared objective. At 64 steps the
actor learns the action task at exactly its trunk's rate (0.7026 vs the
parameter-matched GRU's 0.7029, movement exact 0.083 vs 0.042) and pays
the expected world-loss premium for rollout prediction (1.0853, highest
in the suite); serial_depth still leads and the RCQ-v3 smoke
proceed-criterion remains unmet. Record:
[docs/runs/2026-08-18-baseline-smoke-comparison.md](./docs/runs/2026-08-18-baseline-smoke-comparison.md).
The transfer battery now covers the B2 actor too:
`scripts/transfer_gap_smoke.py --variant world_model_actor` trains the
actor through its declared rollout objective under the identical 128-step
protocol and writes rows to a per-variant artifact subdirectory. Result:
transfer is still at or below the no-op floor on every world — the gap is
not a direct-head artifact — but the actor's maze_chase failure is
markedly less catastrophic than the reference's (−1358 vs −3890, 138 vs
391 collisions), a one-seed-pair hint that is now a falsifiable DGX-scale
question with both zero-points pinned. Record:
[docs/runs/2026-08-18-transfer-gap-smoke.md](./docs/runs/2026-08-18-transfer-gap-smoke.md).
**Erratum — the smoke probes never trained past step 2.** Both
`compare_baselines_smoke.py` and `transfer_gap_smoke.py` built schema-1
configs with `max_optimizer_steps=2`; schema 1 forces the legacy cosine
schedule, whose multiplier is exactly 0.0 from step 2 — so all published
smoke-comparison and transfer-battery rows above (including the B2
addenda) measured two effective optimizer updates plus frozen evaluation.
The distill probes and all campaign recipes are unaffected (they set the
horizon to their real step counts); inter-variant comparisons within each
frozen table stay internally fair; all frozen-regime artifacts are
preserved untouched. Both probes are fixed to schema-2
constant-LR (the campaign baseline regime) and re-run with pinned rows
under `constant-lr/` artifact directories. Corrected headline findings:
the B2 world-model actor **leads the twelve-variant suite** on 64-step
validation action loss (0.5341) and its rollout world loss reverses from
worst-frozen to best-by-4× (0.0325 vs the pack's 0.125–0.24);
serial_depth's earlier "lead" was an initialization artifact; the
reference still does not lead (ties dense_routing at 0.5375) and the
RCQ-v3 smoke proceed-criterion remains unmet; at 128 real steps the
actor's maze_chase transfer sits **exactly at the no-op floor** (−161/17
collisions) while the reference stays catastrophic (−2198/221); and the
first above-floor in-distribution play appears at 512 steps (reference,
non-monotonic) and 1024 steps (actor, never below floor). Record:
[docs/runs/2026-08-18-smoke-probe-schedule-erratum.md](./docs/runs/2026-08-18-smoke-probe-schedule-erratum.md).
**B3 is implemented** at smoke scale: `MixedWorldBatchSource` mixes the
existing lazy moving_shapes and maze_chase datasets for a generalist
(round-robin, affine-shuffled train epochs, identity distinct from either
member; `DatasetConfig.kind` untouched), and specialists fine-tune copies
of that checkpoint on each member source — the §29 unchanged-generalist
versus game-specific fine-tune, not a new factory. A 16+8-step
constant-LR probe pins the maze specialist beating the frozen generalist
on maze_chase action loss (0.490 vs 0.578) and paying a transfer cost on
moving_shapes; the moving_shapes specialist does not beat the generalist
in-distribution at this scale. Campaign-scale materialization remains
DGX; the first-matched architecture campaign stays blocked. 9 new tests;
play-safe gate green, 566 tests, one expected POSIX skip. Record:
[docs/runs/2026-08-18-task-specialist.md](./docs/runs/2026-08-18-task-specialist.md).
The matched baseline suite gains the PLAN §28 item-12 control:
`irene.thought_field.independent_ensemble.v1` — four untied members of
eight slots each at width 352 (29,459,914 trainable, 0.72% under the
reference budget), registered in the architecture manifest (digest
`d19d09bf…`) and inside the verified parameter-matched fairness regime.
Record:
[docs/runs/2026-08-17-matched-ensemble-baseline.md](./docs/runs/2026-08-17-matched-ensemble-baseline.md).
Latency evidence now extends from benchmarks to play:
`evaluation/play_latency.py` converts the closed-loop evaluator's
manual-clock decision milestones into the fail-closed `LatencyEvidence`
contract (analytic pre-pinnable schedule, warmup-aware attempt IDs,
stale-frame rejections as `submission_failed`; 10 tests; play-safe gate
green). Record:
[docs/runs/2026-08-17-play-latency-evidence.md](./docs/runs/2026-08-17-play-latency-evidence.md).
The remaining §28 controls (fixed multi-horizon heads, world-model actor,
task specialist, recurrent transformer) were scoped and found blocked on
shared-objective or campaign-shape changes — the objective trains only a
short-horizon world loss and `horizon_logits` are consumed nowhere — so
they are now preregistered design proposals (B1–B4) awaiting explicit
owner sign-off. Record:
[docs/runs/2026-08-17-remaining-baseline-controls-preregistration.md](./docs/runs/2026-08-17-remaining-baseline-controls-preregistration.md).
Proposal B4 landed as a drop-in under the standing delegation (no
objective or campaign change): `irene.recurrent_transformer.carry_token.v1`
— a standard carry-token Transformer encoder control at width 568
(29,609,034 trainable, 0.22% under the reference budget), inside the
verified parameter-matched regime; the suite now covers ten variants.
Record:
[docs/runs/2026-08-17-recurrent-transformer-baseline.md](./docs/runs/2026-08-17-recurrent-transformer-baseline.md).
The rights ledger required by PLAN §20 now exists at
[docs/RIGHTS_LEDGER.md](./docs/RIGHTS_LEDGER.md): all six in-repo worlds
(moving shapes, pursuit, junction, occlusion, keys/doors, maze_chase) are
accepted as original works across the four required records, and every
external source — including Namco Pac-Man — is marked unreviewed and
un-ingested.

Local play-safe verification after the live v2 registration: every isolated
test module passed (299 tests, one expected POSIX skip in the trusted-final
suite, about 55 seconds, CPU-only). The first-matched campaign remains
fail-closed on its historical 500-step pin. The play-safe runner now sets
`PYTHONDONTWRITEBYTECODE` so later tests cannot leave `__pycache__` in the
registered source tree.

Historical Stage A DGX runs below remain valid failures. They are not a
starting checkpoint for RCQ-v2.

## Phase 0A: safe deterministic core

Status: operational and covered by the play-safe test suite.

Implemented:

- Fail-closed play-safe policy: GPU, capture, HID output, network, subprocess,
  background threads, and artifact writes are denied by default.
- One-thread, below-normal-priority test runner with CUDA hidden.
- Generic USB-HID-like control contract with exactly-once relative mouse
  impulses.
- Versioned, length-prefixed canonical hashes for controls, RGB frames, and
  model-visible observations.
- One canonical `ModelObservation` boundary containing resolved RGB/audio data,
  relative time, visible text, and only the preceding applied control. Storage
  references and privileged simulator fields cannot cross it.
- Provenance-labeled model-visible text.
- Deadline-bearing action envelopes with source frame and model-state version.
- Deterministic 16 × 16 moving-shapes pixel environment.
- Explicit SplitMix64 state; no global random generator.
- Checksummed, versioned snapshots with atomic restore.
- Golden 1,000-step replay hashes.
- Order-independent counterfactual branch execution.
- Immutable lifetime, step, and branch records.
- Strict allowlist projection from stored records to model inputs.
- Previous applied control, rather than requested control, is carried forward.
- Relative QPC timing is exposed to the model; absolute machine uptime is not.
- Split-leakage auditing over world lineage, source session, player, control
  mapping, and snapshot ancestry.
- Fake-clock continuous driver that advances during delayed inference and
  exposes only the newest observation.
- Bounded real-time catch-up that fails neutral on excessive clock jumps;
  unlimited catch-up must be selected explicitly for offline replay.
- Stale, duplicate, late, and wrong-frame action rejection.
- Lossless pending mouse-impulse coalescing, hold expiry, and watchdog
  neutralization.
- Strict environment-outcome provenance and monotonic frame/time validation.
- Immutable experiment manifests with explicit resource capabilities.
- One-checkpoint/no-game-ID/no-privileged-input compliance audit.

Verified:

- All isolated play-safe modules pass locally in CPU-only processes, including
  the real PyTorch forward/backward, sparse-action, multi-thought, checkpoint,
  launcher-contract, and RCQ-v2 tests (294 tests, one expected POSIX skip).
- All 161 tests present in the latest immutable DGX release passed in its
  bounded CUDA smoke container. The additional local tests landed afterward.
- The latest isolated local safe run completed in 18.6 seconds.
- The test process uses one CPU thread at below-normal priority.
- Imports start no threads and load no accelerator/capture libraries.
- No GPU, screen capture, real HID output, network, dependency install, video
  encoding, or background service was used.

## Phase 0B: physical real-time harness

Status: deliberately not started while the owner is gaming.

Still required:

- Native Windows Graphics Capture or DXGI adapter.
- GPU-texture path and no-model capture-to-control loopback.
- Real keyboard/mouse or virtual-gamepad adapter.
- Hardware kill switch and physical watchdog validation.
- QPC instrumentation through first visible action effect.
- 30–120 minute latency and thermal soak.

These operations must be explicitly armed in a non-gaming performance window.
Simulated timing results must never be reported as physical latency.

## Phase 1: model baselines and thought field

Status: trainable implementation and offline trainer completed; bounded DGX
smoke and the one-step full-model canary passed. The first procedural Stage A
bootstrap was stopped after its step-100 gate exposed an all-directions action
shortcut. The sparse-action objective, persistent slot identity, utility writes,
and decisive action/thought metrics were corrected and locally verified. The
fresh bounded 100-step DGX gate then failed action learning with near-zero
logits, while preserving broad thought geometry and learning the short world
target. Both matched tiny-set action overfit runs passed remotely; the
support-aware calibrated B loss reached the exact-set threshold sooner and is
selected for the broad held-out continuation gate. The registered 500-step
cosine-schedule continuation run then validly failed that gate at `51/96`
movement-exact and `6/26` changed-action-exact decisions. The matched
constant-after-warmup, equal-FLOP control improved to `72/96` movement-exact,
`11/26` changed-action-exact, eight movement false positives, and zero
opposite-direction conflicts. It nevertheless validly failed the frozen gate
on exact actions, changed actions, recall, and value loss. The 1,000-step
bootstrap and causal ablation remain blocked while matched baselines and a
larger untouched evaluation are built.

Implemented in the dependency-free contract layer:

- Locked thesis-MVP layout: width 384, 48 belief tokens, 16 working-memory
  tokens, 32 thoughtlets × 3 registers, 8 goal tokens, and three cycles.
- 307 fixed generic actuator queries.
- Direct actuator access to sensors, belief, all thoughts, working/retrieved
  memory, and goals.
- No pooled integration token.
- Private first thought cycle and two routed neighbors in later cycles.
- Shared BrainCell weights across thoughtlets and cycles.
- Exact shape, persistent-state byte, and tied-core parameter estimates.

Implemented behind lazy, opt-in PyTorch imports:

- Spatial CNN token encoder with deterministic exact-grid CUDA backward.
- Persistent belief, working memory, goal context, and K × R thought state.
- One shared BrainCell reused across all thoughtlets and cognitive cycles.
- Private first cycle, sparse later routing, and cycle-0 through cycle-3
  anytime actuator exits.
- Direct 307-query generic-HID readout without one pooled integration token.
- Deterministic, split-namespaced MovingShapes sequence dataset; the lazy
  default describes 1,048,576 transitions without materializing them.
- Causal recurrent unrolling with burn-in and labels kept outside
  `ModelObservation`.
- Class-balanced structured action loss, value loss, one-step future-feature
  loss, and thought-diversity loss.
- BF16 CUDA/FP32 support, gradient accumulation/clipping, validation, canonical
  JSONL metrics, and optimizer-boundary checkpoint/resume.
- Checkpoint identity over config, data, source, runtime, precision, device,
  deterministic CUDA workspace, RNG state, and file SHA-256.
- Foreground `dgx-smoke`, one-step full-thesis `dgx-thesis-canary`, and
  acknowledged detached `phase1-bootstrap` configurations.

First DGX execution record:

- Immutable release: `r20260816t173317z-f01fc3d7e142`.
- Smoke: all 120 tests plus 3 BF16 optimizer steps; checkpoint SHA-256
  `4560130c26a73cb663d7e1b971bf4f6534baa2153f2f788c2e9e9ed1833a2a88`.
- Full-thesis canary: 1 step, 354,292,092-byte checkpoint, training loss
  `0.86640954`, validation loss `7.95379877`, and pre-clipping gradient norm
  `20.923` clipped to the configured maximum of `1.0`. The validation loss is
  expected for an untrained one-step checkpoint.
- Diagnosed baseline: `phase1-bootstrap-r20260816t173317z` was stopped after
  step 110. The validation gate's `0.98994957` key accuracy plus `1.0` recall
  implied about 2.57 false-positive keys per frame, matching an all-W/A/S/D
  shortcut. Its intact step-100 checkpoint SHA-256 is
  `e7e34b528a46455106649f00f3d4f15f766bf98ea3aa6c4e0c7d83fc415cca29`.
- Full details: [first DGX run record](./docs/runs/2026-08-16-first-dgx-run.md).
- Corrected release and failed 100-step gate:
  [corrected Stage A gate record](./docs/runs/2026-08-16-corrected-stagea-gate.md).
- Matched action-path diagnostic:
  [action overfit A/B record](./docs/runs/2026-08-16-action-overfit-ab.md).
- Failed broad held-out gate:
  [Stage A continuation-gate record](./docs/runs/2026-08-16-stagea-continuation-gate.md).
- Improved but failed constant-schedule control:
  [Stage A constant-schedule record](./docs/runs/2026-08-16-stagea-constant-gate.md).

Next execution and coding targets:

1. Add parameter-, data-, and compute-matched monolithic, no-communication,
   and wider recurrent baselines before another architecture claim.
2. Add targets/losses before using confidence, hold-plan, lifecycle, focus,
   horizon, memory-write, or other currently auxiliary heads for live control.
3. Freeze a larger untouched multi-seed evaluation and report both equal-FLOP
   and equal-wall-latency comparisons.
4. Expand split auditing to generator, mechanics, asset, and game-family axes
   once those identifiers are present in the dataset manifest.

This Stage A run is procedural research training, not evidence of general game
understanding and not a live-control-ready checkpoint. Unsupervised auxiliary
heads must remain disconnected from physical control.

Local play-safe verification still forbids GPU execution, capture, HID output,
network access, installs, and background work. Accelerator execution is routed
through the explicit, resource-bounded DGX workflow.
