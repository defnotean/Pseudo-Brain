# Implementation status

Updated: 2026-08-17

## Current campaign: RCQ-v2

Status: live qualification `rcq_v2_reference_v2` is **terminally failed**. The
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
itself is deliberately not yet created. Record:
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
`78ba9cfc1eb56782609f586ac7f3ef393a7fb6ddd6f017546861c3e92d0bb0e2` (previous
`8a41131e…`, `f4e9b355…`, `eb46988b…`, `5decb402…`, and `30d4c119…`; all are
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
