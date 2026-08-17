# Implementation status

Updated: 2026-08-16

## Current campaign: RCQ-v2

Status: local freeze and operator documentation in progress. No qualification
run, registration, or final TEST has been launched.

RCQ-v2 asks whether the existing single model can learn a state-conditioned
W/A/S/D policy and a useful value estimate on synthetic moving-shapes, for one
seed, under a frozen recipe. A later pass would not be closed-loop gameplay or
architecture superiority. Read [CURRENT_WORK.md](../CURRENT_WORK.md) and
[docs/OPERATOR_GUIDE.md](./docs/OPERATOR_GUIDE.md) before [docs/RCQ_V2_PROTOCOL.md](./docs/RCQ_V2_PROTOCOL.md).

The matched-baseline architecture manifest at
`configs/baseline-architecture-manifest.json` is a separate identity used by
architecture-comparison tests. It was stale relative to
`src/irene_brain/training/objective.py` and must be regenerated as an
intentional freeze before those tests, or the full play-safe suite, can pass.
That freeze does not open TEST and does not start DGX training.

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

- All 178 tests pass locally in isolated CPU-only processes, including the real
  PyTorch forward/backward, sparse-action, multi-thought, and checkpoint tests.
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
