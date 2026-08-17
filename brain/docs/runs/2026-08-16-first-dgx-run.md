# First DGX Spark run record

Recorded: 2026-08-16

Immutable source release: `r20260816t173317z-f01fc3d7e142`

## Completed gates

| Gate | Result | Checkpoint |
|---|---|---|
| Bounded CUDA smoke | All 120 tests passed in the CUDA container, followed by 3 BF16 optimizer steps. | `4560130c26a73cb663d7e1b971bf4f6534baa2153f2f788c2e9e9ed1833a2a88` |
| Full-thesis canary | Completed 1 optimizer step and validation. Checkpoint size: 354,292,092 bytes. | `77306c6c476d11d028ee3a4f252a956f843ac7269306a4da50ef290cd91abdeb` |

The full-thesis canary reported:

- Training loss: `0.86640954`.
- Validation loss: `7.95379877`; a high untrained validation loss is expected
  at this one-step gate.
- Pre-clipping gradient norm: `20.923`; gradients were clipped to the
  configured maximum norm of `1.0`.

## First bootstrap diagnosis

Run `phase1-bootstrap-r20260816t173317z` started as the acknowledged Tmux
execution of the 1,000-step `phase1-bootstrap` configuration. At the initial
status snapshot it was active at approximately 70% GPU utilization and 3,781
MiB reported GPU memory. These numbers are an initial observation, not a
throughput benchmark.

The first optimizer telemetry confirmed forward progress through step 30. The
reported training loss moved from `1.02052307` at step 1 to `0.57531011` at
step 30, while the original aggregate key accuracy moved from `0.1062` to
`0.9581`. Positive-key recall at step 30 was `1.0`.

The step-100 validation gate exposed a shortcut in that original metric and
objective. Validation reported key accuracy `0.98994957` and positive-key
recall `1.0`. With no false negatives, that accuracy corresponds to about
`2.57` extra keyboard keys per frame, closely matching the invalid strategy of
turning on all four W/A/S/D keys for a target containing only one or two. The
aggregate balanced BCE diluted each false positive across roughly 294 inactive
button channels.

The same gate also exposed geometric thought-slot collapse. The off-diagonal
cosine-squared diversity measure rose from `0.175` at training step 1 to
`0.405` at step 100, and validation reported `0.937` (lower is better). For 32
slots, those values correspond to a summary-state participation-rank proxy of
approximately `4.98`, `2.36`, and `1.06`, respectively. The original code also
used fresh Gaussian refresh noise on every training frame but zero refresh
noise after the first evaluation frame, so train and validation slot geometry
were not directly comparable. The successor code now uses stable slot-distinct
identities in both paths, fixes the sparse action loss and utility write gradient,
and reports exact-set, conflict, rank, actuator-use, and applied-lifecycle
diagnostics. Those fixes require a fresh release and checkpoint; this historical
run cannot be resumed into them.

The run was therefore stopped after it emitted step 110, before spending the
remaining budget on the wrong objective. The intact step-100 checkpoint is
354,068,357 bytes with SHA-256
`e7e34b528a46455106649f00f3d4f15f766bf98ea3aa6c4e0c7d83fc415cca29`.
No artifact was deleted. This checkpoint is a diagnosed baseline, not a model
candidate.

## Interpretation and safety boundary

This is procedural Stage A training on the deterministic MovingShapes task. It
does not demonstrate general game understanding and is not ready for live
control. The confidence, hold-plan, lifecycle, focus, horizon, memory-write,
and other currently unsupervised auxiliary heads must not control a live
system.

All accelerator work ran on the DGX Spark. The local PC performed no model
compute, screen/audio capture, or physical input injection for these runs.
Checkpoints and run artifacts are not stored in source control; their hashes
above identify the recorded artifacts.
