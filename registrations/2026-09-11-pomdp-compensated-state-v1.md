# POMDP compensated-state v1 — bounded candidate experiment

Registered 2026-09-11 UTC before candidate training. This is separate from all
historical PB21 scientific gates and does not reopen them.

The existing 34,372,860-parameter checkpoint fails the owner's absolute logit
parity tolerance: 1.9073486328125e-05 on the 182-token multi-turn CPU audit,
against a strict limit of 1e-6. It remains unchanged.

Hypothesis: float64 arithmetic plus a high/low float32 representation of each
active recurrent vector can meet the existing tolerance without token replay,
KV caching, additional parameters, or additional working-state storage.
Physical layout remains 16 x 64 float32 = 4,096 bytes. Eight slots hold logical
thought vectors and eight hold their numerical residuals. This reduces logical
slot capacity to eight; it does not claim sixteen independent thought vectors.
The software policy uses four logical slots. Routing is unsupported in this
candidate until a separate compensated-routing implementation is verified.
Float64 weights, transient activations and episodic memory increase memory and
runtime outside the fast-state allocation; no speed advantage is claimed.

Pre-training gates: all existing targeted regressions; actual checkpoint
streaming/parallel logit difference <1e-6 on the fixed 182-token audit and longer
multi-turn/batched probes; physical working state exactly 4,096 bytes float32;
trainable parameters <36M; initial task specification remains the only pointer
source. No rescaling logits or relaxing tolerances is permitted.

Allowed training after those gates: one candidate run, at most 350 optimizer
steps, seed 1337, 120 open procedural tasks, local CPU, one thread, CUDA hidden.
Faulty seed actions are teacher-forced context with zero imitation loss. Actual
environment observations use shared formatting and deterministic error summaries.
The exact training bank is hashed before updates. No held-out reference solution
is provided to policy training or generation.

Post-training evaluation: ten tasks per existing A/B/C tier, seed 42, at most four
actions per task, reset before every task. Level B is reported as resampled open
templates, not unseen algorithms. Preserve the candidate and full trace receipt
in a new run directory even on failure; never replace the historical champion.
Check the original requested capability thresholds and the numerical invariants.
An autonomous repair requires observed failure, an ingested repair phase, a later
successful code mutation and external hidden-test-approved FINISH. Scripted
trajectories do not count. No frontier-equivalence claim follows from this run.
