# Matched multi-seed comparison protocol

Architecture results are accepted only for an externally pinned campaign
registered before any result is inspected. The pre-training registration fixes the exact
ordered training seeds, suites and weights, every variant/seed run, training
data and code, normalized budget, recipe, effective config and optimizer cursor,
checkpoint-selection rule, stopping rule, criterion digest, architecture
manifest digest, and claim scope. Missing or extra cells fail closed; collecting
additional seeds or swapping a suite after seeing results creates a different
campaign rather than extending the registered one.

Checkpoint hashes cannot honestly be known before training. A separately
hashed post-run execution manifest therefore binds the exact registered run
matrix to the checkpoints produced at the preregistered cursor. Evaluation
requires independent expected digests for both phases. It rejects an execution
that omits a planned run, adds a replacement run, changes a run ID or cursor,
or binds itself to another campaign.

The execution-manifest producer is a trust boundary: before pinning that
manifest, it must hash each checkpoint's bytes and verify the checkpoint's
embedded run, effective-config, data, source and cursor identities against the
campaign. The statistical evaluator checks the resulting bindings and all
benchmark cells; it does not itself deserialize model checkpoints.

The architecture manifest has both a self-digest and an externally supplied
expected digest. A modified-and-rehashed manifest is rejected. For the
parameter-matched regime, the evaluator also checks registered variant and
recipe identities, the required fixed controls, allocated trainable parameter
counts, and the declared tolerance. Each template and per-seed effective TOML
is embedded in the registration: its raw and canonical hashes are recomputed,
the two representations must parse identically, and the effective config may
change only `run.name` and `run.seed`. The seed, model factory and optimizer
cursor must match their registered identities. A common training-budget digest
is recomputed from every full config after removing only the run name, seed and
architecture factory, so learning rate, schedule, objective, accumulation,
data settings, precision, determinism, evaluation cadence and step budget
cannot silently differ across variants. Every benchmark cell must match the
registered sample-manifest, evaluator, run, checkpoint, recipe, training-data,
training-code, budget, and cursor identities. The complete canonical raw matrix
and final report are SHA-256 hashed.

`irene_brain.evaluation.multiseed_comparison` operates on normalized scores in
`[0, 1]`. It first averages suites within each training seed, preserving the
training seed as the independent unit. Each reference-versus-control comparison
then reports:

- every paired seed delta;
- paired mean and median deltas;
- positive-seed count;
- every per-suite mean delta;
- an exact two-sided paired sign-flip randomization p-value;
- Holm-Bonferroni familywise correction across registered controls;
- separately preregistered mean-effect and per-suite regression screens.

The criterion object is hashed and must be fixed before training results are
read. At least six seeds are required for a two-sided exact test to possibly
fall below `0.05`; the first comparison campaign should use eight. Exact
enumeration is capped at 20 seeds. The paired sign-flip test assumes that, under
the null, per-seed reference-minus-control differences are exchangeable under
sign reversal (equivalently, their joint null distribution is sign-symmetric).
Training seed—not a frame, episode, suite, or checkpoint—is the independent
statistical unit.

A statistical pass supports only the named benchmark scope and verified
fairness regime. It does not establish causal thought use, real-time advantage,
general game intelligence, or human-like cognition. Those claims require the
separate causal, latency, and cross-environment evidence layers.
