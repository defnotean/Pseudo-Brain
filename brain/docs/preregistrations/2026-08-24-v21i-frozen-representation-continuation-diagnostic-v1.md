# V2.1i frozen-representation continuation diagnostic v1

Status: preregistered diagnostic; never a candidate or qualification run.

## Question

The balanced-intervention seed-42 V2.1i v3 run finished its eighth isolated
hazard-path pass while TRAIN-FIT BCE was still falling. This diagnostic asks
whether the failed V3 discrimination margins are primarily caused by an
undertrained existing hazard path or by belief features whose hazard signal
does not transfer to disjoint TRAIN-CAL episodes.

## Immutable upstream

- V3 result SHA-256:
  `3f91921b7ca63148addd3204cdb1e96b130f26240f49277a269ac8f079b13936`
- Uncalibrated parent checkpoint SHA-256:
  `e5a4c1e92cdf5649679f344ce9a032397e69a0a5c93bb9d3a7bebc4bc1352081`
- Parent state SHA-256:
  `2619b5b0a180bdd55296d3471d89725f4346110e74ef093f84c3068a32e6576d`
- V3 source bundle SHA-256:
  `681f3f45422316ec40c299b6c449a90a4790f544ea21311597416cfa7fbb3862`
- TRAIN-FIT manifest SHA-256:
  `835fa63a0388c23eff17713b55836626f42ddd361ec2829948359d2187f65179`
- TRAIN-CAL manifest SHA-256:
  `aed8e64a9889fab82eea2b2551422c4a7f44cf96ea2b80864f857512218a2a83`

The diagnostic must reject any mismatch. It loads the uncalibrated parent,
never the absent calibrated child.

## Namespace boundary

Only the exact V2.1i TRAIN-FIT and TRAIN-CAL partitions may be constructed.
DEV, CPU-QUAL, and TEST must not be constructed, opened, or evaluated.
TRAIN-CAL is evaluation-only and cannot alter optimization, stopping, model
selection, milestone selection, or interpretation thresholds.

The diagnostic uses a local two-entry contract factory. It constructs exactly
TRAIN-FIT (`train`, offset `16777216`, 128 episodes) and TRAIN-CAL (`train`,
offset `16777344`, 64 episodes). It never calls the development runner's
three-partition factory, so no DEV or VALIDATION configuration is constructed.

## Causal scope limitation

The registered dataset recurrence used by `collect_evidence` injects the prior
factual `actual_reward` and `actual_hazard`. The live `act_rgb` signature has
neither signal. Therefore this diagnostic studies only hazard extraction from
registered, outcome-feedback-conditioned beliefs. It is not live-signature
matched, cannot support live qualification, and cannot conclude that the belief
representation is sufficient for live play. A separate exact-live matched probe
is required; it is deliberately outside this diagnostic.

## Frozen representation and optimizer

All parameters and buffers are frozen except the existing 44,161 parameters
under these outcome-model prefixes:

- `hazard_state_trunk.`
- `hazard_action_embedding.`
- `hazard_outcome_trunk.`
- `hazard_head.`

Beliefs and labels are cached once from the immutable parent. Training uses
only the cached 1,536 complete TRAIN-FIT roots and all five action branches.
TRAIN-CAL contributes 768 cached roots only to fixed evaluations.

The V3 parent did not retain hazard-refinement optimizer state. The diagnostic
therefore records an explicit fresh AdamW restart with learning rate `1e-3`,
weight decay `1e-4`, gradient clip `1`, 24 complete roots per batch, and seed
`30042`. Every pass is a deterministic complete root permutation without
replacement. There are exactly 16 added passes and 1,024 optimizer steps. No
early stopping or model selection is permitted.

The exact 16 permutation digests are derived again from seed `30042` and 1,536
roots during pass-chain validation. A merely well-formed or unique alternative
digest is rejected. Every pass records optimizer-state SHA-256 before and after;
the first digest binds the empty fresh-AdamW state, adjacent optimizer digests
must match, and milestones repeat the contemporaneous optimizer-state digest.

## Fixed milestones and evidence

Create-only diagnostic checkpoints are emitted at added passes
`0, 1, 2, 4, 8, 16`, corresponding to total refinement passes
`8, 9, 10, 12, 16, 24`. Each checkpoint is explicitly
`diagnostic_not_candidate`, carries the full parent/source/dataset/evidence
bindings, model and fresh optimizer states, finite-state audit, pass permutation
digests, and source bindings. After each create-only checkpoint write, the
terminal result or canonical failure artifact carries the checkpoint file's
SHA-256 receipt; a file cannot contain a trustworthy hash of its own final bytes.
Every pass records full-model state
SHA-256 before/after and cumulative optimizer-step start/end. The first pass
must begin at the exact pinned parent state, every adjacent state must match,
and the final chain must end at optimizer step 1,024.

At every milestone, evaluate raw uncalibrated TRAIN-FIT and TRAIN-CAL logits.
Record aggregate and per-action BCE, Brier, ROC AUC, PR AUC, prevalence, factual
metrics, and 20 deterministic cross-episode derangements using seed `31042`.
No calibrator is fit or installed.

## Source freeze and preregistration receipt

The composite diagnostic source bundle includes the frozen V3 production source
bundle plus the exact bytes and individual SHA-256 values of this script, this
document, and the focused test file. Before any execution, the same frozen
script must be invoked in `--register-only` mode to create a separate create-only
JSON preregistration receipt. Normal execution requires that receipt, recomputes
the complete bundle, requires exact payload equality, and records the receipt's
own file SHA-256 in every checkpoint and terminal result. Any later script,
document, or test drift therefore fails before parent loading or dataset-source
construction.

The composite hash cannot be embedded literally in this document because this
document is itself hashed into that composite: adding the hash would change the
hash recursively. The non-self-containing create-only receipt is the canonical
place where the final composite is pinned. Its SHA-256 is supplied and recorded
externally before diagnostic execution.

## Preregistered interpretation

Compare added pass 16 with added pass 0. No intermediate checkpoint may be
selected.

Hazard-path undertraining is supported only if all are true:

- TRAIN-FIT BCE improves by at least `0.02`;
- TRAIN-CAL BCE improves by at least `0.01`;
- TRAIN-CAL aggregate ROC AUC increases by at least `0.02`;
- TRAIN-CAL worst-action ROC AUC increases by at least `0.01`;
- at least four of five TRAIN-CAL per-action ROC AUC values do not decline;
- TRAIN-CAL shuffled/model BCE ratio increases by at least `0.02`.

A frozen-belief generalization limitation is supported when TRAIN-FIT BCE
improves by at least `0.02`, but TRAIN-CAL aggregate ROC AUC gains less than
`0.01` and worst-action ROC AUC gains less than `0.005` (including declines).

All other outcomes are inconclusive. In particular, a TRAIN-FIT BCE improvement
below `0.005` while TRAIN-CAL remains weak is not sufficient to attribute the
failure to the belief representation; it may reflect optimizer or head-capacity
limits.

## Publication prohibition

Every output and checkpoint is diagnostic-only, claims no qualification, and
sets `candidate_publication_allowed=false`. This script has no candidate path,
does not call the V2.1i development gate, and cannot emit a production model.
Every artifact also states that its live-qualification implication is `none`.
