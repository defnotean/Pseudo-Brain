# PB21M-FRESH-BAL-CAL-FIRST-v1 preregistration

Date frozen: 2026-08-25

Canonical basename: `2026-08-25-pb21m-fresh-bal-cal-first-v1`

Classification: fresh architecture qualification, not a checkpoint or model
qualification. A pass can nominate one fixed hazard-head recipe for a later,
separately preregistered scaling study. It cannot authorize DGX/full-model
training or support a scalability, live-play, or human-speed claim.

## Scientific question and hypothesis

PB21K established that the detached NZ hazard representation learns useful raw
hazard ranking on fresh data, while standard all-action affine calibration did
not remove systematic factual-versus-all-action error. PB21K's paired MIX
calibration improved factual BCE and Brier, but missed the frozen all-action
BCE noninferiority lower bound and failed every F2 cell. Within matched raw-logit
deciles, factual-versus-all prevalence gaps persisted for every semantic action.
Those results support moving the factual emphasis upstream into scorer fitting,
rather than another scalar post-hoc calibration retry.

PB21L was intended to test that upstream mechanism on consumed PB21K data, but
failed during evidence packaging before publishing authoritative evidence. It
produced no interpretable candidate metrics and may not be retried. PB21M
therefore makes no arm choice from PB21L observations.

The fixed hypothesis is:

> Relative to the exact all-action BASE objective, a 50/50 mixture of the
> all-action objective and a per-action-balanced factual objective will improve
> factual BCE and Brier after standard train-only per-action affine calibration,
> while remaining within frozen all-action and nonselected-complement
> noninferiority margins, on fresh CAL and DEV namespaces.

This licenses one bounded fresh trial of BAL-UPMIX. It does not assert that the
hypothesis is already true, that every rejected earlier idea was wholly wrong,
or that a successful hazard head is a complete artificial brain.

## Immutable scientific lineage

The run binds and validates the immutable PB21K scientific parent and the
terminal PB21L infrastructure-failure envelope before registration or source
construction.

PB21K identities:

- registration SHA-256:
  `f6bc3e25412ea6c3058bc38ace7622e30fdec72c6732388cb7a7d7dfa38f7058`;
- attempt SHA-256:
  `bd2ae53311733900a421b9928f2db73f32e5aa82d14a523f27f7bcf367b2a094`;
- evidence SHA-256:
  `9d57bf4a6bd60121e8f11e2b9405c1a6d64a9346381503054d6ad9cd4ece385c`;
- evidence-manifest SHA-256:
  `256f25b832aad90457fd8528979a6a60a24992d265dc864771b8a1f569c29be5`;
- result SHA-256:
  `083ac9c44d3327f4f806b1e4842ea3991ec54e5bab65992e7c5446ad09ca02d4`;
- source-bundle SHA-256:
  `b13b764c2ff55dd15649945fec7c05211a3c59fc04c7ecfb667d1b711590e03a`.

PB21L identities:

- registration SHA-256:
  `069ff428f84b772a7b853637248d45deb225738794f918a5e8887bce9f8937f0`;
- attempt SHA-256:
  `7a361cb03cb25c3699ae3b6fe2dcea26e1dd992d3217ea7026dc7b7de8935156`;
- failed result SHA-256:
  `eaf7bc1565666e8ddc371f02643b7e79a7b2c7a010798d35731fb251ffa2ad1c`;
- registered source-bundle SHA-256:
  `21ba2a450fcf2106b60dd596ebc25cf1f3ccbb6cd7f1d2ff935281601f243122`.

PB21L must remain
`exploratory_consumed_data_failed_nonqualifying`, with null authoritative
evidence, `retry_allowed=false`, and its recorded `AttributeError`. PB21M does
not reuse a PB21K or PB21L FIT, CAL, DEV, bootstrap, or derangement array.

## Fresh namespace contract

The old maze-chase environment contract remains fixed: sequence length 16,
four burn-in ticks, twelve post-burn roots per episode, five all-action
counterfactual hazard targets, balanced intervention probability 0.5,
batch-one strict-live replay, current RGB, and internally owned prior action.
Actual reward and actual hazard are absent from the live input, and pending
feedback is installed only after its tick's prediction.

| Label | Role | Split | Half-open local seed range | Episodes | Roots |
|---|---|---|---:|---:|---:|
| F0 | TRAIN-FIT | train | [167772160, 167772416) | 256 | 3072 |
| C0A | TRAIN-CAL-A | train | [167772416, 167772672) | 256 | 3072 |
| C0B | TRAIN-CAL-B | train | [167772672, 167772928) | 256 | 3072 |
| F1 | TRAIN-FIT | train | [167772928, 167773184) | 256 | 3072 |
| C1A | TRAIN-CAL-A | train | [167773184, 167773440) | 256 | 3072 |
| C1B | TRAIN-CAL-B | train | [167773440, 167773696) | 256 | 3072 |
| F2 | TRAIN-FIT | train | [167773696, 167773952) | 256 | 3072 |
| C2A | TRAIN-CAL-A | train | [167773952, 167774208) | 256 | 3072 |
| C2B | TRAIN-CAL-B | train | [167774208, 167774464) | 256 | 3072 |
| DEV | sealed endpoint | validation | [184549376, 184550144) | 768 | 9216 |

The split-prefixed effective DEV range is
`[4611686018611937280, 4611686018611938048)`. Registration performs the
central occupied-range collision audit over all frozen historical and
qualification namespaces and must report 34 records, zero collisions, and a
pass. Registration may compute config-only dataset manifests but may not
construct a source, episode, label, feature, or prediction.

Every FIT/CAL partition must contain exactly 3,072 distinct root IDs and 256
distinct episode-group IDs, with every group repeated for exactly 12 roots;
hazard targets must be exactly binary. Root sets and episode-group sets must be
pairwise disjoint across every constructed PB21M partition. Each ordered
source sequence, split, role, dataset manifest, and partition algorithm is
authoritative evidence. Any collision, identity drift, or ordering drift fails
the attempt.

## Fixed architecture and paired objectives

Both arms use the exact PB21J/PB21K NZ masked-superset scorer. Training begins
with the 485-input, 87,961-parameter masked head. Only N `[0:120]` and Z
`[240:360]` are live; every masked column is canonical float32 positive zero.
After fitting, the exact tensor-copy pruning map produces the 240-input,
58,561-parameter NZ head. Full-FIT padded/pruned batch-one logits must agree
within `1e-5`. Compact-from-step-zero training is not equivalent and is
forbidden.

There are exactly three FIT cohorts, three initialization seeds
`[91042, 92042, 93042]`, and two paired objectives, for 18 heads. Within a
FIT/seed cell, BASE and BAL-UPMIX start from byte-identical parameters and use
the same 32 complete relative permutations from seed `94042`.

For target `y[i,a]`, logit `z[i,a]`, root count `N=3072`, and the factual
action set `F_a={i: factual_action[i]=a}` with `n_a=|F_a|`:

- BASE is exactly
  `J_BASE = mean_(i,a) BCE(y[i,a], z[i,a])`;
- BAL-UPMIX is exactly
  `J_BAL = 0.5 * mean_(i,a) BCE(y[i,a], z[i,a])`
  `+ 0.5 * (1/5) * sum_a mean_(i in F_a) BCE(y[i,a], z[i,a])`.

For BAL-UPMIX, all `n_a` must be positive and are frozen from the complete FIT
cohort before optimization. A minibatch root whose factual action is `a` gets
factual-term weight `N/(5*n_a)`. The theoretical float64 values and exact
float32 values consumed by training are evidence. Current factual action is a
loss selector only; it is never added to the feature vector. There is no POOL
arm, objective-weight search, adaptive arm choice, early stopping, or fallback.

Every head receives 32 complete 3,072-root passes, root batch size 24, exactly
4,096 optimizer steps, fresh AdamW with learning rate `1e-3` and weight decay
`1e-4`, and gradient clipping at 1. Total fitting is exactly 73,728 optimizer
steps. Training telemetry binds initial/final states, pass numbers, losses,
gradient summaries, and every permutation digest. The parent model remains
frozen and must be byte-identical before and after the attempt.

## Standard AA calibration and support

AA is the only permitted calibrator. It is the existing train-only per-action
affine calibrator, fit on CPU float64 with L2 `1e-6`, minimum scale `1e-4`, at
least 20 examples and two examples per class/action, at most 100 iterations,
tolerance `1e-10`, and `per_action_affine` mode. MIX and any calibrator
selection are forbidden.

For each matched cohort and each of the 18 heads:

1. fit `AA-A` on CjA and predict only CjB;
2. fit `AA-B` on CjB and predict only CjA;
3. concatenate out-of-fold rows in canonical CjA then CjB order, where CjA
   predictions come from `AA-B` and CjB predictions come from `AA-A`;
4. fit one final AA on canonical CjA then CjB for possible later DEV use.

Both cross-fit AA calibrators for both arms in every cell must accept. Every
final AA must also accept before DEV can open. Each fitting half must satisfy
at least 20 observations, at least two positives, and at least two negatives
independently for every semantic output action. Every heldout CAL fold must
also have at least 20 observations, two positives, and two negatives for each
action in each claimed all-action, factual, and nonselected-complement domain.
The combined A+B final-AA fit and every claimed DEV action/domain must satisfy
the same at-least-20/two-positive/two-negative minimums; insufficient support
is evaluated before any undefined ranking metric can be requested.
FIT priors must be
finite and strictly between zero and one; every factual action must occur in
FIT, and every FIT factual-action subset must itself contain at least 20 roots,
two selected-target positives, and two selected-target negatives. Evaluated
factual and nonselected-complement domains must contain
both classes for every action. Missing support is a terminal gate failure, not
an invitation to pool folds, change regularization, or retry.

Calibration provenance is evidence, not an unchecked string. For every
cross-fit and final calibrator, it binds source namespace, TRAIN split, exact
source partition, canonical sorted-unique calibration-group-set digest,
canonical sorted-unique upstream-FIT-group-set
digest, pruned-head state SHA-256, dataset-manifest identity, registered source
bundle, and partition algorithm. Authoritative reload reconstructs this
provenance and deterministically refits every calibrator from stored logits and
targets; scales, biases, acceptance, output logits, and probabilities must be
byte exact. Calibrator group digests use the calibrator's canonical sorted
unique group-set digest; separate source evidence hashes preserve ordered root
and group arrays so order and multiplicity cannot be erased by that digest.

## CAL-first lifecycle and separate evidence

The lifecycle is create-only and ordered:

1. freeze the implementation, tests, this preregistration, exact parent
   artifacts, evidence schemas, and resource contract into one registration;
2. before registration, run a fixture-only integration path that fits a real
   `PerActionAffineHazardCalibrator`, builds complete CAL evidence with that
   object, publishes a disposable fixture artifact, reloads it with
   `allow_pickle=False`, reconstructs provenance, refits, and verifies it;
3. validate the canonical registration and immutable parent, run the resource
   preflight, then publish the attempt before constructing any source;
4. construct FIT and both CAL folds, verify finite geometry, frozen support,
   and nine-partition disjointness, and only then fit/prune all 18 heads;
5. publish the deterministic create-only CAL NPZ, reload and validate it, refit
   AA, and only then compute the CAL futility decision;
6. publish the complete CAL decision receipt;
7. if CAL fails, publish a terminal CAL-only negative result and never create a
   DEV-open receipt or DEV source;
8. only if CAL passes, publish the DEV-open receipt before constructing DEV;
9. publish the separate deterministic create-only DEV NPZ, reload and validate
   it, then compute DEV gates and the terminal result.

CAL evidence binds exact axis IDs and maps; FIT/CAL targets, actions, root IDs,
group IDs, source manifests, and ordered source digests; factual counts and
consumed float32 weights; full training telemetry and state hashes; FIT NZ
feature digests; pruning columns, complete map digest, pruned state, and
full-FIT transfer audit; both fold raw-logit tables; cross-fit and final AA
parameters, acceptance and complete provenance; OOF logits/probabilities;
bootstrap indices; and derangements.

DEV evidence is a different frozen schema. It binds the authoritative CAL
evidence SHA-256, complete CAL decision receipt SHA-256, DEV-open receipt
SHA-256, registered DEV source identity, ordered DEV roots/groups/cluster
geometry, fixed final-AA parameters and provenance through the CAL evidence,
raw and calibrated predictions, bootstrap indices, and derangements. It may
not silently duplicate a mutable in-memory CAL table.

Both NPZ files use exact registered key sets, canonical C-contiguous non-object
arrays, deterministic sorted ZIP entries, fixed metadata, and create-only
publication. Reload verifies file SHA-256, byte length, key-set digest, every
array dtype/shape/SHA-256, complete manifest digest, semantic axes, source
cross-links, numerical transforms, and fixed random arrays. Scientific metrics
are computed only from reloaded evidence.

Once the attempt exists, every exception, publication failure, gate failure,
or partial lifecycle is terminal. Existing attempt, evidence, receipt, or
result paths prohibit another attempt. `retry_allowed` is always false. No
PB21M namespace can be recycled under a new basename.

## Frozen resampling and causal tests

Alpha is one-sided `0.025`; quantiles use NumPy's `linear` method. Fixed
initialization replicas are never resampled.

CAL uses seed `95042` to generate one canonical uint16 bootstrap array with
shape `[50000,3,2,256]`: 50,000 whole-episode resamples, independently indexed
for each FIT cohort and CAL fold, shared byte-for-byte across arms, cells,
domains, and metrics. Each draw averages both folds. Cells are evaluated
separately; the three fixed initialization cells are averaged for a cohort;
the three cohort draws are averaged for the grand statistic.

If DEV opens, it uses seed `95042` to generate one canonical uint16 array with
shape `[50000,768]`, shared across all arms, cells, cohorts, domains, and
metrics. The 768 DEV episodes are resampled as whole clusters. Cohort and grand
statistics use the same draw indices; no root or fixed seed is resampled.

CAL and DEV each use seed `96042` for 20 fixed whole-episode derangements.
CAL derangements have shape `[20,3,2,256]` and never map an episode to itself
within a cohort/fold. DEV derangements have shape `[20,768]` and have no fixed
points. The donor row mapping preserves each episode's 12-root block.

For the candidate in every cell and every evaluation domain, the median
shuffled/real calibrated-BCE ratio must be at least `1.05`, aggregate raw-AUC
loss at least `0.05`, and every-action raw-AUC loss at least `0.03`. The
authoritative reload regenerates and byte-compares bootstrap and derangement
arrays before using them. Resampling calculations are block/chunked; no full
root-by-resample tensor is materialized.

## Fixed evaluation domains and support identity

For semantic output action `a`:

- all-action includes every `(root,a)`;
- factual includes only roots whose current applied action equals `a`;
- nonselected complement includes roots whose current applied action is not
  `a`.

Each root contributes one factual and four complement entries. For BCE and
Brier separately, authoritative evaluation verifies
`5*all = factual + 4*nonselected_complement` at every root to absolute
tolerance `1e-12`.

Each domain uses a matching prior frozen from that cell's FIT cohort. All and
complement priors use the corresponding output column; factual priors also
condition on matching factual action. Priors, raw logits, affine parameters,
probabilities, targets, metric values, bootstrap draws, and causal values must
all be finite.

## Absolute candidate gates

BAL-UPMIX+AA must pass every absolute gate below in every cell at both the
cross-fitted CAL endpoint and, if opened, the sealed DEV endpoint. BASE is the
paired control; it must have accepted cross-fit/final AA and finite predictions,
but BASE does not independently have to pass BAL's absolute candidate gates.

- calibrated all-action BCE is no worse than raw all-action BCE;
- all-action BCE is at most `0.98` of its FIT-prior BCE and Brier is at most
  `0.95` of its FIT-prior Brier;
- raw and calibrated all-action ranking each have aggregate AUC at least
  `0.65`, gain over the FIT all-action-prior AUC at least `0.10`, and every
  action AUC at least `0.60`;
- aggregate ECE and absolute calibration bias are at most `0.05` in all three
  domains;
- all-action per-action ECE is at most `0.075`, absolute bias at most `0.05`,
  PR-AUC at least prevalence plus `0.05`, and Brier skill versus the matching
  FIT prior is nonnegative;
- factual per-action targets have both classes, ECE is at most `0.075`, and
  absolute bias is at most `0.075`;
- complement per-action targets have both classes, ECE is at most `0.075`,
  absolute bias at most `0.05`, PR-AUC at least prevalence plus `0.05`, and
  Brier skill versus the matching FIT prior is nonnegative;
- BCE and Brier improvements over the matching FIT prior have strictly
  positive points and one-sided 97.5% whole-episode-bootstrap lower bounds in
  all three domains;
- every causal derangement gate in the prior section passes.

All strict inequalities remain strict. Missing class support, undefined
ranking, nonfinite output, mask-identity failure, or rejected AA is a failure.

## Direct BAL-UPMIX versus BASE gates

Improvement is always `BASE loss - BAL-UPMIX loss`, paired on the same roots,
episodes, cross-fit direction/final AA, and bootstrap indices. Every cell,
every three-initialization FIT cohort average, and the grand three-cohort
average must satisfy both the observed point and its one-sided 97.5% lower
bound:

| Domain | BCE threshold | Brier threshold |
|---|---:|---:|
| factual | > 0 | > 0 |
| all-action | > -0.010 | > -0.005 |
| nonselected complement | > -0.010 | > -0.005 |

At CAL, before any two-fold, cohort, or grand aggregation can pass, the point
estimate from each heldout fold A and B must independently exceed its matching
domain/metric threshold in every cell. A and B use independent cohort-episode
resamples; equal ordinals across F0/F1/F2 are never treated as paired episodes.

CAL passes only if all candidate absolute/support/causal gates, both arms'
cross-fit and final-AA acceptance gates, and every direct cell/cohort/grand
limb pass. Any CAL failure is a valid negative futility result and seals DEV.
DEV applies the same absolute, support, causal, and direct gate families to
the final-AA predictions.

## Latency and resource guard

DEV nomination additionally requires a fixed CPU batch-one pruned-hazard-head
latency probe over the first 256 canonical DEV features for all 18 heads. The
aggregate measured p99 must be at most `5.0 ms`. This is a head-only gate, not
a measurement or claim for capture, representation, orchestration, action
delivery, Minecraft, Pac-Man, or a full live control loop.

Local execution is exactly one foreground CPU-only process. CUDA is hidden and
PyTorch intra-op and inter-op thread counts are each one. Before the attempt
and before every next lifecycle phase, abort without opening that phase if any
of these is true:

- process working set is at least `2.0 GiB`;
- available physical RAM is below `4.0 GiB`;
- system committed memory is at least `85%` of its limit;
- free space on the artifact volume is below `5 GiB`.

The run must not start background workers, network services, screen capture,
HID injection, or a GUI/game loop. The CAL bootstrap is structurally capped at
about 153.6 MB (`50000*3*2*256*2` bytes), and DEV at about 76.8 MB
(`50000*768*2` bytes); draw calculations remain chunked. The run makes no
claim about swap or pagefile thrashing unless that quantity is explicitly
measured. A resource refusal before the create-only attempt receipt consumes no
namespace or data and may be tried again only after resources recover. Once the
attempt receipt exists, any resource abort is terminal and cannot be converted
into a retry on the same data.

## Decision, nonclaims, and scaling boundary

There is one candidate and no fallback:

1. if CAL fails, emit `fresh_CAL_futility_negative_no_DEV`, nominate nothing,
   and stop permanently;
2. if CAL passes but DEV or latency fails, emit a fresh DEV negative, nominate
   nothing, and stop permanently;
3. only if CAL, DEV, latency, evidence, provenance, causal, parent-integrity,
   and resource gates all pass, nominate exactly
   `NZ_masked485_to_pruned240_BAL-UPMIX_plus_standard_AA` for a later scaling
   qualification.

BASE can never nominate. PB21M emits no checkpoint, policy, controller, or
deployable model. A pass does not prove scalable improvement, monotonic gains
with compute, next-state learning, reward learning, general game sense,
cross-game transfer, end-to-end latency, Minecraft competence, Pac-Man
competence, or human-equivalent processing speed. It also does not authorize
DGX or full-model training.

A PB21M pass unlocks only the preparation of a new, preregistered, fresh-data
scaling qualification that tests the nominated recipe against paired BASE over
multiple independent seeds and increasing model/data/compute trajectories. The
scaling study must test monotonic quality, next-state/reward/hazard components,
causal dependence, calibration, full live-loop latency, stability, and bounded
resources before accelerator training can be considered. A PB21M failure does
not justify searching this DEV, changing thresholds, choosing BASE, or reviving
POOL on the same namespaces; a materially new hypothesis requires new ranges
and a new preregistration.
