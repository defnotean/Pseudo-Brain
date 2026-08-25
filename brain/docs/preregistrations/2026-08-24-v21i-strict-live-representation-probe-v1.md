# V2.1i strict-live matched representation/head probe implementation v2

Status: implementation-revision-2 preregistered development diagnostic. It is never a candidate,
qualification run, checkpoint producer, or publication path.

## Revision-2 correction and failed-v1 binding

Revision 1 failed before fitting with `O-warm does not exactly reproduce parent
TRAIN-FIT logits`. Revision 2 binds that immutable failure artifact at
`brain/runs/v21i-diagnostics/2026-08-24-seed42-v3-strict-live-probe-v1.json`,
SHA-256
`3ed153224ee1d038392e0f4e524c69582413b58051fac2dae073420efc748e57`,
and its registration at
`brain/runs/v21i-diagnostics/2026-08-24-seed42-v3-strict-live-probe-v1.registration.json`,
SHA-256
`5138da4ef4fe3f9bf9734e80d8a97eabfbd1e7ed7e0a514172b453af8fa7eba9`.

The failure cause is fixed as `cpu_gemm_batch_shape_rounding`, not topology,
weight-copy, or semantic-action-order drift. Parent logits were collected with
CPU batch size one, but the failed guard recomputed all 1,536 TRAIN-FIT roots in
one batch. An independent 257-root reproduction measured maximum absolute delta
`4.768371582e-7`, mean absolute delta `8.426e-8`, and 953 byte mismatches; the
same copied head evaluated at batch size one had zero mismatches. Semantic
action order was canonical in both paths.

Revision 2 keeps a no-tolerance, byte-exact guard by recomputing O-warm parent
logits at the same batch size one used to construct the tape. It additionally
requires every copied state/action/fusion/head tensor and its projected state
digest to be byte-identical to the parent. Large-batch closeness is not a
substitute and no numerical tolerance is introduced. Registration, success,
and failure records identify implementation revision 2, use revisioned v2 mode
names, and carry the failed-v1 binding and correction cause. The prior v1
receipt cannot authorize a v2 run; a fresh create-only v2 receipt is required.
The source-container filenames retain their historical `v1` suffix solely for
traceability; the authoritative identity is the revision-2 mode plus newly
registered file hashes, never the filename or the obsolete v1 receipt.

## Question and fixed interpretation

This probe separates three explanations for the failed seed-42 v3 hazard AUC:

1. the current 44,161-parameter hazard head was optimization- or
   initialization-underfit;
2. the frozen belief contains sufficient information, but the current head has
   insufficient extraction/action-interaction capacity;
3. the current tick's belief update compressed or made inaccessible information
   that remains present in its live updater tuple.

No threshold in this document was derived from a probe result. Ranking uses the
unchanged V2.1i requirements: aggregate DEV ROC-AUC at least `0.65`, gain over
the TRAIN-FIT action prior at least `0.10`, and every action ROC-AUC at least
`0.60`. Pairwise interpretation additionally requires a paired, episode-
clustered, two-sided 95% AUC-delta interval whose lower bound is above zero.

## Exact immutable upstream

- V3 result SHA-256:
  `3f91921b7ca63148addd3204cdb1e96b130f26240f49277a269ac8f079b13936`
- Raw uncalibrated parent SHA-256:
  `e5a4c1e92cdf5649679f344ce9a032397e69a0a5c93bb9d3a7bebc4bc1352081`
- Parent model-state SHA-256:
  `2619b5b0a180bdd55296d3471d89725f4346110e74ef093f84c3068a32e6576d`
- V3 production source bundle SHA-256:
  `681f3f45422316ec40c299b6c449a90a4790f544ea21311597416cfa7fbb3862`
- TRAIN-FIT manifest SHA-256:
  `835fa63a0388c23eff17713b55836626f42ddd361ec2829948359d2187f65179`
- DEV manifest SHA-256:
  `f2cd82e9a799107800744ee9350e5ebdd4bd795d454bf975280c252f7f78bd64`

Every binding is mandatory. The parent must record exactly 48 joint optimizer
steps and eight/512 isolated hazard-refinement passes/steps, identity hazard
calibration, seed 42, CONFIG_B_PREDICTIVE, and the exact V2.1i configuration.

## Namespace and publication boundary

Only the exact existing TRAIN-FIT and DEV manifests may be materialized.
TRAIN-CAL, CPU-QUAL, and TEST are not constructed. TRAIN-FIT is the only fitting
population. DEV is evaluated exactly twice in conceptual schedule position:
the immutable parent endpoint and the four fixed final arm endpoints. It cannot
control stopping, milestones, hyperparameters, arm selection, or thresholds.

The script emits one create-only JSON diagnostic and no model checkpoint. Every
success or failure payload records `diagnostic_not_candidate`,
`qualification_claimed=false`, `candidate_publication_allowed=false`, and
`checkpoint_emitted=false`. No code path invokes a qualification gate or a
candidate publisher.

Before any run, a separate `--register-only` invocation must create an
immutable registration receipt. It binds the final script, this preregistration,
the focused test file, their ordered composite digest, the exact v3 upstream and
manifests, full fixed schedule, strict sensory scope, and namespace boundary.
The run requires that receipt, recomputes every registered hash before loading
the parent or constructing a source, and records the receipt SHA-256. Any source,
scope, schedule, or receipt drift is a preflight failure. Registration emits no
model and does not open any dataset. The real receipt is not created as part of
implementation or testing.

The implementation owns a local two-entry partition factory containing only
TRAIN-FIT and DEV. It must not call the three-way V2.1i development factory,
even merely to select two returned entries; a spy test pins that no TRAIN-CAL,
CPU-QUAL, or TEST source construction occurs.

## Strict-live sensory replay

Every episode is replayed independently (batch size one), starts with an
internally owned semantic prior action `idle`, and applies the policy's fixed
model initialization seed `2126202608` around that episode's first model tick.
Each tick passes exactly these model arguments:

- current RGB, resized by the existing 32x32 model sensory path;
- the internally owned prior applied action;
- `actual_reward=None`;
- `actual_hazard=None`.

The code does not read `observation.previous_control`. After the model consumes
the current observation, the factual balanced-policy applied action is gathered
from the explicit semantic all-action table and installed as pending prediction,
mirroring `OutcomeAwareCoreV2MazePolicy.act_rgb` after its chosen action. Only
then does it become the next tick's private prior action.

An executable multi-tick equivalence test runs this reference replay beside
`OutcomeAwareCoreV2MazePolicy.act_rgb`, forcing the same logged semantic action
only at `_select_semantic_action`. It pins first-tick seeded initialization and
later recurrent ticks by comparing the hazard-relevant belief, updater tuple,
raw semantic hazard logits, and post-selection pending prediction. Thoughts and
decision outputs are deliberately outside this equivalence claim.

Teacher actions, current/previous event targets, randomized intervention
assignment, environment/episode seeds, root IDs, and episode IDs are forbidden
as model features. Counterfactual event targets are read only after feature
formation and serve only as supervised labels. Root and episode identities are
retained only as hashed provenance and bootstrap clusters.

## Frozen feature tapes

The exact frozen parent is never mutated. Each scored root records:

- `belief`: the 120-D post-update belief;
- `updater`: the 366-D live-available pre-compression tuple
  `[old belief(120), current RGB encoder latent(120), live-only cognitive
  prediction error(120), internally owned prior-action one-hot(5),
  has-pending(1)]`;
- the exhaustive five-action hazard target row;
- the parent's raw semantic hazard logits;
- episode/root provenance digests;
- separate diagnostic-only strictly prior intervention-history counts.

CONFIG_B memory and session inputs are fixed zeros and omitted. The capacity
belief arm receives `[new belief(120), zeros(246)]`, so its trainable topology
and parameter count exactly match the capacity updater arm. Candidate action is
always a separate explicit semantic action ID/embedding. Action-table column
order cannot become a feature.

Because `updater` includes `old belief`, C-updater can diagnose only information
available at the current tick before the new-belief update. It cannot reconstruct
longer history already discarded before `old belief` was formed. Consequently,
C-updater failure cannot distinguish upstream sensory compression from
longer-history information loss.

## Fixed arms and optimization

All fitting uses all 1,536 TRAIN-FIT roots and all five branches, unweighted
binary cross entropy with logits, batch 24, fresh AdamW, learning rate `1e-3`,
weight decay `1e-4`, gradient clip `1`, and seed `30042`. Every pass is one
complete deterministic root permutation without replacement. There is no early
stopping or model selection.

- **O-warm:** exact parent hazard topology and weights, adapted with a fresh
  AdamW restart for 24 passes / 1,536 steps on the strict-live belief tape. The
  parent contributes eight historical refinement passes, so `32` is only a
  nominal cross-signature warm-start count. Those original eight passes used
  v3 recurrence with prior actual reward/hazard and are not represented as
  eight strict-live-distribution passes.
- **O-scratch:** exact current 44,161-parameter topology, fixed reinitialization,
  32 passes / 2,048 steps.
- **C-belief:** fixed 262,801-parameter capacity topology on padded new belief,
  fixed reinitialization, 32 passes / 2,048 steps.
- **C-updater:** the byte-identically initialized same capacity topology on the
  366-D updater tuple, 32 passes / 2,048 steps.

The capacity topology is
`Linear(366,240)-ReLU-LayerNorm`, semantic `Embedding(5,240)-LayerNorm`,
concatenation, `Linear(480,240)-ReLU-Linear(240,240)-ReLU-Linear(240,1)`.
Layer normalization is functional and adds no parameters.

All pass permutation digests, loss/gradient telemetry, arm initial/final state
digests, feature/target/logit digests, exact ordered root/episode sequences with
multiplicity, parent before/after state digests, source bundle, manifests, and
namespace state are recorded. The O-warm record must retain the cross-signature
caveat and fresh-optimizer restart. CUDA is hidden before torch import;
execution is CPU-only and single-threaded.

## Fixed metrics and diagnosis

Report raw, uncalibrated TRAIN-FIT and DEV aggregate/per-action BCE, Brier,
ROC-AUC, PR-AUC, prevalence, and ECE for the parent, TRAIN-FIT action prior, and
four final arms. Calibration is never fit or installed.

Paired DEV AUC-delta intervals use 10,000 deterministic episode-clustered
resamples with seed `31042` for:

- O-warm minus parent;
- O-scratch minus parent;
- C-belief minus O-warm and O-scratch;
- C-updater minus C-belief.

Interpretation is fixed:

- optimization/initialization underfit is supported if O-warm or O-scratch
  passes all frozen ranking gates and its paired lower bound over parent is
  above zero;
- capacity limitation is supported only if both O arms fail, C-belief passes,
  and C-belief's paired lower bound is above both O arms;
- current-tick belief-update compression/accessibility is supported only if
  C-belief fails,
  C-updater passes, and C-updater's paired lower bound over C-belief is above
  zero;
- C-updater success localizes an advantage to the same-tick updater tuple but
  does not establish that longer history was retained; C-updater failure remains
  inconclusive across upstream sensory compression and longer-history loss;
- every other result is inconclusive. It does not authorize unfreezing the
  shared core.

## Intervention-history stratification

The same immutable manifest hash, episode seed, and tick reproduce the dataset's
randomized assignment digest. For root at tick `t`, record separately:

- randomized intervention assignments over ticks strictly `< t`;
- actual applied-versus-teacher disagreements over ticks strictly `< t`.

The current tick is always excluded. Tests pin this prefix-count semantics and
the dataset's exact assignment digest. Exact-count strata report aggregate
metrics only when both classes are present. Counts never enter a head, a gate,
model selection, or the diagnosis. If reconstruction or alignment fails, the
run fails rather than silently omitting or approximating the strata.
