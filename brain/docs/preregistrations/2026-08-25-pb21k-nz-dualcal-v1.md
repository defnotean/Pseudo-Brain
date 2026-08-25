# PB21K-NZ-DUALCAL-v1 preregistration

Date frozen: 2026-08-25

Canonical run basename: `2026-08-25-pb21k-nz-dualcal-v1`

Classification: development diagnostic only. It cannot publish a checkpoint,
candidate, qualification, or production claim. A pass nominates a complete
training/pruning/calibration recipe for a later fresh from-scratch production
runner.

## Scientific question

PB21J established the raw NZ representation path but did not nominate it. Its
only failing NZ deployability endpoint was one fixed replica's factual Brier
lower bound (`F1/init-42042`, -0.00012880979606848126). PB21K asks one narrow,
prospective question: can a calibration objective which gives equal weight to
the all-action and current-factual domains repair that deployability failure
without materially degrading the sealed all-action affine-calibration
control?

This is calibration-only. PB21K must not change the scorer loss, representation,
head topology, pruning, live replay, feature ownership, or causal-shuffle
contract. It does not reuse PB21J DEV data.

## Immutable scientific parent

PB21K binds the exact PB21J registration, attempt, result, source bundle, and
shared calibration implementation:

- PB21J registration SHA-256:
  `8ecebd1b6317f94dbab3d0e2bf82af4cdd44952ebe9e01e478a74aeb8648c25b`
- PB21J attempt SHA-256:
  `cc17c4d507ef3d3fd84e6e6cc01e7757bdf7da3a157304d6e691de47859c43fe`
- PB21J result SHA-256:
  `21fafd0b7107ce2f3edf7cf4c8352caee8d0b519395a15e737249edba7e41182`
- PB21J source-bundle SHA-256:
  `6207c1a93cd732694e2a0508b117636d1fad2632bf51581e91e1306015af9482`
- `hazard_calibration.py` SHA-256:
  `227ba0fddb6ca9fe96b673334865f8a66f0c398baa8f163d87777dfb2256f3b0`

The parent must still say
`representation_supported_deployability_unresolved`, select no arm, and show
that NZ representation inference and calibration acceptance passed while
deployability excluding calibration did not. Any parent or source drift fails
closed.

## Fresh namespaces

All partitions retain the PB21J environment: old maze-chase environment,
sequence length 16, four burn-in ticks, five all-action counterfactual hazard
targets, balanced intervention policy at 0.5, batch-one strict-live replay,
current RGB, and owned prior action. Actual reward and actual hazard are absent.

| Label | Role | Split | Half-open seed range | Episodes | Post-burn roots |
|---|---|---|---:|---:|---:|
| F0 | TRAIN-FIT | train | [134217728, 134217984) | 256 | 3072 |
| C0 | TRAIN-CAL | train | [134217984, 134218240) | 256 | 3072 |
| F1 | TRAIN-FIT | train | [134218240, 134218496) | 256 | 3072 |
| C1 | TRAIN-CAL | train | [134218496, 134218752) | 256 | 3072 |
| F2 | TRAIN-FIT | train | [134218752, 134219008) | 256 | 3072 |
| C2 | TRAIN-CAL | train | [134219008, 134219264) | 256 | 3072 |
| DEV | shared endpoint | validation | [150994944, 150995712) | 768 | 9216 |

Registration performs the central split-prefixed occupied-range audit over
current V21I, CPU-QUAL (through 33,554,624), wallclock bands, PLAY-QUAL,
PB21J, PB21K, and registered TEST bands. Dataset manifests are config-only at
registration; no labels or source objects may be materialized then. Current
V21I/PB21J DEV, CPU-QUAL, PLAY-QUAL, and TEST are forbidden.

## Live feature and scorer contract

The 485-D detached superset is identical to PB21J:

- N `[0:120]`: post-update belief;
- B `[120:240]`: old belief;
- Z `[240:360]`: current RGB encoder latent;
- E `[360:480]`: live cognitive prediction error;
- A `[480:485]`: owned prior-action one-hot;
- pending flag: structurally omitted only after proving its source column is
  exact positive one post burn-in.

The only arms are N, Z, BZ, NZ, and U0. NBZ is omitted and rejected. Masked
columns are canonical float32 positive zero.

Every arm is trained with the same 87,961-parameter masked-485 head:

1. `Linear(485,120)`, ReLU, parameterless layer norm with epsilon 1e-5;
2. `Embedding(5,120)`, parameterless layer norm with epsilon 1e-5;
3. concatenate state and action;
4. `Linear(240,120)`, ReLU;
5. `Linear(120,1)`.

For each F cohort and initialization seed 51042, 52042, and 53042, all five
arms start byte-identically. All heads use the same complete relative
permutations from seed 60042. AdamW uses learning rate 1e-3, weight decay
1e-4, batch 24, gradient clipping at 1, and unweighted all-action BCE for 32
complete passes. This is 4096 optimizer steps/head, 45 heads, and 184,320
total steps. There is no early stopping, selection, retry, or adaptation.

After FIT and before any CAL or DEV prediction, every scorer is pruned with
the exact PB21J tensor-copy map. NZ deploys 240 input columns and 58,561
parameters. The padded/pruned logit audit covers every FIT row at batch size
one and requires maximum absolute difference at most 1e-5. This proves only
inference transfer. A later production runner must reproduce fan-in-485
initialization and optimization, then prune; compact-from-step-zero training
is not claimed equivalent.

## Dual calibrators on one frozen NZ scorer

Only each of the nine pruned NZ scorers is calibrated. Both calibrators consume
the byte-identical matched C-cohort raw NZ logit table.

### AA sealed control

AA calls the existing `fit_train_only_per_action_affine` implementation
directly, with float64 CPU fitting, L2 1e-6, minimum scale 1e-4, minimum 20
all-action examples/action, minimum two examples/class, maximum 100
iterations, and tolerance 1e-10. AA is reported but can never nominate.

### MIX candidate

For semantic action `a`, let `n` be all CAL roots, `F_a` the roots whose
current applied factual action is `a`, and `n_a = |F_a|`. The exact objective
is

`0.5 mean_i BCE(y_i,a, s_a l_i,a + b_a)`

`+ 0.5 mean_{i in F_a} BCE(y_i,a, s_a l_i,a + b_a)`

`+ 0.5e-6 ((s_a - 1)^2 + b_a^2)`, with `s_a >= 1e-4`.

Equivalently, the data weights are
`0.5/n + 0.5 I[i in F_a]/n_a` and sum exactly one within action. The factual
label is the current applied action recorded after inference; updater A is the
owned prior action and may not substitute.

The local frozen MIX type truthfully reports separate all-action, factual, and
mixed-objective fields. It is not serialized as the shared AA type. Fitting is
deterministic damped Newton with exact analytic gradient/Hessian, positive
scale projection, the projected-gradient KKT condition at the scale bound,
and a bias-only Newton direction when the bound KKT condition is satisfied. A
line search permits at most 50 halving attempts per Newton iteration. The L2
penalty appears exactly once. Nonfinite values, non-positive-definite Hessian,
line-search exhaustion, or maximum-iteration exhaustion fail closed.

Each action requires at least 20 factual examples and at least two positive and
two negative factual examples. Every action must improve the identity mixed
objective by more than 1e-10 and strictly improve all-action BCE, all-action
Brier, factual BCE, and factual Brier. The same four proper scores must improve
in aggregate. All scales/biases must be finite and scales positive. If any
check fails, the whole MIX calibrator applies exact identity; proposed
parameters and reports remain diagnostic evidence.

## DEV ordering and authoritative evidence

All 45 scorer fits, all pruning audits, and all 18 calibrator fits finish before
the sole fresh DEV source is constructed. DEV prediction uses the literal
pruned heads with batch size one.

The create-only attempt receipt is published after registration/parent
validation but before any source object. An existing attempt, evidence, or
result forbids retry. After DEV predictions, PB21K atomically publishes a
deterministic create-only compressed `.evidence.npz`, reloads it with
`allow_pickle=False`, validates it, deterministically refits AA and MIX from
its CAL arrays, and only then computes numerical DEV gates and confidence
bounds. All numerical DEV gates and decisions are therefore derived from the
published evidence, not a parallel in-memory table; live training, pruning,
and calibration reports remain diagnostic-only provenance.

Evidence arrays use frozen axes C=9 cells, F=3 cohort pairs, A=5 actions,
Tfit=Tcal=3072 roots, D=9216 DEV roots, G=768 clusters, B=50,000 resamples,
and Q=20 derangements. The NPZ contains:

- schema/action/cell/arm/calibrator IDs plus explicit cell-to-FIT/CAL indices
  and initialization seeds;
- FIT targets/actions and ordered root IDs;
- CAL targets/actions, ordered root IDs, and each cell's shared raw NZ logits;
- AA/MIX applied scales/biases and acceptance, plus rejected-or-accepted MIX
  proposed scales/biases and per-action acceptance;
- DEV targets/actions, ordered root IDs, cluster ordinals, and cluster IDs;
- raw DEV logits and probabilities for all five arms and nine cells;
- AA/MIX NZ calibrated DEV logits and probabilities;
- the shared int32 bootstrap matrix and 20 int32 episode derangements.

All arrays are canonical C-contiguous, little-endian where endian-bearing, and
non-object. ZIP entries are sorted with fixed metadata. The result binds file
SHA-256, byte length, the exact key-set digest, a per-array name/dtype/shape/
SHA-256 manifest, and a digest of that complete sorted manifest. Reload must
verify exact keys, dtypes, shapes, action axes, cell maps, binary targets,
semantic actions, 768x12 cluster order, raw/calibrated numerical consistency,
deterministic AA/MIX refits, and the root-derangement hash reconstructed from
the episode mapping. Each FIT, CAL, and DEV target/action array and ordered
root-ID sequence must match its independently built source-evidence manifest;
each CAL NZ raw-logit table must match its pre-publication calibration digest.
Expanded evidence cluster IDs must match the ordered DEV episode digest.

## Frozen gates

All numerical values must be finite. Alpha is one-sided 0.025. The one shared
50,000x768 int32 episode bootstrap matrix uses seed 82042. The 20 joint
cross-episode derangements use seed 83042. Fixed scorer replicas are averaged,
not resampled.

### Raw representation and ranking

Every one of nine cells must pass raw ranking for Z, BZ, NZ, and U0:

- aggregate AUC at least 0.65;
- gain over that FIT cohort's action-prior DEV AUC at least 0.10;
- every action AUC at least 0.60.

Raw pooled-AUC comparisons use the conditional fixed-grid episode-cluster
jackknife-pseudovalue bootstrap. Every cell point and the grand lower bound
must satisfy NZ-N > 0 and NZ-U0 > -0.025. Ranking uses unsquashed logits.

### Unchanged PB21J NZ deployability gates, applied to MIX

Every MIX cell must satisfy all PB21J NZ gates:

- MIX CAL accepted;
- calibrated DEV BCE no worse than raw;
- all-action BCE no more than 0.98 of FIT-prior BCE and Brier no more than
  0.95 of FIT-prior Brier;
- aggregate all-action ECE at most 0.05;
- every action ECE at most 0.075, absolute bias at most 0.05, PR-AUC at least
  prevalence+0.05, and nonnegative Brier skill versus its FIT prior;
- calibrated aggregate/per-action ranking gates pass on calibrated logits;
- FIT and DEV factual targets have both classes in every action;
- factual BCE and Brier strictly beat the FIT factual action prior;
- all-action and factual BCE/Brier improvements over FIT priors have episode-
  cluster one-sided 97.5% lower bounds above zero;
- joint whole-row derangements have median shuffled/real calibrated BCE at
  least 1.05, aggregate raw AUC loss at least 0.05, and every-action raw AUC
  loss at least 0.03.

PB21K adds factual aggregate absolute bias and ECE at most 0.05, plus factual
per-action absolute bias and ECE at most 0.075.

### Direct AA versus MIX tests

Improvement is always `AA loss - MIX loss`, paired on the same DEV episode.
Every cell point and the grand episode-cluster lower bound must exceed:

- factual BCE: 0;
- factual Brier: 0;
- all-action BCE: -0.010;
- all-action Brier: -0.005.

The noninferiority margins were frozen from PB21J before opening PB21K data.
0.010 is about 2.3% of PB21J mean NZ AA all-action BCE (~0.43); 0.005 is about
3.7% of mean Brier (~0.135). Both are materially tighter than the prior-ratio
gates.

## Decision

NZ+MIX is eligible only if every raw ranking gate, every MIX acceptance and
deployability gate, both raw representation limbs, and all four direct
AA-minus-MIX limbs pass. AA never nominates, and there is no alternate arm or
calibrator selection. A failure produces no nomination and permits no same-DEV
retry. A pass nominates
`masked_superset_train_pruned_deployment_v1_plus_MIX_calibration` for a later
fresh runner, still with no checkpoint or candidate emitted here.
