# PB21L-NZ-UPMIX-CONSUMED-v1 preregistration

Date frozen: 2026-08-25

Canonical basename: `2026-08-25-pb21l-nz-upmix-consumed-v1`

Implementation revision: 2 (pre-registration evidence hardening; scientific
schedule, thresholds, and decision precedence unchanged).

Classification: `exploratory_consumed_data_nonqualifying`, permanently. This
study reuses data already evaluated by PB21K. It cannot nominate a recipe,
publish a checkpoint or candidate, claim qualification, or authorize a retry.
At most it can issue one fixed architecture license for one later,
preregistered, fresh-data trial.

## Question

PB21K showed that the detached NZ representation ranks hazards but did not
resolve factual deployment calibration. PB21L asks a narrower feasibility
question: does adding a factual loss term while fitting the scorer improve the
factual proper scores without materially degrading either the complete
all-action table or the four nonselected counterfactual branches?

No new environment seed is opened. PB21L consumes the exact PB21K F0/F2,
C0/C2, four fixed scorer cells, shared DEV, bootstrap matrix, and derangements.
The result is exploratory even if every gate passes.

## Frozen PB21K parent

- Registration SHA-256:
  `f6bc3e25412ea6c3058bc38ace7622e30fdec72c6732388cb7a7d7dfa38f7058`
- Attempt SHA-256:
  `bd2ae53311733900a421b9928f2db73f32e5aa82d14a523f27f7bcf367b2a094`
- Evidence SHA-256:
  `9d57bf4a6bd60121e8f11e2b9405c1a6d64a9346381503054d6ad9cd4ece385c`
- Result SHA-256:
  `083ac9c44d3327f4f806b1e4842ea3991ec54e5bab65992e7c5446ad09ca02d4`
- Source-bundle SHA-256:
  `b13b764c2ff55dd15649945fec7c05211a3c59fc04c7ecfb667d1b711590e03a`
- Evidence array-manifest SHA-256:
  `256f25b832aad90457fd8528979a6a60a24992d265dc864771b8a1f569c29be5`
- Evidence key-set SHA-256:
  `b11966d1003e1082586adbedbad46c69e6ce2af11964dcad24eb305ffbdc562f`
- Bootstrap-array SHA-256:
  `80c2c95e3273d510ec6f36fea873c78fcd3f5d059e9985132b1bbbdd973dc140`
- Episode-derangement SHA-256:
  `c142d8b756ff30f58417faf2f29546e59b53f792973f4dddcb9833ca4f790672`

PB21K must retain classification `diagnostic_not_candidate`, diagnosis
`NZ_representation_supported_dual_calibration_comparison_unresolved`, no
selected recipe, and validated authoritative evidence. Any byte or envelope
drift fails before data loading.

## Consumed cells and ordering

Only these four cells are permitted, in this order:

1. F0 / init 51042, calibrated on C0;
2. F0 / init 53042, calibrated on C0;
3. F2 / init 51042, calibrated on C2;
4. F2 / init 53042, calibrated on C2.

The PB21K cell indices are `[0, 2, 6, 8]`; FIT/CAL indices are `[0, 2]`.
Authoritative reload requires `cell_fit_index == [0,0,1,1]` and
`cell_init_seed == [51042,53042,51042,53042]` exactly; shape-valid axis
substitution invalidates the attempt.
DEV remains the exact 768-episode, 9,216-root PB21K endpoint. All live tapes
must match the frozen target, factual-action, and ordered-root arrays. The
strict live signature remains current RGB plus internally owned prior action,
with actual reward/hazard absent and pending installed only after each tick.

## Fixed paired scorers

All three arms use PB21K's exact NZ masked-superset feature table, 87,961-
parameter fan-in-485 head, initialization, relative permutations, optimizer,
and pruning map. Current factual action is a loss selector only and never a
feature. The three arms are initialized byte-identically within every cell.

- **BASE:** `J = mean_{root,action} BCE`.
- **POOL-UPMIX:**
  `J = .5 mean_{root,action} BCE + .5 mean_root BCE_factual`.
- **BAL-UPMIX:**
  `J = .5 mean_action mean_root BCE_action`
  `+ .5 mean_action mean_{root:factual=action} BCE_factual`.

For BAL-UPMIX, full-FIT factual counts `n_a` are frozen before fitting. A root
whose current factual action is `a` receives factual-term minibatch weight
`N/(5*n_a)`, where `N=3072`. The minibatch mean is therefore the unbiased
estimator of the exact balanced factual objective under the unchanged
permutations. Counts and per-root weights are immutable evidence.
Evidence stores both the theoretical float64 values and their exact float32
casts consumed by training. Reload recomputes both from the frozen factual
actions, so scientific interpretation follows the implemented objective.

Each head gets 32 complete 3,072-root passes, batch size 24, 4,096 optimizer
steps, fresh AdamW at learning rate 1e-3 and weight decay 1e-4, and gradient
clip 1. Permutation seed is 60042. There is no stopping, retry, selection, or
adaptation. All twelve heads finish before CAL; all twelve AA calibrators
finish before the one consumed DEV endpoint is reconstructed.

Every fitted head is pruned to PB21K NZ columns `[N(120), Z(120)]`, 240 input
features and 58,561 parameters. Tensor copying is byte exact. Padded/pruned
FIT batch-one logit transfer must remain within 1e-5.

## BASE invalidation control

BASE is not merely a conceptual control. Each cell must exactly reproduce the
frozen PB21K NZ evidence:

- FIT target/action/root identities and NZ features;
- initialization and final padded state hashes;
- all 32 pass BCE/gradient/permutation records;
- pruned state hash, map, and FIT transfer digests/maximum difference;
- CAL raw logit table;
- standard AA parameters, acceptance, and calibration provenance;
- DEV raw logits/probabilities and AA logits/probabilities.

The immutable PB21L evidence stores these identities. Reload checks them
against PB21K and deterministically refits every AA calibrator from the
published CAL arrays. Any mismatch invalidates the whole attempt before
scientific interpretation.

For each FIT cohort, the exact recomputed NZ feature-table digest is attached
to prepublication source evidence and copied into authoritative evidence.
Reload receives an immutable snapshot of the prepublication FIT/CAL ordered
group sequences, recomputes their ordered digests, and requires both stored
IDs and digests to match that snapshot byte-for-byte.

Registration binds the exact allowed NPZ key set, key-set digest, and every
array dtype and shape. Evidence includes ordered FIT/CAL group IDs so reload
reconstructs and verifies the calibration-group and upstream-fit-group
provenance digests rather than accepting digest strings. It also stores all
scorer state hashes and pass telemetry, pruning columns/state/transfer, CAL
raw-logit digests, AA lineage identifiers, and the shared inference arrays.
Pass telemetry includes the explicit 1-through-32 pass numbers and every
permutation hash; pruning evidence includes a canonical digest of the complete
mapping record as well as its state and FIT-transfer fields.
Reload also recomputes the evidence array manifest and verifies the published
file byte length, allowed-key-set digest, per-array manifest, and manifest
digest rather than trusting receipt fields.

## Standard AA calibration

Every frozen scorer uses only PB21K's standard all-action per-action affine
calibrator: float64 CPU, L2 1e-6, minimum scale 1e-4, at least 20 examples and
two examples/class/action, at most 100 iterations, tolerance 1e-10. There is
no MIX calibration and no selection among calibrators.

The registered AA provenance is authoritative: source namespace
`maze_chase.pb21k.train-only.v1`, split `TRAIN`, partition `TRAIN-CAL`, the
frozen PB21K partition algorithm and CAL manifest for each cell, and the
registered PB21K/PB21L source-bundle digest for each scorer. Evidence stores
all these fields, and reload compares them to the registered/frozen contract
before reconstructing lineage and deterministically refitting AA. Evidence
cannot supply its own provenance expectations.

## Fixed domains

For semantic output action `a`:

- all-action contains every `(root,a)`;
- factual contains only roots whose current applied action equals `a`;
- nonselected complement contains roots whose current applied action is not
  `a`.

Every root has one factual and four complement entries. The evidence must
satisfy, to absolute tolerance 1e-12 for BCE and Brier,
`5*all = factual + 4*complement` at every root.

Each domain uses a domain-matched prior fit on the cell's FIT cohort. All and
complement action priors use the corresponding semantic output column;
factual priors additionally condition on the matching factual action.

## Absolute gates

Every candidate cell must pass every gate below after standard AA:

- AA accepted and all values finite;
- calibrated all-action BCE no worse than raw;
- all-action BCE no more than .98 of its FIT prior and Brier no more than .95;
- raw and calibrated all-action ranking each retain aggregate AUC >=.65, gain
  over the FIT all-action prior >=.10, and every action AUC >=.60;
- aggregate ECE and absolute bias <=.05 in all three domains;
- all-action per-action gates retain PB21K ECE <=.075, absolute bias <=.05,
  PR-AUC at least prevalence+.05, and nonnegative Brier skill;
- factual per-action gates require both classes, ECE <=.075, and absolute
  bias <=.075; no new factual per-action PR/proper-score gate is introduced;
- complement per-action gates require both classes, ECE <=.075, absolute bias
  <=.05, PR-AUC at least prevalence+.05, and nonnegative Brier skill;
- aggregate BCE and Brier prior improvements have positive points and
  one-sided 97.5% episode-bootstrap lower bounds in all three domains;
- the exact PB21K whole-episode derangements pass separately in all,
  factual, and complement: median shuffled/real calibrated BCE >=1.05,
  aggregate raw AUC loss >=.05, and every-action raw AUC loss >=.03.

The complement aggregate and per-action gates are absolute, not merely
noninferiority checks.

## Paired BASE comparisons

The direct improvement direction is always `BASE loss - candidate loss`.
For each candidate and domain, losses are computed within each cell and
episode; the four fixed-cell episode deltas are averaged before applying the
exact PB21K 50,000x768 bootstrap matrix. Fixed cells are never resampled.

Every cell point and the grand one-sided 97.5% lower bound must be:

- factual BCE >0;
- factual Brier >0;
- all-action BCE >-0.010;
- all-action Brier >-0.005;
- complement BCE >-0.010;
- complement Brier >-0.005.

Strict inequalities apply. The exact shared bootstrap and derangements are
copied byte-for-byte into PB21L evidence.

## Decision and lifecycle

An arm passes only if all four of its absolute cell gates and all six paired
direct limbs pass. Precedence is fixed:

1. issue `BAL-UPMIX` fresh-data license if BAL passes;
2. otherwise issue `POOL-UPMIX` license only if POOL passes;
3. otherwise issue no license.

Observed margin size cannot change this order. The license is not a
nomination and cannot reuse this DEV. A future run requires a new,
preregistered fresh namespace and remains subject to full qualification.

Registration, attempt, evidence, success, and failure records are deterministic
and create-only. Registration fails if attempt/evidence/result already exists;
run fails if any exists. Once an attempt exists, retry is forever false. The
runner is CPU-only, hides CUDA, uses one intra/inter-op thread, never emits a
checkpoint, and never accesses CPU-QUAL, PLAY-QUAL, or TEST.
