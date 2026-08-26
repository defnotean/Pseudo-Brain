# PB21M-POSTHOC-MECHANISM-AUDIT-v1 preregistration

Date frozen: 2026-08-25

Canonical basename: `2026-08-25-pb21m-posthoc-mechanism-audit-v1`

Classification: `post_hoc_consumed_data_nonqualifying`, permanently. Both arms
consume only data already consumed by the sealed PB21M CAL stage. This audit
cannot nominate a qualified model, publish a checkpoint, open DEV, CPU-QUAL,
PLAY-QUAL, or TEST, authorize a same-data retry, or support any scaling,
live-play, or human-speed claim. Its only output is a mechanism read of the
frozen PB21M terminal negative plus a hard cap of **one** single change that
may be carried into a later, separately preregistered fresh PB21N trial.

## Provenance and lineage

- Sole immutable parent: PB21M `2026-08-25-pb21m-fresh-bal-cal-first-v1`,
  classification `fresh_CAL_futility_negative_no_DEV`, retry forbidden.
  Bound file digests:
  - registration `f2ca97558697e4597d57578261a82f13621808758cfd2ec5e1b8542ba516dee6`
  - attempt `e3322c79cd979d6b3b486680360c1eeafb961ad528a194d4706e68d2434d8758`
  - CAL evidence NPZ `5bdf26f0988a1bd79faa07b650eede2fadd08c37616a97b5b21ee739069ffe02`
  - CAL decision `20bc265e17c02e25a78cb76d31ceb281e253b44cfb1478db88dd64bc912706df`
  - result `0833b6c848003f9639c7e69250724639948b6cda334f992b0ff32fb0abd4d18f`
- Arm B additionally binds the exact V2.1i seed-42 uncalibrated parent
  checkpoint (`2026-08-24-seed42-v3.json.uncalibrated.pt`) through
  `strict.load_exact_parent`, and replays the deterministic PB21M maze-chase
  partitions on CPU only. Replay is verified byte-identical against the sealed
  PB21M arrays before any stratum statistic is computed.
- No trainer, no optimizer, no scorer fitting. Only deterministic closed-form /
  Newton calibration fits on already-consumed logits, plus arithmetic on
  already-consumed targets.

## Data layout (from sealed PB21M evidence, verified at load time)

- Scorers: `BASE`, `BAL-UPMIX` (2). Cells: `F{0,1,2}/init-{91042,92042,93042}`
  (9), where the F-index is the FIT cohort that produced the head.
- CAL: three cohorts x 6,144 rows x 5 actions = 18,432 rows. Each cohort is
  the concatenation of two 3,072-row CAL folds (canonical order CjA then CjB;
  row fold index in `oof_row_eval_fold_index`, 0 = CjA rows 0..3071,
  1 = CjB rows 3072..6143).
- `cal_raw_logits[scorer, cell, fold, root, action]` are the pruned-head raw
  hazard logits. `cal_targets` / `cal_actions` are per-cohort hazard targets
  and current applied (factual) actions.
- Sealed OOF identity (used as a load-time self-check): entry `j` of
  `crossfit_*` evaluates fold `crossfit_eval_fold_index[j]`; the OOF table must
  satisfy `oof_calibrated_logits[sc,cell,rows of fold f] ==
  raw[sc,cell,f] * crossfit_scales[sc,cell,j] + crossfit_biases[sc,cell,j]`
  exactly. `final_*` are pooled CAL calibrators and must NOT equal either
  cross-fit entry.

## SOLVER-SHAPING CONTROL (must pass before either arm is scored)

Re-derive the sealed PB21M AA calibrators from the NPZ arrays with the
audit's own Newton implementation:

1. AA cross-fit refit: for every scorer, cell, fold pair, fit
   per-action affine `(s_a, b_a)` on the other fold's rows with the frozen
   objective `mean_i BCE + 1e-6*((s-1)^2 + b^2)`, `s >= 1e-4`, float64,
   deterministic damped Newton (max 100 iterations, tolerance 1e-10,
   projected-gradient KKT at the scale bound, bias-only direction when the
   bound KKT condition holds, halving line search max 50, fail closed).
   All 36 refits must agree with `crossfit_scales`/`crossfit_biases` to
   within 1e-9 parameter difference AND the OOF reconstruction must be
   exact as above.
2. PB21K cross-validation: refit the v21k MIX calibrator (weight 0.5) from
   the sealed `brain/runs/v21k-diagnostics/2026-08-25-pb21k-nz-dualcal-v1.evidence.npz`
   CAL arrays and compare against its sealed `mix_proposed_scales` /
   `mix_proposed_biases` (1e-9) and `mix_per_action_accepted` (exact).
   This proves the generalized weighted solver reproduces a previously
  sealed MIX fit.

If either control fails, the audit halts, publishes a `solver_control_failed`
envelope, and nominates nothing.

## ARM A — fixed MIX35 cross-fit calibrator on sealed raw CAL logits

Frozen definition (the name MIX35 is fixed here; no weight sweep is allowed):

For semantic action `a`, with `n` = all rows of the fitting fold, `F_a` = rows
whose current applied factual action is `a`, `n_a = |F_a|`, and focal weight
`w = 0.35`:

`J_a = (1 - w) * mean_i BCE(y[i,a], s_a*l[i,a] + b_a)`
`     + w * mean_{i in F_a} BCE(y[i,a], s_a*l[i,a] + b_a)`
`     + 0.5e-6 * ((s_a - 1)^2 + b_a^2),  s_a >= 1e-4`

Equivalent data weights: `((1-w)/n) + w * I[i in F_a]/n_a`, summing exactly
one. `w = 0.35` is the single frozen focal weight: it is the
factual-emphasis point chosen to sit between the sealed AA implicit factual
weight 0.0 and the PB21L/PB21M focal 0.5 that already failed at 0.5 upstream
scoring; it is NOT searched. Context variants `w = 0.5` and `w = 1.0` are
computed in the same pass purely to bracket the focal point and may never
drive the decision.

Procedure, per scorer (2) x cell (9):

1. `MIX35-A`: fit on fold 1, predict only fold 0 rows.
2. `MIX35-B`: fit on fold 0, predict only fold 1 rows.
3. OOF table: fold-0 rows then fold-1 rows (same canonical order as the sealed
   OOF table).
4. Secondary probe only (never scored as primary): `MIX35-pooled` fit on both
   folds pooled, applied to both; included to test whether cross-fitting masks
   a pooled-data effect.
5. Acceptance rule (mirrors the sealed AA acceptance): each fit must accept on
   every action; every action needs >= 20 fitting rows and >= 2 positives and
   >= 2 negatives; every held-out fold domain row must have both classes.

Metrics, recomputed from the OOF probabilities with the repo's own
`v21_qualification_metrics` (equal-mass ECE, 10 bins; calibration bias;
BCE; Brier; ROC-AUC; PR-AUC), for each scorer x cell:

- factual domain (rows where factual action == a, per action; aggregate =
  union): the sealed failure domain;
- all-action domain: the sealed control domain;
- per-action tables for both domains.

Frozen decision gates for the focal MIX35 (w = 0.35), relative to the sealed
AA OOF of the same scorer/cell:

- A1 (primary, factual): aggregate factual `|calibration_bias| <= 0.05` AND
  aggregate factual `equal-mass ECE <= 0.05` in ALL 18 scorer/cell OOF
  tables (the exact sealed 0.05 factual limit, not relaxed).
- A2 (paired): grand paired factual `BCE_AA - BCE_MIX35` and `Brier_AA -
  Brier_MIX35` with BOTH held-out fold points positive in every cell, AND
  positive cluster-bootstrap LCBs (2.5% quantile, 50,000 resamples from the
  sealed `bootstrap_indices`, episode-cluster level) for both scores.
- A3 (controls): all-action aggregate `|calibration_bias|` and
  `equal-mass ECE` not worse than the sealed AA OOF values by more than 0.01
  in any scorer/cell; no per-action ROC-AUC drop > 0.05 in the factual
  domain.

Arm A passes only if A1 AND A2 AND A3 all hold. Context variants w = 0.5 /
1.0 report only; they can never rescue a focal-point failure.

## ARM B — prior-action residual strata on reconstructed consumed CAL tapes

Reconstruction (CPU only, single-thread, deterministic, CUDA hidden):

- Load the exact parent through `strict.load_exact_parent` on the sealed
  PB21M result; verify the parent state SHA-256 before and after the whole
  audit (the model must be untouched).
- For each of the six sealed CAL partitions (C0A, C0B, C1A, C1B, C2A, C2B)
  with the registered PB21M partition contracts, replay
  `collect_fresh_live_tape` exactly as PB21M did (same dataset config,
  burn-in 4, sequence length 16, 256 episodes, all-action counterfactual
  targets, balanced-intervention behavior at rate 0.5).
- Byte-identity verification against the sealed evidence before any
  statistics: for each cohort the replayed `root_state_ids` (ordered),
  `factual_actions`, and `hazard_targets` must match
  `cal_root_ids` / `cal_actions` / `cal_targets` exactly. Any mismatch is a
  terminal `replay_drift` failure; nothing is scored.

Prior-action construction (post-hoc arithmetic, no re-inference):

- The tape's `applied` sequence per episode gives the action applied at each
  tick. The **prior action** of a root is the applied action at the previous
  tick of the same episode. This matches the live loop exactly: the collector
  initializes prior = 0 for tick 0 and updates `prior_action = applied` after
  *every* tick, including burn-in ticks 0..3. Therefore the first recorded
  root of each episode (tick 4) carries the burn-in tick-3 applied action as
  its prior — NOT 0. The burn-in applied actions are recorded during replay
  for exactly this purpose.
- Residuals: `r[i, a] = p[i, a] - y[i, a]` in probability space, where
  `p[i, a]` are the sealed AA OOF probabilities (`oof_calibrated_probabilities`)
  for the row's scorer/cell (both scorers) and `y[i, a]` the sealed target.
  (The sealed OOF table is the only calibration state that was consumed
  before the terminal negative; using it keeps the audit strictly on
  consumed data.)

Strata (per cohort x scorer x cell, per prior-action value in 0..4):

- stratum size; stratum mean residual; stratum factual mean residual (rows
  where prior action == factual action at that root);
- per-stratum factual `calibration_bias` (mean probability minus factual
  prevalence within the stratum's factual rows) and equal-mass ECE of the
  factual probabilities restricted to the stratum (10 bins; strata below
  20 rows are reported as `insufficient_support` and excluded from the
  gate).

Frozen decision gates:

- B1 (reconstruction): byte-identity as above, all six cohorts.
- B2 (structure): at least 3 of 9 cells (per scorer) show a
  prior-action-stratum factual `|calibration_bias| >= 0.03` AND
  stratum factual ECE `>= 0.05`, i.e. a non-constant factual calibration
  error organized by prior action.
- B3 (specificity): a fixed prior-label shuffle control — permute prior
  labels within each cohort (30 deterministic permutations, fixed seeds,
  within-cluster to preserve cluster size) and recompute the max-min range
  of stratum factual bias; the observed range must exceed at least 95% of
  the permuted ranges.

Arm B passes only if B1 AND B2 AND B3 all hold.

## Nomination rule (frozen)

- At most ONE single change may be nominated for a later fresh PB21N
  preregistration.
- If exactly one arm passes: nominate that arm's single change
  (A: "replace the standard AA CAL calibrator with a fixed MIX35 weighted
  per-action affine calibrator"; B: "condition the hazard representation on
  the prior applied action").
- If both arms pass: nominate Arm A only (calibration is the closer, cheaper,
  single-mechanism change; the prior-action input is logged to the
  architecture ideas ledger for a later preregistration).
- If neither arm passes: nominate nothing; log both mechanisms to the
  architecture ideas ledger with their measured numbers.
- The two changes may NEVER be combined in a single trial.
- Nomination is a license to write one fresh PB21N preregistration with a
  fresh namespace; it authorizes no run by itself.

## Execution contract

- CPU only; `CUDA_VISIBLE_DEVICES=-1`; one intra/ and one inter-op thread;
  `OMP_NUM_THREADS=1`; deterministic torch mode; `run_provenance`
  metadata (bank/init digests, seeds, det flag, torch version) attached to
  the result.
- Create-only artifacts under `brain/runs/pb21m-posthoc-audit/`:
  `...registration.json`, `...attempt.json`, `...evidence.npz`,
  `...json`. Once an attempt file exists, the audit may not be rerun;
  results are published exactly once.
- No DEV, CPU-QUAL, PLAY-QUAL, or TEST partition may be constructed, loaded,
  or hashed at any point.
- Peak working-set and RAM checks mirror the PB21M resource guard.
- The full repository test suite (including the new audit test module) must
  pass before the result is published.

## Interpretation guardrails

- Both arms are post-hoc on consumed data: a pass is a **mechanism signal**,
  never evidence of a capability. [HYPOTHESIS-LEVEL OUTPUT]
- A MIX35 pass does not imply the scorer was well calibrated in general; it
  only says the frozen factual emphasis removes the measured factual
  miscalibration on consumed CAL.
- An Arm B pass does not mean the model uses prior actions; it means the
  calibration error is organized by prior action, which is one possible
  signature of a missing input channel.
