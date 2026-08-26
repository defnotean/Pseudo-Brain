# PB21O — Isotonic (PAV) per-action hazard calibrator on the prior-action
# conditioned path

Status: **PREREGISTRATION (frozen before any run).** Fresh namespace.
Classification (planned): `fresh_isotonic_hazard_calibration_single_mechanism`.

Date: 2026-08-25

## 0. What is licensed, and why

- The PB21M post-hoc audit nominated the prior-action hazard conditioning
  (Arm B). PB21N trial #1 (fresh namespace, sealed, frozen negative)
  measured that the conditioned path + **affine** (AA) calibrator is
  *specific* (G5/G6/G2/G4 pass) but *not calibrating* (G1 fold-1
  bias/ECE 0.0823; G3 all-action drift 0.011–0.015) on fresh data.
  `brain/docs/runs/2026-08-25-pb21n-prior-action-hazard-conditioning-
  trial1.md`.
- Architecture ideas ledger L4 ([HYPOTHESIS]): the residual miscalibration
  on the conditioned path may be **non-linear in logit space**, so the
  per-action calibrator's function class should be a monotone (isotonic /
  PAV) map, not a shift+scale.
- This prereg licenses exactly one trial: **the same prior-action
  conditioned hazard path, calibrated with per-action isotonic (PAV)
  regression instead of the affine AA map**, against an affine control arm
  on the identical path and partition, with a frozen parent baseline.
- No other change (no new representation mechanism, no combined re-
  weighting, no 2-step prior window, no pruning reorder) is authorized.

## 1. Parent (frozen)

- V2.1i seed42 exact-v3, byte-verified:
  - v3 result JSON SHA-256 `3f91921b7ca63148addd3204cdb1e96b130f26240f49277a269ac8f079b13936`
  - parent state-digest SHA-256 `2619b5b0a180bdd55296d3471d89725f4346110e74ef093f84c3068a32e6576d`
- The parent is read-only. Any drift in its state digest at trial start or
  end fails G4. No parent parameters are trained.

## 2. Single mechanism under test

`H_prior_iso` — the PB21N prior-action conditioned hazard path
(3H trunk + prior-action embedding; stop-gradient upstream; identical
graft and training schedule to PB21N trial #1: 8 passes × 64 optimizer
steps per OOF fold, AdamW lr 1e-3 wd 1e-4 clip 1.0, batch 24, seeded
deterministic permutations, TRIAL_SEED = 43), **calibrated per action with
isotonic regression (PAV)** on the raw OOF logits:

- PAV fit: per action, on the **complementary fold** (1536 rows), the
  pool-adjacent-violators algorithm on the (logit, target) pairs; stable
  sort by (value, row index); float64; piecewise-constant monotone map
  logit → probability. Deterministic; no random tie-breaks.
- PAV application: monotone piecewise-constant evaluation on the target
  fold's OOF logits.
- The fitted step function's **step count** is recorded per (fold, action)
  (diagnostic only; no gate on step count).
- Implementation: a hand-rolled deterministic PAV in the runner; a
  cross-check against `sklearn.isotonic.IsotonicRegression` (available,
  1.9.0) is performed in the test module only, never in the trial path.

Arms (all on the fresh PB21O-CAL partition, upstream belief frozen
byte-exact):
- `H_base` — frozen parent raw hazard logits, AA cross-fit calibrated
  (identical construction to PB21N trial #1).
- `H_prior_aa` — the prior-action conditioned path, **affine (AA)**
  cross-fit calibrated (the PB21N #1 mechanism, re-run on this partition).
- `H_prior_iso` — the prior-action conditioned path, **isotonic (PAV)**
  cross-fit calibrated (the candidate).

`H_prior_aa` and `H_prior_iso` share the exact same conditioned OOF raw
logit tables (one conditioning, two calibrators), so G5 isolates the
calibrator function class from everything else.

## 3. Data partition (fresh, disjoint)

- `PB21O-CAL`: `maze_chase`, split `train`, seed offset **167,774,720**,
  256 episodes (contiguous after PB21N-CAL, which ends at 167,774,720;
  collision-checked against the occupied-range registry before the run).
- burn_in_steps = 4, sequence_length = 16, roots per episode = 12,
  total roots = 3,072.
- OOF folds: 2 × 1536 rows; fold blocks are episode-disjoint
  (episodes 0–127 → fold 0, 128–255 → fold 1).
- No DEV, CPU-QUAL, PLAY-QUAL, or TEST partition may be constructed,
  loaded, or hashed.

## 4. Training schedule (frozen, bound to the V2.1i refinement contract)

- hazard path retrain: `REFINEMENT_PASSES = 8`,
  `REFINEMENT_ROOT_BATCH_SIZE = 24`, `REFINEMENT_LEARNING_RATE = 1e-3`,
  `REFINEMENT_WEIGHT_DECAY = 1e-4`, `REFINEMENT_CLIP_NORM = 1.0`
  (identical to PB21N trial #1; 512 optimizer steps per fold).
- AA control calibrator: canonical `hazard_calibration._fit_one_action`
  (L2 1e-6, min scale 1e-4, max iter 100, tol 1e-10), fit on the
  complementary fold's OOF logits.
- PAV calibrator: as in §2; fit on the complementary fold's OOF
  (logit, target) pairs.

## 5. Frozen gates (decision)

Let `P_iso`, `P_aa`, `P_base` be the three arms' OOF probability tables
on PB21O-CAL (P_iso/P_aa from the same conditioned raw logit tables).

- **G1 (factual absolute, P_iso):** both OOF folds:
  `|factual aggregate calibration_bias| ≤ 0.05` AND
  `factual aggregate ECE (equal-mass, 10 bins) ≤ 0.05`.
- **G2 (paired improvement vs frozen base):** paired `P_iso` vs `P_base`,
  factual, per root (BCE and Brier), episode-clustered bootstrap
  (10,000 resamples, TRIAL_SEED-bound); the one-sided 97.5% linear-
  quantile lower bound must be strictly positive for **both** folds and
  both loss types.
- **G3 (all-action control not degraded vs frozen base):** both folds:
  `|abs(all-action bias)_iso − abs(all-action bias)_base| ≤ 0.01`,
  `all-action ECE_iso − all-action ECE_base ≤ 0.01`, and worst
  per-action factual ROC-AUC drop ≤ 0.05.
- **G4 (parent preservation):** parent state digest byte-exact before and
  after the trial; parent file hashes unchanged.
- **G5 (function-class isolation — the control arm):** paired
  `P_iso` vs `P_aa` (identical conditioned path, PAV vs affine
  calibration), factual BCE per root, episode-clustered bootstrap; the
  one-sided 97.5% linear-quantile LCB must be strictly positive for
  **both** folds. If G5 fails, the PAV calibrator is not shown to
  improve on the affine calibrator beyond noise, and the trial is a
  frozen negative for the non-linearity claim even if G1–G3 pass.
- **G6 (determinism):** the P_iso OOF probability table re-derives
  byte-exactly from the saved evidence (re-fit PAV on the saved
  complementary-fold arrays and re-evaluate); the conditioned OOF raw
  logit tables re-derive byte-exactly from a second conditioning pass.

**PASS = G1 AND G2 AND G3 AND G4 AND G5 AND G6.** Any single gate failing
is a frozen negative for this trial; DEV stays sealed; no retry of this
namespace.

Edge-case guard (faithful extension, frozen with this prereg): any
per-action factual / all-action / PAV stratum without mixed outcomes is
excluded from the affected metric and recorded; this cannot change any
gate value on a fully scorable partition (see PB21N addendum §9).

## 6. Execution contract

- Create-only artifacts under `brain/runs/pb21o-isotonic-hazard-cal/`:
  `…registration.json`, `…attempt.json`, `…evidence.npz`, `…json`.
  Publish exactly once; an existing attempt file forbids a rerun.
- No DEV / CPU-QUAL / PLAY-QUAL / TEST partition may be touched.
- The full repository test suite (including the PB21O test module) must
  pass before the result is published.
- Determinism per G6.
- CPU-only, single-thread, deterministic, CUDA hidden
  (`CUDA_VISIBLE_DEVICES=-1`, OMP/MKL=1, torch threads=1).

## 7. Interpretation guardrails

- A pass is a **mechanism signal** [HYPOTHESIS-LEVEL OUTPUT]: it says the
  per-action miscalibration of the *conditioned* hazard path is
  substantially non-linear and that a monotone calibrator recovers the
  factual gate while holding the all-action control, versus the affine
  calibrator. It does NOT make the model "fully working," nor a
  scaling / live-play / full-model claim.
- It licenses at most one further step: a fresh, separately
  preregistered scaling/qualification trial in yet another fresh
  namespace. It authorizes no run by itself.
- A negative is preserved verbatim and the mechanism returned to the
  architecture ideas ledger with its measured numbers.

## 8. Next step on PASS

On PASS, the next gate is a fresh PB21P-class scaling/qualification
preregistration (new namespace) that would take the prior-action +
isotonic-calibrated hazard path into full-model qualification against
held-out DEV + CPU-QUAL. That is out of scope here.
