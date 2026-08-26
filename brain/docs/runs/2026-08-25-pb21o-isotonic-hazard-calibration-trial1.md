# PB21O trial #1 — isotonic (PAV) per-action hazard calibrator: FROZEN NEGATIVE

Date sealed: 2026-08-25

Canonical run: `2026-08-25-pb21o-isotonic-hazard-calibration-v1`

Classification: `fresh_isotonic_hazard_calibration_single_mechanism`

## 1. Verdict

**FROZEN NEGATIVE.** Gates G1 (factual absolute), G2 (paired improvement vs
base), and G5 (function-class isolation vs the affine control) **fail**;
G3 (all-action control not degraded), G4 (parent preservation), and
G6 (determinism) pass. Per the frozen prereg
(`2026-08-25-pb21o-isotonic-hazard-calibration-v1.md`) this namespace is
closed: no retry, no re-tuning, DEV remains sealed. The negative is
preserved verbatim.

**Headline:** on the prior-action-conditioned hazard path (L1, mechanism
already confirmed specific but not calibrating by PB21N #1), replacing the
per-action *affine* calibrator with a non-linear *isotonic* (PAV)
calibrator is **strictly worse** on every measured gate that the affine
version failed, and it fails the function-class isolation gate (G5) that
existed to ask exactly this question.

## 2. Gate results (fresh PB21O-CAL partition: 256 episodes, 3072 roots,
seed offset 167,774,720 — verified disjoint from the occupied-range
registry, immediately after PB21N-CAL)

| Gate | Meaning | Result | Detail |
|------|---------|--------|--------|
| G1 factual absolute | \|factual bias\| ≤ 0.05 AND factual ECE ≤ 0.05 on P_iso, both folds | **FAIL** | fold0 bias 0.084201, ECE 0.084201; fold1 bias 0.047384, ECE 0.062528 |
| G2 paired vs base | P_iso beats P_base on factual BCE **and** Brier, 97.5% LCB > 0 (episode-clustered, 10,000 resamples @ α=0.025 linear) | **FAIL** | fold0 BCE −0.0948 (LCB −0.1898), fold1 BCE −0.1030 (LCB −0.1905) — P_iso *worsens* factual loss vs the AA-calibrated base |
| G3 all-action control | all-action \|bias\|/ECE not degraded > 0.01 vs P_base; per-action factual AUC drop ≤ 0.05 | **PASS** | fold0 bias_abs Δ −0.0108, ECE Δ −0.0103, worst AUC drop 0.0338; fold1 bias_abs Δ −0.0146, ECE Δ −0.0066, AUC drop 0.0 |
| G4 parent preservation | sealed parent state-digest byte-exact after the whole trial | **PASS** | `2619b5b0…` before = after = sealed |
| G5 function-class isolation | P_iso beats P_aa (isotonic-vs-affine, **same conditioned path**), 97.5% LCB > 0 | **FAIL** | fold0 BCE −0.1209 (LCB −0.2095), fold1 BCE −0.1094 (LCB −0.1917) — **isotonic is worse than affine** on the identical path |
| G6 determinism | conditioning re-derivation + PAV OOF-prob re-derivation byte-exact | **PASS** | both re-derivations byte-exact |

Passed: **false** (G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5 all required).

## 3. Interpretation (hostile-scientist standards)

1. **G5 is the informative gate and it is a clean, meaningful negative.**
   G5 compares the two function classes on the *same* conditioned OOF
   logits (only the calibrator differs), so it isolates the mechanism.
   Result: the richer function class (monotone piecewise-constant, up to
   1,536 blocks per action at the per-fold fit scale) does **not** recover
   the factual miscalibration that the 2-parameter affine map failed to
   fix (PB21N #1: fold1 bias 0.0823/ECE 0.0823); it makes factual loss
   *worse* by ~0.11 BCE in both folds. A non-linear monotone remap of the
   same logit carries no new information beyond the affine fit; with
   1,536 fit rows and sparse positive rates it only adds fit variance.
2. **G3 passing while G1/G2 fail is coherent, not a bug.** All-action
   (5×3,072 targets per fold) the isotonic map smooths the aggregate
   calibration; the factual subset (the applied action's column) is where
   the signal is sparse and the PAV blocks mis-allocate mass. This mirrors
   the PB21M/PB21N pattern: aggregate all-action calibration is not the
   binding constraint; factual-domain calibration is, and it is
   representation-limited, not calibrator-limited.
3. **Cumulative mechanism map (three frozen trials, fresh namespaces):**
   - Prior-action conditioning: **specific** (PB21N G5/G6 passed) but does
     not calibrate the factual domain (PB21N G1/G3).
   - Affine calibrator on the conditioned path: already refuted (PB21N G1/G2).
   - Isotonic calibrator on the conditioned path: refuted *and* strictly
     inferior to affine (PB21O G5).
   The factual hazard miscalibration of the V2.1i frozen scorer therefore
   survives every post-hoc post-processing mechanism in the current family.
   The remaining unexplored direction is **representation-level** (change
   what the hazard path sees — e.g. 2-step prior window, outcome-history
   features, or a different context trunk), which is a new mechanism
   requiring its own fresh namespace and prereg; it is not licensed by any
   closed trial.

## 4. Execution-contract compliance

- **Prereg frozen before first run**; fresh partition verified disjoint
  against the occupied-range registry before launch (no overlap, contiguous
  after PB21N-CAL end 167,774,720).
- **Full test suite green before publication** (103/104 modules; the single
  red module is the pre-existing environmental `test_dgx_launch_contract`
  DGX-surface POSIX tests, off-limits and unchanged).
- **Create-only publish**: `attempt.json` written exactly once
  (`retry_allowed: false`); re-publish raises.
- **Determinism**: G6 passed — the prior-path conditioning re-derivation
  and the PAV OOF-probability re-derivation were byte-exact. (The PB21N #1
  determinism false-negative from `np.empty` OOF complement rows is fixed
  in the shared runner; PB21O inherits the zero-initialized OOF
  convention, so G6 is meaningful here.)
- **PAV implementation** cross-checked against
  `sklearn.isotonic.IsotonicRegression` at machine precision
  (max |Δ| = 1.1e-16 over 5 random fits, 11-test module) — test-only
  cross-check, not on the trial path.
- **Hash provenance**: `result_sha256` = SHA-256 of the result file bytes
  minus the trailing CRLF (= the compact JSON payload the runner hashes).
  Verified: recorded `4a3b9bea982e2cb32f7f3219fa032aec20188cb9e4b135ed70fd1d7506f62719`
  == `sha256(file[:-2])` and == re-serialization of the parsed doc with the
  runner's exact JSON params.

## 5. Sealed artifacts (create-only, do not modify)

Directory: `brain/runs/pb21o-isotonic-hazard-cal/`

| Artifact | Bytes | SHA-256 (raw file bytes) |
|----------|-------|--------------------------|
| `2026-08-25-pb21o-isotonic-hazard-calibration-v1.json` (result) | 18,249 | `c9390921453e2517f06e8bbeb65664149e579377985324036a8eaaacc5ed8e8c` |
| `2026-08-25-pb21o-isotonic-hazard-calibration-v1.attempt.json` | 247 | `33fd9dd0a2b8c9a4431a0a073b385c05024f9c5b4d7643776406db06191f2961` |
| `2026-08-25-pb21o-isotonic-hazard-calibration-v1.registration.json` | 1,017 | `7ade655465812b017b644438392820c304e6dbb0a8dd9dcf808e3565f9a91cd6` |
| `2026-08-25-pb21o-isotonic-hazard-calibration-v1.evidence.npz` | 4,476,630 | `1b3d5ab49839c9bc19c195ea8f5ed486d631dac4967802e66a396e18e9292670` |

Result-payload convention: `sha256(result bytes − CRLF)` =
`4a3b9bea982e2cb32f7f3219fa032aec20188cb9e4b135ed70fd1d7506f62719`
(== attempt `result_sha256`). Evidence NPZ carries all five OOF tables
(`oof_prior_raw`, `oof_H_prior_iso_prob`, `oof_H_prior_aa_prob`,
`oof_H_base_prob`, plus the frozen-replay arrays); eval-row halves of the
OOF tables verified finite; `oof_prior_raw` raw-bytes prefix
`ef3499a8f19a0553…`.

Wall clock: 32.4 s end-to-end (single-thread CPU, deterministic flags on,
CUDA hidden).

## 6. Consequences

- **Namespace closed.** No retry of PB21O; no re-tuning of the PAV (block
  count is already maximal at the data scale — no tunable remained).
- **L4 (isotonic) lead is now `[MEASURED negative]`** in the architecture
  ideas ledger; the L7 combined lead is fully refuted for both the affine
  and the isotonic function classes.
- **The PB21O-class lane remains gated on a calibrated hazard path**,
  unchanged from PB21N #1. The next *untested* mechanism is
  representation-level (new input features to the hazard path), which
  requires a fresh namespace + prereg + control arm and is **not**
  licensed by any closed trial. Until then the operative scorer remains
  the V2.1i frozen raw scorer + five action-specific logit biases, and the
  PB21M terminal scientific CAL negative stands.
- **Next gate (default, when a new mechanism is authorized):** a fresh
  representation-level trial (e.g. 2-step prior-action window or an
  outcome-history feature) in a new namespace, or — if the user prefers
  breadth over the hazard branch — the V2.1 latency/component
  qualification lane. Both are documented in
  `ARCHITECTURE_IDEAS_LEDGER.md` with their evidence labels.
