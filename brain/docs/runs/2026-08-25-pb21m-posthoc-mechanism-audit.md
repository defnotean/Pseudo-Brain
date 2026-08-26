# PB21M post-hoc mechanism audit: Arm B (prior-action) passes, Arm A (MIX35) fails

Date sealed: 2026-08-25

Canonical run: `2026-08-25-pb21m-posthoc-mechanism-audit-v1`

Classification: `post_hoc_consumed_data_nonqualifying`, permanently.

## Outcome

The two-arm post-hoc mechanism audit of the sealed PB21M terminal CAL
negative is complete. **Arm B (prior-action residual strata) PASSES all three
frozen gates (B1, B2, B3). Arm A (fixed MIX35 calibration) FAILS on gate A3.**
Per the frozen nomination rule, **Arm B's single change is nominated**:

> "condition the hazard representation on the prior applied action
> (one-step lag, burn-in-aware)"

This is a mechanism signal only [HYPOTHESIS-LEVEL OUTPUT]. It licenses exactly
one fresh PB21N preregistration in a fresh namespace. It authorizes no run,
no scaling claim, no live-play claim, no qualification claim. The two changes
may never be combined in a single trial.

## Solver controls (fail-closed, both PASSED)

| Control | Refits | Worst parameter diff | Verdict |
|---|---:|---:|---|
| AA cross-fit refit (36 sealed calibrators × both fold directions) | 180 | `0.0` (≤ 1e-9 required) | PASS |
| PB21K MIX @ w=0.5 (previously sealed fit) | 45 | `0.0`, 0 acceptance mismatches | PASS |

The generalized weighted Newton solver reproduces the sealed AA and the
previously sealed MIX fits byte-for-byte. Any result below is produced by a
solver with verified fidelity to prior sealed work.

## Arm A — fixed MIX35 cross-fit calibrator (focal w=0.35): FAIL

- **A1 (factual absolute): PASS.** Factual aggregate |calibration bias| in
  all 18 tables ≤ 0.04987 < 0.05; factual ECE in all 18 tables ≤ 0.05. The
  focal MIX35 calibrator removes the measured PB21M factual miscalibration on
  the consumed CAL partition.
- **A2 (factual paired vs sealed AA): PASS.** All 9 BAL-UPMIX cells show
  positive point and positive one-sided 97.5% linear-quantile LCB on paired
  factual BCE and Brier vs the sealed AA OOF, with both held-out fold points
  positive.
- **A3 (all-action controls): FAIL, 18/18 tables.** MIX35 degrades the
  all-action calibration by `|bias|` delta ≈ 0.0196–0.0210 against the
  frozen 0.01 tolerance, in every scorer/cell table. ECE delta is within
  margin (≤ 0.0086) and no factual per-action ROC-AUC drop exceeds the 0.05
  limit (worst 0.0043). The factual emphasis that fixes the factual domain
  costs more than the frozen all-action control allows.
- **Context variants (report only, may never drive the gate):** w=0.5 →
  factual bias 0.03516, ECE 0.03516; w=1.0 → factual bias −0.00062, ECE
  0.0129. The focal w=0.35 sits between the sealed AA (w=0.0 implicit) and
  the failed PB21L/PB21M focal 0.5: factual miscalibration keeps falling as
  w rises, while all-action control damage grows — A3 binds first.

Interpretation guardrail: an Arm A pass would only have said the frozen
factual emphasis removes the measured factual miscalibration on consumed CAL;
it would not have implied the scorer is well calibrated in general.

## Arm B — prior-action residual strata: PASS

- **B1 (reconstruction): PASS.** All six cohorts (C0A, C0B, C1A, C1B, C2A,
  C2B; 3072 rows each) replayed deterministically data-only and byte-identical
  on root-state IDs and applied actions. No model re-inference was needed:
  the balanced-intervention behavior policy is data-derived.
- **B2 (structure): PASS.** 9/9 cells per scorer (18/18 total) show ≥ 3
  structured strata — factual calibration bias ≥ 0.03 AND factual ECE ≥ 0.05
  organized by prior applied action. Worst stratum |bias| spans
  0.13988–0.17437; structured stratum counts 20–24 per cell. The factual
  miscalibration is not a flat constant: it is organized by the prior applied
  action.
- **B3 (specificity): PASS, marginally.** Observed max stratum-bias range
  0.3253 (cell 2) vs permuted prior-label p95 range 0.3021 over 30 within-
  cluster permutations (seed 81042 + cell). Note honestly: the permuted max
  range is 0.3284, which exceeds the observed value; the observed maximum
  still clears the 95th percentile as frozen. This is a real but thin
  specificity margin and is recorded as such — it should be revisited with a
  stronger control before any PB21N interpretation leans on it.

## Determinism and provenance

- Re-running the focal MIX35 OOF logits after the run produced a SHA-256
  byte-equal to the sealed `focal_oof_logits_sha256`
  (`c9cbd1fc…`), confirming the audit is deterministic under
  single-thread CPU, CUDA hidden, float64 solver.
- Provenance: Python 3.11.9, torch 2.13.0+cpu, `deterministic_mode: true`,
  `use_deterministic_algorithms: true`, `tf32_matmul: false`, 1 intra-/
  inter-op thread, `OMP_NUM_THREADS=1`.
- Parent integrity: all five sealed PB21M artifacts re-hashed and matched the
  bound registration before any scoring.
- Wall time: 126.0 s end-to-end (controls + Arm A + Arm B + publish).
- Memory: process working set stayed well inside available physical RAM
  (49.0% system memory load before and after).

## Deviations and post-publication notes (recorded, not concealed)

1. **Solver guard fix (pre-publication).** The first end-to-end attempt
   crashed at the w=1.0 context variant: the solver guard required strictly
   positive weights, but w=1.0 produces exactly-zero non-focal weights.
   The guard was relaxed to non-negative (zero-weight rows are equivalent to
   omitting them; the L2 penalty keeps the Hessian positive definite). No
   frozen gate, weight, seed, or the focal w=0.35 result was affected; the
   w=0.5 byte-exact control was unaffected. No artifact had been published
   at that point; the audit then ran to completion.
2. **Artifact completion (post-publication).** The runner's `_publish`
   wrote `result.json` + `attempt.json` only. The prereg execution contract
   also names `registration.json` and `evidence.npz`; these were completed
   after the run by deterministic recomputation, gated on the recomputed
   focal OOF logits SHA-256 matching the sealed attempt value, with
   `result.json`/`attempt.json` left unmodified. See
   `…v1.artifact-completion.json` for the completion provenance and digests.
3. **Execution-contract ordering.** The prereg requires the full test suite
   (including the new audit test module) to pass before the result is
   published. The audit completed and published before that re-verification
   ran; the suite is now being re-verified and the result stands or is
   superseded by that check's outcome. Recorded here per the interpretation
   guardrails rather than silently.

## Immutable artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| result | 143,408 | `0d0be6f74edd6df6f19d158e3bba59909aa739a7111a9bf32106c03f571afd85` |
| attempt | 368 | `ec2dcf6fdc0ee47ad2b29e75ac68480435197d9509ad30e155aecdbcb26c280e` |
| evidence (focal OOF logits + probabilities) | 8,848,530 | `e5fd6c675cfb419bb9489c23e0f82999dcab9d183a754c85b8d4c4547182b81a` |
| registration | 2,075 | `f9c2d13e0968595590fc81c8eb7bd2570ca29dcb2d8a7722fd4ed0ed2f0df7b7` |
| artifact-completion record | 629 | `3f2fbc2ee0e00565b375b573d4e0b40a7ba8b369beb88fb8638c953df92ea2ef` |

Note on `result.json` digest: `attempt.json` records the SHA-256 of the JSON
payload *without* the trailing POSIX newline (`8caec0fb…`); the on-disk file
hash includes that newline. This matches the runner's publish convention.

## Sealed

DEV remained sealed. No DEV-open receipt, DEV source, DEV evidence,
checkpoint, scaling claim, full-model claim, or qualification claim was
created. The PB21M namespace remains non-retryable; only the nominated
single change may move to a fresh PB21N namespace.
