# PB21N trial #1 — prior-action-conditioned hazard representation: FROZEN NEGATIVE

Date sealed: 2026-08-25

Canonical run: `2026-08-25-pb21n-prior-action-hazard-conditioning-v1`

Classification: `fresh_hazard_prior_action_single_change` (planned) ·
outcome: **FROZEN NEGATIVE** (G1 and G3 fail) · DEV stays sealed · this
namespace is single-shot (attempt file exists; no re-execution).

Prereg: `brain/docs/preregistrations/
2026-08-25-pb21n-prior-action-hazard-conditioning-v1.md` (frozen; addendum §9
recorded the mixed-outcome edge-case guard appended before first execution).
Evidence (create-only, published exactly once):
`brain/runs/pb21n-hazard-prior-action/2026-08-25-pb21n-prior-action-
hazard-conditioning-v1.{json,attempt.json,registration.json,evidence.npz}`.

## 1. Verdict (frozen, from published result JSON)

| Gate | Verdict | Decisive numbers |
|------|---------|------------------|
| G1 factual absolute (H_prior vs 0.05) | **FAIL** | fold 0: bias 0.0433 / ECE 0.0477 (pass); **fold 1: bias 0.0823 / ECE 0.0823 (fail)** |
| G2 paired vs base (factual BCE+Brier LCB > 0) | PASS | fold 0 point BCE +0.0187 (LCB +0.0104); fold 1 point +0.0185 (LCB +0.0096) |
| G3 all-action controls not degraded vs base | **FAIL** | fold 0: biasΔ 0.0110, ECEΔ +0.0149 (both > 0.01); fold 1: biasΔ 0.0122, ECEΔ +0.0110 (both > 0.01) |
| G4 parent preservation | PASS | parent state SHA byte-exact before/after (`2619b5b0…6576d`) |
| G5 prior-label shuffle specificity | PASS | observed stratum-bias range 0.5206 > permuted p95 0.4247 (max 0.4284) |
| G6 mechanism isolation vs retrain control | PASS | fold 0 point +0.0081 BCE (LCB +0.00045); fold 1 point +0.0336 (LCB +0.0245) |

**PASS = G1∧G2∧G3∧G4∧G5∧G6 → false.** Preserved verbatim; no tuning; DEV
sealed; no retry of this namespace.

## 2. What the mechanism result actually says (negative, but informative)

[MEASURED] On the fresh PB21N-CAL partition (256 train episodes, 3,072
roots, seed offset 167,774,464 — contiguous after PB21M C2B, zero overlap),
the single nominated change **conditions the hazard representation on the
prior applied action** (one-step lag, burn-in-aware; 3H hazard trunk +
prior-action embedding, retrained 8 passes × 64 steps/fold, AdamW 1e-3,
wd 1e-4, clip 1.0):

- **The mechanism is real and specific** [MEASURED]: G5 (within-episode
  prior-label shuffle produces a *lower* factual-structure signal) and G6
  (the prior-action channel beats a pure-retrain control on the fresh
  partition) both pass. This confirms the audit's Arm B finding on
  independent fresh data and isolates the gain from mere retraining.
- **But as a single change it does not achieve calibration** [MEASURED]:
  fold 1 factual bias/ECE 0.082 vs 0.05, and all-action control drifts
  ~0.011–0.015 past the 0.01 no-degradation margin. The channel helps the
  factual mean (G2) and is specific (G5/G6) yet slightly *worsens*
  all-action calibration — the same two-constraint collision pattern the
  PB21M audit saw for MIX35 Arm A.

Honest synthesis [INFERRED]: prior-action conditioning is a genuine,
reproducible hazard-representation mechanism, but neither it alone nor a
focal-weighted calibrator alone clears the frozen gates. The candidate
direction for a future, separately preregistered trial is a **combined**
prior-action hazard path + re-fit calibrator (never combined inside one
frozen gate set) — recorded in the architecture ideas ledger as an
untested lead, labeled [HYPOTHESIS].

## 3. Execution-integrity note: published `determinism: false` is a documented
false negative (runner defect, not a trial defect)

The published result carries `determinism.crossfit_rederivation_byte_equal:
false`. Root cause [MEASURED, post-hoc re-derivation]:
`crossfit_arms` allocated the OOF tables with `np.empty` and writes only
slot[f]'s fold-f eval rows; the complement half holds uninitialized
memory. The determinism check byte-compares the *whole* `[2,3072,5]` array,
so it was comparing uninitialized bytes, not the OOF logits.

Verification [MEASURED]: re-deriving `crossfit_arms` twice from the saved
evidence arrays (read-only, no re-publish) gives **fold-0 and fold-1 eval
rows byte-identical for both arms** (`oof_*` eval blocks bit-equal across
the two re-derivations); only the whole-array (complement-including)
comparison differs. All gates were computed exclusively on eval rows, which
are finite and reproducible. The trial's scientific result therefore stands
as recorded; the determinism flag is a harness defect.

Runner fix (applied after trial #1; does not affect the sealed result):
`crossfit_arms` now zero-initializes the OOF tables so every slot is fully
defined and the whole-array check is meaningful. Regression test added:
`test_crossfit_oof_slots_zero_defined_complement_and_deterministic`
(`brain/tests/test_v21n_prior_action_hazard_conditioning_v1.py`, 19/19 OK).
Consequence: the `oof_logit_sha256` fields of trial #1's result JSON hash
the whole arrays *including* undefined complement bytes; the eval-row
reproducibility is established by the re-derivation above instead.

Additional pre-publish defects fixed before trial #1 (none reached the
published numbers): house `_state_dict_sha256` digest; graft parameter
access; init-identity float32-GEMM tolerance (max|Δ| = 4.8e-7, recorded);
fit-fold = complement-fold semantics in `train_hazard_fold`; AA cross-fit
fit-source = complementary fold's OOF logits (canonical, byte-verified in
`probe_aa_refit.py`); mixed-outcome guards on G1/G3/G5 metric calls (prereg
addendum §9).

## 4. Frozen provenance

- Parent: V2.1i seed42 exact-v3 (result SHA `3f91921b…b13936`, state SHA
  `2619b5b0…6576d`) — byte-verified at trial start and end (G4).
- Init-identity: prior-action graft at init differs from the base graft by
  max|Δ| = 4.77e-07 (float32 GEMM blocking over zero-padded columns); under
  the 1e-6/1e-5 gate. base 44,161 params / prior 59,161 params.
- Training: 4 hazard-path refits (retrain×2, prior×2), 8 passes × 64
  optimizer steps each, 512 steps total; deterministic seeded permutations.
- Calibration: canonical per-action affine AA cross-fit
  (`hazard_calibration._fit_one_action`, L2 1e-6, min scale 1e-4,
  max iter 100, tol 1e-10), fit on the complementary fold's OOF logits.
- Partition: manifest SHA `881d5efe…6ae8612a7`; episode permutation SHA
  `bbd330b1…f953f`. Wall 34.8 s, single-thread, CPU-only, CUDA hidden.

## 5. Decision / next gate

- No nomination from this trial (G1 and G3 fail). The single-change
  mechanism is **confirmed specific but not calibrating** on fresh data.
- DEV remains sealed. This namespace is closed (single-shot).
- **Next gate:** the **PB21O-class lane stays gated on a calibrated hazard
  path**. Candidate fresh-prereg direction (requires its own namespace,
  frozen gates, and a control arm): combined prior-action hazard
  representation + refit calibrator, or a longer-context (2-step)
  prior-action window; either must re-clear G1/G3-style gates on a fresh
  partition. Recorded in the architecture ideas ledger as [HYPOTHESIS]
  with this trial's numbers attached.
- No DGX, no scaling, no full-model work is unlocked by this result.

## 6. Sealed artifact byte-hashes (published exactly once)

- result JSON (payload, no trailing newline) SHA-256
  `87dfdf487eca9146f1efba69af5627e0aff339f668ab909107a8aacc178fd07f`
  — matches `attempt.json` `result_sha256`.
- on-disk byte SHA-256 (prefix):
  - `…-v1.json` `84fa8e81ec2fb909…`
  - `…-v1.attempt.json` `aea20b0e789f75d7…`
  - `…-v1.registration.json` `828f1f2e7e45c094…`
  - `…-v1.evidence.npz` `16c2cf2b00a9e0f8…`
