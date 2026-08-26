# Pseudo-Brain — Architecture Ideas Ledger

Living index of candidate architectural mechanisms, each with an evidence
label and a pointer to the run that produced it. Purpose: keep the ideas
pipeline auditable and prevent re-deriving (or silently discarding) prior
measurements. A mechanism earns candidacy only via a causal battery +
generalization + a control arm that isolates the mechanism (hostile-
scientist standard). This ledger records *measured* and *hypothesized*
ideas; it authorizes no run.

Status legend:
- `[MEASURED]` — a number from a sealed run.
- `[INFERRED]` — reasoned from measured numbers, not directly measured.
- `[HYPOTHESIS]` — a candidate mechanism, not yet shown.
- `[ASPIRATIONAL]` — a long-horizon goal, no causal evidence yet.

---

## L1 — Prior-action conditioning of the hazard representation
Status: `[MEASURED]` — mechanism **confirmed specific but not calibrating**
(PB21N trial #1, frozen negative; namespace closed).

- **What:** add the prior applied action (one-step lag, burn-in-aware) as an
  input to the hazard path, so the hazard representation is conditioned on
  the action the agent took one tick earlier.
- **Why now:** the two-arm PB21M post-hoc audit
  (`2026-08-25-pb21m-posthoc-mechanism-audit-v1`) showed the factual-domain
  miscalibration of the sealed BAL-UPMIX scorer is *organized by prior
  applied action* — 18/18 cells had ≥3 strata with factual
  |calibration_bias| ≥ 0.03 AND factual ECE ≥ 0.05 (worst |bias|
  0.140–0.174), and the specificity control cleared the p95 of within-
  cluster prior-label permutations (observed range 0.3253 vs 0.3021 p95;
  permuted max 0.3284 — a thin margin, recorded honestly). `[MEASURED]`
- **Design constraint:** the hazard path is `dedicated_stopgrad_v1` — it
  reads `context.detach()` plus an action embedding and cannot alter
  next-state, reward, decision, belief, or thought parameters. Adding a
  prior-action embedding keeps the change contained to the hazard path, so
  it is a single mechanism, not a combined change.
- **Cheapest faithful test:** retrain *only* the hazard path (stop-gradient
  trunk + action embedding + outcome trunk + head + per-action affine
  calibration) on a fresh CAL partition with the prior action as a new
  input, holding the upstream representation frozen (V2.1i raw scorer).
  Compare hazard factual miscalibration + all-action control against the
  frozen V2.1i baseline. This is CPU-feasible (no full-model retrain).
- **Prereg:** `brain/docs/preregistrations/2026-08-25-pb21n-prior-action-
  hazard-conditioning-v1.md`.
- **Risk / guardrail:** a pass is a mechanism signal only [HYPOTHESIS-
  LEVEL]; it does not by itself make the model "fully working."
- **PB21N trial #1 result (2026-08-25, FROZEN NEGATIVE):**
  `brain/docs/runs/2026-08-25-pb21n-prior-action-hazard-conditioning-
  trial1.md`. On the fresh PB21N-CAL partition: **G5 PASS** (observed
  stratum-bias range 0.5206 vs permuted p95 0.4247 — mechanism specific,
  stronger than the audit's thin margin), **G6 PASS** (beats the
  pure-retrain control, fold-0 LCB +0.00045 / fold-1 +0.0245), **G2 PASS**
  (factual BCE/Brier improvement vs frozen base, both folds LCB>0), **G4
  PASS** (parent byte-exact) — but **G1 FAIL** (fold-1 factual bias/ECE
  0.0823 vs 0.05 limit) and **G3 FAIL** (all-action bias/ECE drift
  0.011–0.015 vs 0.01 no-degradation margin, both folds). Synthesis
  [INFERRED]: the channel is a genuine hazard-representation mechanism but
  as a *single* change it does not clear the frozen calibration gates —
  same two-constraint collision pattern as L2/MIX35 Arm A. Namespace closed
  (single-shot); DEV sealed; no retry.
  Run-report integrity note: the published `determinism: false` flag is a
  documented harness false-negative (whole-array byte check over
  uninitialized OOF complement half); eval-row OOF logits were re-derived
  byte-exact post-hoc and all gates read eval rows only.

## L2 — Factual-emphasis calibration (MIX35 and context)
Status: `[MEASURED negative]` as a standalone calibrator · logged, not
nominated.

- **What:** replace the standard per-action affine (AA) CAL calibrator with
  a weighted per-action affine calibrator that up-weights factual rows
  (focal weight w).
- **Measured:** the PB21M post-hoc audit's Arm A (focal w=0.35) *removed* the
  factual-domain miscalibration on consumed CAL (worst factual |bias|
  0.04987 < 0.05; A1 + A2 passed) **but degraded the all-action control by
  ≈0.020 |bias| against the 0.01 tolerance in all 18 tables (A3 fail).**
  Context bracket: w=0.5 → factual bias 0.0352; w=1.0 → −0.0006 (fully
  fixed) — i.e. the two constraints (factual accuracy vs all-action
  control) collide; the focal point sits where all-action damage first
  exceeds budget. `[MEASURED]`
- **Read:** re-weighting calibration is the *symptom* treatment; the
  factual bias is not a constant that re-weighting can fix uniformly — it
  is structured (see L1). Logged to the ideas ledger per the frozen
  nomination rule; **may never be combined with L1 in a single trial.**
- **Why not nominated:** A3 (all-action control) is a hard frozen gate, and
  MIX35 fails it. A nominal PB21N must not re-open the calibration weight
  unless a *new* prereg justifies it; the cleaner single mechanism is L1.

## L3 — Pre-prune calibration (calibrate before the prune step)
Status: `[HYPOTHESIS]` · not tested.

- **What:** the V2.1i pipeline fits the raw scorer, then prunes, then fits
  the five per-action logit biases. Hypothesis: fitting the per-action
  biases *before* pruning (or against the un-pruned logit geometry) would
  reduce the factual miscalibration that survives to the pruned table.
- **Basis:** the prune step changes logit geometry
  (`pruning_padded_logit_sha256` / `pruning_pruned_logit_sha256` are bound in
  the PB21M evidence); the calibration fit is downstream of that. `[INFERRED]`
- **Cost:** low (reorder + refit). **Risk:** could change the frozen
  V2.1i pipeline contract; needs its own prereg and a DEV-holding-out
  comparison.

## L4 — Isotonic / non-linear per-action calibrator
Status: `[MEASURED negative]` (PB21O trial #1, frozen negative; namespace
closed) · was the licensed single-change trial after PB21N #1.

- **What:** replace the per-action *affine* (two-parameter) hazard
  calibrator with a monotone (isotonic/PAV) per-action calibrator,
  to fit a non-linear factual-vs-counterfactual probability shape.
- **Result (PB21O trial #1, 2026-08-25, sealed):** **G5 function-class
  isolation FAILED** — isotonic PAV is strictly *worse* than the affine
  control on the identical conditioned path (factual BCE Δ −0.121 fold 0,
  −0.109 fold 1, 97.5% LCB < 0 both). G1 factual absolute FAILED
  (bias 0.0842 / ECE 0.0842 fold 0; bias 0.0474 / ECE 0.0625 fold 1).
  G2 paired vs base FAILED. G3 all-action control PASSED (bias_abs Δ
  −0.0108/−0.0146, ECE Δ −0.0103/−0.0066, worst factual AUC drop
  0.034/0.0). G4 parent + G6 determinism PASSED. `[MEASURED]`
- **Why (mechanistic read):** a non-linear monotone remap of the *same*
  logit carries no new information beyond the affine fit; at 1,536 fit
  rows per fold with sparse positive rates it adds fit variance, not
  signal. The factual miscalibration is **representation-limited, not
  calibrator-limited**. `[INFERRED]`
- **Cost:** one extra PAV fit per action per fold on top of the L1 trial;
  CPU-feasible (32 s wall end-to-end).
- **Prereg:** `brain/docs/preregistrations/
  2026-08-25-pb21o-isotonic-hazard-calibration-v1.md`.
- **Report:** `brain/docs/runs/
  2026-08-25-pb21o-isotonic-hazard-calibration-trial1.md`.
- **Next:** none in this family — both function classes (affine PB21N,
  isotonic PB21O) refuted on the same conditioned path. Remaining
  unexplored direction is representation-level (new hazard-path inputs),
  a separate fresh namespace if ever authorized.

## L5 — Multi-tick prediction error + live episodic memory (V2.2)
Status: `[ASPIRATIONAL]` · roadmap, not a mechanism yet.

- **What:** extend the single-tick predictive trajectory to multi-tick
  prediction error and add a live episodic memory read/write path.
- **Status:** roadmap milestone after V2.1; no causal evidence yet.

## L6 — Meta-learning across trials (V2.3)
Status: `[ASPIRATIONAL]` · roadmap.

- **What:** a meta-learning loop that adapts the recurrent thoughtlets across
  trials rather than a single fixed policy.
- **Status:** roadmap milestone; no causal evidence yet.

## L7 — Combined prior-action hazard path + refit calibrator (lead from PB21N #1)
Status: `[MEASURED negative]` for **both** function classes (affine: PB21N
#1; isotonic: PB21O #1) · lead fully refuted, closed.

- **What (as originally framed):** take the L1 prior-action hazard
  representation and refit the per-action calibrator against its OOF
  logits. **Core correction (2026-08-25, post PB21N #1):** the *affine*
  version of this combination is NOT a new experiment — PB21N trial #1's
  `H_prior_cal` arm WAS exactly "prior-action path + refit per-action
  affine (AA) calibrator" on a fresh partition, and it **failed G1 (fold-1
  factual bias/ECE 0.0823) and G3 (all-action drift 0.011–0.015)** while
  passing G5/G6. `[MEASURED]`
- **Resolution of the remaining hypothesis (2026-08-25, PB21O #1):** the
  follow-up that a *non-linear* (isotonic/PAV) calibrator is the correct
  function class was tested in the fresh PB21O namespace and **FAILED** —
  G5 function-class isolation showed the isotonic map strictly *worse*
  than the affine control on the identical conditioned path (factual BCE
  Δ −0.121/−0.109, LCB < 0). `[MEASURED]`
- **Basis [INFERRED]:** both function classes refuted ⇒ the factual
  miscalibration of the V2.1i frozen scorer on this hazard path is
  **representation-limited**: no post-processing remap of the current
  hazard-path logit (linear or monotone-nonlinear, with or without
  prior-action conditioning) reaches the frozen factual gate. The
  two-constraint collision (factual vs all-action) measured by PB21M Arm
  A / PB21N G1+G3 / PB21O G1+G2 persists.
- **Next:** the remaining unexplored direction is representation-level
  (change what the hazard path *sees*: 2-step prior window, outcome-
  history features, different context trunk), each a new mechanism with
  its own fresh namespace + prereg + control arm. Not licensed by any
  closed trial. Report:
  `brain/docs/runs/2026-08-25-pb21o-isotonic-hazard-calibration-trial1.md`.
- **Measured anchors to carry forward:** PB21N #1 G5 observed range 0.5206
  (strong), G6 fold-1 point +0.0336 BCE, G1 fold-1 bias 0.0823, G3 worst
  ECEΔ +0.0149; PB21O #1 G5 factual-BCE Δ −0.121/−0.109 (isotonic worse
  than affine), G1 fold-0 bias/ECE 0.0842/0.0842, G3 all-action improved
  (bias_abs Δ −0.0108/−0.0146) while G1 failed — the factual-vs-
  all-action two-constraint collision persists across both function
  classes.

---

## Cross-cutting guardrails (all entries)
- A pass on any of these is a **mechanism signal, never a capability
  claim** unless it survives a fresh preregistered qualification with a
  held-out partition and a control arm isolating the mechanism.
- Report module cost separately; no ratios on signed near-zero metrics.
- Unreplicated gains are implementation-sensitive and do not carry forward.
- Preserve negative results verbatim (see L2).
