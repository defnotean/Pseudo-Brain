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
Status: `[HYPOTHESIS]` · not tested.

- **What:** replace the per-action *affine* (two-parameter) hazard
  calibrator with a monotone (isotonic or spline) per-action calibrator,
  to fit a non-linear factual-vs-counterfactual probability shape.
- **Basis:** the per-action bias structure measured in PB21M/PB21L is not a
  single shift+scale; an affine map may be the wrong function class.
  `[INFERRED]`
- **Cost:** moderate. **Risk:** more parameters ⇒ overfit risk on a 6144-
  row CAL split; needs stronger holdout and the all-action control gate.

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
Status: `[HYPOTHESIS]` · untested · **not licensed** — requires a fresh
namespace + its own prereg + a control arm; it deliberately *combines* the
L1 and L2 mechanisms, which the PB21M frozen nomination rule forbids inside
one trial, hence the fresh-prereg requirement.

- **What:** take the L1 prior-action hazard representation (G5/G6-verified
  specific) and, on top of it, refit the per-action affine (or L4 non-linear)
  calibrator against the *conditioned* hazard path's OOF logits on a fresh
  CAL partition.
- **Basis [INFERRED]:** PB21N trial #1 shows L1 alone fixes the factual
  mean (G2) and is specific (G5/G6) yet leaves fold-1 factual bias/ECE at
  0.082 (G1 fail) and drifts all-action control ~0.011–0.015 (G3 fail) —
  the same two-constraint collision that MIX35 Arm A measured on the
  *unconditioned* path (L2). Hypothesis: conditioning changes the residual
  miscalibration shape enough that a refit calibrator now lands inside both
  gates where neither mechanism alone could.
- **Cost:** one extra cross-fit + AA fit on top of the L1 trial; CPU-feasible.
- **Risk:** the combined candidate must re-clear BOTH a factual-absolute
  gate AND an all-action no-degradation gate on fresh data with a control
  arm; if either fails, the mechanism stays "specific but not calibrating"
  and L4/L3 become the next single-mechanism leads.
- **Measured anchors to carry forward:** PB21N #1 G5 observed range 0.5206
  (strong), G6 fold-1 point +0.0336 BCE, G1 fold-1 bias 0.0823, G3 worst
  ECEΔ +0.0149.

---

## Cross-cutting guardrails (all entries)
- A pass on any of these is a **mechanism signal, never a capability
  claim** unless it survives a fresh preregistered qualification with a
  held-out partition and a control arm isolating the mechanism.
- Report module cost separately; no ratios on signed near-zero metrics.
- Unreplicated gains are implementation-sensitive and do not carry forward.
- Preserve negative results verbatim (see L2).
