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

**Program status (2026-08-26):** the hazard branch is fully characterized
at a clean boundary. The **post-processing** family — per-action
calibration with or without prior-action conditioning, affine or
monotone-nonlinear (PAV) or block-capped PAV — is **closed** as a route to
the frozen factual gate (L1/L4/L7 sealed frozen negatives; L8 probe-
refutes the last calibration hypothesis). Two representation-level
directions are **probe-refuted** at probe level: the prior-action
**window-length** axis is **closed at K=1** (L9: a 2-step window's factual
AUC 0.6973 sits *below* the 1-step ceiling 0.7060, no PB21Q trial) and the
**crude outcome-history** feature is **closed** (L10: despite the
hazard target's measured multi-tick autocorrelation — P(h_t|h_{t-k}=1)
lift 1.4–1.8× for k=1–4 — a recent-hazard-count channel drops factual AUC
from 0.6706 to 0.6583; no PB21R trial). The diagnosed binding constraint
is **factual-hazard ranking / signal density** (AUC ~0.70), not calibrator
function class, window length, or a coarse history feature. The remaining
unexplored directions are finer-grained representations (per-action
recency-weighted history, belief-residual history, a dedicated recurrent
hazard-history state, a richer candidate-action encoding, a different
context trunk or upstream representation); none is motivated by the
existing probes and each needs its own fresh namespace + prereg + control
arm — not licensed by any closed trial. See L8/L9/L10.

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

## L8 — Hazard information-ceiling diagnosis + block-cap PAV refutation
Status: `[MEASURED]` (read-only diagnostics on sealed PB21O evidence) ·
**closes the calibration function-class family; no PB21P trial warranted.**

- **Why:** PB21N (affine) and PB21O (full isotonic/PAV) both failed G1, and
  in-sample PAV looked dramatically better than cross-fit PAV. Before
  authoring a representation-level trial, two read-only, publish-free
  probes on the *sealed* PB21O OOF logits decided whether the binding
  failure was calibration-level (fixable by a better/more-constrained
  function class) or ranking-level (no remap can fix it).
- **Probe 1 — information ceiling** (`brain/scratch/probe_hazard_info_\
ceiling.py`):
  - Factual aggregate AUC: **0.631 (frozen base) → 0.708 (prior-
    conditioned OOF)**. Per-action 0.649–0.746 (base) → 0.679–0.746
    (prior). AUC is invariant to *any* monotone remap — 0.71 is the ceiling
    every calibrator can ever reach on this logit. `[MEASURED]`
  - In-sample per-action PAV factual ECE: **agg 0.0039 (fold0) / 0.0082
    (fold1)**, vs the trial's cross-fit PAV 0.0842 / 0.0625. The 0.05 gap
    is cross-fit variance at sparse positive rates (factual positive rates
    per applied action: 0.067–0.375; n_factual 413–747), not a
    function-class impossibility. `[MEASURED]`
  - Representation sufficiency: the hazard logit beats a frozen linear
    belief readout in 7/10 factual (fold, action) cells (worst cell 0.667
    vs 0.725); the representation is not under-fit relative to the beliefs.
    `[MEASURED]`
- **Probe 2 — block-cap PAV sweep** (`brain/scratch/probe_pb21p_blockcap_\
pav.py`): froze a deterministic block-count cap K on PAV (greedy SSE-merge
pooling) and cross-fit K ∈ {2,4,6,8,16, full} on the sealed OOF logits:
  - Factual ECE (fold0/fold1): affine 0.0834/0.0644 · K=2 0.0936/0.0873 ·
    K=4 0.0836/0.0663 · K=6 0.0800/0.0668 · **K=8 0.0798/0.0625 (best)** ·
    K=16 0.0837/0.0621 · full PAV 0.0842/0.0625. **No member clears the
    0.05 G1 limit on either fold.** `[MEASURED]`
- **Conclusion [INFERRED, on top of the measured anchors]:** the entire
  monotone-calibration spectrum (affine → block-capped K=2..16 → full PAV)
  saturates at ~0.06–0.08 factual ECE on this hazard logit. The
  "intermediate block-cap" hypothesis that would have justified a PB21P
  calibration trial is **refuted at probe level** — no trial warranted, and
  none will be run. Combined with the ceiling: the factual failure is
  **ranking/signal-density-limited** (AUC 0.71, sparse positives), not
  calibrator-limited. This completes the L1/L4/L7 closure of the
  hazard post-processing family.
- **Consequence:** the next mechanism must be representation-level
  (change what the hazard path *sees*), and — because ranking is the
  binding constraint — any such trial should carry a **ranking gate**
  (e.g., factual AUC improvement over the 0.708 prior-conditioned
  ceiling), not only a calibration gate. Each candidate is a new
  namespace + prereg + control arm; **not licensed** by any closed trial.
- **Provenance:** both probes are read-only (no partition, no publish, no
  frozen-state change); inputs are the sealed PB21O evidence
  `2026-08-25-pb21o-isotonic-hazard-calibration-v1.evidence.npz` under
  `brain/runs/pb21o-isotonic-hazard-cal/` (artifact hashes in
  `brain/docs/runs/2026-08-25-pb21o-isotonic-hazard-calibration-trial1.md`).
  Both probes were re-run 2026-08-26 and reproduce the numbers above
  exactly (deterministic).

## L9 — 2-step prior-action window (representation-level, single change)
Status: `[MEASURED negative]` at probe level (read-only signal probe) ·
**no PB21Q trial warranted.**

- **What:** extend the measured 1-step prior-action mechanism (PB21N, L1)
  to a 2-step window — condition the hazard representation on the action
  applied one tick earlier *and* two ticks earlier. The single-change,
  CPU-feasible, ranking-testable representation-level extension of the only
  confirmed hazard mechanism (PB21N G5 specific; PB21O G6 retrain control
  passed). The natural first candidate for the "representation-level"
  direction L8 left open.
- **Probe (read-only, no publish, no frozen-state change):**
  `brain/scratch/probe_pb21q_2step_window_signal.py`. On the PB21O-CAL
  partition (seed_offset 167,774,720; 256 ep × 12 roots = 3072; 2 folds of
  1536; burn-in 4), retrained three zero-extended hazard arms per OOF fold
  (deterministic, CPU, single-thread, CUDA hidden; 33 s wall) — base (2H),
  1-step (3H, one prior channel), 2-step (4H, two prior channels) — and
  reported factual aggregate + per-action AUC and the in-sample PAV ECE
  ceiling per arm.
- **Measured (factual aggregate AUC; the L8 binding constraint):**
  - base (2H) **0.6706** · per-action 0.634–0.704
  - **1-step (3H) 0.7060** · per-action 0.674–0.750
  - **2-step (4H) 0.6973** · per-action 0.638–0.746
  - The 2-step arm **fails to clear the 1-step ceiling** (0.6973 < 0.7060)
    and is *worse* on actions 1 (0.664 vs 0.711) and 2 (0.638 vs 0.674).
  - In-sample PAV factual ECE ceiling (per fold / per action):
    base fold0 0.0437/0.0149/0.0312/0.0219/0.0223, fold1
    0.0404/0.0173/0.0325/0.0359/0.0243; 1-step fold0
    0.0416/0.0127/0.0189/0.0269/0.0215, fold1
    0.0562/0.0194/0.0341/0.0147/0.0177; **2-step fold0
    0.0550/0.0146/0.0112/0.0269/0.0263, fold1
    0.0593/0.0271/0.0470/0.0284/0.0371** — the 2-step ceiling is *worse*
    on the binding actions (fold0 a0 0.0550 vs 1-step 0.0416; fold1 a0
    0.0593 vs 0.0562). `[MEASURED]`
- **Conclusion [INFERRED, on the measured anchors]:** extending the
  prior-action window from 1 to 2 steps adds trunk capacity / fit variance,
  **not** ranking signal — the 2-step logit's factual AUC (0.6973) sits
  *below* the 1-step ceiling (0.7060) that L8 showed no calibrator can
  exceed. Within the prior-action family, **1 step is the best window**; a
  longer window does not lift the binding ranking constraint. **No PB21Q
  trial is warranted** (same discipline that saved the PB21P block-cap
  trial — the hypothesis is refuted at probe level). `[MEASURED]`
- **Consequence:** the prior-action *window-length* axis is closed at
  K=1. The remaining unexplored representation-level directions are
  orthogonal to window length (different context trunk, outcome-history
  features, a richer candidate-action encoding, or a different upstream
  representation altogether) — each a new namespace + prereg + control
  arm, **not licensed** by any closed trial, and none is motivated by this
  probe. The hazard branch is at a clean, well-characterized boundary:
  post-processing closed (L1/L4/L7/L8), window-length closed (L9); the
  binding constraint is factual-hazard ranking/signal density (AUC ~0.70).
- **Provenance:** read-only (no partition, no publish, no frozen-state
  change); deterministic replay of the frozen V2.1i parent (state digest
  `2619b5b0…`) over the PB21O-CAL partition. Re-run 2026-08-26; 33 s wall.
  The 1-step arm (0.7060) is a harness self-validation consistent with
  L8's 0.708 prior-conditioned ceiling on the identical partition.

## L10 — Outcome-history hazard features (representation-level, single change)
Status: `[MEASURED negative]` at probe level (read-only signal probes) ·
**no PB21R trial warranted.**

- **What:** condition the hazard representation on recent applied-trajectory
  hazard events (outcome-history features) — a representation-level
  direction *orthogonal* to the L9-refuted prior-action window-length
  axis. Motivated by a first finding [MEASURED]: the factual hazard
  target is **not exchangeable over ticks** (L10 motivation below).
- **Motivation [MEASURED]** (`brain/scratch/probe_outcome_history_\
signal.py`, read-only on the sealed PB21O evidence): factual marginal
  hazard rate 0.1315; P(h_t=1 | h_{t-k}=1) = 0.193/0.182/0.213/0.231 for
  k=1/2/3/4 (lift 1.46/1.38/1.62/1.76×), any-hazard-in-last-K lift
  1.36–1.53× for K=1–4 — persistent, multi-tick hazard autocorrelation.
  This is real target structure, unlike the L9 window-length axis which had
  no supporting signal.
- **Probe (read-only, no publish, no frozen-state change):**
  `brain/scratch/probe_pb21r_outcome_history_signal.py`. Same partition
  (PB21O-CAL), two zero-extended arms retrained per OOF fold
  (deterministic, CPU, single-thread, CUDA hidden; 31 s wall; init-identity
  max|Δ|=0.0): base (2H) vs hist (2H + 1-dim raw scalar channel =
  recent-hazard count in last K=4 applied-ticks / 4, mean 0.0898, 28.5% of
  rows nonzero). The outcome-history mechanism is tested **alone** against
  the base control (single-change discipline; NOT stacked on the 1-step
  action channel).
- **Measured (factual aggregate AUC; the L8 binding constraint):**
  - base (2H) **0.6706** (exactly reproduces L9's base arm — harness
    self-validation passes).
  - **hist (2H+1) 0.6583** — *below* base; per-action 0.623–0.702 vs base
    0.634–0.704 (worse on a1 0.692/0.634-adjacent, a2 0.623 vs 0.634).
  - In-sample PAV factual ECE ceiling: hist fold0
    0.0188/0.0175/0.0328/0.0121/0.0248 vs base
    0.0437/0.0149/0.0312/0.0219/0.0223; fold1 0.0740/0.0097/0.0397/0.0338/
    0.0214 vs base 0.0404/0.0173/0.0325/0.0359/0.0243 — mixed and on
    balance no better (fold1 a0 0.0740 is the worst single cell measured in
    this whole program). `[MEASURED]`
- **Conclusion [INFERRED, on the measured anchors]:** although the hazard
  target has genuine multi-tick autocorrelation, a *coarse recent-hazard
  count* adds trunk fit variance, **not** ranking signal — the hist arm's
  factual AUC (0.6583) sits *below* the base (0.6706) and far below the
  1-step action ceiling (0.7060). Consistent with L8's sufficiency finding:
  the frozen belief context already encodes recent state/outcome
  information; a scalar history feature carries no *new* ranking
  information the hazard path can exploit at this budget. **No PB21R trial
  is warranted** — refuted at probe level. `[MEASURED]`
- **Consequence:** the *crude-count* outcome-history feature is closed.
  The target's measured autocorrelation is preserved as a genuine
  [MEASURED] fact; a finer-grained history mechanism (per-action
  recency-weighted history, belief-residual history, or a dedicated
  recurrent hazard-history state) remains untested but is **not** motivated
  by this probe and would need its own fresh namespace + prereg + control
  arm. As of 2026-08-26 the hazard branch is fully characterized at a clean
  boundary: post-processing closed (L1/L4/L7/L8), window-length closed at
  K=1 (L9), crude outcome-history closed (L10); the binding constraint is
  factual-hazard ranking/signal density (AUC ~0.70).
- **Provenance:** read-only (no partition, no publish, no frozen-state
  change); deterministic replay of the frozen V2.1i parent (state digest
  `2619b5b0…`) over the PB21O-CAL partition. Re-run 2026-08-26; 31 s wall.
  Base arm 0.6706 == L9 base arm (identical parent/partition/schedule).

---

## Cross-cutting guardrails (all entries)
- A pass on any of these is a **mechanism signal, never a capability
  claim** unless it survives a fresh preregistered qualification with a
  held-out partition and a control arm isolating the mechanism.
- Report module cost separately; no ratios on signed near-zero metrics.
- Unreplicated gains are implementation-sensitive and do not carry forward.
- Preserve negative results verbatim (see L2).
