# PB21S — Recurrent applied-trajectory hazard-history state on the
# hazard path (representation-level single mechanism)

Status: **REFUTED BEFORE EXECUTION (2026-08-26) — superseded, no trial
published.** Fresh namespace.
Classification (planned): `fresh_recurrent_hazard_history_single_mechanism`.

**REFUTATION NOTE (appended before execution, post-freeze but pre-run):**
after this prereg was drafted, the L11 signal probe was re-run on the
**fresh disjoint PB21S-CAL partition** (the very partition this trial
would have used). The discovery-partition +0.0307 factual-AUC gain did
**not** generalize: on PB21S-CAL the rec arm scores **0.6523** vs the
retrained base **0.6711** (**−0.0188**), and the specificity permutation
there is p = 0.104 (not significant). Per §0's own fresh-partition
requirement ("a mechanism that does not generalize to fresh data is a
frozen negative for the generalization claim"), **no PB21S trial is
warranted** and this prereg is retained for provenance only. See
ARCHITECTURE_IDEAS_LEDGER L11.

Date: 2026-08-26

## 0. What is licensed, and why

- L8 (ledger) established that the hazard **post-processing** family is
  closed: the binding constraint is **ranking** (factual AUC ceiling ~0.708
  with prior-action conditioning; the factual target is sparse, positive
  rate 6.7–37.5% per action), not calibration function class.
- L9 (ledger, `probe_pb21q_2step_window_signal.py`) refuted the
  prior-action **window-length** direction: a 2-step window is *worse*
  (factual AUC 0.6973) than 1-step (0.7060) — window length is not the
  signal axis.
- L10 (ledger, `probe_outcome_history_signal.py` +
  `probe_pb21r_outcome_history_signal.py`) established a **[MEASURED]
  factual finding** — the factual hazard target is **not exchangeable over
  ticks** (applied-trajectory hazard at t−1..t−4 lifts P(hazard at t) by
  1.4–1.76×) — and refuted a *feedforward* outcome-history *scalar*
  (recent-hazard-count) input: it *degraded* ranking (factual AUC 0.6583
  vs base 0.6706), because a single-count snapshot carries no more than
  the belief context already encodes.
- L11 (ledger, `probe_pb21s_recurrent_history_signal.py` +
  `probe_pb21s_specificity_perm.py`) is the **first representation-level
  mechanism that survives the full evidence battery**: a *recurrent*
  applied-trajectory hazard-history state (a 1-layer GRU, hidden 128,
  reset to zero at tick 0, fed the applied (action-emb + hazard-event)
  sequence through tick t−1 — causal, no current-tick leakage) **conditions
  the hazard representation**. On the PB21O-CAL discovery partition:
  - rec(2H+128) factual aggregate AUC **0.6806** vs retrained
    base(2H) **0.6499** → **+0.0307** (seed 0; multi-seed mean +0.023,
    all 4 seeds positive).
  - **Specificity permutation:** 1000 episode-permutations of the
    applied-trajectory (action, hazard) sequence drop the rec arm to
    **0.6472 ≈ base**, empirical **p = 0.0000** — the gain is
    *specifically* from the applied-trajectory **sequence/persistence**
    (L10's measured autocorrelation), **not** from trunk width/capacity.
  - This is non-redundant with L9 (window length) and L10 (feedforward
    scalar): the GRU carries the *sequence*, not a single snapshot.
- This prereg licenses **exactly one trial**: the **recurrent
  applied-trajectory hazard-history state** conditioning the hazard
  representation, against a retrained base control on the **same fresh
  partition**, with a frozen parent baseline.
- No other change is authorized (no prior-action window change, no
  combined reweighting, no calibrator change, no context-trunk change,
  no 2-step window, no pruning reorder).
- **Fresh partition requirement:** the discovery probes ran on
  PB21O-CAL (seed_offset 167,774,720). A publish-once trial must not be
  tuned against its own test partition. This trial therefore runs on a
  **fresh, disjoint** PB21S-CAL partition (seed_offset 167,774,976,
  contiguous after PB21O-CAL which ends at 167,774,976). The mechanism
  must **re-demonstrate** its +AUC gain on this fresh partition before the
  gate can pass; a mechanism that does not generalize to fresh data is a
  frozen negative for the generalization claim even if it passed on the
  discovery partition.

## 1. Parent (frozen)

- V2.1i seed42 exact-v3, byte-verified:
  - v3 result JSON SHA-256 `3f91921b7ca63148addd3204cdb1e96b130f26240f49277a269ac8f079b13936`
  - parent state-digest SHA-256 `2619b5b0a180bdd55296d3471d89725f4346110e74ef093f84c3068a32e6576d`
- The parent is read-only. Any drift in its state digest at trial start or
  end fails G4. No parent parameters are trained.

## 2. Single mechanism under test

`H_rec` — the hazard path with a **recurrent applied-trajectory
hazard-history state**:

- **Representation:** the hazard state trunk (frozen, parent init,
  LayerNorm) and the candidate-action embedding (frozen, parent init,
  LayerNorm) are unchanged from the base hazard path. A new channel is
  concatenated into the hazard outcome trunk input:
  - a 1-layer **GRU** (hidden **128**) over the **applied-trajectory
    sequence**: at each tick τ the GRU input is the concatenation
    `[applied-action-embedding(applied_ctrl[τ]) (dim 32),
    applied-hazard-event[τ] (dim 1)]` = 33 dims; the GRU state is reset to
    **zero at tick 0** and rolled forward over all 16 ticks (burn-in
    included — real history).
  - the **causal** GRU state at tick t−1 (the agent's committed past,
    **never** including tick t's own hazard event — that is the label
    being predicted) is the history channel.
  - hazard outcome trunk input: `cat([state(120), action(120),
    hist_state(128)])` = 368 dims → trunk(368→256, ReLU) → head(256→1).
- **Determinism:** the GRU + applied-action-embedding are initialized
  N(0,1) from **fixed generators** (zlib.crc32-stable per-parameter
  seeds; the builtin `hash(str)` is process-salted and MUST NOT be used
  — see L11 determinism fix). All OOF folds use deterministic seeded
  permutations.
- **Training schedule (frozen, bound to the V2.1i refinement contract):**
  identical to PB21N/PB21O trial #1 — `REFINEMENT_PASSES = 8`,
  `REFINEMENT_ROOT_BATCH_SIZE = 24`, `REFINEMENT_LEARNING_RATE = 1e-3`,
  `REFINEMENT_WEIGHT_DECAY = 1e-4`, `REFINEMENT_CLIP_NORM = 1.0`
  (512 optimizer steps per OOF fold). **No calibrator** (the raw OOF
  logits are scored directly; the binding constraint is ranking, per L8 —
  a post-hoc calibrator cannot exceed the ranking ceiling).

Arms (all on the fresh PB21S-CAL partition, upstream belief frozen
byte-exact, OOF 2-fold):
- `H_base` — the **retrained** base hazard path (2H trunk,
  `cat([state, action])`), retrained per OOF fold on the complementary
  fold with the frozen schedule. This is the control: it isolates the
  recurrent-history mechanism from "any retraining on this partition."
- `H_rec` — the recurrent applied-trajectory hazard-history path
  (2H + 128-dim GRU state), retrained per OOF fold on the complementary
  fold with the frozen schedule. The candidate.

`H_base` and `H_rec` share the exact same parent init (zero-extended
trunk; the GRU + applied-embedding are fresh N(0,1) for `H_rec` only),
so G2/G5 isolate the recurrent-history channel from everything else.

## 3. Data partition (fresh, disjoint)

- `PB21S-CAL`: `maze_chase`, split `train`, seed offset **167,774,976**,
  256 episodes (contiguous after PB21O-CAL, which ends at 167,774,976;
  collision-checked against the occupied-range registry before the run:
  PB21N-CAL 167,774,464–167,774,719, PB21O-CAL 167,774,720–167,774,975).
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
  (identical to PB21N/PB21O trial #1; 512 optimizer steps per fold).
- No calibrator (raw OOF logits scored directly).

## 5. Frozen gates (decision)

Let `P_rec`, `P_base` be the two arms' OOF **raw-logit** tables on
PB21S-CAL (no calibrator). Factual metrics are computed on the applied
action's logit per root.

- **G1 (ranking — PRIMARY, the L8 binding constraint):** factual
  aggregate **ROC-AUC** of `P_rec` across both OOF folds must be
  **strictly greater than** the factual aggregate ROC-AUC of `P_base`
  by at least **0.010** (one margin, both folds combined). Rationale:
  L8 established ranking as the binding constraint; L11 measured a
  +0.0307 AUC gain on the discovery partition; a +0.010 minimum is
  chosen to be **exceeding cross-fit noise** (L11 multi-seed SE ≈ 0.009)
  while not being so large that a real but modest effect is missed.
- **G2 (paired improvement vs retrained base):** paired `P_rec` vs
  `P_base`, factual, per root (BCE and Brier), episode-clustered
  bootstrap (10,000 resamples, TRIAL_SEED-bound); the one-sided 97.5%
  linear-quantile lower bound must be strictly positive for **both**
  folds and **both** loss types.
- **G3 (all-action control not degraded vs retrained base):** both folds:
  `|abs(all-action bias)_rec − abs(all-action bias)_base| ≤ 0.01`,
  `all-action ECE_rec − all-action ECE_base ≤ 0.01`, and worst
  per-action all-action ROC-AUC drop ≤ 0.05.
- **G4 (parent preservation):** parent state digest byte-exact before and
  after the trial; parent file hashes unchanged.
- **G5 (mechanism specificity — permutation control):** the applied-
  trajectory (action, hazard) sequence is permuted across episodes
  (1000 permutations, seeded); the rec arm's factual aggregate AUC with
  the permuted history must **drop to within 0.005 of `P_base`**'s AUC
  (i.e., the gain must be attributable specifically to the
  applied-trajectory sequence, not trunk width). If the permuted rec
  AUC remains ≥ 0.010 above base, the gain is from trunk capacity and
  the mechanism is not specific — the trial is a frozen negative for the
  recurrent-history claim even if G1–G3 pass.
- **G6 (determinism):** the `P_rec` OOF raw logit table re-derives
  byte-exactly from the saved evidence (re-fit the GRU + re-evaluate on
  a second pass); the `P_base` OOF raw logit table re-derives
  byte-exactly from a second retrain pass. **No process-salted hash may
  be used for any seed (L11 determinism fix: zlib.crc32 only).**

**PASS = G1 AND G2 AND G3 AND G4 AND G5 AND G6.** Any single gate failing
is a frozen negative for this trial; DEV stays sealed; no retry of this
namespace.

Edge-case guard (faithful extension, frozen with this prereg): any
per-action factual / all-action stratum without mixed outcomes is
excluded from the affected metric and recorded; this cannot change any
gate value on a fully scorable partition (see PB21N addendum §9).

## 6. Execution contract

- Create-only artifacts under `brain/runs/pb21s-recurrent-hazard-history/`:
  `…registration.json`, `…attempt.json`, `…evidence.npz`, `…json`.
  Publish exactly once; an existing attempt file forbids a rerun.
- No DEV / CPU-QUAL / PLAY-QUAL / TEST partition may be touched.
- The full repository test suite (including the PB21S test module) must
  pass before the result is published.
- Determinism per G6 (zlib.crc32-stable seeds; no process-salted hash).
- CPU-only, single-thread, deterministic, CUDA hidden
  (`CUDA_VISIBLE_DEVICES=-1`, OMP/MKL=1, torch threads=1).

## 7. Interpretation guardrails

- A pass is a **mechanism signal** [HYPOTHESIS-LEVEL OUTPUT]: it says the
  hazard representation benefits from a *recurrent* applied-trajectory
  hazard-history state — i.e., the applied-trajectory's *sequence/persistence*
  (not a single snapshot, not window length, not trunk width) carries
  hazard-ranking information that the base hazard path does not exploit.
  It does NOT make the model "fully working," nor a scaling / live-play /
  full-model claim.
- It licenses at most one further step: a fresh, separately
  preregistered trial in yet another fresh namespace that would
  integrate the recurrent-history hazard path into the production
  outcome model and re-run the full qualification battery against
  held-out DEV + CPU-QUAL. That is out of scope here.
- A negative is preserved verbatim and the mechanism returned to the
  architecture ideas ledger with its measured numbers.

## 8. Next step on PASS

On PASS, the next gate is a fresh PB21T-class integration/qualification
preregistration (new namespace) that would take the recurrent-history
hazard path into the production outcome model and re-run the full
qualification battery against held-out DEV + CPU-QUAL. That is out of
scope here.
