# PB21N — Prior-Action-Conditioned Hazard Representation

Status: **PREREGISTRATION (frozen before any run).** Fresh namespace.
Classification (planned): `fresh_hazard_prior_action_single_change`.

Date frozen: 2026-08-25
Sole source of the single change: the frozen nomination from
`2026-08-25-pb21m-posthoc-mechanism-audit-v1` (Arm B), which is
`post_hoc_consumed_data_nonqualifying` and authorizes exactly this one
fresh trial. It does not itself run anything.

## 1. Purpose and scope

PB21M's terminal CAL negative showed the sealed BAL-UPMIX scorer's
factual-domain miscalibration is **organized by the prior applied action**
(Arm B of the post-hoc audit: 18/18 cells with ≥3 prior-action strata at
factual |bias| ≥ 0.03 AND ECE ≥ 0.05; specificity range 0.3253 vs 0.3021
p95). This prereg tests the single nominated mechanism:

> **condition the hazard representation on the prior applied action
> (one-step lag, burn-in-aware).**

It is a **single change**. It is NOT combined with the Arm A calibration
change (MIX35) — the frozen nomination rule forbids combining the two in
one trial. No other mechanism, re-weighting, scorer retrain, or data
change is in scope.

## 2. The single change — exact specification

The hazard path is `dedicated_stopgrad_v1`: it reads `context.detach()`
(the frozen upstream belief) plus the *current* proposed-action embedding,
and cannot alter next-state, reward, decision, belief, or thought
parameters. The change adds **one** new input to that path:

1. Add a `hazard_prior_action_embedding: nn.Embedding(actions, hidden)` to
   the all-action outcome model (a NEW parameter; all other parameters
   unchanged at init).
2. In the hazard forward path, concatenate the prior applied action's
   embedding: `cat([hazard_state, hazard_action, hazard_prior_action])`
   feeding `hazard_outcome_trunk` → `hazard_head` → per-action affine
   calibration (scale/bias buffers, unchanged contract).
3. **Input:** for each recorded root, the prior applied action is the
   action physically applied one tick earlier, burn-in-aware: the first
   recorded root of an episode takes the burn-in tick-3 applied action
   (collector semantics — byte-verified in the audit's B1 replay). Action
   IDs use the frozen semantic 5-way class
   (`control_action_class`: idle/W/A/S/D).
4. **Trainable set:** exactly the hazard path
   (`hazard_state_trunk.*`, `hazard_action_embedding.*`,
   `hazard_prior_action_embedding.*`, `hazard_outcome_trunk.*`,
   `hazard_head.*`), mirroring `refine_hazard_path`'s frozen
   prefix-set — **plus** the new prior-action embedding. The upstream
   belief/scorer, next-latent, reward, decision, and thought parameters
   are frozen byte-exact at the V2.1i frozen scorer.
5. **Preservation gate (hard):** after training, every non-hazard-path
   tensor must remain byte-identical (`all_other_tensors_byte_equal`).
   Any drift outside the hazard prefix set aborts the run as an
   infrastructure failure.

## 3. Data partition (fresh, disjoint — frozen)

A fresh, disjoint TRAIN-CAL partition is required (PB21M's C0A/C0B/C1A/
C1B/C2A/C2B are consumed by the parent and may not be reused for this
trial's fit; see §5). New partition spec:

- `PB21N-CAL`: `DatasetSplit.TRAIN`, `seed_offset = 167_774_464`
  (immediately after PB21M's C2B end at 167_774_464), `episodes = 256`,
  giving 256 × 12 recorded roots = 3072 roots (same per-episode recorded
  depth as PB21M: `sequence_length` − `burn_in_steps`).
- All-action intervention dataset, frozen:
  `counterfactual_targets = all_actions_v1`,
  `behavior_policy = balanced_intervention_v1`,
  `behavior_intervention_rate = 0.5`.
- Prior-action column is data-derived from the deterministic dataset
  replay (no model re-inference required for the input; only the hazard
  path is fitted).
- **DEV** (`validation[184_549_376, 184_550_144)`, 768 episodes) remains
  the held-out gate partition and is sealed: it is read only for the
  final frozen gate evaluation, never for stopping, selection, or
  fitting. No DEV-open receipt is created until the gate passes.

## 4. Training contract (frozen)

- Hazard-path-only AdamW, mirroring `refine_hazard_path`:
  - `learning_rate`, `weight_decay`, `clip_norm`, root batch size, and
    pass count are bound to the `HAZARD_REFINEMENT_*` constants from
    `v21i_development_runner` (no new hyperparameters introduced beyond
    the single mechanism).
  - Loss: unweighted `binary_cross_entropy_with_logits` over all five
    actions (aligned by `table.action_ids`), on the frozen upstream
    belief context.
  - Deterministic complete root permutation without replacement per pass;
    `torch.Generator` seeded with the frozen trial seed.
  - Per-pass `maximum_preclip_gradient_norm` and `root_permutation_sha256`
    recorded; non-finite loss or gradient aborts.
- CPU only; `CUDA_VISIBLE_DEVICES=-1`; single intra-/inter-op thread;
  `OMP_NUM_THREADS=1`; deterministic torch mode; `run_provenance`
  metadata attached.

## 5. Frozen gates (decision)

Let `H_prior` be the prior-action-conditioned hazard path's cross-fit OOF
all-action hazard probabilities on `PB21N-CAL`, and `H_base` the V2.1i
frozen baseline hazard path re-evaluated on the same roots (no retraining).
Cross-fit is exactly as the audit's Arm A: fit the hazard path on OOF fold 0,
predict fold 1, and vice-versa; OOF table in canonical order (fold-0 rows
then fold-1 rows), one scorer (the single frozen BAL-UPMIX upstream belief),
two OOF tables (scorer × fold). The "table" unit is therefore the OOF fold
table, not PB21M's 18-head scorer×cell structure.

**Control arm (mechanism isolation, required by the hostile-scientist
standard).** The candidate `H_prior` is *retrained on fresh CAL data*, so
"retraining on new data" is a confound that must be isolated from the
prior-action mechanism. Define `H_retrain`: the parent's exact 2H hazard
path (no prior-action channel), retrained on the *same* OOF folds with the
*same* schedule as `H_prior`. Then:
- `H_base` — frozen parent, no retrain, no prior-action.
- `H_retrain` — retrained hazard path, no prior-action channel (controls for
  data-shift / re-optimization on the fresh partition).
- `H_prior` — retrained hazard path + prior-action channel (the nominated
  single change).
The mechanism is isolated as `H_prior` vs `H_retrain`; the practical claim
is `H_prior` vs `H_base`.

- **G1 (factual absolute):** on `H_prior`, factual aggregate
  |calibration bias| ≤ 0.05 AND factual ECE ≤ 0.05 in both OOF tables.
  (The PB21M failure was exactly this gate.)
- **G2 (paired factual improvement vs baseline):** paired `H_prior` vs
  `H_base` per root, episode-clustered: point and one-sided 97.5%
  linear-quantile LCB both positive for factual BCE and Brier, and both
  held-out fold points positive.
- **G3 (all-action control not degraded):** on all five actions,
  |all-action calibration bias| of `H_prior` must not exceed `H_base`'s
  by more than 0.01, and all-action ECE not worse by more than 0.01, in
  every OOF table; no factual per-action ROC-AUC drop > 0.05.
- **G4 (preservation):** the frozen parent's `model_state_dict` is
  byte-identical before/after the whole trial (the trial only ever reads the
  parent; all fitting is on standalone hazard modules). The grafted
  prior-action embedding and the zero-extended hazard trunk are the ONLY
  permitted new/changed tensors in the candidate.
- **G5 (specificity control, carried from the audit):** the prior-action
  effect must not be an artifact of an arbitrary label — a fixed
  within-cluster prior-label shuffle (30 deterministic permutations,
  seed-bound) must produce a lower factual-structure signal than the
  observed prior-action conditioning (same B3 test as the audit).
- **G6 (mechanism isolation, control arm):** paired `H_prior` vs
  `H_retrain` per root, episode-clustered: the one-sided 97.5% linear-
  quantile LCB for factual BCE improvement (`H_prior` − `H_retrain`) must
  be strictly positive. This proves the prior-action channel contributes
  beyond mere retraining on the fresh partition. If G6 fails, the measured
  gain is attributed to retraining/data-shift, NOT to the prior-action
  mechanism, and the trial is a frozen negative for the mechanism claim
  even if G1–G3 pass.

**PASS = G1 AND G2 AND G3 AND G4 AND G5 AND G6.** Any single gate failing is
a frozen negative for this trial; DEV stays sealed; no retry of this
namespace.

## 6. Execution contract

- Create-only artifacts under `brain/runs/pb21n-hazard-prior-action/`:
  `…registration.json`, `…attempt.json`, `…evidence.npz`, `…json`.
  Publish exactly once; an existing attempt file forbids a rerun.
- No DEV, CPU-QUAL, PLAY-QUAL, or TEST partition may be constructed,
  loaded, or hashed except the sealed DEV read for the final gate.
- Peak working-set / RAM checks mirror the PB21M resource guard.
- The full repository test suite (including the PB21N test module) must
  pass before the result is published.
- Determinism: re-running the hazard-path fit must reproduce the hazard
  state SHA-256 byte-exactly.

## 7. Interpretation guardrails

- A pass is a **mechanism signal** [HYPOTHESIS-LEVEL OUTPUT]: it says the
  prior-action conditioning removes the measured factual miscalibration on
  the consumed/ fresh CAL data with the all-action control intact. It does
  NOT by itself make the model "fully working," nor a scaling / live-play /
  human-speed / full-model claim.
- It licenses at most one further step: a fresh, separately
  preregistered scaling/qualification trial in yet another fresh
  namespace. It authorizes no run by itself.
- A negative is preserved verbatim and the mechanism returned to the
  architecture ideas ledger with its measured numbers.

## 8. Next step on PASS

On PASS, the next gate is a fresh PB21O-class scaling/qualification
preregistration (new namespace) that would take the prior-action
hazard path into full-model qualification against held-out DEV +
CPU-QUAL. That is out of scope here.

## 9. Addendum (appended before first execution; frozen with the run)

Edge-case guard, faithful to the frozen gates: the canonical
`binary_probability_metrics` helper raises when a population has no
mixed outcomes (all-0 or all-1 targets). The audit's B3 did not
encounter this on its consumed CAL strata; a fresh partition might.
Any per-action / (action, prior) stratum without mixed outcomes is
therefore **excluded from scoring and recorded** as
`skipped_unscorable_strata` (G5) / `scorable: false` (G1 per-action)
/ recorded-and-skipped (G3 AUC loop). This cannot change the sign or
magnitude of any gate value on a fully scorable partition, and it
only prevents a hard crash on an unscorable one. All other gate
definitions, thresholds, seeds, and decision rules are unchanged.
