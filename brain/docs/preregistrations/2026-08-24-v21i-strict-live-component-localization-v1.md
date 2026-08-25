# V2.1i strict-live component-localization diagnostic v1

Status: implementation-ready preregistration; development diagnostic only.

This diagnostic follows the exact strict-live v2 result whose SHA-256 is
`2ab123d5d2d42a1d14327bcd76710aa93538564f8897c8c2431a1133127bd2aa`.
That result found that a capacity-matched head on the full live updater tuple
ranked DEV hazards substantially better than a head on the post-update belief.
The present run localizes the smallest live component set that preserves that
ranking evidence. It cannot produce a candidate, checkpoint, qualification
claim, or reusable production DEV result.

## Frozen evidence and namespace

The run imports the strict-live v2 collector and helpers and cryptographically
binds its script, preregistration, test, result, registration, and source bundle.
The new receipt recomputes and embeds the complete strict-live source bundle,
including the nested production-v3 model, data, metric, development-runner, and
provenance dependencies; trusting only the SHA recorded in the old JSON is not
sufficient.
It rematerializes only seed-42 v3 TRAIN-FIT and DEV through RGB plus the
controller-owned prior applied action. `actual_reward=None` and
`actual_hazard=None`. TRAIN-CAL, CPU-QUAL, and TEST are not constructed. The
parent must remain byte-identical. CUDA is hidden and PyTorch uses one CPU
thread.

The DEV tape may be collected up front by the frozen strict-live collector, but
all heads are fitted on TRAIN-FIT before any DEV prediction or metric is
computed. DEV cannot affect training, stopping, hyperparameters, arm
definitions, inference, or selection order.

## Exact 366-D feature layout

Every arm uses the existing `CapacityHazardProbe`: a 366-to-240 state trunk,
240-D action embedding, two 240-D outcome layers, and one scalar output. It has
exactly 262,801 trainable parameters.

The registered slots are:

- `[0:120]`: `B` (old belief) or `N` (new/post-update belief)
- `[120:240]`: `Z` (current RGB encoder latent)
- `[240:360]`: `E` (live-only cognitive prediction error)
- `[360:365]`: `A` (internally owned prior applied-action one-hot)
- `[365:366]`: `P` (has-pending indicator)

`P` must be exactly one at every post-burn-in root. It is therefore
unidentifiable. Every reduced arm sets `P` to exactly zero; only the exact full
updater control retains it.

The eleven fixed arms are:

- `N`: `[new_belief, 0, 0, 0, 0]`, exact post-belief negative control
- `U`: `[old_belief, latent, PE, prior_action_onehot, P]`, exact updater byte control
- `U0`: `[old_belief, latent, PE, prior_action_onehot, 0]`, pending-zero full-updater inferential comparator
- `B`, `Z`, `E`, `A`: one registered component in its native slot, all other slots zero
- `BZ`, `BE`, `ZE`: the three dense-component pairs in their native slots
- `BZE`: all three dense components, with `A=P=0`

`A` is diagnostic-only because it can reveal a behavior-policy/history shortcut.
`BZE` distinguishes dense triple sufficiency from dependence on action or the
otherwise constant pending indicator. `U0` prevents noninferiority from
confounding omitted information with the constant `P` column's intercept or
optimization effect. Exact `U` remains the mandatory byte-reproduction control
and also remains a direct noninferiority reference; `U0` supplements rather than
replaces it.

Forbidden model inputs include actual reward/hazard, teacher action, event
target, intervention assignment, environment seed, episode/root identifiers,
and `observation.previous_control`. Targets and provenance never enter features.

## Fixed optimization

Every arm receives an independently instantiated but byte-identical head at
seed 30042. Every initial state must have exact SHA-256
`4a508e6450c509e877e12cfb03d0532d4c396df25292cd7951d2e18c6005fec5`.
All arms receive the same complete TRAIN-FIT root permutations at seed 30042.
TRAIN-FIT has 1,536 roots. Each arm trains for exactly 32 complete passes,
64 optimizer steps per pass, and 2,048 steps total with batch size 24. The
optimizer is fresh AdamW with learning rate `1e-3`, weight decay `1e-4`, global
gradient clipping at 1.0, and unweighted all-five-action BCE. There is no early
stopping, calibration, or observed-result adaptation.
There is no retry. The implementation accepts only the canonical run ID and
paths below; alternate receipt/result paths are rejected before publication.
A failed execution remains the sole result for this preregistration.

- upstream: `brain/runs/v21i-development/2026-08-24-seed42-v3.json`
- registration: `brain/runs/v21i-diagnostics/2026-08-24-seed42-v3-strict-live-component-localization-v1.registration.json`
- result: `brain/runs/v21i-diagnostics/2026-08-24-seed42-v3-strict-live-component-localization-v1.json`

Every projected feature table must be finite, C-contiguous float32. Every
masked slot must contain canonical positive zero (`+0.0`, with no sign bit).

## Mandatory byte controls

`N` must exactly reproduce prior C-belief:

- final state: `dd3aa2792773c3ecb8ae266d0ac3dc9927cd76ead35bd2c33e4d653fe3ea13c1`
- TRAIN-FIT probabilities: `115956b79b416f7221c4271511b87e51f12d467358a7da4523b79881fc51e020`
- DEV probabilities: `3e6ebc9babfbf2edd06b6b797becdaa3d3b1d3cc84934b5e8de976c771e4c508`
- DEV aggregate ROC AUC: `0.6040297854619737`

`U` must exactly reproduce prior C-updater:

- final state: `dc9a854b596066827d6b0ec576c035520d5c39f0901eebbe8d3cbb667258b00b`
- TRAIN-FIT probabilities: `f1f775e8363cfb7f4d9ec45d9f91d405f0a3499638ec104a7d1316123653d8cd`
- DEV probabilities: `6f4a3e8578b4294d5a48adf9b33644cd91b88bf0e8923b9b676ee635eeac5647`
- DEV aggregate ROC AUC: `0.7321055105872505`

The exact TRAIN-FIT-fitted action-prior DEV AUC remains
`0.5380051490468308`. Comparisons use exact equality with no tolerance. If any
control fails, the only diagnosis is `control_reproduction_failure`; scientific
interpretation and architecture selection are forbidden. The run terminates
before computing or exposing any reduced-arm metric or bootstrap comparison.

## Frozen ranking and simultaneous inference

An arm first passes the unchanged ranking gate only if aggregate DEV ROC AUC is
at least 0.65, gain over the action prior is at least 0.10, and every action ROC
AUC is at least 0.60.

Before resampling, the run must assert the exact DEV geometry: 64 episode
clusters, exactly 12 post-burn-in roots per cluster, and five counterfactual
action branches per root.

One shared matrix of 10,000 paired DEV episode-cluster bootstrap samples is
drawn with replacement using seed 32042. Its exact index-array digest is
recorded. For each of the eight reduced arms, the same samples compute arm-minus-
`N`, arm-minus-`U0`, and arm-minus-`U` deltas. A descriptive paired
`U`-minus-`U0` 95% interval is also recorded but is not an eligibility gate.
All deltas are oriented left minus right. Every quantile uses NumPy's fixed
`linear` method. Descriptive two-sided 95% percentile intervals are recorded for
every comparison.

`U0` must first pass the unchanged ranking gate. Subject to that global
precondition, a reduced arm is eligible only when all three statements hold:

1. its unchanged ranking gate passes;
2. its Bonferroni simultaneous one-sided lower arm-minus-`N` bound is greater than zero;
3. its exploratory Bonferroni lower arm-minus-`U0` bound is greater than `-0.025`;
4. its exploratory Bonferroni lower arm-minus-`U` bound is greater than `-0.025`.

The fixed exploratory reference alpha is 0.05 over eight reduced arms, so the
lower percentile is `0.05 / 8 = 0.00625`. The three comparison limbs inside
each arm form an intersection-union rule and
are not separately alpha-split. Because DEV was
already consumed by the preceding representation diagnostic, these are
deterministic multiplicity-adjusted engineering bounds, not confirmatory
nominal-familywise-error evidence. Fresh preregistered DEV is required for any
confirmation.

The `0.025` noninferiority margin is also exploratory, not scientifically
validated. It retains approximately 80.5% of the previously observed exact
`U`-minus-`N` AUC gain. It is a fixed engineering screen for the next
architecture experiment, not an effect-size claim.

## Fixed interpretation and selection

The result only nominates a component set for a fresh production-form probe;
it does not directly select a production architecture. Nomination excludes
`A`. It prefers any eligible single dense component before a pair, then the
dense triple, in the immutable order:

`Z > E > B > ZE > BZ > BE > BZE`

Observed AUC never breaks a tie. If only `A` is eligible, report
`behavior_history_shortcut_supported` and make no architecture change. If no
reduced arm is eligible while `U0` passes, report
`multi_component_dependence_unresolved`. Otherwise report
`no_architecture_change_supported`. A selected arm supports only the next
production-form experiment; it does not qualify a model or support a direct
architecture change. The consumed DEV split cannot be reused to claim
production qualification.

If exact `U` passes its byte/ranking controls but `U0` fails the ranking gate,
report `pending_intercept_optimization_confounded`, forbid selection, and do not
interpret reduced-arm failures. That result means the constant pending column's
optimization/intercept effect has not been cleanly separated from information
removal.

The native-slot design has a known interpretation limit: `B`, `Z`, `E`, and
their combinations activate different fixed random blocks of the first layer.
Consequently their ordering is initialization/slot-confounded. This diagnostic
may nominate a feature set, but cannot establish unique component causality.
The nominated production-form probe must use multiple preregistered fixed
initializations and fresh seeds.

`E` also is not action-independent: cognitive prediction error includes error
against the pending predicted-next latent produced for the prior selected
action. Therefore `BZE` versus `U0` isolates only the incremental explicit prior
action one-hot `A`; it does not isolate all action dependence.

## Publication and failure semantics

Registration and result files are create-only. Success and failure artifacts
must both state that candidate publication is forbidden and no checkpoint was
emitted. Any execution exception produces a create-only
`diagnostic_run_failed_not_candidate` receipt and is re-raised.
The frozen parent-state digest is asserted equal immediately before any success
artifact is published; recording a false boolean is not sufficient.
