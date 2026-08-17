# RCQ-v3 recipe options implemented behind configuration (2026-08-17)

Status: implementation only. No v3 registration, release, pin, or training run
exists. The v2 registration, candidate, and ranges are untouched and terminal.
The values in `dgx-rcq-v3-reference-candidate.toml` are review proposals
awaiting owner sign-off, not a frozen recipe.

## What was built

Three recipe options from the v2 failure diagnosis
([2026-08-17-rcq-v2-entry-gate-failure-diagnosis.md](./2026-08-17-rcq-v2-entry-gate-failure-diagnosis.md))
are now implemented, config-gated, and off by default:

1. **Deadzone hinge (D1-a).** `objective.continuous_deadzone_hinge_weight`
   and `..._margin` (default 0.0 / 0.04) add
   `weight * mean(relu(|continuous| - margin))` over the eleven continuous
   channels at every anytime exit, inside the action loss, so stage 2's zero
   action weight disables it.
2. **Structural deadzone squash (D1-b).**
   `objective.continuous_output_squash = "deadzone_tanh"` makes
   `build_thesis_model` construct the actuator readout with
   `ActuatorQuerySpec(continuous_squash="deadzone_tanh")`, bounding mouse,
   scroll, and gamepad-axis outputs to `0.046875 * tanh(raw / 0.046875)`.
   The bound is exactly representable in float32 and bfloat16 and strictly
   inside the frozen inclusive ±0.05 deadzone, so saturated outputs cannot
   round above the gate threshold in any evaluation precision. Default
   `none` preserves the historical readout bit-for-bit.
3. **Opposite-key pair penalty (D2-c).**
   `objective.opposite_key_pair_weight` (default 0.0) adds
   `weight * mean(sigmoid(W) * sigmoid(S) + sigmoid(A) * sigmoid(D))` from the
   256 keyboard logits at every exit. The 296-button geometry is unchanged;
   D2-b (categorical movement geometry) was not implemented.

Configuration plumbing keeps every historical identity exact: the four new
fields are optional TOML keys and are omitted from the canonical JSON while at
their defaults. The registered v2 config still hashes to
`b184361881b52189be78dce105ecc59ab52fa15551d63a2a7662a9dd06619366`; this is
pinned by `tests/test_rcq_v3_recipe.py`. The trusted evaluator's objective
construction passes the same fields so a future v3 checkpoint-bound metrics
replay compares like with like.

## Verification

- `tests/test_rcq_v3_recipe.py`: 12 tests covering hash preservation,
  validation, hinge zero/positive/gradient behavior, pair-penalty
  co-activation specificity, and squash bounds in float32 and bfloat16.
- Full play-safe suite: green (runner exit 0, one expected POSIX skip).
- `tests/test_multithought_core.py` was updated for the default-omission
  canonical form and now exempts the unregistered v3 candidate TOML from the
  all-defaults objective assertion.

## Intentional source freeze

This change altered files in the matched-baseline set, so the architecture
manifest was regenerated deliberately:

- Previous live digest: `30d4c119795a3c967e0f1251787cf5fb7effce8d02b814224b4383c40d783fc7`
- New live digest: `5decb402bc9ba68a8b8a80317d09f05458b4aa9584dab81a9d3b8575409f49e7`
- `MANIFEST_IDENTITY=new_comparison`; source bundle
  `883e286f54b56993c765e1668f20440f98f1603408a84b135811fe6d086e4399`

Per the operator guide, a new digest is a new comparison identity: historical
Stage A/RCQ-v2 evidence stays bound to the old tree, and no old claim is
silently rewritten. Any v3 qualification hashes this (or a later) tree into
its own new release and registration.

## What remains before any v3 training

1. Owner sign-off on the candidate recipe values and the D3–D5 decisions in
   [../ROADMAP_TO_PACMAN.md](../ROADMAP_TO_PACMAN.md) (budget, seed, final
   ranges, thresholds).
2. New qualification machinery: a new create-once registration id/path, run
   id, and pinned wrapper support. The existing RCQ wrappers correctly refuse
   everything except the frozen v2 identities.
3. Only then: freeze, register, release, pin, smoke, canary, train — in that
   order, under the same terminal-failure discipline as v2.
