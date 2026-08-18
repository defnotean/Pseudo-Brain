# B2: the latent world-model actor control in its own recipe family (2026-08-18)

Status: implementation. Local CPU-only verification; no training run, no
accelerator, no sealed data. Implements preregistered control **B2** from
[2026-08-17-remaining-baseline-controls-preregistration.md](2026-08-17-remaining-baseline-controls-preregistration.md)
under the owner's explicit 2026-08-18 go.

## What was added

**`irene.world_model_actor.gru_latent.v1`** (`LatentWorldModelActor`,
`model/world_model_actor.py`). The parameter-matched monolithic GRU trunk
plus a small learned latent transition model: an action embedder
(307→64), a residual two-layer transition (396+64→192→396), and a
bottleneck decoder (396→64→396) back to sensor-encoding space. The heads
add exactly 235,800 trainable parameters over the matched trunk for
29,879,700 total — 0.69% above the reference's 29,674,318, inside the
preregistered 1% band and verified by exact allocated-parameter
enumeration. `roll_sensor_embedding(latent, actions)` rolls the latent
through the transition, teacher-forced on a recorded action sequence,
with fail-closed shape checks.

**`LatentRolloutObjective`** (`training/world_model_objective.py`). The
world-model actor's defining feature cannot be expressed by the shared
slot-suite objective — a direct head regressed onto future sensor
encodings — so this variant has its own objective terms and does not ride
on the slot-suite fairness manifest. The shared objective's per-step
horizon world-loss computation was first refactored into an overridable
method (`ThoughtFieldObjective._world_horizon_losses`, behavior-preserving:
the matched-baseline, multi-horizon, and solver suites pass unchanged).
The subclass overrides exactly that surface: for each supported horizon
`k` of the window (the B1 window rule, {1, 2, 4} on the registered
length-8, burn-in-2 campaign window), the GRU latent at the current step
is rolled `k` times through the transition model on the recorded action
targets (`transitions[time_index + j].action_target`, j < k), decoded,
and scored against the frozen pixel-encoder encoding of the frame `k`
steps ahead. Every other objective term — action, value, diversity, and
all diagnostics including the collapse metrics — is inherited unchanged,
so the control stays metric-comparable with the slot suite. The h1
prediction error keeps the `[batch, thoughtlets]` diagnostics contract
with the actor's single latent.

**Training wiring.** `train.py` now resolves a fail-closed
`training_objective_class_path` hook on the model (same importlib
discipline as the model factory); models without the attribute keep the
shared `ThoughtFieldObjective` exactly as before. The actor's recipe is
`configs/training/baseline-stagea-world-model-actor.toml`, a
normalization-identical copy of the parameter-matched monolithic recipe
with only the run name and model factory changed
(`build_thesis_world_model_actor_model`).

**Own manifest.** `scripts/build_world_model_actor_manifest.py` writes
`configs/world-model-actor-manifest.json` with the same canonical-JSON
discipline as the slot-suite manifest: variant identity and identity
SHA-256, exact allocated parameter counts and the delta fraction against
the freshly rebuilt reference, the recipe's raw and canonical-config
SHA-256, the resolved objective class path, source-file digests for the
five files that define the family (`world_model_actor.py`,
`world_model_objective.py`, `factory.py`, `objective.py`, `train.py`),
and a fail-closed tolerance gate that refuses to emit a manifest if the
actor ever drifts outside the 1% band. Checked-in digest `c207401f…`.

## Verification

`tests/test_world_model_actor.py` (7 tests): identity and objective hook
declaration; thesis-factory parameter accounting (actor inside the 1%
band; actor-minus-heads exactly equals the parameter-matched monolith,
with identical trunk state-dict keys); rollout shape, determinism, and
fail-closed shape errors; a finite multi-horizon loss on a real length-8
MovingShapes batch with per-horizon metrics whose even split equals the
combined world loss, inherited metrics intact, and gradients reaching the
transition head; fail-closed rejection of a model without
`roll_sensor_embedding`; checked-in manifest self-consistency (manifest
SHA-256 recomputation, per-file digests, recipe digests, tolerance,
identity SHA-256); and the train.py import-surface resolution path.

The slot-suite architecture manifest was regenerated for the `objective.py`
refactor and the `factory.py` addition (digest `eda3cf38…`). The full
play-safe suite exits 0 (558 tests).

## Notes for the campaign

- B2 is a recipe family of one: its rollout world loss is
  teacher-forced on recorded actions, **not** imagination-based policy
  learning; that limitation is pinned in the variant identity.
- The recipe is a campaign candidate like the other baseline recipes:
  DGX-scale execution remains owner-run through the bounded workflow.
- Smoke-scale comparisons should now include this variant;
  `scripts/compare_baselines_smoke.py` covers the slot-suite registry and
  can be extended to recipe-family manifests as a follow-up.
- B3 (task specialist) is implemented at smoke scale on the existing lazy
  datasets; campaign-scale materialization remains DGX.
