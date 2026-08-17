# Baseline suite extension: persistence and dense-communication ablations (2026-08-17)

Status: implementation only. No training run, release, or campaign registration
exists for these variants. The matched-baseline campaign remains blocked and
fail-closed on its historical 500-step pin; this change widens the suite it
will use when a campaign is preregistered.

## What was added

Two further PLAN.md §28/§29 controls, both exact-allocated-parameter
ablations of the routed thought-field reference (29,674,318 trainable
parameters each, identical state dicts, identical persistent-state bytes):

1. **`irene.thought_field.reset_slots.v1`** ("Thoughtlets with persistence
   removed"; §29 "Persistent versus reset thoughtlets").
   `ResetStateSlotBaseline` overrides `_refresh_thoughts` so incoming thought
   state is discarded every step and slots reseed from current sensors and
   belief. The lifecycle head only gates persistence, so its 1,155 parameters
   (384 × 3 + 3) are reported as architecturally disconnected, mirroring the
   isolated-slot precedent. The seed computation was extracted from
   `IreneBrainModel._refresh_thoughts` into `_seed_thoughts` with no change to
   reference behavior.
2. **`irene.thought_field.dense_routing.v1`** ("Thoughtlets with unrestricted
   dense communication"; §29 "Private/sparse versus unrestricted thought
   communication"). `DenseCommunicationSlotBaseline` builds the same tied
   `BrainCell` with `dense_routing=True`: softmax over every off-diagonal
   peer instead of sparse top-k. Parameter layout and count are exactly the
   reference's; nothing is disconnected. `dense_routing` defaults to `False`,
   so the reference and the isolated-slot control are bit-identical to before.
3. **`irene.thought_field.reactive.v1`** (§28 "Reactive policy with no
   memory"; §29 "Memory retained versus reset"). `ReactiveSlotBaseline`
   discards the entire incoming `BrainState` at the forward boundary, so
   belief, working memory, and thoughts all reseed every step — a pure
   reactive policy at the exact reference parameter budget with zero
   disconnected parameters.
4. **`irene.thought_field.serial_depth.v1`** (§28 "A deeper serial model
   with matched FLOPs"). `SerialDepthSlotBaseline` runs twelve untied serial
   blocks in one cognitive cycle, matching the reference's twelve tied block
   applications (four blocks × three cycles) while removing weight tying.
   Routing is permitted from the first block because serial depth replaces
   recurrence. This is a FLOP-matched, **not** parameter-matched control:
   untied depth costs 75,849,558 trainable parameters (+155.6%), and
   single-cycle execution yields two anytime exits instead of four. Its
   comparison regime stays `verified: false` until a device calibration
   artifact exists, same as the same-width monolith.

Each variant has its own factory
(`build_thesis_reset_slots_model`, `build_thesis_dense_routing_model`,
`build_thesis_reactive_model`, `build_thesis_serial_depth_model`) and
Stage-A-matched recipe (`baseline-stagea-reset-slots.toml`,
`baseline-stagea-dense-communication.toml`, `baseline-stagea-reactive.toml`,
`baseline-stagea-serial-depth.toml`),
byte-identical to the no-communication recipe except run name and factory.

## Intentional source freeze

The change touched matched implementation files (`torch_model.py`,
`brain_cell.py`, `baselines.py`, `factory.py`) and added four manifest
variants, so the architecture manifest was regenerated deliberately:

- Previous live digest: `8a41131e4284b9501864e4f30b4b518f161b2f884043de1255f7d251180ecf4d`
- New live digest: `78ba9cfc1eb56782609f586ac7f3ef393a7fb6ddd6f017546861c3e92d0bb0e2`
- `MANIFEST_IDENTITY=new_comparison`

The historical first-matched campaign pin `52bba6a9…` and every earlier live
digest stay bound to their own trees; nothing is rewritten.

## Verification

- `tests/test_matched_baselines.py`: 10 tests, including the new
  `test_reset_slots_ignore_incoming_thought_state` (incoming thought state and
  ages provably cannot influence any output; allocated parameters exactly
  match the reference; 1,155 disconnected lifecycle parameters),
  `test_dense_routing_reaches_every_other_slot` (cycle-1+ routing diagnostics
  cover exactly the K−1 off-diagonal peers per slot with weights summing to
  one; allocated parameters exactly match),
  `test_reactive_control_ignores_every_incoming_state_field` (a poisoned
  belief/working-memory/thought state produces bit-identical outputs to a
  fresh state), and
  `test_serial_depth_matches_block_applications_without_tying` (untied serial
  depth adds exactly one tied-block parameter set per removed cycle, routes
  from the first block, and yields two anytime exits).
- `tests/test_multithought_core.py`,
  `tests/test_first_matched_campaign_preregistration.py`,
  `tests/test_rcq_v3_recipe.py`, `tests/test_rcq_v3_lineage.py`: green — the
  historical campaign pin and both v3 lineages are unaffected.
- Full play-safe suite: green (runner exit 0).

## What remains before these variants produce evidence

A preregistered multi-seed campaign (MULTISEED_COMPARISON_PROTOCOL): pinned
seed set, per-variant effective configs, execution manifest, and the
registered criterion — then equal-parameter Regime A training on the Spark.
The latency/FLOP Regime B still requires a device calibration artifact.
