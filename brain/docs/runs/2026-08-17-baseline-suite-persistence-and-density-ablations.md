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

Each variant has its own factory
(`build_thesis_reset_slots_model`, `build_thesis_dense_routing_model`) and
Stage-A-matched recipe (`baseline-stagea-reset-slots.toml`,
`baseline-stagea-dense-communication.toml`), byte-identical to the
no-communication recipe except run name and factory.

## Intentional source freeze

The change touched matched implementation files (`torch_model.py`,
`brain_cell.py`, `baselines.py`, `factory.py`) and added two manifest
variants, so the architecture manifest was regenerated deliberately:

- Previous live digest: `eb46988b178d593da6f98f6b99273fd2e791dd62e35f9bcef719c03958d4bb2a`
- New live digest: `f4e9b355299290b7cbf352b1cfb861fd771373cbb0d9707e92c281b998084711`
- `MANIFEST_IDENTITY=new_comparison`; source bundle
  `6b476d011adc47da2835c7bd3d97af60f5f12b45345cd573af3962ab0bff2d00`

The historical first-matched campaign pin `52bba6a9…` and every earlier live
digest stay bound to their own trees; nothing is rewritten.

## Verification

- `tests/test_matched_baselines.py`: 8 tests, including the new
  `test_reset_slots_ignore_incoming_thought_state` (incoming thought state and
  ages provably cannot influence any output; allocated parameters exactly
  match the reference; 1,155 disconnected lifecycle parameters) and
  `test_dense_routing_reaches_every_other_slot` (cycle-1+ routing diagnostics
  cover exactly the K−1 off-diagonal peers per slot with weights summing to
  one; allocated parameters exactly match).
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
