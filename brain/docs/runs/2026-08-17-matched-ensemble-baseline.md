# Matched-cost independent ensemble baseline (2026-08-17)

Status: implementation and registration only. Adds the PLAN §28 item-12
control to the matched baseline suite. No training runs use it yet; the
multi-seed campaign remains fail-closed on its historical pin.

## What was added

**`MatchedEnsembleBaseline`** (`irene.thought_field.independent_ensemble.v1`)
— the matched-cost independent ensemble as an experimental control for the
thought-field claim. Four members each own eight of the 32 thought slots
with fully untied parameters:

- New `EnsembleMemberBlock` (in `model/brain_cell.py`): a lean slot-update
  block carrying only thought cross-attention and the continuous-time
  blend — no routing, belief-maintenance, or workspace-write parameters.
  Its context mirrors the reference block minus the routed-message token.
- New `EnsembleBrainCell`: chunks the thought axis (and per-slot retrieved
  memory) into four disjoint member chunks, runs each member's untied
  stack, and concatenates. Belief and working memory pass through
  unchanged; members never route or write the workspace. One empty routing
  diagnostic per block layer preserves the reference's result arity, and
  weights stay tied across cognitive cycles within each member, exactly
  like the reference ties its cell across cycles.
- Width selected by exact allocated-parameter enumeration over member
  counts {2, 4, 8, 16} and widths: **4 members at core width 352** lands at
  29,459,914 trainable parameters, 0.72% under the reference's 29,674,318 —
  the nearest allocation inside the 1% manifest tolerance
  (`MATCHED_ENSEMBLE_WIDTH` in `training/factory.py`, same discipline as
  the parameter-matched monolith).
- Recipe `configs/training/baseline-stagea-matched-ensemble.toml` is a
  verbatim copy of the reference recipe except name and factory, passing
  the objective scan; the variant joins the manifest's `parameter_matched`
  fairness regime, which still verifies.
- Checked-in manifest regenerated: digest `78ba9cfc…` → `d19d09bf…`.

## Identity-recorded limitations

Members share perception, belief, actuator, and prediction heads — only
thought-slot updates are independent; belief and working memory are
maintained by the shared ingest pathway, not per-block attention. The
control tests whether untied independent subnetworks at matched cost
explain the thought-field's results, not whether a production ensemble
serving design does.

## Verification

`tests/test_matched_baselines.py` grows to 11 tests. New
`test_ensemble_members_are_fully_independent` perturbs member 0's weights
and proves only member 0's slots change while belief and working memory are
untouched, and pins the diagnostics arity. The identity-uniqueness,
fairness-regime, and checked-in-manifest tests now cover the ninth
variant. The full play-safe suite (462 tests) exits 0.

## Remaining §28 gaps

Fixed multi-horizon no-persistence heads (item 8), the recurrent latent
world-model actor (item 13), and the task specialist (item 14) remain
unimplemented. Item 5 (standard recurrent transformer controller) is
likewise absent from the manifest.
