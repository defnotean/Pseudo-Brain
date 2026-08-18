# Recurrent Transformer carry-token baseline (2026-08-17)

Status: implementation and registration only. Adds the PLAN §28 item-5
control to the matched baseline suite, implementing proposal B4 of
[2026-08-17-remaining-baseline-controls-preregistration.md](2026-08-17-remaining-baseline-controls-preregistration.md)
— the one remaining control that needs no shared-objective change and
therefore fits the existing one-recipe fairness contract. No training runs
use it yet; the multi-seed campaign remains fail-closed on its historical
pin.

## What was added

**`RecurrentTransformerBaseline`**
(`irene.recurrent_transformer.carry_token.v1`) — a standard recurrent
Transformer controller:

- New `TransformerCarryCell` (in `model/brain_cell.py`): one token set per
  cycle — belief, working memory, a single recurrent carry token, sensors,
  action/time, goal context, and a pooled retrieved-memory token — through
  untied-per-layer `nn.TransformerEncoderLayer` blocks (norm-first, zero
  dropout). Belief, working memory, and the carry are re-read from their
  own output positions through continuous-time blends. The carry is the
  only slot-shaped cross-step state (config: one thoughtlet, one register,
  like the monolithic controls). One empty routing diagnostic per layer
  preserves the `BrainCell` result arity; the cell stays tied across
  cognitive cycles.
- Width selected by exact allocated-parameter enumeration: **core width
  568** lands at 29,609,034 trainable parameters, 0.22% under the
  reference's 29,674,318 — inside the 1% manifest tolerance
  (`RECURRENT_TRANSFORMER_WIDTH` in `training/factory.py`).
- Recipe `configs/training/baseline-stagea-recurrent-transformer.toml` is
  a verbatim copy of the reference recipe except name and factory; the
  variant joins the manifest's `parameter_matched` fairness regime, which
  still verifies.
- Checked-in manifest regenerated: digest `d19d09bf…` → `869bd923…`; the
  suite now covers ten variants.

## Identity-recorded limitations

This is a conventional carry-token Transformer control, not every possible
recurrent Transformer design; the carry token is the only slot-shaped
cross-step state, and belief/working memory re-read their own token
positions each cycle rather than maintaining separate recurrent pathways.

## Verification

`tests/test_matched_baselines.py` grows to 12 tests. New
`test_recurrent_transformer_carry_persists_across_steps` proves the carry
crosses steps (continued-state output differs from fresh-state, unlike the
reactive control), pins the carry shape and diagnostics arity, and checks
determinism. The identity-uniqueness, fairness-regime, and
checked-in-manifest tests now cover the tenth variant. The full play-safe
suite (473 tests) exits 0.

## Remaining §28 gaps

Fixed multi-horizon no-persistence heads (B1), the recurrent latent
world-model actor (B2), and the task specialist (B3) remain preregistered
proposals awaiting explicit owner sign-off — each changes the shared
objective or the campaign shape.
