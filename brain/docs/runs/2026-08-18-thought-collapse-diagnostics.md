# Thought-collapse diagnostics in the shared objective (2026-08-18)

Status: instrumentation. Local CPU-only verification; no training run, no
accelerator, no sealed data. Implements the collapse-metric gap named in
the 2026-08-18 strategic review ("does thought collapse actually happen
at K=32?") on top of the existing diagnostic surface.

## What was added

Three additive, sample-weighted diagnostic metrics in
`_thought_diagnostic_metric_counts` (`training/objective.py`), reported
by every `ThoughtFieldObjective` forward in both training and evaluation:

- `thought_pairwise_cosine_mean` — mean off-diagonal cosine similarity of
  the thought-summary gram, in [-1, 1]. This is the primary collapse
  signal; the strategic review's early-warning threshold is a sustained
  value above 0.7.
- `thought_duplicate_pair_fraction` — fraction of ordered off-diagonal
  full-register slot pairs with cosine above 0.98, in [0, 1]. Near-1
  values mean slots carry literally identical state.
- `movement_query_slot_entropy` — per-movement-query attention entropy
  over slots, normalized by log(K) into [0, 1]. A collapsed readout
  concentrates all queries on one slot (entropy → 0); a fully utilized
  field spreads uniformly (entropy → 1).

Single-latent controls (thoughtlets = 1) contribute the exact additive
identity for all three, matching the existing diversity-loss convention.

These complement the pre-existing `thought_summary_rank_proxy`,
`thought_full_register_rank_proxy`, `thought_slot_private_energy`,
`movement_query_effective_slot_count`, and
`world_best_second_error_gap`: the objective now reports the full
collapse battery the review asked for, at every logged step, with no new
machinery — the metrics ride the existing JSONL metric writer.

## Verification

`tests/test_multithought_core.py`'s diagnostic test now pins the
nine-metric key set with per-metric range assertions, and a new test
collapses two slots to identical register vectors and checks the
duplicate-pair fraction detects at least the 2/12 ordered pairs.
The architecture manifest was regenerated (objective source hash changed;
new digest recorded in the manifest). The full play-safe suite exits 0
(548 tests).
