# Matched-baseline smoke comparison probe (2026-08-18)

Status: exploratory verification probe. Local CPU-only, single-threaded;
not a training run and not a qualified result. Implements the strategic
review's highest-leverage cheap item: an early matched-baseline signal
*before* any DGX spend.

## Protocol

`scripts/compare_baselines_smoke.py` rebuilds all eleven registered
architecture variants at smoke scale with identical seeds
(`torch.manual_seed(20260818)` before every construction), trains each on
the same deterministic moving_shapes batches for 64 optimizer steps
(sequence_length 8, burn-in 2 — the campaign window shape, so the B1
multi-horizon world loss is exercised by every variant), and scores each
on the same four held-out validation sequences before and after. Per-variant
JSON rows: `docs/runs/artifacts/baseline-smoke-compare/<variant_id>.json`.

## Results (64 steps, seed 20260818)

| variant | params | train loss first→last | val action loss | val movement exact | val world loss | pairwise cos |
| --- | --- | --- | --- | --- | --- | --- |
| **routed (reference)** | 47,258 | 0.8623 → 0.8337 | 1.1175 → 0.7155 | 0.000 → 0.042 | 0.8770 | 0.559 |
| isolated_slots | 47,258 | 0.8632 → 0.8343 | 1.1303 → 0.7169 | 0.000 → 0.042 | 0.8738 | 0.561 |
| reset_slots | 47,258 | 0.8708 → 0.8423 | 1.1138 → 0.7178 | 0.000 → 0.042 | 0.8902 | 0.577 |
| dense_routing | 47,258 | 0.8624 → 0.8338 | 1.1178 → 0.7155 | 0.000 → 0.042 | 0.8774 | 0.559 |
| reactive | 47,258 | 0.8535 → 0.8289 | 0.9548 → 0.7284 | 0.000 → 0.042 | 0.8680 | 0.435 |
| serial_depth | 58,123 | 0.8553 → 0.8399 | 0.8356 → **0.6892** | 0.000 → **0.208** | **0.8150** | 0.560 |
| independent_ensemble | 49,833 | 0.8441 → 0.8345 | 0.7798 → 0.6803 | 0.042 → 0.000 | 0.8846 | 0.575 |
| fixed_multi_horizon (B1) | 47,258 | 0.8796 → 0.8508 | 1.1138 → 0.7183 | 0.000 → 0.042 | 0.9786 | 0.580 |
| monolithic_gru.same_width | 46,313 | 0.8559 → 0.8581 | 0.7218 → 0.7032 | 0.125 → 0.000 | 1.1861 | — |
| monolithic_gru.parameter_matched | 53,909 | 0.9881 → 0.9478 | 1.1124 → 0.7029 | 0.000 → 0.042 | 0.9490 | — |
| recurrent_transformer | 39,689 | 1.1044 → 1.0756 | 1.0541 → 0.7507 | 0.083 → 0.083 | 1.0394 | — |

(—: single-latent controls have no slot pairs; the collapse metrics
contribute their additive identity by design.)

## Reading

Three honest signals, none decisive at this scale:

1. **No thought collapse anywhere.** Pairwise cosine stays in
   0.44–0.58, well under the 0.7 alert threshold; duplicate-pair fraction
   is exactly zero for every slot variant. The review's first go-criterion
   (collapse stable) passes.
2. **The reference does not lead at smoke scale.** It ties its
   parameter-exact ablations (isolated/reset/dense within ±0.003 action
   loss) and trails serial_depth (0.6892, movement exact 0.208 — the
   untied-depth control is the strongest learner here) and the
   parameter-matched monolithic GRU (0.7029). At 64 steps on a tiny model
   this is expected to be noisy — the smoke model is ~0.16% of the thesis
   parameter count and 64 steps is 1/32 of the campaign's shortest stage —
   but it is exactly the cheap warning the review asked for: the
   architecture claim currently rests on nothing at this scale.
3. **B1 behaves as designed.** The fixed-horizon control trains stably
   under the multi-horizon loss (world loss 0.9786, the highest of the
   slot variants — its static partition pays the expected price versus
   flexible assignment, and its pairwise cosine 0.580 is on the family
   trend).

**Consequence for the go/no-go framework:** the review's RCQ-v3
proceed-criterion "reference leading on ≥3/10 metrics at smoke scale" is
**not met** by this probe. Per the standing rule this does not cancel
anything by itself — the probe is explicitly unqualified — but the
qualified matched-baseline comparison is now the highest-value DGX item,
ahead of any further RCQ qualification rounds.

## Verification

Per-variant JSON rows are written incrementally and reruns are
deterministic (fixed seeds, fixed batch order). The play-safe suite is
unaffected (script-only change).
