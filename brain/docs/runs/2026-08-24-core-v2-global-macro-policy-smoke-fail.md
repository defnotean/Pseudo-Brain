# Stage V2.0e global macro policy smoke — exact weighting, learning fail

**Date:** 2026-08-24  
**Release:** `r20260824t211008z-0b1272a178a7`  
**Archive SHA-256:** `0b1272a178a7c9f9f0ad1dbaf62f4022ab0499fc45a038e0ad7b40fcf33e1a50`  
**Container:** `v20e-smoke-20260824` / `c1e8db86d17a...`  
**Artifact:** `runs/v20e-smoke-20260824/v20e_smoke_results.json`  
**Verdict:** smoke FAIL; no 6,000-step run.

## What the correction proved

Both arms completed without OOM or non-finite values. Runtime verified exact
aggregate loss shares `[0.2,0.2,0.2,0.2,0.2]`, so V2.0d's per-batch reduction
defect is fixed. Recurrence remained inside its frozen envelopes.

| Arm | Mean lift | Delta vs V1 | Belief max | Thought max |
|---|---:|---:|---:|---:|
| V2-A global macro | `-0.0393` | `+0.1737` | `0.99935` | `11.93081` |
| V2-C global macro | `-0.0300` | `+0.1830` | `0.99994` | `11.59751` |

These one-seed lift values are smoke diagnostics, not confirmation.

## Frozen gate failures

| Arm | Macro probe | Actions | Balanced accuracy | Per-class recall |
|---|---:|---:|---:|---|
| V2-A | `1.87822 -> 2.68618` | 3 | `0.2126` | `[0,.8137,.1380,.1111,0]` |
| V2-C | `1.61324 -> 2.45916` | 2 | `0.2053` | `[0,.8783,.1481,0,0]` |

Exact macro weighting did not produce macro learning. Both probes worsened,
both balanced accuracies missed 0.25, V2-C missed three actions, and neither arm
selected actions 0 or 4. Confirmatory training was blocked.

## Next diagnostic

The 47 length-bucketed batches repeat in one fixed order. Class 0 appears only
at positions 35 and 36; the 400-update smoke stops at position 23 of the next
cycle, 34 optimizer updates after class 0 was last presented. The loss curve
also oscillates sharply by cycle position (V2-A `3.44728` at update 200 versus
`0.31803` at 300).

This supports, but does not prove, a sequential-interference hypothesis. A
bounded diagnostic will compare fixed order with deterministic random
reshuffling under identical initialization and equal exposure, while measuring
full-bank loss at every complete cycle and directly after the rare-class block.

Research basis:

- PyTorch data loading documentation describes `shuffle=True` as reshuffling
  training data every epoch: https://docs.pytorch.org/docs/stable/data.html
- Malinovsky et al., *Random Reshuffling with Variance Reduction*:
  https://proceedings.mlr.press/v216/malinovsky23a.html
- Gürbüzbalaban et al., *Random Shuffling Beats SGD after Finite Epochs*:
  https://arxiv.org/abs/1806.10077

