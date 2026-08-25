# Stage V2.0d macro-balanced policy smoke — batch-normalization fail

**Date:** 2026-08-24  
**Release:** `r20260824t205829z-aca6c339ac99`  
**Archive SHA-256:** `aca6c339ac991d01a0be8f024c8bb5b1e97629a172da84c2e80cb5b8c8d98340`  
**Container:** `v20d-smoke-20260824` / `7a984e6d0060...`  
**Artifact:** `runs/v20d-smoke-20260824/v20d_smoke_results.json`  
**Verdict:** smoke FAIL; no 6,000-step run.

## Execution integrity

The immutable remote hashes matched the frozen preregistration. Both 400-update
arms completed without OOM, non-finite values, or artifact errors. The strict
JSON is owned by the unprivileged run user. Recurrent state stayed inside its
frozen safety envelopes.

| Arm | Mean lift | Delta vs V1 | Belief max | Thought max |
|---|---:|---:|---:|---:|
| V2-A macro balanced | `-0.0367` | `+0.1763` | `0.99976` | `11.93087` |
| V2-C macro balanced | `-0.0153` | `+0.1977` | `1.00000` | `11.55573` |

These one-seed lift values are smoke diagnostics, not confirmation.

## Frozen gate failures

| Arm | Macro probe | Actions | Balanced accuracy | Per-class recall |
|---|---:|---:|---:|---|
| V2-A | `1.87822 -> 2.24617` | 3 | `0.2017` | `[0,.6882,.2862,.0342,0]` |
| V2-C | `1.61324 -> 2.08694` | 2 | `0.2131` | `[0,.8935,.1717,0,0]` |

Both probe losses worsened, both balanced accuracies missed the frozen 0.25
minimum, V2-C missed three distinct actions, and neither arm ever selected
actions 0 or 4. The confirmatory run was therefore blocked.

## Root-cause measurement

The weights themselves were correct: exposure counts
`[24,236,268,111,113]` times their inverse-frequency weights each equal 150.4.
However, `torch.nn.functional.nll_loss(..., weight=w)` with its default mean
divides by the sum of selected weights *inside every batch*. The 47
length-bucketed batches have different class mixtures, so this changed the
effective class shares to:

`[0.039747, 0.290270, 0.286751, 0.214959, 0.168272]`.

Thus the purported macro loss gave rare class 0 only 4.0% of aggregate
coefficient mass instead of 20%. The observed zero class-0 recall is consistent
with this measured objective mismatch. This explains a flaw in V2.0d's
hypothesis; it does not prove that corrected macro weighting will learn.

Stage V2.0e changes only the reduction to fixed batch-size mean. It keeps the
same bank, weights, order, optimizer, architecture, recurrence, budget, and
evaluation, and asserts 20% aggregate coefficient share for every action.

