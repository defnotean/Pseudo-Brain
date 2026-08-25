# Stage V2.0c normalized recurrence smoke — numerical pass, policy-collapse fail

**Date:** 2026-08-24  
**Release:** `r20260824t204647z-12a4e110c6a3`  
**Archive SHA-256:** `12a4e110c6a32a3bfeddf51f85d544ef5608be138c528e516ce2b842bd3e218a`  
**Container:** `v20c-smoke-20260824` / `623b31701314...`  
**Verdict:** smoke FAIL; no 6,000-step run.

## What passed

Both 400-update arms completed with finite losses, gradients, parameters, and
strictly parseable atomic JSON. The V2.0b recurrent explosion is fixed.

| Arm | Mean lift | Delta vs V1 | Final training loss | Belief max | Thought max |
|---|---:|---:|---:|---:|---:|
| V2-A normalized | `-0.0007` | `+0.2123` | `1.23488` | `0.91449` | `11.32956` |
| V2-C normalized | `-0.0167` | `+0.1963` | `0.94236` | `1.00000` | `11.76836` |

These one-seed deltas are directional smoke diagnostics, not confirmation.

## Why the frozen gate failed

V2-A deployed action 2 on all 750 held-out episodes. V2-C deployed only
actions 1 and 2 (`[0, 566, 184, 0, 0]`). The fixed same-batch probe also
failed its required 10% improvement:

- V2-A: `1.39397 -> 1.36480` (2.09% reduction),
- V2-C: `1.56901 -> 1.68893` (7.64% increase).

Both thought maxima also exceeded the preregistered `11.0` telemetry bound.
The cell's layer-normalized output has that approximate bound, but the
subsequent attention residual is intentionally applied afterward; therefore
the earlier documentation treated a cell-local bound as a full-field bound.
This did not produce non-finite gradients. The new stage records a transparent
16.0 safety envelope rather than claiming an incorrect mathematical bound.

## Measured imbalance and next hypothesis

The pinned raw bank has labels `[24, 208, 232, 92, 44]`. The actual padded
length-grouped training cycle has 752 exposures:

`[24, 236, 268, 111, 113]`.

Thus the observed constant action 2 is the majority class, and V2-C's two
actions are the two largest classes. Stage V2.0d tests a no-hyperparameter
macro-balanced deployed cross entropy using exact inverse padded-exposure
weights. V2.0c remains failed; its gate is not edited or reinterpreted.

