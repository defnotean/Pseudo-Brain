# Stage V2.0f order diagnostic — reshuffling helps, rare classes remain

**Date:** 2026-08-24  
**Release:** `r20260824t211646z-8e7768d24c2a`  
**Archive SHA-256:** `8e7768d24c2af156e8d0f94db73d7b55d664cc5831f50e397dec7725d92b5dff`  
**Container:** `v20f-order-diagnostic-20260824` / `fe2e02cae43e...`  
**Artifact:** `runs/v20f-order-diagnostic-20260824/v20f_order_diagnostic.json`  
**Status:** diagnostic only; not preregistered and not confirmatory.

Two V2-A arms used identical initialization, model RNG sequence, 376 batch
exposures (eight complete 47-batch cycles), exact global macro loss, and all
optimizer settings. Only batch order differed.

| Schedule | Initial macro loss | Final macro loss | Held-out balanced acc. | Held-out actions |
|---|---:|---:|---:|---:|
| Fixed incremental | `1.87822` | `3.54232` | `0.2065` | 3 |
| Deterministic reshuffle/cycle | `1.87822` | `1.41259` | `0.2109` | 3 |

The reshuffled full-bank loss improved 24.79%, while fixed order diverged. The
reshuffled cycle losses were `2.177,2.020,2.058,2.140,1.939,1.814,1.597,1.413`.
Thus fixed ordering is a real optimization defect and reshuffling repairs the
global training-bank objective.

It is not sufficient for the frozen V2.0e scientific gate. Reshuffled held-out
recall was `[0,.1863,.4411,.4274,0]`; actions 0 and 4 were never selected.
Training-bank class-0 NLL worsened `2.37945 -> 3.61231`; class-4 NLL improved
`2.23723 -> 1.70635` but did not reach argmax.

Class 0 occurs in only two of 47 padded batches, including one pure class-0
batch. The next diagnostic measures whether norm-1 gradient clipping cancels
the 6.27× rare-class loss scaling. No promotion claim is made from V2.0f.

