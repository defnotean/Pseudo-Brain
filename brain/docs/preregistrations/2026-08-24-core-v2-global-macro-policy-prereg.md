# Preregistration: Stage V2.0e — globally normalized macro policy

**Date frozen:** 2026-08-24, before V2.0e GPU execution  
**Status:** FROZEN  
**Predecessor:** V2.0d was numerically stable but failed every policy-learning
gate because per-batch weight normalization yielded non-macro effective shares.
Its confirmatory run remains forbidden.

## Falsifiable hypothesis

The V2.0d inverse-frequency weights were globally correct, but PyTorch's
weighted-mean denominator was recomputed from each batch's selected labels.
Across the fixed 47-batch cycle, the effective shares were
`[3.97%,29.03%,28.68%,21.50%,16.83%]`, not 20% each.

**Hypothesis:** using the same per-example weights with a fixed batch-size mean
will make the cycle-level loss exactly macro-balanced, prevent majority-action
collapse, improve the deterministic full-bank macro probe, and preserve
normalized-recurrence stability.

The loss is `mean_i(w[y_i] * -log p[y_i])`. Because every batch has 16 samples
and the global mean weight is one, its full-cycle reduction equals the frozen
full-bank probe reduction. Runtime must assert aggregate effective shares equal
`[0.2,0.2,0.2,0.2,0.2]` within `1e-12`.

No architecture, sampler, order, focal exponent, learning rate, clipping,
optimizer, data, update budget, or evaluation change is allowed.

## Frozen implementation identity

Base Git commit `106c193`; working-tree overlay:

| File | SHA-256 |
|---|---|
| `brain/src/irene_brain/v2/config.py` | `e481a230f8a995b227793cf8805ddb86b3287e8441152284396d4b9abddc29df` |
| `brain/src/irene_brain/v2/aggregator.py` | `7670325f0bba1e6e1c47e464bc22087ccf55da86352d62d8a2b314d3fb4d2717` |
| `brain/src/irene_brain/v2/brain_cell.py` | `03cc6c86d3e41f4ef71ec3f89db0cb79a8609818b79e661f5395330a9da0c298` |
| `brain/src/irene_brain/v2/belief_updater.py` | `cae13b9c8da6d0bb33c732de68d0c21ba98bc15619f122b03cea72c345a76318` |
| `brain/src/irene_brain/v2/losses.py` | `96ac6ea9b89cead659856e13195d662f7a7eff49f90006422c14ffdf9a06b05e` |
| `brain/scripts/stage_v20_core_v2_deployed_loss_baseline.py` | `8bc619ead71713eae78c51f7f51ad27c19744db8bd74cf410bea42e6f98beb71` |
| `brain/scripts/stage_v20e_global_macro_policy.py` | `ccb6987a17ccd60156d7ab5ee65d5cbb262f30d5a08d0bee4d1f26a5c7f92396` |
| `brain/tests/test_core_v2_decision_contract.py` | `e696ba2a8919cffbd39fd26f758752295a3c956a80b3322ebad39081116f41ff` |

Any byte change requires a new experiment identifier.

## Frozen 400-update smoke

Both V2-A and V2-C run at seed 42 with the V2.0c normalized recurrence,
direct-mean aggregation, inverse padded-exposure weights, pinned bank digest
`b3bb5fc33fd5f605`, fixed batch order, AdamW lr `5e-4`, weight decay `1e-4`,
clip `1.0`, and the identical deterministic held-out evaluation.

Each arm must satisfy all of:

- finite loss, gradient norm, parameters, and strict JSON artifact;
- full padded-bank macro probe final `<1.45` and at least 10% below initial;
- at least three distinct deployed held-out actions;
- held-out balanced accuracy `>=0.25`;
- maximum belief magnitude `<=1.00001`; and
- recorded post-attention thought magnitude `<=16.0`.

If any condition fails, stop and do not launch confirmatory training.

## Confirmatory run

Only after smoke PASS: both arms, 6,000 updates, seeds
`{42,142,242,342}`, identical protocol. Per-arm outcome classes against frozen
Core V1 R=`-0.2130` remain ALPHA (`delta>=+0.05`, sigma `<=0.06`), BETA
(`abs(delta)<=0.02`), GAMMA (`delta<=-0.03`), or AMBIGUOUS. Confusion matrices,
recalls, balanced accuracy, state maxima, and pre/post full-bank macro loss are
mandatory diagnostics.

This tests a supervised decision-learning contract only. It is not evidence of
human-level play, grounded prediction, useful memory, or a thoughtlet advantage.

