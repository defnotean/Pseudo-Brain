# Preregistration: Stage V2.0j — accumulated macro policy

**Date frozen:** 2026-08-24, before V2.0j GPU execution  
**Status:** FROZEN  
**Predecessors:** V2.0d–e showed that weighting before per-batch clipping did
not produce macro updates. Diagnostics V2.0f–i identified fixed-order
interference, measured clipping distortion, and found the first five-action
V2-A schedule. No V2-C accumulated run has been observed.

## Falsifiable hypothesis

All 47 independently clipped batch gradients exceeded norm 1 at initialization,
changing nominal 20%-per-class coefficient shares to approximately
`[0.56%,29.65%,28.14%,28.93%,12.71%]`. Accumulating the complete global macro
gradient before one clip restored both rare action pathways. At lr `5e-4` it
oscillated; at lr `1e-4`, 32 accumulated steps retained all five actions and
reduced the V2-A macro probe 36.3%.

**Hypothesis:** exact full-cycle accumulation with one norm-1 clip and AdamW lr
`1e-4` will give both V2-A and previously unobserved V2-C a stable,
non-collapsed deployed policy while preserving normalized-recurrence bounds.

The `1e-4` schedule is selected over diagnostic `2.5e-4` prospectively because
it had lower final macro loss, a more balanced training action histogram, and
lower late gradient volatility. V2.0i's one-seed held-out score is not treated
as confirmation.

Research basis:

- Pascanu et al., *On the difficulty of training recurrent neural networks*:
  https://proceedings.mlr.press/v28/pascanu13.html
- Malinovsky et al., *Random Reshuffling with Variance Reduction*:
  https://proceedings.mlr.press/v216/malinovsky23a.html
- PyTorch data loading/order documentation:
  https://docs.pytorch.org/docs/stable/data.html

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
| `brain/scripts/stage_v20f_order_diagnostic.py` | `092bdb64c2b7d23bc9abb009e84398b6fdf8f6e00cfe738a16e4fd2f4afc4a2f` |
| `brain/scripts/stage_v20j_accumulated_macro_policy.py` | `41c4578c60b6474a576d1f7008914fe80588e5cace57ec679214c0807e5818fc` |
| `brain/tests/test_core_v2_decision_contract.py` | `e696ba2a8919cffbd39fd26f758752295a3c956a80b3322ebad39081116f41ff` |

Any byte change requires a new experiment identifier.

## Frozen 32-cycle smoke

Both V2-A and V2-C run at seed 42. Each optimizer step accumulates the exact
inverse-exposure weighted gradient over all 47 padded batches (752 sample
exposures), divided by 47, before one global norm-1 clip and AdamW update. Batch
groups are deterministically reshuffled per cycle with an order RNG independent
of model RNG. There are 32 optimizer steps / 1,504 batch exposures per arm,
lr `1e-4`, weight decay `1e-4`, normalized recurrence, direct-mean aggregation,
and pinned bank digest `b3bb5fc33fd5f605`.

Each arm must satisfy all of:

- finite losses, accumulated gradient norms, parameters, and strict atomic JSON;
- final full-bank macro probe `<1.45` and at least 10% below initial;
- all five actions emitted on the final training-bank probe;
- at least three actions emitted on held-out evaluation;
- held-out balanced accuracy `>=0.25`;
- maximum belief magnitude `<=1.00001`; and
- maximum post-attention thought magnitude `<=16.0`.

If either arm fails, stop; the confirmatory run is forbidden.

## Confirmatory run

Only after smoke PASS: V2-A and V2-C each run 128 accumulated cycles (6,016
batch exposures), seeds `{42,142,242,342}`, with identical optimizer and
evaluation settings. Results are persisted atomically after every seed.

Per-arm outcome classes against frozen Core V1 R=`-0.2130` remain ALPHA
(`delta>=+0.05`, sigma `<=0.06`), BETA (`abs(delta)<=0.02`), GAMMA
(`delta<=-0.03`), or AMBIGUOUS. Multi-seed confusion matrices, balanced
accuracy, action histograms, state bounds, probe curves, gradient norms, and
wall time are mandatory. V2-C is not preferred unless it improves on V2-A
under a separately preregistered paired gate; this stage establishes whether
the training setup works and replicates.

This stage tests supervised decision learning, not live game competence,
grounded consequence prediction, useful episodic memory, or human-like thought.

