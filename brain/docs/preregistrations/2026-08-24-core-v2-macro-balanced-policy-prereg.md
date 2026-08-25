# Preregistration: Stage V2.0d — macro-balanced deployed policy

**Date frozen:** 2026-08-24, before V2.0d GPU execution  
**Status:** FROZEN  
**Predecessor:** V2.0c fixed numerics but failed its policy-diversity and
fixed-probe gates. Its confirmatory run is forbidden.

## Falsifiable hypothesis

The 752 padded training exposures are distributed across actions as
`[24, 236, 268, 111, 113]`, a maximum/minimum ratio of 11.17. The two V2.0c
policies collapsed onto the most exposed classes.

**Hypothesis:** macro-balancing the deployed cross entropy by exact inverse
padded exposure will prevent majority-action collapse, improve a deterministic
full-bank macro probe, and preserve V2.0c's numerical stability.

No focal exponent, effective-number beta, sampler temperature, architecture
scale, learning rate, or update budget is introduced. For class `c`, the
weight is `752 / (5*n_c)`:

`[6.2666666667, 0.6372881356, 0.5611940299, 1.3549549550, 1.3309734513]`.

Research basis:

- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html
- Lin et al., *Focal Loss for Dense Object Detection* (diagnosis of
  majority/easy-example domination): https://arxiv.org/abs/1708.02002

This stage chooses simpler exact inverse-frequency macro weighting because it
has no new hyperparameter and directly matches the balanced-accuracy gate.

## Frozen implementation identity

Base Git commit `106c193`; working-tree overlay:

| File | SHA-256 |
|---|---|
| `brain/src/irene_brain/v2/config.py` | `e481a230f8a995b227793cf8805ddb86b3287e8441152284396d4b9abddc29df` |
| `brain/src/irene_brain/v2/aggregator.py` | `7670325f0bba1e6e1c47e464bc22087ccf55da86352d62d8a2b314d3fb4d2717` |
| `brain/src/irene_brain/v2/brain_cell.py` | `03cc6c86d3e41f4ef71ec3f89db0cb79a8609818b79e661f5395330a9da0c298` |
| `brain/src/irene_brain/v2/belief_updater.py` | `cae13b9c8da6d0bb33c732de68d0c21ba98bc15619f122b03cea72c345a76318` |
| `brain/src/irene_brain/v2/losses.py` | `dd2e30d5dc3c582782cc5f7af8132bc859cfb106fe760f37ab7e878ca9a4ae86` |
| `brain/scripts/stage_v20_core_v2_deployed_loss_baseline.py` | `32ced52305e71f99ee3371c76c0dfe708d1779d6183846b687be222bd6fcdc4c` |
| `brain/scripts/stage_v20d_macro_balanced_policy.py` | `7ae7e1a26c178d308d600ffd1827ca8dec66f12467f36b61648000a802c0e49f` |
| `brain/tests/test_core_v2_decision_contract.py` | `c6da1455e66cc75dc83925c5735aca254d7f6fc8fad96f850218a59c85ca260e` |

Any byte change requires a new experiment identifier.

## Frozen 400-update smoke

Both V2-A and V2-C run at seed 42 with the V2.0c normalized recurrence,
direct-mean aggregation, pinned bank digest `b3bb5fc33fd5f605`, AdamW
lr `5e-4`, weight decay `1e-4`, clip `1.0`, and the identical deterministic
held-out evaluation.

Each arm must satisfy all of:

- finite loss, gradient norm, parameters, and strict JSON artifact;
- full padded-bank macro probe final `<1.45` and at least 10% below initial;
- at least three distinct deployed held-out actions;
- held-out balanced accuracy `>=0.25` (chance is 0.20 across five classes);
- maximum belief magnitude `<=1.00001`; and
- recorded post-attention thought magnitude `<=16.0`.

The 16.0 thought envelope is a prospective safety threshold informed by
V2.0c's maximum 11.77; it is not presented as a LayerNorm theorem. If any
condition fails, stop and do not launch confirmatory training.

## Confirmatory run

Only after smoke PASS: both arms, 6,000 updates, seeds
`{42, 142, 242, 342}`, identical weights/protocol. Per-arm outcome classes
against frozen Core V1 R=`-0.2130` remain ALPHA (`delta>=+0.05`,
sigma `<=0.06`), BETA (`abs(delta)<=0.02`), GAMMA (`delta<=-0.03`), or
AMBIGUOUS. Full confusion matrices, recalls, balanced accuracy, state maxima,
and pre/post full-bank macro loss are mandatory diagnostics.

This tests balanced decision learning only—not human-level play, grounded
consequence predictions, useful memory, or a thoughtlet advantage.

