# Corrected Stage A gate record

Recorded: 2026-08-16

Immutable source release: `r20260816t181833z-b347a26638d5`

Source archive SHA-256:
`b347a26638d5be0464a4fbbd0220e9e5faad6ef63d26c30123e1c6da7e3b5a75`

## Completed execution gates

| Gate | Result | Checkpoint SHA-256 |
|---|---|---|
| Bounded CUDA smoke | GB10 BF16 backward, all 139 tests, and 3 optimizer steps passed. | `0507067ed43ab77741d6fb8ea04abbff5ce1de04dbfdf11e213c08b5d4c955ee` |
| Full-thesis canary | One full-model BF16 optimizer step and validation completed. | `c0785bf12addd4e024186c9e5ccde5780248f0ac53ac82e0e627d07694e085b7` |
| Bounded Stage A gate | All 100 optimizer steps and held-out validation completed; the scientific action-learning gate failed. | `69c80d9a2875f2d325888a6c5d8f17cb08f5fdecbd7642a04a32b8ef5aa678ff` |

The corrected release fixed the original negative-label dilution, persistent
slot identities, utility-write gradient, and thought diagnostics. The Spark
was idle before and after each bounded job. The 1,000-step bootstrap was not
started.

## Step-100 held-out result

The validation slice contained exactly 96 decisions, including 26 decisions
whose target differed from the previous control. Its copy-previous exact-match
baseline was `70/96 = 0.729167`.

The trained final exit produced:

- Exact W/A/S/D set match: `0/96`.
- Exact match on changed decisions: `0/26`.
- Predicted movement keys per decision: `4.0`, versus target `1.427083`.
- False movement keys per decision: `2.572917`.
- Opposite-direction conflict rate: `1.0`.
- Non-movement keyboard false positives: `0.0`.
- Validation action loss: `0.738146`.

This thresholded output looks like the old all-four shortcut, but its loss
reveals a different failure. The zero-logit baseline for the current button
loss is `1.05 * ln(2) = 0.727805`; training action loss `0.722833` and validation
action loss `0.738146` remained close to it. The logits therefore stayed near
zero and their small, correlated sign changes alternated between all-off and
all-on during training. This is undertrained indecision, not a confident
all-four solution.

The gate used `warmup_steps = max_optimizer_steps = 100`, so it reached the
configured `1e-4` learning rate only on the last step and received about 50.5
full-rate-equivalent updates. The top-one inactive-button loss also sends its
large negative gradient through only one of roughly 294 inactive channels per
sample. Both facts require isolation before a long run.

## What did work

- Summary and full-register rank proxies were `0.642573` and `0.640882`; with
  32 slots these remained far from the rank-one floor of `1/32`.
- Movement-query effective slot count was `31.2471`. This proves diffuse
  attention, not causal utility, so a later slot ablation is still required.
- World loss fell to `0.011092`, although the best-versus-second error gap
  `0.672809` shows that this is not yet evidence of several useful world models.
- Validation value loss `0.358031` only barely matched the optimal constant
  baseline (`0.358065`) and is not evidence of learned long-horizon value.

## Next controlled experiment

Before changing the architecture or inference threshold, run an action-only
tiny-set overfit diagnostic: eight training sequences, 300 optimizer steps,
10-step warmup, no weight decay, and no auxiliary losses. It must achieve at
least 95% exact training-set action accuracy with positive/negative logit
margins. If the unchanged action path cannot do that, replace top-one negative
mining with a loss that trains every movement direction and repeat the same
A/B diagnostic.

This remains procedural Stage A research, not general game understanding and
not a live-control-ready model. All accelerator work ran on the DGX Spark. The
local PC performed no model compute, capture, or physical input injection.
