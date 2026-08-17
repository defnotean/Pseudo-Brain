# Stage A continuation-gate record

Recorded: 2026-08-16

The registered 500-step Stage A continuation run completed normally, but its
held-out gate failed. The metrics artifact was structurally valid and the
fail-closed evaluator reached an ordinary negative decision; this was a model
acceptance failure, not an invalid run or an infrastructure error.

The run used the support-aware calibrated action objective selected by the
matched action-overfit A/B experiment, the full auxiliary objective, 4,096
procedural training sequences, and the registered cosine-after-warmup learning
rate schedule. It evaluated the same fixed 96-decision Stage A validation slice
every 100 optimizer steps.

## Immutable execution

- Release: `r20260816t193757z-ae230800f79c`.
- Release archive SHA-256:
  `ae230800f79cffbc555931a484889c2c4b0a3904f6b4b159a19f57e2a73321a3`.
- CUDA smoke: all 161 release tests passed before the bounded optimizer smoke;
  checkpoint SHA-256:
  `559211d8570d91e1c8a9788ffaa376996e779dd3a19ecb8c38dc58a8547ccf86`.
- Full-thesis canary checkpoint SHA-256:
  `9551c8711eaaef8438f772c14ddabcf73c1da93a9dbe2baf057395b16fec0bf4`.
- Run: `stagea-continuation-r20260816t193757z`.
- Final step-500 checkpoint SHA-256:
  `5384030bef1e0eeb44093bc24f4527d8d7b49e4f09592676a211484904315a97`.

## Registered gate result

Final-exit exact movement decisions improved throughout the run but finished
well below the preregistered acceptance floor:

| Optimizer step | Exact held-out movement decisions |
|---:|---:|
| 100 | `0/96` |
| 200 | `0/96` |
| 300 | `9/96` |
| 400 | `39/96` |
| 500 | `51/96` |

At step 500 the evaluator reported:

| Decisive metric | Observed | Acceptance condition | Result |
|---|---:|---:|---|
| Movement exact match | `51/96` | at least `80/96` | fail |
| Changed-action exact match | `6/26` | at least `13/26` | fail |
| Positive-key recall | `0.6458333` | at least `0.90` | fail |
| Movement false positives | `22/96` | at most `12/96` | fail |
| Predicted active movement keys per decision | `1.05208` | within `0.15` of target `1.42708` | fail |
| Opposite-direction conflicts | `1` | exactly `0` | fail |
| Non-movement false positives | `0` | exactly `0` | pass |
| Value loss | `0.362168` | below `0.35806523` | fail |

The latent-geometry screens passed: summary/register rank proxies were
`0.85483` and `0.85403`, effective thought slots were `31.1377`, and diversity
loss was `0.005484`. Those observations show that the tensor did not collapse
under these proxies; they do **not** show that the thoughts caused better
actions or specialized into distinct roles. World loss was `0.00853` and
remains diagnostic only; the current target and loss do not establish multiple
independent world models.

## Decision and next control

The 1,000-step bootstrap was not launched, and this checkpoint is not eligible
for the causal thought ablation because it did not pass the prerequisite
behavioral gate. It is also not a live-control-ready policy.

The late improvement from `9/96` at step 300 to `51/96` at step 500 makes a
schedule-limited explanation plausible: the registered cosine schedule was
reducing the learning rate while held-out action learning was still improving.
That is a hypothesis, not a conclusion. The next bounded experiment is a
matched, equal-FLOP Stage A run with the same data, objective, warmup, and
500-step budget but a constant learning rate after warmup. Only its held-out
gate result can determine whether the schedule accounts for the shortfall.

All accelerator work remained isolated to the DGX Spark. This run performed no
screen/audio capture and injected no physical controls on the local gaming PC.
