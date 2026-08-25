# Stage V2.0i accumulation LR diagnostic — first non-collapsed policy

**Date:** 2026-08-24  
**Release:** `r20260824t213121z-60ba8a76468a`  
**Archive SHA-256:** `60ba8a76468adf40d69d2d7687536acbcc32d24f9050262d19cad946eeac721b`  
**Container:** `v20i-accumulation-lr-20260824` / `fd3268c9d37e...`  
**Artifact:** `runs/v20i-accumulation-lr-20260824/v20i_accumulation_lr.json`  
**Status:** V2-A diagnostic only; not confirmatory.

Both arms accumulated the exact 47-batch macro gradient, clipped once per
cycle, and ran 32 optimizer steps (1,504 batch exposures). They shared seed 42,
initialization, batch/noise schedule, weight decay, and all model settings.

| LR | Macro loss | Train recalls | Held-out balanced acc. | Held-out lift | Actions |
|---:|---:|---|---:|---:|---:|
| `1e-4` | `1.87822 -> 1.19620` | `[1,.356,.280,.351,.814]` | `0.5110` | `-0.0513` | 5 |
| `2.5e-4` | `1.87822 -> 1.20622` | `[1,.110,.448,.469,.805]` | `0.5200` | `-0.0473` | 5 |

This is the first Core V2 setup to retain and deploy all five actions. Both
arms exceed the earlier smoke requirements for macro loss, reduction, action
diversity, balanced accuracy, held-out lift delta, and recurrence stability.

The `1e-4` arm is selected prospectively for the next preregistration because
it had lower final macro loss, a more balanced training-bank action histogram,
and less late accumulated-gradient volatility (final norm `9.90` versus
`57.73`). The slightly higher single-seed held-out balanced accuracy at
`2.5e-4` is not used to select the noisier schedule.

This remains supervised benchmark evidence only. V2-C has not yet been tested
under accumulation, and no multi-seed confirmation or human-like game-playing
claim follows from this diagnostic.

