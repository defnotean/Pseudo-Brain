# Stage V2.0g clip diagnostic — class weights erased by per-batch clipping

**Date:** 2026-08-24  
**Release:** `r20260824t212414z-cfc99cb2da40`  
**Archive SHA-256:** `cfc99cb2da40f0c6a84ab5f123a625b3309c37071db88a18420e077ad7e080e5`  
**Container:** `v20g-clip-diagnostic-20260824` / `25c9c587b3fe...`  
**Artifact:** `runs/v20g-clip-diagnostic-20260824/v20g_clip_diagnostic.json`  
**Status:** diagnostic only; no parameter updates and no promotion claim.

At the seed-42 initialization, the script computed both unweighted and exact
macro-weighted gradients for every one of the 47 batches. All 47 weighted
gradients exceeded the norm-1 clipping threshold.

Nominal class coefficient shares were exactly 20% each. Applying each batch's
measured scalar clip factor changed the approximate shares to:

`[0.005643, 0.296529, 0.281404, 0.289322, 0.127101]`.

Thus class 0 retained only 0.56% of update mass. The pure class-0 batch had
unweighted gradient norm `1183.45`, weighted norm `7416.31`, and fixed clip
factor `0.000135`. Weighting it by 6.27 increased magnitude, then clipping
divided that magnitude away. A threshold scaled by mean selected weight still
left approximate class shares `[3.8%,24.9%,23.8%,31.0%,16.5%]` because the
underlying per-batch recurrent gradient norms differ greatly.

This establishes that loss weighting before independent per-batch clipping
does not implement the intended macro update. It does not justify removing
clipping: the measured recurrent gradients are genuinely large. The next
diagnostic accumulates the complete 47-batch macro gradient, clips once, and
then updates, preserving both the global objective direction and safety bound.

Research basis: Pascanu, Mikolov, and Bengio, *On the difficulty of training
recurrent neural networks*, https://proceedings.mlr.press/v28/pascanu13.html.

