# V2.1i robust live world-model qualification design

Status: **development protocol draft; CPU-QUAL and TEST remain unopened**

## Objective

Qualify one recurrent model that can update continuously from pixels, maintain a
compact belief state, and predict the immediate result of every legal action in
parallel.  The first environment is maze chase because it exposes the core
requirements for live game interaction: sub-frame state updates, action-specific
danger, reward prediction, and uninterrupted control.  Passing this protocol is
evidence for those mechanisms, not a claim of human-level general intelligence.

The DGX Spark is not used for a substantial run until this protocol passes on
CPU.  All local diagnostics are single-threaded and force CUDA off.

## Mechanism retained from V2.1h

One factual observation advances the recurrent belief exactly once.  From that
same belief, the outcome model predicts the next latent state, reward
distribution, and hazard probability for idle/W/A/S/D in one vectorized call.
Targets for all five actions come from exact restores of the same simulator
snapshot.  Only the factual branch supplies the expert control label and future
recurrent history.

The dedicated nonlinear hazard path is retained.  Two development seeds showed
real state-conditioned ranking (aggregate ROC-AUC 0.6655 and 0.6813, and worse
performance after within-action state shuffling), so removing it would discard a
demonstrated mechanism.  Its absolute calibration did not replicate, so V2.1i
adds a small train-only calibration stage.

## Data namespaces

No range may overlap another range, even when split tags already differ.

| Namespace | Split tag | Proposed offset | Episodes | Purpose |
| --- | --- | ---: | ---: | --- |
| TRAIN-FIT | train | 16,777,216 | 128 | Joint representation and outcome learning |
| TRAIN-CAL | train | 16,777,344 | 64 | Fit hazard calibration only; never select architecture |
| DEV | validation | 25,165,824 | 64 | Mechanism development and falsification |
| CPU-QUAL | validation | 33,554,432 | 192 | One sealed confirmatory opening |
| TEST | test | unassigned | 0 opened | Reserved for a later end-to-end policy claim |

Each episode is a spawn-only 16-tick sequence with four burn-in ticks, the fixed
five-ghost/one-tick dynamics, zero control delay, non-sticky controls, and exact
all-action counterfactual targets.  Every partition uses exactly
`behavior_policy = balanced_intervention_v1` and
`behavior_intervention_rate = 0.5`; this applies to TRAIN-FIT, TRAIN-CAL, DEV,
the sealed CPU-QUAL population, and any later TEST population assigned by this
protocol.  These fields must be explicit in every materialized manifest.  The
scripted planner remains the action-label teacher, while the separately sampled
behavior action advances the recurrent factual history.  With 192
CPU-QUAL episodes, there are 2,304 scored factual roots and 2,304 branch outcomes
per action.  The opening aborts without scoring if any action lacks at least 100
positive and 100 negative branch outcomes or if factual gathering has fewer than
100 hazards.

The fixed-rate development coverage audit produced the following post-burn
support.  `Disagreements` counts factual behavior actions that differ from the
teacher label.  Every `safe/hazard` cell is an exact count, not an expectation.

| Partition | Rows | Disagreements | Factual hazards | Factual safe/hazard by idle/W/A/S/D |
| --- | ---: | ---: | ---: | --- |
| TRAIN-FIT | 1,536 | 641 (`0.4173177083`) | 196 | `147/73`, `336/28`, `233/24`, `298/39`, `326/32` |
| TRAIN-CAL | 768 | 298 (`0.3880208333`) | 93 | `66/41`, `164/7`, `125/12`, `146/11`, `174/22` |
| DEV | 768 | 327 (`0.42578125`) | 103 | `69/38`, `160/10`, `123/21`, `147/15`, `166/19` |

The exhaustive branch support from the same audit was:

| Partition | Branch safe/hazard by idle/W/A/S/D |
| --- | --- |
| TRAIN-FIT | `1273/263`, `1315/221`, `1301/235`, `1244/292`, `1199/337` |
| TRAIN-CAL | `629/139`, `635/133`, `670/98`, `611/157`, `592/176` |
| DEV | `603/165`, `622/146`, `649/119`, `599/169`, `588/180` |

The audited TRAIN-FIT, TRAIN-CAL, and DEV manifest SHA-256 prefixes are
`835fa63a`, `aed8e64a`, and `f2cd82e9`, respectively.  These development
coverage values do not open or precompute CPU-QUAL.

Before CPU-QUAL can be constructed, a create-only seal must bind the namespace
configuration, complete source bundle, model configuration, optimizer schedule,
model seeds, metric implementation, thresholds, and output paths.  The first
permitted opening records materialized sequence/content digests.  TEST access is
an unconditional error.

## Train-only hazard calibration

Joint training learns a raw hazard logit for every state/action pair.  A disjoint
TRAIN-CAL range then fits the smallest correction: one intercept per action

`calibrated_logit[a] = raw_logit[a] + bias[a]`.

The five biases are the only fitted calibration values.  Each convex intercept
fit is solved deterministically in float64 with unweighted binary cross entropy.
The model weights, recurrent belief, reward head, next-state head, and raw hazard
scorer remain byte-identical.  The calibrator is accepted only if it improves
TRAIN-CAL BCE over the identity map; otherwise the identity map is retained.  No
DEV or CPU-QUAL labels influence the parameters, stopping point, or choice
between calibrated and identity output.  The implementation also supports a
positive affine transform as a future fallback, but using it requires a new DEV
version and a new sealed CPU-QUAL namespace; V2.1i preregisters bias-only.

This preserves the ranking signal while correcting seed-dependent probability
bias at negligible live latency.  It is a component of the single deployed model,
not a second agent or an inference-time ensemble.

## Fixed model seeds and optimization

Development first runs bounded seeds 42 and 43 through the new train/calibration
partition to check the specific failure mode.  The schedule is revised only on
DEV.  After the source/protocol seal, confirmatory seeds are exactly
`44, 45, 46, 47, 48`; all five must pass.  No failed seed may be replaced, and no
threshold may change after CPU-QUAL opens.

The final preregistration must specify exact epochs, batch order, learning rates,
weight decay, gradient clipping, initialization digests, and optimizer-step
counts.  Checkpoints and evidence artifacts are create-only.

## Per-seed qualification gates

Learning relative to the true pre-optimizer model:

- hazard BCE is at most 0.90 times initialization BCE;
- next-latent loss is at most 0.60 times initialization;
- reward two-hot cross entropy is at most 0.80 times initialization;
- each action's next-state and reward loss is at most 0.90 times initialization.

Next-state learning must also pass a non-collapse check.  On CPU-QUAL, normalized
target latents and predictions must retain finite nonzero variance, target
covariance effective rank must be at least 8, prediction effective rank at least
4, and mean pairwise cosine distance must be at least 0.05.  These constants are
frozen before the namespace opens; loss reduction alone is not accepted as
evidence when either side is nearly constant.

Hazard quality relative to priors fit only on TRAIN-FIT:

- aggregate BCE is at most 0.98 times action-prior BCE;
- aggregate Brier is at most 0.95 times action-prior Brier;
- aggregate ROC-AUC is at least 0.65 and at least 0.10 above the prior;
- every action has ROC-AUC at least 0.60;
- every action has PR-AUC at least prevalence plus 0.05;
- every action has nonnegative Brier skill;
- equal-mass 10-bin ECE is at most 0.05 aggregate and 0.075 per action;
- absolute calibration bias is at most 0.05 for every action;
- factual-gather Brier beats its TRAIN-FIT factual action-prior baseline;
- train-only calibration does not worsen the uncalibrated parent BCE.

The factual action-prior comparison is eligible only when TRAIN-FIT contains
both safe and hazardous factual outcomes for every idle/W/A/S/D action.  Each
fitted action probability must therefore be strictly between zero and one.
When the frozen prior is evaluated on another partition, its BCE and Brier must
both be finite and strictly greater than zero before the factual improvement or
bootstrap gates may be scored.  Missing class support, a zero-loss baseline, or
a behavior-policy-determined perfect prior is a protocol failure, never a model
pass or a demand that a probabilistic model beat a perfect selector.  The
exhaustive all-action TRAIN-FIT prior and all existing all-action gates remain
required independently.

Causal use of state:

- evaluate 20 fixed cross-episode, within-action derangements;
- median shuffled BCE is at least 1.05 times real-state BCE;
- median shuffled AUC is at most real-state AUC minus 0.05;
- every action loses at least 0.03 ROC-AUC under shuffling.

Uncertainty of improvement:

- run 10,000 deterministic bootstrap resamples clustered by factual
  root/episode;
- the one-sided 95% lower bound is above zero for both BCE and Brier
  improvement over the TRAIN-FIT action priors.

Isolation and preservation:

- calibration changes only its five registered action-bias values (the five
  scale buffers remain exactly one);
- all other tensors are byte-identical;
- next-state, reward, and decision loss worsen by no more than 1%;
- next-latent cosine falls by no more than 0.01;
- raw reward MAE worsens by no more than 2%;
- parameters, buffers, losses, and probabilities are finite.

The `0.5` behavior intervention is a data-contract correction only.  It does
not change any loss, optimizer schedule, qualification threshold, or
preservation rule in this document.  In particular, calibration remains the
registered bias-only transform
`calibrated_logit[a] = raw_logit[a] + bias[a]`; all five scales remain exactly
one, and the existing aggregate/per-action ECE, calibration-bias, AUC, PR-AUC,
BCE, Brier, bootstrap, derangement, learning, preservation, and latency
thresholds are retained without relaxation.

Live speed, measured for each final seed on one CPU thread after 100 warmups and
5,000 ticks in each of three repetitions:

- model-forward p99 is at most 8 ms;
- in-process loop p99 is at most 12 ms;
- p99.9 is at most 16.67 ms;
- fewer than 0.1% of ticks exceed 16.67 ms.

Capture, transport, display, and physical input latency must be measured
separately before claiming a complete human-speed setup.

## Fail-closed outcome

Any missing metric, insufficient class support, source mismatch, artifact path
collision, changed non-hazard tensor, non-finite value, seed failure, or attempted
TEST access fails the entire qualification.  Failure evidence is written before a
nonzero exit.  A CPU-QUAL failure returns the project to DEV with a new version
and a new sealed confirmatory namespace; it never triggers DGX training.

Only a 5/5 pass authorizes a bounded DGX run of the exact source and schedule.
The resulting checkpoint must repeat provenance, preservation, causal-shuffle,
calibration, and latency verification before it can be called a working live-play
model.  The final end-to-end gate then runs the frozen model continuously from
pixels with no teacher or planner labels available at inference: fixed-rate
60 Hz stepping, no deliberate pauses, zero missed controls inside the measured
model deadline, and unseen-episode gameplay that earns pellets and beats the
registered random-action baseline on both reward and catches.  World-model gates
authorize training; this closed-loop gate is what authorizes the phrase
"working model."
