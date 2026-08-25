# Scalable real-time world-model research synthesis

**Date:** 2026-08-25
**Status:** research hypothesis, not qualification evidence

## Question

What architecture and evidence program could turn the present Pseudo-Brain
prototype into one model that perceives, predicts, remembers, plans, and acts at
human game-loop speed, while becoming more capable rather than less reliable as
model, data, and compute increase?

The current answer is a falsifiable hypothesis, not a breakthrough claim.
PB21M tests one necessary piece--balanced learning of action-conditioned hazard
outcomes--and explicitly cannot establish the whole result.

## Primary-source findings

1. **Naive parameter scaling can make an agent worse.** TD-MPC2 reports that
   scaling its predecessor's model and data can reduce performance. Its robust
   replacement couples normalized latent states, action-conditioned latent
   dynamics, reward and value prediction, a policy prior, planning, ensembles,
   and task embeddings. With those changes, one model improved with size through
   317M parameters on an 80-task dataset. This is evidence that architectural
   and optimization invariants must be qualified before spending the scaling
   budget, not evidence that every architecture improves when enlarged.

   Source: [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)

2. **General control benefits from an integrated latent world model.** DreamerV3
   learns dynamics and behavior through imagined futures and uses normalization,
   balancing, and value transformations to keep one configuration stable across
   more than 150 tasks. It collected diamonds in Minecraft from pixels and sparse
   rewards without demonstrations or a hand-built curriculum. This supports
   integrated next-state, reward/value, and policy learning; it does not establish
   cross-game human-equivalent play or low-latency desktop control.

   Source: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)

3. **Model and fresh data must be scaled together.** Microsoft's scaling study
   finds power-law pretraining-loss frontiers for tokenized world models, but the
   compute-optimal model/data coefficients change with tokenizer compression,
   task, and architecture. For one 256-token world-model setting the measured
   exponents were approximately 0.49 for parameters and 0.51 for data. The paper
   uses a diverse, effectively non-repeated-data regime and warns that downstream
   task quality and inference time remain separate open questions.

   Source: [Scaling Laws for Pre-training Agents and World Models](https://arxiv.org/abs/2411.04434)

4. **Predicting useful latent structure can be more scalable than reconstructing
   every pixel.** V-JEPA 2 first learns representations from large action-free
   video, then post-trains an action-conditioned latent predictor on a much smaller
   interaction dataset. Its action-conditioned model predicts future
   representations from previous states and actions and supports planning. This
   backs the present stop-gradient latent-target direction, provided control
   outcomes and causal action tests remain first-class gates.

   Source: [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)

5. **Multiworld data and self-generated experience are central to game
   generalization.** SIMA 2 combines broad game demonstrations, language labels,
   a reasoning core, and later self-directed play. DeepMind reports transfer to
   held-out worlds, while also identifying short memory, long-horizon verification,
   precise keyboard/mouse control, and robust visual understanding as open
   problems. This argues against treating success on one maze generator as proof
   of general game sense.

   Source: [SIMA 2: An Agent that Plays, Reasons, and Learns With You in 3D Virtual Worlds](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)

## Working architectural hypothesis

Keep a **single trainable recurrent agent**, not an orchestration of unrelated
language models. Within that one model, use explicitly different update rates:

- a fast state estimator and reflex/control path that updates every environment
  tick and always has an action ready before the deadline;
- a small set of parallel recurrent thought slots that predict several compact
  alternatives at each tick rather than emitting long verbal chains;
- action-conditioned latent dynamics predicting the next representation for the
  executed action and counterfactual legal actions;
- calibrated reward, hazard, termination, and uncertainty heads tied to semantic
  action IDs;
- a slower persistent memory/planning state updated from prediction error and
  high-information events without blocking the fast path;
- a policy/value prior that supplies an immediate action while bounded latent
  lookahead refines later actions;
- one shared representation and training objective, with stop-gradient target
  paths and loss normalization preventing any one auxiliary head from consuming
  the learning signal.

"Parallel thoughts" therefore means simultaneous compact latent predictions and
state updates inside one model. It does not mean generating three sentences, and
it does not require three separately served foundation models.

## Why this could scale

The hypothesis combines the compatible parts of the primary evidence:

- latent rather than pixel-perfect future prediction for the low-latency core;
- outcome/value supervision so the representation remains useful for control;
- multiple tasks, games, layouts, and behavior qualities in the data;
- explicit action conditioning and counterfactual action coverage;
- stable normalized objectives, deterministic training, and seed replication;
- bounded planning plus a policy prior so thought never stalls the control loop;
- persistent state for information that cannot fit in a short live context.

It can still fail. Additional width may be useless, recurrent slots may collapse,
counterfactual heads may be miscalibrated, planning may exploit model errors, or
fresh data may dominate model size. The word "scalable" is earned only by an
observed frontier, never by parameter count alone.

## Qualification ladder

1. **Component learning:** fresh-seed next-latent, reward, hazard, termination,
   and uncertainty heads must each beat frozen priors and causal controls.
2. **Integrated learning:** joint training must preserve every component and
   improve held-out action selection over the strongest simple recurrent baseline.
3. **Multi-step prediction:** latent rollouts must retain calibrated outcome and
   ranking quality over increasing horizons, including interventions that change
   only the action sequence.
4. **Closed-loop control:** the model must learn and act in a small pixel game with
   no privileged state at a fixed deadline, without pause-between-thought behavior.
5. **Memory and adaptation:** held-out layouts and changed mechanics must show
   within-episode improvement that disappears under preregistered memory knockout.
6. **Cross-environment transfer:** add multiple visually and mechanically distinct
   games; compare single-game, mixed-game, and held-out-game performance.
7. **Three-axis scaling:** preregister paired model-size, fresh-data, and compute
   rungs. Require positive held-out and closed-loop slopes, seed reliability,
   non-collapse, and no component regression. Fit a frontier only after enough
   rungs pass; do not extrapolate from two points.
8. **Real live loop:** measure capture-to-action latency, deadline misses, action
   freshness, and sustained play for at least one hour on the intended hardware.
9. **Minecraft/Pac-Man evidence:** only after the generic gates, evaluate each game
   from pixels and ordinary controls with human-reference reaction-time and task
   metrics. Neither game name may appear in a capability claim before this test.

## PB21M boundary

PB21M asks whether per-action-balanced factual supervision improves a fresh
action-conditioned hazard head without sacrificing all-action or complement
quality. A valid pass nominates a training recipe for the next scaling study. A
valid failure rejects that recipe on those namespaces. Either result advances
the program, but neither authorizes DGX full-model training, a checkpoint claim,
or a claim that the unified architecture is already scalable.

The immediate rule remains: qualify the learning mechanism on fresh data first;
then scale only after the component, integration, and latency gates pass.
