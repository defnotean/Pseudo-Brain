# Pseudo-Brain — DRAFT public README (INTERNAL, NOT PUBLISHED)

> **Status: draft.** This file is not the repo README and must not be
> published until the release gates in `brain/docs/OPEN_SOURCE_READINESS.md`
> are met. Numbers marked ⏳ are placeholders pending measured results —
> they must be replaced with real artifacts or the section deleted.

## What Pseudo-Brain is

Pseudo-Brain is a research project studying a recurrent, sensorimotor
"thoughtlet" architecture: a small brain-like model that maintains K parallel
hypotheses ("thoughtlets"), predicts consequences of candidate actions, and
aggregates those hypotheses into a single deployed decision.

It is not a language model and makes no claim to be one.

## The problem it attacks

Standard supervised training on a policy network optimizes a training output.
If evaluation reads a different output path (e.g. an aggregate over predicted
consequences), gradient signal never reaches what is actually deployed.
Pseudo-Brain Phase 2.7 demonstrated this failure empirically: models trained
to near-zero loss showed no held-out generalization gain across width,
training budget, or data-diversity interventions.

Core V2's central change: **the deployed decision distribution receives the
direct training signal** (deploy-aligned learning).

## What a thoughtlet is

One of K=32 exchangeable latent slots. Each thoughtlet:
- shares weights with all others (no fixed role),
- persists within a trial,
- emits a full consequence hypothesis: action logits, predicted next state,
  reward, hazard, confidence, existence probability.

Physical slot index carries no meaning; permuting slots permutes outputs
(verified by unit test).

## How it differs from a Transformer / GRU

| | Transformer | GRU | Pseudo-Brain V2 |
|---|---|---|---|
| Parallel hypotheses | No | No | K thoughtlets |
| Explicit consequence heads | No | No | reward/hazard/confidence/existence/branch per hypothesis |
| Deployed-output training | N/A | usually direct | explicit contract (V2) |
| Prediction-error feedback loop | No | No | pending predictions compared vs reality each tick |
| Multi-timescale state | KV cache | hidden state | fast (trial) / session (cross-trial) / slow (weights) |

## What is actually measured

- Held-out "torture suite" lift over empirical chance (per-task, preregistered)
- Train/deploy divergence diagnostics
- Causal intervention effects (prediction-error zeroed/scrambled, memory
  read/write disabled, session reset)
- Structural generalization splits (train/holdout/new-rule)

Chance floors are computed empirically per task, not assumed.

## What is still hypothetical

- That consequence-head learning improves generalization (V2.1+)
- That cross-trial adaptation without optimizer steps works at scale
- Anything about scaling behavior (deliberately untested until V2 proves out)

## Core V1 historical result (negative, preserved)

Core V1 trained with replicated per-slot cross-entropy reached ~0 training
loss but flat held-out lift (≈ −0.21 mean lift on the torture suite).
Width (120→480), budget (6k→24k steps), and data diversity (finite bank vs
fresh stream) all failed to improve it. These negative results motivated V2's
design and are preserved in `brain/docs/runs/` with frozen preregistrations.

## Core V2 architecture

```
observation ──▶ SensorEncoder ──▶ z_t
                                   │
pending prediction ──▶ ERROR(e_t) ◀┘   (error computed BEFORE cognition)
        │                    │
        ▼                    ▼
     belief update  ◀── episodic/session retrieval
        │
        ▼
   thought field (K=32 BrainCell updates × C=3 cycles + attention)
        │
        ▼
   consequence hypotheses (per thoughtlet)
        │
        ▼
   multiplicity-proof weighted-mean aggregator ──▶ action_dist  ◀═ TRAINED DIRECTLY
        │
        ▼
   world model ──▶ pending prediction for t+1
```

## Flagship reproducible result

⏳ Pending Stage V2.0 classification and confirmation campaign. This section
will either cite exact commit + artifact + reproduction command or be removed.

## How to reproduce

⏳ Placeholder — will document: environment, hardware requirement (any CUDA
GPU ≥ 8GB or CPU for smoke tests), exact command sequence, expected artifact
hashes.

## Limitations

- Tiny visual domain (32×32 RGB), 5 discrete actions
- Research code quality; APIs unstable
- Negative results are reported as such and are load-bearing
- No claim of biological plausibility beyond loose inspiration

## Hardware requirements

⏳ CPU-only works for unit tests; training runs used a single NVIDIA DGX
Spark (GB10). Any modern GPU should suffice for reproduction-scale runs.

## Citation / license

No license applied yet. Licensing decision deliberately deferred
(see internal `LICENSE_OPTIONS_INTERNAL.md`). Citation block TBD upon release.
