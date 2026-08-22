# Pseudo-Brain Master Roadmap

> **THIS IS THE CANONICAL FORWARD ROADMAP.**
> Historical run reports (`brain/docs/runs/`) describe what happened.
> [CURRENT_WORK.md](../CURRENT_WORK.md) describes immediate execution.
> This file describes the canonical future phase order.
> Historical planning documents that predate this file (e.g.
> [ROADMAP_TO_PACMAN.md](ROADMAP_TO_PACMAN.md), PLAN.md phase sections) are
> **superseded for forward planning** by this document; their history is
> preserved and remains valid as records.

**Last updated:** 2026-08-22 (roadmap rebase after Phase 2.5 discovery-set result)
**Rebase commit:** see git history at/after `c387503`

---

## Claim discipline (applies everywhere)

- **[MEASURED]** — direct experimental result.
- **[INFERRED]** — interpretation strongly suggested by measurements.
- **[HYPOTHESIS]** — explanation requiring further evidence.
- **[ASPIRATIONAL]** — long-term objective; NOT a current empirical claim.

---

## 1. Long-term objective

**[ASPIRATIONAL]** Develop Pseudo-Brain into a scalable general foundation
architecture whose persistent parallel cognition can eventually compete with
and, if architecture and scaling evidence support it, surpass frontier
conventional models on important dimensions (reasoning, research, coding,
planning, memory, agency, uncertainty handling, real-time cognition,
intelligence per parameter / FLOP / latency / energy).

**Current evidence is tiny research-scale prototypes only.** No general
frontier superiority has been demonstrated or is claimed. The credibility of
the project depends on never blurring [ASPIRATIONAL] into [MEASURED].

## 2. Governing philosophy

**Do not scale around a bad foundation.** Before large models, large data, or
language, make the Pseudo-Brain core as principled, stable, causally clean,
and scalable as reasonably possible. A foundational weakness found at 1M
parameters costs almost nothing; found at 1B+ it costs everything.

## 3. Canonical phase structure

### Phase 0 — Experimental foundation — **COMPLETE**
Deterministic environments, canonical observation/control types, snapshot /
branch / leakage audits, timing contracts, CPU-only play-safe suite.

### Phase 1 — Real-time runtime — **COMPLETE**
Persistent multi-thought execution inside physical deadlines. Vectorized
slot-recurrent core: ~1.1 ms GB10 forward at K=32 (vs 16.67 ms 60 Hz budget);
historically reduced from ~18.55 ms. [MEASURED]

### Phase 2 / 2.5 — Architecture validation — **NEAR COMPLETION (closure in progress)**
Question: is thought-mediated parallel cognition causal, useful, stable, and
behaviorally meaningful vs strong recurrent baselines?

Established [MEASURED]:
- Belief→action bypass removed; thought content causally load-bearing
  (transplant, knockout, binding interventions).
- Whole-slot permutation invariance holds.
- Vectorized real-time core (Phase 1).
- Behavioral K-capacity transition on long-horizon uncertainty (K≥16 reaches
  GRU ceiling on one task).
- **Reliability discovery set (n=10 independent training seeds, 8-hypothesis
  escalation, 3000 steps): PB K=32 catastrophic collapse 1–3/10 vs GRU 6–7/10;
  higher PB median return and Stage-1 survival.** [MEASURED, p≈0.027–0.036 —
  strong empirical evidence, not yet a confirmed claim]
- **Negative replication preserved:** ephemeral single-cue memory task shows
  no PB advantage (GRU equal/better on return). [MEASURED]

Closure queue (in progress):
1. Mechanistic battery on surviving/collapsed checkpoints (normal, reset,
   stale D={1,5,20}, probability scramble, binding scramble, consequence
   scramble, permutation, VoI probes) — comparing surviving-PB vs collapsed-PB
   vs surviving-GRU vs collapsed-GRU.
2. Pre-collapse trajectory analysis (telemetry already captured in
   `runs/mechanism-discovery-v1/` on the Spark).
3. Freeze mechanism hypothesis.
4. Fresh independent-training-seed confirmation (new 10+10 seeds, architecture
   untouched).
5. Phase 2 closure memo.

### Phase 2.6 — Foundation hardening & Core V1 — **NEXT MAJOR PHASE**
Make the core as strong as reasonably possible BEFORE scaling. Output:
a formally specified **Pseudo-Brain Core V1** with stable contracts.

Workstreams (each: baseline → weakness → preregistered candidate → smallest
clean change → torture tests → efficiency → multi-seed → keep/reject):

- **A. BrainCell** — systematic small candidate set for the recurrent update
  (persistence, selective update, evidence integration, revision, memory
  retrieval, goal conditioning, anti-overwrite). Resource-matched.
- **B. Thought register contract** — learned registers (content / working
  state / confidence / provenance / goal relation / consequence); no fixed
  slot-index semantics; exchangeability preserved.
- **C. Dynamic thought capacity** — Kmax fixed, Kactive learned; sleeping
  slots; compute efficiency; reduced interference.
- **D. Thought lifecycle** — sleep/wake/seed/strengthen/merge/retire; learned,
  permutation-safe.
- **E. Sparse thought communication** — mostly-private computation + selective
  routing (top-k / learned); measure useful communication vs interference vs
  latency.
- **F. World belief vs thought field** — belief stores world state; thoughts
  store hypotheses/plans/alternatives; audit so belief never becomes a new
  bypass.
- **G. Memory hierarchy** — working / thought / episodic / long-term; gated
  writes; relevance-weighted retrieval; causal-use tests.
- **H. Goals & subgoals** — general goal representation, subgoal generation,
  maintenance through distraction, replanning.
- **I. Adaptive compute** — learned halting; uncertainty/surprise-driven
  cycles; anytime output preserved; quality-vs-compute measured.
- **J. Prediction-error self-correction** — error as first-class signal for
  confidence, hypothesis survival, memory writes, replanning.
- **K. Uncertainty** — aleatoric vs epistemic distinction; information
  gathering only when it can help; no handcrafted bonuses.
- **L. Architectural constitution** — every Phase-2 failure lesson becomes a
  permanent invariant + regression test (see `ARCHITECTURAL_CONSTITUTION.md`).
- **M. Vectorization / systems** — batched [B,K,R,W] hot path; profile p50/95/99
  per subsystem; real-time preserved at K=128+.
- **N. Training stability** — normalized losses, gradient diagnostics, stable
  init, collapse detection, curriculum/replay controls. Investigate WHY high-K
  resists collapse (link to Phase 2 mechanism result) before redesigning.

### Core V1 freeze — **GATED**
Freeze only when: architectural cleanliness (constitution green), stability
(seed variance acceptable, collapse controlled), efficiency (vectorized,
real-time, K-scaling measured), cognitive mechanics (torture-suite gates met),
scalability (no unjustified O(K²), scaling path understood). Necessary
fundamental redesign afterward = Core V2, evidence required.

### Phase 2.7 — Scaling-law / intelligence-per-compute research
Ladder ≈ 1M → 5–10M → 30–60M → 100M → 300M → 1B (long-term). Vary W, Kmax,
Kactive, R, C, memory, routing sparsity. Fair strong baselines at each scale
(GRU/RNN, modern SSM/recurrent, Transformer, recurrent Transformer, world-model
agents). Measure params / FLOPs / latency / memory / performance / sample
efficiency / stability. Question: which extra unit of compute buys the most
cognition? **[ASPIRATIONAL] target: intelligence per compute, e.g. a smaller
PB matching a larger conventional model on persistent reasoning/agency.**

### Phase 2.8 — Cognitive capability training
Long-term memory, multi-step planning, goals/subgoals, uncertainty, world
modeling, information gathering, changed dynamics, self-correction, abstract
reasoning, decomposition, multi-objective, long-duration tasks. Make the
system significantly smarter before generalization claims.

### Phase 2.9 — Foundation modalities / language / tools
Only after Core V1 + scaling evidence. Text/code/vision/audio/tool observations
as modalities; pretrained language components as encoder/decoder ("ears and
mouth") while Pseudo-Brain remains the persistent cognitive controller.

### Phase 3 — Generalization
Unseen worlds/layouts/rules, compositional transfer, few-shot adaptation,
changed dynamics, novel instructions. "Can the brain figure out something new?"

### Phase 4 — General agent / research / coding
Persistent tool use, coding, research, verification, long tasks, episodic
experience, long-lived goals — exploiting persistent cognition, not a chat
wrapper.

### Phase 5+ — Frontier-scale foundation system
Multi-billion scale, multimodal cognition, large agentic RL environments,
world/video/action learning, distributed training, robotics. Aspirational
comparison against the strongest conventional systems, honestly.

## 4. Foundation Torture Suite (Phase 2.6 gate)

Unit tests for cognition; each isolates one property, with causal
interventions, strong conventional baselines, and NOT designed to favor PB:
remember 1 cue; remember several independent cues; maintain 2 incompatible
possibilities; maintain 8+; update one without erasing others; reject a
disproven hypothesis; ignore irrelevant evidence; preserve goal through
distraction; create/abandon subgoals; retrieve/avoid episodic memory;
distinguish randomness from ignorance; probe only when useful; adapt after
dynamics change; hold information across blank delays; handle contradictory
evidence; act while thinking; dynamic active-K; permutation invariance;
stale/noisy-thought robustness; multi-object uncertainty; long-horizon
planning; recover from wrong internal model.

## 5. Frontier evaluation discipline (Phase 5+)

Evaluation must eventually leave PB-designed environments: external held-out
reasoning, math, science, coding, research, agent benchmarks, computer use,
visual reasoning, planning, tool use, multimodal. Winning only PB-shaped
benchmarks is not sufficient evidence of anything general.

## 6. Hard rules (goalpost discipline)

- Every phase defines question / baseline / metric / gate / failure condition.
- Failures are recorded and preserved. No endless hypothesis renaming.
- No fake timelines. Progress through phase gates only.
- Data quality × training compute × curriculum × post-training matter as much
  as architecture; never attribute everything to architecture.
- Baselines are never weakened. GRU isolates recurrent-state organization
  today; it is not the final frontier baseline.

## 7. Agent startup sequence (mandatory)

1. `git status`, `git branch --show-current`, `git rev-parse HEAD`,
   `git log --oneline -20`, `git remote -v`.
2. Read `CURRENT_WORK.md` (immediate execution).
3. Read this roadmap (phase order).
4. Read newest `brain/docs/runs/*` and any active preregistration.
5. Check running jobs and hardware status (Spark occupancy).
6. Then act. Never trust a stale task file over HEAD + newest logs.

## 8. Model family naming (placeholder, not locked)

PB-S / PB-M / PB-L / PB-XL conceptually; each configuration documents params,
W, Kmax, expected active K, R, C, memory, routing, modalities, training
compute, measured latency. Do not lock branding before Core V1.
