# PHASE 2 CLOSURE MEMO — Architecture Validation (2026-08-22)

**Status:** Phase 2 / 2.5 CLOSED. Verdict: **MIXED — task-specific reliability
advantage, no general superiority.** Forward transition: Phase 2.6 Foundation
Hardening per [MASTER_ROADMAP.md](MASTER_ROADMAP.md).

**Session HEAD range:** 25b16e3 → closure commit (see git log). All experiments on
DGX Spark gx10-db18 (GB10, CUDA 13.0), except where labeled local CPU.

---

## The question Phase 2 asked

*Does persistent thought-mediated parallel cognition provide a measurable advantage
over strong resource-matched monolithic recurrent baselines?*

## Answer

**Sometimes yes — in training reliability, not in best-case performance.**

### Positive findings [MEASURED]

1. **Collapse resistance (the session's central result).** On the 8-hypothesis
   multi-stage uncertainty task across 20 independent training seeds per architecture:
   PB K=32 catastrophic collapse 4/20 vs Proposal-GRU 12/20 (Fisher one-sided p=0.035;
   discovery set alone p=0.027; fresh confirmation alone p=0.27 — direction replicated,
   pooled significant). PB medians above GRU medians in both sets. Stage-1 survival
   roughly doubles for high-K PB.

2. **Mechanistic corroboration.** Hostile battery on preserved checkpoints: surviving-PB
   solutions are causally dependent on correctly-bound persistent hypothesis
   probabilities (prob/binding scrambles destroy ~74% of return; reset costs ~54%;
   stale D≤20 and whole-slot permutation harmless). Surviving-GRU solutions *improve*
   under reset (−190→−134) — the GRU succeeds by refusing to carry hypotheses.
   Collapsed checkpoints of both architectures are indistinguishable floor behavior.

3. **K-capacity is behaviorally real.** K-scaling transitions replicate (K≥16 reaches
   GRU ceiling on long-horizon uncertainty; accuracy rises monotonically with K on
   ephemeral memory).

4. **Real-time + lean.** Vectorized core: 824,667 params @ 1.11 ms GB10 forward at
   K=32 (vs GRU 907,859 @ 0.62 ms) — both far inside the 60 Hz budget.

5. **Causality machinery holds.** Thought knockout/transplant/binding interventions
   behave as designed; permutation invariance exact everywhere it was tested.

### Negative findings [MEASURED — preserved]

6. **No return superiority anywhere.** Best-case/asymptotic return never separates the
   architectures on any task. Where means separate, they do so within seed noise.
7. **Ephemeral single-cue memory: GRU wins.** n=10 replication: no PB advantage
   (permutation p=0.80, point estimate favors GRU); GRU's reset-collapse proves the
   task needs memory — PB doesn't convert its accuracy edge into control there.
8. **Original unmediated thesis:** falsified (Phase 2 original). **Legacy ranked-K8/K4
   branch:** closed negative. **CPU preliminary "K=8 beats GRU":** not reproduced
   under proper training-seed methodology.
9. **PB collapse still happens** (2–3/10 seeds): the architecture reduces, not
   eliminates, catastrophic instability.

### [INFERRED] mechanism

Parallel persistent slots preserve learned competing-hypothesis structure through the
optimization window where a monolithic state tends to overwrite/compress it. Evidence:
trajectory shapes (GRU collapse = early bifurcation after near-ceiling; PB fails later),
battery asymmetry (PB's surviving solution is scramble-sensitive, GRU's is reset-helped),
and the task boundary (advantage appears exactly where multiple latent variables must
coexist; absent where one memory suffices).

### [HYPOTHESIS] carried into Phase 2.6

Persistent parallel hypothesis organization reduces interference-driven optimization
collapse when multiple incompatible latent possibilities must coexist. Phase 2.6
Workstream N must treat collapse-resistance as a measurable design property, not luck.

---

## What would have made this stronger (honest gaps)

- Confirmation set alone underpowered (n=10/arm; pooled n=20 reaches α=0.05).
- Single lab family of tasks; third multi-latent task untested.
- VoI/probe interventions designed but not run in the battery.
- FLOPs formally counted nowhere; latency/params only.
- GRU remains a narrow baseline class for 2026 (see MASTER_ROADMAP §Frontier).

## Resource summary [MEASURED]

| Model | Params | GB10 fwd p50 |
|---|---|---|
| Proposal-GRU | 907,859 | 0.62 ms |
| PB K=32 | 824,667 | 1.11 ms |

## Artifact index

Runs: reliability10, escalation-3000, hostile-battery, mechanism-discovery-v1
(checkpoints + trajectories), confirmation10-v1, ephemeral campaigns — Spark
`~/projects/pseudo-brain/runs/`; records `brain/docs/runs/2026-08-22-*.md`;
statistics inline in each record. Git: every result committed and pushed.

## Transition

Phase 2.6 begins with the FOUNDATION AUDIT (component matrix → highest-leverage
weaknesses → disciplined smallest-change experiments), per MASTER_ROADMAP.md.
The constitution (ARCHITECTURAL_CONSTITUTION.md) now encodes every lesson above as
a permanent invariant.
