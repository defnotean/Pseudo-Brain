# T1-4 closed-loop competence probe — reactive baseline (diagnostic, read-only)

**Date:** 2026-08-28
**Mode:** local CPU only, single-thread, below-normal OS priority; CUDA hidden.
**Status:** DIAGNOSTIC PROBE (L-series discipline: read-only, no partition
opened, no training, no publish). It answers one question the T1-4
acceptance gates do NOT: does the trained `ReactiveBaselineV1` actually
**play** the env closed-loop, or only classify the teacher's stored frames?

**Checkpoint under probe:** `brain/runs/embodied-reactive-baseline-v1/seed_42.pt`
(133,773 params; T1-4 build, all 4 acceptance gates PASS).

## 1. Method

For 5 TRAIN seeds per family (25 episodes; the first 5 of
`partition_seeds(f, "TRAIN")[:50]`), run BOTH arms on the identical env:
- **Teacher:** the frozen pixel-only expert
  (`ScriptedMazeChasePlannerPolicy`, family-matched knobs; the T1-3 corpus
  teacher).
- **Model:** `ReactiveBaselineV1` seed-42 closed-loop — last-4 rendered
  frames (left-padded with the episode's reset frame, exactly as trained)
  + previous-applied-action one-hot → argmax action → env step.

Metric: the corpus M_P = 0.5·pellet-fraction + 0.5·(survival/max_ticks),
so the two arms are directly comparable. Probe:
`brain/scratch/probe_t14_live_competence.py`; raw output:
`brain/scratch/t14_live_competence_probe.log` (wall 247.9 s).

## 2. Results [MEASURED]

| Family | Teacher mean M_P | Model mean M_P | Model pellets | Model survival |
|---|---|---|---|---|
| F0_calm (1 ghost, shy, 4000t) | 0.806 | **0.500** | **0/150** | full 4000t (never caught) |
| F1_standard (3 ghosts, mixed, 6000t) | 0.524 | 0.867 | 93–116/142 (~72%) | full 6000t |
| F2_pressured (4 ghosts, direct, 6000t) | 0.521 | 0.836 | 84–102/138 (~65%) | full 6000t |
| F3_fast (4 ghosts, mixed, elroy, 8000t) | 0.750 | 0.940 | 104–136/142 (~88%) | full 8000t |
| F4_open_slow (2 ghosts, ambush, 8000t) | 0.773 | **0.515** | **1–8/166** | full 8000t |
| **All 25** | **0.670** | **0.731** | — | — |

Aggregate (25 episodes): teacher mean pellet-fraction **0.911** vs model
**0.463** (ratio 0.508); model survival 6400t vs 2899t (ratio 2.21 — the
model was **never caught** in any of the 25 episodes; the teacher frequently
clears the maze in ~200–420 ticks, which its 0.5 survival weight
penalizes).

## 3. Reading (claims labeled)

- **[MEASURED]** The model is a **cautious grazer, not a sweeper**: it
  never dies, survives to the max-tick horizon in all 25 episodes, and
  collects ~half the teacher's pellets overall (0.463 vs 0.911
  pellet-fraction). Its M_P *beats* the teacher on average (0.731 vs 0.670)
  ONLY because M_P's 0.5 survival weight rewards its never-dying at the
  cost of slow clearing. **Pellet-fraction is the cleaner competence
  metric here, and on it the model is at ~0.51× the teacher.**
- **[MEASURED]** The closed-loop behavior is **family-dependent**: on
  F0_calm and F4_open_slow the model is near-inert (0–8 pellets; on F0
  it emitted class 1 (W) on 80/80 observed ticks at one seed —
  constant-`W` loop); on F1/F2/F3 it survives pressure and forages
  productively. This pattern is consistent with [INFERRED]
  covariate-shift: on the calm/open families the teacher's clearing
  trajectories leave the stored-frame manifold early and the reactive head
  locks up, while under ghost pressure the high-visibility state
  distribution keeps it near its training data.
- **[INFERRED]** This is the expected naive-BC compounding-drift limit of a
  non-recurrent reactive head, NOT a training defect: on-distribution
  (stored frames) accuracy is ~0.78–0.945 (T1-4 gate U; this probe's
  stored-frame check on one F0 episode: 390/500 = 0.78) while
  off-manifold closed-loop behavior degrades family-dependently.
- **Corrections to earlier session summary:** an earlier 5-episode
  F0-only probe was over-generalized as "constant-W everywhere". The full
  25-episode probe shows the constant-lockup is confined to F0/F4; F1–F3
  behavior is productive grazing. This record supersedes that summary.

## 4. What this does and does not establish

- It is **not** the Tier-3 Q2 qualification comparison (that is the
  sealed-TEST paired one-sided 97.5% LCB under its own preregistration).
  It is a pre-T2-1 diagnostic on the TRAIN partition the baseline was
  trained on.
- It **preserves a negative result**: the T1-4 baseline, while passing all
  4 build gates, is a weak closed-loop player (~half the teacher's
  pellet rate, inert on 2 of 5 families). It is a **genuine, never-weakened
  comparison arm** — a weak baseline makes the Q2 contrast *more*
  informative, not less. No gate is re-tuned, no §-contract edited.
- No TEST/DEV/CAL seed was touched; no checkpoint claim is published.

**Next:** proceed to T2-1 (integrated Core V2 + EmbodiedInterfaceV1
training) as scheduled; this probe's numbers are the honest reference for
what "just a reactive head" achieves.
