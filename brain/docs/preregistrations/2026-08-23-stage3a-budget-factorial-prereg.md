# Preregistration: Stage 3a — training-budget × LR-schedule factorial (frozen before run)

**Date:** 2026-08-23. **Architecture:** Core V1 frozen (W=120 K=32 C=3).
**Deterministic research mode ON; pinned banks; provenance recorded per run.**

## Question

Can Core V1 convert additional training compute into additional torture-suite
capability under the current objective, and does the optimizer regime gate that
conversion?

## Design — clean factorial

Cells: LR schedule ∈ {FIXED, COSINE} × budget ∈ {6k, 12k, 24k} steps.

- FIXED: constant lr 5e-4. One 24k-step trajectory per seed; 6k/12k cells are
  checkpoint evaluations of that same trajectory (identical by construction,
  since a fixed-LR run's prefix IS the shorter run).
- COSINE: cosine anneal 5e-4 → 0 over the DECLARED horizon. Each budget is a
  separate run (the horizon is part of the cell definition — declared here so
  no cell silently changes recipe).

Seeds {42, 142, 242} per cell. Batch size, weight decay, grad clip, data
bank (seed 42 pinned, digest recorded) identical across ALL cells.

## Metrics per checkpoint

mean torture lift · fraction of tasks > chance+2% · per-faculty lifts ·
train loss trajectory (probes) · held-out eval CE loss (held-out bank,
base_seed offset +77777, never trained on) · learning-curve AUC (lift probes
at 3k/6k/9k/12k/18k/24k) · grad-norm samples · action distribution on eval ·
wall time.

## Predeclared outcome classes

- A BUDGET-LIMITED: monotone clear improvement with budget, σ contained
  (e.g. −0.227 → ≤ −0.10 @24k) → width curve must be rerun at adequate
  budget before W is condemned.
- B SCHEDULE-GATED: FIXED flat, COSINE clearly better at 24k → optimizer
  regime is the bottleneck; establish stable scaling-compatible optimizer.
- C PLATEAU: all cells within noise of −0.227 → capability ceiling under the
  current objective; parameter scaling pointless; hunt the missing signal.
- D VARIANCE EXPLOSION: mean improves but σ grows sharply → fragile-
  optimization workstream first.

Ambiguous mixes are reported honestly against these four labels.

## Fidelity rules

No early stopping into a different cell; no tuning between arms; any deviation
voids the affected cell only. All raw JSON + provenance archived under runs/.
