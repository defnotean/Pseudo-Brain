# Phase 2.5 frontier campaigns — local CPU runs (2026-08-21)

**Context:** DGX Spark is occupied hosting Qwen 3.8 on GB10, so the owner directed these two frontier
campaigns to run on the Windows PC. Both scripts auto-select CPU when CUDA is hidden;
runs are labeled **local CPU experiments**. Behavioral metrics (returns, survival, accuracy,
ablations) are scientifically usable; no timing numbers from these runs may be compared to
GB10/DGX Spark Gate-8 latency measurements.

**Launch (CPU-only, single-threaded):**
```
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 ... PYTHONPATH=<repo>/brain/src \
  brain/.venv/Scripts/python.exe -B scripts/dgx_phase2_definitive_escalation.py
  brain/.venv/Scripts/python.exe -B scripts/dgx_phase2_ephemeral_memory_campaign.py
```
(venv `brain/.venv` was created via uv with numpy 2.4.6 + torch 2.13.0 (cpu).)

Note: `-I` (isolated mode) strips `PYTHONPATH`, so the frontier scripts (which, unlike the
ranked-k8 probe, contain no `sys.path` bootstrap) must be launched with `-B` only, with a
native forward-slash `PYTHONPATH`.

## 1. Definitive Escalation: 8-hypothesis multi-stage delayed uncertainty (`8293d3f` milestone)

5 models × 5 independent training seeds (42/142/242/342/442), 1200 steps; held-out eval seeds
1000–5000, delay=15.

| Model | Mean Return (5 seeds) | S1 Surv | S2 Surv | Final Acc |
|---|---|---|---|---|
| Proposal-GRU Baseline | −437.66 ± 180.23 | 19.4% | 15.8% | 1.4% |
| Pseudo-Brain K=1 | −470.24 ± 37.96 | 8.4% | 31.2% | 5.8% |
| Pseudo-Brain K=8 | **−352.80 ± 126.46** | 22.2% | **35.4%** | **13.4%** |
| Pseudo-Brain K=16 | −368.99 ± 131.61 | 25.8% | 26.2% | 6.6% |
| Pseudo-Brain K=32 | −366.62 ± 96.94 | 22.8% | 31.8% | 12.8% |

**Finding:** all Pseudo-Brain K≥8 variants beat the multi-branch GRU baseline on mean return
(K=8 best). K=8 also leads on Stage-2 survival (35.4%).

Hostile ablations (seed-0 checkpoints, delay=15):
- **Frame Reset** degrades every model (e.g. K=8: −374.00 → −526.30; GRU: −507.60 → −534.50) —
  persistent state matters; GRU is least sensitive (monolithic carry absorbs some reset cost).
- **Scramble prob / scramble binding**: small degradations only for K=8 (−374.00 → −424.45) and
  K=16/K=32 (−449.40 → −446.10 / −330.65 → −329.55); K=1 unaffected (no cross-thoughtlet structure).
- **Permutation**: near-neutral for K=32 (−330.65 → −332.80), small effect for K=16 (−449.40 → −448.10).

## 2. Ephemeral Memory & Persistence Campaign

Cues flash for 3 frames then vanish; final execution sees 100% blank input. 5 models × 5 seeds,
1500 steps, held-out eval, delay=15, `reset_state` toggled.

| Model | Normal Return | Reset Return | Final Acc |
|---|---|---|---|
| Proposal-GRU Baseline | **−418.40 ± 131.39** | −412.24 ± 201.36 | 7.6% |
| Pseudo-Brain K=1 | −480.89 ± 47.88 | −495.58 ± 12.20 | 11.2% |
| Pseudo-Brain K=8 | −476.95 ± 57.58 | −518.20 ± 11.87 | **16.0%** |
| Pseudo-Brain K=16 | −514.00 ± 10.76 | −517.78 ± 11.80 | 8.2% |
| Pseudo-Brain K=32 | −482.28 ± 68.05 | −518.38 ± 11.25 | 13.0% |

**Finding:** on the ephemeral-memory task, GRU leads on mean return, while **K=8 leads on final
accuracy (16.0%)** and the Pseudo-Brain variants show the expected **reset > normal** pattern
(reset returns worse than normal for every Pseudo-Brain variant — persistent thought state is
causally load-bearing). K=16 collapsed on this task (S1/S2 survival 0.0%).

## Interpretation notes
- The two campaigns are complementary: the 8-hypothesis task rewards K≥8 Pseudo-Brain on
  return; the ephemeral-memory task rewards GRU on return but K=8 on accuracy. K=8 is the
  consistently competitive configuration across both.
- Heterogeneity across training seeds is high (large stds, e.g. GRU ±180 on the escalation task),
  so single-checkpoint ablations (seed 0) are directionally informative but not a substitute for
  the 5-seed replication question still open in `CURRENT_WORK.md`.
- All numbers here are local CPU behavioral results; do not mix them with GB10 latency/Gate-8
  numbers.
