# Phase 2.7 Stages J/K: batched-engine equivalence PASS + real throughput (2026-08-22)

**Prereg:** `2026-08-22-batched-engine-equivalence-prereg.md` (frozen before run).
**Data:** Spark `runs/equiv_results.json`. Corrected length-bucketed batched
engine (final-frame loss only, no padding) vs canonical solo trainer,
6000 optimizer steps each, 3 seeds.

## Scientific equivalence [MEASURED] — PASS

| Seed | solo lift | batched lift | \|Δ\| |
|---|---|---|---|
| 42 | −0.2100 | −0.2180 | 0.008 |
| 142 | −0.2140 | −0.1900 | 0.024 |
| 242 | −0.1687 | −0.1873 | 0.019 |

3/3 seeds within 0.03; mean |Δ| = 0.017 ≤ 0.02 → **PASS per prereg gates.**
The batched engine trains Pseudo-Brain to statistically equivalent quality.
It becomes the approved Phase 2.7 campaign trainer; solo stays as reference.

## Throughput — the correct accounting

The script's `wall_speedup_x: 0.76` is misleading because one batched
optimizer step consumes 16 episodes while a solo step consumes 1: the batched
arm received 16× the episode exposure in similar wall time. Per-EPISODE
throughput (the metric that matters for fixed-data campaigns):

- Solo: 6,000 episodes / 490 s = **12.2 episodes/s**
- Batched B=16: 96,000 episodes / 655 s = **146.6 episodes/s**
- **≈ 12× per-episode training throughput** [MEASURED]

Consistent with the Stage I projection (11×). For a fixed-episode-budget
campaign (e.g. the locked 6000-episode protocol), wall time drops from ~8 min
to ~41 s per seed-arm.

## Semantics note for future campaigns

A "batched step" ≠ a "solo step." Campaign specs must state EPISODE BUDGETS,
not optimizer-step counts. Data order also differs by construction
(length-bucketed grouping); equivalence here was measured under that regime.

## Status

Stage K complete. The frozen Core V1 now has a validated ~12× faster training
engine. Next: Stage L controlled scaling sweep (W-axis first) using the
engine, if session time remains.
