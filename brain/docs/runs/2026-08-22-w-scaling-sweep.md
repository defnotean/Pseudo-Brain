# Phase 2.7 Stage L: W-axis scaling sweep — first result (2026-08-22)

**Setup:** frozen Core V1, batched engine (equivalence-approved), K=32 C=3
fixed, width ∈ {120, 240, 480}, 3 seeds each, 6000 steps, locked eval.
Data: Spark `runs/w_scaling_results.json`.

## Results [MEASURED]

| Width | Params | Mean lift | Seed σ | Latency p50 |
|---|---|---|---|---|
| 120 (frozen) | 0.82M | −0.2327 | ±0.0039 | 1.09 ms |
| 240 | 2.13M | −0.2184 | ±0.0261 | 0.99 ms |
| 480 | 6.30M | **−0.1722** | **±0.1344** | 0.93 ms |

## Findings

1. **Mean lift improves with width** (−0.233 → −0.172 from W=120→480):
   learned capacity helps the torture suite even under the fixed light
   training budget. [MEASURED]
2. **Variance explodes with width** (σ 0.004 → 0.134). At W=480 one seed
   reached −0.009 lift (near-zero gap to chance — by far the best single run
   of this phase) while another sat at −0.338. Wide models are high-variance:
   they can be much better OR no better, seed-dependent.
3. **Latency does NOT grow with width** in the tested range (1.09 → 0.93 ms;
   slightly improved, likely wider parallelism per kernel). Real-time budget
   intact at all widths.

## Interpretation [INFERRED]

W-scaling shows a genuine but noisy capability gain — consistent with the
Phase 2.6 finding that PB's bottleneck is optimization/sample efficiency, not
parameter count: more capacity helps on good seeds but amplifies seed
sensitivity at this budget. The obvious follow-up is a W=480 × more-steps or
× multi-seed(n≥5) confirmation to separate "wide models need more data" from
"wide models are unstable."

## Scaling-curve status

| Axis | Tested | Result |
|---|---|---|
| K (thoughts) | 32→256 (Phase 2.6) | latency flat; params flat; capacity effects behavioral |
| W (width) | 120→480 (tonight) | mean lift ↑, variance ↑↑, latency flat |

Both primary hardware-relevant axes now have first scaling curves.
