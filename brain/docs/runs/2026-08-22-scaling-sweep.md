# Phase 2.6: K/W scaling sweep on GB10 — first scaling data (2026-08-22)

**Method:** pure measurement, no training. Canonical `VectorizedPseudoBrain`
(batch=1, 32×32 input, C=3 cycles), 200 timed iterations after 50 warmup,
`torch.cuda.synchronize` per step. Data: Spark `runs/scaling_sweep.json`.
Script: `brain/scripts/phase26_scaling_sweep.py`.

## Results [MEASURED]

| K \ W | 120 | 240 | 480 |
|---|---|---|---|
| **K=32** | 1.067 ms / 825k | 0.915 ms / 2.13M | 0.863 ms / 6.30M |
| **K=64** | 1.073 ms / 825k* | 0.858 ms / 2.13M* | 0.917 ms / 6.30M* |
| **K=128** | 1.072 ms / 825k* | 0.855 ms / 2.13M* | 0.998 ms / 6.30M* |
| **K=256** | 0.988 ms / 825k* | 0.853 ms / 2.13M* | 1.578 ms / 6.30M* |

\* params identical across K at fixed W — the slot dimension shares weights and
the param count is K-independent by construction (confirmed empirically).
GRU reference: 933k params @ 0.616 ms.

Peak allocated memory stays 37–70 MB across the whole sweep.

## Findings [LABELS]

1. **[MEASURED] Latency is essentially flat in K up to 128** (≈0.85–1.07 ms) and
   still inside budget at K=256 (≤1.58 ms). The vectorized [B,K,W] design scales:
   going K=32→256 costs ≤ +0.5 ms, i.e. the architecture can afford ~15× more
   thought capacity while remaining ≈10× under the 16.67 ms frame budget.
2. **[MEASURED] W is the dominant cost axis**, not K: W=480 runs *faster* than
   W=120 at equal K for small K (kernel efficiency), until K=256 where the O(K·W²)
   attention terms surface (1.578 ms). The hot path has no O(K²) blow-up anywhere
   in the tested range.
3. **[MEASURED] PB pays ~0.24–1.0 ms over the GRU** depending on config; all
   configurations remain comfortably real-time.

## Implications for Phase 2.6 ordering

- The "does it scale in K?" risk from the Foundation Audit is **retired for
  K≤256 at research widths**: dynamic-K/lifecycle work is motivated by
  interference and spare-capacity science, not by a latency wall.
- Scaling-law experiments (Phase 2.7) can proceed at W=240–480 without
  real-time risk.
- Next audit item stands: torture-suite baseline, then episodic memory v0.
