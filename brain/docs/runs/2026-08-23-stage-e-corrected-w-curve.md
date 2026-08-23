# Phase 2.7 Stage E: corrected deterministic W-scaling curve (2026-08-23)

**Design:** frozen architecture; W ∈ {120, 240, 480} × seeds {42, 142, 242,
342} × 6000 steps; pinned banks (`crc32` offsets, bank digest
`b3bb5fc33fd5f605` verified cross-process in Stage A); strict deterministic
execution (validated bitwise-reproducible in Stage C). Locked eval.
Data: Spark `runs/det_E.json`.

## Results [MEASURED]

| Width | Params | Mean lift | Seed σ | Per-seed lifts |
|---|---|---|---|---|
| 120 | 0.82M | **−0.227** | ±0.013 | −0.242 / −0.237 / −0.214 / −0.214 |
| 240 | 2.13M | −0.232 | ±0.039 | −0.285 / −0.174 / −0.233 / −0.238 |
| 480 | 6.30M | −0.257 | ±0.057 | −0.311 / −0.209 / −0.315 / −0.191 |

## Interpretation

**The director's Outcome 3: the apparent W-scaling improvement was artifact.**
Once episode banks are pinned and kernels deterministic:

- Mean lift does not improve with width — it mildly degrades
  (−0.227 → −0.257).
- Seed variance grows monotonically with width (σ 0.013 → 0.039 → 0.057):
  wider cores are *less* stable under the fixed light-training recipe, but
  not better on average even at their good seeds.
- The v1 sweep's headline (W480 mean −0.172 incl. a −0.009 seed) is
  attributable to its confounds: different per-process banks + ULP-level GPU
  drift amplified by training dynamics.

## Conclusions

1. **Width is not the scaling axis for Core V1's torture performance.** This
   is consistent with Phase 2.6's K-scaling finding: parameter count (via K)
   was also flat. Both primary capacity axes show flat-to-negative returns at
   this budget; the binding constraint remains optimization/sample efficiency
   and the evidence-intake deficit, not capacity.
2. Core V1 stays W=120 — now supported by a clean measurement rather than
   default conservatism.
3. Scaling-law work should pivot axes: training regime/budget (steps, LR
   schedule), data curriculum, or architectural mechanisms targeting intake —
   before any larger parameter counts.
4. The σ growth with width is itself a real, reproducible phenomenon worth
   one diagnostic pass later (gradient-scale vs width) if wide configs are
   ever revisited.

## Provenance record (adopted permanently per owner directive)

Every future run saves: bank digest · init param digest · training seed ·
eval seed · deterministic flag · torch/CUDA versions.
