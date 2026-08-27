# Run Report — V2.0j confirmatory: V2-C arm closeout (V2-C four-seed confirmation)

**Date:** 2026-08-26  
**Preregistration:** `2026-08-24-core-v2-accumulated-macro-policy-prereg` (frozen)  
**Classification basis:** frozen outcome classes vs Core V1 R=`-0.2130`: ALPHA
(delta >= +0.05 AND sigma <= 0.06); BETA (|delta| <= 0.02); GAMMA (delta <=
-0.03, stop); otherwise AMBIGUOUS.  
**Mode:** confirmatory, 4 seeds {42,142,242,342}, 128 cycles, bank digest
`b3bb5fc33fd5f605`, lr `1e-4`, complete-47-batch cycle clipping,
fixed-exposure-mean loss normalization.

## Verdict

**V2-C_accumulated_macro: AMBIGUOUS** (as published; re-verified today from the
sealed result JSON, no re-execution).

Per-seed held-out mean lifts: `[-0.0047, +0.0020, -0.1500, +0.0047]`.

| Quantity | Value | Gate | Pass |
|---|---|---|---|
| mean lift | -0.0370 | — | — |
| delta vs Core V1 | +0.1760 | >= +0.05 | yes |
| sigma (population, n=4) | 0.0653 | <= 0.06 | **no** |
| |delta| <= 0.02 | no | — |
| delta <= -0.03 | no | — | — |

- ALPHA fails on the sigma limb (0.0653 > 0.06). The seed-242 run (lift −0.150)
  is the outlier driving sigma; three of four seeds sit within ±0.005 of each
  other, all five deployed actions retained every seed (unique_actions=5), and
  no seed is non-finite or constant-policy.
- BETA and GAMMA both fail; the arm is not stopped.
- Per the frozen release, AMBIGUOUS **closes this stage's claim window without
  promotion and without a stop directive.** It licenses no retry, no
  sigma-tolerance relaxation, no seed-242 exclusion (post-hoc outlier removal
  against a frozen dispersion gate would be tuning, and is not performed), and
  no extension. V2-A is recorded ALPHA (mean −0.0457, delta +0.1673, sigma
  0.0208) — unchanged from the published artifact.

## What AMBIGUOUS means for downstream work

The stage tested trainability and held-out decision behavior only. The V2-C
(normalized-recurrence) arm's dispersion is the unresolved dimension: the mean
improvement over V1 is real in sign and magnitude (delta +0.176) but exceeds the
frozen seed-to-seed tolerance, so it cannot be converted into a capability claim
on this protocol. This is exactly the failure mode the frozen gate was written
to catch; the negative is preserved verbatim per hostile-scientist standards.

**No re-run.** The confirmatory protocol was executed once as frozen; AMBIGUOUS
is its terminal classification for claim purposes. Downstream, the integrated
training program (Tier 2 of the construction run) proceeds under its own fresh
namespace and preregistration; this stage's result informs architecture
selection (V2-A-style convex gates preferred where dispersion matters) but is
never reopened.

## Provenance

- Artifact: `runs/v20j-confirmatory-20260824/v20j_confirmatory_results.json`
  (schema_version 1, mode `confirmatory`).
- Re-verification method: read-only JSON parse + recomputation of mean/sigma
  (population n) today; reproduced published aggregates exactly.
- No DEV/CPU-QUAL/TEST partitions touched by this report. No model changed.
