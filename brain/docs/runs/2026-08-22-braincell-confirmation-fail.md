# Phase 2.6: Evidence-residual confirmation round — FAIL per preregistered gates (2026-08-22)

**Protocol:** gates frozen in `2026-08-22-braincell-screening.md` before the run.
5 seeds × {current, evidence_residual}, torture lift + escalation collapse +
permutation property + synthetic-memory probe. Data: Spark
`runs/bc_confirmation_results.json`.

## Results [MEASURED]

| Metric | current | evidence_residual |
|---|---|---|
| Torture mean lift (n=5) | −0.1943 ± 0.0186 | −0.2145 ± **0.0082** |
| Escalation collapse rate | 0.015 | 0.005 |
| Permutation exact (all seeds) | ✅ | ✅ |
| Memory causal signature | 0/5 seeds | **1/5 seeds** |

Per-seed memory probes (correct/zero/wrong): evidence_residual is inconsistent —
s442 shows the full causal pattern (0.275/0.150/0.075), but s142 and s342 show
*inverted* patterns (wrong evidence HELPS: 0.35 wrong vs 0.10 correct). Current
shows flat zero/causal response on all 5 seeds, replicating screening.

## Gate verdicts

- G1 lift no-regression: **FAIL** (−0.2145 < −0.1943 − 0.0186; outside noise,
  small but real)
- G2 collapse no-worse: PASS (0.005 ≤ 0.015)
- G3 permutation exact: PASS
- G4 memory causal 5/5: **FAIL** (1/5)

**PROMOTED_TO_CORE_V1_CANDIDATE = false. Evidence-residual v1 is NOT promoted.**

## Interpretation [INFERRED — flagged for the next experiment]

1. The screening's 3/3 causal signature did NOT survive to n=5 — the seed spread
   warning in review was correct. The keep/accept pathway CAN carry evidence
   (s442) but training does not reliably shape it into a causal channel.
2. The variance reduction replicated (σ 0.0082 vs 0.0186 at n=5): the
   decomposition stabilizes optimization even though it does not yet deliver
   causal intake.
3. Lift regression is small but outside noise; combined with failed intake this
   is a clean FAIL, not MIXED.

## Consequences

- Evidence-residual v1 → REJECTED for Core V1. Recorded honestly.
- `Memory-Intake v2` prereg (`2026-08-22-memory-intake-v2-prereg.md`) was
  conditional on confirmation PASS → **VOID by its own terms.** Not executed.
- BrainCell workstream returns to candidate design with two hard constraints
  learned: (a) variance stabilization is achievable via explicit decomposition;
  (b) causal intake requires more than a static gate pair — the accept pathway
  must be *trained against evidence contrastively*, not merely parameterized.

## What survives

- σ-reduction finding (decomposition stabilizes training) [MEASURED, n=5]
- The ideal-evidence intake deficit of the current core [MEASURED, replicated]
- All telemetry harnesses (gate inspection, memory probes, escalation eval)
- Phase-2 strengths remain untouched (current core still the reference)

Phase 2.6 workstream order resumes: adaptive-gate/halting decision →
constitution CI → Core V1 freeze review, carrying these negatives as inputs.
