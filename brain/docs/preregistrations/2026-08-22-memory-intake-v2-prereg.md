# Preregistration: Memory-Intake v2 (relevance-conditioned per-register integration)

**Frozen:** 2026-08-22, BEFORE any v2 training. **Conditional:** executes only if
`evidence_residual` clears its already-frozen confirmation gates. If confirmation
fails, this document is void and the BrainCell workstream re-plans.

## Candidates compared (3-arm, matched)

| Arm | Description |
|---|---|
| A | current BrainCell (frozen control) |
| B | Evidence-Residual v1 (candidate control — only if it passed confirmation) |
| C | Memory-Intake v2: relevance-conditioned per-register keep/accept |

Matched: K=32, W=120, same actuator/head, same torture tasks, same optimizer
(AdamW 5e-4, wd 1e-4), 6000 steps, seeds {42,142,242} for screening.

## Architecture (smallest principled change over v1)

Per-register gates over the existing register dimension:

    new_r = keep_r * old_r + accept_r * relevance_evidence_r + residual_update

- keep/accept gates: [B,K,R,1], computed from [thought; evidence; observation]
  through a SHARED adapter (no per-slot private networks, no Python loops over K).
- relevance conditioning: a shared Evidence Adapter scores evidence against the
  current thought; accept is modulated by the relevance score.
- Registers remain learned latent state; slot IDs semantically meaningless;
  whole-slot permutation invariance exact; vectorized hot path; no belief→action
  or memory→action bypass.

R (register count) = W split into R=W/8 groups of 8 dims (practical per-register
granularity without parameter explosion). Exact factorization may be adjusted
only BEFORE the run, recorded here as an addendum with timestamp.

## Causal battery (frozen — conditions A–I from the owner's directive)

A correct-relevant · B zero · C donor/wrong · D irrelevant-structured ·
E stale · F contradictory-credible · G register-specific modification ·
H norm-matched noise · I present-but-unread.

Tasks: t01/t17-style cue tasks for A–C, H; dedicated abstract contradiction
task for F (belief established from early cue, counter-cue later — no semantics
hardcoded); distractor-evidence variants for D, E, G, I.

## Per-condition telemetry (all recorded, all conditions)

decision accuracy · Δaction probability · thought-state change norm ·
per-register change · keep/accept gate stats (mean/std/min/max/p10/p50/p90) ·
relevance score · unrelated-information preservation (cosine on untouched
registers) · confidence change.

## Metrics (preregistered definitions)

- RELEVANT EVIDENCE GAIN = acc(correct) − acc(zero)
- DONOR STEERING = P(donor action | donor) − P(donor action | normal)
- IRRELEVANT SENSITIVITY = |decision shift from condition D|
- STALE REJECTION = acc preferring current evidence over superseded memory
- CONTRADICTION RESPONSE = confidence drop or appropriate revision under F
- PRESERVATION = cosine of untouched registers pre/post update under G

## Promotion gates (numeric, frozen now)

Screening (3 seeds) → confirm (5 seeds) requires ALL:
1. RelevantEvidenceGain ≥ +0.10 on memory-dependent tasks, all seeds
2. DonorSteering ≥ +0.10 (evidence pathway causally steers decisions)
3. IrrelevantSensitivity ≤ 1/3 of RelevantEvidenceGain
4. StaleRejection ≥ 0.60 absolute
5. ContradictionResponse: confidence decreases OR state revises (either
   measured), not ignored (shift ≠ 0 at p < 0.05 sign test)
6. Preservation ≥ 0.8 cosine on untouched registers
7. Causal signature (1–3) holds in ≥ 4/5 confirmation seeds
8. Torture lift ≥ v1 mean − 1σ (no multitask regression beyond noise)
9. Escalation collapse rate ≤ current BrainCell + 0.05; Phase-2 battery no
   material regression
10. Exact permutation invariance (property test, all seeds)
11. Latency p50 ≤ 1.5× evidence_residual measured p50 on the same DGX run

## Failure handling

Any gate failure → record FAIL with diagnostics. One substantially-different
revision permitted ONLY if telemetry identifies a concrete cause. No re-tuning
until pass. MIXED verdicts recorded as MIXED.

## Same-store retest (only after PASS)

Rejected Episodic Memory v1 store, unmodified, + {old BrainCell, new BrainCell},
same protocol as the v1 fail run. This isolates intake vs store/retrieval
bottleneck. Then instrument the full pipeline: write → retrieve → integrate.

## Systems measurements (required)

params, FLOP estimate, latency mean/p50/p95/p99, peak VRAM on GB10, same
harness as screening. Tensor structure [B,K,R,W]-compatible.
