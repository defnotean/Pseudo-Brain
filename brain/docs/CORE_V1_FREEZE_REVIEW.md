# Pseudo-Brain Core V1 — FREEZE REVIEW (2026-08-22)

**Status at review start:** HEAD `0d4237d`. Constitution CI green (15/15).
Regression run pending (final gate). This matrix follows directive section 13.

## Component matrix

| Component | Current implementation | Promoted? | Evidence | Known weakness | Scaling risk | Latency cost | Constitution | Inclusion rationale |
|---|---|---|---|---|---|---|---|---|
| Sensor/observation interface | 3×32×32 RGB → conv encoder (canonical core) | YES | Locked torture baseline; all campaigns | Fixed resolution | Low (W is cost axis) | ~0.2 ms | A,B,K,L pass | Functional, tested |
| World belief | Implicit in slot state (no separate belief buffer) | YES (as-is) | Phase 2 hostile battery: bound persistent probabilities causal | No explicit belief object limits interpretability | None | 0 | H passes | Present behavior verified; explicit belief = future work |
| Thought registers (slots) | K=32 slots × W=120, slot_identities init | YES | Scaling sweep flat to K=256; Phase 2 capacity effects | K fixed at runtime | LOW (measured to 256) | flat | A,M,L exact | Core identity of PB |
| BrainCell update | Shared GRU + slot attention, C=3 cycles [MEASURED intake deficit] | YES with documented limitation | 4 candidate replacements failed (gated-GRU, evidence-residual, gate-wiring ×2 protocols); σ-stabilization finding preserved | **Idealized-evidence intake deficit: 0/5 seeds causal. Multitask sample efficiency substantially below GRU** | Medium (recurrent serialization) | ~0.9 ms | E,F,G,H pass | Best verified foundation despite limitation |
| Communication/routing | Slot attention within cycles | YES | Hostile battery mechanism work | No dynamic routing targets | Low | included above | A,M pass | Verified behavior |
| Lifecycle | Static keep/expiry blend | YES (static only) | Dynamic-K v0/v1 REJECTED; gate telemetry kept | No adaptive capacity allocation | n/a | 0 | C,I pass | Simplest verified form |
| Memory interface | `retrieved_memory` parameter exists, unused by canonical core | INTERFACE ONLY | Episodic v0/v1 REJECTED; oracle injection showed core-side deficit | Interface dormant until intake solved | n/a | 0 | B relevant | Kept as seam for post-V1 memory program |
| Episodic store | NONE | NO — rejected | Preregistered failures v0+v1 | — | — | — | — | Excluded per gates |
| Uncertainty representation | branch_probability per-slot confidence + hazard heads | YES | Hostile battery binding tests; J/M contracts | Confidence not calibrated | Low | small | J bounded check | Verified binding semantics |
| Consequence prediction | Utility/reward/displacement heads per slot | YES | Phase 2 consequence battery lineage | — | Low | small | E,F pass | Load-bearing for aggregation |
| Action aggregation | Probability-weighted multiplicity-proof Q(a), temperature 0.1 pinned | YES | I contract (duplicate safety); decode-time suppression | Temperature was implicit before tonight; now explicit | Low | small | D,I,J pass | Multiplicity-proof design verified |
| Adaptive compute | NONE (fixed C=3) | NO — halting DESCOPED, Dynamic-K REJECTED | Halting audit: telemetry-only, no savings; DK failed replication | Compute spent uniformly | n/a | 0 | O pass | Simplicity; revisit needs real early-exit |
| Reflex | tanh-bounded ±0.02 sensor reflex | YES | D contract bound-verified | Reflex adds sensor dependence to logits (by design, bounded) | Low | tiny | D pass | Bounded, non-bypassing |
| Training stability | AdamW + grad clip 1.0; collapse documented (Phase 2 bifurcation analysis) | YES | Reliability10 campaign; confirmation runs | Collapse onset varies by seed | Medium | 0 | N pass | Documented protocol |
| Vectorization/runtime | Fully vectorized [B,K,W]; no per-slot loops | YES | O contract timing check; scaling sweep | — | verified to K=256 | — | O pass | Real-time budget met |

## Frozen gates checklist (directive §15)

- ARCHITECTURAL VALIDITY: Constitution CI green 15/15 ✅; permutation exact;
  no bypass (B); action mediation verified (E/F)
- SYSTEMS: latency p50 ~1.09 ms ≪ 16.67 ms real-time ✅; vectorization
  protected (O); K-scaling healthy to 256 ✅
- TRAINING: canonical trainer reproducible from frozen seeds ✅; collapse
  behavior documented (Phase 2 + confirmation rounds); seed protocol
  established (3 screening / 5 confirmation) ✅
- COGNITION: locked torture baseline recorded ✅; Phase-2 strengths retained
  (pending final regression gate) ⏳; rejected mechanisms excluded ✅
- DOCUMENTATION: this matrix + known-limitations section ⏳ finalize after
  regression

## KNOWN LIMITATIONS carried into Core V1 (explicit, per directive §14)

1. **Evidence-intake deficit [MEASURED]:** ideal injected evidence does not
   reliably move decisions (0/5 causal seeds). Four principled fixes attempted
   and preregister-failed: episodic memory v0/v1, gated-GRU, evidence-residual,
   adaptive-gate wiring. Root cause partially localized (update-rule side);
   full fix deferred as a new research program.
2. **Multitask sample efficiency [MEASURED]:** PB −0.215 vs GRU −0.042 mean
   lift on locked suite (PB learns the 25-task mixture substantially less
   efficiently under light budgets).
3. **No adaptive compute [MEASURED]:** all states spend identical cycles.
4. **Variance finding preserved [MEASURED]:** explicit keep/accept decomposition
   reduced training seed variance 7× (screening) / 2× (confirmation) but did
   not produce usable intake — a design input, not a mechanism.

## Verdict

**PASSED — CORE V1 FROZEN at commit `890e4d0` (tag `core-v1`).**

Final gate (torture regression, `2026-08-22-torture-regression.md`):
- PB reproduces locked baseline cleanly (−0.226 vs −0.215, within noise) —
  the temperature pin and CI additions changed no behavior ✅
- GRU n=3 revealed its locked single-seed estimate (−0.042) sat at the
  optimistic end of a high-variance distribution (−0.36..−0.05); baseline note
  updated: GRU's multitask edge is real but seed-erratic, PB uniformly weak ✅

Core V1 = the verified foundation above, WITH its four documented limitations.
Next program: Phase 2.7 throughput/scaling preparation on the frozen core.
