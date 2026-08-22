# PHASE 2.6 FOUNDATION AUDIT — Pseudo-Brain Core (2026-08-22)

**Scope:** audit of the current core against Phase 2 evidence, scaling risk, and
Core V1 freeze criteria (MASTER_ROADMAP §Phase 2.6 / §Core V1). No mechanisms
changed here; this document decides what earns a change.

**Code inspected:** `model/brain_cell.py` (683 L, shared BrainCell: gated blend,
routing top-k, memory attention/write), `model/torch_model.py` (662 L,
IreneBrainModel: belief/working-memory/thought-field/goal-context state layout),
`model/consequence_thought_actuator.py` (270 L, proposal heads + aggregation),
`model/spec.py` (326 L, frozen tensor contracts), `model/adaptive_thought_gate.py`
(192 L, surprise gate + halting controller), `model/phase2_baselines.py`,
plus Phase 2 run records.

---

## Component matrix

| # | Component | Current implementation | Phase 2 evidence | Weakness | Scaling risk | Candidate improvement | Proving experiment |
|---|---|---|---|---|---|---|---|
| 1 | **BrainCell** (`brain_cell.py`) | Gated continuous-time blend + sparse top-2 routed neighbor messages + memory cross-attention; shared weights across K slots. | Causal thought content (transplant/knockout); collapse-resistance at high K. | Update rule never ablated; register mean-pooling history shows update-rule details matter; no evidence the *gating* is doing useful work at scale. | GRUCell per slot is O(K) — fine. Routing scores O(K²) but k=2 keeps it linear-ish. At W=1024 the per-slot attention cost grows. | Small preregistered candidate set (e.g. gated blend vs GRU-cell vs minimal residual) resource-matched on torture tasks. | Torture suite + reliability protocol (n=10 collapse counts). |
| 2 | **Thought registers** ([B,K,R,W]) | R registers per slot; historical mean-pooling commutativity bug fixed; actuator projects r*w→w lazily (rank-dependent layer rebuild — see Battery bug note). | Binding interventions show binding matters when present. | Register semantics unlearned/unaudited; no notion of confidence/provenance per register; lazy projection rebuild is fragile (bit the battery twice). | R>1 multiplies actuator input dims → O(K·R·W²) heads. | Learned register roles with static shapes; explicit per-slot confidence channel; kill the lazy rebuild (fixed rank contract). | Register-ablation intervention suite; binding scrambles. |
| 3 | **Thought lifecycle** | None. All K slots live forever from init; `slot_identities` buffer is the only seed structure. | Collapse-resistance suggests unused slots may act as protective spare capacity [HYPOTHESIS]. | No sleep/wake/retire → interference risk grows with Kmax; can't test "spare capacity protects" without lifecycle. | At K=256, dead slots waste compute and gradient noise. | Learned per-slot activity gate (keep permutation invariance); retire-on-evidence. | Dynamic-K torture task (easy→hard load shift); measure interference. |
| 4 | **Thought routing** | Top-k neighbor messaging inside BrainCell cycle 2+. | Not isolated by any Phase 2 experiment. | Never ablated; unknown whether routing helps or just adds parameters. | O(K²) score computation at large K. | Ablate routing entirely vs denser vs sparser; decide keep/kill on evidence. | Matched ±routing runs on escalation + torture tasks. |
| 5 | **World belief** (`belief` tokens) | Persistent belief token bank, gated updates. | Belief bypass removed; belief feeds BrainCell + actuator queries. | Belief vs thought-field information split unaudited — belief may duplicate hypothesis content or become a covert bypass. | Low. | Information-flow audit: knockout belief path vs thought path on same task. | Pathwise ablation battery extension. |
| 6 | **Uncertainty machinery** | Branch probability head per slot; BCE to staged targets; multiplicity-safe aggregation. | Probability scramble destroys surviving-PB return (−74%) — probability channel is causally real. Calibration unaudited post-fix. | No aleatoric/epistemic separation; calibration only checked indirectly. | Low. | Calibration metrics (Brier/NLL) as first-class telemetry; epistemic-vs-aleatoric probe later. | Calibration telemetry in torture suite. |
| 7 | **Memory hierarchy** | Working-memory tokens + retrieved-memory inputs exist as interfaces; **no episodic store implemented**. | Memory-write gating had a gradient-starvation bug historically (documented in brain_cell comments); causal use unproven. | The biggest structural gap vs PLAN.md's own contract. Interfaces without a store = dead weight. | Episodic store must be vectorized or it will dominate latency. | Implement minimal vectorized episodic store with gated writes/retrieval; prove causal use before expanding. | Write-gate ablation + retrieval-relevance tests. |
| 8 | **Goals/subgoals** | Goal-context tokens exist as persistent inputs; nothing generates subgoals. | Untested. | Entirely aspirational component currently. | Low. | Defer until memory works (goals need memory for subgoal persistence). | Goal-persistence torture task after memory lands. |
| #9 | **Adaptive compute** | `AdaptiveThoughtUpdateGate` + halting controller implemented (192 L) but NOT wired into the frontier campaign models used in Phase 2. | Untested in Phase 2.5 campaigns. | Orphaned module — built, tested at unit level, not part of live core. | Halting must stay batched/vectorized. | Either wire into Core V1 candidate set or explicitly descope. | Anytime-quality curve vs cycles C. |
| 10 | **Self-correction** | Prediction-error features feed the adaptive gate; no dedicated correction loop. | Untested. | Same orphan status as adaptive compute. | Low. | Fold into adaptive-gate workstream decision. | Surprise-driven rewrite test. |
| 11 | **Training stability** | AdamW+clip; collapse happens (GRU 60%, PB 30% on escalation task). | Central Phase 2 finding. | No collapse detection, no loss normalization diagnostics, no recovery tooling. | Worse at scale. | Collapse-detection telemetry (from mechanism-discovery harness) promoted to standard training loop; normalized loss reporting. | Reliability protocol reused as regression gate. |
| 12 | **Systems/scaling** | Vectorized [B,K,W] hot path; 1.11 ms @ K=32. | Real-time proven. | No measurements beyond K=32; routing O(K²) unquantified at K=128+. | Unknown >K=32 — this is the single biggest unknown for "scalable". | Latency/params sweep K∈{32,64,128,256} × W∈{120,240,480}. | Pure benchmarking, cheap — do first. |

---

## Highest-leverage weaknesses (ranked)

1. **No scaling data beyond K=32 (#12)** — the roadmap's central claim is
   "scalable foundation architecture" and we have one measured point. Cheapest to fix.
2. **No episodic memory (#7)** — PLAN.md's own architecture contract includes it;
   current implementation has interfaces but no store. Blocks goals (#8) too.
3. **Lifecycle/dynamic-K missing (#3)** — needed both for efficiency and to test
   whether spare capacity explains collapse resistance.
4. **Orphaned modules (#9/#10)** — adaptive gate/halting/self-correction exist but
   are not wired into the campaign core; either wire or descope honestly.
5. **Stability tooling (#11)** — collapse detection exists in my session harness but
   not in the canonical trainer.

## Recommended Phase 2.6 execution order

1. **Scaling sweep (pure measurement, days):** latency/params/memory at
   K∈{32,64,128,256} × W∈{120,240,480} on GB10. Decides everything downstream.
2. **Torture-suite baseline:** define the ~25 tests (MASTER_ROADMAP §4), implement
   the environments, run current core + GRU baseline → the reference numbers every
   future change must beat or justify itself against.
3. **Episodic memory v0** (vectorized, gated write/retrieve) → causal-use proof.
4. **Dynamic-K/lifecycle v0** → interference + spare-capacity experiments (this may
   explain the Phase 2 collapse result).
5. **BrainCell candidate set** → small preregistered comparison.
6. **Wire-or-descope adaptive gate/halting.**
7. **Constitution regression harness** → CI-runnable invariant checks.
8. **Then Core V1 freeze review.**
