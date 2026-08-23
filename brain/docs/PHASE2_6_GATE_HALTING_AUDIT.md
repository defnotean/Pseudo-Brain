# Phase 2.6: Adaptive-Gate / Halting wire-or-descope audit (2026-08-22)

**Scope decision per owner:** inspection/verdict only — no prereg needed for the
audit itself. Wiring either module into the active core as a behavioral
mechanism requires a preregistration FIRST (boundary respected below).

Both modules live in `brain/src/irene_brain/model/adaptive_thought_gate.py`,
instantiated ONLY by `IreneBrainModel(torch_model.py)` behind
`enable_adaptive_cognition=True`. **The canonical Phase 2/2.6 science core
(`VectorizedPseudoBrain`) contains neither module** — every current-era result
(escalation campaigns, torture baseline, memory v0/v1, Dynamic-K, BrainCell
screening/confirmation) was produced without them.

## Module A: AdaptiveThoughtUpdateGate ("change-my-mind" gate)

| Question | Answer | Evidence |
|---|---|---|
| Reachable in current forward path? | NO for canonical core; YES inside legacy IreneBrainModel when flag set | torch_model.py L462 |
| Receives gradients? | Only via legacy `cognitive_losses` gate_loss; current-era trainers never call it | grep: 0 refs in escalation/torture trainers |
| Changing output changes thoughts/actions? | Yes within IreneBrainModel (modulates thought refresh blend) | torch_model.py L462-474 |
| Duplicates functionality elsewhere? | Partially — lifecycle keep/expiry head already blends seeds into thoughts in the else-branch | torch_model.py L475-480 |
| Phase-2 evidence supporting it? | NONE. All Phase 2/2.6 results ran without it | absent from VectorizedPseudoBrain |
| Params/FLOPs/latency cost | Small (~72k params @ W=120: context_proj 14.5k + gate_mlp 57.7k) | source inspection |
| Vectorized / K=256-scalable? | Yes — shared MLP over [B,T,R,W], no per-slot params | forward() shape flow |
| Preserves permutation invariance? | Yes — slot-symmetric operations only | forward() |
| Clear failure it should solve? | YES — its documented purpose ("sensory match → preserve; hazard → rewrite") is VERBATIM the evidence-intake deficit measured in the BrainCell campaign [MEASURED] | module docstring L4-10 |

**Verdict: PROMOTE-TO-EXPERIMENT.** Orphaned from the canonical core, but it is
the ONE existing mechanism whose intended job matches a replicated, measured
weakness (ideal-evidence intake deficit, confirmed 0/5 seeds causal in the
current core). Owner's caveat honored: static gates (Gated-GRU) and static
keep/accept (Evidence-Residual) both failed — the untested hypothesis is that
conditioning the gate on explicit discrepancy/prediction-error signals differs
from both. Minimal preregistered experiment written BEFORE any wiring
(see prereg file). No core modification until that prereg governs the run.

## Module B: AdaptiveHaltingController

| Question | Answer | Evidence |
|---|---|---|
| Reachable in current forward path? | Legacy IreneBrainModel only | torch_model.py L589 |
| Receives gradients? | Only via cognitive_losses halting loss (needs complexity_level labels); unused in current era | cognitive_losses.py L130 |
| Changing output changes thoughts/actions? | **NO — by construction.** `should_halt` is computed and discarded; the cycle loop ALWAYS executes max_cycles | grep: zero uses of should_halt; L588-597 appends to diagnostics only |
| Duplicates functionality elsewhere? | No | — |
| Phase-2 evidence supporting it? | None | absent from canonical core |
| Params/FLOPs/latency cost | ~7.3k params; adds a forward-pass sync point (`.item()` batch-mean) per cycle — an anti-feature for GPU throughput | forward() L188 |
| Vectorized / scalable? | NO — data-dependent Python bool via `.item()`; breaks vectorization/CUDA-graph capture; halts on BATCH-MEAN confidence (cross-batch coupling) | forward() L185-191 |
| Preserves permutation invariance? | Yes (slot-mean input) | — |
| Clear failure it should solve? | Would be compute/quality tradeoff — but it delivers ZERO compute savings even in legacy form because masked cycles still execute | construction |

**Verdict: DESCOPE from Core V1.** It is pure telemetry today: no causal path to
behavior, no compute savings, a latent `.item()` sync defect, and no supporting
evidence. Archived in place (module file untouched, historical scripts keep
working). If adaptive-depth is revisited post-Core-V1, it needs real early-exit
execution (skip masked cycles) and wall-clock measurement — a NEW mechanism,
preregistered separately.

## Summary verdicts

- **Adaptive-gate: PROMOTE-TO-EXPERIMENT** (prereg frozen before wiring)
- **Halting: DESCOPE** (archived; recorded rationale above)

## Carried finding for Core V1 docs [MEASURED]

Current core has a reproducible evidence-intake weakness (0/5 causal seeds);
explicit decomposition reduces optimization variance (σ 0.0082 vs 0.0186, n=5)
but has not learned reliable causal integration. Both statements enter the
Core V1 limitations section regardless of the gate experiment's outcome.
