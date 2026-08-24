# Preregistration: Stage V2.0 — Core V2 Deployed-Loss Baseline vs Core V1

**Date frozen:** 2026-08-24
**Status:** FROZEN before any GPU run. No gate may be edited after launch.
**Motivation:** Phase 2.7 established that Core V1's training objective
(replicated per-slot `action_logits` CE) does not supervise the evaluation
path (consequence-utility aggregation → `action_dist`). Stage 3a showed
perfect training loss with zero generalization gain; Stage 3b rejected the
data-diversity hypothesis (AMBIGUOUS, MECHANISM-leaning). Core V2 implements
the contract fix directly: gradient flows through the deployed aggregator.

## Fixed configuration (identical to Stage 3b ARM A unless stated)

| Item | Value |
|---|---|
| Bank | `build_bank_pinned(seed=42, TASKS)`, digest must equal `b3bb5fc33fd5f605` |
| Steps | 6000 |
| Optimizer | AdamW lr=5e-4, weight_decay=1e-4, grad-clip 1.0 |
| Batch | length-grouped B=16, padded groups, modulo repeat |
| Seeds | {42, 142, 242, 342} |
| Determinism | mandatory (`apply_deterministic_mode()`), provenance recorded |
| Eval | 30 episodes/task, eval seed `20260822 + i*7919`, argmax on deployed `action_dist`, lift = acc − chance, identical to Stage 3b `eval_full` |
| Reference R | Core V1 FIXED @6k mean lift **−0.2130** (n=4, σ=0.0071, same bank) |

## Arms (both trained with the DEPLOYED decision loss only)

- **ARM V2-A (`CONFIG_A_DECISION_ONLY`):** minimal architecture — encoder,
  belief updater, thought field, single shared degenerate action head,
  multiplicity-proof weighted-mean aggregator. Isolates the contract change.
- **ARM V2-C (`CONFIG_C_FULL`):** full Core V2 — consequence heads
  (reward/hazard/confidence/existence/branch), world-model pending predictions
  (detached), episodic memory + session latent present but INERT in this
  harness (no writes occur; single supervised pass per sequence). Tests the
  contract fix inside the full architecture.

Explicit scope limits (declared up front): V2 has no button-control intake;
the torture harness supplies `ctrl = 0` and constant `dt`, so no information
is lost relative to V1's runs. Consequence heads receive NO direct labels —
any learning they show is implicit through the deployed-loss gradient.

## Frozen outcome classes (evaluated per arm, independently)

Let `Δ = mean_lift(arm) − R = mean_lift(arm) − (−0.2130)`.

- **OUTCOME α (contract fix works):** `Δ ≥ +0.05` AND seed `σ ≤ 0.06`
  → deployed-loss training materially improves held-out generalization.
  Next: V2.1 (direct consequence supervision) + TE-V2-batched ARM-B
  matched-training-loss disambiguation.
- **OUTCOME β (parity):** `|Δ| ≤ 0.02`
  → contract fix alone insufficient at 6k on the finite bank; bottleneck is
  upstream of the objective (evidence intake). Resume 3c0-style diagnostics
  ON V2 (gradient audit, oracle-consequence, aggregator ablation) before any
  scaling or longer runs.
- **OUTCOME γ (regression):** `Δ ≤ −0.03`
  → V2 skeleton underperforms V1; STOP. Root-cause locally before any
  further GPU claims. No rescue tuning within this stage.
- **Otherwise:** AMBIGUOUS. Report full per-seed distributions for both arms;
  no gate shopping; no post-hoc band adjustment.

The two arms are an exploratory CONTRAST (different architectures), not
replicates; only per-arm comparisons against R are confirmatory.

## Secondary (recorded, non-gated)

- Final training deployed-loss curve (every 100 steps).
- Per-slot CE at decision frame (diagnostic only, computed under `no_grad`,
  never optimized).
- Thought diversity proxy: fraction of the 32 slots whose argmax matches the
  deployed argmax at the decision frame (eval time).
- Mass entropy: entropy of `mass_per_action` at decision frame.
- Wall time per run.

## Falsification discipline

Any deviation from this document discovered post-launch (bank digest
mismatch, eval-seed drift, nondeterministic execution) voids the run:
results are reported as VOID with the defect named, and the stage reruns
only after a corrected, re-frozen prereg.
