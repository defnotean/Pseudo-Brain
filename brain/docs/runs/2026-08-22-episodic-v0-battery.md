# Phase 2.6 Workstream G: Episodic Memory v0 — first causal battery (2026-08-22)

**Setup:** PB K=32 (frozen recipe) + `EpisodicMemoryV0` (64-entry vectorized
key-value store, learned write gate + key encoder, 77,241 memory-specific params,
61 KB store). Trained 4000 steps on t17/t18 + fillers. Six-condition battery run.
Data: Spark `runs/episodic_v0_results.json`.

## Battery results [MEASURED]

| Condition | t17 (recall) | t18 (stale rejection) |
|---|---|---|
| normal | **0.175** | **0.250** |
| store disabled | 0.075 | 0.200 |
| present-but-unread | 0.075 | 0.200 |
| correct value injected | 0.175 | — |
| donor value injected | 0.075 | — |

Telemetry: 92,824 writes attempted / 89,350 committed (96% gate-open rate),
93,784 retrievals.

## Interpretation [LABELS]

[MEASURED] The six-condition signature is **qualitatively correct but weak in
magnitude**:
- normal > disabled/unread on BOTH tasks (+0.10 t17, +0.05 t18): the store's
  contents causally contribute beyond mere module presence — and "unread"
  exactly matches "disabled", so there is no hidden side-channel benefit.
- correct-injected (0.175) = normal; donor-injected (0.075) is *worse than
  disabled*: injecting wrong evidence actively misleads, i.e. retrieval content,
  not just its presence, drives behavior.
- BUT absolute accuracies are near chance (t17 chance=1/3, t18 chance=1/2) and
  well below the owner's success bar ("normal → succeeds"). At 4000 steps the
  model has not learned to *use* memory reliably; it has learned only a weak
  dependence on it.
- Write-gate telemetry shows the gate is saturated open (96%) — it is not yet
  doing selective writing. Retrieval precision was not yet measurable against
  ground truth keys (needs instrumentation next round).

## Verdict against the owner's bar

**NOT YET PASSED.** The causal direction is right (all six conditions behave as
predicted qualitatively), but effect sizes are small and absolute performance is
below the bar. Per the directive: this does not earn Core V1 candidacy yet.

## Diagnosis and v0→v1 plan

1. **Undertrained**: escalation needed 3000 steps for ONE task; here 4000 steps
   covers 4 tasks. Next: 12–16k steps, t17/t18-only curriculum first.
2. **Write gate saturated**: add a sparsity pressure or lower temperature so
   writes become selective (96% open ≈ no selection).
3. **Retrieval precision unmeasured**: instrument next run with known-key probes.
4. **Injection strength** (2.5 amplitude) may saturate; sweep injection gain.
