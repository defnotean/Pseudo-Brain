# Phase 2.5: ephemeral-memory reliability replication (n=10) — NEGATIVE for PB (2026-08-22)

**Machine:** gx10-db18, GB10. **HEAD:** ee238fa.
**Derived script:** `ephemeral_reliability10_dump.py` (10 training seeds {42..942},
configs {GRU, K=32}, 1500 steps, per-seed dump; derivation inline, syntax verified
after a heredoc-escape corruption was caught by compile-check before launch).
Console: `runs/ephemeral-reliability10-v1/console.log`.

## Per-seed normal returns [MEASURED]

| Rank | GRU | PB K=32 |
|---|---|---|
| 1 | −123.5 | −220.4 |
| 2 | −136.3 | −258.9 |
| 3 | −185.1 | −280.8 |
| 4 | −301.4 | −301.4 |
| 5 | −354.8 | −503.7 |
| 6 | −450.1 | −504.1 |
| 7 | −486.2 | −504.7 |
| 8 | −498.4 | −512.6 |
| 9 | −504.6 | −517.4 |
| 10 | −511.6 | −522.1 |

## Statistics

- Permutation test (one-sided PB better): **p = 0.802** — no advantage; GRU nominally better.
- Catastrophic seeds (≤ −500): GRU 3/10 vs K=32 6/10, Fisher p = 0.47 — not significant
  either direction, but the sign is *against* the reliability story here.

## Interpretation [LABELS]

[MEASURED] The escalation-task reliability advantage of PB K=32 did **not replicate** on
the ephemeral-memory task: per-seed returns are statistically indistinguishable with a
point estimate favoring GRU. The earlier 5-seed ephemeral result (GRU wins return) stands.

[INFERRED] The Phase 2.5 picture is now task-split:
- 8-hypothesis multi-stage uncertainty: PB K=32 significantly more reliable than GRU
  (fewer catastrophic training collapses, p≈0.027–0.036).
- Ephemeral single-cue memory: GRU equal-or-better on return; PB's accuracy edge does
  not convert to control.

[HYPOTHESIS] Parallel hypothesis slots pay off specifically when multiple latent
variables must be simultaneously maintained and separately updated (the escalation
task's joint LL/LR/RL/RR structure), not when a single memory must simply persist.

## Where this leaves Phase 2

This is close to the §33 closure condition forming honestly: one defensible positive
(task-specific reliability), one defensible negative (no return advantage on ephemeral).
Remaining before closure memo:
1. Latency/params table re-measurement on GB10 (resource accounting).
2. Persistence causal battery on surviving K=32 escalation checkpoints (mechanistic why).
3. Optionally: a second multi-latent-variable task to test the HYPOTHESIS above.
