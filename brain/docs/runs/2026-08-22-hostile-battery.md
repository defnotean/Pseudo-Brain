# Phase 2.5: hostile mechanistic battery — results (2026-08-22)

**Checkpoints:** 12 preserved from the discovery run (3 surviving-PB, 3 collapsed-PB,
3 surviving-GRU, 3 collapsed-GRU). **Interventions:** normal, reset-every-frame,
stale D∈{1,5,20}, probability scramble, binding scramble, permutation (PB only).
Data: Spark `runs/hostile-battery-out/battery_results.jsonl` (per-seed detail included).

## Mean returns by cell × intervention [MEASURED]

| Cell | normal | reset | stale1/5/20 | prob-scramble | binding-scramble | permute |
|---|---|---|---|---|---|---|
| **surviving-PB** | **−127.3** | −195.4 | ≈−126.8–127.0 | **−364.1** | **−364.1** | −127.3 (=normal) |
| collapsed-PB | −532.5 | −532.2 | ≈−532.5 | −480.1 | −480.1 | −532.5 |
| **surviving-GRU** | **−189.8** | **−133.9 (improves!)** | ≈−189.8 | −285.3 | −285.3 | n/a |
| collapsed-GRU | −528.0 | −428.2 (improves) | ≈−528 | −528.6 | −528.6 | n/a |

## Findings

1. **Surviving PB is causally thought-mediated. [MEASURED]** Probability scramble and
   binding scramble each destroy ~74% of its return (−127 → −364); frame-reset costs
   ~54%. Stale state D=1..20 is harmless and permutation is exactly neutral. The
   surviving high-K checkpoints genuinely depend on correctly-bound persistent
   hypothesis probabilities.

2. **Surviving GRU is *anti*-persistent. [MEASURED, surprising]** Its return
   *improves* when reset every frame (−190 → −134). Scrambles hurt it less than PB
   (−190 → −285). The winning GRU solution does not rely on a well-bound persistent
   hypothesis structure the way PB does — if anything its carried state is mildly
   harmful at evaluation time.

3. **Collapsed checkpoints of both architectures look alike. [MEASURED]** Collapsed-PB
   and collapsed-GRU sit at the catastrophic floor under every intervention, are
   indifferent to stale/reset, and their scrambles slightly *improve* returns
   (−532 → −480 / −528 → unchanged). A collapsed model has no functional hypothesis
   structure left to scramble — consistent with collapse = loss of usable internal
   structure rather than miscalibration.

4. **Permutation invariance held everywhere.** Exact zero effect on both surviving
   and collapsed PB.

## Mechanism picture [INFERRED]

The reliability result now has a causal correlate: what distinguishes a surviving
high-K PB checkpoint is not just higher return but a *qualitatively different
solution* — one that maintains and uses correctly-bound persistent per-hypothesis
probabilities (scramble-sensitive), whereas the GRU's successful solutions avoid
persistent commitment entirely (reset helps). The GRU "wins" by refusing to carry
hypotheses; PB wins by carrying them safely. On tasks where carrying is necessary,
the GRU's strategy has a 60–70% chance of collapsing into the no-structure attractor
during training; PB's slot architecture resists that.

Caveats: stale-D semantics here only rewind before the final decision; battery used
3 seeds/cell; single task. Fresh-seed confirmation remains the gate before claiming.

## Next

Fresh independent confirmation run: new 10+10 seeds {GRU, K=32}, same envelope,
architecture untouched — then Phase 2 closure memo.
