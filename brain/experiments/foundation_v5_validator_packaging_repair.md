# V5 validator packaging repair

The first driver preflight failed on import, before model initialization or any
optimizer update: the isolated training bundle lacked
irene_brain.evaluation.independent_pilot. The driver had been changed to call
that module's finite/state-aware parity validator after its earlier import-only
check. Preserve the failed bundle, preflight log and source manifest.

Create a separate r1 training bundle containing the existing audited module
from the frozen independent checkpoint evaluator. Do not change the corpus,
model, optimizer, training schedule, driver logic, preflight cases or thresholds.
Freeze a new complete source manifest and rerun the CPU-hidden CLI import first,
then the identical64-vs32+32 native driver preflight. Full training stays gated.
The added module contains validation/scoring functions, but no benchmark answers
or test cases; the training driver calls only validate_streaming_parity.
