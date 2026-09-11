# Coding score adapter, frozen before model generation

`score_independent_math.py --include-code` scores the same frozen coding bank
through `evaluation/independent_code.py` and the verified Bubblewrap executor.
Before scoring any model output, every official canonical solution must pass
and every deliberately incorrect return-None control must fail. Dependency or
control failure aborts scoring; it is not a model failure or a valid score.

Accept a complete Python source response, optionally enclosed in one entire
plain or `python` fenced block. Require a top-level function with the official
entry-point name. No syntax repair, body completion, additional prompt prefix,
extraction from surrounding prose, or retries. Execute the official tests once.
Missing, malformed and failed responses count as incorrect in the full task
denominator. Report executor output/status and source identity with each result.

The model generator never reads these tests or canonical solutions. The ordinary
same-interpreter test harness is not an adversarially tamper-proof grader; retain
this limitation in results. These32-task diagnostics cannot prove frontier parity.
