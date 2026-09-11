# Executed clean MBPP trajectories v1

The target-order probe did not restore module binding. Preserve its negative
result and keep the audited original MBPP prompts unchanged. Add broader actual
implementation demonstrations rather than more prompt-order tuning.

Use all357 audited official-training cases from the corrected r2 MBPP bank,
gzip e7645a3c98749815098ec66293e583087180b06c6bfd89249d6b1bb96f04d5d0. Verify every
case digest, training ID and private-validator hash. Preserve stored case order,
all programs/prompts/files and source provenance; no filtering/replacement quota.

For each case execute exactly READ_FILE test_public.py, WRITE_FILE target with
the original reference program, RUN_TESTS, FINISH. Use the same isolated workspace
and original private validator. Require all four actual actions to succeed and
FINISH to verify completion. Save every attempted trace before admission; any
failed case, encoding, identity check or infrastructure error makes the collection
incomplete. No silent repair of upstream programs or tests.

Encode with the unchanged v1 scripted-teacher codec: only teacher action/EOS
labels, one task reset, immutable initial prompt as pointer source,8192 whole
record/1024 action+EOS/4096 prompt+observation limits, no truncation. Actual
program execution stays on Colab CPU. Hard deadline600seconds. No actor/model
generation, optimization or independent capability scoring. These357 scripted
demonstrations supplement the58 actual model-prefix continuations and previous
varied-fault examples; they do not themselves prove learned competence.

Freeze source before execution. Record counts, per-case sizes/hashes, raw actual
feedback and compressed corpus identity. Archive and hash-verify after download.
Any model training still requires a separately frozen mixture, schedule and
matched evaluation protocol with foundation retention measurements.
