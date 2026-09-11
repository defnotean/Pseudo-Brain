# Parallel-depth episode integration

Connect the tested token session and isolated tool environment in a bounded
episode API. Each task creates a fresh session with its immutable initial prompt.
Each action uses the fixed RESP marker and raw greedy tokens. Do not strip or
repair malformed actions, inject phase hints, replay history, or accept truncated
actions. Require a separate externally verified FINISH for success.

After each executed action, ingest only its new JSON observation frame. Escape
brackets inside JSON data to avoid literal tokenizer control markers; preserve
the exact decoded text. This is framing, not prompt-injection protection.
Record complete raw output and feedback in an audit trace that is never fed
back to the model. Keep4096-byte recurrent state separate from permitted prompt,
environment state, returned output and audit records.

Expose positive integer limits for cycles, action tokens, observation tokens and
initial prompt tokens. Reject overlong initial prompts. Stop incomplete on
overlong observations without silently truncating them or executing further
actions. No model weights, decoder policy or active V5 bundle changes.

Colab CPU checks, CUDA hidden/one thread: scripted wrong-code/test/fail-finish/
edit/test/verified-finish control through real isolated tools; exact framing and
event ordering, no history replay, raw invalid output rejection, truncated action
nonexecution, observation bounds and invalid argument rejection. An actual
reduced-width model must repeat a fresh-task episode after an unrelated task and
must not self-certify success. Scripted test doubles are infrastructure controls,
not evidence of learned repair. Native trained autonomous evaluation still required.
