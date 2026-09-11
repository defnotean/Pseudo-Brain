# Transformer tool-loop comparison preparation

Add a separate full-prefix session for the existing transformer baseline. It
retains every action/observation token and recomputes the full prefix; it does
not satisfy the recurrent4KB constraint and is not an optimized generation-speed
baseline. Keep its pointer source the immutable initial prompt. Consume emitted
EOS exactly once, preserve raw greedy output, and report prefix length explicitly.
Use a32768-token cap with explicit context_limit failure, no silent truncation.

Reuse the same episode execution loop through a session-construction hook.
Recurrent behavior remains the default, with4096 state bytes and no prefix
history. Transformer results use state_bytes=null and report retained prefix
tokens. Same raw-action handling, tool environment, observation framing, bounds
and external FINISH validation. Preserve all frozen prior source/evidence roots.

Colab CPU verification, CUDA hidden/one thread: exact agreement with independent
full-prefix transformer evaluation across three turns, EOS ingestion, immutable
prompt, fresh-task/context-limit behavior, and shared-loop memory/failure
reporting. Rerun the existing seven recurrent episode controls because the shared
loop changes. No new GPU jobs or model updates. Native transformer rollout and
capability comparisons remain pending.
