# Opt-in parallel prefill in the independent decoder

Apply only after the registered trained native prefill gate passes all lengths,
including32768, with the4096-byte state and registered timing thresholds.
Add an explicit opt-in to greedy_decode and the pilot generation CLI. Keep the
default serial path and all frozen v4 evaluator source/artifacts unchanged.
Reject the option for the transformer. Preserve immutable initial prompt,
greedy token selection, EOS/length behavior and streaming recurrent continuation.
Record the selected prefill mode and helper hash in new generation metadata.

Verify CPU token-for-token equality against the serial decoder at31/257 context,
pointer on/off, fresh state each call, and explicit rejection for transformer.
Run only on Colab. The trained native gate already covers full-vocabulary logits,
state and continuation; integration additionally checks exact generated tokens
for the first two registered HumanEval tasks and first registered GSM8K task
against their archived v4 serial responses, unchanged512-token greedy limit.
Those three observed tasks are integration checks, not new capability evidence.
Run that native check after the separately registered GPU restart gate terminates.
Preserve any mismatch and do not alter old responses or silently fall back.
