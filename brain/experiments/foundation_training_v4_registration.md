# Foundation learning v4: conditional training protocol

Registered2026-09-11 before any v4 training. Start only after the separately
registered foundation corpus finishes with all quotas, hash checks, complete
response labels and isolation checks passing. Freeze its manifest hash explicitly
in the launch command and archive it first. No silent quota or dataset changes.

This is a fresh experiment after both v3 models failed the64-task capability
diagnostic. It changes data coverage, response length distribution and total
updates. It is not a continuation of v3 or evidence that either architecture
already rivals frontier models.

Use the unchanged v3 recurrent and transformer architectures, tokenizer,
parameter counts, precision, pointer implementation and loss paths. Recurrent
training alone uses256-token length buckets; baseline tensors stay unchanged.
The v3 helper runner is frozen at SHA256:
1b4f833db439a2031d2f085085fe7082baeebd084ea9f2281c4d08a4f7b8e547.
Reuse its verified training/evaluation functions rather than rewriting them.

For each model independently, initialize from seed198, train one pass over all
20480 frozen training examples in identical order, and evaluate all384 development
examples at initialization and at the final fixed checkpoint. One complete
document per optimizer update, response-only mean next-token loss. AdamW3e-4,
betas(.9,.95), epsilon1e-8, weight decay.1, gradient norm clip1; warmup64 then
cosine to3e-5 at the final update. Both models train from scratch; never load v3
weights or select a checkpoint using benchmark outcomes.

Bound training to1800 seconds per model, as before. Save the final or partial
checkpoint and full per-update log. A timeout/nonfinite value/OOM is incomplete,
not a completed equal-exposure comparison. The current helper stores weights
and tokenizer, not a resumable optimizer state; do not claim exact resumability.
No repetition of a failed long run or automatic budget extension.

After training, record token-weighted NLL by domain, fixed six sample outputs,
and recurrent trained31/257 parity under1e-6. Preserve every failure. If complete,
evaluate the same registered64-task diagnostic without changing questions,
decoding limits or scorers; recheck longer trained parity. Report prior benchmark
observation and GSM8K training-split exposure. Fresh frontier qualification remains
separate. No checkpoint promotion based solely on this experiment.

Record exact source/config/tokenizer/corpus hashes, parameters, peak memory and
elapsed time. All execution stays on the user's Colab; no local model work.
