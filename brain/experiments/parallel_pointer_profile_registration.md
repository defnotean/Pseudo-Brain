# Parallel pointer readout profile v1

The owner requires a parallel copy head without an avoidable throughput bottleneck.
The current head uses independent query rows, but each call recomputes ptr_k over
all immutable prompt embeddings. Measure that cost before changing caching or
state semantics. Do not infer end-to-end generation speed from this profile.

Use unchanged V4 recurrent weights, FP64, strict determinism, seed1065 on the
authorized Colab GPU. Prompt lengths31/4096/32768 and query-row counts1/32. Include
repeated source tokens, zero-masked padding and disabled query pointer masks.
Compare parallel readout against concatenated independent single-row readouts;
all full-vocabulary logits must be finite and differ by less than1e-6, with the
input prompt unchanged. No serial attention history, generated-token cache or
new persistent state is introduced.

For each case, warm each measured path once and collect three synchronized wall
times; report all samples, median and peak allocated CUDA bytes. Measure parallel
readout, serial row loop and vocabulary readout without the pointer. Report the
isolated row-batching speedup and pointer overhead without claiming a full-model
or transformer-generation speed advantage. Negative overhead from noise remains
visible. Bound the profile to300seconds and preserve partial results on failure.

No training, architecture change, cache optimization or checkpoint modification
is part of this registration. Run only after active GPU jobs are terminal.
CPU preflight is an import check, not a substitute for native measurements.
