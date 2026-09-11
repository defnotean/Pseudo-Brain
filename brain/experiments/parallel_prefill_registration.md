# Parallel initial-prompt prefill diagnostic

Registered2026-09-11 while the frozen v4 training run is active. Do not modify
that run's model source, weights, training functions or benchmark decoder.

An opt-in helper scans the complete initial context in parallel through each
of the existing eight recurrent layers, retains the final logical state of each
layer, converts it to the existing paired float32 representation, and reads out
only the final position. It returns exactly the ordinary4096-byte per-stream
state. Full-sequence features are transient workspace. No generated-token history
or persistent key/value cache is introduced. The immutable task prompt is caller
input as before. This does not parallelize autoregressive generation.

First run CPU functional tests on Colab with CUDA hidden: lengths1/31/257, batch2,
pointer on/off, unequal per-stream reset positions, seven subsequent tokens,
1e-6 absolute logit tolerance and1e-12 reconstructed-state tolerance. Check shape,
physical state bytes and no gradient graph. Also reject empty and non-FP64 inputs.

Native GPU parity, trained-checkpoint correctness, and timing are still required
before integrating this path. Run those only after current GPU training and
registered checkpoint evaluation are terminal. Preserve frozen decoder results;
record this as a separate inference optimization with its own source hashes.
Do not infer a speedup from CPU functional tests or claim frontier capability.
