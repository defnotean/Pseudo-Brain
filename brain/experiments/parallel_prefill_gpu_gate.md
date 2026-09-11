# Native GPU prefill gate

Registered2026-09-11 before GPU execution or timing outcomes. This follows the
eight passing CPU tests in parallel_prefill_registration.md. Do not run until
v4 training and its registered independent evaluation are terminal.

Use the completed v4 recurrent checkpoint, verified against its recorded SHA256,
and unchanged full-width model source. Run in an isolated diagnostic source copy;
the frozen training/evaluation sources and results remain unchanged.

Correctness: batch1, lengths31/257/2049/32768, seed1063, nonzero random token IDs,
reset at0 and the midpoint. At32768 include a repeated-token segment4096:8192.
Use all but the final context token as the immutable pointer prompt for the
three shorter cases; use first31 tokens for the long case, consistent with the
existing maximum-context parity diagnostic. No generated-token cache.

Compare every vocabulary logit at every position from the existing parallel
model path against sequential step; then compare the new prefill helper's final
logits, reconstructed state and seven subsequent step outputs against sequential
state. Require finite values,4096-byte float32 state, logit max absolute error
strictly<1e-6, reconstructed-state error<1e-12. Preserve per-case maxima.

Timing: only after each short case's correctness check warms both paths, run
three whole-prefill repetitions with CUDA synchronization, alternating execution
order. Report each duration and median. At257/2049 require parallel median at
least2x faster than the existing serial decoder prefill. Report31 without a
speed gate. Do not time repeated32768 serial passes or infer long-context speed
from correctness-check wall time. This measures initial-context ingestion for
these inputs, not autoregressive decoding or architecture-wide performance.

Overall deadline900 seconds. Any failed bound, OOM, timeout or incomplete case
prevents integration. Never loosen criteria after seeing outcomes. Archive the
checkpoint/source hashes, GPU/torch, raw timings, errors and semantic pass/fail.
Even a pass is not evidence of frontier language/coding/reasoning competence.
