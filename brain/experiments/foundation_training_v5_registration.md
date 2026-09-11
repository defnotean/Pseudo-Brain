# Foundation training v5: deterministic, resumable, broader learning

Conditional on the complete registered v5 corpus:69632 training and1536
development documents, verified manifest/source/tokenizer hashes and isolation.
Use the unchanged full-width recurrent and transformer models, fresh seed198
for each, one pass in the identical frozen document order. No prior weights.
Response-only mean next-token loss and recurrent-only256-token buckets remain.
Use strict deterministic settings from the separately passed native restart gate:
CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python, deterministic algorithms required,
cuDNN deterministic with benchmarking off and TF32 off. Record this numerical
change; v4 historical results and artifacts remain unchanged.

AdamW3e-4, betas(.9,.95), epsilon1e-8, decay.1, clipping1. Warm64 updates then
cosine decay to3e-5 at the last of69632 updates. All resumes retain the global
step and full schedule; they do not restart warmup. One complete document/update.

Use immutable optimizer-boundary checkpoints every4096 updates and on a planned
pause or completion. Save model/optimizer/RNG via the tested checkpoint adapter;
save a hash-bound receipt with cursor, committed token counters and cumulative
training time. A latest pointer may reference only a fully published receipt.
Require explicit receipt hash plus checkpoint/data/config/source/runtime checks
on resume. Logs are immutable per attempt; never overwrite failed-attempt evidence.
No exact resume claim for older v4 inference-only checkpoints.

Limit cumulative update-loop plus checkpoint-writing time to3600 seconds/model,
checked between updates. Finish an already running update and save a valid boundary;
record any overshoot and call a time-limited run incomplete. Initialization,
encoding, restoration and development evaluation are measured separately. A
planned update-count pause may resume within the same total budget. Unexpected
failures do not authorize an automatic repeat or budget extension.

Before full training, validate the new driver itself on the first64 frozen v5
training documents for both models:64 uninterrupted vs32 + save/reconstruct +32,
using the full69632-step schedule. Require identical all64 logged losses/LRs,
final model/optimizer/RNG tensors, cursor64 and token counters. This is an
implementation gate, not model capability. Bound each preflight process180s;
skip development for this preflight only. Preserve and diagnose failures.

For full training evaluate all1536 development documents at initialization and
completion, save inference-compatible weights/tokenizer, six fixed sample outputs,
and trained recurrent parity. Then perform registered independent capability
scoring with unchanged tasks and decoding limits. The observed benchmark has
already influenced research direction and GSM8K training exposure is explicit;
this is not fresh frontier qualification. No promotion based only on loss.
