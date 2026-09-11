# Recurrent-only bucket v3 — timing gate and conditional learning run

The prior both-model padding gate FAILED because changing the transformer's
Flash attention shape exceeded its frozen logit/loss bounds. Preserve that
failure and do not change its thresholds. This experiment leaves every
transformer input/label/prompt shape exactly as in broad pilot v1.

Only the recurrent language-loss path appends ignored padding to multiples
of256. Reuse its passed full-width GPU equivalence evidence at31/257/2049 from
the preserved gate: largest valid-logit difference4.21885e-15, all loss and
gradient differences below1e-8. Its unchanged helper/model source hashes must
match that report's provenance. Original prompts and all supervised targets
remain unchanged. Generation and parity use unpadded inputs.

Before learning, run the still-pending recurrent timing arms specified in
bucket_gpu_gate_registration.md, using new independent cache directories:
lengths[1025,1031,1057,1111,1173,1259,1281,1347], seed909, full-width model,
one cold then one warm forward/backward pass, no optimizer. Require lower
bucketed cold total and bucketed warm total<=1.25x exact warm total. Record
all outcomes. A failed timing gate blocks this learning run; no silent retry.

If timing and a CPU routing/target-preservation check pass, start a new output
directory and fresh seed198 initializations for both models. Keep the exact
frozen3072 training/192 development documents, tokenizer, example order, token
counts,3072 updates, AdamW schedule and1800-second/model limit of broad v1.
Use train_broad_recurrent_bucket_v3.py. The transformer's loss call is unchanged;
the recurrent loss wrapper alone pads trailing ignored targets. No checkpoint
resume, extra training tokens, alternative precision or outcome-based selection.

Save source/gate/registration hashes, failure/completion status, checkpoint
hashes, initial/final development NLL, samples and trained parity as in v1.
After a complete run, apply the frozen diagnostic evaluator. Its now-observed
64 tasks remain excluded from training and are diagnostics, not fresh frontier
qualification. The v1 partial checkpoint's0/32 coding and0/32 math are preserved.
Completing this small comparison is not success on the user's frontier goal.
