# Native carried-state ingestion gate

Run once on the existing trained V4 recurrent checkpoint SHA256
7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf,
after the active V5 training controller is terminal and before any other GPU job.
This checkpoint and core already passed initial-prefill validation. No updates,
model edits, decoding policy changes, or benchmark score claims are authorized
by this gate. Preserve any failure; do not retry under the same registration.

Use the isolated Colab ingestion source which passed 12 CPU tests, with unchanged
prefill/model/scan files. Full width, vocabulary and pointer remain enabled.
Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python, use strict deterministic mode
through run_provenance, and record checkpoint, source, token-bank and runtime
provenance. Evaluation seed 1064. A 900-second alarm bounds the native gate.

Start each case from a nonzero state produced by 31 initial prompt tokens. The
pointer source stays that immutable prompt throughout. Ingest lengths
1/31/257/2048/2049/32768 without reset, including repeated tokens at 4096:8192
in the longest case. Also check 257 tokens with batch two and independent resets
at the midpoint/end. Repeat the single-stream 257 case in chunks 1/3/13/31/61/148.

Compare every returned boundary's full-vocabulary logits against ordinary
streaming steps, require finite logits/states, absolute logit error <1e-6,
decoded high/low state error <1e-12, and exactly 4096 bytes per stream in a
16x64 float32 state. Require input state and prompt immutability. Follow each
case with seven predetermined serial tokens and require logit parity <1e-6.
The helper returns only boundary logits; this gate does not claim to compare
intermediate logits it does not return. No timing/speedup claim is part of this
gate. Passing does not establish autonomous POMDP or frontier competence.
