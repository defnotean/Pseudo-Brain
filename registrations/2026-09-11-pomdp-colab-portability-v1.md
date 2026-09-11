# POMDP Colab portability and corrected-feedback candidate v1

Registered before remote allocation/training. The owner explicitly selected their
Google Colab setup instead of DGX Spark on 2026-09-11 UTC. Local verification stays
CPU-only; this registration authorizes a bounded Colab A100 experiment.

First upload a source archive, exact cached 32k tokenizer, and unchanged historical
POMDP checkpoint, with a manifest that verifies every payload hash after extraction.
Do not git-pull an unrelated remote checkout or depend on unpublished changes.
Run the actual checkpoint parity/budget audit in compensated mode on A100, including
native Triton execution and batched sequences of 31, 257 and 513 tokens. Require
max absolute streaming/parallel logit error <1e-6, <36M parameters, and a physical
16x64 float32 fast working tensor. Also compare native scan forward and backward
results against the float64 PyTorch scan. No AMP or TF32; preserve float64 math.

Only after these pass, allow at most 350 optimizer steps, seed 1337, the existing
120-task open procedural curriculum, response-only loss and supervised-position
readout. This remote candidate uses the corrected real feedback validator (unique
bytecode cache prefix and compact actual exception), and consumes every returned
token before EOS even on repetition cutoff. These are explicit differences from
the already running, snapshotted CPU compensated-v1 experiment. Do not present
the remote run as an exact resume of that CPU job or a new independent hypothesis.

Evaluate ten tasks per A/B/C tier with seed 42, reset per task, maximum four actions.
Preserve every trace and the original acceptance thresholds. Save the tokenizer in
the candidate checkpoint. Download artifacts before releasing the session; do not
assume a Drive mount exists. A failed scientific gate blocks longer training.
Both candidates remain separate from the historical champion, with no promotion
or frontier claim unless the corresponding evidence actually supports it.
