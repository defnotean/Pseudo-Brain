# Resumable language training preparation

The v4 pilot saved inference weights and tokenizer but omitted optimizer/RNG
state. Prepare exact optimizer-boundary restart support before larger training.
Reuse training.checkpoint's atomic, non-overwriting, restricted-loading envelope.
Preserve all v4 artifacts, source hashes and registered training/evaluation paths.

The opt-in language_checkpoint adapter saves model buffers/parameters, AdamW
moments and parameter groups, training mode, Python and Torch CPU/CUDA RNG, and
the existing TrainerCursor. Verify source/data/config hashes, model configuration,
named optimizer parameter order and numerical runtime before restoring. Reject
uncleared gradients: accumulation mid-step is deliberately unsupported.

The caller must include tokenizer, loss policy, total learning-rate schedule,
optimizer configuration and deterministic example order in its frozen run
identities. This adapter does not capture NumPy RNG, custom generators, data
workers or streaming packer buffers. Current pilots do not use those for updates;
a future caller needing them must extend and validate the contract first.

Before use in longer training, compare an uninterrupted update sequence with a
saved/reconstructed/resumed sequence using real recurrent and transformer models,
varying token lengths and response masks. Require identical subsequent losses,
model parameters and optimizer tensors on the same runtime; verify next RNG draws
and cursor, mismatch rejection and preservation of an existing checkpoint.
First run reduced-width CPU tests on Colab with CUDA hidden; native/full-width
CUDA restart equivalence remains a separate required gate before long GPU use.
Do not infer native determinism or capability from CPU restart correctness.
