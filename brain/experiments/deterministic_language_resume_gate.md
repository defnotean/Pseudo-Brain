# Strict deterministic native restart gate

The default-mode transformer prefix control completed on2026-09-11: two fresh
three-update runs differed in45 model tensors (max8.719624e-5) and110 optimizer
tensors while Python/CPU/CUDA RNG states and model identities matched. Divergence
therefore occurs without restoration. The particular contributing kernel is not
yet identified. Preserve the original failed restart gate and control artifacts.

Register a new native restart gate with the same full-width models, seed198,
seed965 tensor bank, six-update schedule, optimizer/loss/padding, three fresh
subprocess phases and exact comparisons as language_resume_gpu_gate.md.
Change only execution determinism: set CUBLAS_WORKSPACE_CONFIG=:4096:8 before
importing Torch and call the existing run_provenance.apply_deterministic_mode.
This requests strict deterministic algorithms, cuDNN deterministic mode,
disables cuDNN benchmarking and TF32. Do not use warn_only or silently select
a different attention backend if an unsupported operation throws.

The checkpoint adapter additionally records CUBLAS_WORKSPACE_CONFIG in its
runtime fingerprint, so checkpoints cannot silently cross that setting.
Freshly freeze all source hashes in an isolated new directory. Do not edit
the v1 gate's runner/source manifest or call its failed results passing.

Same180 seconds per fresh phase and1200 total; no GPU overlap. Require exact
prefix/suffix losses, final model/optimizer tensors, per-update/next RNG draws
and cursor. A failure leaves this new gate unpassed. A pass establishes this
specific same-runtime restart check, not universal determinism or capability.

Reference: https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html
and https://docs.pytorch.org/docs/2.11/notes/randomness.html . These describe the
deterministic execution controls; our measured gates establish their behavior
on this workload. Run only on Colab, with no larger training until the gate passes.
