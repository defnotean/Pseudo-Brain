# Independent FP64 gradient validation

Registered before execution on 2026-09-11. No model weights, training budget,
data selection, checkpoint, or existing numerical gate changes.

The prior Triton gradient test checked finite nonzero gradients, without an
independent gradient oracle. This is a coverage gap, not evidence of a defect.

On the owned Colab GPU after the live training/evaluation jobs finish, run
`tests/test_native_scan_gradient_reference.py`. Never run this on the gaming PC.
Use the frozen v3 model and scan sources, verifying their hashes before running.

Eight scan cases compare FP64 outputs and gradients for a, b, and optional h0
against ordinary sequential PyTorch autograd: lengths 31, 257, 2048, 2049;
batch 2, width 64; near-one retention and an exact internal reset. Random
upstream derivatives avoid checking only the derivative of a sum of squares.
Observe native dispatch in both forward and backward at lengths <=2048 and
the existing fallback at 2049. Frozen absolute and relative tolerance: 1e-8.

Two model cases compare every parameter gradient against the same independent
serial recurrence at lengths 31 and 257, with the candidate's eight layers and
64-wide recurrent state, reduced residual width 32/vocabulary 128, immutable
prompt pointer, internal reset, ignored labels, and checkpointed chunked loss.
Require all gradients present and finite, and absolute/relative tolerance 1e-8.
This is a gradient correctness check, not evidence of full-width capability,
frontier competence, or maximum-context streaming parity.

Report the entire test result, including any failure; do not loosen tolerance
after inspecting outcomes. Archive source and remote log with v3 diagnostics.
