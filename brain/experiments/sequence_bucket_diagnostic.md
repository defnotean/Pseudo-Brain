# Length bucket diagnostic — separate from the active pilot

The active broad pilot uses exact document lengths and remains unchanged.
Native scan kernels specialize by length, causing repeated CUDA compilation.
Evaluate a future optimization that rounds input length up to a multiple of256
and marks all appended targets -100. No original input/label is removed or moved.
The original immutable prompt and supervised-token count remain unchanged.

Before use, verify loss and gradient equivalence on causal models, including
fully ignored trailing readout chunks. CPU checks use float64 tolerance1e-8 for
the recurrent model and float32 tolerance1e-5 for the transformer. CUDA checks
and actual compile/time measurements remain required before a subsequent pilot.
Do not reuse a recurrent state after trailing padding for streaming inference.
Do not modify or restart the currently registered learning run for this diagnostic.

Remote CPU outcome: all3 tests passed in3.21s on the owned Colab runtime,
with CUDA hidden in a separate subprocess. Source SHA256:
8e05b3849eec0af591b86e0e396d6feacb59ba6951189af80c66b87c041c07de.
Evidence and source archive are preserved in
`brain/runs/broad-pilot-20260911-v1/evaluation-preparation`.
No CUDA equivalence or speed claim follows from this CPU result.
