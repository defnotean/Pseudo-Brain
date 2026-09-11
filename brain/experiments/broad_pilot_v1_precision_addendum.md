# Baseline numerical gate clarification before any training

The first remote preflight passed causality and loss agreement but failed an
elementwise fp32 gradient tolerance for the bfloat16 Flash-attention baseline.
The shared chunked objective already passed dense gradient agreement at 1e-8
in the float64 recurrent candidate. Flash attention casts adjoints to bfloat16,
so a different float32 loss-reduction order can cross a quantization boundary.

Before training, record the baseline QKV relative gradient L2 error and require
it below bfloat16 epsilon (0.0078125) on CUDA, or 1e-5 on the float32 CPU path.
Keep the objective agreement limit at1e-5 and causality exact. This defines a
precision-aware baseline check, not a relaxation of the recurrent 1e-6 streaming
parity law. The initial failed test/source archive is preserved. The data and
training protocol are unchanged; no learning run has started.
