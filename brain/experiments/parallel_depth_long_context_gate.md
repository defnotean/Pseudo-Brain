# Parallel depth v1 — bounded long-context execution gate

After the v1 architecture gate passes, run one forward/backward diagnostic on
the longest of the 96 immutable context-audit samples, without truncation or
optimizer updates. Fixed bank SHA-256:
d931e2f42519a4dc2be90f2426d45453ce353cf841bbf47c8193de808adbbe74.

Use the default 384-wide, eight-layer candidate, fixed seed 198, existing 32K
BPE tokenizer, response-only targets, whole-document packing, and an immutable
pointer prompt consisting only of tokens before the first response marker.
Use readout chunks of 256. Bound the diagnostic to 600 seconds on the owned
A100 Colab runtime. Require all 18,574 tokens to be ingested, finite loss and
finite nonzero language gradients in all eight recurrent layers. Record
supervised tokens, wall time, peak CUDA allocation and exact source hashes.

This tests execution and learning-path feasibility, not full-length streaming
parity, held-out generalization, or model competence. No optimizer step or
checkpoint promotion is authorized by this gate. A failure requires diagnosis.
