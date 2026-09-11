# Parallel depth v1 — architecture diagnostic registration

Hypothesis: the existing single affine language recurrence and 64-wide recurrent
readout limit learned context composition. Merely increasing retention did not
improve the development capability gate. Use eight trainable affine recurrent
layers, interleaved with nonlinear residual feed-forward computation, before
considering broader training. The existing constructed recurrent stack is not
used by the current unified language logits.

Candidate: width 384, eight logical states of width 64, eight paired float32
residual slots, tied 32K embedding/readout, feed-forward expansion four, optional
independent-time predecessor pointer. No generated history, KV cache or replay.
Float64 parameters/arithmetic support strict parity; fast state alone is 4096 B.
The slots represent layers, not eight independently addressable task threads.
The existing POMDP adapter is not yet integrated; no champion replacement.

Before training, require on remote Colab:
- Trainable parameters strictly below 36M; physical state exactly [B,16,64]
  float32, constant after arbitrary steps.
- Native GPU parallel/streaming logit max absolute error below 1e-6, with
  reset boundaries and a repeated-token/padded immutable pointer prompt.
- No future-token influence; all-pad prompts contribute exactly zero.
- Nonzero finite gradients in each recurrent layer's gate, candidate and output
  projection from language loss. Compare current unified module gradient coverage.
- Chunked recomputed language loss and parameter gradients agree with dense
  loss within 1e-8 on a small case with ignored labels and pointer masking.
- Checkpoint round trip preserves logits. Existing regression tests remain intact.

Failure means diagnose this candidate; do not extend training or promote it.
This registration authorizes architecture diagnostics only, not a capability
claim. A subsequent bounded training comparison must fix corpus, exclusions,
token budget, parameter/compute comparison and fresh held-out evaluations.
The 96 pinned source examples are diagnostics, not a representative qualification
set. Long-context gates and POMDP adapter integration remain separate work.
