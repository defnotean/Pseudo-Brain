# Paired action-loss correction v2

Preserve the failed v1 native gate and completed read-only diagnostic. Recurrent
full/chunk gradients already pass both records with errors below1.3e-15. On the
unchanged transformer, repeated full and repeated chunked runs are bit-exact;
activation-checkpointed and uncheckpointed chunks are also bit-exact. Full dense
ignore-index CE equals the selected-target reference exactly. Chunked readout
changes final-feature gradients and yields parameter errors up to1.56284e-4;
no optimizer update or attention-backend change occurred in the diagnostic.

Correction: only the comparison_transformer_v1 branch of action_trajectory_loss
uses one full readout and ignore-index mean cross entropy. Keep the recurrent
chunk256 route and256-token bucket padding. Both branches retain exact whole
trajectory/prompt/reset validation. No model architecture, checkpoint, inference
path, training data, loss target, optimizer or tolerance change. The transformer
route trades transient memory for the same numerical readout path as reference.
Frozen V4/V5 foundation drivers and archived v1 training bundles stay unchanged.

First rerun the existing three action-loss CPU controls on Colab, CUDA hidden
and one thread. Then use a separate v2 native gate on the original two records
and both unchanged V4 checkpoints, comparing loss and every parameter gradient
against full selected-target CE with unchanged tolerances (recurrent1e-10/1e-8;
transformer1e-6/1e-4). Require finite gradients and record peak allocated memory.
Bound the gate to300seconds; preserve failure without tolerance relaxation.

Separately measure a full8192-token transformer training/backward/one-optimizer
step on a predetermined synthetic token fixture, with the first128 tokens as
immutable pointer prompt, every next token supervised and one task-start reset.
This is the maximum accepted trajectory length. Use seed1066 and the same
optimizer configuration as the policy pilot. Report peak allocated/reserved
CUDA memory, finite loss/gradients/weights and actual GPU total memory. Discard
all fixture-updated weights; no checkpoint is saved or reused. Bound this memory
preflight to300seconds and run only after the corrected native gradient gate
passes. This is a resource check, not policy training or capability evidence.

Only after these checks pass, freeze a new policy-training source bundle and
an explicit v2 amendment that records the full transformer readout. Update the
post-training adapter to bind to that new source digest. Never repurpose the
failed v1 gate, change its artifacts, or silently change a frozen source hash.
