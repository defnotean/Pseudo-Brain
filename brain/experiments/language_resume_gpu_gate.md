# Native language restart gate

Run only after the active v4 evaluation and parallel prefill GPU gate terminate.
Use the same authorized Colab A100 and frozen v4 model/scan sources. No overlap
with GPU work. This validates training infrastructure, not learned capability.

For each full-width model, run three fresh subprocesses: six uninterrupted
updates; three updates plus optimizer-boundary checkpoint; reconstruct and
restore that checkpoint, then perform the remaining three updates. Seed198,
unchanged default models and AdamW(3e-4, betas .9/.95, eps1e-8, decay .1), clipping1.
Use six fixed synthetic whole-document examples of lengths31,257,2049 repeated;
recurrent-only256 buckets, response masks, pointer to the immutable first half.
LR at zero-based update i is3e-4/(i+1). These are diagnostic inputs, not a new
capability training corpus. Hash actual tensors and record init provenance.

Require bit-identical suffix losses, model state, all optimizer state, next
Python/CPU/CUDA RNG draws, per-update RNG sentinels, and cursor3. Preserve all
differences/failures; do not relax equality after seeing results. Use ordinary
runtime precision and record deterministic flags rather than silently changing
the v4 numerical path. Each subprocess has180 seconds; total gate has1200 seconds.
Preserve source, logs, tensor-bank identity, boundary digest and results. Checkpoint
tensors remain on Colab until results are archived. CPU checks do not substitute
for this gate. No longer run may use the new adapter unless this gate passes.
