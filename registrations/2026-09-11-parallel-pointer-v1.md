# Parallel pointer candidate, version 1

Registered before GPU execution or candidate training. The owner requested a
parallel pointer head and explicitly selected Google Colab as the accelerator.

The existing attention recurrence depends on the previous softmax and argmax;
its time loop is serial. This experiment replaces that recurrence, without
changing its parameter tensors, with independent time rows:

`score[t,i] = content[t,i] + seq_boost * (i > 0 and prompt[i-1] == input[t])`.

Padding cannot be a predecessor or a copy target. All-padded prompts contribute
zero. The immutable initial task specification is the only pointer source.
No generated-token history or previous attention is retained. Duplicate
predecessors are ambiguous and must be resolved by content scores; this is an
accuracy risk, not a claim of equivalence to the legacy mechanism. Old
checkpoints default to `sequential`; new checkpoint metadata records the mode.

Gates before training:

- Batched and streaming logits differ by less than 1e-6, including masks,
  duplicated prompt tokens, padded prompts, and internal resets.
- Causality, meaningful pointer gradients, selective readout, and unique-token
  copy continuation pass. The physical fast state remains 4096 bytes and
  trainable parameter count remains below 36 million.
- Native A100 checks use the actual checkpoint and lengths 31, 257, and 513.
- Compare legacy and candidate pointer-only forward and forward/backward
  latency, plus complete model forward/backward with the same batch shapes,
  precision, supervised positions, warmups, and device synchronization. Record
  medians and peak GPU allocation; no end-to-end speedup claim from head-only
  timing. Benchmark shapes B=4, T=128/512, N=64, 3 warmups and 10 measured runs.

If these gates pass, permit one 350-step seed-1337 warm start from the original
champion on the same corrected procedural bank as the Colab portability run.
Use compensated state, float64 arithmetic, no AMP/TF32, and a maximum one-hour
job. Evaluate A/B/C with 10 tasks each, four actions, reset per task. Preserve
all source hashes, checkpoints, traces and receipts in a new run directory.
Do not overwrite or promote the champion. Compare task completion, exact
module/function binding and verified repairs against the legacy Colab run;
loss or synthetic copying alone does not qualify the candidate. Do not expand
training on a failed gate; diagnose or preregister a revised experiment.

Separately fix reset-mask initialization: the scan must start with the learned
initial state even if its first reset occurs after token zero. Test before and
after the internal reset. This applies to both pointer modes.

This is an engineering and limited procedural capability experiment. It does
not establish general-purpose or frontier-level ability.
