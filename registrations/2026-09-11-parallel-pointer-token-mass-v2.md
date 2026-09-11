# Parallel pointer token-mass supervision, version 2

Registered before v2 training and before v1 task evaluation completes.
The v1 diagnostic found 2,920 of 3,559 supervised pointer targets have another
prompt position with identical key and predecessor features. An exact-position
loss asks this head to distinguish scores that are identical by construction.
The generated output is a token, so equivalent copy positions should contribute
their combined probability. This diagnosis uses only the training bank.

Change only the optional auxiliary attention objective from probability at one
position to the sum of attention at all prompt positions containing the correct
target token. Keep gate supervision, token loss, architecture, optimizer,
retention loss, state budget, and decoding unchanged. Store the selected loss
in run metadata and checkpoint metadata. Preserve positional loss as the default
for historical runs. Metrics must distinguish positional and token-mass accuracy.

Test identical-token probability aggregation, invariance to prompt-position
permutation, finite nonzero gradients, masked-target exclusion and unchanged
legacy objective. This adds no inference input or oracle information: supervised
labels are used only by the training loss.

After focused tests and native invariant preflight, permit one fresh 350-step
seed-1337 warm start from the original champion on the exact corrected bank.
Use the owned Colab A100, compensated state and parallel_predecessor mode,
float64 with AMP/TF32 disabled, maximum one hour. Preserve v1 and legacy runs.
Evaluate the existing fixed 30-task development suite with four actions and
per-task resets. Compare completion, exact binding, and verified repairs; do not
promote on lower auxiliary loss. Repeated inspection makes this a development
comparison, not a newly sealed or frontier benchmark. Final qualification needs
an independent frozen evaluation set untouched by this development cycle.

Engineering note added before v2 execution: prompt/observation ingestion may
omit discarded intermediate vocabulary logits. All cognitive state tensors and
the next prediction were checked for exact equality, with 33 focused tests
passing. A 29-token local CPU diagnostic measured 389.84 ms full readout versus
39.82 ms final-only readout, five measured repetitions after one warmup. This
changes neither the optimizer trajectory nor decoding outputs. Include the
exact source in the v2 manifest; do not modify the running v1 release.
