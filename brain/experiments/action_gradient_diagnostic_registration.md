# Native transformer action-gradient diagnosis v1

The original native paired action-gradient gate failed on transformer
embedding.weight in the first809-token record. Both recurrent records passed
all95 parameter gradients with maximum errors below1.3e-15. Preserve that failed
gate; do not loosen its tolerances or launch the queued policy training.

Read-only diagnostic on the same unchanged V4 transformer, same first two frozen
policy training records, same strict deterministic native settings. No optimizer
or checkpoint changes. Bound execution to300seconds, equal to the original gate.

For each record compare six backward passes: original full selected-target CE,
an exact repeat, full dense ignore-index sum/count CE, existing checkpointed
chunk256 action loss, its exact repeat, and chunk256 sum/count without activation
checkpointing. Capture the final trunk feature gradient and every parameter
gradient. Report bitwise repeatability, absolute and relative-L2 errors, number
of parameters/elements outside the original atol1e-6/rtol1e-4, and worst-element
values. Preserve per-parameter details and scalar losses, including mismatches.

This distinguishes repeat instability, target selection/reduction effects,
chunk shape, activation checkpointing, and whether differences arise before the
trunk backward pass. It does not itself isolate a particular attention kernel
or authorize a changed tolerance. A later backend/precision intervention must
be separately defined if the evidence motivates one.

PyTorch documents backend-dependent floating-point behavior for SDPA; that is
context for a hypothesis, not evidence that it caused this failure:
[PyTorch2.11 SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).
