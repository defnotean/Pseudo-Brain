# Staging-canary invariance diagnosis (2026-08-16)

Status: root cause identified and a capture fix confirmed on GB10. This is not
an RCQ competency result. TEST was not opened. The v1 registration is **not**
reusable for a new pin of the fixed tree.

## What failed

Pin-bound smoke passed on release `r20260817t013858z-db1f586ef3a0`. The
staging canary died at the value-head transition:

```text
RuntimeError: frozen-stage invariance failed for action_outputs_sha256
```

Record: [2026-08-16-rcq-v2-smoke-pass-canary-invariance-fail.md](./2026-08-16-rcq-v2-smoke-pass-canary-invariance-fail.md).

## What the probes showed

Bounded diagnosis only. CPU float32 on the workstation, then the Spark
container with the frozen release mounted read-only. Dataset geometry was the
canary split (not sealed TEST).

| Probe | CPU float32 | CUDA bf16+TF32 | CUDA float32 |
|---|---|---|---|
| Two eval captures, no freeze | match | match | match |
| `set_loss_weights` only | match | match | match |
| Flip `requires_grad` freeze mask | match | **mismatch** (`action_outputs_sha256` and `recurrent_states_sha256`) | **mismatch** (same keys) |
| Same freeze, both captures under `inference_mode` + `requires_grad=False` | match | match | match |
| Official `transition_to_stage` | match | **mismatch** (unfixed capture) | **mismatch** |

`non_value_state_sha256` matched in every failing freeze probe. Frozen weights
did not change. CUDA selected different kernels once most parameters had
`requires_grad=False`, even in `eval` + `no_grad`. Consecutive identical
forwards were already bit-identical, so this is not generic two-call noise.

## The fix

`_capture_invariance_snapshot` now:

1. records the live `requires_grad` flags
2. sets every parameter to `requires_grad=False`
3. runs the eval forward under `torch.inference_mode()`
4. restores the flags

That is the audit's intended claim: same-runtime bit identity of action and
recurrent outputs when non-value weights are unchanged. It must not depend on
whether the optimizer freeze mask is already applied.

An overlay of that file onto the frozen GB10 release made
`transition_to_stage` match under canary bf16+TF32. The installed release was
not modified.

## Why DGX cannot continue under v1

This edits `brain/src/irene_brain/training/torch_system.py`, so
`source_tree_sha256` changes. Registration
`registrations/rcq-v2-reference-v1.json` is create-once and must stay. Do not
edit it to match the new tree. Start a newly named qualification, new
immutable release, and new create-only pin. Do not start the 2,048-update
reference on pin `dbcb6afc…`.

Follow-up: live qualification `rcq_v2_reference_v2` is registered at
`registrations/rcq-v2-reference-v2.json`. Record:
[2026-08-16-rcq-v2-reference-v2-registration.md](./2026-08-16-rcq-v2-reference-v2-registration.md).
