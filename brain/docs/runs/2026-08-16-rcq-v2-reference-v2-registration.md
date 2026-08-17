# RCQ-v2 reference qualification v2 registration (2026-08-16)

Status: create-once live registration published. No DGX training, no TEST
construction, and no qualification claim. Historical
`registrations/rcq-v2-reference-v1.json` is unchanged.

## Why a new name

The v1 Spark pin passed smoke and failed the staging canary: CUDA kernel
selection followed the live `requires_grad` freeze mask, so
`action_outputs_sha256` changed across an otherwise identical value-head
transition. The capture path now saves `requires_grad`, forces every parameter
off, runs under `torch.inference_mode()`, and restores the flags. That changes
`source_tree_sha256`. Protocol forbids editing the v1 JSON; this is a newly
named qualification with the same TEST ranges and the same training config.

## What was published

Isolated local builder, CPU-only, CUDA hidden, no bytecode. The registrations
directory already held historical v1, so the builder now resolves an existing
parent instead of requiring an empty exclusive receipt root. Only the live
filename is create-once.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\dgx\New-RcqV2Registration.ps1
```

| Field | Value |
|---|---|
| File | `registrations/rcq-v2-reference-v2.json` |
| `registration_sha256` | `6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690` |
| `sealed_test_examples_opened` | `0` |
| Qualification id | `rcq_v2_reference_v2` |
| Run id | `dgx-rcq-v2-reference-seed-1702` |
| Pin directory | `qualification-pins/rcq-v2-reference-v2` |
| `source_tree_sha256` | `c49777ea45ec6bd5b238e76c78a9cb25db0a6f270bd7cdb3299da97545b40a35` |
| `evaluator_bundle_sha256` | `fcbf79e8e597ef646f62792e5f5f386cc64938716fbf7ce0565d11476ec00303` |
| `config_raw_sha256` | `d0532c3cd6d3ee4a549cfdc9bcd5c667c12d4a7528e1c0d1cff25d1aebf69985` |
| `config_canonical_sha256` | `b184361881b52189be78dce105ecc59ab52fa15551d63a2a7662a9dd06619366` |

Config hashes match historical v1. Source and evaluator-bundle hashes do not:
they include the invariance-capture fix and the live v2 identity strings.

Historical v1 file SHA-256 remains
`33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0`.

Replacement is forbidden. If source, configuration, or the evaluator bundle
changes after this pin, discard this registration/release pair and start a new
qualification. Do not edit either JSON to "fix" a hash.

Copy `registration_sha256` to a note that is **not** this repository and **not**
a training run directory. The in-repo copy of the file is required for the
immutable DGX release; the external copy of the digest is the launch pin.
`.gitattributes` pins this JSON to LF so a Windows checkout cannot silently
rewrite the bytes and break the pin.

## What this registration binds

Geometry and identities only:

- TRAIN `[1048576, 1056768)` — 8,192 sequences, 49,152 scored decisions
  (8,192 × 6). Those decision counts are sequence geometry, not opened labels.
- Development `[1048576, 1048832)` — 256 sequences, 1,536 scored decisions.
- Sealed recipient TEST `[3145728, 3146240)` — 512 sequences, 3,072 scored
  decisions reserved, not constructed.
- Sealed donor TEST `[3146240, 3146752)` — 512 sequences, zero scored
  decisions in registration.
- Unused guard `[3146752, 3147264)`.
- Future matched-campaign offset `2097152`.

The payload contains no final-label statistics, no changed-action counts, and
no value aggregates from sealed ranges.

## What must not happen next

- Do not construct recipient, donor, or guard TEST sequences.
- Do not edit `registrations/rcq-v2-reference-v1.json`.
- Do not chmod an installed immutable release writable.
- Do not point generic `Start-DgxBrainTraining.ps1` at this config or run id.
- Do not start the 2,048-update reference until preflight, sync, pretraining
  pin, RCQ smoke, and staging canary all agree with this SHA-256.
- Leave the v1 pin directory historical. Create the new pin in
  `qualification-pins/rcq-v2-reference-v2/`.
- The first v2 pin is unused. See
  [2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md](./2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md).
- Live follow-up: smoke and canary passed, reference train started.
  [2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md](./2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md).

Next operator step: [OPERATOR_GUIDE.md](../OPERATOR_GUIDE.md) step 3.
