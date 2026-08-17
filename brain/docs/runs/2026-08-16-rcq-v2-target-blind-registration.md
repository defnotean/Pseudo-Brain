# RCQ-v2 target-blind registration (2026-08-16)

Status: create-once registration published. No DGX training, no TEST
construction, and no qualification claim.

## What was published

Isolated local builder, CPU-only, CUDA hidden, no bytecode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\dgx\New-RcqV2Registration.ps1
```

| Field | Value |
|---|---|
| File | `registrations/rcq-v2-reference-v1.json` |
| `registration_sha256` | `33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0` |
| `sealed_test_examples_opened` | `0` |
| Qualification id | `rcq_v2_reference_v1` |
| Run id | `dgx-rcq-v2-reference-seed-1702` |
| `source_tree_sha256` | `03a579c88965aea6fa31bcad2084ab69e98d824ac6c8ebfd5e2263f1cd4c0260` |
| `evaluator_bundle_sha256` | `280af1ae3526f9943419cbfef8b14a83a28950006fbe29aae8971ff8b213d00a` |
| `config_raw_sha256` | `d0532c3cd6d3ee4a549cfdc9bcd5c667c12d4a7528e1c0d1cff25d1aebf69985` |
| `config_canonical_sha256` | `b184361881b52189be78dce105ecc59ab52fa15551d63a2a7662a9dd06619366` |

Replacement is forbidden. If source, configuration, or the evaluator bundle
changes after this pin, discard this registration/release pair and start a new
qualification. Do not edit the JSON to "fix" a hash.

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

## Setup that made the builder runnable on Windows

Windows PowerShell strips double quotes from `python -c` payloads, so the
wrapper does not inline the bootstrap. It runs `python -I -B` against
`brain/scripts/run_rcq_v2_registration.py`. `-B` is required: importing the
package would otherwise write `__pycache__` into `brain/src/irene_brain`, and
the source-tree hasher would fail closed.

## What must not happen next

- Do not construct recipient, donor, or guard TEST sequences.
- Do not point generic `Start-DgxBrainTraining.ps1` at this config or run id.
- Do not start the 2,048-update reference until preflight, sync, pretraining
  pin, RCQ smoke, and staging canary all agree with this SHA-256.

Next operator step: [OPERATOR_GUIDE.md](../OPERATOR_GUIDE.md) step 3.
