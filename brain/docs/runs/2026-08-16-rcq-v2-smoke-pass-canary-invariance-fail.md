# RCQ-v2 smoke pass and staging-canary invariance failure (2026-08-16)

Status: pin-bound smoke passed. Staging canary failed during the schema-3
stage transition. The 2,048-update reference was **not** started. TEST was
not opened. This is not an RCQ competency result.

## Live identities

Registration SHA-256 is unchanged:
`33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0`.

| Item | Value |
|---|---|
| Release | `r20260817t013858z-db1f586ef3a0` |
| Archive SHA-256 | `db1f586ef3a03c49f297f56436d142438b13d50a2d829f84fd888126137a23e1` |
| Pretraining pin file SHA-256 | `dbcb6afc798c18121491af6688933b039d81f80e0ff8baf0de060cd7494b113b` |
| Pretraining pin semantic SHA-256 | `89214c15d5e83153bef7d33986b190ae0f2054f3a48e2ae4b58720da3a217b78` |
| Cached image ID | `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b` |
| `source_tree_sha256` | `03a579c88965aea6fa31bcad2084ab69e98d824ac6c8ebfd5e2263f1cd4c0260` |
| Smoke receipt | `/home/defnotean/projects/pseudo-brain/smoke-receipts/r20260817t013858z-db1f586ef3a0.env` |
| Smoke receipt SHA-256 | `0d1ab2d25f490d1170db839684fcee7497690df00e208c5502ac14007511816b` |
| Canary run id | `rcq-staging-dbcb6afc798c1812` |
| Canary log | `/home/defnotean/projects/pseudo-brain/logs/rcq-staging-dbcb6afc798c1812.log` |

The two earlier unused pins never received a smoke receipt. They remain
historical only:
[2026-08-16-rcq-v2-unused-pretraining-pin-discard.md](./2026-08-16-rcq-v2-unused-pretraining-pin-discard.md).

## What smoke proved

Foreground pin-bound smoke on GB10:

- CUDA bf16 backward passed.
- Isolated unit-test modules passed, including the two launcher-test fixes
  (writable evaluator copy; OS-independent absolute checkpoint paths).
- Bounded trainer smoke completed 3 optimizer steps.
  Checkpoint SHA-256
  `0f7fb9d2cb1cec088b7dc5f9e935b3c16a5634b3d7cfb2ff4adca65cd464e9fe`.

That is not a development gate and not a value-head freeze.

## What the canary did

`Invoke-DgxRcqStagingCanary.ps1` with 2 CPUs and 8 GiB. Config
`brain/configs/training/dgx-rcq-v2-staging-canary.toml` (`build_smoke_model`,
2 updates, seed 1703). Phase `initial stop_after_step=1` started. After the
first optimizer step the trainer attempted the registered transition into the
value-head-only stage (`invariance_audit = "nonvalue_action_state_v1"`).

It failed before resume:

```text
RuntimeError: frozen-stage invariance failed for action_outputs_sha256
```

No staging-canary receipt was written. The exclusive canary run directory
`runs/rcq-staging-dbcb6afc798c1812` now exists, so this pin cannot retry the
same canary identity. Keep that directory as a failed operational artifact.

## Why the 2,048-update train did not start

The reference recipe uses the same `nonvalue_action_state_v1` audit after the
development gate. The canary is operational, not scientific, but a failed
canary is still a stop. Do not treat this pin as launch-ready.

The two invariance snapshots are consecutive eval forwards of the same
weights: capture, then `_configure_optimizer` (freeze mask, new AdamW, stage
loss weights), then capture again. Action outputs are hashed from
`ActionPrediction` fields, not from `value_per_thought`. A mismatch means
those two no-grad forwards were not bit-identical on this GB10 / CUDA 13 /
bf16 / TF32 runtime.

Do not chmod the installed release. Do not open TEST. Do not edit
`brain/src` or the RCQ reference TOML under this registration. A source or
reference-config fix is a new qualification. A canary-only retry still needs
a new immutable release and a new create-only pin because this canary run id
is consumed.

Diagnosis (2026-08-16): CUDA selects different eval kernels after the freeze
mask flips `requires_grad`, even though non-value weights are bit-identical.
See [2026-08-16-rcq-v2-canary-invariance-diagnosis.md](./2026-08-16-rcq-v2-canary-invariance-diagnosis.md).
