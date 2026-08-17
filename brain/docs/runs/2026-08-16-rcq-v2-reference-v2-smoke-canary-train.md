# RCQ-v2 live qualification smoke, canary, and reference start (2026-08-16)

Status: pin-bound smoke passed. Staging canary passed, including the schema-3
value-head transition invariance receipt. The 2,048-update reference is
running detached on Spark. TEST was not opened. This is not an RCQ competency
result.

## Live identities

Registration SHA-256
`6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`.
Do not rebuild it. Historical v1 JSON is unchanged.

| Item | Value |
|---|---|
| Qualification id | `rcq_v2_reference_v2` |
| Release | `r20260817t021531z-7a2967ebec60` |
| Archive SHA-256 | `7a2967ebec6004787eabd8695a5b75d41117fb61fe8d15e77faace1bb49d7f69` |
| Pretraining pin file SHA-256 | `adf79ccccbb98c73e0d8104ef5b575b5bf260de0bec3b8ac9c0d81dc4fedf2ba` |
| Pretraining pin semantic SHA-256 | `552a3f9b43b1c283f623cc37248077e3a5054f39ceed4e936d646e658c78df69` |
| Pin created UTC | `2026-08-17T02:16:00Z` |
| Cached image ID | `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b` |
| `source_tree_sha256` | `c49777ea45ec6bd5b238e76c78a9cb25db0a6f270bd7cdb3299da97545b40a35` |
| `evaluator_bundle_sha256` | `fcbf79e8e597ef646f62792e5f5f386cc64938716fbf7ce0565d11476ec00303` |
| Smoke receipt | `/home/defnotean/projects/pseudo-brain/smoke-receipts/r20260817t021531z-7a2967ebec60.env` |
| Smoke receipt SHA-256 | `41dabbad602900f18bc34ac398e35ec5dd528a059da2a1bba1839721e1c366db` |
| Canary run id | `rcq-staging-adf79ccccbb98c73` |
| Canary receipt SHA-256 | `fb152fcec3862273cd7634cfbfba6262e4980297c6b3c203c19bca6179950eca` |
| Reference run id | `dgx-rcq-v2-reference-seed-1702` |

The first live-v2 pin (`2847e786…`) never received a smoke receipt and was
discarded:
[2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md](./2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md).

## What smoke proved

Foreground pin-bound smoke on GB10:

- CUDA bf16 backward passed.
- Isolated unit-test modules passed. Historical v1 registration pinning
  skipped because that file is git-only.
- Bounded trainer smoke completed 3 optimizer steps. Checkpoint SHA-256
  `563f7bc4a9a730c0366ea8f45af52e1bcd440ca8414f4c5affb17bd3865059e8`.

That is not a development gate and not a value-head freeze.

## What the canary proved

`Invoke-DgxRcqStagingCanary.ps1` with 2 CPUs and 8 GiB. Config
`brain/configs/training/dgx-rcq-v2-staging-canary.toml`. Phase `initial
stop_after_step=1` wrote checkpoint schema 2
`672f41c08dacc85cccba7256343c80287c5ea81ff8cb1260c6a700346ca562b8`. The resume
phase exact-resumed that checkpoint and completed update 2. Final checkpoint
SHA-256
`1ddca8fcb10cb25fe3b5e54d20b2b6e1c9696864939b95f74dee45e0472340e3`.
Invariance receipt SHA-256
`5c1d03d078efa64ade58bed54c155e534eb1e157412ad931b587fd2fce3d1577`.
Stage-transition SHA-256
`945b748559487e44560bb6e4432b19c121fce4bd5aab0ad6a762d3e9ed039df7`.

This is an operational check, not RCQ competency. It does confirm that the
live capture path (freeze-mask `requires_grad` saved, all parameters off,
`torch.inference_mode()`, then restore) keeps action outputs bit-identical
across the value-head transition on this GB10 / CUDA 13 / bf16 / TF32 runtime.

## Reference train

`Start-DgxRcqV2Reference.ps1 -AcknowledgeDetached` launched after smoke and
canary receipts agreed with the pin.

- Handshake: `/home/defnotean/projects/pseudo-brain/runs/dgx-rcq-v2-reference-seed-1702/launch.ready`
- tmux session and container: `pseudo-brain-dgx-rcq-v2-reference-seed-1702`
- Config canonical SHA-256 `b184361881b52189be78dce105ecc59ab52fa15551d63a2a7662a9dd06619366`
- `max_optimizer_steps`: 2048, precision bfloat16, two stages
- Status about 18 seconds after launch: tmux running, container up, GPU busy

First checkpoint at optimizer step 256 (~24 minutes after launch):

| Item | Value |
|---|---|
| Checkpoint | `step-00000256.pt` (338 MiB) |
| Checkpoint SHA-256 | `95f607675808ea0f012edc4738443192d7835197a54f78a0c405d4b007d6d426` |
| Train loss | 0.489 |
| Train movement exact match | 0.229 |
| Validation loss | 0.485 |
| Validation movement exact match | 0.247 |

That validation slice is the open development namespace, not sealed TEST.
These numbers are not a development gate. The entry gate remains step 1,536.

Host disk is about 550 GiB free after the non-3.8 weight cleanup. Do not chmod
installed releases writable. Do not open TEST. Resume only with
`Resume-DgxRcqV2Reference.ps1 -AcknowledgeDetached`. The entry gate is at
optimizer step 1,536. Failure there is terminal for this candidate.
