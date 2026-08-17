# RCQ-v2 live qualification smoke, canary, and reference start (2026-08-16)

Status: pin-bound smoke passed. Staging canary passed, including the schema-3
value-head transition invariance receipt. The 2,048-update reference ran
detached on Spark and stopped at optimizer step 1,536: the frozen development
entry gate **failed**, which is terminal for this candidate. TEST was not
opened. This is not an RCQ competency result.

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

Second checkpoint at optimizer step 512 (~42 minutes after launch):

| Item | Value |
|---|---|
| Checkpoint | `step-00000512.pt` (338 MiB) |
| Checkpoint SHA-256 | `00001fcab2de0791668cff60fd5538463b5bca2e11264e42e83ba7a01b5274d1` |
| Train loss | 0.218 |
| Train movement exact match | 0.688 |
| Validation loss | 0.203 |
| Validation movement exact match | 0.721 |
| Validation previous-control exact | 0.696 |
| Validation positive-key recall | 0.860 |

Third checkpoint at optimizer step 768 (~62 minutes after launch):

| Item | Value |
|---|---|
| Checkpoint | `step-00000768.pt` (338 MiB) |
| Checkpoint SHA-256 | `3543bf9521205cd14ee34b78106a4b62ae33fc3fe63c13609432a2e78f99b348` |
| Train loss | 0.169 |
| Train movement exact match | 0.833 |
| Validation loss | 0.124 |
| Validation movement exact match | 0.842 |
| Validation previous-control exact | 0.696 |
| Validation positive-key recall | 0.928 |
| Validation continuous-outside-deadzone (logged mean) | 3.22 |

Fourth checkpoint at optimizer step 1,024 (~82 minutes after launch):

| Item | Value |
|---|---|
| Checkpoint | `step-00001024.pt` (338 MiB) |
| Checkpoint SHA-256 | `d5436e2681942f583c45be5c96af96285bb0d1dfca844369ac9f8af5d0b21fcd` |
| Train loss | 0.207 |
| Train movement exact match | 0.708 |
| Validation loss | 0.139 |
| Validation movement exact match | 0.842 |
| Validation previous-control exact | 0.696 |
| Validation positive-key recall | 0.988 |
| Validation continuous-outside-deadzone (logged mean) | 3.71 |

Fifth checkpoint at optimizer step 1,280 (~103 minutes after launch):

| Item | Value |
|---|---|
| Checkpoint | `step-00001280.pt` (338 MiB) |
| Checkpoint SHA-256 | `d36a6def9d0da23afe7d170856ed902e8a8a90c23d3a986eb9cfa14ccd682d60` |
| Train loss | 0.131 |
| Train movement exact match | 0.854 |
| Validation loss | 0.100 |
| Validation movement exact match | 0.900 |
| Validation previous-control exact | 0.696 |
| Validation positive-key recall | 0.954 |
| Validation continuous-outside-deadzone (logged mean) | 7.08 |

The validation slice is the open development namespace, not sealed TEST.
These logger lines are not the development gate. Exact 0.90 and recall 0.95
are above the 0.80 / 0.90 floors on that eval dump. False positives and
opposite-key conflicts dropped versus step 1,024. Continuous outputs are
still leaving the `[-0.05, 0.05]` deadzone, and that logged mean rose to
7.08, so the official step-1,536 report can still fail quiescence even if
WASD exact stays high.

Host disk is about 549 GiB free after the non-3.8 weight cleanup. Do not chmod
installed releases writable. Do not open TEST.

## Terminal outcome: entry gate failed at step 1,536 (2026-08-17 UTC)

At optimizer step 1,536 the trainer wrote `step-00001536.pt`, evaluated the
frozen `rcq_v2_development_v1` entry gate on the full 256-sequence development
slice (1,536 scored decisions), and exited with
`stop_reason: development_gate_failed`. Stage 2 (value-head-only) never
started. This candidate is terminal: no resume, no tuning, no retry, and no
parallel run under this registration. Final TEST stays sealed with
`sealed_test_examples_opened: 0`.

Terminal identities:

| Item | Value |
|---|---|
| Gate report | `development-gate-step-00001536.json` |
| Gate report SHA-256 | `6e1bcdec69949e1b1e43f271ebf300f07a39e25318790aa82de99c5b641e7724` |
| Gate | `rcq_v2_development_v1`, schema 1, `passed: false` |
| Terminal checkpoint | `step-00001536.pt` (stage-0 terminal, 338 MiB) |
| Checkpoint SHA-256 | `56cbf6d34325582ca4d68a3b6620193effe37825cb59f67f057f9c656c770fb2` |
| `checkpoints/latest.json` SHA-256 | `c2e7b046bc0a4e6aa22180b0f5c1101c5db9fdee7931e24db8dbeb55aed12cf7` (points at `step-00001536.pt`, schema 1) |
| `metrics.jsonl` SHA-256 | `10d7faa18912fd89bef2ecab1622245247bd550acf863997755be6d83a96302e` (31 rows) |

Canonical gate checks (1,536 scored development decisions):

| Check | Observed | Requirement | Result |
|---|---|---|---|
| movement exact | 1,422 | >= 1,229/1,536 (0.80) | pass |
| changed movement exact | 402 | >= 234/467 (0.50) | pass |
| sample-macro positive-key recall | 0.97624 | >= 0.90 | pass |
| movement false-positive keys | 64 | <= 230 | pass |
| opposite-direction conflicts | 9 | <= 7 | **fail** |
| non-movement keyboard false positives | 0 | = 0 | pass |
| predicted-active buttons outside W/A/S/D (296 channels) | 0 | = 0 | pass |
| off-support target buttons | 0 | = 0 | pass |
| nonzero continuous targets | 0 | = 0 | pass |
| continuous outputs outside the [-0.05, 0.05] deadzone | 875 | = 0 of 16,896 | **fail** |

Final logged train row (step 1,536): loss 0.0887, action loss 0.0664, value
loss 0.2141, movement exact 0.875. Final logged development row: loss 0.0742,
action loss 0.0440, value loss 0.2928, movement exact 0.9258. The action
metrics stayed above their floors at the gate; the candidate failed on nine
opposite-direction W/A/S/D conflicts and 875 continuous outputs outside the
quiescence deadzone, both hard requirements of the frozen gate. The entry
report's `value_loss` 0.2928 is recorded context only; it has no pass/fail
role in stage 1, and the stage-2 value-completion gate never ran.

Consequences, per the frozen protocol: the canonical failed entry report makes
the step-1,536 checkpoint terminal, so `Resume-DgxRcqV2Reference.ps1` refuses
it; do not tune or warm-start from this candidate; do not preclaim; do not
open sealed TEST ranges. Seed 1702 did not qualify as the RCQ-v2 reference
candidate. Any next attempt is a newly preregistered experiment with its own
registration, release, and pin.
