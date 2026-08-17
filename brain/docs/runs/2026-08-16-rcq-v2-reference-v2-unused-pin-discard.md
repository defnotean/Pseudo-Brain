# Unused RCQ-v2 reference-v2 pretraining pin discard (2026-08-16)

Status: operational discard. This is not a model result, not a development-gate
failure, and not permission to open TEST.

The live v2 registration, immutable release, and create-only pretraining pin
completed. Pin-bound RCQ smoke then failed in an isolated test that reads
historical `registrations/rcq-v2-reference-v1.json`. That file is git-only.
Release sync allowlists only the live registration, so the immutable Spark
tree does not contain v1. CUDA backward had already passed. No smoke receipt,
no staging canary, no reference run, and no TEST construction exist for this
pin.

Keep live registration SHA-256
`6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`.
`source_tree_sha256`, the evaluator bundle, and the RCQ training TOML did not
change. Do not rebuild the registration. Do not edit historical v1 JSON.

## Why the pin cannot be reused

`qualification-pins/rcq-v2-reference-v2/` is create-only. Each pin binds one
`RELEASE_ID`. Smoke runs tests from that frozen release, so a test fix is
invisible until a new release is synced. A new release needs a new pin. Unused
pin files are discarded rather than overwritten. Installed unused releases and
their smoke logs stay on the host. Do not chmod those releases writable.

## Unused pin — historical v1 file absent from the live release

Release `r20260817t021020z-6f2359089eeb`. Smoke log
`/home/defnotean/projects/pseudo-brain/logs/smoke-r20260817t021020z-6f2359089eeb-20260817T021135Z.log`.

`test_historical_v1_registration_bytes_are_preserved` resolved
`/workspace/repo/registrations/rcq-v2-reference-v1.json` and raised
`FileNotFoundError`. Local play-safe passed because the Git tree still holds
v1. The test now skips when that historical file is absent and still pins the
v1 bytes when the Git checkout is present.

| Item | Value |
|---|---|
| Unused archive SHA-256 | `6f2359089eebd2810c97acdb2b0d18b724038efe21fec161da5efbae1435c802` |
| Unused pin file SHA-256 | `2847e78625714bde22b4f8eb4c364b6480a2eff1222e5dde9cacb6d854b7210f` |
| Unused pin semantic SHA-256 | `8c1acce108ad1c654a438c3f0c69037d4d219138d83a04e715aba26538bcb7af` |
| Pin created UTC | `2026-08-17T02:11:11Z` |
| Cached image ID | `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b` |

## What must happen next

Discard this unused pin file so the v2 pin directory is empty. Sync a new
immutable release that includes the skip, create a new pin bound to the same
live registration SHA, then rerun smoke. Do not start the 2,048-update
reference until that smoke and the staging canary both pass.

Do not open sealed TEST ranges. Do not rerun final-once. Do not point generic
train/smoke wrappers at the RCQ reference config.
