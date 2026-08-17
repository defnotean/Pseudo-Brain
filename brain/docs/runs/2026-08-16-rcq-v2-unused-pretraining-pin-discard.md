# Unused RCQ-v2 pretraining pin discard (2026-08-16)

Status: operational discard. This is not a model result, not a development-gate
failure, and not permission to open TEST.

The first Spark preflight, immutable release, and create-only pretraining pin
completed. Pin-bound RCQ smoke then failed inside an isolated launcher test
that tried to edit a copy of the immutable release. CUDA backward had already
passed. No smoke receipt, no staging canary, no reference run, and no TEST
construction exist for that pin.

## Why the pin cannot be reused

`qualification-pins/rcq-v2-reference-v1/` is create-only. The pin binds one
`RELEASE_ID`. Smoke runs tests from that frozen release, so a test fix is
invisible until a new release is synced. A new release needs a new pin. The
unused pin file is therefore discarded rather than overwritten.

This discard does **not** rebuild the target-blind registration. The failure
was in `brain/tests/test_dgx_launch_contract.py`. `source_tree_sha256`, the
evaluator bundle, the RCQ training TOML, and
`registrations/rcq-v2-reference-v1.json` are unchanged. Keep
`33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0`.

## Discarded identities

| Item | Value |
|---|---|
| Registration SHA-256 (kept) | `33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0` |
| `source_tree_sha256` (kept) | `03a579c88965aea6fa31bcad2084ab69e98d824ac6c8ebfd5e2263f1cd4c0260` |
| Unused release | `r20260817t012309z-c07f1a23ca35` |
| Unused archive SHA-256 | `c07f1a23ca35224652fa02ec10f1b9b1ea5b5e5ca816907a3e750d703a6f37e6` |
| Cached image ID | `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b` |
| Unused pin file SHA-256 | `cee1cb7676e2eae00c58236b8eda2e00d3495bb20b212c8ff62ca4c90ec1333b` |
| Unused pin semantic SHA-256 | `bd207cb38bae32139a43d31dd20106e4c01c927ca8fbb0d9f733eccf352072c7` |
| Pin created UTC | `2026-08-17T01:26:48Z` |
| Smoke log | `/home/defnotean/projects/pseudo-brain/logs/smoke-r20260817t012309z-c07f1a23ca35-20260817T012705Z.log` |

The installed unused release and its smoke log stay on the host as historical
artifacts. Do not chmod that release writable. Do not treat it as the current
campaign pin.

## Test bug

`test_resume_preflight_refuses_terminal_failed_stage_gates` copied
`irene_brain` into `/tmp` with `shutil.copytree`, which preserved the
immutable-release mode `444`. It then called `Path.write_text` on
`evaluation/rcq_v2.py`. Local Windows checkouts are writable, so play-safe
did not see this. The Spark smoke did.

The test now copies that tree and then `chmod`s files `0600` and directories
`0700` before mutating the evaluator. That is a launcher-test fix only.

## What must happen next

1. Empty `qualification-pins/rcq-v2-reference-v1/` (remove `pretraining.json`
   after `chmod u+w`; leave the parent lock file).
2. Sync a **new** immutable release that includes the test fix and the same
   registration file.
3. Require remote read-back of registration SHA-256 `33f7900c…`.
4. Publish a **new** pretraining pin bound to the new `RELEASE_ID`.
5. Re-run pin-bound RCQ smoke, then the staging canary.
6. Start the 2,048-update reference only if both pass.

Do not open sealed TEST ranges. Do not rerun final-once. Do not point generic
train/smoke wrappers at the RCQ reference config.
