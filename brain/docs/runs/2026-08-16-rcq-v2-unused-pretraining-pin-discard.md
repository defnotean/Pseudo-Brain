# Unused RCQ-v2 pretraining pin discards (2026-08-16)

Status: operational discards. These are not model results, not development-gate
failures, and not permission to open TEST.

Two create-only Spark pins were published and then discarded because pin-bound
smoke died in isolated tests that only fail on the immutable Linux release.
CUDA backward passed both times. No smoke receipt, no staging canary, no
reference run, and no TEST construction exist for either pin.

Keep registration SHA-256
`33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0`.
`source_tree_sha256`, the evaluator bundle, and the RCQ training TOML did not
change. Do not rebuild the registration.

## Why the pins cannot be reused

`qualification-pins/rcq-v2-reference-v1/` is create-only. Each pin binds one
`RELEASE_ID`. Smoke runs tests from that frozen release, so a test fix is
invisible until a new release is synced. A new release needs a new pin. Unused
pin files are discarded rather than overwritten. Installed unused releases and
their smoke logs stay on the host. Do not chmod those releases writable.

## Unused pin 1 — read-only evaluator copy

Release `r20260817t012309z-c07f1a23ca35`. Smoke log
`/home/defnotean/projects/pseudo-brain/logs/smoke-r20260817t012309z-c07f1a23ca35-20260817T012705Z.log`.

`test_resume_preflight_refuses_terminal_failed_stage_gates` copied `irene_brain`
with `shutil.copytree`, which preserved immutable-release mode `444`, then
called `Path.write_text` on `evaluation/rcq_v2.py`. Local Windows checkouts are
writable, so play-safe did not see this.

The test now `chmod`s copied files `0600` and directories `0700` before
mutating the evaluator.

| Item | Value |
|---|---|
| Unused archive SHA-256 | `c07f1a23ca35224652fa02ec10f1b9b1ea5b5e5ca816907a3e750d703a6f37e6` |
| Unused pin file SHA-256 | `cee1cb7676e2eae00c58236b8eda2e00d3495bb20b212c8ff62ca4c90ec1333b` |
| Unused pin semantic SHA-256 | `bd207cb38bae32139a43d31dd20106e4c01c927ca8fbb0d9f733eccf352072c7` |
| Pin created UTC | `2026-08-17T01:26:48Z` |

## Unused pin 2 — OS-dependent absolute checkpoint path

Release `r20260817t013316z-4dcad4af91bb`. Smoke log
`/home/defnotean/projects/pseudo-brain/logs/smoke-r20260817t013316z-4dcad4af91bb-20260817T013428Z.log`.

The resume-preflight copy now passes. Smoke then failed in
`test_binder_rejects_noninteger_schema_and_unsafe_checkpoint_paths`. The case
used `str((checkpoint_root / "absolute.pt").resolve())`. On Windows that string
contains backslashes and raises `POSIX relative`. On Linux it is a POSIX
absolute path and raises `checkpoint_path must stay beneath checkpoint_root`.

The test now uses fixed strings `/tmp/absolute.pt` and `C:\outside.pt` so both
rejection branches are OS-independent.

| Item | Value |
|---|---|
| Unused archive SHA-256 | `4dcad4af91bb904393f6c83ffdeca5e0149d43d4ec67e2b0011736df718dc253` |
| Unused pin file SHA-256 | `04608d704329422f7eef45f846ed45c97cceb344c87d51b532bab5a0b4ee5f96` |
| Unused pin semantic SHA-256 | `8da8ab64adae14c1fde1d6b30759b7169390b4268586efca2e0ef299d5d287da` |
| Pin created UTC | `2026-08-17T01:33:52Z` |
| Cached image ID (both pins) | `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b` |

## What must happen next

1. Empty `qualification-pins/rcq-v2-reference-v1/` (remove `pretraining.json`
   after `chmod u+w`; leave the parent lock file).
2. Sync a **new** immutable release that includes both test fixes and the same
   registration file.
3. Require remote read-back of registration SHA-256 `33f7900c…`.
4. Publish a **new** pretraining pin bound to the new `RELEASE_ID`.
5. Re-run pin-bound RCQ smoke, then the staging canary.
6. Start the 2,048-update reference only if both pass.

Do not open sealed TEST ranges. Do not rerun final-once. Do not point generic
train/smoke wrappers at the RCQ reference config.
