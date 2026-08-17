# Operator guide: freeze, register, then DGX

This is the how-to for the current RCQ-v2 campaign. Exact thresholds, hexadecimal
value identities, and legal stopping rules are in
[RCQ_V2_PROTOCOL.md](RCQ_V2_PROTOCOL.md). Host and container mechanics are in
[DGX_SPARK_TRAINING.md](DGX_SPARK_TRAINING.md). The campaign checklist is
[CURRENT_WORK.md](../../CURRENT_WORK.md).

Run every command from the Git top-level of this repository unless a snippet
says otherwise.

## What you are operating

There are two different launch families. Mixing them is a protocol failure.

| Family | Scripts | Use for RCQ-v2? |
|---|---|---|
| Generic | `Invoke-DgxBrainSmoke.ps1`, `Start-DgxBrainTraining.ps1`, `Resume-DgxBrainTraining.ps1` | **No.** They reject the frozen RCQ reference config and run ID. |
| Pinned RCQ | `New-DgxRcqV2PretrainingPin.ps1`, `Invoke-DgxRcqV2Smoke.ps1`, `Invoke-DgxRcqStagingCanary.ps1`, `Start-DgxRcqV2Reference.ps1`, `Resume-DgxRcqV2Reference.ps1`, preclaim / final-auth / final-once / verify | **Yes**, after the local freeze and registration. |

Generic preflight and sync (`Invoke-DgxPreflight.ps1`, `Sync-DgxBrainRelease.ps1`)
are still required. They install the immutable release that the pinned actions
later consume.

## Step 1. Freeze the local tree

Do this on the Windows workstation. Do not use the local GPU.

1. Confirm you are in the canonical Git root and that
   `.pseudo-brain-workspace-v2` is present.
2. Confirm `brain/src` contains only the `irene_brain` package, and that the
   package contains only `.py` files (no `__pycache__`, no links, no extra
   files).
3. If implementation files under the matched-baseline set changed, regenerate
   the architecture manifest:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\Update-BaselineArchitectureManifest.ps1
   ```

   Record the printed `MANIFEST_SHA256` and the per-file source hashes in the
   freeze run record. A new digest is a new comparison identity.
4. Run the play-safe suite:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\run_play_safe_tests.ps1
   ```

   Stop if any module fails. Do not register a tree that cannot pass local
   identity tests.
5. Commit and push the frozen source, configuration, documentation, and
   manifest **before** creating the registration. After the registration SHA
   exists, do not edit `brain/src`, the RCQ training TOML, or the registration
   file. If a later smoke or canary forces a **source** change, discard that
   registration/release pair and start a new freeze. A launcher-test-only fix
   that leaves `source_tree_sha256` unchanged still needs a new immutable
   release and a new pretraining pin; empty the unused create-only pin
   directory first, and do not rebuild the registration.

## Step 2. Create the target-blind registration

The builder hashes source and configuration. It derives dataset *geometry*
manifests only. It must report `sealed_test_examples_opened: 0`. It must not
instantiate a sequence dataset.

Windows PowerShell strips double quotes from ``python -c`` payloads, so the
wrapper does not use an inline bootstrap. It runs isolated ``python -I -B``
against `brain/scripts/run_rcq_v2_registration.py`, which rewrites ``sys.path``
and ``sys.argv`` and then executes the target-blind builder.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\dgx\New-RcqV2Registration.ps1
```

Expected outputs:

- File: `registrations/rcq-v2-reference-v2.json`
- Printed line: `REGISTRATION_SHA256=<64 lowercase hex>`
- Printed line: `EXTERNAL_PIN_REQUIRED=true`

The live 2026-08-16 qualification already published that file. Its digest is
`6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`. Copy that
value into a note that is not this repository and not `brain/runs/`. See
[runs/2026-08-16-rcq-v2-reference-v2-registration.md](./runs/2026-08-16-rcq-v2-reference-v2-registration.md).
Historical `registrations/rcq-v2-reference-v1.json` stays in git and must not
be edited. Replacement of the live file is forbidden: if it already exists,
delete nothing; start a newly named qualification instead.

`PlanOnly` prints the intended action without writing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\dgx\New-RcqV2Registration.ps1 -PlanOnly
```

## Step 3. Remote preflight, then sync

These are short foreground SSH checks. They never pull Docker images and never
use the workstation GPU. The wrapper invokes remote `/bin/bash -p -s` because
Spark GNU bash 5.2 rejects combining `-p` with `--noprofile`/`--norc`.

```powershell
& .\brain\scripts\dgx\Invoke-DgxPreflight.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -ContainerImage 'vllm/vllm-openai:nightly-aarch64' `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 16
```

If another compute process is using the Spark, or disk/memory is too low, stop.
Do not kill competing jobs from these scripts.

Then install one immutable release. Sync allowlists `brain/**` plus the exact
registration path:

```powershell
& .\brain\scripts\dgx\Sync-DgxBrainRelease.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -MinFreeDiskGiB 20
```

Write down `RELEASE_ID`, the source-archive SHA-256, and the cached image
content digest. Require remote read-back of the externally pinned registration
SHA at `registrations/rcq-v2-reference-v2.json` inside that release.

## Step 4. Trusted pretraining pin

This ceremony binds release, archive, registration, and image identities. Keep
the file and the printed hashes outside the run directory.

```powershell
& .\brain\scripts\dgx\New-DgxRcqV2PretrainingPin.ps1 `
  -SshTarget defnotean `
  -ReleaseId '<release-id>' `
  -ReleaseArchiveSha256 '<archive-sha256>' `
  -RegistrationSha256 '<registration-sha256>' `
  -ContainerImage '<cached-image-reference>' `
  -ContainerImageId 'sha256:<64-lowercase-hex>'
```

After this pin exists, production RCQ actions ignore caller-selected workspace,
release, image, config, run, and checkpoint identities. They resolve only the
canonical remote workspace `~/projects/pseudo-brain`.

## Step 5. RCQ smoke, then staging canary

```powershell
& .\brain\scripts\dgx\Invoke-DgxRcqV2Smoke.ps1 -SshTarget defnotean
```

Smoke must pass CUDA backward, isolated CPU tests, and the bounded trainer
smoke. Then:

```powershell
& .\brain\scripts\dgx\Invoke-DgxRcqStagingCanary.ps1 `
  -SshTarget defnotean `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 16 `
  -ContainerCpuCount 2 `
  -ContainerMemoryGiB 8
```

The canary is an operational check, not a scientific gate. It runs two CUDA
processes: update 1 with a stage transition, then exact-resume of checkpoint
schema 2 through update 2. Require the invariance receipt. Do not treat a
canary pass as RCQ competency.

## Step 6. Pinned reference training

Start only when the immutable release, smoke receipt, canary receipt,
registration, pretraining pin, and external registration SHA all agree.

```powershell
& .\brain\scripts\dgx\Start-DgxRcqV2Reference.ps1 `
  -SshTarget defnotean `
  -AcknowledgeDetached
```

This is seed `1702`, config `dgx-rcq-v2-reference.toml`, 12 CPUs, 96 GiB. There
are no resource or identity overrides.

- At optimizer step 1,536 the trainer evaluates the development entry gate.
  Failure is terminal for this candidate. Do not tune and retry under RCQ-v2.
- On pass it freezes non-value state and trains only the value head through
  step 2,048.
- Resume only with `Resume-DgxRcqV2Reference.ps1 -AcknowledgeDetached`.

## Step 7. Preclaim, review, one-shot TEST

Preclaim revalidates everything that is not TEST and must report
`sealed_test_examples_opened: 0`:

```powershell
& .\brain\scripts\dgx\Invoke-DgxRcqV2Preclaim.ps1 -SshTarget defnotean
```

After independent review of the readiness SHA and checkpoint SHAs, run the
final-authorization ceremony, then the one TEST-consuming action **once**:

```powershell
& .\brain\scripts\dgx\New-DgxRcqV2FinalAuthorization.ps1 `
  -SshTarget defnotean `
  -PretrainingPinSha256 '<pretraining-pin-file-sha256>' `
  -LatestPointerSha256 '<latest.json-file-sha256>' `
  -EntryCheckpointSha256 '<step-00001536-sha256>' `
  -CheckpointSha256 '<step-00002048-sha256>' `
  -ReadinessSha256 '<readiness-file-sha256>'

& .\brain\scripts\dgx\Invoke-DgxRcqV2FinalOnce.ps1 `
  -SshTarget defnotean `
  -AcknowledgePermanentTestRetirement
```

Do not rerun final-once after a claim exists. Verify the receipt without
opening TEST:

```powershell
& .\brain\scripts\dgx\Test-DgxRcqV2FinalReceipt.ps1 -SshTarget defnotean
```

A failure after claim is still the RCQ-v2 result. The sealed ranges stay
retired. Later architecture work must use the separately reserved 2^21
namespace.

## Common failure modes, in plain language

- **Stale architecture manifest.** Local tests stop because
  `brain/configs/baseline-architecture-manifest.json` no longer matches the
  implementation files. Regenerate only as an intentional freeze.
- **Registration already exists.** The file is create-once. Do not overwrite it
  to "fix" a hash. Freeze a new source tree and a new qualification id.
- **Generic train/resume/smoke pointed at RCQ files.** The wrappers refuse.
  Use the pinned scripts.
- **Opening TEST to inspect labels.** That retires the range as evidence. The
  old `[1048576, 1049088)` band is already dead for that reason.
- **Continuing after a failed frozen gate.** Diagnose or preregister a new
  experiment. Do not treat a failed candidate as a warm start.
- **Smoke dies on a launcher test that only fails in the Linux release.**
  Isolated tests run from a `chmod a-w` tree, and Windows play-safe can miss
  POSIX path or mode behavior. If smoke never receipted, discard the unused
  pin and sync a new release. Do not chmod the installed release. The
  2026-08-16 unused pins are that case:
  [runs/2026-08-16-rcq-v2-unused-pretraining-pin-discard.md](./runs/2026-08-16-rcq-v2-unused-pretraining-pin-discard.md)
  and
  [runs/2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md](./runs/2026-08-16-rcq-v2-reference-v2-unused-pin-discard.md).
- **Staging canary fails frozen-stage invariance.** After update 1 the value-head
  transition requires bit-identical action outputs. A mismatch stops the
  campaign. Do not start the 2,048-update reference. The 2026-08-16 canary is
  that case:
  [runs/2026-08-16-rcq-v2-smoke-pass-canary-invariance-fail.md](./runs/2026-08-16-rcq-v2-smoke-pass-canary-invariance-fail.md).
  Live v2 canary passed after the capture fix:
  [runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md](./runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md).
  Diagnosis: CUDA kernel selection follows the `requires_grad` freeze mask.
  [runs/2026-08-16-rcq-v2-canary-invariance-diagnosis.md](./runs/2026-08-16-rcq-v2-canary-invariance-diagnosis.md).
  Bounded probes: `brain/scripts/diagnose_rcq_stage_invariance.py`.

## What to copy down at each gate

Keep these outside the repository and outside `brain/runs/`:

- architecture `manifest_sha256` (matched-baseline freeze)
- `REGISTRATION_SHA256`
- `RELEASE_ID`, release archive SHA-256, container image id
- pretraining pin file SHA and semantic SHA
- smoke and canary receipt hashes
- step-1536 and step-2048 checkpoint SHAs, `latest.json` SHA
- preclaim `readiness_sha256`
- final authorization hashes and the terminal receipt SHA
