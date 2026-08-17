# Pseudo-Brain DGX Spark launch scripts

These PowerShell entry points operate only through an explicit remote OpenSSH
target. Generic train/resume/smoke wrappers reject the frozen RCQ-v2 reference
config and run ID; production RCQ actions derive every identity from the
create-only pins.

Generic workspace-taking actions:

- `Invoke-DgxPreflight.ps1`: fail-closed ARM64/GPU/Python/PyTorch/disk/unified-memory check.
- `Sync-DgxBrainRelease.ps1`: source-only Git-aware sync to an immutable release.
- `Invoke-DgxBrainSmoke.ps1`: bounded foreground CUDA/backward/tests/trainer smoke. Rejects the RCQ reference config.
- `Start-DgxBrainTraining.ps1`: generic trainer entry point. Rejects the RCQ reference config/run and the schema-3 staging-canary config.
- `Resume-DgxBrainTraining.ps1`: generic exact-checkpoint resume. Same RCQ/canary rejection as start.
- `Get-DgxBrainTrainingStatus.ps1`: single read-only status and log-tail snapshot.
- `Receive-DgxBrainArtifact.ps1`: non-overwriting log, checkpoint, or resume-attempt retrieval.

Pinned RCQ-v2 production actions. After the pretraining pin exists, these accept
no caller-selected workspace, release, image, config, run, or checkpoint:

- `New-DgxRcqV2PretrainingPin.ps1`: trusted create-only pretraining pin ceremony.
- `Invoke-DgxRcqV2Smoke.ps1`: pin-bound foreground smoke of the pinned release/image.
- `Invoke-DgxRcqStagingCanary.ps1`: pin-bound two-update schema-3 CUDA canary.
- `Start-DgxRcqV2Reference.ps1`: pin-bound detached start of `dgx-rcq-v2-reference-seed-1702`.
- `Resume-DgxRcqV2Reference.ps1`: pin-bound detached resume of that same run.
- `Invoke-DgxRcqV2Preclaim.ps1`: non-TEST readiness publication. No identity arguments.
- `New-DgxRcqV2FinalAuthorization.ps1`: trusted create-only final-authorization ceremony.
- `Invoke-DgxRcqV2FinalOnce.ps1`: one-shot TEST claim/evaluation. No identity arguments.
- `Test-DgxRcqV2FinalReceipt.ps1`: read-only receipt verification. No identity arguments.
- `New-RcqV2Registration.ps1`: local target-blind registration builder before release sync.

The plain-language campaign checklist is
[CURRENT_WORK.md](../../../CURRENT_WORK.md). The ordered operator commands are
[OPERATOR_GUIDE.md](../../docs/OPERATOR_GUIDE.md). Local matched-baseline
manifest regeneration uses `../Update-BaselineArchitectureManifest.ps1` and is
not a DGX action.

The current workspace contract is `pseudo-brain-workspace-v2`. The canonical
production workspace is `~/projects/pseudo-brain`. Generic preflight, sync,
smoke, start, and resume may target a direct child of the remote `projects`
directory named `pseudo-brain` (or an explicit `pseudo-brain-<suffix>`). Those
write-capable actions reject the historical `irene-brain` root and any path
nested beneath it. Archive, container, and tmux names also use `pseudo-brain`.
Pinned RCQ actions ignore caller workspace arguments and resolve only that
canonical account path.

Historical v1 runs remain readable in place. Only the status and artifact
wrappers accept `~/projects/irene-brain`, only when the caller supplies
`-HistoricalIreneWorkspace`; their remote actions validate the old v1 marker
and contain no write or migration operation. Local artifact destinations are
anchored inside the Pseudo-Brain repository.

Remote dispatch runs under `/bin/bash -p` with `PATH=/usr/sbin:/usr/bin`,
`env -i`, and `--noprofile --norc`. Production Python uses `python3 -I`.
No script pulls images, reads secret `.env` files, accepts arbitrary remote
commands, stops services, launches systemd units, or invokes a local
accelerator. See [DGX_SPARK_TRAINING.md](../../docs/DGX_SPARK_TRAINING.md) and
[RCQ_V2_PROTOCOL.md](../../docs/RCQ_V2_PROTOCOL.md) for the required order.
Runs created before the resumable lock and launch-digest contract are
intentionally not resumable by these scripts.
