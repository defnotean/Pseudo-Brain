# DGX Spark training boundary

The scientific qualification sequence, frozen gates, sealed TEST chronology,
and stopping rules are specified in
[RCQ_V2_PROTOCOL.md](RCQ_V2_PROTOCOL.md). This document covers the DGX host and
container operations that implement that protocol.

Pseudo-Brain uses the DGX Spark only for offline training. The Spark is never in
the live capture-to-action loop, and these scripts never use the workstation
GPU.

Every command requires an explicit SSH target and a dedicated remote project
directory. New releases and runs use the existing OpenSSH alias `defnotean` and
the Pseudo-Brain v2 workspace `~/projects/pseudo-brain`. The historical
`~/projects/irene-brain` v1 workspace is read-only: only status and artifact
retrieval accept it, and only with `-HistoricalIreneWorkspace`. Sync, smoke,
start, and resume reject it before contacting the remote helper. The scripts
reject localhost, an SSH alias resolving
to this workstation, broad remote directories, unknown host keys, interactive
password prompts, active GPU compute processes, inadequate unified memory, and
inadequate disk space.

## Verified Spark snapshot

The following was observed on 2026-08-16 and is only a snapshot; the preflight
must be rerun before every new release or job:

- The alias `defnotean` resolves to the ARM64 DGX Spark account.
- The accelerator is NVIDIA GB10 with driver 580.173.02.
- System Python is 3.12 and does not contain PyTorch.
- The already-cached `vllm/vllm-openai:nightly-aarch64` image contains Python
  and PyTorch 2.13.0+cu130.
- The filesystem had only about 33.4 GiB free. Do not pull another framework
  image until storage is deliberately reclaimed.
- `irene-qwen38-heretic` previously occupied most unified memory. It was
  explicitly stopped for this training window; after that, the Spark reported
  no compute processes and about 119 GiB `MemAvailable`.

The existing vLLM image is acceptable for the first bounded smoke because it is
already present. It is not a reproducible long-term training base because its
tag is mutable. The smoke receipt records the local Docker image ID, and a
later training launch refuses to proceed if that ID changes. Once enough disk
is available, move to a pinned NVIDIA PyTorch NGC image that has been validated
on Spark.

The remote scripts never stop, remove, restart, or reconfigure an existing
container. When this training window is over, the previously stopped Irene
service can be restored manually on the Spark with:

```bash
docker start irene-qwen38-heretic
```

Do that only after the training container has exited.

## One-time SSH trust

The launchers use `BatchMode=yes` and `StrictHostKeyChecking=yes`. They do not
accept a new key automatically. If this Windows account has never connected to
the Spark, first verify its host-key fingerprint through a trusted local source
and make one deliberate interactive connection:

```powershell
ssh defnotean
```

Exit without starting anything. Do not put passwords, private keys, API tokens,
or host fingerprints in this repository.

If a restricted automation account cannot read your normal OpenSSH config, the
same commands may use the explicit shell-safe target
`defnotean@gx10-db18.local`. This contains no credential; authentication still
comes from OpenSSH. Prefer the alias in an ordinary user terminal.

## Safe launch sequence

Run commands from the repository root. All threshold and resource values are
explicit so a copied command cannot silently inherit a larger workload.

### 1. Remote preflight

This is a short foreground check. It verifies ARM64, disk, `MemAvailable`, an
idle compute queue, `nvidia-smi`, system Python, Docker, the presence of the
selected cached image, and PyTorch/CUDA from inside that image. It never pulls
an image.

```powershell
& .\brain\scripts\dgx\Invoke-DgxPreflight.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -ContainerImage 'vllm/vllm-openai:nightly-aarch64' `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 16
```

If another compute process appears or memory falls below the threshold, the
preflight aborts. It does not stop the competing process.

### 2. Sync an immutable source release

The sync tool asks Git for tracked plus non-ignored untracked files under
`brain/`, builds one temporary source archive, and rejects secrets, caches,
datasets, logs, runs, checkpoints, weights, and symbolic links. It installs to
a new read-only `releases/<release-id>` directory. It does not overwrite a
release or any existing service directory.

```powershell
& .\brain\scripts\dgx\Sync-DgxBrainRelease.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -MinFreeDiskGiB 20
```

Copy the printed `RELEASE_ID`; every later step requires it explicitly.

### 3. Bounded foreground smoke

Use the tiny CUDA smoke configuration produced by the training package. The
smoke runs three gates in a container with networking disabled and source
mounted read-only:

1. A 64 x 64 BF16 CUDA matrix multiply and backward pass.
2. Every repository unit-test module in an isolated Python process, so the
   dependency-free import test cannot be contaminated by a different
   torch-enabled test module.
3. `python -m irene_brain.training.train --config <toml>` with a hard five
   minute outer timeout and `PSEUDO_BRAIN_MAX_STEPS=3`.

```powershell
& .\brain\scripts\dgx\Invoke-DgxBrainSmoke.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -ReleaseId '<release-id>' `
  -ContainerImage 'vllm/vllm-openai:nightly-aarch64' `
  -ConfigRelativePath 'brain/configs/training/dgx-smoke.toml' `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 16 `
  -ContainerCpuCount 2 `
  -ContainerMemoryGiB 8
```

A successful smoke creates a receipt tied to both the release and the cached
image ID. Training cannot bypass this receipt. The supplied image tag remains
human-readable metadata, but Docker is always launched by the inspected
immutable `sha256:...` image ID so a concurrent retag cannot change the runtime.

### 4. Schema-3 transition/resume canary

Before the 2,048-update RCQ reference run, execute the dedicated two-update
schema-3 canary. This is an operational check, not a scientific gate: it uses
the smoke model, tiny offset-zero data, and declares no RCQ development or TEST
gate. It never materializes the sealed RCQ TEST ranges.

```powershell
& .\brain\scripts\dgx\Invoke-DgxRcqStagingCanary.ps1 `
  -SshTarget defnotean `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 16 `
  -ContainerCpuCount 2 `
  -ContainerMemoryGiB 8
```

Release, image, and run identity come from the create-only pretraining pin.
Do not pass a workspace, release ID, image, config, or run ID; the generic
train/resume wrappers refuse this canary config.

The launcher fixes
`brain/configs/training/dgx-rcq-v2-staging-canary.toml`, has no tmux mode, and
has a ten-minute outer timeout. Its first foreground CUDA process performs
update 1, transitions from the all-parameter stage to the exact two-parameter
value-head stage, resets the optimizer, snapshots the frozen action/state
invariants, writes checkpoint schema 2, and exits at the registered operational
stop. The second process exact-resumes that checkpoint and its bound metrics
prefix, performs update 2 with only the value head trainable, and rechecks the
invariants.

Success requires both checkpoints, the exact four train/validation metric
records, the transition record, and step-1/step-2 invariance records. A receipt
binds their hashes to the immutable release, cached image ID, pretraining pin,
and canary config. The general training launcher refuses this canary config, so
it cannot silently skip the resume half. Conversely, launching the RCQ-v2
reference requires `Start-DgxRcqV2Reference.ps1` after the exact release's
canary receipt; generic start/resume wrappers refuse that config and run ID.
Do not start that reference run until its source, registration, pretraining
pin, and evaluation contract are frozen. The complete pinned production
sequence, including preclaim, final authorization, final-once, and
verify-receipt wrappers, is in [RCQ_V2_PROTOCOL.md](RCQ_V2_PROTOCOL.md).

### 5. Full-model canary, then first training run

Keep the first full-thesis optimization canary foreground and bounded to one
optimizer step. This validates full-model memory, BF16 forward/backward,
metrics, and checkpoint writing without starting the 1,000-step bootstrap:

```powershell
& .\brain\scripts\dgx\Start-DgxBrainTraining.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -ReleaseId '<release-id>' `
  -ContainerImage 'vllm/vllm-openai:nightly-aarch64' `
  -ConfigRelativePath 'brain/configs/training/dgx-thesis-canary.toml' `
  -RunId 'thesis-canary-001' `
  -LaunchMode Foreground `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 96 `
  -ContainerCpuCount 12 `
  -ContainerMemoryGiB 96
```

The container has no network, receives the selected source release read-only,
and writes only to its unique run directory. It cannot overwrite an existing
run ID. Both smoke and training set the fixed deterministic CUDA requirement
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts.

`phase1-bootstrap.toml` contains 1,000 optimizer steps and is not a foreground
pilot. Only after the full-model canary is healthy may that persistent job be
requested. There is no default detached mode; both the mode and acknowledgement
are required:

```powershell
& .\brain\scripts\dgx\Start-DgxBrainTraining.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -ReleaseId '<release-id>' `
  -ContainerImage 'vllm/vllm-openai:nightly-aarch64' `
  -ConfigRelativePath 'brain/configs/training/phase1-bootstrap.toml' `
  -RunId 'phase1-train-001' `
  -LaunchMode Tmux `
  -AcknowledgeDetached `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 96 `
  -ContainerCpuCount 12 `
  -ContainerMemoryGiB 96
```

No systemd unit is installed or modified.
For detached launches, the command returns success only after the tmux child has
acquired the per-run lock, claimed its log, written a unique readiness
handshake, and its exact-name Docker container is observed live.

### 6. Resume an interrupted run

Do not invoke resume while the original job is running. The resume launcher
refuses an active `pseudo-brain-<run-id>` tmux session, any active or stopped
Docker container record with that exact name, a busy GPU compute queue, or a
held per-run training lock.

Only runs created by this resumable launcher contract are eligible. Missing or
invalid `.training.lock`, `launch_sha256`, CPU/memory metadata, or runtime
boundary metadata is fatal; legacy runs are not guessed or upgraded in place.

The safest selection is the strictly validated `latest.json` pointer:

```powershell
& .\brain\scripts\dgx\Resume-DgxBrainTraining.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -ReleaseId '<original-release-id>' `
  -ContainerImage 'vllm/vllm-openai:nightly-aarch64' `
  -ConfigRelativePath 'brain/configs/training/phase1-bootstrap.toml' `
  -RunId 'phase1-train-001' `
  -UseLatest `
  -LaunchMode Tmux `
  -AcknowledgeDetached `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 96 `
  -ContainerCpuCount 12 `
  -ContainerMemoryGiB 96
```

If `latest.json` was lost but an exact digest was recorded elsewhere, name both
the checkpoint and its required SHA-256:

```powershell
& .\brain\scripts\dgx\Resume-DgxBrainTraining.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -ReleaseId '<original-release-id>' `
  -ContainerImage 'vllm/vllm-openai:nightly-aarch64' `
  -ConfigRelativePath 'brain/configs/training/phase1-bootstrap.toml' `
  -RunId 'phase1-train-001' `
  -CheckpointFile 'step-00000100.pt' `
  -CheckpointSha256 '<64-lowercase-hex-characters>' `
  -LaunchMode Foreground `
  -MinFreeDiskGiB 20 `
  -MinAvailableMemoryGiB 96 `
  -ContainerCpuCount 12 `
  -ContainerMemoryGiB 96
```

Resume is deliberately strict:

- The checkpoint must be a regular, non-symlinked
  `runs/<run-id>/checkpoints/step-XXXXXXXX.pt` file.
- Its bytes must match `latest.json` or the caller-supplied SHA-256.
- It must be the highest checkpoint in the run. Resuming an older file could
  overwrite later checkpoint names, so it is refused.
- A checkpoint at the configured final optimizer step is refused because no
  work remains and the trainer would rewrite that final checkpoint.
- A canonical step-1536 `development-gate-step-XXXXXXXX.json` using
  `rcq_v2_development_v1` with `passed:false` makes the selected
  stage-transition checkpoint terminal. A canonical step-2048
  `final-development-step-XXXXXXXX.json` using
  `rcq_v2_value_development_v1` with `passed:false` likewise makes the final
  completion checkpoint terminal. The remote preflight reconstructs each
  report with the immutable release evaluator. For the completion report it
  also requires the canonical passed step-1536 entry report, its exact SHA-256
  binding, and the registered value-baseline fields. This validates every
  check, metric, scope, and aggregate status before creating a resume attempt
  or starting a container. Latest and exact selection cannot bypass the rule;
  malformed, semantically inconsistent, or linked artifacts fail closed.
- Release ID, source/config bytes, cached image ID, CPU count, memory limit,
  minimum disk/memory gates, container UID:GID, mounts, networking, IPC, PID
  limit, and deterministic CUDA settings must match the original launch.
- The original launch script must exactly match its required SHA-256. The
  resume container is launched by the same immutable image ID, never by its
  mutable user-facing tag.

The original `run.env`, `launch.sh`, logs, and selected checkpoint are never
rewritten by the launcher. Each resume gets a new
`resume-attempts/<timestamp>-<digest>/` directory containing immutable launch
metadata, its own launch script, and an append-only attempt log. The trainer
continues the original metrics stream and creates only later checkpoint step
names. Both `resume.env` and the attempt log record the checkpoint step plus the
exact pre-resume byte and line boundary of `metrics.jsonl`, making any records
after that boundary attributable without rewriting the old metric stream.

### 7. Status and artifact retrieval

Status is one snapshot, not a background monitor. It reports the tmux/container
state, `MemAvailable`, `nvidia-smi`, and the last 80 log lines:

```powershell
& .\brain\scripts\dgx\Get-DgxBrainTrainingStatus.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -RunId 'phase1-train-001'
```

Logs and checkpoints are retrieved separately into a destination that must not
already exist. The remote probe resolves a canonical workspace-contained path
and rejects linked or special files anywhere in recursive artifact trees before
copying. This prevents path escape, accidental local merge, or overwrite:

```powershell
& .\brain\scripts\dgx\Receive-DgxBrainArtifact.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -RunId 'phase1-train-001' `
  -Kind Logs `
  -LocalDestination '.\brain\artifacts\phase1-train-001-logs'

& .\brain\scripts\dgx\Receive-DgxBrainArtifact.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -RunId 'phase1-train-001' `
  -Kind Checkpoints `
  -LocalDestination '.\brain\artifacts\phase1-train-001-checkpoints'

& .\brain\scripts\dgx\Receive-DgxBrainArtifact.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/pseudo-brain' `
  -RunId 'phase1-train-001' `
  -Kind ResumeAttempts `
  -LocalDestination '.\brain\artifacts\phase1-train-001-resume-attempts'
```

Create `brain/artifacts` yourself first; the retrieval tool requires an
existing parent, anchors relative destinations to the Pseudo-Brain repository,
rejects destinations outside it, and never chooses a destination implicitly.

### Historical v1 status and retrieval

Historical releases and runs stay where they were created under
`~/projects/irene-brain`. They are not migrated, renamed, resumed, or used as a
destination for new data. To make the read-only compatibility path deliberate,
status and retrieval require the explicit legacy switch:

```powershell
& .\brain\scripts\dgx\Get-DgxBrainTrainingStatus.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/irene-brain' `
  -RunId '<historical-run-id>' `
  -HistoricalIreneWorkspace

& .\brain\scripts\dgx\Receive-DgxBrainArtifact.ps1 `
  -SshTarget defnotean `
  -RemoteWorkDir '~/projects/irene-brain' `
  -RunId '<historical-run-id>' `
  -Kind Checkpoints `
  -HistoricalIreneWorkspace `
  -LocalDestination '.\brain\artifacts\historical-run-checkpoints'
```

Omitting the switch rejects the old root. Supplying it with the new root also
fails. The remote read path requires the existing v1 ownership marker and has
no sync, launch, resume, delete, rename, or marker-upgrade action.

## Precision and platform notes

DGX Spark is an ARM64 Grace Blackwell system with unified CPU/GPU memory. NVIDIA
documents that `nvidia-smi` may show `Memory-Usage: Not Supported` on this iGPU,
so the guard combines the compute-process query with Linux `MemAvailable` and a
four GiB reserve above the Docker memory limit. The DGX Dashboard is useful for
an independent live view.

Start this model with BF16 autocast and FP32 loss/reduction math. NVIDIA's Spark
PyTorch fine-tuning playbook also defaults to BF16. FP8 is a later measured
optimization, not a bootstrap assumption. Record the exact container image ID,
PyTorch/CUDA versions, config hash, release hash, seed, and checkpoint hash for
every accepted run.

## NVIDIA primary sources

- [DGX Spark system overview and supported remote access](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)
- [NVIDIA Container Runtime for Docker on DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
- [NGC and ARM64 framework containers](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)
- [DGX Spark known issues and unified-memory reporting](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)
- [DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)
- [DGX Dashboard monitoring](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html)
- [NVIDIA DGX Spark PyTorch fine-tuning playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/README.md)
- [NVIDIA PyTorch framework support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/)
