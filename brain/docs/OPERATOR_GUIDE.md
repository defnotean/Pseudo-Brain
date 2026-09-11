# Current POMDP operator entrypoint (2026-09-11)

The owner has selected **Google Colab instead of DGX Spark** for this effort.
The active remote registration is `registrations/2026-09-11-pomdp-colab-portability-v1.md`.
`--device cuda` is available only for the explicit Colab runtime; defaults and local
tests remain CPU-only. Run the full native-GPU parity preflight before a training
job. Archive and verify local source plus tokenizer/checkpoint identities before
upload, download results before releasing the Colab session, and do not assume
Drive persistence. Candidate checkpoints now embed their tokenizer definition.
The root `scripts/colab_dispatch.py exec-file FILE --session NAME --timeout SECONDS`
uses the installed CLI's file transport and verifies a remote completion marker.
It propagates Python failures even if the underlying CLI returns zero. The legacy
`run` recipe requires a real Drive mount and is not the POMDP experiment entrypoint.

The POMDP effort is separate from the historical RCQ/DGX campaigns below. See
[the active audit](runs/2026-09-11-pomdp-setup-audit.md) and the repository's
`CURRENT_WORK.md` before running anything. From `brain/`, use fresh run names:

```powershell
python -u experiments/train_and_eval_target_relevance.py --audit-only --run-dir runs/NEW-AUDIT-NAME
python -u experiments/train_and_eval_target_relevance.py --audit-only --compensated-state --run-dir runs/NEW-COMPENSATED-AUDIT
python -u experiments/train_and_eval_target_relevance.py --eval-only --checkpoint runs/CANDIDATE/candidate.pt --run-dir runs/NEW-EVAL-NAME
python scripts/run_all_tests.py -q
```

The runner exits 2 on a failed invariant or capability gate. This is a recorded
negative result, not permission to loosen the threshold. Training is capped at
350 steps and must follow its registration. It saves `candidate.pt` in the run
directory, never over `checkpoints/pb_pomdp_champion.pt`. Run `--audit-only` first;
the old float32 champion currently fails strict pointer-enabled parity.
The compensated checkpoint records its precision mode and reloads it explicitly.
Only the fast physical working tensor is 4 KB; total resident state and weights
are larger. Compensated mode has eight logical slots and disallows routing.

The test dispatcher uses pytest so that both pytest and unittest tests run.
All local runs remain CPU-only, CUDA-hidden and single-threaded. The old DGX
gates and sealed artifacts remain unchanged.

The dispatcher and pytest bootstrap set `PSEUDO_BRAIN_CPU_ONLY=1`; this also
disables DirectML discovery, which CUDA visibility alone does not control.
Training/audit module imports no longer change device settings as a side effect.
The two Stage-A continuation TOMLs now retain their exact registered LF bytes on
Windows checkout. Source pins are never rewritten to match changed code.
DGX source releases exclude `src/irene_brain/console/web/`; that local console UI
is not part of the Python-only training package. The remote source guard remains
strict. CPU checkpoint tests support the installed scheduler's static fields,
including both old `verbose` and newer `_is_initial` schemas, without weakening
missing-field validation. Optional absent recurrent state is explicitly hashed.
Outcome-table predictions are computed in canonical action order and rearranged
after the output heads so presentation order cannot perturb hazard probabilities.

---

# Operator guide: freeze, register, then DGX

This is the how-to for the current RCQ-v2 campaign. Exact thresholds, hexadecimal
value identities, and legal stopping rules are in
[RCQ_V2_PROTOCOL.md](RCQ_V2_PROTOCOL.md). Host and container mechanics are in
[DGX_SPARK_TRAINING.md](DGX_SPARK_TRAINING.md). The campaign checklist is
[CURRENT_WORK.md](../../CURRENT_WORK.md).

The live campaign is **Phase 2.5 thought-mediated parallel cognition**. Gate 6
is **FAIL** (`dgx-gate6-matched-gru-v1`, IQM −27.758 vs −25.694). Do not rerun
that K=32 probe. Diagnosis:
[runs/2026-08-20-gate6-fail-diagnosis.md](runs/2026-08-20-gate6-fail-diagnosis.md).
Next named CPU probe: `ranked-k8-unmatched-suppress-v1` (`--smoke` first).
Do not kill Irene sglang on GB10. Play-gated maze-chase distill is
**historical** (38-pellet champion), not the Gate 6 metric.

Historical note: play-gated maze-chase distill v1 is not RCQ.
Spark-only compute. Turn-weighted exclusive CE **passed** the campaign
gate at 32 steps (A×377 + S×103, 38 pellets, 43 collisions, reward −392,
val match 0.25). The 128-step continuation **failed** sticky S (idle×26
+ S×454, 20 pellets, 390 collisions, reward −3880, val match 0.0). Do
not scale 128. `play_moved` is false because reward is below the no-op
floor. Window-32 and episode-windows both **failed** idle no-op.
Exclusive-argmax play decode **unstuck idle** then sticky S.
Exclusive-direction softmax **failed sticky S**. Action-only and
value-only exclusive CE **failed idle no-op**. Tiled 1:1 planner
windows **failed sticky A**. Full-episode tiled update **failed
sticky D**. Multi-episode tiled tiles **failed mixed W/A/D** at 17
pellets. Do not retry or scale those recipes, and do not retune
hold×0.1. Campaign success is pellets ≥ 32 with a non-idle non-D-only
histogram. One GB10 train at a time; named CPU farm jobs run in
parallel on the host. Play-peak / early-stop
(`dgx-play-maze-chase-distill-play-peak-v1`) **passed** on the kept
step-32 peak (38 pellets, A×377 + S×103) and stopped at step 40 (15
pellets). Exact resume from the 32-step champion is messy (config
identity), so the collision-aware probe was a new 32-step run with
`ghost_hit_penalty_v1` and play-peak. That job **failed**: 23 pellets /
20 collisions at the kept step-8 peak, then 15 pellets at step 16.
Champion remains 38 pellets / 43 collisions. Spark GPU is idle. Do
not scale 64 or 128, and do not retune hold×0.1. The next GPU job must
be a new distinct idea.
A turn-hold audit found 261/720 change ticks (82/90 windows have a
change).

The 128-step turn-weighted exclusive-CE launch already ran and **failed**
sticky S. Do not launch it again. Do not start a 256-step scale. Play-peak
kept the 32-step champion and confirmed more steps hurt. Ghost-hit
penalty + play-peak **failed** (23 pellets / 20 collisions). Do not
scale that recipe. Spark GPU is idle. Record:
[runs/2026-08-18-play-gated-maze-chase-distill.md](runs/2026-08-18-play-gated-maze-chase-distill.md).

Scale only if `play-gate.json` shows `campaign_success: true` (pellets ≥ 32
and a WASD histogram that is not idle / D-only). Turn-weighted exclusive
CE passed that gate at 32 steps (38 pellets, A×377 + S×103, collisions
43, val match 0.25). The 128-step continuation failed sticky S (20
pellets, 390 collisions). Do not scale 128. Do not retune hold×0.1.
Ghost-hit penalty + play-peak failed (23 pellets, below the ≥32 floor).
Do not scale it. Spark GPU is idle. The next GPU job must be a new
distinct idea, not a longer fixed budget of turn-weighted exclusive CE.
Failed recipes stay failed: exclusive-argmax sticky S,
exclusive-CE sticky S, 128-step turn-weighted sticky S,
ghost-hit 23 pellets, action-only/value-only idle, tiled-windows sticky A, episode-update
sticky D, multi-episode 17 pellets. Record:
[runs/2026-08-18-play-gated-maze-chase-distill.md](runs/2026-08-18-play-gated-maze-chase-distill.md).

The live `rcq_v2_reference_v2` reference on seed 1702 already failed the
step-1,536 development entry gate. Do not start or resume that run. A later
RCQ attempt needs a newly named qualification. Record:
[runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md](runs/2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md).

RCQ-v3 machinery exists, but **do not run** `New-RcqV3Registration.ps1`
yet. The 2026-08-18 smoke-probe schedule erratum found the probes had
trained for two effective updates; the corrected constant-LR table still
does not put the reference ahead on ≥3/10 metrics, so the review's
proceed-criterion for another RCQ round is unmet. Record:
[runs/2026-08-18-smoke-probe-schedule-erratum.md](runs/2026-08-18-smoke-probe-schedule-erratum.md).
Do not start a generic or pinned Spark train from this guide until that
criterion is met or the owner explicitly overrides it.

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


### Parallel pointer candidate (2026-09-11)

`--pointer-mode parallel_predecessor` selects independent pointer time rows for
training. The score combines the existing content query with a learned boost
when the current input token matches the preceding immutable prompt token.
It has no Python loop over time and retains no previous attention distribution.
The recurrent scan still carries the bounded working state; autoregressive
generation still consumes one newly generated token per step. This does not
make all generation tokens simultaneous.

The default remains `sequential` for historical checkpoints. Candidate files
record `pointer_mode`, and the agent restores it on load. This is a different
copy mechanism requiring capability validation. Repeated predecessors may be
ambiguous; no target labels or generated-token history resolve them at runtime.
All-padded prompts contribute zero, and pointer masks disable copy contributions
on observations. Fast state bytes and parameter tensors are unchanged.

The independent GPU latency and capability gates are registered in
`registrations/2026-09-11-parallel-pointer-v1.md`. Do not promote on speed alone.
An internal-reset correction also preserves the learned scan initial state
before the first reset, regardless of pointer mode.

`--pointer-loss token_mass` is an optional training-only objective registered
separately in `2026-09-11-parallel-pointer-token-mass-v2.md`. It sums attention
over every prompt position containing the correct target token, avoiding an
impossible position distinction when multiple keys and predecessor features
are identical. The default `position` objective preserves historical behavior.
Logs separately report positional and token-mass accuracy (probability >0.5).
Neither metric substitutes for isolated task completion and verified repair.

Prompt/observation ingestion now requests a vocabulary readout only for the last
input token. Every token still updates the full cognitive state. Intermediate
discarded logits are omitted; generation and pointer-active steps retain their
normal readout. State tensors and the next prediction are checked for exact
equality against the full-readout path. This optimization adds no replay buffer.

The experimental `--retention-profile multiscale` assigns half the active
channels their legacy gates, one quarter a 0.99 retention floor and one quarter
a 0.999 floor. Both streaming and scan use the same affine equations. This adds
static model constants but no trainable parameters or fast-state bytes. Slower
channels may acquire new information too slowly; remote numerical and capability
gates in the v3 registration must pass before any promotion. Historical
checkpoints default to the `legacy` profile. All verification is remote while
the owner is gaming.

Measured v2 outcome: 350 Colab steps took 9.88 seconds; the fixed development
suite completed 1/10 A, 4/10 B, 0/10 C tasks. One recovery handled a premature
FINISH and missing file, then wrote a correct implementation and passed hidden
tests. This meets the current recovery predicate but does not establish faulty
code repair. The overall milestone remains failed and the champion is unchanged.

Future episode traces record target-file presence and SHA-256 before and after
actions as evaluation provenance, never as model input. Validated recoveries
are classified as missing-file recovery, rejected-mutation recovery, or repair
of changed existing code. Missing hashes or unchanged files remain unclassified.
The existing overall recovery predicate and historical receipts are preserved;
the new categories prevent broader repair claims than the trace supports.

### Broader corpus source integrity

Structured message, instruction/input/output, and problem/solution records take
precedence over standalone output fields. Unsupported schemas are skipped rather
than serialized into training text. Dataset exhaustion restarts the real stream
without inserting a synthetic example at the boundary.

For reproducible broader experiments, configure `allow_synthetic_fallback=False`
and `hf_dataset_revisions` with immutable revisions. Historical/offline workflows
retain explicit synthetic operation. Every HF reader document identifies its
actual origin; packed blocks and batches retain `source_records`, document hashes
and token counts. Training receipts must record these records, not merely the
requested dataset names. A failed required source stops the stream.

The existing broad synthetic corpus includes algorithm families used by the
old Level C suite. Training on that corpus invalidates an unseen-family claim
for that suite. Broad pretraining requires fresh independent qualification data;
task resets alone do not establish training/evaluation decontamination.

Use `hf_dataset_configs` when a source requires a named subset, such as the
CodeSearchNet `python` subset. Its documentation/code adapter retains both the
request and function source, and provenance includes the selected config.
Source access probes using the dataset viewer establish schema/access only;
training must read the recorded immutable revision and preserve its own bank hash.

For a model that resets state every block, set `document_policy="whole"` on
`StreamingConfig` or `SequencePacker`. This keeps complete documents inside a
block, adds masked padding before an overflowing document, and rejects documents
longer than `seq_len`. Increase the bound or record an explicit exclusion before
packing; there is no silent truncation or dropping. Call `packer.finish()` for a
finite corpus's final partial block. Padding has no targets, resets state, and
appears in source records as `origin="padding"` for accurate token accounting.

Always pass the returned `reset_mask` into the model. Colab CPU tests verify that
packed document logits match independent runs. This gate uses no pointer prompt;
future pointer-enabled packed training must separately isolate each document's
immutable prompt. The default `document_policy="split"` preserves historical
packing but is unsuitable for whole-context assistant training with per-block
state resets. Existing training scripts have not been silently switched.

### Experimental parallel-depth language core

`unified/parallel_depth_model.py` is a separate, unpromoted candidate. Eight
affine recurrent layers each consume the previous layer's current-token output;
each layer scans all time positions in parallel. Residual feed-forward layers
and a tied embedding/readout carry a wider representation between recurrent
updates. It does not invoke the dormant `UnifiedPseudoBrain.recurrent_stack`.

Streaming state is exactly 16 x 64 float32 values per example: eight logical
layer states plus eight numerical residual slots. Parameters and arithmetic
use float64; total model memory is larger than the 4096-byte fast state. The
layer slots are not independently addressable conversation/task threads.
The existing POMDP adapter and champion checkpoint are not switched to this core.

An optional immutable prompt pointer uses independent predecessor/content scores,
with no previous-attention state or generated-token cache. A caller must supply
only the initial task prompt, never the answer or later generated history.
`language_loss(..., chunk_size=256)` recomputes vocabulary readout chunks during
backward to bound their workspace while retaining complete recurrent context.
Chunking changes neither token weights nor next-token target alignment. For
long contexts, the current scan implementation falls back to the PyTorch parallel
scan above 2048 positions; long-context performance is not yet established.

The architecture diagnostic gates are recorded in
`experiments/parallel_depth_v1_registration.md`. Numerical/gradient success is
not learned competence; training and independent evaluation remain required.

Measured remote v1 gate: 22,387,458 parameters; exactly 4096 B fast state; five
tests passed. Full-width native A100 parity at 31/257 tokens was within 5.357e-15.
One full 18,574-token reasoning forward/backward, with response-only supervision
and an 83-token initial pointer prompt, used 7,855,505,408 B peak CUDA allocation.
All eight layers had finite nonzero gradients. This is a feasibility diagnostic
without optimizer steps, not evidence of generalization or frontier capability.
The >2048-token path uses PyTorch parallel scan; full-length streaming parity and
broader training comparisons remain required. Source and receipts are archived
under `brain/runs/colab-source-20260911/parallel-depth-v1`.

### Broad learning comparison protocol

`experiments/broad_pilot_v1_registration.md` defines a first learning comparison
between the eight-layer recurrent candidate and a six-layer causal transformer.
Both use the same tokenizer, tied readout, initial-prompt pointer, complete
documents, example order and optimizer schedule. The transformer is a control
with full-prefix attention, not an implementation of the 4 KB invariant.
Its parameters are float32 and its Flash attention uses bfloat16. Parameter
counts must be within 5%; compute and precision are explicitly not matched.

`build_broad_pilot_corpus.py` freezes 1024 training and 64 development examples
per domain from pinned sources. It joins related questions/repositories/exact
normalized code transitively before splitting, records oversized/unusable
exclusions, and verifies no isolation-key overlap. This does not certify that
semantic paraphrases or all forked repositories are disjoint. Development loss
is a source-derived diagnostic, not a frontier correctness benchmark.

`train_broad_pilot.py` uses one fixed pass, one full document per update, and
response-only loss. It records initialization/final development NLL, per-update
training receipts, final checkpoint hashes, diagnostic greedy responses and
trained recurrent parity. Any timeout, OOM, nonfinite value or parity failure
is incomplete/failed, not a qualified comparison. Models are not promoted.
The bfloat16 baseline gradient-test precision is documented separately in
`broad_pilot_v1_precision_addendum.md`; recurrent parity remains below1e-6.

`data/sequence_buckets.py` provides optional trailing padding to a fixed length
multiple, with appended labels ignored. It never truncates or shifts original
targets and must not be used to obtain a streaming final state. Three isolated
Colab CPU tests passed; CUDA equivalence and measured performance are pending.
This helper is not enabled in the active registered pilot.

Independent evaluation preparation is archived under
`brain/runs/broad-pilot-20260911-v1/evaluation-preparation/independent_eval`.
The 64 requests contain 32 HumanEval and 32 GSM8K tasks, with pinned upstream
snapshots and separate grader data. Model generation must read requests only.
Exact normalized overlap checks found no exclusions against the frozen pilot
corpus; semantic contamination is not certified. Correctness scoring remains
pending, and generated Python needs a verified execution sandbox.

`experiments/generate_independent_pilot.py` accepts an explicit checkpoint hash,
frozen request bank/manifest and a new output directory. It runs greedy generation
with a 512-token limit and fresh state per task, preserving all responses and
recording timeout/failure as incomplete. It does not load grader data.
`experiments/score_independent_math.py` runs separately with the bank and generation
directory, verifies identity and scores the registered math tasks. Missing answers
remain failures in the full denominator; code correctness stays unscored pending
a verified execution sandbox. The protocol is frozen in
`experiments/independent_pilot_v1_registration.md`.

Five tests passed on Colab CPU, including state reset, immutable pointer inputs,
answer-field rejection, strict numerical grading and duplicate/missing task
handling. All32 frozen math references passed the grader and all32 invalid
controls failed. These checks validate the evaluator, not model competence.

`evaluation/code_sandbox.py` now provides verified remote Python execution for
the code diagnostic. It requires Bubblewrap/libseccomp and the documented
Linux runtime; absent dependencies cause failure, never unsandboxed fallback.
Four Colab integration checks passed, followed by32/32 canonical HumanEval
passes and0/32 deliberately incorrect control passes. Generated model outputs
have not been scored yet. See `experiments/code_sandbox_protocol.md` for exact
filesystem, namespace, syscall, resource and output restrictions and the
same-interpreter grader limitation. Execution controls do not establish model
competence or adversarial grading integrity.

Add `--include-code` to `score_independent_math.py` to score HumanEval as well.
The adapter runs every canonical/incorrect control before model answers, accepts
complete Python functions with at most one surrounding code fence, and never
repairs malformed output. All missing/incorrect responses remain in totals.
Seven remote tests and a full CLI run on explicitly labeled reference controls
passed. This verifies scoring integration; actual model scores remain pending.
The interpretation rules are frozen in `experiments/independent_code_v1_addendum.md`.

Before checkpoint generation, the recurrent evaluator now checks trained
streaming/parallel parity at31/257/2049 tokens with a reset. It rejects nonfinite
values, a changed4 KB physical state, or logit error>=1e-6. This does not cover
the full32768-token bound. Six remote CPU tests passed for the evaluator gate.

Broad pilot v1 hit its1800-second training limit at933/3072 recurrent updates;
the transformer arm did not start. Its partial checkpoint and logs are preserved
under `brain/runs/broad-pilot-20260911-v1/recurrent`. A partial-checkpoint diagnostic
is allowed only with that incomplete status carried into the result.

`bucket_gpu_gate.py` and `bucket_gpu_gate_registration.md` define a separate
GPU check for padding equivalence and compilation costs. Conditional v2 runner
`train_broad_bucket_v2.py` uses the same fixed data/order/budget and only appends
ignored padding. Its48-document encoding check passed on Colab CPU; GPU gates
have not run. Do not launch v2 unless those gates pass and prior GPU work is
terminal. Details are frozen in `broad_bucket_v2_registration.md`.

Current outcomes supersede that pending status: the GPU bucket gate failed the
transformer logit/loss thresholds, so conditional v2 must not launch as written.
The recurrent padding checks passed; timing was skipped after the failure.
See `broad_bucket_v2_resolution.md`; failed receipts remain preserved.

The933-update v1 checkpoint completed its64-task diagnostic:0/32 HumanEval,
0/32 GSM8K, all responses at512 tokens, median only4 distinct tokens. Trained
parity passed through2049 tokens with maximum error2.3093e-14 and4 KB state.
A six-prompt pointer ablation did not remove repetition. These are explicit
negative capability results, not a qualified model or transformer comparison.

The separate recurrent-only v3 timing gate passed: cold forward/backward time
44.09→13.94s across eight registered lengths, warm time0.406→0.419s, and cubins
8→2. These are compilation diagnostics, not full training throughput. CPU checks
confirmed recurrent-only padding and identical baseline input/label/prompt tensors.
`train_broad_recurrent_bucket_v3.py` is now running fresh initializations on Colab
with the original data/order/budget. Its protocol is
`recurrent_bucket_v3_registration.md`; source and gates are archived under
`brain/runs/broad-pilot-20260911-v3/source`. No v3 capability result exists yet.

Colab CLI connection expiry can produce404/401 and delete its local alias while
the VM/job remains alive. Inspect the authenticated assignment list and match
the endpoint to creation history before deciding the job is lost. Refresh the
proxy connection and reuse the existing kernel; never restart just because
the local alias disappeared. This recovery succeeded without restarting v3's
preflight timing job.

The first96 actual recurrent training updates matched v1's example order and
learning rates, with maximum loss difference1.77636e-15. Their elapsed training
time fell from185.1887s to10.7053s, including compilation/optimizer work. This
early-run comparison supports the optimization's intended effect; a full training
comparison and new capability assessment remain pending.
