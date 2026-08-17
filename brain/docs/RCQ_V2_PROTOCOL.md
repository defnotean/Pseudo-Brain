# Reference Candidate Qualification v2 protocol and runbook

Status: implementation-ready protocol. No qualification run or final TEST has
been launched. The canonical registration and its external SHA-256 pin must be
created from the final frozen local source/configuration tree, then included in
the immutable synced release before training begins.

If you are executing the campaign rather than auditing thresholds, start with
[CURRENT_WORK.md](../../CURRENT_WORK.md) and
[OPERATOR_GUIDE.md](OPERATOR_GUIDE.md). This file remains the frozen
scientific contract.

## What this experiment can establish

RCQ-v2 is the smallest candidate-competency experiment for one reference Pseudo-Brain
policy. It asks whether the existing single model can learn a genuinely
state-conditioned W/A/S/D policy and a useful value estimate on the synthetic
moving-shapes task. Pseudo-Brain still uses its shared-weight recurrent thought slots,
workspace, sparse thought communication, recurrent state, and anytime action
heads; RCQ-v2 does not alter that architecture.

A pass means only that seed 1702 qualified as a one-seed, open-loop,
teacher-forced reference policy under this frozen protocol. It is not evidence
of closed-loop gameplay, human-speed control, general game competence,
architecture superiority, causal use of distinct thoughts, or a breakthrough.
Those claims require new closed-loop and matched multi-seed experiments on new
data.

## Frozen training recipe

The sole training recipe is
`brain/configs/training/dgx-rcq-v2-reference.toml`. It is schema 3 and hashes
both stages into one configuration. The run cannot be split into two configs,
truncated through an environment override, or resumed under a different hash.

- Run seed: `1702`.
- Training namespace: split `train`, local offset `1048576`, 8,192 sequences.
- Development namespace: split `validation`, local offset `1048576`, 256
  sequences. This slice is already open and may select the reference candidate;
  it is never final evidence.
- Sequence geometry: 8 timesteps, 2 burn-in timesteps, 6 scored decisions.
- Batch size: 1 sequence; gradient accumulation: 8 sequences per optimizer
  update.
- Schedule: log every 64 updates; evaluate and checkpoint every 256 updates.
- Precision/runtime: one CUDA process, BF16 autocast, TF32 allowed,
  deterministic algorithms, no compilation, and zero data workers.

The run exposes 16,384 training-sequence instances and 98,304 scored training
decisions. At update 1,536 the cursor is epoch 1, batch 4,096; at update 2,048
it is exactly epoch 2, batch 0. This is two passes over 8,192 unique training
sequences, not 98,304 independent examples.

| Stage | Updates | Trainable parameters | Optimizer and objective | Gate |
|---|---:|---|---|---|
| `joint` | `[0, 1536)` | All model/objective parameters | Fresh AdamW, LR `1e-4`, weight decay `0.01`, 20-update warmup; action/value/world/diversity weights `1/.1/.1/.05` | `rcq_v2_development_v1` at 1,536 |
| `value-head-only` | `[1536, 2048)` | Exactly `model.value_per_thought.bias` and `.weight` | Reset AdamW and scheduler, LR `3e-4`, weight decay `0`, no warmup; value weight `1`, all other loss weights `0` | `rcq_v2_value_development_v1` at 2,048 |

The successful transition captures hashes of every non-value state tensor, all
action outputs at every exit, and recurrent-state outputs. Stage 2 must preserve
those hashes exactly in the same registered runtime. This makes action
invariance a structural consequence of the freeze mask and an audited runtime
fact; it does not show that thoughts caused the action.

## Development gates

Both gates use the entire fixed 256-sequence development slice: 1,536 scored
decisions. The registered data invariants are 2,275 active target movement keys,
467 changed decisions, and 1,069 decisions on which copying the previous W/A/S/D
set is exact. A count off this lattice is an input/provenance failure.

The step-1,536 entry gate requires all of the following:

- at least 1,229/1,536 exact W/A/S/D decisions;
- at least 234/467 exact changed decisions;
- sample-macro positive-key recall at least `0.90`;
- no more than 230 movement false-positive keys and 7 opposite-direction
  conflicts;
- zero false positives among the other keyboard keys and zero predicted active
  buttons outside W/A/S/D across all 296 button channels;
- zero off-support target buttons and zero nonzero continuous targets; and
- zero of 16,896 continuous outputs outside the inclusive `[-0.05, 0.05]`
  deadzone.

The entry report records `value_loss` but does not use it to pass stage 1. Its
canonical newline-terminated SHA-256 is embedded in the stage transition and
every later checkpoint.

The step-2,048 completion gate repeats every action and quiescence requirement,
binds the exact passed entry report, and additionally requires:

- final `value_loss <= 0x1.4ff8d56cab52bp-2`
  (`0.328097662694591`);
- final `value_loss <= 0x1.ccccccccccccdp-1` times the step-1,536 value loss
  (a `0.90` ratio); and
- reported development R-squared at least `0.20` against variance
  `0x1.be77815f41fbfp-2` (`0.4360027517691342`).

The R-squared criterion is deliberately explicit but redundant: the absolute
MSE ceiling implies approximately R-squared `0.2475` on this development slice.
The TRAIN-fitted timestep-plus-previous-W/A/S/D baseline underlying the absolute
ceiling is `0x1.754d5eea85785p-2` (`0.36455295854954556`). Exact hexadecimal
identities, rather than rounded decimals, are authoritative.

For context, the already-open TRAIN/development analysis produced:

| Diagnostic | Development value |
|---|---:|
| Target mean | `0.6259369390854818` |
| Target variance | `0.4360027517691342` |
| Zero predictor MSE | `0.8277998034808363` |
| TRAIN global-mean MSE | `0.4361457901076875` |
| TRAIN timestep lookup MSE | `0.3714460879257104` |
| TRAIN timestep + previous-W/A/S/D lookup MSE | `0.36455295854954556` |

These diagnostics used TRAIN plus the already-open development slice only. No
fresh final recipient, donor, or guard sequence was constructed to set them.

## Final TEST namespaces and chronology

Offsets are local to the named split. The old TEST proposal was opened during
evaluator development and is permanently retired:

- opened recipient TEST `[1048576, 1049088)`;
- retired adjacent donor TEST `[1049088, 1049600)`; and
- retired adjacent guard TEST `[1049600, 1050112)`.

No label-derived count from those ranges is a production criterion. The full
chronology is recorded in
`brain/docs/runs/2026-08-16-rcq-v2-test-range-retirement.md`.

The replacement final TEST namespace is sealed until the trusted evaluator has
durably and atomically claimed it:

- 512 recipient sequences `[3145728, 3146240)`, yielding 3,072 scored
  decisions;
- 512 donor-only sequences `[3146240, 3146752)`; and
- unused guard `[3146752, 3147264)`.

The permanent chronology record conservatively notes that a denied fresh-range
constructor invocation may have occurred during guard testing. It created no
dataset object/state or manifest and performed no materialization, indexing,
iteration, content access, or label access. Future tests use only the
target-blind overlap predicate; see
`brain/docs/runs/2026-08-16-rcq-v2-test-range-retirement.md`.

The existing future matched-campaign reservation remains unchanged at
`[2097152, 2097408)`, `[2097408, 2097664)`, and `[2097664, 2097920)`. RCQ-v2
does not consume or supersede it.

Before the final claim, registration may contain only target-blind slice
geometry and manifests. Target-active counts, changed-action counts, value
statistics, donor assignments, and all denominators are post-claim facts. The
evaluator creates a target-blind one-to-one permutation of the 512 donor
sequences. Each recipient keeps one donor for all eight timesteps, each donor is
used once, and the deterministic assignment minimizes mismatch in timestep-wise
previous W/A/S/D state while rejecting every pair with an identical RGB frame.
Normal and RGB-deranged conditions use independent recurrent states, and RGB is
replaced during burn-in as well as scored timesteps.

## Final one-shot criteria

The final evaluator uses batch size 1, the final actuator exit, strict
`logit > 0` button activation, the registered BF16 autocast/TF32 runtime, and a
10,000-resample one-sided 95% recipient-sequence-paired bootstrap. It requires:

- movement exact fraction at least `0.85` and changed-decision exact fraction
  at least `0.60`;
- sample-macro positive recall at least `0.93`, movement false positives at
  most `0.10` per decision, and opposite conflicts at most `0.002` per
  decision;
- zero positive buttons outside W/A/S/D across all 296 button channels and zero
  continuous outputs outside `[-0.05, 0.05]`;
- exact-action accuracy at least `0.10` above the stronger of copying the
  previous W/A/S/D set and the frozen TRAIN timestep/previous-W/A/S/D lookup;
- normal-minus-RGB-deranged exact accuracy at least `0.10`, changed accuracy at
  least `0.20`, and their one-sided bootstrap lower bounds strictly above
  `0.05` and `0.10`, respectively;
- value R-squared at least `0.10`, Pearson correlation at least `0.30`, model
  MSE at most `0.90` times the best frozen TRAIN-fit value baseline, and normal
  MSE at most `0.90` times RGB-deranged MSE.

Zero or malformed denominators fail closed. The final receipt reports the
observed changed denominator rather than assuming a preregistered label count.

## Required preregistration

Before the first training update, publish one canonical, newline-terminated
registration and record its SHA-256 outside the run directory. It must bind at
least:

- raw and canonical training-config hashes;
- immutable source-tree, evaluator-bundle, and batch-source-manifest hashes;
- seed, run ID `dgx-rcq-v2-reference-seed-1702`, model factory, stages, gates,
  runtime protocol, and expected artifact names;
- TRAIN/development manifests and target-blind final recipient/donor manifests;
- sealed, retired, guard, and future-campaign ranges;
- all gate/final thresholds, exact hexadecimal value constants, donor-matching
  seed, bootstrap seed/resample count/index, and the canonical receipt
  directory.

Do not generate or accept a registration that contains final labels or
label-derived invariants. Do not launch if any registered source/config/runtime
identity changes after the registration SHA is pinned; make a new release and a
new registration instead.

The target-blind builder is the only supported registration author. Freeze the
local source and configuration first, purge generated Python caches and other
disallowed source-side files, create the fixed output directory, and run the
builder from that exact local repository *before* release sync:

```text
python -m irene_brain.evaluation.rcq_v2_registration \
  --training-release-root <frozen-local-repository-root> \
  --config <frozen-local-repository-root>/brain/configs/training/dgx-rcq-v2-reference.toml
```

Record the printed `registration_sha256` in an external launch pin. The builder
derives configuration manifests only and reports `sealed_test_examples_opened`
as zero; it does not instantiate any dataset. Release sync allowlists that one
exact registration path in addition to `brain/**`, installs both into the same
immutable release, and the evaluator requires the resulting fixed container
path `/workspace/repo/registrations/rcq-v2-reference-v1.json`. Do not generate
the registration inside an already installed read-only release. Do not edit
source, configuration, or registration after its SHA is pinned; if smoke or
canary work requires a change, discard that registration/release pair and begin
again with a new immutable release.

## DGX run order

The operator commands below are the only supported production sequence. Generic
`Start-DgxBrainTraining.ps1`, `Resume-DgxBrainTraining.ps1`, and
`Invoke-DgxBrainSmoke.ps1` reject the frozen RCQ-v2 reference config
`brain/configs/training/dgx-rcq-v2-reference.toml` and the fixed run ID
`dgx-rcq-v2-reference-seed-1702`. After the pretraining pin exists, production
actions resolve the canonical workspace `~/projects/pseudo-brain`, the pin
directory `qualification-pins/rcq-v2-reference-v1`, and the claim registry
`final-claims` on the host; they do not accept caller-selected workspace,
release, image, config, run, or checkpoint identities. Container Python is
`python3 -I` with argv limited to the subcommand `preclaim`, `final-once`, or
`verify-receipt`. Image launch uses the inspected immutable image ID and
`--pull never`.

1. Freeze the local source/configuration tree, purge generated caches and
   disallowed source entries, build the target-blind registration at the one
   fixed repository path above, and record its SHA-256 outside the repository
   and run directories.
2. Without changing any registered input, run `Invoke-DgxPreflight.ps1` and
   `Sync-DgxBrainRelease.ps1` to install one immutable release containing
   `brain/**` plus that exact registration file. Record the release ID, source
   archive SHA-256, and cached image content digest, and require remote
   read-back of the externally pinned registration SHA at its fixed path.
3. Run the trusted pretraining-pin ceremony and preserve/review the file and
   printed hashes outside the repository and run directories:

   ```powershell
   & .\brain\scripts\dgx\New-DgxRcqV2PretrainingPin.ps1 `
     -SshTarget defnotean `
     -ReleaseId '<release-id>' `
     -ReleaseArchiveSha256 '<archive-sha256>' `
     -RegistrationSha256 '<registration-sha256>' `
     -ContainerImage '<cached-image-reference>' `
     -ContainerImageId 'sha256:<64-lowercase-hex>'
   ```

4. Run `Invoke-DgxRcqV2Smoke.ps1` in the foreground. It derives release and
   image from the pin. It must pass CUDA backward, isolated CPU tests, and the
   bounded trainer smoke.
5. Run `Invoke-DgxRcqStagingCanary.ps1`. Resource gates remain explicit for
   this operational check; release, image, and run identity come from the pin.
   Its first CUDA process stops after update 1; its second process exact-resumes
   checkpoint schema 2 and completes update 2. Require the reset/freeze
   transition, metrics-prefix replay, and non-value/action/state invariance
   receipt.
6. Confirm that the immutable release, smoke receipt, canary receipt, canonical
   registration, pretraining pin, and external registration SHA all agree. Only
   then start the pinned reference:

   ```powershell
   & .\brain\scripts\dgx\Start-DgxRcqV2Reference.ps1 `
     -SshTarget defnotean `
     -AcknowledgeDetached
   ```

   The dedicated action uses run ID `dgx-rcq-v2-reference-seed-1702`, config
   `dgx-rcq-v2-reference.toml`, 12 container CPUs, and 96 GiB memory. It accepts
   no resource or identity overrides.
7. Let the trainer execute the entry gate at 1,536. On pass it resets into the
   value-only stage and checkpoints stage 1 at local step 0. Continue or
   exact-resume with `Resume-DgxRcqV2Reference.ps1 -AcknowledgeDetached` only.
   Stop immediately if either DEVELOPMENT gate fails.
8. Require the passed value-completion report, final invariance report, retained
   step-1536 checkpoint, terminal schema-2 step-2048 checkpoint, `latest.json`,
   and their mutually bound digests.
9. In the same registered release/runtime, run the non-TEST preclaim action.
   It strictly revalidates the source/config/runtime/checkpoint, both gate
   records, checkpoint-bound metrics, RNG and optimizer state, invariants, and
   an exact full-development replay. It categorically forbids TEST construction
   and publishes one immutable readiness receipt:

   ```powershell
   & .\brain\scripts\dgx\Invoke-DgxRcqV2Preclaim.ps1 -SshTarget defnotean
   ```

   Require status `ready_for_once_only_final`,
   `sealed_test_examples_opened: 0`, and externally record the printed
   `readiness_sha256` together with the step-1536 and step-2048 checkpoint
   SHAs and the terminal `latest.json` file SHA.
10. After independent external review of those artifacts, run the trusted
    final-authorization ceremony. It binds the pretraining pin file and
    semantic SHA, retained step-1536 checkpoint path and SHA, terminal
    `latest.json` file SHA, terminal step-2048 checkpoint SHA, readiness file
    path/file SHA/semantic SHA, and the fixed range-claim ID:

    ```powershell
    & .\brain\scripts\dgx\New-DgxRcqV2FinalAuthorization.ps1 `
      -SshTarget defnotean `
      -PretrainingPinSha256 '<pretraining-pin-file-sha256>' `
      -LatestPointerSha256 '<latest.json-file-sha256>' `
      -EntryCheckpointSha256 '<step-00001536-sha256>' `
      -CheckpointSha256 '<step-00002048-sha256>' `
      -ReadinessSha256 '<readiness-file-sha256>'
    ```

11. Then invoke the sole TEST-consuming action once. The dedicated wrapper
    mounts the immutable release, run directory, pins, and workspace-global
    claim registry at their registered container paths:

    ```powershell
    & .\brain\scripts\dgx\Invoke-DgxRcqV2FinalOnce.ps1 `
      -SshTarget defnotean `
      -AcknowledgePermanentTestRetirement
    ```

    Do not rerun this action after a claim exists.
12. Verify the authoritative terminal receipt without opening TEST:

    ```powershell
    & .\brain\scripts\dgx\Test-DgxRcqV2FinalReceipt.ps1 -SshTarget defnotean
    ```

The in-container evaluator entry points remain `python3 -I` running
`irene_brain.evaluation.rcq_v2_torch` with argv `{preclaim|final-once|verify-receipt}`
only. There is no supported operator path that passes `--training-release-root`,
`--config`, `--registration`, `--run-dir`, `--checkpoint`, or related identity
flags into those actions.

The generic trainer and `--evaluate-only test` path reject schema-3 TEST. There
is no public evidence factory, caller-supplied authorization object, or
supported path that intentionally exposes or returns a production score
without first publishing the create-only claim and attempting a terminal
receipt. A claim-only or partial-ledger crash remains a terminal invalid result,
not a supported unreceipted score or a reason to retry.

The once-only enforcement boundary is the canonical workspace's
workspace-global claim registry on the canonical host (or a deliberately
shared registry mounted by every authorized host). It is not a global
cryptographic defense against copying the release and run to another host with
an empty registry. Operators must preserve and back up the authoritative
registry, share that same registry with every authorized evaluator host, and
never clone, clear, reset, or reinitialize it. Losing registry continuity
invalidates the once-only guarantee and does not make the sealed ranges
eligible for reuse.

The source-tree checks, clean source-root policy, and isolated launcher defend
the cooperative frozen-run workflow against accidental shadow modules,
bytecode, native extensions, links, and stale caches. They do not prove safety
against a malicious same-account host racing already loaded code, mutating
process memory, copying the release, or resetting the registry. The immutable
archive/release identity, cached image identity, externally preserved
pretraining pin, launch receipts, and trusted operator/host remain part of the
root of trust; a Python source digest alone is not a hostile-runtime proof.

The create-only pretraining-pin ceremony and the later, separately reviewed
final-authorization ceremony are trusted operator actions. Their canonical
files and printed hashes must be independently reviewed and preserved outside
the run before proceeding. After final authorization, the launch/evaluator
caller, command-line input, and environment are treated as untrusted: the
dedicated action derives the fixed identity chain from the two read-only pins
and fails closed rather than accepting caller-supplied identities. A malicious
operator, root account, or same-UID host process is outside this threat model.

## Failure and stopping rules

- Entry-gate failure writes the canonical failed report and a stage-0
  step-1,536 terminal checkpoint, exits with scientific failure, and forbids a
  futile stage-2 resume. Do not tune on the failed candidate under RCQ-v2.
- Completion-gate failure writes the canonical failed value report, final
  invariance evidence, and a stage-1 step-2,048 terminal checkpoint, then exits
  with scientific failure. Do not open final TEST.
- A crash before a gate may resume only the exact latest checkpoint, cursor,
  optimizer/scheduler/scaler/RNG state, gate receipts, and checkpoint-bound
  metric prefix. A durable post-checkpoint metric tail must replay byte for byte
  before another checkpoint may be written.
- The trusted final evaluator completes every non-TEST authorization check
  before publishing its create-only claim. Once that claim exists in the
  authoritative canonical/shared registry, recipient and donor TEST ranges are
  retired by this protocol regardless of pass, scientific failure, or later
  runtime error. Preserve that registry; a copied host with an empty registry
  is not evidence that the ranges are untouched.
- After claim, publish create-only donor and raw-decision ledgers plus a terminal
  receipt. Ordinary caught errors attempt to publish an invalid-after-claim
  receipt. `SIGKILL`, power loss, or artifact publication/fsync failure can
  instead leave only the claim or partial ledgers with no terminal receipt.
  A canonical create-only terminal receipt is authoritative whenever later
  strict validation proves its self-digest and every immutable artifact it
  references, regardless of the publisher's exit status or missing stdout.
  The receipt cannot be overwritten or downgraded. An absent, partial,
  malformed, self-digest-invalid, or artifact-mismatched receipt is a terminal
  invalid result: the ranges remain retired and the evaluator must never be
  retried.
- Stop after the one final receipt. A failure is the RCQ-v2 result, not a reason
  to rerun, adjust thresholds, or reuse these ranges. Any later architecture
  campaign must use its separately registered 2^21 namespace and its own
  preregistration.
