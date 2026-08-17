# Start here: what Pseudo-Brain is doing now

Read this file first. The numbered scientific protocol, exact thresholds, and
hexadecimal identities live in [brain/docs/RCQ_V2_PROTOCOL.md](brain/docs/RCQ_V2_PROTOCOL.md).
This page is the plain-language map: what the project is, what the current
experiment can prove, what to do next, and which documents to open when you
need the precise rules.

## In one paragraph

Pseudo-Brain is one model, not a committee. Shared-weight BrainCell thoughtlets
keep a persistent internal state, talk sparsely, and can emit an action after
any internal cycle. The long-term aim is human-speed closed-loop play. That is
**not** what the current experiment is measuring.

The current experiment is **Reference Candidate Qualification v2 (RCQ-v2)**.
It asks a smaller question: can this already-built model learn a genuinely
state-conditioned W/A/S/D policy, plus a useful value estimate, on the
synthetic moving-shapes task, under a frozen recipe, for one seed.

A pass means only that seed `1702` qualified as a one-seed, open-loop,
teacher-forced reference policy. It is not closed-loop gameplay, not
Minecraft or Pac-Man competence, not proof that distinct thoughts caused the
action, and not an architecture-superiority claim.

## Where we are

| Step | What it is | Status |
|---|---|---|
| 0 | Source lives in this Git repository and on private GitHub | Done. `defnotean/Pseudo-Brain`, branch `defnotean/pseudo-brain` |
| 1 | Write operator documentation and freeze tooling | Done |
| 2 | Freeze implementation source and regenerate the matched-baseline architecture manifest | Done. Live digest `30d4c119…`. Historical campaign pin `52bba6a9…` unchanged. |
| 3 | Run local CPU-only tests, including the regenerated manifest identity | Done. 299 tests passed, one expected POSIX skip. |
| 4 | Build the create-once, target-blind RCQ-v2 registration and record its SHA-256 | Live v2 `6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`. Historical v1 `33f7900c…` preserved. Copy the live digest off-repo. |
| 5 | DGX preflight and immutable release sync | Next for the live v2 tree. Historical release `r20260817t013858z-db1f586ef3a0` stays unused for training. |
| 6 | Trusted pretraining pin, then RCQ smoke, then staging canary | Next after a new v2 sync. First v2 pin `2847e786…` is unused: smoke died because historical v1 JSON is not in the live release. |
| 7 | Pinned 2,048-update reference train on seed 1702 | Only if live v2 smoke and canary both pass. |
| 8 | Newly named qualification after the invariance capture fix | Done. Live file `registrations/rcq-v2-reference-v2.json`. Do not edit v1. |
| 9 | Preclaim, independent review, final authorization, one-shot TEST | Only after this qualification's development gates pass |

Do not skip ahead. Do not open sealed TEST ranges to "check" labels. Do not
resume a failed frozen gate as if it had passed. Stage A historical DGX runs
are recorded failures; they are not a starting checkpoint for RCQ-v2.

## Local rules that always apply

- This Windows workstation stays **CPU-only**. Hide CUDA (`CUDA_VISIBLE_DEVICES=-1`),
  use one thread, and do not capture the screen or inject input. The owner may
  be gaming.
- Accelerator work happens only on the DGX Spark, through the PowerShell
  wrappers under `brain/scripts/dgx/`.
- The Python import name remains `irene_brain` so historical checkpoints stay
  verifiable. Do not rename it as a drive-by cleanup.
- The sibling Irene desktop/chat project is not part of this workspace.

## How to verify locally

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\run_play_safe_tests.ps1
```

That runner is the only supported local test entry. It is CPU-only, one
thread, below-normal priority, and isolated per test module.

To regenerate the matched-baseline architecture manifest after an intentional
source freeze:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\Update-BaselineArchitectureManifest.ps1
```

A changed manifest digest is a **new comparison identity**. Do not silently
rewrite it to make an old claim look current.

To build the RCQ-v2 registration (create-once; replacement is forbidden):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\dgx\New-RcqV2Registration.ps1
```

Copy the printed `REGISTRATION_SHA256` somewhere that is not this repository
and not a training run directory. The live file is
`registrations/rcq-v2-reference-v2.json` and must be included in the immutable
DGX release. Historical `registrations/rcq-v2-reference-v1.json` stays in git
and is not part of the live sync allowlist.

## What "target-blind" means

Registration hashes geometry and source/config identities. It does **not**
construct the sealed TEST sequences, read their pixels, or count their labels.
Those counts are allowed only after a durable one-shot claim. If a registration
file ever contains final-label statistics, discard it.

Sealed replacement TEST ranges (never construct before the claim):

- recipient `[3145728, 3146240)`
- donor `[3146240, 3146752)`
- unused guard `[3146752, 3147264)`

Retired, already-opened ranges must never be reused as evidence. See
[brain/docs/runs/2026-08-16-rcq-v2-test-range-retirement.md](brain/docs/runs/2026-08-16-rcq-v2-test-range-retirement.md).

## Document map

Read in this order unless you already know the file you need:

1. This file — current campaign and checklist.
2. [brain/docs/OPERATOR_GUIDE.md](brain/docs/OPERATOR_GUIDE.md) — how to execute
   each step, in order, with the exact commands.
3. [brain/STATUS.md](brain/STATUS.md) — what is implemented and what already
   failed.
4. [brain/docs/RCQ_V2_PROTOCOL.md](brain/docs/RCQ_V2_PROTOCOL.md) — frozen
   thresholds and stopping rules.
5. [brain/docs/DGX_SPARK_TRAINING.md](brain/docs/DGX_SPARK_TRAINING.md) — Spark
   host, SSH, and container operations.
6. [brain/docs/BASELINE_PROTOCOL.md](brain/docs/BASELINE_PROTOCOL.md) — later
   architecture-comparison rules, not the current qualification.
7. [brain/docs/PLAY_SAFE.md](brain/docs/PLAY_SAFE.md) — local safety policy.
8. [brain/PLAN.md](brain/PLAN.md) — long-term architecture blueprint.

Historical Stage A DGX records are under `brain/docs/runs/`. They document
valid failures. Do not treat them as a green light to scale.

## GitHub

The private source of record is `https://github.com/defnotean/Pseudo-Brain`.
Implementation and documentation are kept in the same commit stream so the
remote copy matches the frozen local tree.
