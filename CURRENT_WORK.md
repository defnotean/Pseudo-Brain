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

The current experiment is **Play-gated maze-chase distill v1**
(`play_gated_maze_chase_distill_v1`). It asks whether the existing thesis
thought-field can learn to play maze-chase closed-loop after planner
distillation, measured by play (reward / collisions / pellets), not by
action loss. Probe v2 / 128-step sat at sticky D (~10 pellets). Window-32
and episode-windows failed idle no-op (9 pellets, mask 0). Exclusive
WASD argmax **unstuck idle** then stuck on S (S×478 + A×2). Exclusive
softmax loss matching that decode **failed sticky S**: histogram S×480,
20 pellets, 391 collisions, reward −3890. Action-only exclusive CE
**failed idle no-op** (mask 0 × 480, 9 pellets, reward −161; val match
0.0; logits ≈ −5). Value-only exclusive CE **failed idle no-op** (mask
0 × 480, 9 pellets, reward −161; val match 0.0; inactive logit max
≈ −4.82). Tiled 1:1 planner windows **failed sticky A**: histogram
A×476 + D×4, 15 pellets, 18 collisions, reward −165; val exclusive-argmax
match **0.083** (down from exclusive-CE 0.167). Full-episode tiled
update (accum 30) **failed sticky D**: histogram **D×431 + A×49**, 10
pellets, 16 collisions, −150; val exclusive-argmax match **0.417**
(= teacher D, not a ranking gain). Do not scale accumulation-30.
Closed-loop BC at spawn is not next: off-policy planner labels on
idle/W/A/D were S×32. Next GPU probe is multi-episode tiled tiles at
the same accum 30 (`dgx-play-maze-chase-distill-multi-episode-v1`), now
live on Spark release `r20260818t192855z-6d85cc69dd69`. Spark CPU farm
also finished planner seeds 132–147 (5/16 clear), tiled teacher mix,
three-episode coverage, and an episode-update thoughtlet dump
(open-loop A; closed-loop sticky D). Record:
[brain/docs/runs/2026-08-18-play-gated-maze-chase-distill.md](brain/docs/runs/2026-08-18-play-gated-maze-chase-distill.md).

All training, probes, and play evals run on the Spark. This workstation is
orchestration only (git, docs, DGX wrappers, SSH, hashes).

RCQ-v2 seed `1702` is **terminally failed**. Do not resume, retune, or open
TEST on v2. Do not create `registrations/rcq-v3-reference-v1.json` for this
campaign.

## Where we are

| Step | What it is | Status |
|---|---|---|
| 0 | Source lives in this Git repository and on private GitHub | Done. `defnotean/Pseudo-Brain`, branch `defnotean/pseudo-brain` |
| 1 | Write operator documentation and freeze tooling | Done |
| 2 | Freeze implementation source and regenerate the matched-baseline architecture manifest | Done. Live digest `5e0f2536…` (2026-08-18 exclusive WASD softmax loss on maze-chase). Historical `eda3cf38…`, `78ba9cfc…`, `8a41131e…`, `f4e9b355…`, `eb46988b…`, `5decb402…`, `30d4c119…`, and first-matched pin `52bba6a9…` unchanged. |
| 3 | Run local CPU-only tests, including the regenerated manifest identity | Done. 299 tests passed, one expected POSIX skip. |
| 4 | Build the create-once, target-blind RCQ-v2 registration and record its SHA-256 | Live v2 `6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`. Historical v1 `33f7900c…` preserved. Copy the live digest off-repo. |
| 5 | DGX preflight and immutable release sync | Done for live release `r20260817t021531z-7a2967ebec60`. Historical v1 release stays unused for training. |
| 6 | Trusted pretraining pin, then RCQ smoke, then staging canary | Done on pin `adf79ccc…`. Smoke and canary both passed. First v2 pin `2847e786…` is unused. |
| 7 | Pinned 2,048-update reference train on seed 1702 | Failed. Official `rcq_v2_development_v1` at step 1,536 is `passed:false` (9 opposite conflicts, 875 continuous outsides). Terminal for this candidate. Do not resume. |
| 8 | Newly named qualification after the invariance capture fix | Done. Live file `registrations/rcq-v2-reference-v2.json`. Do not edit v1. |
| 9 | Preclaim, independent review, final authorization, one-shot TEST | Blocked. This qualification failed the entry gate. Do not preclaim or open TEST. |
| — | RCQ-v3 registration ceremony | **Deferred.** Do not run `New-RcqV3Registration.ps1`. |
| — | Current campaign | Play-gated maze-chase distill v1. Sticky D at 32/128 spawn-only. Window-32 and episode-windows failed idle. Exclusive argmax unstuck idle then sticky S. Exclusive softmax failed S×480. Action-only exclusive CE failed idle no-op. Value-only exclusive CE failed idle no-op. Tiled 1:1 windows failed sticky A (A×476 + D×4, 15 pellets, val match 0.083). Full-episode tiled update failed sticky D (D×431 + A×49, 10 pellets, val match 0.417 = teacher D). Next GPU: multi-episode tiled tiles at accum 30. Do not scale the failed recipes. |
| — | Compute | Spark only. One scientific GPU train at a time on the GB10; many named CPU jobs in parallel on the ARM host. Generic wrappers, never RCQ-v2 start/resume. |

Do not skip ahead. Do not open sealed TEST ranges to "check" labels. Do not
resume a failed frozen gate as if it had passed. Stage A historical DGX runs
are recorded failures; they are not a starting checkpoint for RCQ-v2.
Do not start another RCQ round while the smoke proceed-criterion is unmet.

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

The RCQ-v3 qualification (`rcq_v3_reference_v1`) is fully preregistered in
code and config but its registration file does not exist yet. **Do not
create it while the smoke proceed-criterion is unmet.** The 2026-08-18
constant-LR erratum confirmed the reference does not lead the twelve-variant
suite; jumping to another RCQ round would freeze a campaign the cheap
signal does not support. Record:
[brain/docs/runs/2026-08-18-smoke-probe-schedule-erratum.md](brain/docs/runs/2026-08-18-smoke-probe-schedule-erratum.md).

If that criterion is later met, or the owner explicitly overrides it, the
ceremony sequence (console-attached terminal, in order) is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\dgx\New-RcqV3Registration.ps1
```

then commit the new `registrations/rcq-v3-reference-v1.json`, sync a fresh
release, pin, smoke, canary, and start the reference with the
`Invoke-DgxRcqV3*` / `Start-DgxRcqV3Reference.ps1` wrappers. The exact
sequence and the frozen D1–D5 decisions are in
[brain/docs/runs/2026-08-17-rcq-v3-reference-v1-preregistration.md](brain/docs/runs/2026-08-17-rcq-v3-reference-v1-preregistration.md).
That wrapper is create-once: running it now would freeze a registration
this campaign is not ready to spend.

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
9. [brain/docs/ROADMAP_TO_PACMAN.md](brain/docs/ROADMAP_TO_PACMAN.md) — the path
   from the failed RCQ-v2 to a 60 Hz Pac-Man-like arcade proof, including the
   now-frozen RCQ-v3 redesign decisions. The owner-run v3 ceremony is
   deferred until the smoke proceed-criterion is met.

Historical Stage A DGX records are under `brain/docs/runs/`. They document
valid failures. Do not treat them as a green light to scale.

## GitHub

The private source of record is `https://github.com/defnotean/Pseudo-Brain`.
Implementation and documentation are kept in the same commit stream so the
remote copy matches the frozen local tree.
