# Start here: what Pseudo-Brain is doing now

Read this file first. The live campaign is Phase 2.5 thought-mediated
parallel cognition. Gates and stopping rules are in
[brain/docs/preregistrations/2026-08-20-thought-mediated-parallel-cognition.md](brain/docs/preregistrations/2026-08-20-thought-mediated-parallel-cognition.md).
The numbered RCQ-v2 protocol and hexadecimal identities remain in
[brain/docs/RCQ_V2_PROTOCOL.md](brain/docs/RCQ_V2_PROTOCOL.md) for the frozen
historical qualification. This page is the plain-language map: what the
project is, what the current experiment can prove, what to do next, and
which documents to open when you need the precise rules.

## In one paragraph

Pseudo-Brain is one model, not a committee. Shared-weight BrainCell thoughtlets
keep a persistent internal state, talk sparsely, and can emit an action after
any internal cycle. The long-term aim is human-speed closed-loop play.

**ACTIVE FRONTIER:**
Phase 2 / 2.5 final mechanistic closure (collapse-resistance mechanism)

**NEXT MAJOR PHASE (do not start until Phase 2 closes):**
Phase 2.6 Foundation Hardening — see
[brain/docs/MASTER_ROADMAP.md](brain/docs/MASTER_ROADMAP.md) (canonical forward roadmap)

**LONG-TERM TARGET [ASPIRATIONAL]:**
frontier-scale general cognitive foundation architecture; current evidence is
tiny research-scale prototypes only

**Latest verified milestone:**
c387503

**CURRENT QUESTIONS:**
- WHY does PB K=32 resist catastrophic training collapse? (mechanistic battery
  on preserved checkpoints + pre-collapse trajectory analysis — telemetry and
  checkpoints captured in Spark `runs/mechanism-discovery-v1/`)
- Fresh independent-training-seed confirmation once a mechanism hypothesis is frozen
- Persistence/reset causal battery on surviving checkpoints
- Task-specificity boundary: multi-latent uncertainty (PB wins reliability) vs
  single-cue ephemeral memory (GRU wins return)

**Discovery-set result driving closure [MEASURED, n=10 independent training seeds,
8-hypothesis escalation, 3000 steps]:** PB K=32 catastrophic collapse 1–3/10 vs GRU
6–7/10 (Fisher p≈0.027); higher PB median return and Stage-1 survival. Strong
empirical evidence; NOT yet a confirmed claim. Ephemeral-memory negative replication
preserved. Records:
[brain/docs/runs/2026-08-22-dgx-reliability10.md](brain/docs/runs/2026-08-22-dgx-reliability10.md),
[brain/docs/runs/2026-08-22-reliability-statistics.md](brain/docs/runs/2026-08-22-reliability-statistics.md),
[brain/docs/runs/2026-08-22-ephemeral-reliability-negative.md](brain/docs/runs/2026-08-22-ephemeral-reliability-negative.md).

The live experiment is **Phase 2.5 thought-mediated parallel cognition**
(`PREREG-PHASE2-REVISED-THOUGHT-MEDIATED-V1`). Main action intent must flow
through per-thought proposals; the direct belief-to-action bypass is gone.

**LEGACY PREREGISTRATION:**
ranked-k8-unmatched-suppress-v1
status: completed/archived side experiment (local CPU, full run)
verdict: FAIL (IQM advantage 1.11% < 10% margin; no superiority claimed).
Results: `C:\Users\Demon\AppData\Local\Temp\ranked_k8_full_results.json`.
K=4 follow-up (the preregistration's prescribed distinct second idea): smoke
FAIL — PB IQM −141.667 vs GRU −24.467 (−479%), knockout sanity failed
(knockout *improved* Family B, −92.4%); full not launched (smoke not ahead).
Record: [brain/docs/runs/2026-08-22-ranked-k4-smoke-fail.md](brain/docs/runs/2026-08-22-ranked-k4-smoke-fail.md).
Branch closed.

**Gate 6 is FAIL.** Named CPU probe `dgx-gate6-matched-gru-v1` finished while
GB10 stayed on Irene sglang. Thought-mediated IQM **−27.758** vs Proposal-GRU
**−25.694** (**−8.04%**, GRU wins). Family-B knockout sanity **passed** (1343%
mean degradation): thoughts still sit on the action path. That does **not**
beat the matched GRU. Architecture superiority is **not** claimed. Do not
retry the same probe as if it passed. Record:
[brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md](brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md).

Diagnosis `why-gru-wins-despite-knockout-v1` (not a rerun): seed 45 knockout
**0%** and Family E **−796 vs −41.5** likely decide the mean; extra K
anti-scales (98/98/32/−98/−98); scramble/stale/donor are 0% because slots are
interchangeable. Next named design is **`ranked-k8-unmatched-suppress-v1`**
(cap K=8, unmatched-slot suppression, multiplicity Q(a), collapse/register
guard). Smoke first (seeds 43+45, 60 steps); full 5×300 only if smoke PB is
ahead. Record:
[brain/docs/runs/2026-08-20-gate6-fail-diagnosis.md](brain/docs/runs/2026-08-20-gate6-fail-diagnosis.md).

Closed-loop play is still the Aug 18 **38-pellet** turn-weighted distill
champion versus the planner's **284** pellets (0 neural clears). Variant E
(5.65 pellets / 83 catches) is historical and has no persisted maze-chase
checkpoint. RCQ-v2 seed `1702` is **terminally failed**. Do not resume,
retune, or open TEST on v2. Do not create `registrations/rcq-v3-reference-v1.json`.

All training, probes, and play evals run on the Spark. This workstation is
orchestration only (git, docs, DGX wrappers, SSH, hashes).

### Historical: Variant E and the 38-pellet distill champion

**Variant E (Full Adaptive System, Seed 43/42)** (`full_adaptive_system_v1`)
was the 2026-08-18 balanced ablation champion: **5.65** mean pellets and
**83** ghost catches on 20 held-out seeds. Variants F (7.80 pellets, reckless)
and G (4.90 pellets, conservative) were exploratory. Seed-sensitivity of E
is a closed historical question, not the live campaign. Record:
[brain/docs/runs/2026-08-18-five-way-cognitive-ablation.md](brain/docs/runs/2026-08-18-five-way-cognitive-ablation.md).

The maze-chase play artifact remains the 32-step turn-weighted exclusive-CE
champion (`dgx-play-maze-chase-distill-turn-weighted-v1`, 38 pellets). Do not
scale 128. Do not retune hold ×0.1.

## Where we are

| Step | What it is | Status |
|---|---|---|
| 0 | Source lives in this Git repository and on private GitHub | Done. `defnotean/Pseudo-Brain`, branch `defnotean/pseudo-brain` |
| 1 | Write operator documentation and freeze tooling | Done |
| 2 | Freeze implementation source and regenerate the matched-baseline architecture manifest | Done. Live digest `7aeaa44e…` (2026-08-19 Topological goal routing & counterfactual foresight). Historical pins unchanged. |
| 3 | Run local CPU-only tests, including the regenerated manifest identity | Done. 59 test modules passed (678 unit tests). |
| 4 | Build the create-once, target-blind RCQ-v2 registration and record its SHA-256 | Live v2 `6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`. Historical v1 `33f7900c…` preserved. Copy the live digest off-repo. |
| 5 | DGX preflight and immutable release sync | Done for live release `r20260817t021531z-7a2967ebec60`. Historical v1 release stays unused for training. |
| 6 | Trusted pretraining pin, then RCQ smoke, then staging canary | Done on pin `adf79ccc…`. Smoke and canary both passed. First v2 pin `2847e786…` is unused. |
| 7 | Pinned 2,048-update reference train on seed 1702 | Failed. Official `rcq_v2_development_v1` at step 1,536 is `passed:false` (9 opposite conflicts, 875 continuous outsides). Terminal for this candidate. Do not resume. |
| 8 | Newly named qualification after the invariance capture fix | Done. Live file `registrations/rcq-v2-reference-v2.json`. Do not edit v1. |
| 9 | Preclaim, independent review, final authorization, one-shot TEST | Blocked. This qualification failed the entry gate. Do not preclaim or open TEST. |
| — | RCQ-v3 registration ceremony | **Deferred.** Do not run `New-RcqV3Registration.ps1`. |
| — | Phase 0 / Phase 1 | **Locked.** Phase 0 7/7; Phase 1 9/9 on DGX Spark. |
| — | Phase 2 original thesis | **Falsified.** Unmediated designs tied or lost to matched GRU. |
| — | Live campaign | **Phase 2.5** thought-mediated parallel cognition. |
| — | Phase 2.5 Gate 6 | **FAIL.** CPU probe `dgx-gate6-matched-gru-v1`: thought-mediated IQM **−27.758** vs Proposal-GRU **−25.694** (−8.04%; GRU wins). Knockout sanity passed (1343%). Thoughts remain causal; not better than matched GRU. Architecture superiority not claimed. Do not retry the same probe. Record: [brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md](brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md). Diagnosis: [brain/docs/runs/2026-08-20-gate6-fail-diagnosis.md](brain/docs/runs/2026-08-20-gate6-fail-diagnosis.md). Next named design `ranked-k8-unmatched-suppress-v1`. |
| — | Play competence | Neural **38 pellets / 0 clears** vs planner **284 / 2 clears**. Record: [brain/docs/runs/2026-08-20-play-competence-closed-loop.md](brain/docs/runs/2026-08-20-play-competence-closed-loop.md). |
| — | Historical champion | Variant E: **5.65 pellets / 83 catches**. Distill play champion: 38 pellets. |
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

## Research horizon

The project maintains a strict, non-derailing three-tiered hierarchy:

1. **CURRENT WORK (Active Priority):**
   - **Phase 2.5 is live.** Gate 6 **FAIL**: thought-mediated IQM **−27.758** vs
     Proposal-GRU **−25.694** (**−8.04%**, GRU wins). Knockout sanity passed
     (1343% mean Family-B degradation): thoughts remain causal. They are **not**
     better than the matched GRU. Architecture superiority is not claimed. Do
     not retry `dgx-gate6-matched-gru-v1` as if it passed. Record:
     [brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md](brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md).
   - **Play competence (2026-08-20, Spark CPU):** thought-mediated campaign and
     Variant E have **no persisted maze-chase checkpoints**. Re-eval of the
     turn-weighted distill champion (`e58f323f…`) vs planner:
     protocol `distill-direct-5-9-240` neural **38 pellets / 43 collisions / 0
     clears** vs planner **284 / 1 / 2 clears**; protocol
     `e-heldout-direct-2001-2020-120` neural **8.55 mean pellets / 265 collisions
     / 0 clears** vs planner **84.5 / 21 / 0**. Still much worse than the planner.
     GB10 left alone. Record: [brain/docs/runs/2026-08-20-play-competence-closed-loop.md](brain/docs/runs/2026-08-20-play-competence-closed-loop.md).
   - Phase 2.5 CPU diagnostics are in-tree: belief collapse / VoI, multi-horizon
     ambiguity arena, sequential probe resolution, analytic WAIT-vs-commit VoI,
     and a multiplicity-proof action aggregator. Play-safe tests cover them.
     Record: [brain/docs/runs/2026-08-20-phase25-belief-collapse-diagnostics.md](brain/docs/runs/2026-08-20-phase25-belief-collapse-diagnostics.md).
     Do not launch the full thought-mediated Spark campaign from this workstation;
     use `--cpu-smoke` only. Do not resume RCQ-v2.
   - **CPU-only play-safe test suite maintenance and manifest integrity.**
   - **Phase 2.5 frontier campaigns completed on local CPU (2026-08-21):** definitive
     8-hypothesis escalation (K=8 best return, beats multi-branch GRU) and the
     ephemeral-memory campaign (GRU leads return; K=8 leads accuracy; reset > normal
     for all Pseudo-Brain variants). Record:
     [brain/docs/runs/2026-08-21-phase25-cpu-frontier-campaigns.md](brain/docs/runs/2026-08-21-phase25-cpu-frontier-campaigns.md).
     Behavioral results only — Spark (GB10) is hosting Qwen, so no latency/Gate-8
     comparison from these runs.

2. **NEAR-TERM RESEARCH:**
   - Real-time $60\text{ Hz}$ closed-loop arcade benchmark (Pac-Man family, frame skip 1, latency $\text{p99} < 16.67\text{ ms}$).
   - Hardware capture-to-control latency harness (RTX 5070 non-gaming window).
   - Procedural skill ladder data scaling (junctions, occlusion memory, key/door sequencing).
   - Named follow-up `ranked-k8-unmatched-suppress-v1` after Gate 6 FAIL diagnosis; do not rerun `dgx-gate6-matched-gru-v1`.

3. **LONG-TERM GENERAL AGENT GOAL:**
   - Single general-purpose cognitive agent with persistent task belief, dynamic subgoals, and multi-thoughtlet decomposition ([PLAN.md §43](brain/PLAN.md)).
   - Unified sensorimotor abstraction over digital software tools (`READ_FILE`, `SEARCH_WEB`, `RUN_COMMAND`, `EDIT_CODE`) and physical robotics.
   - Pretrained language encoder ("ears") and decoder ("mouth") interfacing with Pseudo-Brain's recurrent core.
   - Latent counterfactual tool evaluation and prediction-error self-correction across digital and physical domains.

## Document map

Read in this order unless you already know the file you need:

1. This file — current campaign and checklist.
2. [brain/docs/OPERATOR_GUIDE.md](brain/docs/OPERATOR_GUIDE.md) — how to execute
   each step, in order, with the exact commands.
3. [brain/STATUS.md](brain/STATUS.md) — what is implemented and what already
   failed.
4. [brain/docs/preregistrations/2026-08-20-thought-mediated-parallel-cognition.md](brain/docs/preregistrations/2026-08-20-thought-mediated-parallel-cognition.md) — Phase 2.5 gates.
5. [brain/docs/RCQ_V2_PROTOCOL.md](brain/docs/RCQ_V2_PROTOCOL.md) — frozen
   thresholds and stopping rules.
6. [brain/docs/DGX_SPARK_TRAINING.md](brain/docs/DGX_SPARK_TRAINING.md) — Spark
   host, SSH, and container operations.
7. [brain/docs/BASELINE_PROTOCOL.md](brain/docs/BASELINE_PROTOCOL.md) — later
   architecture-comparison rules, not the current qualification.
8. [brain/docs/PLAY_SAFE.md](brain/docs/PLAY_SAFE.md) — local safety policy.
9. [brain/PLAN.md](brain/PLAN.md) — long-term architecture blueprint and general-agent roadmap (§43).
10. [brain/docs/ROADMAP_TO_PACMAN.md](brain/docs/ROADMAP_TO_PACMAN.md) — the path
    from the failed RCQ-v2 to a 60 Hz Pac-Man-like arcade proof and beyond (§9).

Historical Stage A DGX records are under `brain/docs/runs/`. They document
valid failures. Do not treat them as a green light to scale.

## GitHub

The private source of record is `https://github.com/defnotean/Pseudo-Brain`.
Implementation and documentation are kept in the same commit stream so the
remote copy matches the frozen local tree.
