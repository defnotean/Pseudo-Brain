# Start here: what Pseudo-Brain is doing now

Read this file first. The numbered scientific protocol, exact thresholds, and
hexadecimal identities live in [brain/docs/RCQ_V2_PROTOCOL.md](brain/docs/RCQ_V2_PROTOCOL.md).
This page is the plain-language map: what the project is, what the current
experiment can prove, what to do next, and which documents to open when you
need the precise rules.

## In one paragraph

Pseudo-Brain is one model, not a committee. Shared-weight BrainCell thoughtlets
keep a persistent internal state, talk sparsely, and can emit an action after
any internal cycle. The long-term aim is human-speed closed-loop play.

The current **best single-run balanced checkpoint** is **Variant E (Full Adaptive System, Seed 43/42)**
(`full_adaptive_system_v1`). It combines adaptive thought-update gating ($\alpha$), multi-horizon
future prediction, and situation-dependent dynamic cognitive depth ($C \in [1, 6]$).
Across the initial **20 held-out evaluation seeds** (2,400 decision steps), Variant E achieved:
- **Mean pellets collected**: **5.65** (highest among safe models; +71.2% over baseline 3.30).
- **Total ghost catches**: **83** (vs **727** for baseline, an **88.6% reduction**).
- **Situation-dependent cognitive cycles**: ~1.35 cycles in open corridors vs ~3.80 cycles in ghost hazards.
Record: [brain/docs/runs/2026-08-18-five-way-cognitive-ablation.md](brain/docs/runs/2026-08-18-five-way-cognitive-ablation.md).

### Immediate Priority: Training Robustness & Multi-Seed Policy Variance
Early findings from the 5-seed replication battery show that while Variant E achieves a strong mean of **6.36 pellets / episode**, safety policy varies dramatically across random seeds:
- **Good Seeds (42, 43):** 125–155 catches (disciplined hazard evasion).
- **Reckless Seeds (45, 46):** 689–734 catches (hazard blindness / collapsed penalty weighting).

**Architectural Distinction:** An individual *Variant E checkpoint* can be outstanding, but whether *Variant E reliably trains into a safe policy* is currently an active open investigation. Diagnosing internal telemetry differences (danger recall, gate dynamics, thought rank, risk weighting) between Good and Bad seeds is now prioritized over adding new cognitive modules.

Recent exploratory experiments:
- **Variant F (Counterfactual Foresight)**: High-reward candidate (**7.80 pellets**), evaluating multi-seed stability.
- **Variant G (Topological Goal Routing)**: Conservative candidate (**4.90 pellets**), evaluating whether topological routing provides multi-seed stability.
- Evaluation principle: Task performance $\uparrow$ while maintaining safety bounds $\downarrow$. Multi-seed distribution ($\text{mean} \pm \text{std}$) determines architectural validity, not single lucky checkpoints.

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
| 2 | Freeze implementation source and regenerate the matched-baseline architecture manifest | Done. Live digest `7aeaa44e…` (2026-08-19 Topological goal routing & counterfactual foresight). Historical pins unchanged. |
| 3 | Run local CPU-only tests, including the regenerated manifest identity | Done. 59 test modules passed (678 unit tests). |
| 4 | Build the create-once, target-blind RCQ-v2 registration and record its SHA-256 | Live v2 `6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690`. Historical v1 `33f7900c…` preserved. Copy the live digest off-repo. |
| 5 | DGX preflight and immutable release sync | Done for live release `r20260817t021531z-7a2967ebec60`. Historical v1 release stays unused for training. |
| 6 | Trusted pretraining pin, then RCQ smoke, then staging canary | Done on pin `adf79ccc…`. Smoke and canary both passed. First v2 pin `2847e786…` is unused. |
| 7 | Pinned 2,048-update reference train on seed 1702 | Failed. Official `rcq_v2_development_v1` at step 1,536 is `passed:false` (9 opposite conflicts, 875 continuous outsides). Terminal for this candidate. Do not resume. |
| 8 | Newly named qualification after the invariance capture fix | Done. Live file `registrations/rcq-v2-reference-v2.json`. Do not edit v1. |
| 9 | Preclaim, independent review, final authorization, one-shot TEST | Blocked. This qualification failed the entry gate. Do not preclaim or open TEST. |
| — | RCQ-v3 registration ceremony | **Deferred.** Do not run `New-RcqV3Registration.ps1`. |
| — | Current champion | **Variant E (Full Adaptive System)**: **5.65 pellets / 83 catches** (-88.6% catches vs baseline 727). |
| — | Active investigation | 5-training-seed statistical replication battery (E vs F vs G) & calibrated counterfactual risk utility. |
| — | Phase 2.5 Gate 6 | **FAIL.** CPU probe `dgx-gate6-matched-gru-v1` finished: thought-mediated IQM **−27.758** vs Proposal-GRU **−25.694** (−8.04%; GRU wins). Knockout sanity passed (mean degradation 1343%). Architecture superiority not claimed. Record: [brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md](brain/docs/runs/2026-08-20-gate6-matched-proposal-gru.md). |
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
   - **Phase 2.5 Gate 6 (2026-08-20):** **FAIL.** Thought-mediated IQM −27.758 vs
     Proposal-GRU −25.694 (−8.04%; GRU wins). Knockout sanity passed.
     Architecture superiority not claimed.
   - **Play competence (2026-08-20, Spark CPU):** thought-mediated campaign and
     Variant E have **no persisted maze-chase checkpoints**. Re-eval of the
     turn-weighted distill champion (`e58f323f…`) vs planner:
     protocol `distill-direct-5-9-240` neural **38 pellets / 43 collisions / 0
     clears** vs planner **284 / 1 / 2 clears**; protocol
     `e-heldout-direct-2001-2020-120` neural **8.55 mean pellets / 265 collisions
     / 0 clears** vs planner **84.5 / 21 / 0**. Still ≪ planner. GB10 left
     alone. Record: [brain/docs/runs/2026-08-20-play-competence-closed-loop.md](brain/docs/runs/2026-08-20-play-competence-closed-loop.md).
   - Phase 2.5 CPU diagnostics are in-tree: belief collapse / VoI, multi-horizon
     ambiguity arena, sequential probe resolution, analytic WAIT-vs-commit VoI,
     and a multiplicity-proof action aggregator. Play-safe tests cover them.
     Record: [brain/docs/runs/2026-08-20-phase25-belief-collapse-diagnostics.md](brain/docs/runs/2026-08-20-phase25-belief-collapse-diagnostics.md).
     Do not launch the full thought-mediated Spark campaign from this workstation;
     use `--cpu-smoke` only. Do not resume RCQ-v2.
   - 5-seed statistical replication battery ($E$ vs $F$ vs $G$) across 100 evaluation episodes per variant.
   - Calibrated counterfactual risk utility (targeting the sweet spot: $F$'s $\sim 7.8$ pellet drive with $E$'s $\sim 83$ safe catches).
   - Wall recovery latency reduction (reducing collision-recovery response from $23.1 \to \le 10$ ticks).
   - CPU-only play-safe test suite maintenance and manifest integrity.

2. **NEAR-TERM RESEARCH:**
   - Real-time $60\text{ Hz}$ closed-loop arcade benchmark (Pac-Man family, frame skip 1, latency $\text{p99} < 16.67\text{ ms}$).
   - Hardware capture-to-control latency harness (RTX 5070 non-gaming window).
   - Procedural skill ladder data scaling (junctions, occlusion memory, key/door sequencing).
   - Matched-baseline comparisons at equal FLOPs, parameters, and latency.

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
4. [brain/docs/RCQ_V2_PROTOCOL.md](brain/docs/RCQ_V2_PROTOCOL.md) — frozen
   thresholds and stopping rules.
5. [brain/docs/DGX_SPARK_TRAINING.md](brain/docs/DGX_SPARK_TRAINING.md) — Spark
   host, SSH, and container operations.
6. [brain/docs/BASELINE_PROTOCOL.md](brain/docs/BASELINE_PROTOCOL.md) — later
   architecture-comparison rules, not the current qualification.
7. [brain/docs/PLAY_SAFE.md](brain/docs/PLAY_SAFE.md) — local safety policy.
8. [brain/PLAN.md](brain/PLAN.md) — long-term architecture blueprint and general-agent roadmap (§43).
9. [brain/docs/ROADMAP_TO_PACMAN.md](brain/docs/ROADMAP_TO_PACMAN.md) — the path
   from the failed RCQ-v2 to a 60 Hz Pac-Man-like arcade proof and beyond (§9).

Historical Stage A DGX records are under `brain/docs/runs/`. They document
valid failures. Do not treat them as a green light to scale.

## GitHub

The private source of record is `https://github.com/defnotean/Pseudo-Brain`.
Implementation and documentation are kept in the same commit stream so the
remote copy matches the frozen local tree.
