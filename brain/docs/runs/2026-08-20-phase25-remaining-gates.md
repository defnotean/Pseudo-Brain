# Phase 2.5 remaining gates (2026-08-20)

Status: **scored from existing artifacts**. No GPU train. No Gate 6 retry.
Local CPU-only parse plus an untrained actuator permutation probe
(`CUDA_VISIBLE_DEVICES=-1`, one thread). Architecture superiority is
**not** claimed.

Sources:

- Preregistration: [`docs/preregistrations/2026-08-20-thought-mediated-parallel-cognition.md`](../preregistrations/2026-08-20-thought-mediated-parallel-cognition.md) (Gates 1–8)
- Campaign JSON: [`docs/phase_closure/thought_mediated_campaign_results.json`](../phase_closure/thought_mediated_campaign_results.json) (`cuda:0`, 2026-08-20 06:35:40 UTC, seeds 42–46)
- Gate 6 named probe: [`docs/phase_closure/gate6_matched_gru_results.json`](../phase_closure/gate6_matched_gru_results.json) and [`2026-08-20-gate6-matched-proposal-gru.md`](2026-08-20-gate6-matched-proposal-gru.md) (`a615223`)
- Original thesis (retired, not reused for Gates 3/7): [`docs/phase_closure/PHASE_2_ORIGINAL_THESIS_FALSIFICATION_REPORT.md`](../phase_closure/PHASE_2_ORIGINAL_THESIS_FALSIFICATION_REPORT.md)
- Play competence: [`2026-08-20-play-competence-closed-loop.md`](2026-08-20-play-competence-closed-loop.md) (no thought-mediated maze-chase `.pt`)

Parser: `irene_brain.evaluation.phase25_remaining_gates`.

## Gate table

| Gate | Name | Verdict | Measured |
| :--- | :--- | :---: | :--- |
| Gate 1 | Thought mediation | **UNKNOWN** | Family B IQM **-32.0** -> **-486.0** (**1418.75%** collapse); Gate 6 Family-B knockout sanity **1343%**. Families C–E and reflex: **not measured**. |
| Gate 2 | Monotonic capacity scaling | **FAIL** | `multi_seed_scaling_curve` decision_critical_score K=1/4/8/16/32 = **98.0 / 98.0 / 32.0 / -98.0 / -98.0** (anti-monotonic). Stochastic EU: -37.74 / -38.71 / -39.68 / -38.71 / -38.71. |
| Gate 3 | Causal slot usefulness | **UNKNOWN** | **not measured**. No per-slot knockout table in the thought-mediated campaign JSON. Need ≥8/32 slots with ≥5% drop. |
| Gate 4 | Permutation equivariance | **PASS** | Campaign `B_permute` vs `A_normal` decision_critical **98.0 vs 98.0 (0.00%)**; IQM -32.0 vs -31.5. CPU actuator argmax change **0.00%**, action_dist L1 **0.0000%**. Gate wants ≤1.0% action change. |
| Gate 5 | Cognitive scrambling | **FAIL** | `C_register_swap` IQM **-32.0** vs `A_normal` **-32.0** (**0.00%** degradation). Need ≥15%. |
| Gate 6 | Baseline superiority | **FAIL** | PB IQM **−27.758** vs GRU **−25.694**, relative **−8.04%** (need ≥+10%). Knockout sanity passed (1343%). Do not claim superiority. |
| Gate 7 | Breadth | **UNKNOWN** | **not measured**. No per-family 95% bootstrap CI in the thought-mediated campaign JSON. Need positive CI on ≥4/5 families. |
| Gate 8 | DGX Spark real-time deadline | **UNKNOWN** | **not measured**. Campaign JSON has no kernel p99 / loop p95. Phase 1 GB10 numbers are a different model. Needs GB10. |

## Literal notes

### Gate 1 (UNKNOWN, Family B proxy holds)

Campaign closed-loop interventions ran on Family B only
(`FAMILY_B_PURSUIT_EVASION`). `H_zero_knockout` IQM **-486.0** vs
`A_normal` **-32.0** is a **1418.75%** collapse (online top-1 95.67% ->
1.33%; decision-critical 98.0 -> -100.0). The Gate 6 probe repeats the
Family B knockout at **1343%** mean degradation (sanity passed). The
registered bar is Families **B–E** ≥30% **and** reflex tasks stable.
Families C–E and reflex (Family A) knockout/stability are **not
measured** in the campaign JSON, so the full gate stays UNKNOWN.

### Gate 2 (FAIL)

Need a statistically significant K=32 > 16 > 8 > 4 > 1 trend. The
committed `multi_seed_scaling_curve` is the opposite: 98, 98, 32, -98,
-98. Larger K is worse. The stochastic-occluded table is also
non-monotonic (best expected utility is K=1 / Proposal-GRU at -37.74).

### Gate 3 (UNKNOWN)

The thought-mediated campaign JSON has no `useful_slots` / per-slot
knockout table. Design 1/2 reported **0/32** useful slots; that is the
retired unmediated thesis and is not reused here.

### Gate 4 (PASS)

The campaign did not store a dedicated action-output delta field.
`B_permute` vs `A_normal` decision-critical score is **98.0 vs 98.0
(0.00%)**. IQM moved from -32.0 to -31.5 (slightly better, not an
action-change fail). A cheap untrained `ConsequenceThoughtActuator`
whole-slot permutation on CPU measured **0.00%** argmax change and
**0.0000%** action_dist L1. 0.00% equivariance is PASS (≤1.0%).

### Gate 5 (FAIL)

`C_register_swap` (register-binding break) is identical to `A_normal`
on IQM (-32.0), online top-1 (95.67%), and decision-critical (98.0).
0.00% degradation vs a ≥15% bar.

### Gate 6 (FAIL)

Named probe `dgx-gate6-matched-gru-v1` (`a615223`): thought-mediated
K=32 W=60 C=3 IQM **−27.758** vs Proposal-GRU H=180 **−25.694**.
Relative IQM advantage **−8.04%**. Need ≥+10% and GRU must not tie or
win. **FAIL.** Knockout sanity passed (1343%); thoughts are on the
Family B action path. That does not rescue Gate 6. Do not claim
architecture superiority. Do **not** silently rerun the same 300-step
CPU probe.

### Gate 7 (UNKNOWN)

No per-family 95% bootstrap CI exists in
`thought_mediated_campaign_results.json`. Design 2 family scores belong
to the retired thesis.

### Gate 8 (UNKNOWN)

No thought-mediated kernel p99 or end-to-end p95 on GB10 is in the
campaign JSON. Measuring this needs the accelerator. This analysis did
not start a GPU job.

## Play competence (context, not a prereg gate)

Phase 2.5 has **no persisted thought-mediated maze-chase checkpoint**.
The live play artifact is still the Aug 18 turn-weighted exclusive-CE
champion. Neural 38 pellets / 43 collisions vs planner 284 / 1 on
seeds 5+9. Not evidence for Gates 1–8.

## Next work (not a Gate 6 retry)

The make-or-break comparison already failed while mediation knockout on
Family B is huge (1418.75% campaign / 1343% Gate 6). Next work is a
**newly named** design or diagnosis of *why Proposal-GRU still wins
IQM while zeroing thoughts collapses Family B*, not another
`dgx-gate6-matched-gru-v1` 300-step CPU probe.

Named diagnosis: [`2026-08-20-gate6-fail-diagnosis.md`](2026-08-20-gate6-fail-diagnosis.md)
(`why-gru-wins-despite-knockout-v1`). Leave-one-out without seed 45: PB IQM
**−28.040** vs GRU **−28.480** (**+1.54%**). Extra K anti-scales. Scramble is
unused. Follow-up named design **`ranked-k8-unmatched-suppress-v1`**
(prereg: [`2026-08-20-ranked-k8-unmatched-suppress.md`](../preregistrations/2026-08-20-ranked-k8-unmatched-suppress.md)):
cap K=8, unmatched-slot suppression, multiplicity Q(a), collapse guard.
Smoke (seeds 43+45, 60 steps) before a full Gate 6 envelope.

Do not start a GB10 train from this document. Do not resume RCQ-v2.

## Local CPU verification

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Resolve-Path .\brain\src).Path
C:\Users\Demon\AppData\Local\Programs\Python\Python311\python.exe -m unittest discover -s brain\tests -t brain -p test_phase25_remaining_gates.py -v
C:\Users\Demon\AppData\Local\Programs\Python\Python311\python.exe -m irene_brain.evaluation.phase25_remaining_gates
```

2026-08-20 local run: **2 tests, 0.006 s, exit 0**. Actuator permutation
argmax change 0.00%.
