# Gate 6 FAIL diagnosis (2026-08-20)

**Probe ID**: `why-gru-wins-despite-knockout-v1`  
**Status**: diagnosis, **not** a Gate 6 rerun  
**Parent FAIL**: `dgx-gate6-matched-gru-v1` (`a615223`)

Frozen numbers (do not silently edit):

- Gate 6 FAIL: thought-mediated IQM **−27.758** vs Proposal-GRU **−25.694** (**−8.04%**).
- Knockout sanity **1343%** overall. Seed 45 Family-B knockout **0%** (−21 → −21).
- Gate 2 FAIL: decision_critical K=1/4/8/16/32 = **98 / 98 / 32 / −98 / −98**.
- Gate 5 FAIL: `C_register_swap` **0.00%** degradation.
- Gate 4 PASS: permutation **0.00%**.

This note names three hypotheses, records the cheapest CPU measurements that
distinguish them, and says what a **future named** thought-mediated design
should change. It does not claim architecture superiority. It does not resume
RCQ-v2. It does not kill Irene sglang on GB10.

## Hypotheses (preregistered before the cheap parse)

**H1 — Seed-45 collapsed policy.**  
On seed 45 the thought field is inert: zeroing thoughts does not change Family
B return. GRU IQM **−19.333** vs PB **−26.917**. This seed can decide the
pooled mean even when the other four seeds are close or tied.

**H2 — Extra K duplicates / hurts ranking.**  
Gate 2 is anti-monotonic. K=1 and K=4 score 98; K=8 drops to 32; K=16 and K=32
are −98. Branch targets are 10 futures. Hungarian matching at K=32 leaves ~22
unmatched slots still voting. Softmax-over-slots then lets leftover clones
swamp the matched specialist.

**H3 — Scramble unused because slots are already interchangeable / ranking is
broken.**  
`C_register_swap`, `D_stale_thoughts`, and `E_donor_thoughts` all sit at IQM
**−32.0**, identical to `A_normal`. Whole-slot permutation is 0.00% (Gate 4
PASS, as designed). If register 0 ≈ register 1 and every slot proposes the
same action, a binding break cannot hurt. Zero-knockout still collapses Family
B to **−486.0**, so thoughts are on the path as a *blob*, not as ranked
distinct hypotheses.

## Cheap measurements (JSON + untrained actuator, no 5×300 train)

Sources: `docs/phase_closure/gate6_matched_gru_results.json` and
`thought_mediated_campaign_results.json`. Parser:
`irene_brain.evaluation.gate6_fail_diagnosis`.

### H1 from existing Gate 6 JSON

| Seed | PB IQM | GRU IQM | Family-B knockout | Family E PB | Family E GRU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | −31.000 | −30.417 | −32.0 → −526.0 | −506.0 | −536.0 |
| 43 | −28.417 | −31.417 | −20.5 → −477.0 | −46.5 | −562.0 |
| 44 | −27.500 | −27.500 | −27.5 → −681.0 | −685.0 | −685.0 |
| **45** | **−26.917** | **−19.333** | **−21.0 → −21.0 (0%)** | **−796.0** | **−41.5** |
| 46 | −30.833 | −30.833 | −27.0 → −181.0 | −535.0 | −535.0 |

Seeds 44 and 46 are exact PB/GRU ties across families. Seed 43 is the only PB
win. Seed 45 is the GRU blowout: Family E **−796** vs **−41.5**, and Family B
thoughts do nothing. Leave-one-out pooled IQM **without seed 45** (same JSON,
no new train): PB **−28.040** vs GRU **−28.480** (**+1.54%**). H1 is the Gate 6
decider. The frozen five-seed FAIL remains **−8.04%**; this leave-one-out is
commentary, not a rewritten Gate 6 score.

Action histogram for seed 45 is **not** in the FAIL JSON. Re-training seed 45
for 300 steps would be a partial Gate 6 rerun and is **out of scope**. The
follow-up named probe records Family-B WASD histograms on every seed,
including 45.

### H2 from existing campaign JSON

`multi_seed_scaling_curve` decision_critical:

- K=1: **98**
- K=4: **98**
- K=8: **32**
- K=16: **−98**
- K=32: **−98**

Commentary: extra slots past 8 invert competence. The matched Gate 6 pair used
**K=32**, the worst point on that curve. Stochastic expected utility is also
best at K=1 / Proposal-GRU (−37.74) vs K=32 (−38.71).

### H3 from existing campaign JSON

| Intervention | IQM | decision_critical |
| :--- | ---: | ---: |
| A_normal | −32.0 | 98 |
| B_permute | −31.5 | 98 |
| C_register_swap | −32.0 | 98 |
| D_stale_thoughts | −32.0 | 98 |
| E_donor_thoughts | −32.0 | 98 |
| H_zero_knockout | −486.0 | −100 |

Scramble/stale/donor are unused. Only zeroing the whole field matters.

### Untrained actuator (local CUDA-hidden)

Untrained `ConsequenceThoughtActuator` (`CUDA_VISIBLE_DEVICES=-1`, 2026-08-20
local): permutation argmax change **0.00%** (Gate 4 invariant still holds);
register-swap with distinct registers **50.00%** argmax change (scramble can
matter when r0 ≠ r1); multiplicity Q(a) unit test prefers the specialist over
eight close-utility clones.

## What to change in a future named design

Do **not** retire parallel thoughts on this evidence. Knockout collapse on
four of five seeds shows mediation can be causal. Do **not** rerun K=32
300-step Gate 6.

Next named design (`ranked-k8-unmatched-suppress-v1`):

1. **Cap K=8** (resource-matched width to Proposal-GRU ±6%). Gate 2 says
   K>8 anti-scales with the old vote aggregator.
2. **Unmatched-slot suppression + ranking**: leftover Hungarian slots get
   branch_prob → 0 and utility below the matched floor. Live aggregator uses
   multiplicity-proof Q(a) so duplicate RIGHT clones cannot outvote a better
   LEFT.
3. **Collapse detector**: penalize near-zero active fraction (seed-45 inert
   thoughts) and register-0/1 cosine collapse so `C_register_swap` can matter.
4. Keep the thought-mediated invariant: no belief→action bypass, shared
   proposal head, bounded reflex.

A second distinct design, only if this one still loses to GRU: K=4 (Gate 2
peak) with the same losses — not a retune of the failed K=32 probe.

Smoke vs GRU (seeds 43+45, 60 steps) is allowed before a full 5×300. Full
Gate 6 envelope only if the smoke shows PB ahead. New run id. Spark CPU if
GB10 is on Irene sglang.

## Local CPU verification

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Resolve-Path .\brain\src).Path
C:\Users\Demon\AppData\Local\Programs\Python\Python311\python.exe -m irene_brain.evaluation.gate6_fail_diagnosis
```

2026-08-20 local: diagnosis exit 0. Leave-one-out without seed 45 is **+1.54%**. Play-safe CPU suite exit 0 (new modules `test_gate6_fail_diagnosis.py`, `test_ranked_k8_unmatched_suppress.py`).
