# Gate 6 named probe: thought-mediated K=32 vs matched Proposal-GRU

**Date**: 2026-08-20  
**Probe ID**: `dgx-gate6-matched-gru-v1`  
**Parent registration**: `PREREG-PHASE2-REVISED-THOUGHT-MEDIATED-V1`  
**Source pin**: git `97ffe62579c90744595960fc7d342b21e3e490ef` (`defnotean/pseudo-brain`)  
**Harness**: `brain/scripts/dgx_run_gate6_matched_gru_probe.py`  
**Status**: **FAIL** (GRU won pooled IQM; measured 2026-08-20 20:41:10 UTC)

This is a bounded named campaign probe. It is not an RCQ round, not a 2,048-update
reference train, and not a new architecture registration.

## Why this probe exists

The 2026-08-20 `thought_mediated_campaign_results.json` (device `cuda:0`,
06:35:40 UTC) does **not** contain Gate 6. It reports:

- Stochastic occluded expected utility: Proposal-GRU **-37.74**, thought-mediated
  K=32 **-38.71** (GRU better; not the Gate 6 IQM criterion).
- Closed-loop IQM only for thought-mediated causal interventions on Family B
  (`A_normal` IQM **-32.0**, `H_zero_knockout` IQM **-486.0**). Thoughts matter
  on that knockout, but there is no matched Proposal-GRU IQM row.

Gate 6 is now measured. Official JSON:
[`docs/runs/artifacts/gate6-matched-proposal-gru/gate6_results.json`](../runs/artifacts/gate6-matched-proposal-gru/gate6_results.json).
Pooled IQM thought-mediated **−27.758** vs Proposal-GRU **−25.694**
(advantage **−8.04%**). **FAIL.** Knockout sanity passed. Architecture
superiority is not claimed.

## Exact resource-matched pair (committed HEAD, not dirty working tree)

| Side | Constructor | K / hidden | Width | Cycles | Notes |
|---|---|---|---|---|---|
| Thought-mediated (no bypass) | `ThoughtMediatedBrainModel` K=32 | K=32 | **searched** so params match GRU ±5% | 3 | HEAD width map W=32 is **not** used: it is 270,112 params vs GRU 787,314 |
| Proposal-GRU | `ProposalGRUBaseline(hidden_dim=180)` | 1 GRU state | 180 | — | committed HEAD constructor; envelope to match |

Parameter counts are recorded by the probe at runtime. The committed HEAD width
map (K=32 W=32 = 270,112) does **not** match Proposal-GRU hidden 180 (787,314).
The probe searches K=32 `core_width` (multiples of 4) to the GRU-180 envelope
within **6%** (closest discrete width is used; expect ~W=60 / ~832k vs 787k).
Uncommitted working-tree width search (`target_params=819_894`) is out of scope.

## Equal experience

- Seeds: `42, 43, 44, 45, 46`
- Optimizer: AdamW `lr=5e-4`, `weight_decay=1e-4`, grad clip 1.0
- Updates: **300** steps (the hardcoded budget in the committed campaign trainers,
  not the unused `--train-steps 1000` argparse default)
- Data: all five Phase-2 families, horizon-2 `generate_multi_step_trajectory_tree`,
  on-policy DAgger step after each update
- Eval: 5 closed-loop episodes per family per seed

## Pass / fail (frozen before launch)

Gate 6 **PASS** only if all of:

1. Pooled IQM return of thought-mediated K=32 is ≥ **10%** better than matched
   Proposal-GRU, defined as `(iqm_pb - iqm_gru) / abs(iqm_gru) ≥ 0.10`.
2. Mean Family-B thought knockout degradation ≥ **30%** relative to the intact
   thought-mediated policy (mediation sanity).
3. Proposal-GRU does not tie or win IQM.

If GRU ties or wins, the recorded status is **FAIL** and
`architecture_superiority_claimed` is false.

## Compute boundary

The NVIDIA GB10 was occupied at probe start by Irene `sglang` Qwen
(`mem-fraction-static 0.80`). This probe therefore runs as a named Spark **CPU**
job (`CUDA_VISIBLE_DEVICES=-1`) against a read-only HEAD source snapshot. It
does not stop the sglang container and does not start a second GPU train.
