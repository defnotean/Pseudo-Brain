# V2.1i seed-42 V2 development qualification — FAIL

**Date:** 2026-08-24  
**Artifact:** `brain/runs/v21i-development/2026-08-24-seed42-v2.json`  
**Artifact SHA-256:** `e3d45586f04420c26649f217ec29c02401878a32c8e5fed670e3467bf16ad7af`  
**Verdict:** 39 of 43 frozen development checks passed; V2 remains failed.

The immutable artifact failed exactly four checks:

| Check | Observation | Required |
| --- | ---: | ---: |
| `dev_every_action_ece` | idle ECE `0.07506943200375342` | every action at most `0.075` |
| `dev_factual_brier_beats_train_prior` | model `0.09819492985147087`; prior `0.0` | model strictly below prior |
| `factual_bootstrap_bce_lower_bound` | `-0.3654399898297399` | strictly above `0.0` |
| `factual_bootstrap_brier_lower_bound` | `-0.10675083215847629` | strictly above `0.0` |

The ECE miss is `0.00006943200375342`; it is not waived or rounded into a
pass.  The three factual failures expose a separate protocol defect.  The V2
runner used the scripted teacher as the factual behavior policy with zero
interventions.  On TRAIN-FIT, the action-conditioned factual prior was exactly
`idle=1.0` and `W/A/S/D=0.0`: idle contained `74/74` hazards, while every
movement action contained zero.  The same teacher-selection relationship was
perfect on DEV, giving the frozen factual prior Brier `0.0`, BCE approximately
`1e-12`, and ROC/PR-AUC `1.0`.  This is not TRAIN/DEV label leakage; it is a
behavior-policy selection confound that makes the factual comparison
unwinnable and non-predictive for a learned policy's live action distribution.

The corrected development data contract pins
`balanced_intervention_v1` at rate `0.5` for every partition.  The exact
post-burn coverage audit was:

| Partition | Rows | Teacher disagreements | Factual hazards | Factual safe/hazard by idle/W/A/S/D |
| --- | ---: | ---: | ---: | --- |
| TRAIN-FIT | 1,536 | 641 (`0.4173177083`) | 196 | `147/73`, `336/28`, `233/24`, `298/39`, `326/32` |
| TRAIN-CAL | 768 | 298 (`0.3880208333`) | 93 | `66/41`, `164/7`, `125/12`, `146/11`, `174/22` |
| DEV | 768 | 327 (`0.42578125`) | 103 | `69/38`, `160/10`, `123/21`, `147/15`, `166/19` |

Exhaustive branch safe/hazard counts were:

- TRAIN-FIT: `1273/263`, `1315/221`, `1301/235`, `1244/292`, `1199/337`;
- TRAIN-CAL: `629/139`, `635/133`, `670/98`, `611/157`, `592/176`;
- DEV: `603/165`, `622/146`, `649/119`, `599/169`, `588/180`.

No calibrated child checkpoint was published: the artifact records
`published=false` with reason `development gate failed`.  The valid
uncalibrated parent remains
`2026-08-24-seed42-v2.json.uncalibrated.pt`, checkpoint SHA-256
`422b12230b14a51333ff1f9f4202c65bcdab676c6d72dbe0a19e6a85bfdb7ba0`,
and state SHA-256
`ce8ef0fc4e9de3c505644792108697fafd3d4afae921d7679b17b9b7683e4544`.
Its validity does not convert V2 into a passing candidate or authorize a child.

The V2 artifact and 39/43 result remain immutable failures.  The correction
requires a new development protocol/version with the existing thresholds and
bias-only calibration unchanged.  CPU-QUAL and TEST remain unopened, and this
failure authorizes no DGX run or live-play claim.
