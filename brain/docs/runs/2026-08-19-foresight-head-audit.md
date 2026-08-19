# Counterfactual Foresight Head Multi-Seed Audit & Convergence Analysis

**Date**: August 19, 2026  
**Audited Models**: Multi-Seed Battery (Seeds 42, 43, 44, 45, 46)  
**Evaluation Protocol**: 20 Held-Out Evaluation Worlds (Seeds 2001–2020) $\times$ 25 Steps $\times$ 5 Candidate Actions = **2,500 Counterfactual Predictions per Seed**  
**Harness**: `scripts/audit_foresight_convergence.py`  
**Raw Data**: `docs/runs/2026-08-19-foresight-head-audit.json`  

---

## 1. Executive Summary & Core Discovery

To resolve the seed divergence observed in the Grounded Replication Battery (where Seed 44 achieved a 90% catch reduction while other seeds showed flat or minor changes), we performed a deep diagnostic audit directly on the **Counterfactual Foresight Head** itself across all 5 seeds.

### Key Finding: Severe Base-Rate Imbalance & Prior Convergence
Across 2,500 evaluated steps per seed, ground-truth ghost collisions occurred in only **3.4%** of transitions (86 to 93 positive hazards vs 2,407 to 2,414 negative hazards). 
Under unweighted binary cross-entropy, the optimization objective is overwhelmingly dominated ($28:1$) by the negative class. Consequently:
- Networks converge toward predicting the global prior ($\hat{p}_{\text{haz}} \approx 0.11\text{--}0.34$) regardless of the specific spatial ghost distance.
- At default threshold $\tau = 0.50$, recall is $0.0\%$ because no prediction crosses $0.50$.
- With threshold calibration ($\tau = 0.05\text{--}0.30$), the head surfaces weak directional gradients, but the separation margin between positive and negative hazard probabilities remains narrow ($\Delta \approx -0.005 \text{ to } +0.003$).

---

## 2. Multi-Seed Foresight Diagnostic Matrix

| Metric | Seed 42 | Seed 43 | Seed 44 (Best Policy) | Seed 45 | Seed 46 | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brier Calibration Score** (lower is better) | 0.1269 | 0.0815 | **0.0731** | **0.0484** | 0.0884 | **$0.0837 \pm 0.028$** |
| **Mean Predicted Positive Hazard ($\bar{p}_{y=1}$)** | 0.3412 | 0.2537 | 0.2246 | 0.1126 | 0.2553 | $0.2375 \pm 0.082$ |
| **Mean Predicted Negative Hazard ($\bar{p}_{y=0}$)** | 0.3395 | 0.2536 | 0.2294 | 0.1130 | 0.2526 | $0.2376 \pm 0.081$ |
| **Discrimination Margin ($\Delta = \bar{p}_+ - \bar{p}_-$)** | +0.0017 | +0.0001 | -0.0047 | -0.0004 | **+0.0027** | $-0.0001 \pm 0.003$ |
| **Optimal Hazard Threshold ($\tau^*$)** | 0.30 | 0.05 | 0.05 | 0.05 | 0.25 | $0.14 \pm 0.12$ |
| **Optimal $F_1$ Score** | 0.0686 | 0.0665 | 0.0717 | 0.0880 | **0.1124** | $0.0814 \pm 0.019$ |
| **Safest Predicted Action Accuracy** | **98.0%** | 96.0% | 95.6% | 94.0% | 96.2% | **$96.0\% \pm 1.4\%$** |
| **Escape Margin MAE** (grid units) | 4.68 | 4.61 | 5.32 | **3.88** | 4.32 | **$4.56 \pm 0.53$** |

---

## 3. Horizon-Wise Calibration & Multi-Step Error

| Prediction Horizon | Horizon 1 (Tick $+1$) | Horizon 2 (Tick $+3$) | Horizon 3 (Tick $+5$) |
| :--- | :---: | :---: | :---: |
| **Seed 42 Brier / Acc** | 0.1243 / 96.6% | 0.1270 / 96.6% | 0.1294 / 96.6% |
| **Seed 43 Brier / Acc** | 0.0792 / 96.6% | 0.0815 / 96.6% | 0.0838 / 96.6% |
| **Seed 44 Brier / Acc** | 0.0681 / 96.3% | 0.0734 / 96.3% | 0.0778 / 96.3% |
| **Seed 45 Brier / Acc** | **0.0460** / 96.7% | **0.0485** / 96.7% | **0.0507** / 96.7% |
| **Seed 46 Brier / Acc** | 0.0852 / 96.7% | 0.0886 / 96.7% | 0.0914 / 96.7% |

Across all seeds, prediction error monotonically increases with horizon depth ($H_1 \to H_3$), confirming that immediate collision danger is more easily calibrated than distant multi-step convergence.

---

## 4. Architectural Resolution: Balanced Hazard Supervision

To turn seed-dependent survival (874, 822, 82, 874, 822) into consistent high-survival policies across all training seeds, two architectural enhancements are implemented:

1. **Positive Class Loss Weighting (`pos_weight = 12.0`) in `CognitiveAuxiliaryLoss`**:
   $$\mathcal{L}_{\text{hazard}} = - \frac{1}{B} \sum_{i=1}^B \left[ w_{\text{pos}} y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right]$$
   where $w_{\text{pos}} = 12.0$ offsets the $96.6\%$ negative class imbalance, compelling gradient updates during dangerous states.

2. **Epistemic Calibration Gating Rule**:
   $$\text{Grounded Planning Gate} = \begin{cases} \text{Active}, & \text{if } \text{Brier} < 0.05 \land \Delta > 0.10 \land \text{RankAcc} > 0.90 \\ \text{Fallback to Direct Actuator}, & \text{otherwise} \end{cases}$$
   *Rule: "Don't trust your own imagination until it is calibrated."*

---

## 5. Artifacts and Lineage

- **Audit Script**: `brain/scripts/audit_foresight_convergence.py`
- **Result Log**: `brain/docs/runs/2026-08-19-foresight-head-audit.json`
- **Loss Module**: `brain/src/irene_brain/training/cognitive_losses.py`
- **Phase 0 Status**: `COMPLETE & LOCKED ✅` (`brain/docs/phase_closure/PHASE_0_CLOSURE_REPORT.md`)
- **Phase 1 Status**: `OPEN` (`brain/docs/phase_closure/PHASE_1_CLOSURE_REPORT.md`)
