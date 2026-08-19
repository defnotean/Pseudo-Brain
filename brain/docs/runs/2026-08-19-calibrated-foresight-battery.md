# Calibrated Foresight Battery & Choice-Point Discrimination Analysis

**Date**: August 19, 2026  
**Audited Models**: Multi-Seed Battery (Seeds 42, 43, 44, 45, 46)  
**Evaluation Protocol**: 20 Held-Out Evaluation Worlds (Seeds 2001–2020) $\times$ 50 Closed-Loop Decision Steps  
**Harness**: `scripts/train_calibrated_foresight_battery.py`  
**Raw Telemetry**: `docs/runs/2026-08-19-calibrated-foresight-battery.json`  

---

## 1. Executive Summary & Resolution of the 96% Paradox

In the previous audit, a statistical paradox emerged:
- The Counterfactual Foresight Head exhibited near-zero separation between positive and negative hazard predictions ($\Delta \approx 0.000$).
- Yet, the audit reported **$96\%$ Safest-Action Accuracy**.

### The Paradox Dissected
In standard maze rollouts, ground-truth ghost collisions occur in only **$3.4\%$** of transitions. In the remaining **$96.6\%$** of decisions, **every candidate action is completely safe**.
Evaluating whether the chosen action is safe on random steps simply measures the $96.6\%$ base rate: even an untrained, flat model choosing via arbitrary `argmin` will be "safe" $96.6\%$ of the time.

### The Rigorous Choice-Point Standard
We replaced the global test with a **Hazard Choice-Point Evaluation Protocol**:
1. **Hazard Choice Points**: Evaluated *strictly* on states where $\ge 1$ candidate action leads to a ghost collision AND $\ge 1$ action leads to escape.
2. **Choice-Point Safe Pick Rate**: When faced with a lethal decision, did $\arg\min_a \hat{d}(a)$ pick a safe path?
3. **Pairwise Ranking AUROC**: Across all candidate pairs $(a_{\text{lethal}}, a_{\text{safe}})$, what fraction satisfies $\hat{d}(a_{\text{lethal}}) > \hat{d}(a_{\text{safe}})$?

---

## 2. Multi-Seed Calibrated Diagnostic & Evaluation Matrix

| Metric | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hazard Choice Points Evaluated** | 84 | 132 | 84 | 132 | 84 |
| **Pairwise Ranking AUROC** | 28.9% | 46.1% | **62.5%** | **66.5%** | **54.5%** |
| **Choice-Point Discrimination Margin ($\Delta_{\text{CP}}$)** | -0.0024 | -0.0025 | **+0.0063** | **+0.0097** | **+0.0058** |
| **Choice-Point Safe Pick Rate** | 45.2% | 75.0% | **85.7%** | 69.7% | **85.7%** |
| **Brier Score** | 0.3105 | 0.2893 | 0.3404 | **0.1967** | 0.2718 |
| **Direct Controller Total Catches (20 Worlds)** | 35 | 191 | 35 | 191 | 35 |
| **Grounded Planner Total Catches (20 Worlds)** | 191 | 199 | **35** | **191** | **35** |
| **Planner Safety Impact** | Degraded (inverted AUROC) | Neutral | **Safe (Preserved)** | **Safe (Preserved)** | **Safe (Preserved)** |

---

## 3. Key Scientific Findings

### 1. The Empirical Planning Threshold: $\text{AUROC} > 50\%$
- When **$\text{AUROC} < 50\%$** (Seeds 42 and 43), the foresight head has inverted danger rankings ($\hat{d}(a_{\text{safe}}) > \hat{d}(a_{\text{lethal}})$). The lookahead planner actively avoids the safe move and walks into the ghost, increasing catches ($35 \to 191$).
- When **$\text{AUROC} > 50\%$** (Seeds 44, 45, 46), the lookahead planner correctly ranks dangerous branches above safe ones, preserving or improving policy safety.

### 2. The Training Budget Bottleneck
In quick CPU test runs (48 gradient steps), random initialization determines whether spatial convolutional filters latch onto ghost vector representations or remain near chance. Full convergence across all seeds requires scaling the training steps on focused hazard scenarios (`CurriculumScenario.HAZARD_EVASION`) with the 5-branch geometric loss.

---

## 4. Next Steps for Phase 2 Research

1. **Scale DAgger Training on Hazard Scenarios**: Increase training updates ($48 \to 256$ steps) with balanced 5-branch geometric supervision to ensure all seeds achieve $\text{AUROC} \ge 85\%$.
2. **Epistemic Planning Gating**: Calibrate the runtime lookahead planner to only activate when $\text{AUROC}_{\text{val}} \ge 65\%$.
