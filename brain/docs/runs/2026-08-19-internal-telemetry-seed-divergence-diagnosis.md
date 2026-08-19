# Internal Telemetry Diagnosis: Good Seed 43 vs Reckless Seed 46
**Date**: 2026-08-19  
**Models**: `Variant E` trained on Seed 43 ("Good Model") vs Seed 46 ("Reckless Model")  
**Evaluation**: 10 Held-Out Validation Worlds (Seeds 100-109), 100 Ticks per World  
**Execution**: CPU-only, single-threaded, CUDA-hidden  

---

## 1. Executive Summary

To answer the central scientific question:
> *"Why does the exact same Pseudo-Brain architecture sometimes train into a careful brain and sometimes into a reckless brain?"*

We captured synchronized internal neural telemetry, spatial geometry, future forecasts, slot representations, and action distributions during live closed-loop play across held-out test environments.

---

## 2. Telemetry Comparison Table

| Metric | Good (Seed 43) | Reckless (Seed 46) | Diagnostic Finding |
| :--- | :---: | :---: | :--- |
| **Mean Pellets Collected** | 3.10 | 4.50 | Reckless model is greedier |
| **Total Ghost Catches** | **34** | **415** | **12x higher catch rate in Seed 46** |
| **Danger Prediction Recall (%)** | **100.0%** | **100.0%** | Both models see danger approaching |
| **Danger Prediction Precision (%)** | 17.5% | 61.2% | Seed 43 has wider defensive safety buffer |
| **Hazard Compute Cycles ($C_{\text{hazard}}$)** | 3.00 | 3.00 | Full recurrent depth utilized |
| **Surprise Gate $\alpha$ (Hazard)** | **0.208** | **0.102** | Seed 46 thought updates are 50% suppressed |
| **Thoughtlet Effective Rank (1-4)** | 3.87 | 3.54 | Dimensional capacity across 4 slots |
| **Thoughtlet Pairwise Cosine Sim** | **0.088** | **0.579** | **Seed 46 suffers 6.6x slot collapse / redundancy** |
| **Active Evasion Rate (%)** | **26.3%** | **4.2%** | Seed 43 actively navigates away from ghost |
| **Suicide / Toward Ghost Rate (%)** | **5.7%** | **88.4%** | **Seed 46 moves INTO the ghost in 88.4% of steps!** |
| **Wall Collision Rate (%)** | 94.4% | 70.1% | Seed 43 hugs corners to escape corridors |

---

## 3. Four Hypotheses Analysis

### Possibility 1: Does Seed 46 literally not see danger coming?
- **Result**: **FALSE / RULED OUT**.
- **Evidence**: Both Seed 43 and Seed 46 achieve **100.0% danger prediction recall**. The world model in Seed 46 correctly forecasts imminent ghost proximity.

### Possibility 2: Does Seed 46 predict danger but ignore or charge into it?
- **Result**: **CONFIRMED (PRIMARY ROOT CAUSE)**.
- **Evidence**: Under danger ($\text{dist} \le 2.5$), Seed 46 takes actions directed *toward* the incoming ghost in **88.4%** of steps (compared to only **5.7%** in Seed 43). Seed 46 has learned a reckless greedy policy that treats the ghost as an obstacle to bypass rather than a fatal hazard.

### Possibility 3: Do reckless models fail to think longer during danger?
- **Result**: **SECONDARY CONTRIBUTOR**.
- **Evidence**: Both models allocate the maximum 3 cycles ($C_{\text{hazard}} = 3.00$), but Seed 46's surprise gate $\alpha$ is suppressed ($0.102$ vs $0.208$), preventing sensory surprise from updating working thoughts.

### Possibility 4: Do the thoughtlets collapse in bad seeds?
- **Result**: **CONFIRMED (ARCHITECTURAL BOTTLENECK)**.
- **Evidence**: In Good Seed 43, the 4 thoughtlet slots are near-orthogonal (pairwise cosine similarity = **0.088**), allowing specialized sub-functions (e.g. navigation slot vs evasion slot). In Reckless Seed 46, pairwise cosine similarity surges to **0.579** (6.6x higher!), causing slot collapse where all 4 thoughtlets redundantly track pellets and neglect evasion dynamics.

---

## 4. Prescribed Architectural Fixes for Robustness

To transform training robustness such that *all* random seeds converge to safe champion policies:
1. **Thoughtlet Orthogonalization Penalty**: Add $\mathcal{L}_{\text{ortho}} = \lambda \sum_{i \neq j} (\cos(z_i, z_j))^2$ to prevent slot collapse during initial warmup.
2. **Directional Hazard Repulsion Loss**: Supervise movement vector projection during danger steps to strictly penalize velocity vectors pointing along the ghost approach vector.
3. **Surprise Gate Floor**: Ensure $\alpha \ge 0.20$ during detected danger horizons.
