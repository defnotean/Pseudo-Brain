# Phase 2 Matched-Compute Baseline Campaign Report — DGX Spark

## 1. Executive Summary & Verdict

```text
================================================================================
PHASE 2 — DESIGN 1 STATUS: DOES NOT PASS ❌
(Attempt 1 of 2 Allowed by Registered Falsification Protocol)
================================================================================
```

The initial matched-compute comparison was executed natively on the **NVIDIA DGX Spark** (`cuda:0`, NVIDIA GB10 GPU, CUDA 13.0, PyTorch 2.13.0+cu130) across **5 independent training/evaluation seeds (42, 43, 44, 45, 46)** and all **5 procedural task families**.

**Primary Result**: In its unspecialized reference state, **Pseudo-Brain ranks 8th (last)** with an overall IQM return of **-32.63**, compared to the leading conventional baselines (**GRU: -24.26**, **World-Model Actor: -24.26**, **SSM: -26.66**, **Wider Monolith: -27.71**).

---

## 2. Model Performance Matrix (Measured on NVIDIA DGX Spark GB10 CUDA)

| Rank | Model Variant | Parameters | Latency p95 (ms) | Family A (Multi-Obj) | Family B (Pursuit) | Family C (Junctions) | Family D (Occlusion) | Family E (Dynamics) | Total IQM Return |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Baseline 2 — GRU Recurrent** | 125,419 | $3.76\text{ ms}$ | +1.08 | **-61.24** | **-63.80** | **-96.16** | -122.52 | **-24.26** |
| 1 | **Baseline 8 — Latent World-Model Actor** | 127,779 | $3.78\text{ ms}$ | +1.08 | **-61.24** | **-63.80** | **-96.16** | -122.52 | **-24.26** |
| 3 | **Baseline 4 — State-Space SSM (S4)** | 126,119 | $3.96\text{ ms}$ | +0.68 | -39.84 | -43.96 | -181.16 | -192.32 | **-26.66** |
| 4 | **Baseline 5 — Wider Monolith (Capacity-Matched)** | 128,491 | $3.84\text{ ms}$ | **+3.60** | -129.68 | -138.16 | -127.84 | **-40.60** | **-27.71** |
| 5 | **Baseline 3 — Recurrent Transformer** | 126,859 | **$3.05\text{ ms}$** | +1.60 | -38.48 | -57.20 | -137.44 | -268.64 | **-28.94** |
| 6 | **Baseline 1 — Reactive (No Memory)** | 127,019 | $4.83\text{ ms}$ | +2.60 | -256.84 | -263.80 | -208.60 | -117.88 | **-29.34** |
| 7 | **Baseline 6 — Deeper Serial Model** | 127,019 | $4.91\text{ ms}$ | +2.60 | -207.08 | -186.88 | -176.16 | -195.48 | **-32.39** |
| 8 | **Pseudo-Brain Reference ($K=32, C=3$)** | 127,019 | **$4.79\text{ ms}$** | **+2.60** | -206.68 | -182.52 | -176.20 | -197.88 | **-32.63** |
| - | **Baseline 7 — Fixed Multi-Horizon** | 127,019 | $4.73\text{ ms}$ | +2.56 | -191.20 | -223.24 | -176.88 | -190.64 | -31.35 |

---

## 3. Scientific Inferences & Diagnostic Reality Check

### **A. Pseudo-Brain and Deeper Serial Baseline Behave Almost Identically [MEASURED]**
Comparing Pseudo-Brain directly to Baseline 6 (Deeper Serial):
- **Family A**: $+2.60$ vs $+2.60$
- **Family B**: $-206.68$ vs $-207.08$
- **Family C**: $-182.52$ vs $-186.88$
- **Family D**: $-176.20$ vs $-176.16$
- **Family E**: $-197.88$ vs $-195.48$
- **Total IQM**: $\mathbf{-32.63}$ vs $\mathbf{-32.39}$

**Inference**: Spending equivalent computation across 32 parallel thoughtlets currently provides no distinguishable advantage over executing equivalent serial computation.

### **B. Zero Causal Slot Specialization (0 / 32 Slots) [MEASURED]**
- The systematic single-slot knockout test across all 32 slots resulted in **0 / 32 causally useful slots** (0.0% degradation under ablation).
- The controller currently does not depend on any individual slot to execute its decisions; the 32 channels act as a diffuse, unspecialized mixture that the model can ignore without penalty.

### **C. Task Family Performance Realities [MEASURED]**
- **Family A (Multi-Object Tracking)**: Pseudo-Brain is competitive ($+2.60$).
- **Family B (Pursuit) & Family C (Junctions)**: Monolithic GRU and World Model Actor drastically outperform Pseudo-Brain (Pursuit: $-61.24$ vs $-206.68$; Junctions: $-63.80$ vs $-182.52$).
- **Family E (Changed Dynamics)**: Pseudo-Brain performs poorly ($-197.88$). The **Wider Monolith is the standout performer ($-40.60$)**, demonstrating that increased capacity in a continuous monolithic vector currently adapts better to control remapping than unspecialized thoughtlets.

---

## 4. Phase 2 — Design 2 Objective: Forcing Causal Parallel Thought Utilization

Under the registered falsification protocol, Phase 2 allows two substantially different thought-field designs before the hard-stop rule applies.

Design 2 will focus exclusively on **forcing causal parallel thought utilization** without adding architectural bloat:

1. **Slot Dropout During Training ($p_{\text{drop}} \in [0.10, 0.25]$)**:
   - Randomly masks subsets of thought slots during training passes to mechanically prevent diffuse codependence and force individual slots to carry standalone predictive utility.
2. **Unordered Multi-Future Branch Supervision**:
   - Supervise distinct thoughtlets on alternative future hypotheses (e.g. branch choices at junctions, threat trajectory vs open corridor).
3. **Causal Marginal Utility Loss**:
   - Penalize representations where slot removal has zero marginal effect on predicted future outcomes.
4. **Multi-Scenario On-Policy DAgger Distillation**:
   - Aggregate closed-loop rollouts across all 5 task families to close the performance gap against the GRU baselines.

### Target Metrics for Design 2:
- $\ge 8 / 32$ causally useful thoughtlets ($\ge 5\%$ degradation under single-slot knockout)
- Thought field shuffling causes $\ge 10\%$ relevant degradation
- Close the performance gap against GRU (target: IQM $\ge -20.0$, beating GRU's $-24.26$)
