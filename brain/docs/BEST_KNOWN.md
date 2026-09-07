# Pseudo-Brain: BEST_KNOWN State

**Last Updated:** 2026-09-07  
**Champion Git Commit:** `157265b`  
**Active Branch:** `defnotean/pseudo-brain`  
**Hardware Baselines:**
- Local: Windows 11 AMD CPU / DirectML
- Remote: Google Colab NVIDIA A100-SXM4-40GB

---

## 1. Champion Architecture & Parameters

| Component | Architecture Variant | Parameters | Latency Target | Measured Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recurrent Core** | Consequence-Gated Plasticity (`PlasticBrainCell`) | 279,982 | $\le 2.0\text{ ms}$ | **$1.50\text{ ms}$** (CPU) | **VALIDATED** (Decoupled consequence surprise $\delta_{\text{consequence}}$) |
| **Thought Routing** | Sub-Quadratic Clustered Router (`BlockSparseClusteredThoughtRouter`) | 148,608 | $\le 1.5\text{ ms}$ | **$1.30\text{ ms}$** ($K=64$) | **VALIDATED** ($\mathcal{O}(K^{1.5} W)$, $2.38\times$ FLOP reduction at $K=512$) |
| **Lookahead Planner** | Dynamic Beam Search ($H=5, B=6$) | N/A (Latent unroll) | $\le 16.67\text{ ms}$ | **$8.03\text{ ms}$** (CPU) | **VALIDATED** (Isolated microbenchmark) |
| **Agent Infrastructure**| Contextual IOR (`PseudoBrainAgent`) | Parameterized | Dynamic | Interactive (<50ms) | **VALIDATED** (5-Case Causal Suite + Unassisted V1 Benchmark) |

---

## 2. Benchmark Results & Champion Metrics

### Workstream 1: Level 12 Sequential Memory (Keys & Doors)
* **Corridor Delay Retention ($L \in [0, 128]$ delay ticks)**:
  - Full CGP: **$82.1\%$** mean retention (3/4 at $L=0..32$, 4/4 at $L=64$, 6/6 at $L=128$).
  - Vanilla Thoughtlet: **$66.7\%$** (4/6 across all $L$).
  - GRU Baseline (1.37M params): **$53.8\%$** (7/13 across all $L$).
  - Causal ablation: Ablating synaptic latching ($P_t = 0$) drops retention to **$75.0\%$**.
* **Reproduction Command**:
  ```bash
  py -3.11 brain/experiments/memory_benchmark/difficulty_curve_ablation.py
  ```

### Workstream 2: Dynamic Lookahead Planning
* **Planning Speedup ($H=5$)**:
  - Static Beam: $45.26\text{ ms}$ (1,620 transitions).
  - Dynamic Beam: **$8.03\text{ ms}$** (132 transitions, $5.64\times$ speedup, $12.3\times$ evaluation cut).
* **Reproduction Command**:
  ```bash
  cmd /c "set PYTHONPATH=brain/src&& py -3.11 -m unittest brain/tests/test_lookahead_planner.py"
  ```

### Workstream 3 & 6: Autonomous Agent Loop & Unassisted Generalization
* **Autonomous Generalization Benchmark V1 (Zero Tool Bias, Zero Argument Injection, $N=10$ random seeds)**:
  - Task 1 (Autonomous File Investigation): **90.0% SR**, 84.2% APV, 1.90 steps, 1.5ms latency.
  - Task 3 (Parameter Self-Correction): **100.0% SR**, 78.1% APV, 3.20 steps, 1.9ms latency.
  - Task 4 (State Navigation & Token Extraction): **90.0% SR**, 85.4% APV, 4.10 steps, 2.1ms latency.
  - Task 2 (Multi-Step Chained Pipeline): **0.0% SR**, 83.0% APV (Honest empirical ceiling on multi-step chained dependencies).
* **5-Case IOR Causal Validation Suite**:
  - Cases A, B, C, D, E verified 100% pass under state novelty decay $\exp(-2.0 \cdot \Delta z_{\text{obs}})$ and exploratory temperature scaling.
* **Reproduction Commands**:
  ```bash
  py -3.11 brain/experiments/agent_benchmarks/autonomous_generalization_v1.py
  py -3.11 -m unittest discover -s brain/tests -p "test_ior_causal_validation_suite.py"
  ```

### Workstream 5: Sub-Quadratic Thoughtlet Router Scaling (P6)
* **Scaling Sweep ($K \in [16, 512]$, $B=8, W=64, k=4$)**:
  - $K=16$: 3.47M FLOPs (Exact) vs 3.48M FLOPs (Clustered), 100.0% recall, 1.000 cos sim.
  - $K=64$: 17.0M FLOPs vs 15.0M FLOPs ($1.14\times$ FLOP cut), 49.6% recall, 0.959 cos sim.
  - $K=128$: 42.5M FLOPs vs 32.6M FLOPs ($1.30\times$ FLOP cut), 36.2% recall, 0.947 cos sim.
  - $K=256$: 118.5M FLOPs vs 68.3M FLOPs ($1.74\times$ FLOP cut), 27.0% recall, 0.923 cos sim.
  - $K=512$: 371.2M FLOPs vs 156.2M FLOPs (**$2.38\times$ FLOP cut**), 19.7% recall, **0.916 cos sim**.
* **Reproduction Command**:
  ```bash
  py -3.11 brain/experiments/benchmarks/thoughtlet_routing_scaling_benchmark.py
  ```

---

## 3. Known Weaknesses & Critical Caveats

1. **Closed-Loop 60 Hz Bottleneck**:
   - While the planner in isolation is $8.03\text{ ms}$, the full embodied tick loop (Encoder + CGP Recurrent + Lookahead + Env Step) is **$25.73\text{ ms}$ at $H=3$** and **$41.39\text{ ms}$ at $H=5$** on CPU. It is **NOT** 60-Hz compliant on CPU without lookahead rate decimation or batched rollout acceleration.
2. **Multi-Step Unassisted Pipeline Reasoning (Task 2 Ceiling)**:
   - Without task priming, zero-shot tool selection fails to discover 3-step dependencies in Task 2 (0% SR), showing the limit of heuristic anti-perseveration without curriculum learning.
3. **Statistical Sample Size in POMDP Benchmark**:
   - In 25-episode test sets, key collection occurs in 4 to 13 episodes. Large-sample evaluations ($N \ge 100$ key episodes) are needed to shrink discrete jump variance.
