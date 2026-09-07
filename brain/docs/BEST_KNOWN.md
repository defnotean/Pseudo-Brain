# Pseudo-Brain: BEST_KNOWN State

**Last Updated:** 2026-09-07  
**Champion Git Commit:** `f81300d`  
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
| **Agent Infrastructure**| Contextual IOR (`PseudoBrainAgent`) | Parameterized | Dynamic | Interactive (<50ms) | **VALIDATED** (5-Case Causal Suite + Procedural DAG Ladder) |

---

## 2. Benchmark Results & Champion Metrics

### Workstream 1: Level 12 Sequential Memory (Keys & Doors)
* **Corridor Delay Retention with Real Sensory Masking ($L \in [0, 64]$ delay ticks)**:
  - Full CGP: **$85.0\%$** mean retention (**$100.0\%$** at $L=32$ and $L=64$).
  - GRU Baseline (1.37M params): **$89.3\%$** mean retention ($85.7\%$ at $L=64$).
  - Vanilla Thoughtlet: **$78.1\%$** mean retention (collapses to $62.5\%$ at $L=8$, $75.0\%$ at $L=64$).
  - Causal ablation: Ablating synaptic latching ($P_t = 0$, `cgp_no_cgsl`) drops mean retention to **$68.8\%$** ($50.0\%$ at $L=8$).
* **Critical Observability Finding**:
  - In standard unmasked `KeysDoorsEnv`, birds-eye 16x16 visibility allows feedforward reactive models to achieve **$81.0\%$ Key $\to$ Door conversion** with zero recurrent memory by detecting key absence directly from the pixel frame.
  - Working memory retention is only tested when partial observability is enforced (via observation masking in `CorridorDelayKeysDoorsEnv` or Chebyshev fog-of-war in `OcclusionEnv`).
* **Reproduction Command**:
  ```bash
  py -3.11 -c "from memory_benchmark.difficulty_curve_ablation import run_difficulty_curve_benchmark; run_difficulty_curve_benchmark()"
  ```

### Workstream 2: Dynamic Lookahead Planning Pareto Frontier (P5)
* **Pareto Audit vs. Exhaustive Search ($H \in [2, 3, 4, 5, 6]$, diverse decision states)**:
  - $H=2$: **100.0% Agreement**, 0.00 Regret, 0.0% False Pruning, $1.16\times$ speedup ($8.49\text{ ms}$ vs $9.83\text{ ms}$).
  - $H=3$: **100.0% Agreement**, 0.00 Regret, 0.0% False Pruning, $1.99\times$ speedup ($16.30\text{ ms}$ vs $32.46\text{ ms}$).
  - $H=4$: **100.0% Agreement**, 0.00 Regret, 0.0% False Pruning, **$4.22\times$ speedup** ($25.79\text{ ms}$ vs $108.74\text{ ms}$).
  - $H=5$: **100.0% Agreement**, 0.00 Regret, 0.0% False Pruning, **$12.06\times$ speedup** ($33.83\text{ ms}$ vs $408.06\text{ ms}$).
  - $H=6$: **100.0% Agreement**, 0.00 Regret, 0.0% False Pruning, **$38.20\times$ speedup** ($48.86\text{ ms}$ vs $1,866.59\text{ ms}$).
* **Closed-Loop Arcade Benchmark (MazeChase, $H=3$)**:
  - Pellets: 3.0 (Exhaustive) vs **3.0** (Dynamic Beam).
  - Ghost Collisions: 1.0 vs **1.0**.
  - Tick Latency: $47.99\text{ ms}$ $\to$ **$30.40\text{ ms}$** ($1.58\times$ full loop speedup).
* **Reproduction Command**:
  ```bash
  py -3.11 brain/experiments/benchmarks/planner_quality_pareto_benchmark.py --horizons 2 3 4 5 6
  ```

### Workstream 3 & 6: Autonomous Agent Loop & Unassisted Generalization
* **Procedural Task DAG Capability Ladder Benchmark (WS3 / WS6, N=100 seeds, L1-L8, Shuffled Tools, Zero Bias, 2,400 runs)**:
  - Blind Anti-Perseveration: **25.6% Overall SR** (collapses to 20.0% on L4 retry, 0.0% on L6/L7, 8.0% on L8).
  - Uninhibited Agent (No-IOR): **37.3% Overall SR** (collapses to 16.0% on L4 retry, 1.0% on L8 due to perseveration).
  - Contextual IOR PseudoBrainAgent: **96.6% Overall SR** (100% on L1-L7, 73.0% on L8, 5.1 steps on L4).
  - Double Dissociation on L4 proves that state novelty $\Delta z_{\text{obs}}$ decays inhibition to allow retrying previously failed tools after repair.
* **Autonomous Generalization Benchmark V1 (Zero Tool Bias, Zero Argument Injection, $N=10$ random seeds)**:
  - Task 1 (Autonomous File Investigation): **70.0% SR**, 83.8% APV, 3.70 steps, 3.0ms latency.
  - Task 2 (Multi-Step Dependent Pipeline): **60.0% SR**, 85.7% APV, 7.70 steps, 6.2ms latency (upgraded from 0.0% via endogenous stage tracking).
  - Task 3 (Parameter Self-Correction): **100.0% SR**, 88.9% APV, 3.60 steps, 2.8ms latency.
  - Task 4 (State Navigation & Token Extraction): **100.0% SR**, 82.4% APV, 3.40 steps, 2.7ms latency.
  - Overall Portfolio Mean: **82.5% SR**, **85.2% APV**.
* **5-Case IOR Causal Validation Suite**:
  - Cases A, B, C, D, E verified 100% pass under state novelty decay $\exp(-2.0 \cdot \Delta z_{\text{obs}})$ and exploratory temperature scaling.
* **Reproduction Commands**:
  ```bash
  py -3.11 brain/experiments/agent_benchmarks/procedural_task_dag_benchmark.py --episodes 25
  py -3.11 brain/experiments/agent_benchmarks/autonomous_generalization_v1.py
  py -3.11 -m pytest brain/tests/test_procedural_dag_capability_ladder.py brain/tests/test_ior_causal_validation_suite.py
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

### Workstream 7: Matched-Budget Architecture Comparison (P10)
* **Master Architecture Comparison ($L=16$ Delay Corridor POMDP, CPU Inference)**:
  - **Reactive Baseline**: 133k params, 0 state bytes, 2.52 MFLOPs, 0.48 ms, **0.0% retention** (cannot retain key).
  - **Heavy GRU Baseline**: 1,371k params, 1,536 state bytes, 4.99 MFLOPs, 0.60 ms, **50.0% retention** (0.036 ret/kParam).
  - **Dense Thoughtlet (32 slots)**: 65k params, 1,536 state bytes, 4.22 MFLOPs, 0.91 ms, **0.0% retention**.
  - **Sparse Thoughtlet ($k=4$)**: 65k params, 1,536 state bytes, 4.22 MFLOPs, 0.80 ms, **75.0% retention**.
  - **CGP Thoughtlet (Ours)**: **280k params**, 1,556 state bytes, 4.27 MFLOPs, 1.28 ms, **100.0% retention** (**0.357 ret/kParam, $9.92\times$ higher than GRU**).
  - **Full Pseudo-Brain (CGP + Router + Lookahead)**: 280k params, 5.96 MFLOPs, 26.36 ms, **100.0% retention**.
* **Reproduction Command**:
### Workstream 8: Multi-Threaded Latent Dependency Benchmark (MTLD-Bench / "The Kill Shot")
* **Multi-Variable Retention Across Extreme Grid ($M \in [2, 32]$ concurrent variables, Delay Horizons $L \in [16, 128]$ ticks)**:
  - **Non-CGP Baseline Collapse Across Grid**: Reactive (10.8%–13.3%), Monolithic Heavy GRU (10.8%–13.3%), Monolithic Matched GRU (10.8%–14.2%), Modern Diagonal SSM / GLRU (10.8%–14.2%), Single Thoughtlet (10.3%–14.2%), Dense Thoughtlets (10.8%–14.2%), and Sparse Thoughtlets (10.8%–14.2%) all **collapse completely to chance (~12.5%)** under dual delay corridors with intervening partial updates.
  - **CGP Multi-Variable Retention Across $M$**: CGP Thoughtlets sustain **42.9% at $M=2$, 38.3% at $M=4$, 29.8% at $M=8$, 22.3% at $M=16$, and 18.4% at $M=32$** (graceful capacity decay) at only 131k parameters ($4.0\times$ to $10.5\times$ smaller than GRU baselines).
  - **CGP Retention Across Delay Horizons ($L \in [16, 128]$)**: At canonical $M=4$, CGP maintains **38.3% at $L=16$, 34.6% at $L=32$, 31.5% at $L=64$, and 19.1% at $L=128$** (preserving untouched variables across 256 total ticks of masked distraction).
  - **The "Isolate on Orthogonality" Principle**: On orthogonal variables, `CGP - Disconnected Slots Ablation` reaches **43.3% at $M=2$, 39.7% at $M=4$, 31.0% at $M=8$, and 23.0% at $M=16$ with $2\times$ lower latency (1.46 ms vs 2.96 ms)**, confirming that shielding slots from inter-slot routing eliminates cross-slot noise when variables are independent.
  - **Cognitive Scaling Roadmap**: Formal specification detailing the $131\text{k} \to 10\text{M} \to 50\text{M} \to 100\text{M} \to 300\text{M} \to 1\text{B} \to 3\text{B}$ parameter ladder documented in [`COGNITIVE_SCALING_ROADMAP.md`](COGNITIVE_SCALING_ROADMAP.md).
* **Reproduction Command**:
  ```bash
  py -3.11 brain/experiments/memory_benchmark/multi_threaded_latent_dependency_benchmark.py --extreme-grid --num-seeds 15 --train-steps 120
  ```

### Workstream 9: MTLD Degradation Causal Diagnostic Suite
* **Causal Decomposition of the Delay Failure Curve ($L \in [16, 128]$)**:
  - **Hypothesis 1 (Passive Latch Decay)**: Setting $\lambda = 0.9999$ ($t_{1/2} \approx 6,931$ steps) prevents passive latch decay across 256 delay ticks, improving $L=128$ retention from **$17.9\%$ to $25.7\%$** (while avoiding numerical explosion of unconstrained $\lambda=1.0$).
  - **Hypothesis 2 (CIG Gate Bleed & Sharpening)**: Gate temperature sharpening ($T=0.5$) slashes delay salience on uninformative sensory noise by **$88\%$** ($0.336 \to 0.039$), boosting $L=128$ retention to **$30.4\%$**.
  - **Hypothesis 3 (Slot Width / Subspace Volume)**: Increasing slot width from $W=12$ to $W=48$ expands hyperspherical representation volume, lifting $L=128$ retention from $17.0\%$ to **$29.3\%$**.
  - **Hypothesis 4 (Routing Modality)**: On orthogonal variables, Disconnected routing achieves **$25.2\%$ at $L=128$** with $9.52\text{ ms}$ latency, outperforming Static Top-4 ($17.7\%$, $17.81\text{ ms}$).
* **Reproduction Command**:
  ```bash
  py -3.11 brain/experiments/memory_benchmark/mtld_degradation_diagnostic.py --num-seeds 15 --train-steps 120
  ```

### Workstream 10: Cognitive Scaling Ladder Benchmark (131k -> 8M)
* **Multi-Scale Empirical Validation vs. Monolithic GRU and Modern Diagonal SSM (GLRU)**:
  - **Empirical Baseline Non-Improvement**: Scaling the tested Monolithic GRU and Modern Diagonal SSM baselines from ~0.19M to ~10.7M parameters (+55x scale) did not materially improve performance on this MTLD configuration (remaining trapped at chance: $10.9\% - 14.7\%$).
  - **Pseudo-Brain Multi-Scale Invariance**: Pseudo-Brain CGP Thoughtlets sustain **$38.0\% - 39.4\%$ retention at $L=128$** (+25.5% to +27.9% margin over GRU/SSM) and graceful degradation at $M=32$ (**$20.9\% - 21.7\%$**) across tiers 131k to 2M.
  - **Sub-2ms CPU Inference**: Latency scales gracefully from **$0.96\text{ ms}$** at 131k to **$1.92\text{ ms}$** at 8M on single-threaded CPU.
* **Reproduction Command**:
  ```bash
  py -3.11 brain/experiments/memory_benchmark/cognitive_scaling_ladder_benchmark.py --num-seeds 15 --train-steps 120
  ```

---

## 3. Known Weaknesses & Critical Caveats

1. **The $M=32$ Capacity & Optimization Wall**:
   - While CGP degrades gracefully from $M=2$ ($42.9\%$) to $M=16$ ($22.3\%$) and holds ~21% at $M=32$ across 131k–2M tiers, the 8M model dips to **15.8%** at $M=32$. With $K=32$ and slot width $W=96$, monolithic flattened readouts ($K \cdot W = 3,072$ dimensions) suffer sample-starvation under standard training budgets, demonstrating an architectural boundary where slot-wise readouts or cross-slot attention are needed.
2. **Closed-Loop 60 Hz Bottleneck**:
   - While the planner in isolation is sub-10ms ($9.83\text{ ms}$ at $H=2$, $18.28\text{ ms}$ at $H=3$), the full embodied tick loop (Encoder + CGP Recurrent + Lookahead + Env Step) is **$25.24\text{ ms}$ at $H=3$** and **$39.53\text{ ms}$ at $H=5$** on single-threaded CPU. To achieve strictly $<16.67\text{ ms}$ closed-loop execution, the agent requires reflexive execution ($H=0$, $8.54\text{ ms}$), 1-step lookahead ($H=1$, $13.01\text{ ms}$), or dual-rate planning decimation.
3. **Horizon Dead-End Utility Calibration**:
   - When evaluating deep lookaheads ($H \ge 5$), fixed negative pruning thresholds can trigger spurious dead-end classification due to unnormalized cumulative discounting. Dynamic lookahead requires horizon-scaled thresholds ($\theta_{\text{dead}} = -20 \cdot H$).
4. **Multi-Step Unassisted Pipeline Recovery**:
   - Endogenous parameter inference achieves 60.0% SR on multi-step pipelines, leaving 40.0% where persistent exploration or explicit DAG sub-goal tracking is required.
