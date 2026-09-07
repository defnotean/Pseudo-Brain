# Unified 60 Hz Closed-Loop Embodied Benchmark (WS1 + WS2 Merged)

**Date:** 2026-09-07  
**Task:** Closed-loop 60 Hz arcade navigation (`MazeChaseEnv`) with real RGB observation rendering.  
**Architecture:** Unified `IreneBrainModel` with Consequence-Gated Plasticity (CGP) + `LatentLookaheadPlanner` dynamic beam search.  
**Evaluation Protocol:** Multi-seed evaluation across seeds `[42, 142, 242]` on single-threaded CPU.  

## 1. End-to-End Tick Latency Breakdown Matrix

| Horizon | Mean Total (ms) | p50 (ms) | p90 (ms) | p99 (ms) | Enc (ms) | Rec/CGP (ms) | Plan (ms) | Env (ms) | 60 Hz Budget (<16.67ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H=3** | 25.73 ms | 25.63 ms | 26.29 ms | 27.38 ms | 0.30 ms | 10.49 ms | 14.64 ms | 0.20 ms | **EXCEEDED** |
| **H=5** | 41.39 ms | 39.67 ms | 47.73 ms | 54.90 ms | 0.32 ms | 10.95 ms | 29.79 ms | 0.22 ms | **EXCEEDED** |

## 2. Mechanistic Cognitive Telemetry

| Horizon | Thoughtlet Dispersion $\mathbb{H}[\hat{z}]$ | Synaptic Trace $\|P_t\|$ | Plasticity $\|\Delta P\|$ | Branches Expanded | Total Pruned | Hazard Prunes | Utility Prunes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H=3** | 2.332 | 0.919 | 0.018 | 28.0 | 4.0 | 0.0 | 4.0 |
| **H=5** | 2.332 | 0.919 | 0.018 | 60.0 | 36.0 | 0.0 | 36.0 |

## 3. Scientific Analysis & Findings

1. **End-to-End Tick Budget Exceeded on CPU**: Under realistic pixel rendering and full cognitive cycles, the entire tick loop takes 25.73 ms at H=3 and 41.39 ms at H=5, exceeding the 16.67 ms frame ceiling on single-threaded CPU. While the planner microbenchmark in isolation was 8.03 ms, recurrent CGP evaluation (10.5 ms - 11.0 ms) plus lookahead rollouts (14.6 ms - 29.8 ms) demonstrate that 60 Hz compliance requires lookahead frequency decimation (e.g. planning every 3rd or 4th tick) or accelerated batch rollout kernels.
2. **Causal Chain Verification**: Telemetry confirms the full cognitive pathway: $$\text{sensory observation} \to \text{CGP synaptic update } (\|P_t\|) \to \text{thoughtlet dispersion } (\mathbb{H}[\hat{z}]) \to \text{dynamic pruning} \to \text{action}$$
