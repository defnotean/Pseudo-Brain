# Cognitive Scaling Ladder Benchmark Report (131k -> 500k -> 2M -> 8M)
**Date:** 2026-09-07  
**Status:** `[MEASURED]` Multi-scale empirical comparison between Pseudo-Brain CGP, Monolithic GRU, and Diagonal SSM (GLRU).  

## 1. Executive Summary
This experiment tracks whether the architectural advantage of Pseudo-Brain's shared recurrent core, CIG gating, and synaptic latching ($P_t$) scales monotonically across parameter tiers (131k to 8M) and stress regimes:
- **Condition A**: Baseline MTLD ($M=4, L=16$)
- **Condition B**: Temporal Delay Stress ($M=4, L=128$)
- **Condition C**: Massive Concurrent Load ($M=32, L=32$)

## 2. Scaling Trajectories across Parameter Tiers

### Condition A: Baseline (M=4, L=16)
| Tier | CGP Accuracy | GRU Accuracy | GLRU (SSM) Accuracy | CGP Margin vs GRU | CGP Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **131k** | **39.0%** | 13.9% | 13.3% | **+25.1%** | 0.96 ms |
| **500k** | **38.6%** | 12.5% | 12.5% | **+26.1%** | 1.05 ms |
| **2M** | **39.3%** | 14.7% | 12.7% | **+24.6%** | 1.26 ms |
| **8M** | **38.9%** | 10.9% | 13.0% | **+27.9%** | 1.92 ms |

### Condition B: Temporal Stress (M=4, L=128)
| Tier | CGP Accuracy | GRU Accuracy | GLRU (SSM) Accuracy | CGP Margin vs GRU | CGP Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **131k** | **38.0%** | 11.6% | 11.5% | **+26.5%** | 6.20 ms |
| **500k** | **37.6%** | 11.5% | 11.5% | **+26.1%** | 6.73 ms |
| **2M** | **38.4%** | 11.1% | 12.1% | **+27.3%** | 8.13 ms |
| **8M** | **39.4%** | 13.9% | 11.5% | **+25.5%** | 12.20 ms |

### Condition C: Massive Load (M=32, L=32)
| Tier | CGP Accuracy | GRU Accuracy | GLRU (SSM) Accuracy | CGP Margin vs GRU | CGP Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **131k** | **20.9%** | 12.4% | 12.8% | **+8.6%** | 4.61 ms |
| **500k** | **21.7%** | 12.7% | 13.1% | **+9.0%** | 6.14 ms |
| **2M** | **21.6%** | 12.5% | 12.4% | **+9.1%** | 9.16 ms |
| **8M** | **15.8%** | 12.7% | 12.7% | **+3.1%** | 24.19 ms |
