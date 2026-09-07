# MTLD Degradation Causal Diagnostic Report
**Date:** 2026-09-07  
**Status:** `[MEASURED]` Causal decomposition of the delay degradation failure curve and routing crossover.  

## 1. Executive Summary
Following the discovery of the delay failure curve in MTLD-Extreme (where retention decays from 38.3% to 19.1% across $L=16 \to 128$), this diagnostic systematically audited the 4 root-cause hypotheses:
1. **Hypothesis 1 (Passive Latch Decay)**: Tested whether passive decay in $P_t = \lambda P_{t-1} + \eta \Delta P$ erodes memory over 256 delay ticks.
2. **Hypothesis 2 (CIG Gate Bleed)**: Audited mean salience $\bar{g}_{\text{delay}}$ on uninformative sensory noise.
3. **Hypothesis 3 (Slot Width / Dimensional Capacity)**: Swept slot width $W \in [12, 24, 48, 64]$ to determine if hyperspherical subspace collapse causes long-horizon forgetting.
4. **Hypothesis 4 (Routing Modality & Crossover)**: Audited the Disconnected vs. Routed crossover and evaluated Dependency-Gated Dynamic Routing.

## 2. Experiment 1: Latch Persistence Sweep (Lambda in [0.99, 0.999, 0.9999, 1.0])
| Lambda Decay | Delay L=16 | Delay L=32 | Delay L=64 | Delay L=128 | Retention Drop (16->128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **lambda_0.99** | 36.0% | 33.3% | 24.1% | **16.2%** | -19.8% |
| **lambda_0.999** | 36.1% | 35.7% | 24.0% | **17.9%** | -18.2% |
| **lambda_0.9999** | 33.1% | 32.3% | 29.8% | **25.7%** | -7.4% |
| **lambda_1.0** | 32.1% | 30.2% | 26.5% | **17.7%** | -14.4% |

## 3. Experiment 2: CIG Gate Bleed & Sharpening
| Gate Temperature | Delay L=16 Acc | Delay L=64 Acc | Delay L=128 Acc | Mean Gate Salience |
| :--- | :---: | :---: | :---: | :---: |
| **temp_1.0** | 36.8% | 32.0% | **28.6%** | 0.4086 |
| **temp_0.5** | 36.9% | 33.5% | **30.4%** | 0.1671 |
| **temp_0.2** | 39.6% | 32.6% | **29.3%** | 0.2403 |

## 4. Experiment 3: Slot Width Capacity Scaling (W in [12, 24, 48, 64])
| Slot Width | Params | Delay L=16 | Delay L=32 | Delay L=64 | Delay L=128 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **width_12** | 99,559 | 36.6% | 30.9% | 23.2% | **17.0%** | 2.91 ms |
| **width_24** | 361,627 | 31.9% | 31.7% | 30.3% | **24.6%** | 3.26 ms |
| **width_48** | 925,507 | 39.5% | 36.2% | 34.1% | **29.3%** | 3.83 ms |
| **width_64** | 1,227,187 | 37.9% | 34.5% | 29.8% | **23.8%** | 4.41 ms |

## 5. Experiment 4: Routing Modality & The Long-Horizon Crossover
| Routing Strategy | Delay L=16 | Delay L=32 | Delay L=64 | Delay L=128 | Latency (L=16) | Latency (L=128) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disconnected (A=0)** | **40.9%** | 34.9% | 32.7% | **25.2%** | 1.47 ms | 9.52 ms |
| **Static Top-4 Routing** | **33.5%** | 27.8% | 16.1% | **17.7%** | 2.86 ms | 17.81 ms |
| **Dense All-to-All Attention** | **30.3%** | 26.0% | 27.4% | **24.7%** | 2.19 ms | 13.92 ms |
| **Dependency-Gated Dynamic Routing** | **38.5%** | 34.8% | 29.2% | **20.4%** | 1.99 ms | 12.91 ms |