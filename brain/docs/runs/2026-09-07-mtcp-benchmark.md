# Multi-Threaded Cognitive Process Benchmark Report (MTCP-Bench)
**Date:** 2026-09-07  
**Status:** `[MEASURED]` Evaluates 4 parameter-matched architectures across realistic multi-threaded workloads.  

## 1. Executive Summary
MTCP-Bench directly evaluates whether Pseudo-Brain's specialized cognitive primitive survives realistic multi-threaded workloads with:
- **Interleaved Asynchronous Threads** ($N \in [8, 16]$)
- **Mid-flight Interruptions & Preemption** ($D_{\text{interrupt}} \in [8, 16]$ steps)
- **Cross-Thread Dependencies** (Thread $B$ dependent on output of Thread $A$)
- **Orthogonal Independent Threads** (zero cross-talk tolerance)
- **Delayed Consequence Feedback**

## 2. Benchmark Results & Compound Cognitive Scores

### 8 Concurrent Cognitive Threads
| Architecture | Params | Retention Acc | Isolation Acc | Recovery Acc | Dependency Acc | Latency | Compound Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pseudo-Brain CGP** | 130.2k | **60.4%** | **76.2%** | **99.6%** | 100.0% | 1.57 ms | **45.88** |
| **Monolithic GRU** | 276.6k | **18.3%** | **13.1%** | **20.8%** | 17.1% | 0.28 ms | **0.50** |
| **Modern Diagonal SSM (GLRU)** | 269.0k | **15.0%** | **9.2%** | **12.1%** | 12.1% | 0.30 ms | **0.17** |
| **Recurrent Linear Attention** | 173.7k | **12.9%** | **13.1%** | **12.1%** | 12.1% | 0.18 ms | **0.20** |

### 16 Concurrent Cognitive Threads
| Architecture | Params | Retention Acc | Isolation Acc | Recovery Acc | Dependency Acc | Latency | Compound Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pseudo-Brain CGP** | 130.2k | **54.2%** | **91.9%** | **76.2%** | 62.9% | 2.37 ms | **37.95** |
| **Monolithic GRU** | 276.6k | **20.0%** | **14.8%** | **15.4%** | 17.9% | 0.36 ms | **0.46** |
| **Modern Diagonal SSM (GLRU)** | 269.0k | **10.0%** | **8.5%** | **11.7%** | 14.2% | 0.37 ms | **0.10** |
| **Recurrent Linear Attention** | 173.7k | **9.6%** | **9.2%** | **13.8%** | 15.0% | 0.21 ms | **0.12** |
