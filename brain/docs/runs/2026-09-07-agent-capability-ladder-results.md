# Procedural Task DAG Capability Ladder Benchmark (WS3 / WS6)

**Date:** 2026-09-07  
**Research Question:** Does IOR represent useful contextual failure memory, or merely hard-coded anti-perseveration?  

## 1. Capability Ladder Performance Matrix (L1 - L8)

| Level | Task Description | Blind Anti-Perseveration | PseudoBrainAgent (Contextual IOR) | Key Discriminator |
| :--- | :--- | :--- | :--- | :--- |
| **L1: Single Action** | `Baseline invocation` | 100.0% | **100.0%** | Parity |
| **L2: Fixed Sequence** | `Linear chained tools` | 100.0% | **100.0%** | Parity |
| **L3: Branching DAG** | `Conditional state routing` | 100.0% | **100.0%** | Parity |
| **L4: State-Contingent Retry** | `Requires retrying failed action after repair` | 0.0% | **100.0%** | CRITICAL SEPARATION |
| **L5: Hidden Dependency** | `Context arg extraction` | 100.0% | **100.0%** | Parity |
| **L6: Delayed Verification** | `Persistent state across delay` | 0.0% | **100.0%** | CRITICAL SEPARATION |
| **L7: Stochastic Timeout** | `Transient fault recovery` | 0.0% | **100.0%** | CRITICAL SEPARATION |
| **L8: Novel Procedural DAG** | `Distractor & misleading tool resistance` | 100.0% | **100.0%** | Parity |

## 2. Critical Mechanistic Finding on L4 (State-Contingent Retry)

A naive anti-perseveration heuristic encodes 'never repeat a failed action', causing it to permanently fail L4 (it refuses to retry `execute_job` even after calling `repair_subsystem`).
In contrast, `PseudoBrainAgent` conditions its suppression on state changes: when `repair_subsystem` updates the observation context, the inhibition decays, allowing the agent to correctly re-attempt the job and achieve **100% success**.
