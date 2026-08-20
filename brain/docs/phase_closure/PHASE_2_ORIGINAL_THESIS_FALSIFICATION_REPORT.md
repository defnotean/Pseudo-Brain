# Phase 2 Original Microthought Thesis Falsification Report

```text
================================================================================
PHASE 2 ORIGINAL HYPOTHESIS: FORMALLY RETIRED & FALSIFIED ✅ (Clean Negative Result)
Preregistered 2-Design Protocol: Completed across 5 seeds on NVIDIA DGX Spark
================================================================================
```

## 1. The Original Central Phase 2 Hypothesis

> *At matched parameters, FLOPs, training experience, observation/action interface, and physical latency budget, does a persistent field of parallel thoughtlets provide a reproducible causal advantage over strong conventional recurrent architectures (monolithic GRU, SSM, Transformer, world model) without structural routing mediation?*

---

## 2. Empirical Evidence & Failure Modes across 2 Substantial Designs

Both designs were evaluated across **5 random seeds (42, 43, 44, 45, 46)** on **NVIDIA DGX Spark** (`cuda:0`, NVIDIA GB10 GPU, CUDA 13.0, PyTorch 2.13.0+cu130) across all **5 procedural task families**:

### **A. Design 1: Reference Architecture with Direct Global Actuator Attention**
- **Outcome**: Pseudo-Brain scored **-32.63 IQM Return** (ranked 8th out of 8 architectures, trailing GRU at **-24.26**).
- **Equivalence**: Pseudo-Brain (-32.63) performed almost identically to the Deeper Serial Baseline (-32.39).
- **Causal Knockout**: **0 / 32 slots** caused $\ge 5\%$ degradation under single-slot removal.
- **Verdict**: Failed to achieve $\ge 5\%$ improvement over baseline.

### **B. Design 2: Hungarian Multi-Future Branch Supervision & DAgger**
- **Outcome**: Improved return to **-27.45 IQM Return**, beating Transformer (-28.94) and Deeper Serial (-32.39).
- **Equalized Experience Control**: When the monolithic GRU received the exact same multi-family interactive transitions and branch targets, **it matched Pseudo-Brain point-for-point at -27.45**.
- **The Attention Illusion**: The actuator allocated **44.98% of its raw cross-attention mass to thoughts**, yet zeroing all thoughts produced **0.00% degradation**. The policy decoded actions directly from `sensors` and `belief`, completely bypassing `thoughts`.
- **Capacity Scaling Curve**: Flat ($-28.00$ across $K \in \{1, 4, 8, 16, 24, 32\}$).
- **Verdict**: Failed to achieve $\ge 5\%$ improvement over the matched GRU under equal experience.

---

## 3. Preregistered Stopping Rule Execution

In accordance with Section 1.3 of the Phase 2 Research Directive:
> *"After two substantially different thought-field designs with at least three independent seeds each, if neither prediction nor control improves by at least ~5% over the strongest matched conventional baseline: RETIRE PARALLEL MICROTHOUGHTS AS THE CENTRAL THESIS. A clean negative result is scientifically successful."*

The original unmediated microthought thesis is now **formally retired and archived as a successful scientific falsification**.

---

## 4. Scientific Diagnosis: The Structural Bypass Flaw

The failure of the original architecture was not caused by a lack of representational capacity in the thoughtlets (which successfully learned alternative future branch predictions), but by the **unrestricted information bypass** in the actuator:
$$\text{Action} = \text{Decoder}(\text{Sensors}, \text{Belief}, \text{Thoughts}, \text{WorkingMemory})$$
Given direct access to `Belief` and `Sensors`, the gradient optimizer consistently took the path of least resistance, routing all action decisions around the thoughtlets.

---

## 5. Successor Milestone: Thought-Mediated Parallel Cognition

A new preregistered hypothesis is registered in [`docs/preregistrations/2026-08-20-thought-mediated-parallel-cognition.md`](../preregistrations/2026-08-20-thought-mediated-parallel-cognition.md):
- **Core Principle**: Action choice must flow **through** the thought field via per-thought action proposals and permutation-invariant aggregation.
- **Main Action Bypass**: Mechanically eliminated.
- **Reflex Path**: Separate, strictly bounded $[-0.1, 0.1]$ reflex correction for short-timescale physical stabilization only.
