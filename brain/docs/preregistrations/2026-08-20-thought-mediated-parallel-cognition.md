# Formal Preregistration: Thought-Mediated Parallel Cognition

**Registration Date**: 2026-08-20  
**Registration ID**: `PREREG-PHASE2-REVISED-THOUGHT-MEDIATED-V1`  
**Status**: ACTIVE 🧠  

---

## 1. Central Scientific Hypothesis

> **Thought-Mediated Parallel Cognition Hypothesis**:  
> *Parallel persistent thought states provide a reproducible causal advantage over strong monolithic recurrent architectures at strictly matched compute and data budgets when decision-relevant prediction, memory, and alternative-action information must flow through them rather than being bypassable by a monolithic belief representation.*

---

## 2. Structural Architecture & Elimination of Action Bypass

```text
                 CURRENT OBSERVATION
                         │
                  Sensory Encoder
                         │
                         ▼
                 PERSISTENT BELIEF
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      WORKING / EPISODIC       THOUGHT FIELD
           MEMORY              K = 32 initially
                                     │
                       ┌─────────────┼─────────────┐
                       ▼             ▼             ▼
                   Future A      Future B      Future C
                   / hypothesis  / hypothesis  / hypothesis
                       │             │             │
                       └──────┬──────┴──────┬──────┘
                              │
                   PER-THOUGHT ACTION PROPOSALS
                              │
                  permutation-invariant aggregation
                              │
                              ▼
                       MAIN ACTION INTENT
                              │
          fresh sensors ──────┼────── bounded reflex correction ([-0.1, +0.1])
                              │
                              ▼
                        FINAL CONTROL
```

### **Key Architectural Invariants**:
1. **No Belief-to-Action Bypass**: The main action intent is decoded exclusively from the aggregation of per-thoughtlet action proposals. The belief state updates the thought field, but has no direct connection to the main action logits.
2. **Shared Per-Thoughtlet Proposal Head**: All $K$ thoughtlets share an identical proposal head mapping slot representation $z_k \in \mathbb{R}^d$ to:
   - $\hat{y}_k$: candidate future hypothesis (displacement $\Delta \hat{x}, \Delta \hat{y}$, hazard $\hat{d}$, reward $\hat{r}$)
   - $\hat{a}_k$: candidate action logits (WASD intent)
   - $\hat{u}_k$: scalar utility / confidence logit
3. **Permutation-Invariant Proposal Aggregation**:
   $$\text{Weight}_k = \frac{\exp(\hat{u}_k / \tau)}{\sum_{j=1}^K \exp(\hat{u}_j / \tau)}$$
   $$\text{MainActionIntent} = \sum_{k=1}^K \text{Weight}_k \cdot \hat{a}_k$$
4. **Bounded Reflex Path**:
   - Fresh sensors pass through a single small linear layer bounded to $[-\delta_{\max}, +\delta_{\max}]$ ($\delta_{\max} = 0.10$).
   - Incapable of solving navigation, route planning, memory, or pursuit strategy by itself.

---

## 3. Resource-Matched Scaling Protocol

To rigorously isolate parallelism as an architectural variable, $K \in \{1, 4, 8, 16, 32\}$ configurations will be constructed with compensating hidden width:

| Configuration | Thoughtlets ($K$) | Compensating Core Width ($W$) | Parameter Count | Total FLOPs |
| :---: | :---: | :---: | :---: | :---: |
| **$K = 1$ (Matched Monolith)** | 1 | 180 | $\approx 127\text{k}$ | Matched |
| **$K = 4$** | 4 | 90 | $\approx 127\text{k}$ | Matched |
| **$K = 8$** | 8 | 64 | $\approx 127\text{k}$ | Matched |
| **$K = 16$** | 16 | 45 | $\approx 127\text{k}$ | Matched |
| **$K = 32$ (Standard Field)** | 32 | 32 | $127,019$ | Matched |

---

## 4. Fair Baseline Suite under Equal Experience

All baselines receive the **identical training transition budget, multi-family curriculum, and counterfactual branch datasets**:
1. **Baseline 0 (Critical Control)**: Monolithic GRU with the same Proposal-Style Action Aggregator.
2. **Baseline 1**: Monolithic GRU standard readout.
3. **Baseline 2**: State-Space Model (SSM / S4 style).
4. **Baseline 3**: Recurrent Transformer.
5. **Baseline 4**: Latent World-Model Actor.

---

## 5. Preregistered Go / No-Go Pass Gates

| Gate ID | Criterion | Target Threshold |
| :---: | :--- | :--- |
| **Gate 1** | **Thought Mediation** | Knocking out thoughts collapses Families B–E returns by $\ge 30\%$, while reflex tasks remain stable. |
| **Gate 2** | **Monotonic Capacity Scaling** | $K=32 \succ K=16 \succ K=8 \succ K=4 \succ K=1$ (statistically significant trend). |
| **Gate 3** | **Causal Slot Usefulness** | $\ge 8 / 32$ slots demonstrate context-dependent positive marginal utility ($\ge 5\%$ drop on knockout). |
| **Gate 4** | **Permutation Equivariance** | Whole-slot permutation causes $\le 1.0\%$ change in action output. |
| **Gate 5** | **Cognitive Scrambling** | Breaking internal thoughtlet register binding causes $\ge 15\%$ degradation. |
| **Gate 6** | **Baseline Superiority** | $\ge 10\%$ normalized return / IQM advantage over matched Proposal-GRU baseline. |
| **Gate 7** | **Breadth** | Positive 95% bootstrap CI across $\ge 4 / 5$ procedural task families. |
| **Gate 8** | **DGX Spark Real-Time Deadline** | Kernel p99 $\le 8.00\text{ ms}$, end-to-end loop p95 $\le 16.67\text{ ms}$ on NVIDIA GB10 GPU. |

## 6. Gate 6 measurement pair (named probe)

The first closed-loop IQM comparison of the **mediated** (no bypass) model against
Proposal-GRU is the bounded probe
[`2026-08-20-gate6-matched-proposal-gru-probe.md`](2026-08-20-gate6-matched-proposal-gru-probe.md):

- Thought-mediated: K=32 with core width **searched** so parameter count matches Proposal-GRU ±5% (HEAD W=32 map is not matched: 270k vs 787k).
- Proposal-GRU: `ProposalGRUBaseline(hidden_dim=180)`.
- Equal experience: 300 AdamW steps, seeds `{42,43,44,45,46}`, all five Phase-2 families.
- Pass remains the table above (≥10% IQM/return). GRU tie or win is **FAIL**.
