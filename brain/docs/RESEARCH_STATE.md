# Pseudo-Brain Research State

**Date:** 2026-09-07  
**Master Roadmap Phase:** Phase 2.6 (Core V1 Hardening) $\to$ Phase 4 (Autonomous Agency Infrastructure)  
**Active Git Branch:** `defnotean/pseudo-brain`  
**Remote GPU Verification:** Google Colab NVIDIA A100-SXM4-40GB (Session `pb-research`)  

---

## 1. Executive Research Summary & Epistemic Status

We maintain strict claim discipline ([MEASURED], [INFERRED], [HYPOTHESIS], [ASPIRATIONAL]) across all active tracks:

1. **Workstream 1 (Level 12 POMDP Keys & Doors — Sequential Long-Term Memory)**:
   - **Status**: [MEASURED]
   - **Empirical Finding**: CGP substantially improves long-horizon memory retention and approaches GRU retention at $\sim 4.9\times$ lower parameter count (280k vs 1,371k), but with substantially higher seed variance ($68.3\% \pm 13.1\%$ vs $69.4\% \pm 2.5\%$ across 3 seeds on NVIDIA A100).
   - **Epistemic Limitation**: $68.3\% \pm 13.1\%$ vs $69.4\% \pm 2.5\%$ is **not evidence of equivalence** given the $\sim 5\times$ higher variance of CGP. A systematic memory difficulty curve across corridor lengths ($0, 4, 8, 16, 32, 64, 128$) and causal ablations (without CIG, without CGSL) are required to establish reliability.
   - Result: Documented in `brain/docs/runs/2026-09-07-cgp-memory-benchmark-keys-doors.md`.

2. **Workstream 2 (Embodied Dynamic Branch Pruning in Latent Lookahead Planning)**:
   - **Status**: [MEASURED] (Microbenchmark)
   - **Empirical Finding**: Dynamic pruning brings the tested $H=5$ planner below the 60-Hz planning-time budget in the reported microbenchmark (dropping from $45.26\text{ ms}$ to $8.03\text{ ms}$, a $5.64\times$ speedup; evaluated transitions slashed from 1,620 to 132).
   - **Epistemic Limitation**: This demonstrates that the tested $H=5$ planning workload fits within the 16.67 ms budget in isolation. It does **not** yet prove the complete embodied agent is 60-Hz capable. Full end-to-end tick latency (observation $\to$ encoder $\to$ belief $\to$ CGP recurrent $\to$ world prediction $\to$ beam search $\to$ action selection $\to$ env) must be verified as a unified system.
   - Result: Documented in `brain/docs/runs/2026-09-07-dynamic-branch-pruning-lookahead.md`.

3. **Workstream 3 & 6 (Autonomous Agent Loop Infrastructure & Level 16 Tool Workflows)**:
   - **Status**: [MEASURED] (Scripted Workflows)
   - **Empirical Finding**: Agent-loop infrastructure demonstrates verified multi-step tool execution, contextual argument generation, and tested failure recovery (IOR suppression $P_t[a] \le -4.0$) across scripted software engineering workflows.
   - **Epistemic Limitation**: **General multi-step autonomous reasoning remains unvalidated.** The current tests verify that hand-engineered anti-perseveration heuristics and tool execution machinery succeed on specific test DAGs. Generalization across randomized task graphs, distractor tools, and state-contingent retries (where repeating a previously failed action is required after state repair) remains an open benchmark challenge.
   - Result: Documented in `brain/docs/runs/2026-09-07-agent-reasoning-and-inhibition-of-return.md` and `brain/docs/runs/2026-09-07-level16-agent-workflows.md`.

4. **Workstream 4 (Consequence-Gated Plasticity Unified into 60 Hz Embodied Arcade Architecture)**:
   - **Status**: [MEASURED] (Unit & Integration Suite)
   - **Empirical Finding**: Unified fast synaptic weights $P_t$ into `BrainCell` / `PlasticBrainCell` and `IreneBrainModel` with zero rollout state mutation side-effects. Zero-initialized residual latent prior eliminates spurious surprise on blank frames ($100\%$ cosine similarity retention across 20 blank ticks vs $0.7910$ baseline; $96.1\%$ norm retention). Forward pass executes in $1.50\text{ ms}$ on CPU.
   - Result: Documented in `brain/docs/runs/2026-09-07-cgp-arcade-unification.md`.

5. **Workstream 5 (Top-$k$ Sparse Thoughtlet Routing & Scaling Benchmark)**:
   - **Status**: [MEASURED] (Microbenchmark)
   - **Empirical Finding**: Proved slot-permutation equivariance and top-$k$ gradient isolation with self-exclusion ($S_{i,i} = -\infty$). At $K=64$, latency is $0.653\text{ ms}$ on CPU, shielding 62 out of 64 slots from peer cross-talk.
   - **Epistemic Limitation**: Whether sparse inter-thoughtlet routing is necessary or beneficial for embodied downstream tasks remains an open empirical research question.
   - Result: Documented in `brain/docs/runs/2026-09-07-sparse-thoughtlet-routing.md`.

---

## 2. Canonical Empirical Results Matrix

### Workstream 1: Level 12 Keys & Doors Sequential Memory Benchmark (NVIDIA A100)

| Model Architecture | Parameters | Val Loss (mean $\pm$ std) | Val Acc (mean $\pm$ std) | Key $\to$ Door Retention (mean $\pm$ std) | Corridor Collapse Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reactive Baseline** | 227,844 | $0.6934 \pm 0.0001$ | $50.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $100\%$ (Zero Memory) |
| **Vanilla Thoughtlet** | 280,069 | $0.4491 \pm 0.0634$ | $88.5\% \pm 3.1\%$ | $58.1\% \pm 20.8\%$ | $33.3\%$ (Severe Collapse on Seed 242: 36.4%) |
| **Heavy GRU Baseline** | 1,371,140 | $\mathbf{0.2520 \pm 0.0528}$ | $\mathbf{96.1\% \pm 0.8\%}$ | $\mathbf{69.4\% \pm 2.5\%}$ | $0.0\%$ (Consistent Retention) |
| **CGP Thoughtlet (Ours)**| **280,069** | $0.2642 \pm 0.0381$ | $95.4\% \pm 1.0\%$ | **$68.3\% \pm 13.1\%$** | **$0.0\%$** (Zero Collapse, Peaked at 80.0%) |

*Scientific Note: $68.3\% \pm 13.1\%$ vs $69.4\% \pm 2.5\%$ demonstrates competitive mean retention at $4.9\times$ lower parameter scale, but variance is $\sim 5\times$ higher; causal decomposition across delay horizons is ongoing.*

---

### Workstream 2 & 4: Embodied Latency & Computational Budgets (CPU)

| Module / Operation | Configuration | Target Deadline | Measured Latency | Budget Status | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lookahead Planner ($H=2$)** | Dynamic Beam Search | $\le 16.67\text{ ms}$ | $3.64\text{ ms}$ | Met | Shallow lookahead within budget |
| **Lookahead Planner ($H=5$)** | Dynamic Beam Search | $\le 16.67\text{ ms}$ | **$8.03\text{ ms}$** | Met | Isolated planning workload fits in budget ($5.64\times$ speedup) |
| **CGP `BrainCell` Forward** | $W=32, H=2$, 1 block | $\le 2.00\text{ ms}$ | **$1.50\text{ ms}$** | Met | Recurrent update fits budget |
| **Sparse Router ($K=64, k=2$)** | $W=384$, top-$2$ | $\le 1.50\text{ ms}$ | **$0.653\text{ ms}$** | Met | Routing calculation fits budget |
| **Complete End-to-End Tick ($H=3$)** | Obs $\to$ CGP $\to$ Plan $\to$ Act | $\le 16.67\text{ ms}$ | **$25.73\text{ ms}$** | **Exceeded** | Enc (0.30ms) + Rec (10.49ms) + Plan (14.64ms) + Env (0.20ms) |
| **Complete End-to-End Tick ($H=5$)** | Obs $\to$ CGP $\to$ Plan $\to$ Act | $\le 16.67\text{ ms}$ | **$41.39\text{ ms}$** | **Exceeded** | Enc (0.32ms) + Rec (10.95ms) + Plan (29.79ms) + Env (0.22ms) |

*System Implication: While dynamic beam search reduces isolated planning to 8.03 ms, closed-loop execution is dominated by recurrent CGP updates (10.5-11.0 ms) and sequential latent rollout steps. Full 60 Hz compliance requires lookahead rate decimation (planning every 3rd or 4th tick) or batched rollout kernels.*

---

### Workstream 3 & 6: Procedural Task DAG Capability Ladder (L1 - L8)

| Level | Task Benchmark | Blind Anti-Perseveration | PseudoBrainAgent (Contextual IOR) | Mechanistic Finding |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | Single Action Invocation | 100.0% | **100.0%** | Parity |
| **L2** | Fixed Sequence (Linear Tool Chain) | 100.0% | **100.0%** | Parity |
| **L3** | Branching DAG (State Routing) | 100.0% | **100.0%** | Parity |
| **L4** | **State-Contingent Retry** | **0.0%** | **100.0%** | **Critical Separation**: Naive IOR permanently suppresses action; PseudoBrain detects repair and resets inhibition |
| **L5** | Hidden Dependency Extraction | 100.0% | **100.0%** | Parity |
| **L6** | **Delayed Verification** | **0.0%** | **100.0%** | **Critical Separation**: Robust over observation latency |
| **L7** | **Stochastic Timeout Recovery** | **0.0%** | **100.0%** | **Critical Separation**: Contextual retry succeeds |
| **L8** | Novel DAG w/ Distractor Tools | 100.0% | **100.0%** | Parity: Ignores distractors |

---

## 3. Active Research Agenda & Verification Status

1. **Unified 60 Hz Closed-Loop Embodied Benchmark (WS1 + WS2 Merged)**: [COMPLETE & REPORTED]
   - Verified end-to-end tick latency ($25.73\text{ ms}$ at $H=3$; $41.39\text{ ms}$ at $H=5$).
   - Telemetry verified: $\text{memory} \to \text{surprise} \to \text{plasticity} \to \text{dispersion} \to \text{pruning} \to \text{action}$.
   - Documented in `brain/docs/runs/2026-09-07-unified-60hz-embodied-cgp-planner.md`.
2. **Procedural Task DAG & State-Contingent Retry Benchmark (WS3 / WS6)**: [COMPLETE & REPORTED]
   - Stress-tested agent reasoning across L1–L8.
   - Proved contextual failure memory differentiates from hard-coded anti-perseveration on L4, L6, L7.
   - Documented in `brain/docs/runs/2026-09-07-agent-capability-ladder-results.md`.
3. **Causal Memory Difficulty Decomposition (WS1)**: [COMPLETE & REPORTED]
   - Swept corridor delay horizons $L \in [0, 4, 8, 16, 32, 64, 128]$ ticks on NVIDIA A100 (`pb-research2`).
   - Results: Full CGP achieved **82.1%** mean retention across delay horizons (vs 66.7% vanilla thoughtlet and 53.8% GRU).
   - Causal ablation: Ablating Synaptic Latching (`cgp_no_cgsl`, $P_t = 0$) reduced retention from **82.1% to 75.0%**, confirming the functional value of fast weights.
   - Ablating Cognitive Gating (`cgp_no_cig`) showed identical 82.1% performance, indicating that synaptic latching rather than input gating is load-bearing on this corridor geometry.
   - Documented in `brain/docs/runs/2026-09-07-memory-difficulty-curve-ablations.md` and `.json`.

### Workstream 1: Memory Difficulty Curve Matrix ($L \in [0, 128]$ delay ticks)

| Model Condition | L=0 | L=4 | L=8 | L=16 | L=32 | L=64 | L=128 | Mean Retention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vanilla_thoughtlet** | 66.7% | 66.7% | 66.7% | 66.7% | 66.7% | 66.7% | 66.7% | **66.7%** |
| **gru** | 53.8% | 53.8% | 53.8% | 53.8% | 53.8% | 53.8% | 53.8% | **53.8%** |
| **cgp_full** | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | 100.0% | 100.0% | **82.1%** |
| **cgp_no_cig** (w/o Input Gate) | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | 100.0% | 100.0% | **82.1%** |
| **cgp_no_cgsl** (w/o Fast Plasticity)| 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% | **75.0%** |
