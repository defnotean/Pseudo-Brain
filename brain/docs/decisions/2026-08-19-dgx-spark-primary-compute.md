# Decision Record: Designation of NVIDIA DGX Spark as Primary Compute Platform

**Date**: August 19, 2026  
**Status**: APPROVED & REGISTERED ✅  
**Type**: Formal Roadmap & Hardware Architecture Amendment  
**Scope**: All training, inference, cognitive unrolling, world modeling, memory, future language/tool-use, and robotics intelligence.

---

## 1. Context & Motivation

The original deployment target for Pseudo-Brain during initial Phase 1 specification was registered as a local NVIDIA RTX 5070 (12,227 MiB VRAM). 

As the Pseudo-Brain research roadmap progresses through internal cognition (Phase 2), memory-augmented navigation (Phase 3), rich arcade control (Phase 4), and general-purpose sensorimotor/language/robotics expansion (PLAN.md §43), the cognitive compute requirements scale substantially beyond small experimental configurations:
- Scaling parallel thought fields from $K=32 \to K=64 \to K=128+$.
- Deep persistent belief state filtering and predictive representation.
- Multi-horizon counterfactual world-model unrolling ($H \ge 2$).
- Episodic memory retrieval across long temporal horizons.
- Dynamic cognitive halting and adaptive thought-depth cycles ($C \in [1, 6]$).
- Future multi-modal language/tool-use token generation and real-time robotics perception.

To support this unified scaling trajectory under strict real-time deadlines, the project owner has formally designated the **NVIDIA DGX Spark** as the canonical unified compute platform for Pseudo-Brain.

---

## 2. Hardware Target Transition Summary

| Attribute | Previous Registered Target | New Registered Target |
| :--- | :--- | :--- |
| **Primary Compute Platform** | Local NVIDIA RTX 5070 (12 GB VRAM) | **NVIDIA DGX Spark Unified Compute Platform** |
| **Scope of Compute** | Inference on RTX 5070; training on DGX Spark | **Unified compute**: Training, Inference, Cognitive Unrolling, World Modeling, Memory, Planning, and Actuation on DGX Spark |
| **Interface Role** | Workstation ran full model locally | Workstation / Physical host runs only I/O capture & action dispatch |
| **Phase 1 Hardware Gates** | Reopened for DGX Spark validation | Requires DGX Spark compilation, latency profile, and 1-hour physical deadline test |

---

## 3. "All Compute on the Spark" Architecture Invariant

To preserve the fundamental scientific integrity of Pseudo-Brain as a singular, unified cognitive entity:

### A. Operations Residing Strictly on the DGX Spark:
1. Sensory neural preprocessing and modality patch encoders.
2. Persistent belief state recurrent updates.
3. Thoughtlet updates and cross-thought attention.
4. Recurrent `BrainCell` cognitive cycles ($C=1 \dots 3+$).
5. Adaptive surprise-gated thought updates ($\alpha_t$).
6. Multi-horizon spatial and hazard predictions ($\hat{d}_{t+k}, \hat{r}_{t+k}$).
7. Counterfactual lookahead planning and branch evaluation.
8. Episodic and working memory retrieval and storage.
9. Dynamic cognitive halting allocation.
10. Actuator and structured control decoding.
11. Future language encoder/decoder and tool-decision reasoning.
12. Future high-level robotic sensorimotor cognition.

### B. Physical Interface Boundaries (Host / Edge):
External edge hosts (e.g. Windows workstation, game capture PC, or robotic microcontrollers) perform **only physical interface functions**:
- Frame acquisition / sensor sampling.
- Frame serialization and streaming.
- Action packet receipt and physical dispatch (HID injection or motor PWM).

Microcontrollers performing high-frequency electrical or joint stabilization do **not** count as an AI brain and must not perform cognitive decision-making.

---

## 4. Strict Network Latency Accounting Contract

In networked deployment topologies where the environment resides on a host PC and Pseudo-Brain runs on the DGX Spark:

$$\text{End-to-End Latency } = T_{\text{capture}\to\text{serialize}} + T_{\text{host}\to\text{Spark}} + T_{\text{preprocess}} + T_{\text{inference}} + T_{\text{plan}} + T_{\text{Spark}\to\text{host}} + T_{\text{dispatch}}$$

### Mandatory Latency Reporting Requirements:
1. **Network Transfer Must Count**: A 5 ms neural kernel with an 18 ms network roundtrip is a **23 ms physical controller**, and will be formally audited against the 16.67 ms (60 Hz) deadline as a failing configuration.
2. **Component Sub-Stage Breakdown**: Every Phase 1 benchmark must explicitly report:
   - Host capture & serialization latency.
   - Host-to-Spark transfer latency.
   - Spark neural kernel latency (sensory + belief + cycles + actuator).
   - Spark-to-host action transfer latency.
   - Action dispatch latency.

---

## 5. Impact on Roadmap & Phase Closure Status

1. **No Weakening of Scientific Standards**:
   - The real-time deadline targets remain invariant:
     - Kernel latency: $\text{p99} \le 8.0\text{ ms}$.
     - End-to-end observation-to-submit: $\text{p95} \le 16.67\text{ ms}$ (60 Hz).
     - Continuous real-time deadline misses: $< 0.1\%$ over 1 wall-clock hour ($216,001$ ticks).
2. **Phase 1 Closure Status**:
   - Phase 1 is **OPEN ⏳** pending official DGX Spark benchmark execution and 1-hour physical soak evidence.
   - Historical CPU and RTX 5070 measurements are preserved as baseline reference artifacts in `docs/phase_closure/`.
3. **Phase 2 Cognition Research**:
   - Phase 2 research continues uninterrupted on the CPU/Spark verification harness, focusing on empirical intervention value and selective cognitive control.
