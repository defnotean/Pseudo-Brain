# Pseudo-Brain Hardware Manifest & DGX Spark Unified Architecture

**Registered Platform**: NVIDIA DGX Spark Unified Compute Platform  
**Amendment Date**: August 19, 2026  
**Reference Decision**: `docs/decisions/2026-08-19-dgx-spark-primary-compute.md`  
**Classification**: ARCHITECTURE SPECIFICATION & DEPLOYMENT MANIFEST  

---

## 1. Primary Compute Platform Specification

The **NVIDIA DGX Spark** is the canonical unified compute platform for all Pseudo-Brain operations:
- Training (DAgger, cognitive auxiliary supervision, multi-seed replication).
- Live Neural Inference & Recurrent State Processing.
- Cognitive Unrolling & Counterfactual Tree Evaluation.
- Episodic & Working Memory Retrieval.
- Future Multimodal Sensor Perception, Tool Decision Making, and Robotics Intelligence.

```text
+--------------------------------------------------------------------------------+
|                   NVIDIA DGX SPARK UNIFIED COMPUTE ENGINE                      |
|                                                                                |
|  +------------------------+  +------------------------+  +------------------+  |
|  | Sensory Patch Encoders |  | Persistent Belief Rec. |  | Thoughtlet Field |  |
|  | (Vision / Audio / IMU) |  | (Unobserved Tracking)  |  | (K=4 -> 32 -> 128)  |
|  +------------------------+  +------------------------+  +------------------+  |
|               |                           |                       |            |
|  +------------------------+  +------------------------+  +------------------+  |
|  | Cognitive Halting (C)  |  | Memory Retrieval (M)   |  | Lookahead Planner|  |
|  | Dynamic Depth C in 1..6|  | Episodic Workspace     |  | Multi-Step Fores.|  |
|  +------------------------+  +------------------------+  +------------------+  |
|                                           |                                    |
|                              +------------------------+                        |
|                              | Actuator Readout Head  |                        |
|                              | 307-Channel HID / PWM  |                        |
|                              +------------------------+                        |
+--------------------------------------------------------------------------------+
                                       | (Control Command Stream)
                                       v
                     +----------------------------------+
                     | Edge / Host Physical Interface   |
                     | - Capture & Serialization        |
                     | - Physical Action Dispatch (HID) |
                     +----------------------------------+
```

---

## 2. Physical Interface Boundaries & Zero-Cheating Invariant

1. **Edge / Host Role**:
   External client workstations, game capture devices, or robotic microcontrollers perform **only physical interface functions**:
   - Frame capture and compression/serialization.
   - Action packet receipt and hardware injection (HID keypresses or motor PWM).
2. **No Secondary AI Brains**:
   Microcontrollers performing high-frequency electrical or joint stabilization do not count as a second AI brain and must not perform cognitive path planning or task decision-making.

---

## 3. Strict Network Latency Accounting Protocol

For distributed or networked topologies (e.g. Host Capture PC $\leftrightarrow$ DGX Spark):

$$\text{End-to-End Latency } = T_{\text{capture}\to\text{serialize}} + T_{\text{host}\to\text{Spark}} + T_{\text{Spark\_inference}} + T_{\text{Spark}\to\text{host}} + T_{\text{dispatch}}$$

### Mandatory Budget Matrix (60 Hz Real-Time Target)

| Sub-Stage | Budget Allocation | Benchmark Measurement Method |
| :--- | :---: | :--- |
| **Host Frame Capture & Serialization** | $\le 2.0\text{ ms}$ | High-resolution timer ($T_1 - T_0$) |
| **Host-to-Spark Network Ingestion** | $\le 2.5\text{ ms}$ | Timestamped TCP/UDP socket roundtrip |
| **DGX Spark Neural Kernel** | $\le 6.0\text{ ms}$ | CUDA Event Timers (`cudaEventElapsedTime`) |
| **DGX Spark Lookahead / Planning** | $\le 2.0\text{ ms}$ | Sub-stage cognitive timer |
| **Spark-to-Host Action Transfer** | $\le 2.5\text{ ms}$ | Timestamped socket response |
| **Host Action Dispatch (HID Injection)** | $\le 1.5\text{ ms}$ | Dispatch driver timer |
| **Total End-to-End Closed-Loop Budget** | **$\le 16.67\text{ ms}$ (60 Hz)** | Wall-clock $T_{\text{submit}} - T_{\text{capture}}$ |

*Rule*: If the network roundtrip is 18 ms, the controller is a 23 ms system and fails the 60 Hz deadline requirement. Real network latency must be reported in all Phase 1 closure evidence.

---

## 4. Scaling Ladder Protocol ($K=4 \to K=128$)

To ensure rigorous, controlled scaling on the DGX Spark platform without premature compute bloat:

| Stage | Thoughtlet Scale ($K$) | Parameter Target | Cognitive Depth ($C$) | Primary Research Focus |
| :---: | :---: | :---: | :---: | :--- |
| **Scale A** | $K=4$ | $\sim 29.7\text{M}$ | $C \in [1, 3]$ | Foundational cognitive dynamics, DAgger stability, selective gating |
| **Scale B** | $K=8$ | $\sim 35.0\text{M}$ | $C \in [1, 3]$ | Multimodal sensor fusion (vision + control discovery) |
| **Scale C** | $K=16$ | $\sim 45.0\text{M}$ | $C \in [1, 4]$ | Episodic memory augmentation & long-horizon occlusion tracking |
| **Scale D** | $K=32$ (Canonical Phase 1) | $\sim 60.0\text{M}$ | $C=3$ | Registered real-time Continuous Sensorimotor Kernel |
| **Scale E** | $K=64$ | $\sim 90.0\text{M}$ | $C \in [1, 6]$ | Multi-agent coordination & rich 3D voxel navigation |
| **Scale F** | $K=128+$ | $\sim 150.0\text{M}+$ | $C \in [1, 6]$ | Language, software tool-use & continuous robotics intelligence |

For every scale $K$, the following metrics must be reported:
- Task performance (pellets, catches, survival).
- Effective thoughtlet rank ($R_{\text{eff}} / K$).
- Thoughtlet pairwise cosine similarity.
- Causal slot utility under targeted ablation.
- FLOPs, CUDA kernel latency, power draw, and VRAM footprint.
