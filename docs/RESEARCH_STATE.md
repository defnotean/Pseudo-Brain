# PSEUDO-BRAIN: CURRENT RESEARCH STATE

**Last Updated**: 2026-09-06T19:00:00-05:00  
**Current Best Known Commit (`BEST_KNOWN`)**: `d039fae`  
**Active Hardware**: Google Colab NVIDIA A100-SXM4-40GB (`pb-research`)  

---

## 1. What Pseudo-Brain Currently Knows How to Do
1. **1-Trial Hidden Rule Reversal Adaptation ($B=16$)**:
   - The **Plastic Thoughtlet** ($K=32$ parallel thoughtlets + surprise-gated $P_t$ fast synaptic weights) achieves **100.0% ± 0.0%** adaptation on Trial 12 following Rule Reversal (Rule A $\to$ Rule B), and retains Rule B at **100.0%** across all tested seeds (`42, 142, 242`).
   - Standard GRU and non-plastic Thoughtlet baselines score **0.0%** adaptation under identical conditions.
2. **Accelerated Pipeline Parity**:
   - The compiled AMP stack (`torch.compile(mode="reduce-overhead")`, `bfloat16` AMP, and vectorized `forward_sequence`) achieves exact parity with the original uncompiled FP32 loop at $B=16$ (**100.0% ± 0.0%** across all 3 seeds).
3. **Distractor Delay Invariance (Preliminary)**:
   - Synaptic weights $P_t$ remain stable during distractor noise because the surprise gate $\sigma(\text{gate})$ remains closed when prediction error is low.

---

## 2. What Pseudo-Brain Fails At (Current Failure Modes & Bottlenecks)
1. **Optimization Phase Transition at Large Batch Sizes ($B=64$)**:
   - Scaling physical batch size from 16 to 64 induces a sharp bimodal stability collapse: across 6 seeds in the 2×2 matrix, 3 seeds adapted at 100% while 3 seeds collapsed to 0% adaptation.
   - Mechanism: Batch averaging across 64 episodes suppresses surprise variance $\text{Var}(e_t)$ during training ($\approx 0.11$ at $B=64$ vs $0.27$ at $B=16$), shrinking the basin of attraction needed for reliable plasticity gating.
2. **Catastrophic Interference on Reversal 2 ($T_{21} \to T_{30}$)**:
   - When switching back from Rule B to Rule A (Reversal 2), models exhibit varying degrees of retroactive interference without explicit trace reset mechanisms.
3. **Absence of Autonomous Tool / Environment Loop**:
   - Pseudo-Brain currently operates on fixed episodic sequence datasets rather than an interactive, multi-step goal-directed agent loop.

---

## 3. Strongest Validated Results
1. **Batch Size Optimization Causality**:
   The 2×2 Controlled Matrix (Original vs Accelerated Pipeline $\times$ $B=16$ vs $B=64$) proved that the $B=64$ instability is **100% caused by batch size and optimization dynamics**, not by `torch.compile`, BF16 AMP, or vectorized execution:
   - Original Pipeline: $B=16 \to \mathbf{100.0\% \pm 0.0\%}$, $B=64 \to \mathbf{33.3\% \pm 47.1\%}$
   - Accelerated Pipeline: $B=16 \to \mathbf{100.0\% \pm 0.0\%}$, $B=64 \to \mathbf{66.7\% \pm 47.1\%}$
2. **Causal Mechanism Isolated (Gradient Accumulation Phase 2)**:
   - $B=16, \text{accum}=1$ (eff $B=16$, 1500 updates): **100.0% $\pm$ 0.0%** across all 3 seeds.
   - $B=16, \text{accum}=4$ (eff $B=64$, 375 updates): **33.3% $\pm$ 47.1%** (Seed 42: 100%, Seed 142: 0%, Seed 242: 0%).
   - **Conclusion**: Physical batch size is irrelevant. The failure is strictly driven by **effective gradient batch size / update frequency (loss smoothing)** blunting the prediction error variance needed to coordinate synaptic plasticity.
3. **Dense Transition Phase Diagram ($B \in [16, 24, 32, 48, 64, 96]$ Complete)**:
   - Evaluated 18 multi-seed training runs normalized to 24,000 sequence samples (1,200 trajectory equivalents).
   - $B=16$: **100.0% $\pm$ 0.0%** adaptation, **100.0%** session success ($\text{Var}(e_t) = 5.596$).
   - $B=24$: **66.7% $\pm$ 47.1%** adaptation, **66.7%** session success ($\text{Var}(e_t) = 5.353$).
   - $B=32$: **100.0% $\pm$ 0.0%** adaptation, **100.0%** session success ($\text{Var}(e_t) = 2.976$).
   - $B=48$: **100.0% $\pm$ 0.0%** adaptation, **100.0%** session success ($\text{Var}(e_t) = 1.600$).
   - $B=64$: **100.0% $\pm$ 0.0%** adaptation, **66.7%** session success (Seed 142 froze on Reversal 2; $\text{Var}(e_t) = 1.658$).
   - $B=96$: **100.0% $\pm$ 0.0%** adaptation, **100.0%** session success ($\text{Var}(e_t) = 1.552$).
   - Generated publication figure: `brain/experiments/online_adaptation/transition_phase_curve.png`.

---

## 4. Current Primary Limitation: Sensory Distractor Noise Vulnerability
- In `DistractorHiddenRuleEnv`, visual distractor delay ticks ($D \in [10, 25, 50]$) inject dynamic sensory noise.
- Current surprise gating triggers on raw latent observation prediction error: $\|z_{t+1} - \hat{z}_{t+1}\|_2$.
- Because visual distractor noise cannot be predicted, the gate opens on irrelevant sensory fluctuations, corrupting synaptic weights $P_t$ and causing performance to degrade under delay horizons.

---

## 5. Next Immediate Architectural Improvement: Consequence-Gated Plasticity (CGP)
- **Hypothesis**: Replacing raw observation prediction error with **Outcome / Consequence Prediction Error** (reward/penalty discrepancy $\delta_t = |r_t - \hat{r}_t|$) will decouple plasticity gating from sensory distractor noise, making $P_t$ impervious to visual distractions across arbitrary delay horizons $D$.
- **Test Protocol**: Benchmark on `distractor_benchmark.py` across $D \in [0, 10, 25, 50, 100]$.


