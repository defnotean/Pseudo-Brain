# Fly-Inspired Architecture Sketch

**Date:** 2026-08-29 | **Status:** DRAFT

---

## 1. Module Topology (Inspired by Fly Neuropils)

Current VectorizedPseudoBrain has K=32 thoughtlet slots in a flat topology. The fly brain has specialized modules. We propose organizing the 32 slots into 4 functional modules of 8 slots each:

| Module | Fly Analogue | Role |
|---|---|---|
| Sensory Encoding | Lamina / Medulla | Raw input → sparse representation |
| Integration | Central Complex (ring attractor) | Heading/path integration |
| Memory | Mushroom Body (KC→MBON) | Associative memory, past-state recall |
| Action Selection | MBON → Output | Decode thoughtlet state into actions |

## 2. Sparse Coding (KC-Inspired)

Replace dense activations with Kenyon-cell-inspired sparse coding:
- Each thoughtlet applies CRC32-seeded random sparse projection
- Sparsity target: ~2-3% active thoughtlets per module per step
- Deterministic, CPU-only, matches project constraints
- Benefit: lower interference, higher signal-to-noise

## 3. Modulatory Gating (DAN-Inspired)

DAN-like signal broadcasts scalar gating value to target module:
- Gating value from learned reward/prediction-error signal
- Only ~6% of inter-module connections get direct modulatory input
- Implementation: scalar gate × target module output

## 4. Feedback Loop Motifs (KC→DAN→MBON)

Implement fly circuit motifs as architectural priors:
- Sensory encoding → modulatory gating (KC→DAN)
- Modulatory signal → memory module readout (DAN→MBON)
- Structured recurrence replacing ad-hoc recurrence

## 5. Local Learning Rule

Replace backprop for gate training with self-supervised predictive learning:
- Each gate: predict next state from current state + action
- Supervisory signal: allothetic cue (next sensory observation)
- Local rule: prediction_error × pre_activity × post_activity
- No global gradient computation needed

## 6. Expected Benefits

1. Robustness: sparse coding + modular topology → better OOD generalization
2. Interference: module isolation reduces cross-talk
3. Sample efficiency: local learning eliminates full backprop pass
4. Biological plausibility: grounded in known circuit motifs

## 7. Evaluation Plan

Compare fly-inspired vs. canonical VectorizedPseudoBrain on:
- RCQ v2 evaluation (primary metric)
- OOD generalization (perturbed sensory inputs)
- Sensory dropout tolerance (progressive input removal)
- Interference metric (cross-module activation overlap)