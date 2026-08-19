# Continuous Autonomous Campaign Round 01 Evidence Record

**Date**: 2026-08-18
**Status**: `NEW_VALIDATED_CHAMPION`
**Model Parameter SHA256 Digest**: `ef7a8bc35bdc6939`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_01.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 2
- **Sequences in Replay Buffer**: 52
- **Training Loss (Mean)**: `1.1674`
- **Pellet Yield**: Mean `3.3` (Total: `10`) | **Champion Record**: `3.3`
- **Ghost Catches by Topology**:
  - Corridor Catches: `1`
  - Junction Catches: `4`
  - Dead-End Catches: `6`
  - Total Catches: `11`
- **Wall Bumps (Refused Steps)**: `341`
- **Nearest Ghost Distance**: Mean `6.82` tiles | Min `1.00` tiles
- **Junction Entries**: Total `11` | Unsafe Crossing Count (ghost dist <= 2): `0`
- **Expert Planner Disagreement**:
  - Total Disagreements: `357` (99.2%)
  - Productive Disagreements (Pellet Gained + Survived): `6` (1.7%)
  - Benign Safe Disagreements: `240`
  - Fatal Disagreements (Caught): `111`

## Internal Thought-Field Dynamics
- **Effective Thought Dimensionality (SVD Rank)**: `2.87` / 4 thoughtlets
- **Thoughtlet Variance (Inter-Slot Differentiation)**: `0.4238`
- **Temporal Thought Persistence (Cosine Similarity)**: `0.9998`

## Invariant Compliance
- Structural Mutual Exclusion ($W+S=0, A+D=0$): PASS
- Deadzone Bounding: PASS
- Single-threaded CPU execution: PASS
- CUDA-hidden compliance: PASS
- Zero-cheating policy compliance: PASS

## Cognitive Depth Scaling Ablation

| Cognitive Cycles | Mean Pellets | Mean Ghost Catches | Effective Thought Rank | Latency (ms/step) |
|---|---|---|---|---|
| **1** | 3.33 | 3.67 | 2.87 | 26.19 ms |
| **2** | 4.00 | 89.67 | 2.87 | 45.12 ms |
| **3** | 3.33 | 3.67 | 2.87 | 64.78 ms |
| **4** | 3.33 | 3.67 | 2.88 | 87.23 ms |
| **6** | 3.33 | 3.67 | 2.88 | 128.86 ms |
