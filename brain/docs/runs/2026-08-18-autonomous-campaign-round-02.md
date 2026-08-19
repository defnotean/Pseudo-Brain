# Continuous Autonomous Campaign Round 02 Evidence Record

**Date**: 2026-08-18
**Status**: `NEW_VALIDATED_CHAMPION`
**Model Parameter SHA256 Digest**: `215be7a8f6e85f0d`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_02.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 4
- **Sequences in Replay Buffer**: 84
- **Training Loss (Mean)**: `1.2593`
- **Rapid Metric (3-seed)**: Mean `4.3` (Total: `13`) | **Rapid Record**: `4.3`
- **Validated Champion (20-seed)**: Pellets `5.20` | Catches `70`
- **Ghost Catches by Topology**:
  - Corridor Catches: `2`
  - Junction Catches: `3`
  - Dead-End Catches: `7`
  - Total Catches: `12`
- **Wall Interaction Dynamics**:
  - Total Wall Bump Ticks: `335`
  - Distinct Wall Contact Events: `14`
  - Mean Repeated Pushes per Event: `24.2` (Max Streak: `40`)
  - Mean Wall Recovery Latency: `24.2` ticks
- **Nearest Ghost Distance**: Mean `6.08` tiles | Min `1.00` tiles
- **Junction Entries**: Total `12` | Unsafe Crossing Count (ghost dist <= 2): `0`

## Event-Triggered Thought Dynamics (Delta T = 1 - cos(T_t, T_t+1))
- **Normal Movement Step Delta T**: `0.00001`
- **Wall Bump Frame Delta T**: `0.00023`
- **Ghost Danger Frame Delta T (dist <= 2.5)**: `0.00000`

## Expert Disagreement & Outcome Conditioning
- **In-Sample Direct Imitation Match**: `0.0%`
- **Rollout Lookahead Disagreement Rate**: `98.1%` (353 decisions)
  - Productive Disagreements (Pellet Gained + Survived): `3` (0.8%)
  - Benign Safe Disagreements: `243`
  - Fatal Disagreements (Caught): `107`

## Internal Thought-Field Representation Health
- **Effective SVD Rank**: `2.94` / 4 thoughtlets
- **Pairwise Thoughtlet Cosine Similarity**: `0.037`
- **Mean Thoughtlet Norm**: `5.63`
- **Inter-Slot Variance**: `0.7166`
- **Global Temporal Persistence**: `0.9998`

## Invariant Compliance
- Structural Mutual Exclusion ($W+S=0, A+D=0$): PASS
- Deadzone Bounding: PASS
- Single-threaded CPU execution: PASS
- CUDA-hidden compliance: PASS
- Zero-cheating policy compliance: PASS

## Cognitive Depth Scaling Ablation

| Cognitive Cycles | Mean Pellets | Mean Ghost Catches | Effective Thought Rank | Latency (ms/step) |
|---|---|---|---|---|
| **1** | 7.00 | 5.00 | 2.94 | 28.14 ms |
| **2** | 4.33 | 4.00 | 2.94 | 45.95 ms |
| **3** | 3.33 | 3.67 | 2.94 | 67.31 ms |
| **4** | 4.33 | 4.00 | 2.93 | 89.78 ms |
| **6** | 3.33 | 3.67 | 2.94 | 128.30 ms |
