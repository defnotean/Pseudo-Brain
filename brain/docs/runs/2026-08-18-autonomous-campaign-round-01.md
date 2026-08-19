# Continuous Autonomous Campaign Round 01 Evidence Record

**Date**: 2026-08-18
**Status**: `NEW_VALIDATED_CHAMPION`
**Model Parameter SHA256 Digest**: `5fa1075c96e9371d`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_01.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 2
- **Sequences in Replay Buffer**: 52
- **Training Loss (Mean)**: `1.2497`
- **Rapid Metric (3-seed)**: Mean `3.3` (Total: `10`) | **Rapid Record**: `3.3`
- **Validated Champion (20-seed)**: Pellets `4.25` | Catches `854`
- **Ghost Catches by Topology**:
  - Corridor Catches: `1`
  - Junction Catches: `4`
  - Dead-End Catches: `6`
  - Total Catches: `11`
- **Wall Interaction Dynamics**:
  - Total Wall Bump Ticks: `341`
  - Distinct Wall Contact Events: `14`
  - Mean Repeated Pushes per Event: `24.7` (Max Streak: `40`)
  - Mean Wall Recovery Latency: `24.7` ticks
- **Nearest Ghost Distance**: Mean `6.82` tiles | Min `1.00` tiles
- **Junction Entries**: Total `11` | Unsafe Crossing Count (ghost dist <= 2): `0`

## Event-Triggered Thought Dynamics (Delta T = 1 - cos(T_t, T_t+1))
- **Normal Movement Step Delta T**: `0.00000`
- **Wall Bump Frame Delta T**: `0.00027`
- **Ghost Danger Frame Delta T (dist <= 2.5)**: `0.00000`

## Expert Disagreement & Outcome Conditioning
- **In-Sample Direct Imitation Match**: `0.0%`
- **Rollout Lookahead Disagreement Rate**: `99.2%` (357 decisions)
  - Productive Disagreements (Pellet Gained + Survived): `6` (1.7%)
  - Benign Safe Disagreements: `240`
  - Fatal Disagreements (Caught): `111`

## Internal Thought-Field Representation Health
- **Effective SVD Rank**: `2.90` / 4 thoughtlets
- **Pairwise Thoughtlet Cosine Similarity**: `0.437`
- **Mean Thoughtlet Norm**: `5.60`
- **Inter-Slot Variance**: `0.4120`
- **Global Temporal Persistence**: `0.9997`

## Invariant Compliance
- Structural Mutual Exclusion ($W+S=0, A+D=0$): PASS
- Deadzone Bounding: PASS
- Single-threaded CPU execution: PASS
- CUDA-hidden compliance: PASS
- Zero-cheating policy compliance: PASS

## Cognitive Depth Scaling Ablation

| Cognitive Cycles | Mean Pellets | Mean Ghost Catches | Effective Thought Rank | Latency (ms/step) |
|---|---|---|---|---|
| **1** | 3.33 | 3.67 | 2.90 | 27.04 ms |
| **2** | 3.33 | 3.67 | 2.90 | 47.37 ms |
| **3** | 3.33 | 3.67 | 2.89 | 68.39 ms |
| **4** | 3.33 | 3.67 | 2.88 | 90.12 ms |
| **6** | 3.33 | 3.67 | 2.87 | 131.29 ms |
