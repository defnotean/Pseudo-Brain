# Continuous Autonomous Campaign Round 01 Evidence Record

**Date**: 2026-08-18
**Status**: `NEW_VALIDATED_CHAMPION`
**Model Parameter SHA256 Digest**: `f04baf1265824aa2`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_01.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 2
- **Sequences in Replay Buffer**: 52
- **Training Loss (Mean)**: `1.0657`
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
- **Wall Bump Frame Delta T**: `0.00019`
- **Ghost Danger Frame Delta T (dist <= 2.5)**: `0.00000`

## Expert Disagreement & Outcome Conditioning
- **In-Sample Direct Imitation Match**: `0.0%`
- **Rollout Lookahead Disagreement Rate**: `99.2%` (357 decisions)
  - Productive Disagreements (Pellet Gained + Survived): `6` (1.7%)
  - Benign Safe Disagreements: `240`
  - Fatal Disagreements (Caught): `111`

## Internal Thought-Field Representation Health
- **Effective SVD Rank**: `2.90` / 4 thoughtlets
- **Pairwise Thoughtlet Cosine Similarity**: `0.451`
- **Mean Thoughtlet Norm**: `5.54`
- **Inter-Slot Variance**: `0.3942`
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
| **1** | 6.33 | 3.67 | 2.90 | 26.49 ms |
| **2** | 3.33 | 3.67 | 2.90 | 46.45 ms |
| **3** | 3.33 | 3.67 | 2.91 | 67.22 ms |
| **4** | 3.33 | 3.67 | 2.91 | 83.11 ms |
| **6** | 4.00 | 89.67 | 2.91 | 129.44 ms |
