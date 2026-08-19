# Continuous Autonomous Campaign Round 08 Evidence Record

**Date**: 2026-08-18
**Status**: `STABLE_PROGRESSION`
**Model Parameter SHA256 Digest**: `0487e37898c25651`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_08.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 16
- **Sequences in Replay Buffer**: 276
- **Training Loss (Mean)**: `2.4066`
- **Rapid Metric (3-seed)**: Mean `4.0` (Total: `12`) | **Rapid Record**: `5.0`
- **Validated Champion (20-seed)**: Pellets `5.20` | Catches `70`
- **Ghost Catches by Topology**:
  - Corridor Catches: `1`
  - Junction Catches: `4`
  - Dead-End Catches: `7`
  - Total Catches: `12`
- **Wall Interaction Dynamics**:
  - Total Wall Bump Ticks: `336`
  - Distinct Wall Contact Events: `14`
  - Mean Repeated Pushes per Event: `24.3` (Max Streak: `38`)
  - Mean Wall Recovery Latency: `24.3` ticks
- **Nearest Ghost Distance**: Mean `6.43` tiles | Min `1.00` tiles
- **Junction Entries**: Total `10` | Unsafe Crossing Count (ghost dist <= 2): `0`

## Event-Triggered Thought Dynamics (Delta T = 1 - cos(T_t, T_t+1))
- **Normal Movement Step Delta T**: `0.00130`
- **Wall Bump Frame Delta T**: `0.00015`
- **Ghost Danger Frame Delta T (dist <= 2.5)**: `0.00001`

## Expert Disagreement & Outcome Conditioning
- **In-Sample Direct Imitation Match**: `0.0%`
- **Rollout Lookahead Disagreement Rate**: `94.7%` (341 decisions)
  - Productive Disagreements (Pellet Gained + Survived): `7` (2.1%)
  - Benign Safe Disagreements: `227`
  - Fatal Disagreements (Caught): `107`

## Internal Thought-Field Representation Health
- **Effective SVD Rank**: `2.99` / 4 thoughtlets
- **Pairwise Thoughtlet Cosine Similarity**: `-0.122`
- **Mean Thoughtlet Norm**: `5.60`
- **Inter-Slot Variance**: `0.8252`
- **Global Temporal Persistence**: `0.9998`

## Invariant Compliance
- Structural Mutual Exclusion ($W+S=0, A+D=0$): PASS
- Deadzone Bounding: PASS
- Single-threaded CPU execution: PASS
- CUDA-hidden compliance: PASS
- Zero-cheating policy compliance: PASS
