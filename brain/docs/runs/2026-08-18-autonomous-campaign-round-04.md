# Continuous Autonomous Campaign Round 04 Evidence Record

**Date**: 2026-08-18
**Status**: `RAPID_IMPROVEMENT_NO_VALIDATION_BEAT`
**Model Parameter SHA256 Digest**: `99f35def624e9ff1`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_04.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 8
- **Sequences in Replay Buffer**: 148
- **Training Loss (Mean)**: `1.2306`
- **Rapid Metric (3-seed)**: Mean `5.0` (Total: `15`) | **Rapid Record**: `5.0`
- **Validated Champion (20-seed)**: Pellets `5.20` | Catches `70`
- **Ghost Catches by Topology**:
  - Corridor Catches: `3`
  - Junction Catches: `2`
  - Dead-End Catches: `6`
  - Total Catches: `11`
- **Wall Interaction Dynamics**:
  - Total Wall Bump Ticks: `335`
  - Distinct Wall Contact Events: `14`
  - Mean Repeated Pushes per Event: `24.3` (Max Streak: `40`)
  - Mean Wall Recovery Latency: `24.3` ticks
- **Nearest Ghost Distance**: Mean `5.83` tiles | Min `1.00` tiles
- **Junction Entries**: Total `10` | Unsafe Crossing Count (ghost dist <= 2): `0`

## Event-Triggered Thought Dynamics (Delta T = 1 - cos(T_t, T_t+1))
- **Normal Movement Step Delta T**: `0.00002`
- **Wall Bump Frame Delta T**: `0.00023`
- **Ghost Danger Frame Delta T (dist <= 2.5)**: `0.00000`

## Expert Disagreement & Outcome Conditioning
- **In-Sample Direct Imitation Match**: `0.0%`
- **Rollout Lookahead Disagreement Rate**: `92.2%` (332 decisions)
  - Productive Disagreements (Pellet Gained + Survived): `1` (0.3%)
  - Benign Safe Disagreements: `223`
  - Fatal Disagreements (Caught): `108`

## Internal Thought-Field Representation Health
- **Effective SVD Rank**: `2.97` / 4 thoughtlets
- **Pairwise Thoughtlet Cosine Similarity**: `-0.000`
- **Mean Thoughtlet Norm**: `5.63`
- **Inter-Slot Variance**: `0.7435`
- **Global Temporal Persistence**: `0.9998`

## Invariant Compliance
- Structural Mutual Exclusion ($W+S=0, A+D=0$): PASS
- Deadzone Bounding: PASS
- Single-threaded CPU execution: PASS
- CUDA-hidden compliance: PASS
- Zero-cheating policy compliance: PASS
