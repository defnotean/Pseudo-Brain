# Continuous Autonomous Campaign Round 03 Evidence Record

**Date**: 2026-08-18
**Status**: `ELEVATED_GHOST_COLLISIONS`
**Model Parameter SHA256 Digest**: `f351946e07a8dc87`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_03.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 6
- **Sequences in Replay Buffer**: 116
- **Training Loss (Mean)**: `1.0698`
- **Rapid Metric (3-seed)**: Mean `4.0` (Total: `12`) | **Rapid Record**: `4.3`
- **Validated Champion (20-seed)**: Pellets `5.20` | Catches `70`
- **Ghost Catches by Topology**:
  - Corridor Catches: `8`
  - Junction Catches: `1`
  - Dead-End Catches: `260`
  - Total Catches: `269`
- **Wall Interaction Dynamics**:
  - Total Wall Bump Ticks: `330`
  - Distinct Wall Contact Events: `6`
  - Mean Repeated Pushes per Event: `55.0` (Max Streak: `94`)
  - Mean Wall Recovery Latency: `55.0` ticks
- **Nearest Ghost Distance**: Mean `2.31` tiles | Min `1.00` tiles
- **Junction Entries**: Total `4` | Unsafe Crossing Count (ghost dist <= 2): `1`

## Event-Triggered Thought Dynamics (Delta T = 1 - cos(T_t, T_t+1))
- **Normal Movement Step Delta T**: `0.00222`
- **Wall Bump Frame Delta T**: `0.00001`
- **Ghost Danger Frame Delta T (dist <= 2.5)**: `0.00000`

## Expert Disagreement & Outcome Conditioning
- **In-Sample Direct Imitation Match**: `0.0%`
- **Rollout Lookahead Disagreement Rate**: `96.9%` (349 decisions)
  - Productive Disagreements (Pellet Gained + Survived): `3` (0.9%)
  - Benign Safe Disagreements: `40`
  - Fatal Disagreements (Caught): `306`

## Internal Thought-Field Representation Health
- **Effective SVD Rank**: `2.94` / 4 thoughtlets
- **Pairwise Thoughtlet Cosine Similarity**: `0.001`
- **Mean Thoughtlet Norm**: `5.65`
- **Inter-Slot Variance**: `0.7474`
- **Global Temporal Persistence**: `0.9998`

## Invariant Compliance
- Structural Mutual Exclusion ($W+S=0, A+D=0$): PASS
- Deadzone Bounding: PASS
- Single-threaded CPU execution: PASS
- CUDA-hidden compliance: PASS
- Zero-cheating policy compliance: PASS
