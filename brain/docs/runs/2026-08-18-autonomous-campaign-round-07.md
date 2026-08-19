# Continuous Autonomous Campaign Round 07 Evidence Record

**Date**: 2026-08-18
**Status**: `ELEVATED_GHOST_COLLISIONS`
**Model Parameter SHA256 Digest**: `bb978c150e8b35c7`
**Checkpoint**: `C:\Users\Demon\OneDrive\Desktop\Projects\Pseudo-Brain\brain\artifacts\checkpoints\curriculum_dagger_round_07.pt`

## Quantitative Physical & Spatial Telemetry
- **DAgger Iterations Completed**: 14
- **Sequences in Replay Buffer**: 244
- **Training Loss (Mean)**: `2.5506`
- **Rapid Metric (3-seed)**: Mean `4.7` (Total: `14`) | **Rapid Record**: `5.0`
- **Validated Champion (20-seed)**: Pellets `5.20` | Catches `70`
- **Ghost Catches by Topology**:
  - Corridor Catches: `4`
  - Junction Catches: `4`
  - Dead-End Catches: `25`
  - Total Catches: `33`
- **Wall Interaction Dynamics**:
  - Total Wall Bump Ticks: `336`
  - Distinct Wall Contact Events: `19`
  - Mean Repeated Pushes per Event: `18.2` (Max Streak: `46`)
  - Mean Wall Recovery Latency: `18.2` ticks
- **Nearest Ghost Distance**: Mean `6.79` tiles | Min `1.00` tiles
- **Junction Entries**: Total `9` | Unsafe Crossing Count (ghost dist <= 2): `0`

## Event-Triggered Thought Dynamics (Delta T = 1 - cos(T_t, T_t+1))
- **Normal Movement Step Delta T**: `0.00047`
- **Wall Bump Frame Delta T**: `0.00052`
- **Ghost Danger Frame Delta T (dist <= 2.5)**: `0.00022`

## Expert Disagreement & Outcome Conditioning
- **In-Sample Direct Imitation Match**: `0.0%`
- **Rollout Lookahead Disagreement Rate**: `98.1%` (353 decisions)
  - Productive Disagreements (Pellet Gained + Survived): `32` (9.1%)
  - Benign Safe Disagreements: `193`
  - Fatal Disagreements (Caught): `128`

## Internal Thought-Field Representation Health
- **Effective SVD Rank**: `2.98` / 4 thoughtlets
- **Pairwise Thoughtlet Cosine Similarity**: `-0.061`
- **Mean Thoughtlet Norm**: `5.60`
- **Inter-Slot Variance**: `0.7810`
- **Global Temporal Persistence**: `0.9995`

## Invariant Compliance
- Structural Mutual Exclusion ($W+S=0, A+D=0$): PASS
- Deadzone Bounding: PASS
- Single-threaded CPU execution: PASS
- CUDA-hidden compliance: PASS
- Zero-cheating policy compliance: PASS
