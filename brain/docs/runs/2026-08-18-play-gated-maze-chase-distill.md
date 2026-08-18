# Play-gated maze-chase distill campaign v1 (2026-08-18)

Status: **current campaign**. Probe v1 trained 32 steps then **crashed
before `play-gate.json`**. `play_moved` is missing. Probe v2 is the next
bounded Spark job (same 32-step recipe after the CUDA play-eval fix).
RCQ-v2 seed 1702 stays terminal. No v3 registration. No sealed TEST.
Compute is Spark-only; the workstation is orchestration.

## Probe v1 result (2026-08-18)

Training completed. The play gate did not run. Official `play_moved` is
**missing**, so this is not a scale-up.

| Field | Value |
|---|---|
| Release | `r20260818t155535z-ac06f47b314b` |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-probe-v1` |
| Canonical config SHA-256 | `f74732be8b03535a31e0fa4178b183283653c4d0eec4117bb384c8dc2916f124` |
| Launch `run.env` config SHA-256 | `e5a584bab2c67e09c4a8885313f48da1e96c2e7d3e7345a77e1870f91c572ce3` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `b1fec71a6f4e8924f4226c118911fd7b1ea91930c97058cb3d2f9d8d355ee8ca` |
| `latest.json` SHA-256 | `211dc99734bc260bcb15b0e012a52a09204afe0178cffd54d26b3027b650779d` |
| Metrics SHA-256 | `3608402766c03c601ec8bbea41895d7cfdf0711fc8b3b56e467b98441821d494` |
| `play-gate.json` | **absent** (process exited during closed-loop eval) |

Logged metrics (not the gate):

| Split | Step | Loss | Action loss | Movement exact |
|---|---|---|---|---|
| train | 1 | 2.191 | 0.804 | 0.000 |
| train | 8 | 2.107 | 1.692 | 0.000 |
| train | 16 | 0.857 | 0.566 | 0.000 |
| train | 24 | 0.557 | 0.446 | 0.500 |
| train | 32 | 0.402 | 0.362 | 0.500 |
| validation | 32 | 0.726 | 0.538 | 0.458 |

Validation also shows D predicted-positive 1.0 vs target 0.458
(movement false positives 0.542) and zero W/A/S true positives. That is
teacher-agreement texture, not play.

Crash (after `status: completed` on step 32):

```
RuntimeError: Expected all tensors to be on the same device, but got
mat2 is on cuda:0, different from other tensors on cpu
```

in `ThoughtField.initial_state` → `noise_projection(thought_noise)`.
Cause: `evaluate_closed_loop_play` hardcoded `torch.device("cpu")` while
the trained thesis model stayed on CUDA. Fix: derive the inference
device from `next(model.parameters()).device`, the same pattern as the
RCQ evaluators. Do not resume v1: checkpoint `code_sha256` would
disagree with the fixed tree. Next job is a newly named 32-step probe.

## Probe v2 (preregistered)

Same question, same play floor, same 32-step recipe. New run id only.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-probe-v2` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-probe.toml` (unchanged recipe) |
| Budget | 32 optimizer steps, then seeds 5/9 × 240 ticks |
| Pass | `play-gate.json` with `play_moved: true` (`reward_sum > -161`) |

Live v1 identities (historical; do not relaunch):

- Release `r20260818t155535z-ac06f47b314b` (archive SHA-256 `ac06f47b314b…`)
- Smoke receipt on that release (image `sha256:177a406d7cb2…`)
- Run id `dgx-play-maze-chase-distill-probe-v1`
- Config SHA-256 `f74732be8b03535a31e0fa4178b183283653c4d0eec4117bb384c8dc2916f124`

## Question

Can the existing thesis thought-field learn to **play** maze-chase
closed-loop, measured by play (reward / collisions / pellets), after
distilling the in-repo planner teacher?

A pass is not Minecraft, not desktop play, not architecture superiority,
and not an RCQ qualification.

## Frozen identities

| Field | Value |
|---|---|
| Campaign id | `play_gated_maze_chase_distill_v1` |
| First probe run id | `dgx-play-maze-chase-distill-probe-v1` (trained; play-gate crashed) |
| Next probe run id | `dgx-play-maze-chase-distill-probe-v2` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-probe.toml` |
| Model factory | `irene_brain.training.factory:build_thesis_model` |
| Data | lazy `irene.maze_chase.planner_teacher.v1` via `dataset.kind = "maze_chase"` |
| Probe budget | 32 optimizer steps, schema 2, constant after warmup |
| Play eval | seeds 5/9, 240 ticks, canonical maze slot (3 ghosts, period 2, 16 extra loops) |

## Success / fail / stop

The no-op floor is frozen from the 2026-08-18 constant-LR transfer table
on the same play config: **reward_sum -161, collisions 17, zero pellets**.

- **Play moved** (probe pass, may scale): `reward_sum > -161`.
- **Probe fail**: play at or below that floor, even if action loss drops.
  Action-loss-only improvement is a fail for this campaign.
- **Stop**: if the probe fails, do not start a longer Spark train. Diagnose
  (teacher/closed-loop mismatch, action decode, reset, horizon, twitch)
  and run the next **newly named** bounded probe.
- **Campaign pass** (later, only after play has moved): a preregistered
  longer run that stays above the floor and does not regress collisions
  above 17. Not opened by this probe.

`train.py` writes `play-gate.json` into the run directory after a
maze_chase train or evaluate-only pass. The play gate uses reward_sum
against the no-op floor. Maze `pellet_eaten` stays out of
`targets_collected` so the cross-world no-op floor stays world-flat.

## What this probe will not claim

- RCQ competence or TEST labels
- Architecture superiority against matched baselines
- Physical 60 Hz latency
- Transfer to other ladder worlds (one transfer world waits until play
  has moved and hygiene is tight)

## Spark sequence (this probe)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (new immutable release; do not chmod it)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (required receipt)
4. `Start-DgxBrainTraining.ps1` with this config, run id
   `dgx-play-maze-chase-distill-probe-v2` (v1 already ran), Foreground or
   Tmux with `-AcknowledgeDetached`
5. Read `play-gate.json`. Scale only if `play_moved` is true.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

The B2 world-model-actor manifest was regenerated because `train.py` is in
that family's source set (new digest `3d2e3cc0…`; previous `c207401f…`).
No actor parameter or recipe identity changed.
