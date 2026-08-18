# Play-gated maze-chase distill campaign v1 (2026-08-18)

Status: **current campaign**. 32-step probe v2 and 128-step probe both
**passed** play at the same numbers: `play_moved: true`, reward_sum
**-150**, collisions 16, **0 pellets**. 4× steps did not add pellets.
Spark is idle. Next bounded idea is a zero-pellet diagnosis, not a
2048-step train. RCQ-v2 seed 1702 stays terminal. No v3 registration. No
sealed TEST. Compute is Spark-only; the workstation is orchestration.

## Probe v2 result (2026-08-18)

Play moved. Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/probe-v2-play-gate.json](./artifacts/play-gated-maze-chase-distill/probe-v2-play-gate.json).

| Field | Value |
|---|---|
| Release | `r20260818t160829z-82aef6dbc370` (archive SHA-256 `82aef6dbc370…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-probe-v2` |
| Canonical config SHA-256 | `f74732be8b03535a31e0fa4178b183283653c4d0eec4117bb384c8dc2916f124` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `e0515628943633f1078eb89ba39424df0566aa4862743e9367352c823e153fb8` |
| `latest.json` SHA-256 | `1020dbadd9dea3ac0f662db745863bc10cb3ecd4f5720c97c29c6c4387da7bce` |
| Metrics SHA-256 | `3608402766c03c601ec8bbea41895d7cfdf0711fc8b3b56e467b98441821d494` (bit-identical to v1 train/val logs) |
| `play-gate.json` SHA-256 | `df9d88f47181f911ea96ae71424ff841e9e55bd291ab861e3f6615ea9f08e4bb` |
| Report SHA-256 | `e9f68c644c7b0609cc2622709fd609b63c333815147444d6b8c1deb76c21636c` |
| `play_moved` | **true** |
| `gate` | `passed` |
| reward_sum | **-150** (floor -161) |
| collisions | **16** (floor 17) |
| pellets_eaten | **0** |
| decisions_rejected | 0 |

This is a thin pass: one fewer collision than no-op across 480 ticks, no
pellets. Logger train/val numbers match v1 (loss 0.402 / 0.726, movement
exact 0.500 / 0.458) and are still not the gate. Next job asks whether a
bounded 128-step train keeps play above the floor and whether pellets
appear. Not a 2048-step scale-up.

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

## 128-step probe result (2026-08-18)

Play held, did not improve. Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/probe-128-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/probe-128-v1-play-gate.json).

| Field | Value |
|---|---|
| Release | `r20260818t162040z-2382206346a0` (archive SHA-256 `2382206346a0…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-probe-128-v1` |
| Canonical config SHA-256 | `267d8e8c99a00a73e2781464f3985ceabadd1933951b2d52669c0b78fcb8e397` |
| Checkpoint | `checkpoints/step-00000128.pt` |
| Checkpoint SHA-256 | `6b0d64c22db576178d8f98a4d9811c37bdd4cea688d121050ec24733764ef7c2` |
| `latest.json` SHA-256 | `d00a23272ff0e77c5b4b51af09f1af4bc78c3a67320535569fe2c072745cad95` |
| Metrics SHA-256 | `a78a3e75c68066a3e88f20797eeeb1e1badd361635157e0141b2b7b4e0275061` |
| `play-gate.json` SHA-256 | `6401a9223032fc30dc6418470fbc15a6fd5f1f45888b6dada938d53babf784a2` |
| Report SHA-256 | `c0b2efb2c9d6414f074fc75c7acf77626a5783cc91d6f5d5d237e117efde7158` |
| `play_moved` | **true** |
| `gate` | `passed` |
| reward_sum | **-150** (same as 32-step) |
| collisions | **16** (same as 32-step) |
| pellets_eaten | **0** |
| decisions_rejected | 0 |

Logged metrics at step 128 (not the gate): train loss 0.438, action loss
0.422, movement exact 0.667; validation loss 0.501, action loss 0.485,
movement exact 0.458. Teacher agreement moved; closed-loop play did not.
Do not treat this as a green light for an unlabeled long train.

## Next bounded idea (not started)

Zero-pellet diagnosis: the model is one collision below no-op and eats
nothing at 32 and 128 steps. Check closed-loop action decode vs the
planner teacher (sticky D? no pellet approach), reset/horizon, and
whether 240 ticks on seeds 5/9 can show pellet play at all for this
checkpoint family. A newly named short Spark probe only after that
hypothesis is written down.

## 128-step probe (preregistered, now completed)

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-probe-128-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-probe-128.toml` |
| Budget | 128 optimizer steps, then seeds 5/9 × 240 ticks |
| Pass | `play-gate.json` with `play_moved: true` and collisions ≤ 17 |

## Probe v1 identities (historical; do not relaunch)

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
| Passing 32-step run id | `dgx-play-maze-chase-distill-probe-v2` (`play_moved: true`) |
| Passing 128-step run id | `dgx-play-maze-chase-distill-probe-128-v1` (`play_moved: true`, same play numbers) |
| Next probe run id | none running; next idea is a zero-pellet diagnosis |
| 32-step config | `brain/configs/training/dgx-play-maze-chase-distill-probe.toml` |
| 128-step config | `brain/configs/training/dgx-play-maze-chase-distill-probe-128.toml` |
| Model factory | `irene_brain.training.factory:build_thesis_model` |
| Data | lazy `irene.maze_chase.planner_teacher.v1` via `dataset.kind = "maze_chase"` |
| Probe budget | 32-step pass done; next bounded budget 128 steps, schema 2, constant after warmup |
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
- **Campaign pass** (later): a preregistered longer run that stays above
  the floor and does not regress collisions above 17. The 128-step probe
  is the next bounded step, not that campaign pass.

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

## Spark sequence (128-step probe, completed)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t162040z-2382206346a0`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-probe-128.toml`, run id
   `dgx-play-maze-chase-distill-probe-128-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` held `play_moved: true` at reward -150 / collisions 16
   / 0 pellets. Spark is idle.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

The B2 world-model-actor manifest was regenerated because `train.py` is in
that family's source set (new digest `3d2e3cc0…`; previous `c207401f…`).
No actor parameter or recipe identity changed.
