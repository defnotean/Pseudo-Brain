# Play-gated maze-chase distill campaign v1 (2026-08-18)

Status: **current campaign**. First probe is **running** on Spark.
RCQ-v2 seed 1702 stays terminal. No v3 registration. No sealed TEST.
Compute is Spark-only; the workstation is orchestration.

Live probe identities (2026-08-18):

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
| First probe run id | `dgx-play-maze-chase-distill-probe-v1` |
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
   `dgx-play-maze-chase-distill-probe-v1`, Foreground or Tmux with
   `-AcknowledgeDetached`
5. Read `play-gate.json`. Scale only if `play_moved` is true.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

The B2 world-model-actor manifest was regenerated because `train.py` is in
that family's source set (new digest `3d2e3cc0…`; previous `c207401f…`).
No actor parameter or recipe identity changed.
