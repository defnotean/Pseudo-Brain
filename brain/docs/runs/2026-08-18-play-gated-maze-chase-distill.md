# Play-gated maze-chase distill campaign v1 (2026-08-18)

Status: **current campaign**. Episode-window probe **failed**: idle no-op
(`play_moved: false`, `campaign_success: false`, reward **-161**,
collisions 17, **9 pellets**, histogram mask 0 × 480). Spark is idle.
Do not scale. Window-32 stays falsified. RCQ-v2 seed 1702 stays terminal.
No v3 registration. No sealed TEST.

## Episode-window probe result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/episode-windows-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/episode-windows-v1-play-gate.json).

The later-tick / full-episode sampling hypothesis is **falsified** at this
budget. Sampling itself worked: validation teacher D fell from the spawn
snippet's 45.8% to **16.7%**, with W the plurality at **37.5%**. Closed-loop
play is still the no-op floor. All WASD predicted-positive rates stayed
**0.0**; inactive movement logit max **-0.80**. Same idle decode as
window-32, without that probe's value-loss explosion (val value loss 3.80
vs ~85).

| Field | Value |
|---|---|
| Release | `r20260818t170141z-55b96fa56a6e` (archive SHA-256 `55b96fa56a6e…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-episode-windows-v1` |
| Canonical config SHA-256 | `8738d61a34216dc6749919171ad60eaec64c7e7cf6edc80932cde7a1ff296e7b` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `72e42ca0f706b8c0fd170d2dcca3b39e19f2c4e8bdd438cde2ae209ff2b38911` |
| `latest.json` SHA-256 | `1edd611d2a022c9d9817ba385ea7399f0e15e57d4d168b54b23f9ec8cacb6dcf` |
| Metrics SHA-256 | `67c71b1513c43222e4171ecce5796bb3b69803a224769581e1f09fc660cd4ba7` |
| `play-gate.json` SHA-256 | `84331559972d836f0b7c1a2766f87d8936885ff29cc844c1ddde1bddb03a0716` |
| Report SHA-256 | `3dac20c529a1d60396ea05d1598b03921b9b678dad033f515fde29ab5c4607b9` |
| `play_moved` | **false** |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **-161** (floor) |
| collisions | **17** (floor) |
| pellets_eaten | **9** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[0, 480]]** (idle every tick) |
| sticky_or_idle | **true** |

Logged metrics at step 32 (not the gate): train loss 1.063, action loss
0.590, movement exact 0.0, value loss 4.51; validation loss 0.945,
action loss 0.544, movement exact 0.0, value loss 3.80. Val teacher mix
W/A/S/D = 0.375 / 0.292 / 0.167 / 0.167. Do not scale.

## Episode-window probe (preregistered; now completed / failed)

Hypothesis: 8-tick spawn-only snippets never showed pellet-seeking or
corridor choice (planner clears seed 5 at tick 208). Sampling the same
8-tick window length from **across a 240-tick planner trajectory**
exposes later ticks, including pellets and the clear path, without the
value-scale blow-up that falsified window-32. **Falsified at 32 steps:**
the teacher mix unstuck from D, but closed-loop logits stayed below the
`> 0` decode threshold.

Sampling change:

- Spawn-only (`episode_horizon = 0`, default) is unchanged:
  `irene.maze_chase.planner_teacher.v1`, `max_ticks = sequence_length`.
  Historical maze_chase dataset hashes stay byte-identical.
- Positive `episode_horizon` is a named source:
  `irene.maze_chase.planner_teacher.episode_windows.v1` /
  batch source `maze_chase_episode_windows`. The planner rolls through
  the horizon, then a deterministic uniform start picks an 8-tick window.
  Value targets are recomputed on the sliced window (zero-bootstrap at
  the window end) so the 8-tick value scale is preserved.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-episode-windows-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-episode-windows.toml` |
| Canonical config SHA-256 | `8738d61a34216dc6749919171ad60eaec64c7e7cf6edc80932cde7a1ff296e7b` |
| Batch-source SHA-256 | `ad167a9518bbdc948b98e442f24d14fea6c6297a4b1c08f470bd0804ca5a742d` |
| Budget | 32 optimizer steps, `sequence_length = 8`, `episode_horizon = 240` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only |
| Result | idle no-op; 9 pellets; mask 0 × 480 |

## Window-32 probe result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/window32-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/window32-v1-play-gate.json).

The 32-tick teacher-window hypothesis is **falsified** at this budget.
Play is exactly the no-op floor, and the decode histogram is idle, not
sticky D.

| Field | Value |
|---|---|
| Release | `r20260818t164127z-e23fdae191c4` (archive SHA-256 `e23fdae191c4…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-window32-v1` |
| Canonical config SHA-256 | `94829d3d13f66f12dd75f54c87ff57c3eaa19859cbfc1e21bf738f4130316b4e` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `52ac305793b48973cc526c5a680f57f2dc3994a03858661300547c7cd6b9a658` |
| `latest.json` SHA-256 | `384bb193f006edb37433fc60098d9704b564d8dedb4fee3ac9e3709f71e2072d` |
| Metrics SHA-256 | `b1014e6790e7ca02c39a305801b37733665b4e493cf12b0308c5d942eb24a33f` |
| `play-gate.json` SHA-256 | `2cdedaf93e9fa36e05b9b281f9001054fb018249ca5f865721dcd5b0e53f2a4e` |
| Report SHA-256 | `d192b59a56b95ec4c04045b8aefb5ca54b1582a7762dd3ea3e2ade4997477276` |
| `play_moved` | **false** |
| `gate` | **failed** |
| reward_sum | **-161** (floor) |
| collisions | **17** (floor) |
| pellets_eaten | **9** (matches no-op arithmetic; counting is now honest) |
| movement_mask_histogram | **[[0, 480]]** (idle every tick; mask 0 = no WASD) |
| decisions_rejected | 0 |

Logged metrics at step 32 (not the gate): train loss 3.558, action loss
0.523, movement exact 0.0, value loss 29.83; validation loss 9.092,
action loss 0.534, movement exact 0.008, value loss 85.06. Lengthening
the teacher window at the same 32-step budget blew up value targets and
left closed-loop logits below the `> 0` decode threshold, so the player
never pressed a key. Worse than sticky D. Do not scale.

Pellet counting is confirmed: no-op-equivalent play now reports 9
pellets, matching 17×−10 + 9 = −161.

## Zero-pellet diagnosis (2026-08-18)

The 32-step and 128-step probes are the same closed-loop player: reward
**-150**, collisions **16**, JSON `pellets_eaten` **0**. Train loss moved;
play did not. That is not "need more steps."

### 1. The JSON "zero pellets" column is the wrong event

`play-gate.json` mapped `totals["targets_collected"]`, which counts
`target_collected`. maze_chase emits `pellet_eaten`. Historical JSON files
therefore always print 0 pellets even if the player ate some.

Reward arithmetic on the frozen play config (pellet +1, caught −10):

| Policy | collisions | reward | implied pellets |
|---|---|---|---|
| no-op floor | 17 | −161 | **9** (17×−10 + 9) |
| v2 / 128-step neural | 16 | −150 | **10** (16×−10 + 10) |

So the neural player is not pellet-blind in the world; it is one extra
corridor pellet and one fewer catch than no-op. Campaign success is still
eating / clearing, not this.

Play-gate now counts `pellet_eaten` and writes `movement_mask_histogram`
(W=bit0, A=bit1, S=bit2, D=bit3; mask 8 is D-only).

### 2. 240 ticks on seeds 5/9 can show pellet collection

The pixel-only planner on the same canonical slot clears seed 5 at tick
**208** (142 pellets, 0 catches). Seed 9 is in the same 3-seed planner
row that clears 3/3. Lengthening eval ticks is not the next idea.

### 3. The policy is sticky D, not idle, not a player

128-step Spark `metrics.jsonl` (run
`dgx-play-maze-chase-distill-probe-128-v1`): every validation row at
steps 32, 64, 96, and 128 is bit-identical on actions:

| Field | Value (val, all four evals) |
|---|---|
| `movement_d_predicted_positive_rate` | **1.0** |
| `movement_w/a/s_predicted_positive_rate` | **0.0** |
| `movement_d_true_positive_rate` | 0.458 |
| `movement_w/a/s_true_positive_rate` | **0.0** |
| `movement_exact_match` | **0.458** (= teacher D rate) |
| `movement_false_positive_count` | 0.542 |

The teacher is **not** sticky D: val target D is 45.8%, W 20.8%, A 16.7%,
S 16.7%. Closed-loop decode is independent `logit > 0` on WASD (HID 26 / 4
/ 22 / 7, same as `button_support_control_indices`). Always-D plus
`sticky_direction=False` is a wall-hug east / respawn loop. Eval device
was already fixed (probe v1 CUDA/CPU crash). Action mapping is not
swapped.

### 4. Teacher covers pellets; the window does not cover turns

`irene.maze_chase.planner_teacher.v1` labels the planner. Dataset
`max_ticks = sequence_length`, so the 8-tick probe only ever sees ticks
0–7 from spawn. Those snippets include pellet-path actions (W/A/S are in
the val targets) but not the first junction. 4× optimizer steps over the
same 8-tick openings overfit D and froze val exact-match at 0.458.

### Next bounded probe (completed; failed)

Hypothesis: **32-tick teacher windows at the same 32-step budget as v2**
unstick D. **Falsified.** Play is idle no-op (mask 0 × 480). Do not
retry or scale. The next distinct idea (episode-window sampling of 8-tick
windows from 240-tick planner rollouts) is preregistered above.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-window32-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-window32.toml` |
| Budget | 32 optimizer steps, `sequence_length = 32`, seeds 5/9 × 240 ticks |
| Result | `play_moved: false`; 9 pellets; idle histogram |

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
| Next probe run id | none started; episode-windows **failed** idle no-op |
| Failed episode-windows run id | `dgx-play-maze-chase-distill-episode-windows-v1` |
| Failed window-32 run id | `dgx-play-maze-chase-distill-window32-v1` (idle no-op; do not retry) |
| 32-step config | `brain/configs/training/dgx-play-maze-chase-distill-probe.toml` |
| 128-step config | `brain/configs/training/dgx-play-maze-chase-distill-probe-128.toml` |
| Window-32 config | `brain/configs/training/dgx-play-maze-chase-distill-window32.toml` |
| Episode-windows config | `brain/configs/training/dgx-play-maze-chase-distill-episode-windows.toml` |
| Model factory | `irene_brain.training.factory:build_thesis_model` |
| Data | `irene.maze_chase.planner_teacher.episode_windows.v1` via `dataset.kind = "maze_chase"` and `episode_horizon = 240` |
| Probe budget | 32 optimizer steps with `sequence_length = 8` windows drawn from 240-tick planner rollouts; schema 2, constant after warmup |
| Play eval | seeds 5/9, 240 ticks, canonical maze slot (3 ghosts, period 2, 16 extra loops) |

## Success / fail / stop

The no-op floor is frozen from the 2026-08-18 constant-LR transfer table
on the same play config: **reward_sum -161, collisions 17**. Implied
no-op pellets are 9 (reward arithmetic). Historical play-gate JSON
`pellets_eaten: 0` is the wrong event name, not a world fact.

- **Play moved** (thin probe instrument): `reward_sum > -161`. Sticky D
  at **-150** / 16 collisions / ~10 pellets still fails the campaign.
- **Campaign pass**: `pellets_eaten >= 32` and a movement histogram that
  is not idle (mask 0) or D-only (mask 8). A planner-like clear is ~142
  pellets on seed 5.
- **Window-32 probe** failed as idle no-op. Do not retry or scale it.
- **Episode-windows probe** failed as idle no-op after the teacher mix
  unstuck from D. Do not scale it.
- **Sticky D / no-op is fail**, even if `play_moved` is true.

`train.py` writes `play-gate.json` into the run directory after a
maze_chase train or evaluate-only pass. The play gate uses reward_sum
against the no-op floor and now counts `pellet_eaten`. Maze pellets stay
out of `targets_collected` so the cross-world no-op floor stays world-flat.

## What this probe will not claim

- RCQ competence or TEST labels
- Architecture superiority against matched baselines
- Physical 60 Hz latency
- Transfer to other ladder worlds (one transfer world waits until play
  has moved and hygiene is tight)

## Spark sequence (episode-windows probe, completed; failed)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t170141z-55b96fa56a6e`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-episode-windows.toml`, run id
   `dgx-play-maze-chase-distill-episode-windows-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed: reward -161 / collisions 17 / 9 pellets /
   histogram [[0, 480]]. Sampling unstuck teacher D (val D 16.7%, W 37.5%)
   but WASD logits stayed negative. Spark is idle. Do not scale.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

## Spark sequence (window-32 probe, completed; failed)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t164127z-e23fdae191c4`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-window32.toml`, run id
   `dgx-play-maze-chase-distill-window32-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed: reward -161 / collisions 17 / 9 pellets /
   histogram [[0, 480]]. Spark is idle. Do not scale.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

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
