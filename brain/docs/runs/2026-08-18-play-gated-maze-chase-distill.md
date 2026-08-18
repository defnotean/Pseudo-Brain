# Play-gated maze-chase distill campaign v1 (2026-08-18)

Status: **current campaign**. Exclusive-direction softmax probe
**completed / failed**: histogram **S×480**, 20 pellets, 391 collisions,
reward −3890. Val exclusive-argmax match **0.167** (= teacher S rate);
teacher logit gap **−0.445**. Do not scale. Next GPU probe is
action-only exclusive CE. Named Spark CPU farm jobs run in parallel.
Exclusive-argmax play-decode stays failed sticky S. Window-32 and
episode-windows stay falsified idle no-op. RCQ-v2 seed 1702 stays
terminal. No v3 registration. No sealed TEST.

## Exclusive-direction softmax loss result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/exclusive-ce-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/exclusive-ce-v1-play-gate.json).

The matched-loss hypothesis is **falsified at this budget**. Exclusive
softmax did not make val select the teacher direction. Play is stickier
than decode-only exclusive-argmax (S×480 vs S×478 + A×2) with the same
pellet/collision/reward numbers as that sticky-S floor.

| Field | Value |
|---|---|
| Release | `r20260818t175216z-c804bcd101c1` (archive SHA-256 `c804bcd101c1…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-exclusive-ce-v1` |
| Canonical config SHA-256 | `6419c66f4fff8c8a6105d198beea33cf7f2131aab51ff4e28693243a7148c5f3` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `9cbb6ff81ccd3404cc1985ea91c3982f909516e889d1c579dad8e47912943318` |
| `latest.json` SHA-256 | `b9d36ede307d94a759ffd84513be121c91478710ab3692bd9c35a594622fd7c7` |
| Metrics SHA-256 | `606d8332124b9e56d1dcfb8d0cb385e92a4d5d54fea309d5a229a2884b1a6b34` |
| `play-gate.json` SHA-256 | `173883447e214293df1ab71cabeba787d563271fabe3c571d46816c79c0be2aa` |
| Report SHA-256 | `33de3f98498d966603ba8110e529aa9dfadd8aa2fa3c0f740dca827b36e2f771` |
| `play_moved` | **false** (reward far below the no-op floor) |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−3890** |
| collisions | **391** |
| pellets_eaten | **20** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[4, 480]]** (S×480; mask 4 = S-only) |
| sticky_or_idle | **false** (not idle, not D-only) |

Logged metrics at step 32: train loss 1.911, action loss 1.507, movement
exact 0.0, exclusive-argmax match 0.0, value loss 3.77, S predicted-positive
**1.0**; validation loss 1.722, action loss 1.314, movement exact 0.0,
exclusive-argmax match **0.167**, value loss 3.83. Val teacher mix
W/A/S/D = 0.375 / 0.292 / 0.167 / 0.167. Every val WASD predicted-positive
**0.0**; teacher movement logit gap **−0.445**; inactive movement logit max
**−0.070**. Exclusive-argmax match equals teacher S, so val still ranks S
first among negatives. Do not scale.

## Exclusive-direction softmax loss (preregistered; now completed / failed)

Hypothesis: independent multi-label BCE trained ranking-free WASD logits, so
exclusive-argmax play collapsed to a sticky S among negatives (val
predicted-positive 0.0). Teacher maze-chase movement is one of W/A/S/D, or
idle. Training with softmax / exclusive CE on the four directions matches
the kept `exclusive_argmax_wasd_v1` decode.

Named objective `exclusive_wasd_softmax_v1` (maze_chase only; requires
`exclusive_argmax_wasd_v1` play decode):

- Directed teacher rows: softmax cross-entropy toward the unique WASD key
  (first-on in W/A/S/D order if several are labeled).
- Idle teacher rows: hinge the winning logit strictly below **−4.0**, the
  same margin closed-loop exclusive-argmax uses to stay idle.
- Non-movement buttons keep the frozen support-aware background tail.
- Default `support_aware_calibrated_v1` is unchanged. RCQ-v2 / moving-shapes
  keep that default. Do not retarget `independent_logit_gt_zero_v1` play.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-exclusive-ce-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-ce.toml` |
| Canonical config SHA-256 | `6419c66f4fff8c8a6105d198beea33cf7f2131aab51ff4e28693243a7148c5f3` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Batch-source | same episode-windows teacher as exclusive-argmax |
| Budget | 32 optimizer steps, `sequence_length = 8`, `episode_horizon = 240` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only / one-key sticky |
| Result | S×480; 20 pellets; 391 collisions; val exclusive-argmax match 0.167 (= teacher S); gap −0.445 |
| Stop | Val did not select the teacher direction. Do not scale this loss. |

## Exclusive-argmax play decode result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/exclusive-argmax-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/exclusive-argmax-v1-play-gate.json).

The decode hypothesis is **confirmed for idle**, **falsified for play**.
Independent `logit > 0` was why episode-windows sat idle (all WASD logits
negative). Exclusive argmax pressed a key every tick. Training was
bit-identical to episode-windows (same metrics SHA-256 `67c71b15…`); only
play decode changed. The ranking among negative logits collapsed to **S**.

| Field | Value |
|---|---|
| Release | `r20260818t172702z-e492020c6fca` (archive SHA-256 `e492020c6fca…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-exclusive-argmax-v1` |
| Canonical config SHA-256 | `cecd8f4b59791c5abf79515beb2ef810ea77ab5ba174c33e1886af73ab6204f4` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0) |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `4663253841457f4dbe14e701354a0bd4ed6ec79f297385d18d3978283113fab2` |
| `latest.json` SHA-256 | `69c31a7023648f1b3bb8a747b5e8f212437a62ffc162be9f51f66cdffcbc9d67` |
| Metrics SHA-256 | `67c71b1513c43222e4171ecce5796bb3b69803a224769581e1f09fc660cd4ba7` (identical to episode-windows) |
| `play-gate.json` SHA-256 | `865927debc345c72435fe3eaf136c72d9988242165af13c7409945f583b5b845` |
| Report SHA-256 | `013a2aefbf47fd52ddf9faffe55abad1bd57246e8cfaee11a578561d4c1eaf4d` |
| `play_moved` | **false** (reward far below the no-op floor) |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−3890** |
| collisions | **391** |
| pellets_eaten | **20** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[2, 2], [4, 478]]** (A×2, S×478; mask 4 = S-only) |
| sticky_or_idle | **false** (not idle, not D-only) |

Logged metrics at step 32 match episode-windows: train loss 1.063, action
loss 0.590, movement exact 0.0, value loss 4.51; validation loss 0.945,
action loss 0.544, movement exact 0.0, value loss 3.80. Val teacher mix
W/A/S/D = 0.375 / 0.292 / 0.167 / 0.167. Every WASD predicted-positive
**0.0**; inactive movement logit max **−0.80**. Do not scale that recipe.
Exclusive-direction softmax is recorded above (also failed sticky S).

## Exclusive-argmax play decode (preregistered; now completed / failed)

Hypothesis: idle play is a decode / head / threshold mismatch, not “need
more spawn data.” Closed-loop (and val exact) uses independent `logit > 0`
on WASD. After 32 episode-window steps every WASD predicted-positive was
**0.0** and inactive-movement logit max was **−0.80**, so the player never
pressed a key. Pac-Man-style play needs **exactly one** direction.
**Confirmed for idle:** argmax unstuck the no-op histogram. **Failed for
play:** sticky S, 20 pellets, 391 collisions.

Named decode `exclusive_argmax_wasd_v1`:

- Press the unique WASD argmax (HID 26 / 4 / 22 / 7).
- Ties keep the earliest key in W/A/S/D bit order.
- Idle only when the winning logit is strictly below margin **−4.0**
  (well below the observed −0.80 cluster).
- Non-movement buttons stay on frozen `logit > 0`.
- Default `independent_logit_gt_zero_v1` is unchanged. RCQ moving-shapes
  closed-loop reports keep that default. `exclusive_argmax_wasd_v1` is
  maze_chase-only via `objective.play_decode_kind`.

Training loss stayed independent multi-label. Teacher stayed
`irene.maze_chase.planner_teacher.episode_windows.v1`.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-exclusive-argmax-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-argmax.toml` |
| Canonical config SHA-256 | `cecd8f4b59791c5abf79515beb2ef810ea77ab5ba174c33e1886af73ab6204f4` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0) |
| Batch-source | same episode-windows teacher as the failed later-tick probe |
| Budget | 32 optimizer steps, `sequence_length = 8`, `episode_horizon = 240` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only |
| Result | idle unstuck (S×478 + A×2); 20 pellets; 391 collisions; val predicted-positive 0.0 |
| Stop | Val predicted-positive stayed 0. Do not scale. Exclusive-direction softmax is the next named probe. |

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
| Next probe run id | `dgx-play-maze-chase-distill-exclusive-ce-action-only-v1` |
| Failed exclusive-CE run id | `dgx-play-maze-chase-distill-exclusive-ce-v1` |
| Failed exclusive-argmax run id | `dgx-play-maze-chase-distill-exclusive-argmax-v1` |
| Failed episode-windows run id | `dgx-play-maze-chase-distill-episode-windows-v1` |
| Failed window-32 run id | `dgx-play-maze-chase-distill-window32-v1` (idle no-op; do not retry) |
| 32-step config | `brain/configs/training/dgx-play-maze-chase-distill-probe.toml` |
| 128-step config | `brain/configs/training/dgx-play-maze-chase-distill-probe-128.toml` |
| Window-32 config | `brain/configs/training/dgx-play-maze-chase-distill-window32.toml` |
| Episode-windows config | `brain/configs/training/dgx-play-maze-chase-distill-episode-windows.toml` |
| Exclusive-argmax config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-argmax.toml` |
| Exclusive-CE config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-ce.toml` |
| Action-only exclusive-CE config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-ce-action-only.toml` |
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
- **Exclusive-argmax probe** unstuck idle (S×478 + A×2) but failed
  campaign (20 pellets, 391 collisions). Val predicted-positive stayed
  0.0. Do not scale that recipe.
- **Exclusive-CE probe** kept exclusive-argmax decode and changed only the
  WASD training loss to `exclusive_wasd_softmax_v1`. It failed as S×480
  (20 pellets, 391 collisions, −3890). Val exclusive-argmax match 0.167
  equals teacher S; teacher logit gap −0.445. Do not scale.
- **Action-only exclusive-CE probe** keeps that loss and decode and zeros
  value, world, diversity, and continuous weights so exclusive WASD
  softmax is the only trained term. Sticky D / no-op / sticky-S is fail,
  even if `play_moved` is true.

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

## Spark sequence (exclusive-CE probe, completed; failed campaign)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t175216z-c804bcd101c1`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-exclusive-ce.toml`, run id
   `dgx-play-maze-chase-distill-exclusive-ce-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed campaign: reward −3890 / collisions 391 /
   20 pellets / histogram [[4, 480]] (S-sticky). Val exclusive-argmax
   match 0.167, teacher logit gap −0.445. Do not scale.

## CPU farm (2026-08-18, Spark host)

Named CPU jobs against release `r20260818t175216z-c804bcd101c1` and the
exclusive-CE checkpoint. No GB10. Artifacts:
[artifacts/play-gated-maze-chase-distill/cpu-farm/](./artifacts/play-gated-maze-chase-distill/cpu-farm/).

| Job | Run id | Result |
|---|---|---|
| Planner closed-loop | `play-gated-cpu-planner-v1` | Seed 5 clears at tick 208 (142 pellets, 0 collisions, reward 152). Seed 9 clears at tick 237 (142 pellets, 1 collision, reward 142). Mixed WASD. |
| Teacher WASD hist | `play-gated-cpu-teacher-hist-v1` | Spawn train is D-heavy (D 52.3%, S 37.5%). Episode-windows train is mixed (S 35.2%, A 25.8%, W 21.9%, D 17.2%). Zero idle, zero multi-key. |
| Teacher exclusive | `play-gated-cpu-teacher-exclusive-v1` | 256 episode-window ticks: 251 exclusive WASD, 5 idle, 0 multi-key. |
| Thoughtlet dump | `play-gated-cpu-thoughtlets-v1` | Exclusive-CE ckpt, 8 ticks seed 5 on CPU. Mean thought-attention entropy 3.453. WASD logits stay S-ranked (S ≈ 0.00 to −0.04; others more negative). |
| CPU play-gate | `play-gated-cpu-play-gate-v1` | Running on host while the next GPU probe launches. |

## Action-only exclusive CE (preregistered)

Hypothesis: value/world/diversity terms diluted exclusive WASD softmax so
the head still failed to rank the teacher (gap −0.445; train S
predicted-positive 1.0). Same teacher, decode, and 32-step budget; only
the trained term is exclusive CE.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-exclusive-ce-action-only-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-ce-action-only.toml` |
| Canonical config SHA-256 | `6af0d222e61175421a819360796422e4f0f9ed0f5ec7405c4c4f9666e21fed3a` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Weights | `action_weight=1.0`; value/world/diversity/continuous = 0 |
| Batch-source | same episode-windows teacher as exclusive-CE |
| Budget | 32 optimizer steps, `sequence_length = 8`, `episode_horizon = 240` |
| Stop | Sticky S / idle / D-only is fail. Do not scale if val still does not rank the teacher. |

## Spark sequence (action-only exclusive-CE probe)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1`
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml`
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-exclusive-ce-action-only.toml`, run id
   `dgx-play-maze-chase-distill-exclusive-ce-action-only-v1`, Tmux with
   `-AcknowledgeDetached`
5. Keep named CPU farm jobs on the Spark host while the GB10 trains.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

## Spark sequence (exclusive-argmax probe, completed; failed campaign)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t172702z-e492020c6fca`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-exclusive-argmax.toml`, run id
   `dgx-play-maze-chase-distill-exclusive-argmax-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed campaign: reward −3890 / collisions 391 /
   20 pellets / histogram [[2, 2], [4, 478]] (S-sticky). Idle unstuck.
   Val predicted-positive 0.0. Spark is idle. Do not scale.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

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

The B2 world-model-actor manifest was regenerated because `objective.py` is in
that family's source set (new digest `be571ba4…`; previous `885aff85…`,
then `3d2e3cc0…`, then `c207401f…`).
No actor parameter or recipe identity changed.
The matched-baseline architecture manifest is a new comparison identity
`5e0f2536…` (previous live `eda3cf38…`) for the same source-file reason.
