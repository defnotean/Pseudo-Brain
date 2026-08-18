# Play-gated maze-chase distill campaign v1 (2026-08-18)

Status: **current campaign**. Play-peak / early-stop
(`dgx-play-maze-chase-distill-play-peak-v1`) **completed / passed**:
closed-loop play peaked at step 32 (A×377 + S×103, **38 pellets**, 43
collisions, reward −392) and dropped at step 40 (A×450 + D×30, 15
pellets). The trainer kept the step-32 checkpoint and wrote
`play-gate.json` on that peak (`campaign_success: true`). The 32-step
turn-weighted exclusive CE remains the campaign-pass checkpoint; more
steps still hurt. Do not scale 64 or 128. 128-step turn-weighted
exclusive CE **completed / failed** sticky S (idle×26 + S×454, 20
pellets, 390 collisions, reward −3880, val match 0.0). Exact resume
from the 32-step champion is messy (config identity). The licensed next
GPU job is collision-aware `ghost_hit_penalty_v1` with play-peak, 32
steps (`dgx-play-maze-chase-distill-ghost-hit-v1`). Spark GPU is idle
until that launch. RCQ-v2 seed 1702 stays terminal. No v3 registration.
No sealed TEST.

## Collision-aware ghost-hit penalty (preregistered)

Hypothesis: the 32-step champion is clumsy eating (38 pellets, 43
collisions), not a player. Exact resume from checkpoint `e58f323f…` is
messy because checkpoint identity includes the config hash. A new
bounded 32-step probe keeps turn-weighted exclusive CE (hold ×0.1
unchanged), exclusive-argmax decode, the exclusive-CE aux mix, 90 tiled
windows, and accum 30. The named extra term is
`ghost_hit_penalty_v1`: exclusive-softmax mass on the WASD step that
would land on a currently visible ghost cell, weight 1.0. Play-peak
keeps most pellets, then fewest collisions, and stops on a drop so
training cannot walk through the known step-40 collapse. Goal:
collisions below 43 with pellets still ≥ 32. If pellets collapse toward
~20, do not scale.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-ghost-hit-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-ghost-hit.toml` |
| Canonical config SHA-256 | `a1a15e5702b161c3afcd017c4cf9ca40eeb3408addaa5280942410a53c64dbf8` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1; unchanged) |
| Extra term | `ghost_hit_penalty_v1` weight 1.0 |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (same 90-window manifest) |
| Batch-source | `maze_chase_tiled_windows` |
| Weights | exclusive-CE aux mix (value/world/diversity/continuous restored) |
| Budget | max 32 optimizer steps, play eval every 8, `play_peak_v1` early-stop |
| Play eval | seeds 5/9 × 240 ticks during training; official gate on the kept peak |
| Campaign pass vs champion | collisions < 43 and pellets ≥ 32; histogram not idle / D-only / one-key sticky |

Champion to beat: 38 pellets, 43 collisions, A×377 + S×103, reward
−392, val match 0.25, checkpoint `e58f323f…`. Do not overwrite it
without beating that play row.

## 128-step turn-weighted exclusive CE result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/turn-weighted-128-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/turn-weighted-128-v1-play-gate.json).

More updates of the same recipe **destroyed** the 32-step pass. Pellets
dropped 38 → **20** (sticky-S band, toward 17). The mixed A/S histogram
collapsed to **idle×26 + S×454**. Collisions exploded 43 → **390**
(exclusive-CE sticky-S floor was 391). Reward **−3880**. Val
exclusive-argmax match **0.0** (was chance 0.25; not copy-majority).
`campaign_success: false`. Do **not** scale to 256. Do **not** retune
hold×0.1.

The 128-step run matched the passing 32-step metrics at step 32 (train
loss 1.169, val match 0.25). Exclusive-argmax ranking then collapsed
between train step 40 (0.211) and 48 (0.033). From val step 64 onward
match stayed 0.0 and inactive movement logit max sat near **−5.9**
(step 32 was −0.69). CPU open-loop thoughtlets on the 128-step
checkpoint rank **S×32** on seeds 5/9 (the 32-step checkpoint ranked
A×32). Closed-loop and open-loop now agree on sticky S.

| Field | 32-step (pass) | 128-step (fail) |
|---|---|---|
| pellets_eaten | **38** | **20** |
| histogram | A×377 + S×103 | idle×26 + S×454 |
| collisions | 43 | **390** |
| reward_sum | −392 | **−3880** |
| val exclusive-argmax match | 0.25 | **0.0** |
| `campaign_success` | true | **false** |
| `play_moved` | false | false |
| mazes_cleared | 0 | 0 |

| Field | Value |
|---|---|
| Release | `r20260818t202732z-ab1001f8af37` (archive SHA-256 `ab1001f8af37…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-turn-weighted-128-v1` |
| Canonical config SHA-256 | `7ad447d0e09f304751bcf2337419255b2355875c84b00dc1e682cb13c6fd4ad9` |
| Launch `run.env` config SHA-256 | `c1a2873afd241187e0c3901996ecffca9e5b7de30043e8cbe9268c055be5b7bf` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1; unchanged) |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (90 windows / 3 episodes) |
| Checkpoint | `checkpoints/step-00000128.pt` |
| Checkpoint SHA-256 | `d1feae85b5a13ee36c5c2c899ccac3c53418416d250f7afaf87242bbf970ee28` |
| `latest.json` SHA-256 | `12eea18eae874ed77bbc9a03674815870f5eb12202f47881797ef58d6a5c6968` |
| Metrics SHA-256 | `9e99c263540dd4c73af34e1782c3eeb28fec900bdef2ed19da465ff2c924c6d4` |
| `play-gate.json` SHA-256 | `2c09427c60f354fa6b1b5f76ecbb599c727bfba917aea19202ed6dd277a6cdb8` |
| Report SHA-256 | `372fa0fddbd3bc3f4eda9697e22d0dc6b73e2a7d1d750f5c7a2e46099c1ce1ba` |
| `play_moved` | **false** |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−3880** |
| collisions | **390** |
| pellets_eaten | **20** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[0, 26], [4, 454]]** (idle×26 + S×454) |
| sticky_or_idle | **false** (not idle-only, not D-only; still one-key sticky S) |

Logged metrics at step 128: train loss 1.552, action loss 0.449, movement
exact 0.022, exclusive-argmax match 0.022, value loss (train not the
gate); validation loss 0.525, action loss 0.409, movement exact 0.0,
exclusive-argmax match **0.0**, value loss 1.02. Val teacher mix W/A/S/D
≈ 0.250 / 0.083 / 0.250 / 0.417. Every val WASD predicted-positive
**0.0**; teacher movement logit gap **−0.115**; inactive movement logit
max **−5.94**. Spark GPU is idle. Do not scale.

## Play-peak / early-stop result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/play-peak-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/play-peak-v1-play-gate.json).

Scoring play every 8 steps **did** catch the collapse. The kept peak is
step 32: A×377 + S×103, **38 pellets**, 43 collisions, reward **−392**,
`campaign_success: true`. That play-gate JSON is byte-identical to the
32-step turn-weighted champion (`34af40b0…`; report SHA `7e9d916f…`).
Step 40 dropped to 15 pellets (A×450 + D×30, 18 collisions, −165) and
`play_peak_v1` stopped. Do **not** scale past the peak.

| Step | Pellets | Collisions | Histogram | Gate |
|---|---|---|---|---|
| 8 | 19 | 20 | A×474 + S×4 + D×2 | failed |
| 16 | 15 | 18 | A×476 + D×4 | failed |
| 24 | 20 | 391 | A×2 + S×478 | failed sticky S |
| **32** | **38** | **43** | **A×377 + S×103** | **passed (kept)** |
| 40 | 15 | 18 | A×450 + D×30 | failed; early-stop |

| Field | Value |
|---|---|
| Release | `r20260818t215024z-ad7ce0bfdd99` (archive SHA-256 `ad7ce0bfdd99…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-play-peak-v1` |
| Canonical config SHA-256 | `1d5e29963993766cd945ab344b29d149f47a873057923385251b7af6d6163059` |
| Launch `run.env` config SHA-256 | `560f9fe3e82d015b51dbe30bc500cd0d63179cc28e6db3481da39cc0205de2a9` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1; unchanged) |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (90 windows / 3 episodes) |
| Selected checkpoint | `checkpoints/play-best.pt` from `step-00000032.pt` |
| Selected checkpoint SHA-256 | `5df3a7c1d2a67257481c9990ad8af39703f2a6709cefba25c080fdeb16eb767d` |
| Terminal checkpoint | `checkpoints/step-00000040.pt` |
| Terminal checkpoint SHA-256 | `474c2f1ae04396c0a523135d2548903892da9a8dbc5f94610e0d97981a6b497b` |
| `latest.json` SHA-256 | `27ba663605001b293116d574c7eaf42d237b00ed881b25ca5f9308ee1d2f71d9` |
| Metrics SHA-256 | `ca1e4616ae1a5478d3b5b4b7ac652b3a676e124284a0a9bd0a53b54c247c7339` |
| `play-gate.json` SHA-256 | `34af40b019bc66a80d7204f599b4aa13e7d50db0d4f15e67b1d638f757f28019` |
| `play-best.json` SHA-256 | `56a5e2694413bc1b22def701e8458402ad58129e22798137265c35f83bcc304f` |
| `play-peak.json` SHA-256 | `4986a6ad592d3df5e44a0276c9f71aeeeac3774869171a82d15aae91c47bf323` |
| `play-trace.jsonl` SHA-256 | `0599ee527b4993a71d93c97d61d3529c172dbc735a196f47f774a38b502d56a6` |
| Report SHA-256 | `7e9d916f6f401ef0cd183596ba0f41b2a3d0a9405163c23382c621281b2357ac` |
| `play_moved` | **false** (reward below the no-op floor) |
| `campaign_success` | **true** |
| `gate` | **passed** |
| `stop_reason` | **play_peak_drop** at step 40 |
| reward_sum | **−392** |
| collisions | **43** |
| pellets_eaten | **38** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[2, 377], [4, 103]]** (A×377 + S×103) |
| sticky_or_idle | **false** |

Logged metrics at the kept step 32 match the 32-step champion: train
loss 1.169, action loss 0.564, exclusive-argmax match 0.256; validation
loss 0.928, action loss 0.496, exclusive-argmax match **0.25**. Step 40
train exclusive-argmax match **0.211** (the 128-step log's collapse
start). Spark GPU is idle. Do not scale.

## Play-peak / early-stop (preregistered; now completed / passed)

Hypothesis: more optimizer steps of the passing 32-step turn-weighted
recipe **hurt**. Ranking collapsed train steps 40→48 (exclusive-argmax
match 0.211→0.033) and the 128-step endpoint was sticky S. Keep the
same teacher, loss (hold ×0.1), decode, aux mix, 90 tiled windows, and
accum 30. Score closed-loop play every 8 steps on seeds 5/9, keep the
checkpoint with most pellets then fewest collisions, and stop when
play drops from that peak (`play_peak_v1`: pellet drop of 8, drop into
the 20-pellet sticky band, or one-key WASD). Bound 64, not 128 or 256.
Do not retune hold×0.1.

| Field | Value |
|---|---|
| Release | `r20260818t215024z-ad7ce0bfdd99` (archive SHA-256 `ad7ce0bfdd99…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-play-peak-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-play-peak.toml` |
| Canonical config SHA-256 | `1d5e29963993766cd945ab344b29d149f47a873057923385251b7af6d6163059` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1; unchanged) |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (same 90-window manifest) |
| Batch-source | `maze_chase_tiled_windows` |
| Weights | exclusive-CE aux mix (value/world/diversity/continuous restored) |
| Budget | max 64 optimizer steps, play eval every 8, `play_peak_v1` early-stop |
| Play eval | seeds 5/9 × 240 ticks during training; official gate on the kept peak |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only / one-key sticky |

## Turn-weighted exclusive CE result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/turn-weighted-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/turn-weighted-v1-play-gate.json).

Down-weighting corridor holds (×0.1) on the same 90 tiled windows **did**
beat the pellet floor. Closed-loop play mixed A and S, ate **38 pellets**
(≥ 32), and stayed far from the 391-collision S-sticky explosion
(43 collisions). Val exclusive-argmax match **0.25** is four-way chance,
not copy-majority (teacher A 0.083 / D 0.417). `play_moved` is still
**false**: 43 collisions pulled reward to **−392**, below the −161 no-op
floor. A is dominant (377/480). Every val WASD predicted-positive stayed
**0.0**. The licensed 128-step continuation of this recipe then
**failed** sticky S. Do not retune hold×0.1.

| Field | Value |
|---|---|
| Release | `r20260818t195814z-872818a4fa68` (archive SHA-256 `872818a4fa68…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-turn-weighted-v1` |
| Canonical config SHA-256 | `3af3cd9974020d3b72f202552605fc6b1910a52f3937f06dc69b66286d759f7b` |
| Launch `run.env` config SHA-256 | `eed48dc31fc91a57931015bf365a9ee458f02bf0492f9a271db83efa1d1d4d88` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1) |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (90 windows / 3 episodes) |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `e58f323fb90893c4953b04211002753dd3162f398d1b3b4152390203ba8131cf` |
| `latest.json` SHA-256 | `3fe8cd2fcd259c0343fe44363101d8145f0b72d92bbcc7a8405fb46a34375e6a` |
| Metrics SHA-256 | `5943ee6347c3129a7686acb41d4b0db99d9de592e94a52023e0ddcdc2f07ba0f` |
| `play-gate.json` SHA-256 | `34af40b019bc66a80d7204f599b4aa13e7d50db0d4f15e67b1d638f757f28019` |
| Report SHA-256 | `7e9d916f6f401ef0cd183596ba0f41b2a3d0a9405163c23382c621281b2357ac` |
| `play_moved` | **false** (reward below the no-op floor) |
| `campaign_success` | **true** |
| `gate` | **passed** |
| reward_sum | **−392** |
| collisions | **43** |
| pellets_eaten | **38** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[2, 377], [4, 103]]** (A×377 + S×103) |
| sticky_or_idle | **false** |

Logged metrics at step 32: train loss 1.169, action loss 0.564, movement
exact 0.006, exclusive-argmax match 0.256, value loss 5.89; validation
loss 0.928, action loss 0.496, movement exact 0.0, exclusive-argmax
match **0.25**, value loss 4.15. Val teacher mix W/A/S/D ≈ 0.250 /
0.083 / 0.250 / 0.417. Every val WASD predicted-positive **0.0**;
teacher movement logit gap **−0.139**; inactive movement logit max
**−0.690**. CPU turn-hold on the same 90 tiles: 261/720 change ticks
(36.3%), 82/90 windows have a change. Do not retune hold×0.1.

Closed-loop GIFs of this checkpoint on Spark **CPU** (CUDA hidden; the
128-step GB10 train was left running). Same play-gate decode
`exclusive_argmax_wasd_v1`, 240 ticks, seeds 5 and 9. Combined they
reproduce the gate row (38 pellets, 43 collisions, A×377 + S×103).
The play looks clumsy: A/S sticky-ish, lots of ghost hits, no maze
clear.

| Seed | Pellets | Catches | Histogram | GIF |
|---|---|---|---|---|
| 5 | 21 | 12 | A×188 + S×52 | [turn-weighted-v1-seed5.gif](./artifacts/play-gated-maze-chase-distill/turn-weighted-v1-seed5.gif) |
| 9 | 17 | 31 | A×189 + S×51 | [turn-weighted-v1-seed9.gif](./artifacts/play-gated-maze-chase-distill/turn-weighted-v1-seed9.gif) |

Teacher comparison (planner, not the model):
[planner-teacher-seed5.gif](./artifacts/play-gated-maze-chase-distill/planner-teacher-seed5.gif)
(copy of the 2026-08-17 seed-5 planner clear). Renderer:
`brain/scripts/render_maze_chase_neural_replay.py`.

## Multi-episode tiled tiles result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/multi-episode-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/multi-episode-v1-play-gate.json).

Three 240-tick planner episodes **did mix closed-loop play** (first
non-one-key histogram: W×48 + A×200 + D×232) but did **not** beat
mode-collapse on validation or the pellet floor. Pellets 17 are below
exclusive-argmax's sticky-S 20 and far below 32. Collisions stayed at
the no-op floor (17), not a chase. Val exclusive-argmax match **0.083**
equals teacher A, the same copy-a-constant signature as tiled-windows.
CPU window-majority already showed 79/90 mixed tiles and a nearly
balanced teacher (W 23.9% / A 23.5% / S 24.6% / D 26.9%), so this is
not a coverage miss. Do not scale 90-seq. Next teaching signal is
turn-weighted exclusive CE (hold ×0.1, change ×1.0).

| Field | Value |
|---|---|
| Release | `r20260818t192855z-6d85cc69dd69` (archive SHA-256 `6d85cc69dd69…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-multi-episode-v1` |
| Canonical config SHA-256 | `4231288135f8465008a106f1133a8f1a9321f7ee0a8cf7aa76494a0788c72c54` |
| Launch `run.env` config SHA-256 | `82d8a17475159cf0f027af09ae2ab5bd5bc11e08eabd9bf4040a5bf9cf5bbac6` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (90 windows / 3 episodes) |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `650e7de46c3b9ee4ca638440e88e9edcc77cf52072b54dd2b43991082b079378` |
| `latest.json` SHA-256 | `9c3dc46511914231f64906a05f7b7fa438ee8b35b14ab8fffb411bda4cc75139` |
| Metrics SHA-256 | `09d62434f55004cac3c3923a056c271451672bedb46f92cba9664e847eb59747` |
| `play-gate.json` SHA-256 | `c6367334a1429c8c4d1a6f40a6958de2c9555c0b570b63ca7bd8c6ae3cb2b15f` |
| Report SHA-256 | `1a2e1573d71b09462dc4ef21e933641603bd060b0500f4ade320fba92c3a2fde` |
| `play_moved` | **true** (thin: −153 vs no-op −161) |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−153** |
| collisions | **17** |
| pellets_eaten | **17** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[1, 48], [2, 200], [8, 232]]** (W×48 + A×200 + D×232) |
| sticky_or_idle | **false** (mixed, not idle / D-only; still campaign-fail) |

Logged metrics at step 32: train loss 1.821, action loss 1.218, movement
exact 0.006, exclusive-argmax match 0.294, value loss 5.88; validation
loss 1.648, action loss 1.205, movement exact 0.0, exclusive-argmax
match **0.083**, value loss 4.26. Val teacher mix W/A/S/D ≈ 0.250 /
0.083 / 0.250 / 0.417. Every val WASD predicted-positive **0.0**;
teacher movement logit gap **−0.082**; inactive movement logit max
**−0.562**. Train exclusive-argmax match tracked the batch D rate
(0.294 ≈ 0.306). Do not scale.

## Full-episode tiled update result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/episode-update-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/episode-update-v1-play-gate.json).

Averaging all 30 tiles of one 240-tick episode **did not** unstick play.
The 1:1 teacher is mixed (CPU tiled hist: W 16.3% / A 25.4% / S 27.5% /
D 29.6%), but closed-loop collapsed to spawn-band sticky D with a little
A. Pellets 10 match the original 32/128-step sticky-D band, not the
campaign floor of 32. Val exclusive-argmax match **0.417** equals teacher
D and is not a ranking gain. Do not scale accumulation-30.

| Field | Value |
|---|---|
| Release | `r20260818t190103z-ddf0904b5d81` (archive SHA-256 `ddf0904b5d81…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-episode-update-v1` |
| Canonical config SHA-256 | `b9b7888c194f72d91b6464b6e3e99dc2e52103db35c9a4d441ca69b60ee80c40` |
| Launch `run.env` config SHA-256 | `7fc53427da716009b7b11b0a29f1a6264fdca40204f30c263cda94e852e6fd81` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `ea9bb3ada69e495f12757cc6847e88c62c2228a0b5aeccf2b66b671d0d3eb6e7` |
| `latest.json` SHA-256 | `1a9cbe6b874e59fb375f61921f1e181f98b2e7da0fddbb8058191b5aa29fbe5c` |
| Metrics SHA-256 | `eab3edac495d7d4f06a22881eb37671124534a54f396a336d2659a98b53913c1` |
| `play-gate.json` SHA-256 | `ba622c99426d114a85b02a10afdf65cf0d59321cd1498acce56b9014d718b8ff` |
| Report SHA-256 | `b8b379ad4bfc44ca7e81f61ed6b96b47814e69fdf0e7c4a18bff6be20b0aa6da` |
| `play_moved` | **true** (thin: −150 vs no-op −161) |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−150** |
| collisions | **16** |
| pellets_eaten | **10** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[2, 49], [8, 431]]** (A×49 + D×431; mask 8 = D-only) |
| sticky_or_idle | **false** (not idle, not D-only; still D-dominant sticky) |

Logged metrics at step 32: train loss 2.166, action loss 1.156, movement
exact 0.011, exclusive-argmax match 0.372, value loss 9.92; validation loss
1.306, action loss 1.084, movement exact 0.0, exclusive-argmax match
**0.417**, value loss 1.99. Val teacher mix W/A/S/D ≈ 0.250 / 0.083 /
0.250 / 0.417. Every val WASD predicted-positive **0.0**; teacher
movement logit gap **−0.247**; inactive movement logit max **−1.56**.
Open-loop idle-step thoughtlets rank A (seed 5 A×32; seed 9 A×31 + D×1)
while closed-loop play is sticky D. Do not scale.

## Tiled 1:1 planner windows result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/tiled-windows-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/tiled-windows-v1-play-gate.json).

The 1:1 tiled-window hypothesis is **falsified at this budget**. Consecutive
8-tick windows covering one 240-tick planner episode did not spread the
play histogram or raise val exclusive-argmax match. Play swapped sticky S
for sticky A. Pellets 15 are below exclusive-CE's 20 and far below the
campaign floor of 32. Collisions fell to 18 because A-sticky play sits
near the no-op wall, not because the policy learned. Val match **0.083**
is down from exclusive-CE's 0.167. Do not scale tiled-windows.

| Field | Value |
|---|---|
| Release | `r20260818t184756z-49bac601d0ca` (archive SHA-256 `49bac601d0ca…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-tiled-windows-v1` |
| Canonical config SHA-256 | `2c126296c820830d037c0fc14053f1cbdbd05ec40d218ea2e84975cb02002b83` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `fc7c50cba42e65f8b05c0cae768a63913f375d9966a320aed8ec07ad8f937ebf` |
| `latest.json` SHA-256 | `6545caf27762c9f578b0e8e07489ce6a997e7e3ba80ceaa9cd2315fc3ac9413d` |
| Metrics SHA-256 | `aacc6315d175db148019fee5f22129a5da29ab366674d16f583a8e9b847cded8` |
| `play-gate.json` SHA-256 | `9fac7fd9b25e5f00019ebbd3374ca75c419a1842e2974470186a2f437f3aaa03` |
| Report SHA-256 | `b1128bdf4ee56d384a932d52240396afc715fd608c31235f6587485e4e259742` |
| `play_moved` | **false** |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−165** |
| collisions | **18** |
| pellets_eaten | **15** |
| mazes_cleared | **0** |
| movement_mask_histogram | **[[2, 476], [8, 4]]** (A×476 + D×4; mask 2 = A-only) |
| sticky_or_idle | **false** (not idle, not D-only; still one-key sticky A) |

Logged metrics at step 32: train loss 1.653, action loss 1.250, movement
exact 0.0, exclusive-argmax match 0.0, value loss 3.75; validation loss
2.004, action loss 1.260, movement exact 0.0, exclusive-argmax match
**0.083**, value loss 7.16. Val teacher mix W/A/S/D ≈ 0.250 / 0.083 /
0.250 / 0.417. Every val WASD predicted-positive **0.0**; teacher
movement logit gap **−0.145**; inactive movement logit max **−0.162**.
Step-16 train movement exact 1.0 is one D-only window in the logger, not
the gate. Do not scale.

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
| Passing turn-weighted run id | `dgx-play-maze-chase-distill-turn-weighted-v1` (`campaign_success: true`, 38 pellets) |
| Failed 128-step turn-weighted run id | `dgx-play-maze-chase-distill-turn-weighted-128-v1` (sticky S, 20 pellets; do not scale) |
| Play-peak / early-stop run id | `dgx-play-maze-chase-distill-play-peak-v1` (`campaign_success: true` on kept step 32; early-stop at 40) |
| Failed multi-episode tiled run id | `dgx-play-maze-chase-distill-multi-episode-v1` |
| Failed full-episode tiled-update run id | `dgx-play-maze-chase-distill-episode-update-v1` |
| Failed tiled-window exclusive-CE run id | `dgx-play-maze-chase-distill-tiled-windows-v1` |
| Failed value-only exclusive-CE run id | `dgx-play-maze-chase-distill-exclusive-ce-value-only-v1` |
| Failed exclusive-CE run id | `dgx-play-maze-chase-distill-exclusive-ce-v1` |
| Failed action-only exclusive-CE run id | `dgx-play-maze-chase-distill-exclusive-ce-action-only-v1` |
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
| Value-only exclusive-CE config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-ce-value-only.toml` |
| Tiled-window exclusive-CE config | `brain/configs/training/dgx-play-maze-chase-distill-tiled-windows.toml` |
| Full-episode tiled-update config | `brain/configs/training/dgx-play-maze-chase-distill-episode-update.toml` |
| Multi-episode tiled config | `brain/configs/training/dgx-play-maze-chase-distill-multi-episode.toml` |
| Turn-weighted exclusive-CE config | `brain/configs/training/dgx-play-maze-chase-distill-turn-weighted.toml` |
| 128-step turn-weighted exclusive-CE config | `brain/configs/training/dgx-play-maze-chase-distill-turn-weighted-128.toml` |
| Play-peak / early-stop config | `brain/configs/training/dgx-play-maze-chase-distill-play-peak.toml` |
| Model factory | `irene_brain.training.factory:build_thesis_model` |
| Data | Next probe: same `irene.maze_chase.planner_teacher.tiled_windows.v1` 90-window teacher and `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1), play-peak early-stop, max 64. Passed 32-step turn-weighted used that loss. Failed 128-step turn-weighted of that loss. Failed multi-episode used unweighted exclusive CE on that teacher. Failed episode-update used 30 sequences / accum 30 on one episode. Failed tiled-windows used batch 1 / accum 1. Failed exclusive-CE family used `irene.maze_chase.planner_teacher.episode_windows.v1`. |
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
- **Action-only exclusive-CE probe** zeros value, world, diversity, and
  continuous weights. It failed as idle no-op (mask 0 × 480). Do not scale.
- **Value-only exclusive-CE probe** restores `value_weight=0.1` only. It
  failed as idle no-op (mask 0 × 480; val match 0.0; inactive max ≈ −4.82).
  Value_weight alone is not the idle-margin hack. Do not scale.
- **Tiled-window exclusive-CE probe** kept exclusive softmax, exclusive
  argmax decode, and the exclusive-CE aux mix that stayed above idle
  (value/world/diversity/continuous). The variable was consecutive 8-tick
  windows covering one 240-tick planner episode 1:1. It failed sticky A
  (A×476 + D×4, 15 pellets, 18 collisions, −165). Val exclusive-argmax
  match 0.083, down from exclusive-CE 0.167. Do not scale.
- **Full-episode tiled-update probe** kept that tiled teacher and loss
  with `gradient_accumulation_steps = 30` so each optimizer update saw
  all 30 tiles of one 240-tick episode. It failed sticky D (D×431 + A×49,
  10 pellets, 16 collisions, −150). Val exclusive-argmax match 0.417
  equals teacher D. Do not scale accumulation-30.
- **Multi-episode tiled probe** keeps exclusive softmax, exclusive-argmax
  decode, the exclusive-CE aux mix, and accum 30. The variable is 90
  tiled windows covering three planner episodes so each update can mix
  openings. Closed-loop BC at spawn is not this probe: off-policy planner
  labels on idle/W/A/D rollouts were S×32. Sticky one-key / idle is fail,
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

Named CPU jobs. No GB10. Artifacts:
[artifacts/play-gated-maze-chase-distill/cpu-farm/](./artifacts/play-gated-maze-chase-distill/cpu-farm/).

| Job | Run id | Result |
|---|---|---|
| Planner closed-loop | `play-gated-cpu-planner-v1` | Seed 5 clears at tick 208 (142 pellets, 0 collisions, reward 152). Seed 9 clears at tick 237 (142 pellets, 1 collision, reward 142). Mixed WASD. |
| Planner seeds 100–115 | `play-gated-cpu-planner-seeds-v1` | 4/16 clear inside 240 ticks (seeds 105/107/108/111). All 16 mixed WASD and positive (114–142 pellets). 240 ticks is tight; the older robustness row used a longer horizon. |
| Teacher WASD hist | `play-gated-cpu-teacher-hist-v1` | Spawn train is D-heavy (D 52.3%, S 37.5%). Episode-windows train is mixed (S 35.2%, A 25.8%, W 21.9%, D 17.2%). Zero idle, zero multi-key. |
| Teacher exclusive | `play-gated-cpu-teacher-exclusive-v1` | 256 episode-window ticks: 251 exclusive WASD, 5 idle, 0 multi-key. |
| Uniform coverage | `play-gated-cpu-coverage-v1` | 16 train sequences = 16 different episodes × one 8-tick window each (3.3% of 240). Starts scatter (37, 173, 53, 230, …). Not 1:1 with a play-eval trajectory. |
| Thoughtlet dump | `play-gated-cpu-thoughtlets-v1` | Exclusive-CE ckpt, 8 ticks seed 5 on CPU. Mean thought-attention entropy 3.453. WASD logits stay S-ranked (S ≈ 0.00 to −0.04; others more negative). |
| Thoughtlet dump 32 | `play-gated-cpu-thoughtlets-long-v1` | Exclusive-CE ckpt, 32 ticks seeds 5 and 9. Argmax S×32 on both seeds from tick 0. Sticky S is not a later-tick collapse on that checkpoint. |
| CPU play-gate | `play-gated-cpu-play-gate-v1` | Finished. Same campaign numbers as the GPU exclusive-CE gate: S×480, 20 pellets, 391 collisions, reward −3890. Report SHA `ce7911a5…` (CPU path). |
| Planner seeds 116–131 | `play-gated-cpu-planner-seeds-v2` | 4/16 clear inside 240 ticks (seeds 122/126/127/130). All 16 mixed WASD and positive (108–142 pellets). Same 240-tick tightness as 100–115. |
| Tiled coverage | `play-gated-cpu-tiled-coverage-v1` | Confirmed: 30 windows, starts 0,8,…,232, one episode, coverage 1.0. Teacher `tiled_windows.v1`. Tiled sampling is 1:1; the failed GPU still updated one window at a time. |
| Thoughtlet dump tiled | `play-gated-cpu-thoughtlets-tiled-v1` | Tiled-windows ckpt, 32 ticks seeds 5 and 9 on CPU idle-step. Argmax D×32 on both seeds from tick 0. Closed-loop play was sticky A; open-loop idle ranks D. |
| Planner seeds 132–147 | `play-gated-cpu-planner-seeds-v3` | 5/16 clear inside 240 ticks (seeds 132/137/142/143/144). All 16 mixed WASD and positive (101–142 pellets, rewards 62–152). |
| Tiled teacher hist | `play-gated-cpu-tiled-hist-v1` | One 240-tick tiled train episode is mixed: W 16.3%, A 25.4%, S 27.5%, D 29.6%, 3 idle, 0 multi-key. Manifest `a4dd4529…`. Sticky play is not the teacher. |
| Multi-episode coverage | `play-gated-cpu-multi-episode-coverage-v1` | 90 windows cover three episodes 1:1 (seeds 0/1/2, coverage 1.0 each). Teacher `tiled_windows.v1`. Manifest `e3970174…`. |
| Off-policy teacher | `play-gated-cpu-offpolicy-teacher-v1` | On seeds 5/9, idle/W/A/D 32-tick rollouts label **S×32**. Only sticky S mixes (seed 5 D+S; seed 9 W+S+D). Closed-loop BC at spawn would teach sticky S. |
| Thoughtlet dump episode-update | `play-gated-cpu-thoughtlets-episode-update-v1` | Episode-update ckpt, 32 ticks seeds 5 and 9 on CPU idle-step. Argmax A×32 / A×31+D×1. Closed-loop play was sticky D; open-loop idle ranks A. |
| Window majority | `play-gated-cpu-window-majority-v1` | 90 tiled 8-tick windows (three episodes): **79 mixed**, **10 pure one-key**, mean majority fraction **0.614**. Majority-key histogram W×21 / A×23 / S×20 / D×26. Artifact SHA-256 `b3ade679…`. Exclusive-CE collapse is not “windows are one-key corridors.” |
| Turn/hold audit | `play-gated-cpu-turn-hold-v1` | Same 90 tiled windows: **261/720 change ticks** (36.3%), **459 holds**, 8 idle, **82/90 windows have a change**. Artifact SHA-256 `84099816…`. Turn-weighted CE had signal; the 32-step pass used it. |
| Thoughtlet dump multi-episode | `play-gated-cpu-thoughtlets-multi-episode-v1` | Multi-episode ckpt, 32 ticks seeds 5 and 9 on CPU idle-step. Argmax **A×31 + D×1** both seeds. Closed-loop play was mixed W/A/D; open-loop idle ranks A. Artifact SHA-256 `cc104343…`. |
| Thoughtlet dump turn-weighted | `play-gated-cpu-thoughtlets-turn-weighted-v1` | 32-step turn-weighted ckpt, 32 ticks seeds 5 and 9 on CPU idle-step. Argmax **A×32** both seeds. Closed-loop play mixed A×377 + S×103; open-loop idle ranks A. Artifact SHA-256 `deaa5526…`. |
| Thoughtlet dump turn-weighted 128 | `play-gated-cpu-thoughtlets-turn-weighted-128-v1` | 128-step turn-weighted ckpt, 32 ticks seeds 5 and 9 on CPU idle-step. Argmax **S×32** both seeds. Closed-loop play is sticky S (S×454 + idle×26); open-loop now agrees. Artifact SHA-256 `1e13d99c…`. |

## Action-only exclusive CE result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/exclusive-ce-action-only-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/exclusive-ce-action-only-v1-play-gate.json).

Zeroing value/world/diversity/continuous **did not** make exclusive CE
rank the teacher. It drove every WASD logit below the −4.0 idle margin
(inactive max ≈ −4.81). Play is the no-op floor.

| Field | Value |
|---|---|
| Release | `r20260818t181123z-a3164aca95ef` (archive SHA-256 `a3164aca95ef…`) |
| Run id | `dgx-play-maze-chase-distill-exclusive-ce-action-only-v1` |
| Canonical config SHA-256 | `6af0d222e61175421a819360796422e4f0f9ed0f5ec7405c4c4f9666e21fed3a` |
| Checkpoint SHA-256 | `98c8dd831efb73daaddb93cf468d7684732f4d6e0b26303a240c107df467cca9` |
| `latest.json` SHA-256 | `ff0f12151979f4b5450729faa0d6bb2c99de2d8e918c101cf343e327e0706852` |
| Metrics SHA-256 | `13cf3e12b86111a0c06ee256ddcc9dec8a2b5ae1f85928ca6c7f8129c1ed08d4` |
| `play-gate.json` SHA-256 | `114a3aec3fb0364cc469eb7693fc9df3d8bbae7552f9ba3f1b33dac2848b5edf` |
| Report SHA-256 | `32725e929cc818810874acce1a1b415f16968d8e4bbbd4fa5fd67d0d707e5451` |
| `play_moved` | **false** |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−161** |
| collisions | **17** |
| pellets_eaten | **9** |
| movement_mask_histogram | **[[0, 480]]** (idle) |
| sticky_or_idle | **true** |

Val exclusive-argmax match **0.0**; teacher logit gap **−0.526**. Do not scale.

## Value-only exclusive CE result (2026-08-18)

Official `play-gate.json`:
[artifacts/play-gated-maze-chase-distill/exclusive-ce-value-only-v1-play-gate.json](./artifacts/play-gated-maze-chase-distill/exclusive-ce-value-only-v1-play-gate.json).

Restoring only `value_weight=0.1` **did not** keep exclusive-CE logits
above the −4.0 idle margin. Play is the same no-op floor as action-only.
Val exclusive-argmax match stayed **0.0** (it did not rise from exclusive-CE's
0.167), so a longer exclusive-CE train is not licensed.

| Field | Value |
|---|---|
| Release | `r20260818t181900z-85606667a9fa` (archive SHA-256 `85606667a9fa…`) |
| Container image | `sha256:177a406d7cb2…` |
| Run id | `dgx-play-maze-chase-distill-exclusive-ce-value-only-v1` |
| Canonical config SHA-256 | `868e067ac061f49334077080331ee710d003bacef2325a355e2000766368158d` |
| Checkpoint | `checkpoints/step-00000032.pt` |
| Checkpoint SHA-256 | `6784d46e19b5c0f1cd825e4efeaa5ded0b54d3523cd7020f8ec3fa66cfdace9c` |
| `latest.json` SHA-256 | `ed4cb1923706f32386d92a40bd71719e2c7991eb5e59edf5e82e176395934e9e` |
| Metrics SHA-256 | `681a8d24f5ff75ded737ad5e6c3c08e8701e776e438951307480f06c6b6c4c3c` |
| `play-gate.json` SHA-256 | `7dc8959cd8a0c1f7da2e926d9ba06bc360f37f15054e0575d04aa00838cbc373` |
| Report SHA-256 | `2c4f65f3e396c4b74a249557333aaebc195ab16d0e8c52b6e891018cd8ec7072` |
| `play_moved` | **false** |
| `campaign_success` | **false** |
| `gate` | **failed** |
| reward_sum | **−161** |
| collisions | **17** |
| pellets_eaten | **9** |
| movement_mask_histogram | **[[0, 480]]** (idle) |
| sticky_or_idle | **true** |

Logged metrics at step 32: train loss 1.982, action loss 1.259, movement
exact 0.0, exclusive-argmax match 0.0, value loss 7.23; validation loss
1.635, action loss 1.156, movement exact 0.0, exclusive-argmax match
**0.0**, value loss 4.79. Val teacher mix W/A/S/D = 0.375 / 0.292 /
0.167 / 0.167. Inactive movement logit max **−4.82**; teacher logit gap
**−0.361**. Do not scale. Do not chase `value_weight`.

## Value-only exclusive CE (preregistered; now completed / failed)

Hypothesis: auxiliary value/world/diversity terms are what kept
exclusive-CE play above the idle margin (sticky S instead of no-op).
Restore only `value_weight=0.1`. **Falsified:** idle no-op, same as
action-only. The idle-margin mix that stayed above −4 was
world/diversity/continuous together with value, not value alone.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-exclusive-ce-value-only-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-exclusive-ce-value-only.toml` |
| Canonical config SHA-256 | `868e067ac061f49334077080331ee710d003bacef2325a355e2000766368158d` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Weights | `action_weight=1.0`, `value_weight=0.1`; world/diversity/continuous = 0 |
| Budget | 32 optimizer steps |
| Result | idle no-op; 9 pellets; val match 0.0; inactive max ≈ −4.82 |

## Tiled 1:1 planner windows (preregistered; now completed / failed)

Hypothesis: uniform 8-tick lottery never shows a full planner trajectory
(CPU coverage: 16 episodes × 8 ticks = 3.3% of 240). Cycle supervision
and label-smoothing-off are already the exclusive-CE loss. Teach the
planner 1:1 with consecutive non-overlapping windows of one 240-tick
episode (`window_sampling = "tiled"`, 30 train sequences = 240/8). Keep
exclusive softmax, exclusive-argmax decode, and the exclusive-CE aux mix
that stayed above idle (value 0.1 / world 0.1 / diversity 0.05 /
continuous 0.25). Not a step-count scale-up of failed exclusive-CE.
**Falsified:** sticky A, 15 pellets, val match 0.083.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-tiled-windows-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-tiled-windows.toml` |
| Canonical config SHA-256 | `2c126296c820830d037c0fc14053f1cbdbd05ec40d218ea2e84975cb02002b83` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` |
| Batch-source | `maze_chase_tiled_windows` |
| Weights | exclusive-CE aux mix (value/world/diversity/continuous restored) |
| Budget | 32 optimizer steps, `sequence_length = 8`, `episode_horizon = 240`, `train_sequences = 30` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only / one-key sticky |
| Result | A×476 + D×4; 15 pellets; 18 collisions; reward −165; val match 0.083 |

## Full-episode tiled update (preregistered; now completed / failed)

Hypothesis: tiled-windows still updated on **one** 8-tick window
(`batch_size = 1`, `gradient_accumulation_steps = 1`), so 32 steps can
overfit the current corridor direction and collapse to sticky A. Keep
the same 30-tile teacher, exclusive softmax, exclusive-argmax decode,
and exclusive-CE aux mix. Set `gradient_accumulation_steps = 30` so each
optimizer update averages all 30 tiles of the 240-tick episode. Same
32-update budget, not a step-count scale-up.
**Falsified:** sticky D, 10 pellets, val match 0.417 = teacher D.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-episode-update-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-episode-update.toml` |
| Canonical config SHA-256 | `b9b7888c194f72d91b6464b6e3e99dc2e52103db35c9a4d441ca69b60ee80c40` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (same manifest as tiled-windows) |
| Batch-source | `maze_chase_tiled_windows` |
| Weights | exclusive-CE aux mix (value/world/diversity/continuous restored) |
| Budget | 32 optimizer steps, `batch_size = 1`, `gradient_accumulation_steps = 30` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only / one-key sticky |
| Result | D×431 + A×49; 10 pellets; 16 collisions; reward −150; val match 0.417 |

## Multi-episode tiled tiles (preregistered; now completed / failed)

Hypothesis: one planner episode still has a dominant opening, so accum-30
over those 30 tiles collapsed to sticky D (val match = teacher D).
Closed-loop BC at spawn is not the next idea: off-policy planner labels
on idle/W/A/D 32-tick rollouts were S×32. Keep exclusive softmax,
exclusive-argmax decode, the exclusive-CE aux mix, and
`gradient_accumulation_steps = 30`. Set `train_sequences = 90` so tiled
windows cover three 240-tick episodes 1:1 and each update can mix
openings. Not an accumulation scale-up.
**Falsified as a campaign pass:** mixed W/A/D play, 17 pellets, val
match 0.083 = teacher A. Do not scale 90-seq.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-multi-episode-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-multi-episode.toml` |
| Canonical config SHA-256 | `4231288135f8465008a106f1133a8f1a9321f7ee0a8cf7aa76494a0788c72c54` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_v1` |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (90 windows / 3 episodes) |
| Batch-source | `maze_chase_tiled_windows` |
| Weights | exclusive-CE aux mix (value/world/diversity/continuous restored) |
| Budget | 32 optimizer steps, `batch_size = 1`, `gradient_accumulation_steps = 30`, `train_sequences = 90` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only / one-key sticky |
| Result | W×48 + A×200 + D×232; 17 pellets; 17 collisions; reward −153; val match 0.083 = teacher A |

## Turn-weighted exclusive CE (preregistered; now completed / passed)

Hypothesis: 32-step exclusive CE copies a constant key even when the
teacher is mixed, because corridor holds dominate the window. CPU
window-majority: 79/90 mixed, mean majority 0.614, tick mix already
near-uniform. Multi-episode mixed closed-loop play but val match still
equals one teacher key (A at 0.083). Keep the same 90 tiled windows,
accum 30, exclusive-argmax decode, and exclusive-CE aux mix. Switch the
loss to `exclusive_wasd_softmax_turn_weighted_v1`: teacher direction
*changes* keep weight 1.0; holds are ×0.1. Teaching-signal change, not
more episodes.
**Passed as a campaign pass:** A×377 + S×103, 38 pellets, 43 collisions,
val match 0.25 (chance, not copy-majority). Do not retune hold×0.1.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-turn-weighted-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-turn-weighted.toml` |
| Canonical config SHA-256 | `3af3cd9974020d3b72f202552605fc6b1910a52f3937f06dc69b66286d759f7b` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1) |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (same 90-window manifest as multi-episode) |
| Batch-source | `maze_chase_tiled_windows` |
| Weights | exclusive-CE aux mix (value/world/diversity/continuous restored) |
| Budget | 32 optimizer steps, `batch_size = 1`, `gradient_accumulation_steps = 30`, `train_sequences = 90` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only / one-key sticky |
| Result | A×377 + S×103; 38 pellets; 43 collisions; reward −392; val match 0.25 |

## 128-step turn-weighted exclusive CE (preregistered; completed / failed)

Hypothesis: 32-step turn-weighted exclusive CE already passed pellets ≥ 32
with a mixed A/S histogram and collisions far below the 391 S-sticky
explosion. Val match is chance (0.25), not copy-majority. A longer
bounded train of the **same** recipe tests whether more updates raise
pellets, spread keys, and cut collisions without retuning hold×0.1.
**Failed sticky S:** idle×26 + S×454, 20 pellets, 390 collisions,
reward −3880, val match 0.0. Do not scale 128.

| Field | Value |
|---|---|
| Run id | `dgx-play-maze-chase-distill-turn-weighted-128-v1` |
| Config | `brain/configs/training/dgx-play-maze-chase-distill-turn-weighted-128.toml` |
| Canonical config SHA-256 | `7ad447d0e09f304751bcf2337419255b2355875c84b00dc1e682cb13c6fd4ad9` |
| Play decode | `exclusive_argmax_wasd_v1` (idle margin −4.0; kept) |
| Action loss | `exclusive_wasd_softmax_turn_weighted_v1` (hold ×0.1; unchanged) |
| Teacher | `irene.maze_chase.planner_teacher.tiled_windows.v1` (same 90-window manifest) |
| Batch-source | `maze_chase_tiled_windows` |
| Weights | exclusive-CE aux mix (value/world/diversity/continuous restored) |
| Budget | 128 optimizer steps, `batch_size = 1`, `gradient_accumulation_steps = 30`, `train_sequences = 90` |
| Play eval | seeds 5/9 × 240 ticks; honest `pellet_eaten`; WASD histogram |
| Campaign pass | `pellets_eaten >= 32` and histogram not idle / D-only / one-key sticky |
| Result | **failed** sticky S: idle×26 + S×454; 20 pellets; 390 collisions; reward −3880; val match 0.0 |

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t181900z-85606667a9fa`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml`
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-exclusive-ce-value-only.toml`, run id
   `dgx-play-maze-chase-distill-exclusive-ce-value-only-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed idle no-op: reward −161 / collisions 17 /
   9 pellets / histogram [[0, 480]]. Val match 0.0. Do not scale.

## Spark sequence (tiled-window exclusive-CE probe, completed; failed sticky A)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t184756z-49bac601d0ca`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml`
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-tiled-windows.toml`, run id
   `dgx-play-maze-chase-distill-tiled-windows-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed sticky A: reward −165 / collisions 18 /
   15 pellets / histogram [[2, 476], [8, 4]]. Val match 0.083. Do not scale.

## Spark sequence (full-episode tiled-update probe, completed; failed sticky D)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t190103z-ddf0904b5d81`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml`
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-episode-update.toml`, run id
   `dgx-play-maze-chase-distill-episode-update-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed sticky D: reward −150 / collisions 16 /
   10 pellets / histogram [[2, 49], [8, 431]]. Val match 0.417 = teacher D.
   Do not scale accumulation-30.

## Spark sequence (multi-episode tiled probe, completed; failed mixed W/A/D)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t192855z-6d85cc69dd69`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml`
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-multi-episode.toml`, run id
   `dgx-play-maze-chase-distill-multi-episode-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed mixed W/A/D: reward −153 / collisions 17 /
   17 pellets / histogram [[1, 48], [2, 200], [8, 232]]. Val match
   0.083 = teacher A. Do not scale 90-seq.

## Spark sequence (turn-weighted exclusive-CE probe, completed; passed)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t195814z-872818a4fa68`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-turn-weighted.toml`, run id
   `dgx-play-maze-chase-distill-turn-weighted-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` passed: reward −392 / collisions 43 / 38 pellets /
   histogram [[2, 377], [4, 103]] (A×377 + S×103). Val match 0.25.
   Do not retune hold×0.1.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

## Spark sequence (128-step turn-weighted exclusive-CE probe, completed; failed sticky S)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t202732z-ab1001f8af37`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-turn-weighted-128.toml`, run id
   `dgx-play-maze-chase-distill-turn-weighted-128-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` failed sticky S: reward −3880 / collisions 390 /
   20 pellets / histogram [[0, 26], [4, 454]] (idle×26 + S×454). Val
   match 0.0. Do not scale 128.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.

## Spark sequence (play-peak / early-stop, completed / passed)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1` (release `r20260818t215024z-ad7ce0bfdd99`)
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml` (receipt written)
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-play-peak.toml`, run id
   `dgx-play-maze-chase-distill-play-peak-v1`, Tmux with
   `-AcknowledgeDetached`
5. `play-gate.json` passed on the kept step-32 peak: reward −392 /
   collisions 43 / 38 pellets / histogram [[2, 377], [4, 103]].
   `stop_reason: play_peak_drop` at step 40 (15 pellets). Spark GPU is
   idle. Do not scale.

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

The B2 world-model-actor manifest was regenerated because `train.py` and
`objective.py` are in that family's source set (new digest `74ec3510…`;
previous `898e8a0e…`, `4e603895…`, `be571ba4…`, `885aff85…`, `3d2e3cc0…`,
then `c207401f…`).
No actor parameter or recipe identity changed.
The matched-baseline architecture manifest is a new comparison identity
`cc04cb4f…` (previous live `3d7ff5bd…`, then `5e0f2536…`) for the same
source-file reason.

## Spark sequence (ghost-hit penalty + play-peak, preregistered)

1. `Invoke-DgxPreflight.ps1`
2. `Sync-DgxBrainRelease.ps1`
3. `Invoke-DgxBrainSmoke.ps1` on `dgx-smoke.toml`
4. `Start-DgxBrainTraining.ps1` with
   `dgx-play-maze-chase-distill-ghost-hit.toml`, run id
   `dgx-play-maze-chase-distill-ghost-hit-v1`, Tmux with
   `-AcknowledgeDetached`
5. Watch `play-gate.json` vs the 38/43 champion. If collisions drop and
   pellets stay ≥ 32, that is the new champion. If pellets collapse,
   Spark idle; do not scale.

Generic wrappers only. Never `Start-DgxRcqV2Reference.ps1`. Never point
generic train at an RCQ config.
