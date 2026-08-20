# Play competence: current neural checkpoint vs planner (2026-08-20)

Status: **measured**. Thought-mediated Phase 2.5 has **no persisted maze-chase
weights** on Spark. Variant E has **no persisted checkpoint**. The live play
artifact is still the Aug 18 **32-step turn-weighted exclusive-CE champion**
(`dgx-play-maze-chase-distill-turn-weighted-v1`,
`step-00000032.pt` SHA-256 `e58f323fb90893c4953b04211002753dd3162f398d1b3b4152390203ba8131cf`).
Closed-loop play was re-run on Spark **CPU** (`CUDA_VISIBLE_DEVICES=-1`). GB10
was left alone (Irene sglang occupied it; no second GPU train). Protocols are
labeled and not mixed.

JSON: [artifacts/play-gated-maze-chase-distill/play-competence-20260820.json](./artifacts/play-gated-maze-chase-distill/play-competence-20260820.json)
(SHA-256 `82a3d8c8f9fb72296893d380aba547d4eca48ab74acf12df97927dda74ec0663`).
GIFs (new names, do not confuse with Aug 18 `turn-weighted-v1-seed*.gif`):
[play-competence-20260820-turn-weighted-champion-seed5.gif](./artifacts/play-gated-maze-chase-distill/play-competence-20260820-turn-weighted-champion-seed5.gif)
(21 pellets / 12 collisions) and
[seed 9](./artifacts/play-gated-maze-chase-distill/play-competence-20260820-turn-weighted-champion-seed9.gif)
(17 pellets / 31 collisions).

## What was evaluated

| Field | Value |
|---|---|
| Agent | `turn-weighted-v1-champion` |
| Spark run | `play-competence-cpu-20260820-v1` |
| Device | CPU docker, image `177a406d7cb2`, 4 CPUs, 12 GiB |
| Release (load) | `r20260818t195814z-872818a4fa68` |
| Decode | `exclusive_argmax_wasd_v1` |
| World | `MazeChaseEnv(ghost_count=3, ghost_period=2, extra_loops=16)` |
| Thought-mediated campaign `.pt` | none |
| Variant E `.pt` | none |
| GPU train started | **no** |

Entry point: `brain/scripts/eval_play_competence.py`. Launcher:
`brain/scripts/dgx/start_play_competence_eval.sh`.

## Protocol `distill-direct-5-9-240`

Same gate as the Aug 18 play-gated distill campaign (seeds 5/9, 240 ticks).
Neural totals **match** the frozen `play-gate.json` (38 pellets / 43 collisions).
The net does not clear. The planner clears both seeds.

| Agent | Seed | Pellets | Collisions | Ticks | Clear |
|---|---|---:|---:|---:|---|
| neural | 5 | 21 | 12 | 240 | no |
| neural | 9 | 17 | 31 | 240 | no |
| **neural total** | 5+9 | **38** | **43** | 480 | **0** |
| planner | 5 | 142 | 0 | 208 | **yes** |
| planner | 9 | 142 | 1 | 237 | **yes** |
| **planner total** | 5+9 | **284** | **1** | 445 | **2** |

Histogram remains A×377 + S×103 (seed 5: A×188+S×52; seed 9: A×189+S×51).
Clumsy corridor eating, not a player. Planner is ~7.5× pellets with 43× fewer
ghost hits and two maze clears.

## Protocol `e-heldout-direct-2001-2020-120`

Variant E's **seed/tick grid** (seeds 2001–2020, 120 ticks, 2,400 steps) but
**direct** exclusive-argmax play. This is **not** a reproduction of Variant E's
5.65-pellet / 83-catch table, which used a LatentLookaheadPolicy wrap and a
checkpoint that is not on Spark.

| Agent | Mean pellets | Total pellets | Mean collisions | Total collisions | Clears |
|---|---:|---:|---:|---:|---:|
| neural (turn-weighted champion, direct) | **8.55** | 171 | 13.25 | 265 | **0** |
| planner | **84.5** | 1690 | 1.05 | 21 | **0** |

No clears on this protocol: 120 ticks is below the planner's seed-5 clear
(~tick 208, ~142 pellets). Neural is still ≪ planner (~10× fewer pellets,
~13× more ghost hits).

Do not mix this 8.55 mean with Variant E's 5.65, or with the distill 38-pellet
two-seed total.

## Honest play verdict

The current best neural maze-chase checkpoint **does not play** at planner
competence. It passes the distill campaign pellet floor (38 ≥ 32) by grinding
A/S in corridors, dies to ghosts, and never clears. Phase 2.5 thought-mediated
campaign scores are diagnostic knockouts on other worlds; they are not maze-chase
play evidence until a named maze-chase checkpoint exists.

## Next bounded play train (not started)

GB10 was busy (sglang). Do **not** start this while Gate 6 or any GPU job owns
the accelerator.

Named 32-step **play-gated maze-chase distill of the thought-mediated
(no-bypass) actuator** on the existing tiled planner teacher
(`irene.maze_chase.planner_teacher.tiled_windows.v1`). Keep
`exclusive_argmax_wasd_v1`, turn-weighted exclusive CE (hold ×0.1 unchanged),
play eval every 8 on seeds 5/9 × 240, `play_peak_v1` early-stop. Pass if
pellets ≥ 32 **and** (collisions < 43 or a maze clear). Fail and stop if
pellets drop into the sticky ~20 band. Do not resume RCQ-v2. Do not retune
hold ×0.1. Do not scale 128.
