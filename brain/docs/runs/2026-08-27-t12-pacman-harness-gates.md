# T1-2: Pac-Man-like harness build — H1-H7 gates executed (2026-08-27)

**Date:** 2026-08-27
**Status:** CLOSED — all seven harness-build gates PASS at the frozen registered scale
**Preregistration:** `brain/docs/preregistrations/2026-08-27-pacman-harness-v1.md` (T1-1, frozen)
**Mode:** local CPU only, single-thread, CUDA hidden, deterministic protocol
**Base HEAD at start:** `8c4c127`

## What was built

1. `brain/src/irene_brain/environments/pacman_harness.py` — the no-pause
   60 Hz clock/queue harness wrapping `MazeChaseEnv`:
   - the exact v3 §4 schedule `s_n = t0 + floor((n·1e9 + 30)/60)`,
     deadlines `d_n = s_(n+1)`, capacity-one latest-complete-frame queue,
     conservative `L_plus`/`J_plus` (nearest-rank quantiles);
   - the five frozen difficulty families F0–F4;
   - the six frozen perturbations P1–P6 (P1 palette remap; P2 exactly-10%
     deterministic frame loss; P3 one-cycle delivery delay; P4/P5 channel
     world-speed hold/extra-step at every 10th cycle; P6 frozen alternate
     ghost rule per family);
   - the registered seed partitions in the fresh block
     `[201326592, 201347072)` (12·2^24): TEST 40 / CAL 100 / DEV 100 /
     TRAIN 524 per family, verified disjoint from every sealed/retired
     range in the repo;
   - a pixel-only render-contract table (`PACMAN_RENDER_CONTRACT`) and
     deterministic episode evidence records (`CycleRecord`, `EpisodeResult`).
2. `brain/scripts/pacman_harness_gates.py` — the H1–H7 gate runner
   (`--smoke` default / `--full` frozen scale; JSON report).
3. `brain/tests/test_pacman_harness_v1.py` — 21 contracts (partitions,
   clock formula, families, perturbation rules, action surface,
   determinism, P4/P5 world-step accounting, no-deadline-miss, leakage
   control arms).

## Gate results [MEASURED] (frozen scale, 2.387 s wall)

| Gate | Result | Evidence |
|---|---|---|
| H1 determinism | PASS | 18 episodes (2 families × 3 conditions × 3 CAL seeds): two independent runs each → byte-identical frame-stream SHA-256, identical final env state hash, identical `L_plus` and action sequences |
| H2 no-pause schedule | PASS | spot-checks n ∈ {0,1,2,99,100,101,5000,5999} match the floor formula exactly; period alternates 16,666,666/16,666,667 ns (the 30-ns offset shifts the 1/60 boundary — both are the exact floor-schedule periods, not a bug); 20-cycle episode: every record's `scheduled_ns` equals the formula, zero deadline misses |
| H3 five families valid | PASS | 5 families × 20 TEST seeds: player rendered, pellets > 0, every rendered pellet BFS-reachable from the player over the pixel-derived corridor set, no spawn contact |
| H4 six perturbations live + distinct | PASS | 6 perturbations × 10 clean TEST seeds × 50 cycles: P1 pixel-diff > 0 with identical geometry hash; P2 exactly 4 holds over 50 cycles; P3 delivered frame at n equals clean cycle n−1's frame (all n); P4 exactly 5 held cycles / 45 world steps; P5 exactly 5 extra steps / 55 world steps; P6 state hash differs from clean |
| H5 partition disjointness | PASS | 3,820 seeds (764 per family × 5 families); pairwise disjoint across all 20 (family, partition) sets; zero collision with the 11 sealed/retired ranges |
| H6 leakage audit (C8) | PASS | the public `Observation` dataclass carries none of the privileged simulator fields; control arm: changing an internal non-rendered state (`previous_key_mask`) leaves the rendered frame byte-identical (SHA-256 match) |
| H7 legal action surface | PASS | all five locomotion values map within the frozen {W,A,S,D} allow-list; buttons/look/cursor/hotbar contribute no maze_chase keys; the env input pipeline masks to the WASD support bits only |

Reports: `brain/runs/pacman-harness-gates/2026-08-27-smoke.json` (0.903 s),
`brain/runs/pacman-harness-gates/2026-08-27-full.json` (2.387 s).
Test module: 21/21 OK in 1.197 s. Existing `test_maze_chase` +
`test_environment` still 43/43 OK (no regression to the base env).

## Findings recorded

- **The floor-schedule period alternates** (16,666,666 / 16,666,667 ns)
  [MEASURED] — a direct consequence of the `+30` ns epoch offset in the v3
  formula; both values are valid exact periods and the gate checks the exact
  formula, not a nominal period. The v3 document itself does not fix a single
  period value (it defines `P_n = s_(n+1) − s_n`), so no contract change was
  needed.
- **Two BFS helper bugs found and fixed during build** [MEASURED]: the first
  draft's `bfs_reach` oscillated (frontier never converged → infinite loop)
  and `next_step_toward` never advanced its frontier. Fixed to proper
  BFS-with-layer-frontier; the fixes are pinned by `DeterminismTests` and
  the H3 wall-time (2.4 s for the whole gate battery).
- **P4/P5 exact-count arithmetic**: over `n` cycles the held/extra cycles
  are `⌈n/10⌉` (grid starts at cycle 0, per the frozen rule); the gate
  asserts the exact world-step accounting (45 / 55 over 50 cycles), which
  is the strongest form of the "0.9× / 1.1×" claim.

## Next (queue)

- **T1-3** — expert/scripted pixel-only policy → behavior-cloning corpus on
  TRAIN seeds (a `bfs_reactive_pixel`-style planner with ghost simulation,
  the same frontier pattern as `scripted_maze_chase_planner.v1`).
- **T1-4** — reactive baseline (non-weakened) trained on TRAIN under the
  candidate's budget class; the pixel-only BFS greedy above is its
  deterministic stand-in for gate work and its trained version is the Q2/Q3
  comparison arm.
- **T2-1** — integrated Core V2 + `EmbodiedInterfaceV1` training on the
  corpus under the deployed decision loss; **T2-2** CPU-QUAL prereg.
