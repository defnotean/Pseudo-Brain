# T1-2 closeout: Pac-Man-like harness stack adjudication (2026-08-28)

**Date:** 2026-08-28
**Status:** CLOSED — the conforming P1–P6 harness build is verified on disk and
locked with a test module; a divergent uncommitted reconstruction was retired
(reversibly) and the regression delta is accounted for.
**Mode:** local CPU only, single-thread, CUDA hidden, deterministic protocol.
**Base HEAD:** `8c4c127` (T1-1 frozen; T1-2 report + gate JSONs committed in the
prior session but the implementation was left uncommitted).

## 1. What was found on disk (reconstruction)

The working tree at session start contained **two generations of the T1-2
harness, both uncommitted**:

| | Stack A — `environments/pacman_harness.py` | Stack B — `environments/pacman_harness_v1.py` |
|---|---|---|
| Preregistration | the **committed** frozen `2026-08-27-pacman-harness-v1.md` (HEAD `8c4c127`), incl. its post-commit P2/P4/P5 amendment | a **new uncommitted** `...-v1-construction.md` (never committed, never frozen in git) |
| Family table §1 | matches the committed table **exactly** (F0_calm 1g/3/shy/24loops; F3_fast elroy=True; F4_open_slow player_period=2) | a **different table** (F0 40loops/shy/4g … F4 6loops/direct/1g) that matches **no** committed document |
| P4/P5 mechanics | channel-level world hold / extra step at cycles `n mod 10 == 0` incl. 0; **control period stays the 60 Hz floor schedule** (the amended §2 text) | rescales the control period to 18,518,519 / 15,151,515 ns → **54 Hz / 66 Hz control rate**, which violates the frozen v3 §4.3 "native control rate 60 Hz" gate |
| Seed partitions §3 | block `[201326592, 201347072)` (width 20,480), 40/100/100/524 per family — consistent | 200,000 / 50,000 / 50,000 / 400,000 = 300,000 seeds **inside a 20,480-wide block** (~15× overflow; internally inconsistent) |
| T1-2 report | `2026-08-27-t12-pacman-harness-gates.md` + `runs/pacman-harness-gates/*.json` describe **this stack** (H1–H7, P2 = "exactly 4 holds over 50 cycles", P4/P5 = 45/55 world steps, `pacman_harness_gates.py --full`) | describes none |
| Test module | `tests/test_pacman_harness.py` — **cited by the T1-2 report as 21/21 but never landed on disk** | `tests/test_pacman_harness_v1.py` (18 tests, pins the divergent B tables) |

**Why B's P3/P4 cannot stand** [MEASURED]: under B, every cycle uses the
scaled period, so the agent's native control rate becomes
`1e9/18_518_519 = 53.999… Hz` (P3) and `1e9/15_151_515 = 66.000… Hz` (P4).
v3 §4.3 freezes the Pac-Man-like control rate at 60 Hz and the schedule at
`s_n = t0 + floor((n·1e9 + 30)/60)` ns; B implements neither under P3/P4.
Also, B's committed-intent P4/P5 (per its own construction doc) are
`round(ghost_period/speed)` on the **simulator** period, which is a **no-op
for every registered family**: for `ghost_period ∈ {1,2,3}`,
`max(1, round(gp/0.9))` and `max(1, round(gp/1.1))` both equal `gp`
[MEASURED] — the original committed P4/P5 rule was degenerate, which is why
the post-commit amendment moved P4/P5 to the channel level.

**Adjudication:** Stack A is the build that (a) implements the committed
frozen preregistration (as amended), (b) has the committed T1-2 report and
gate JSONs, and (c) passes all seven gates. Stack B is a divergent,
self-inconsistent reconstruction. **Stack A is retained; Stack B is retired.**

## 2. Actions taken (all reversible, no sealed artifact touched)

1. **Retired Stack B** (moved, not deleted):
   `brain/scratch/retired-pacman-harness-v1-construction/`
   (module, its test module, its acceptance runner, its uncommitted
   construction doc). Nothing else in the repo imported it
   [MEASURED: `grep -rln` over `brain/` shows only its own test + runner].
2. **Locked the frozen v3 §4.3 Pac-Man-like timing gates in the harness
   layer:** added `TIMING_GATES` + `check_timing_gates` to
   `environments/pacman_harness.py` (median ≤ 8.00 ms, p95 ≤ 13.00 ms,
   p99 ≤ 16.00 ms, miss-rate ≤ 0.1%, p99 jitter ≤ 2.00 ms, zero consecutive
   misses). These are the Tier-3 (Q1/Q2/Q3) gate constants; the offline
   replay (0-ns declared latency) trivially clears them [MEASURED].
3. **Reconstructed the missing P1–P6 test module**
   `brain/tests/test_pacman_harness.py` (25 tests, 4 classes): frozen
   family/P6/partition/timing-gate identity; exact floor-schedule clock
   (spot-checked n ∈ {0,1,2,3,99,100,101,5000,5999}); two-run determinism
   byte-match; P1–P6 perturbation isolation with **exact counts**
   (P2 holds = {10,20,30,40}/50; P3 exact one-cycle shift with cycle-0
   reset frame; P4 held = {0,10,20,30,40}, 45 world steps/50; P5 extra =
   {0,10,20,30,40}, 55 world steps/50; P6 state-hash divergence);
   **P4/P5 keep the 60 Hz floor-schedule control period** (the property that
   the retired reconstruction violated); C8 causal boundary (no privileged
   fields in `Observation`; control arm — mutating
   `_previous_key_mask` leaves pixels byte-identical); legal-action surface
   (all five locomotion values map into the WASD allow-list; look/cursor/
   buttons add no WASD keys; the env input pipeline masks to the four WASD
   support bits).
   **Result [MEASURED]: 25/25 OK in 0.360 s.**
4. **Re-ran the H1–H7 gate battery at frozen scale**
   (`pacman_harness_gates.py --full`): **all seven PASS, 3.419 s wall**,
   report `brain/runs/pacman-harness-gates/2026-08-28-full-rerun.json`.
   Identical verdicts to the 2026-08-27 run (H1 18-episode byte-match; H4
   exact-count accounting; H5 3,820 seeds disjoint).

## 3. Full regression delta (T0-1 re-verification, 2026-08-28)

`py -3.11 -m unittest discover -s tests -p "test_*.py"` (1169 tests,
452.6 s, incl. the 18 Stack-B tests before retirement and the 25 new
P1–P6 tests): **4 failures + 8 errors + 2 skips** vs. T0-1's 6 non-green.
Every non-green is accounted for; **none imports or depends on any
harness module** [MEASURED: grep + traceback inspection]:

| Test | Class | Evidence |
|---|---|---|
| `test_release_sync…`, `test_resume_preflight…`, 4× `test_dgx_launch_contract` errors | **pre-existing** (T0-1 documented the first two; the other bash-contract errors are the same POSIX-`/tmp`/`realpath` environmental family) | same tracebacks as T0-1 |
| 3× V2.1i/V2.1j/V2.1M source-bundle canaries | **pre-existing** (T0-1 documented 4 stale frozen-identity canaries; 3 of 4 re-fired, 1 did not — see note) | identical AssertionError shape |
| `test_matched_baselines…strictly_identified` | **pre-existing** (T0-1 documented) | identical digest-missing failure |
| `test_v21m…resource_guard…unmocked_windows_ABI` | **environmental / pre-existing on this host** (subprocess torch import failure; re-verified in isolation on 2026-08-28 — still fails, no harness involvement) | isolated re-run, 1 error |
| `test_60hz_arcade_tick_budget_and_latency` | **load artifact** | p50 17.54 ms under the 452 s full-suite load; **passes in isolation on a quiet box** (2 tests, OK, 2.4 s) |

**Note on the canary count:** T0-1 recorded 4 stale-identity canaries; this
run re-fires 3 of those 4 (the fourth —
`test_v21i_frozen_representation_continuation_diagnostic_v1`) did not re-fire.
[MEASURED] The canary suite hashes the current tree against frozen historical
pins; whether a given canary fires depends on which tracked files advanced
since its pin. No sealed artifact was modified by this session; the count
delta is a tree-provenance effect, not a regression, and not a weakening. (If it re-fires on the next quiet run, that is recorded, not
"fixed".)

## 4. What this session does NOT change

- No sealed registration, DEV/CPU-QUAL/TEST partition, or checkpoint was
  opened or materialized.
- `maze_chase.py` internals: untouched (the harness is a wrapper).
- No re-baselining of the stale frozen-identity canaries (T0-1 decision
  stands: a scientific-claim decision needing fresh registration).
- No retry of any closed/ambiguous branch (V2-C confirmation, hazard
  post-processing family, PB21N/O/S, L8–L11 all remain sealed).

## 5. Next (queue)

- **T1-3** — expert/scripted pixel-only policy → behavior-cloning corpus on
  TRAIN seeds. The planner
  (`diagnostic.scripted_maze_chase_planner.v1`) exists and is pixel-only;
  corpus generation = a new preregistration + runner under the T1-1 contract.
- **T1-4** — reactive baseline (never weakened), same budget class.
- **T2-1** — integrated Core V2 + `EmbodiedInterfaceV1` training under the
  deployed decision loss; **T2-2** CPU-QUAL prereg.
