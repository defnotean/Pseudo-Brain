"""Pac-Man-like (maze_chase) harness gates H1-H7 — T1-2 runner.

Executes the harness-build acceptance gates frozen in
``brain/docs/preregistrations/2026-08-27-pacman-harness-v1.md`` section 6
against ``irene_brain.environments.pacman_harness``.  Deterministic,
CPU-only, single-thread, no model, no network: pure simulator + channel
logic plus pixel-only (public render-contract) frame analysis.

Usage (from the repo root):
  py -3.11 brain/scripts/pacman_harness_gates.py --smoke
  py -3.11 brain/scripts/pacman_harness_gates.py --full --json <out.json>

--smoke runs the reduced instance counts (default, for iteration);
--full runs the frozen registered scale (H1: 3 seeds x 2 families x 3
conditions; H3: 20 TEST seeds x 5 families; H4: 10 clean TEST seeds x 6
perturbations).  Both scales are fully deterministic.

Exit code 0 iff every gate passes; the JSON report (with --json) carries
each gate's verdict plus the measured evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from irene_brain.environments.maze_chase import MazeChaseEnv  # noqa: E402
from irene_brain.environments.pacman_harness import (  # noqa: E402
    FAMILIES,
    PACMAN_RENDER_CONTRACT,
    PacmanHarness,
    action_class_of,
    all_harness_seeds,
    assert_seed_partition_disjointness,
    control_for_class,
    partition_seeds,
    scheduled_start_ns,
    Controller,
)
from irene_brain.types import GenericControl, RgbFrame  # noqa: E402
from irene_brain.v2.embodied_interface import (  # noqa: E402
    LOCOMOTION,
    EmbodiedInterfaceRecord,
    translate_record,
)

GRID = 16
PLAYER_C = PACMAN_RENDER_CONTRACT["player"]
PELLET_C = PACMAN_RENDER_CONTRACT["pellet"]
GHOST_C = PACMAN_RENDER_CONTRACT["ghost"]
WALL_C = PACMAN_RENDER_CONTRACT["wall"]
BG_EVEN = PACMAN_RENDER_CONTRACT["background_even"]
BG_ODD = PACMAN_RENDER_CONTRACT["background_odd"]
OVERLAP_C = PACMAN_RENDER_CONTRACT["player_ghost_overlap"]


# --- Pixel-only frame analysis (public render contract only) ---------------


def classify_frame(frame: RgbFrame) -> dict[str, object]:
    """Classify one frame using ONLY the public render-contract colors.

    Returns player cell, rendered pellet cells, ghost cells, and wall cells.
    No simulator coordinates or privileged state are read (H6 discipline).
    """
    if frame.width != GRID or frame.height != GRID:
        raise ValueError(f"expected {GRID}x{GRID} frame, got {frame.width}x{frame.height}")
    players, pellets, ghosts, walls = [], [], [], []
    for y in range(GRID):
        for x in range(GRID):
            off = (y * GRID + x) * 3
            rgb = (frame.pixels[off], frame.pixels[off + 1], frame.pixels[off + 2])
            cell = (x, y)
            if rgb == PLAYER_C:
                players.append(cell)
            elif rgb == PELLET_C:
                pellets.append(cell)
            elif rgb == GHOST_C:
                ghosts.append(cell)
            elif rgb == WALL_C:
                walls.append(cell)
            elif rgb in (BG_EVEN, BG_ODD):
                pass
            elif rgb == OVERLAP_C:
                # player sitting on a ghost: count as player cell (and the
                # ghost under it is still rendered beneath by the sim; the
                # overlap color replaces it, so record the player cell).
                players.append(cell)
            else:
                raise ValueError(f"frame contains out-of-contract color {rgb!r} at {cell}")
    return {
        "player": players[0] if len(players) == 1 else None,
        "n_players": len(players),
        "pellets": tuple(sorted(pellets)),
        "ghosts": tuple(sorted(ghosts)),
        "walls": tuple(sorted(walls)),
    }


def corridor_neighbors(cell: tuple[int, int], walls: set[tuple[int, int]]):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (x + dx, y + dy)
        if 0 <= n[0] < GRID and 0 <= n[1] < GRID and n not in walls:
            yield n


def bfs_reach(player, walls: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """All corridor cells reachable from ``player`` (proper BFS)."""
    reachable = set()
    frontier = {player}
    while frontier:
        new: set[tuple[int, int]] = set()
        for cell in frontier:
            for n in corridor_neighbors(cell, walls):
                if n not in reachable and n not in new:
                    new.add(n)
        reachable |= new
        frontier = new
    return reachable


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Controllers -------------------------------------------------------------


def idle_controller() -> Controller:
    return Controller(name="idle", fn=lambda cycle, frame: 0)


def wasd_cycle_controller() -> Controller:
    """Deterministic non-idle controller: cycles W/A/S/D every cycle."""

    def fn(cycle: int, frame: RgbFrame) -> int:
        return 1 + (cycle % 4)

    return Controller(name="wasd_cycle", fn=fn)


def bfs_reactive_controller() -> Controller:
    """Pixel-only greedy reactive controller (T1-4 baseline stand-in).

    Every cycle: classify the frame; BFS on the pixel-derived wall set to
    the nearest rendered pellet; if the next step would put the player
    within Chebyshev distance 1 of a rendered ghost, prefer the corridor
    neighbor that maximizes the minimum distance to any rendered ghost;
    idle when no pellet is reachable.  No simulator coordinates ever enter:
    only the pixel classification.  Pure Python (fast enough for H4 smoke).
    """

    def next_step_toward(target, walls):
        # BFS from the target over corridor cells; dist[cell] = path length.
        dist = {target: 0}
        frontier = {target}
        while frontier:
            new: set = set()
            for cell in frontier:
                for n in corridor_neighbors(cell, walls):
                    if n not in dist:
                        dist[n] = dist[cell] + 1
                        new.add(n)
            frontier = new
        return dist

    def fn(cycle: int, frame: RgbFrame) -> int:
        cls = classify_frame(frame)
        player = cls["player"]
        walls = set(cls["walls"])
        pellets = cls["pellets"]
        ghosts = cls["ghosts"]
        if player is None or not pellets:
            return 0
        dist = next_step_toward(pellets[0], walls)
        if player not in dist:
            return 0
        # nearest rendered pellet by BFS distance
        best = min(pellets, key=lambda p: dist.get(p, 10**9))
        # candidate corridor neighbors, ordered by descending distance to
        # the target; prefer the one farthest from any rendered ghost.
        cands = [
            n for n in corridor_neighbors(player, walls)
            if n in dist and dist[n] < dist[player]
        ]
        if not cands:
            return 0
        scored = []
        for n in cands:
            danger = min((manhattan(n, g) for g in ghosts), default=99)
            scored.append((dist[n], -min(danger, 99), n))
        scored.sort()
        step = scored[0][2]
        if step[0] > player[0]:
            return 4  # D
        if step[0] < player[0]:
            return 2  # A
        if step[1] < player[1]:
            return 1  # W
        if step[1] > player[1]:
            return 3  # S
        return 0

    return Controller(name="bfs_reactive_pixel", fn=fn)


# --- Gate implementations (preregistration section 6) ------------------------


def _h1_determinism(full: bool) -> dict:
    """H1: two independent runs of fixed (family, seed, perturbation)
    episodes produce byte-identical rendered-frame streams and timing
    records."""
    seeds = list(partition_seeds(1, "CAL")[: (3 if full else 2)])
    families = [0, 1]
    conditions = ["clean", "P2_frame_loss_10pct", "P3_obs_delay_1"]
    mismatches = []
    checked = 0
    for fam in families:
        for cond in conditions:
            h = PacmanHarness(fam, perturbation=cond, max_cycles=60)
            for seed in seeds:
                r1 = h.run_episode(seed, idle_controller(), partition="CAL")
                r2 = h.run_episode(seed, idle_controller(), partition="CAL")
                checked += 1
                if (
                    r1.frame_sha256_stream_sha256 != r2.frame_sha256_stream_sha256
                    or r1.env_state_hash_final != r2.env_state_hash_final
                    or [r.l_plus_ns for r in r1.records] != [r.l_plus_ns for r in r2.records]
                    or [r.action_class for r in r1.records] != [r.action_class for r in r2.records]
                ):
                    mismatches.append({"family": fam, "condition": cond, "seed": seed})
    return {
        "gate": "H1",
        "passed": bool(checked) and not mismatches,
        "episodes_checked": checked,
        "seeds": seeds,
        "families": families,
        "conditions": conditions,
        "mismatches": mismatches,
    }


def _h2_no_pause_schedule() -> dict:
    """H2: the schedule exactly matches s_n = floor((n*1e9+30)/60), the
    period P_n is its difference, and a real episode's record timestamps
    equal the schedule with no deadline miss."""
    spot = []
    ok = True
    # The floor schedule s_n = floor((n*1e9+30)/60) has a period that
    # alternates between 16,666,666 and 16,666,667 ns (the 30-ns offset
    # shifts the 1/60 boundary). Both are valid; the deadline/miss logic
    # below uses the exact s_n values, not a nominal period.
    for n in (0, 1, 2, 99, 100, 101, 5000, 5999):
        s = scheduled_start_ns(n)
        expected = (n * 1_000_000_000 + 30) // 60
        p = scheduled_start_ns(n + 1) - s
        match = s == expected and p in (16_666_666, 16_666_667)
        spot.append({"n": n, "s_n": s, "period_ns": p, "ok": match})
        ok = ok and match
    h = PacmanHarness(0, max_cycles=20)
    result = h.run_episode(partition_seeds(0, "TEST")[0], idle_controller(), partition="TEST")
    for rec in result.records:
        expected = (rec.cycle_id * 1_000_000_000 + 30) // 60
        if rec.scheduled_ns != expected:
            ok = False
        if rec.t_accept > rec.deadline_ns:
            ok = False
        if rec.period_ns not in (16_666_666, 16_666_667):
            ok = False
    return {
        "gate": "H2",
        "passed": ok,
        "spot_checks": spot,
        "episode_cycles": len(result.records),
    }


def _h3_families_valid(full: bool) -> dict:
    """H3: each family on N TEST seeds produces a valid start: the player is
    rendered, pellets > 0, every rendered pellet is BFS-reachable from the
    player over the pixel-derived corridor set, and no ghost renders on the
    player cell at spawn."""
    n_seeds = 20 if full else 5
    seeds = []
    # draw from different families' TEST ranges so every family is checked
    # on layouts from its own registered partition
    per_family = []
    all_ok = True
    for fam in range(len(FAMILIES)):
        seeds = list(partition_seeds(fam, "TEST")[:n_seeds])
        starts = []
        for seed in seeds:
            env = PacmanHarness(fam).build_env(seed)
            cls = classify_frame(env.current_observation.rgb)
            player = cls["player"]
            walls = set(cls["walls"])
            pellets = set(cls["pellets"])
            ghosts = set(cls["ghosts"])
            if player is None or not pellets:
                valid = False
                reachable_fraction = 0.0
            else:
                reachable = bfs_reach(player, walls)
                valid = pellets.issubset(reachable) and player not in ghosts
                reachable_fraction = len(pellets & reachable) / len(pellets)
            all_ok = all_ok and valid
            starts.append(
                {
                    "seed": seed,
                    "pellets": len(pellets),
                    "ghosts": len(ghosts),
                    "all_pellets_reachable": valid and pellets.issubset(reachable),
                    "reachable_fraction": round(reachable_fraction, 4),
                    "spawn_contact": player in ghosts,
                    "valid": bool(valid),
                }
            )
        per_family.append(
            {
                "family": FAMILIES[fam].name,
                "starts": starts,
                "mean_start_pellets": round(sum(s["pellets"] for s in starts) / len(starts), 2),
            }
        )
    return {"gate": "H3", "passed": all_ok, "seeds_per_family": n_seeds, "families": per_family}


def _h4_perturbations_live(full: bool) -> dict:
    """H4: each perturbation runs to completion on N TEST seeds and its
    observation stream differs from the clean twin in the frozen way."""
    n_seeds = 10 if full else 3
    n_cycles = 50
    fam = 1
    seeds = list(partition_seeds(fam, "TEST")[:n_seeds])
    results = {}
    all_ok = True
    for pert in (
        "P1_palette_shift",
        "P2_frame_loss_10pct",
        "P3_obs_delay_1",
        "P4_speed_90pct",
        "P5_speed_110pct",
        "P6_enemy_policy_change",
    ):
        detail = []
        ok = True
        for seed in seeds:
            hc = PacmanHarness(fam, max_cycles=n_cycles)
            hp = PacmanHarness(fam, perturbation=pert, max_cycles=n_cycles)
            rc = hc.run_episode(seed, wasd_cycle_controller(), partition="TEST")
            rp = hp.run_episode(seed, wasd_cycle_controller(), partition="TEST")
            crashed = rp.elapsed_cycles == 0 or rc.elapsed_cycles == 0
            differs = False
            if pert == "P1_palette_shift":
                differs = (
                    rc.frame_sha256_stream_sha256 != rp.frame_sha256_stream_sha256
                    and rc.env_state_hash_final == rp.env_state_hash_final
                )
            elif pert == "P2_frame_loss_10pct":
                holds = sum(1 for r in rp.records if r.is_hold)
                differs = holds == 4  # drops at cycles 10, 20, 30, 40
            elif pert == "P3_obs_delay_1":
                differs = all(
                    rp.records[n].delivered_frame_sha256 == rc.records[n - 1].frame_sha256
                    for n in range(1, min(len(rp.records), len(rc.records)))
                )
            elif pert == "P4_speed_90pct":
                held = sum(1 for r in rp.records if r.world_steps_this_cycle == 0)
                steps = sum(r.world_steps_this_cycle for r in rp.records)
                expected_held = (n_cycles + 9) // 10  # cycles 0,10,20,...< n_cycles
                differs = held == expected_held and steps == n_cycles - expected_held
            elif pert == "P5_speed_110pct":
                extra = sum(1 for r in rp.records if r.world_steps_this_cycle == 2)
                steps = sum(r.world_steps_this_cycle for r in rp.records)
                expected_extra = (n_cycles + 9) // 10  # cycles 0,10,20,...< n_cycles
                differs = extra == expected_extra and steps == n_cycles + expected_extra
            else:  # P6_enemy_policy_change
                differs = rc.env_state_hash_final != rp.env_state_hash_final
            detail.append(
                {
                    "seed": seed,
                    "crashed": crashed,
                    "differs": bool(differs),
                    "perturbed_cycles": rp.elapsed_cycles,
                    "clean_cycles": rc.elapsed_cycles,
                }
            )
            ok = ok and not crashed and bool(differs)
        results[pert] = {"seeds": detail, "passed": ok}
        all_ok = all_ok and ok
    return {"gate": "H4", "passed": all_ok, "perturbations": results}


def _h5_partition_disjointness() -> dict:
    """H5: every (family, partition) seed set is pairwise disjoint and free
    of collisions with the repo's sealed/retired ranges; totals match the
    frozen counts (40/100/100/524 per family)."""
    try:
        assert_seed_partition_disjointness()
        total = len(all_harness_seeds())
        expected_total = (40 + 100 + 100 + 524) * 5
        counts_ok = total == expected_total
        return {
            "gate": "H5",
            "passed": counts_ok,
            "seeds": total,
            "expected_seeds": expected_total,
        }
    except AssertionError as exc:
        return {"gate": "H5", "passed": False, "error": str(exc)}


def _h6_leakage_audit() -> dict:
    """H6: no privileged field reaches the observation the model sees.

    (a) The env's public Observation dataclass carries none of the
    simulator's privileged fields (maze, pellets, ghosts, coordinates,
    reward events, cleared/caught).
    (b) Control arm: changing an internal, non-rendered state
    (previous_key_mask) leaves the rendered frame byte-identical — the
    observation is a pure function of rendered pixels.
    """
    fam = 1
    seed = partition_seeds(fam, "TEST")[0]
    h = PacmanHarness(fam)
    env = h.build_env(seed)
    obs = env.current_observation

    privileged_names = {
        "maze",
        "pellets",
        "pellets_remaining",
        "player_x",
        "player_y",
        "player_dx",
        "player_dy",
        "ghosts",
        "episode_seed",
        "rng",
        "times_caught",
        "cleared",
        "pellets_eaten",
        "previous_key_mask",
        "sticky_mask",
        "delay_queue",
        "tick",
    }
    obs_fields = set(type(obs).__dataclass_fields__)  # type: ignore[attr-defined]
    leaked = sorted(obs_fields & privileged_names)

    env2 = h.build_env(seed)
    env2._previous_key_mask = 0x0F  # noqa: SLF001 (control-arm internal)
    frame_a = env.current_observation.rgb
    frame_b = env2.current_observation.rgb
    internal_change_no_pixel_change = frame_a.pixels == frame_b.pixels

    return {
        "gate": "H6",
        "passed": not leaked and internal_change_no_pixel_change,
        "observation_fields": sorted(obs_fields),
        "leaked_privileged_fields": leaked,
        "control_arm": {
            "internal_previous_key_mask_changed": True,
            "pixels_unchanged": internal_change_no_pixel_change,
            "frame_sha_a": frame_a.sha256,
            "frame_sha_b": frame_b.sha256,
        },
    }


def _h7_legal_action_surface() -> dict:
    """H7: the V1 adapter actuates only W/A/S/D for maze_chase; every
    locomotion value maps within the frozen allow-list; the environment
    input pipeline drops any key outside the WASD support mask."""
    allowed = {0x1A, 0x04, 0x16, 0x07}  # W A S D
    cases = []
    all_ok = True
    for loc in LOCOMOTION:
        keys = translate_record(EmbodiedInterfaceRecord(locomotion=loc))
        within = set(keys) <= allowed
        cases.append({"locomotion": loc, "keys": list(keys), "within_allow_list": within})
        all_ok = all_ok and within
    # buttons/look/cursor/hotbar never contribute maze_chase keys:
    record = EmbodiedInterfaceRecord(
        locomotion="NOOP",
        look_yaw_deg=30,
        look_pitch_deg=-30,
        cursor_dx_px=64,
        cursor_dy_px=-64,
        buttons=(True, True, True, True, True, False),
    )
    keys = translate_record(record)
    within = set(keys) & allowed == set()  # no WASD key from buttons/look alone
    all_ok = all_ok and within
    # the environment's own pipeline drops non-WASD keys:
    h = PacmanHarness(0)
    env = h.build_env(partition_seeds(0, "TEST")[0])
    for action_class in range(5):
        mask = env._mask_from_control(control_for_class(action_class))  # noqa: SLF001
        if action_class == 0:
            all_ok = all_ok and mask == 0
        else:
            all_ok = all_ok and mask in (0x01, 0x02, 0x04, 0x08)
    return {
        "gate": "H7",
        "passed": all_ok,
        "allow_list": sorted(allowed),
        "non_wasd_keys_from_buttons": list(keys),
        "cases": cases,
    }


# --- Driver ------------------------------------------------------------------


def run_all_gates(full: bool) -> dict:
    started = time.time()
    gates = {
        "H1": _h1_determinism(full),
        "H2": _h2_no_pause_schedule(),
        "H3": _h3_families_valid(full),
        "H4": _h4_perturbations_live(full),
        "H5": _h5_partition_disjointness(),
        "H6": _h6_leakage_audit(),
        "H7": _h7_legal_action_surface(),
    }
    return {
        "mode": "pacman_harness_gates_v1",
        "preregistration": "brain/docs/preregistrations/2026-08-27-pacman-harness-v1.md",
        "scale": "full" if full else "smoke",
        "started_unix": int(started),
        "wall_seconds": round(time.time() - started, 3),
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the frozen registered scale")
    parser.add_argument("--json", default=None, help="write the JSON report to this path")
    args = parser.parse_args()
    report = run_all_gates(full=args.full)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"report -> {out}")
    for name, gate in report["gates"].items():
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] {name}")
    print(f"all_passed={report['all_passed']} wall={report['wall_seconds']}s")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
