"""Focused contracts for the Pac-Man-like v3 section-4 harness (P1-P6).

Preregistration:
``brain/docs/preregistrations/2026-08-27-pacman-harness-v1.md``
(frozen 2026-08-27, commit ``8c4c127``; P2/P4/P5 amended to the channel-level
rules in the working tree before T1-2 closed).  This module pins the
*committed* P1-P6 family / partition / perturbation identity and the clock,
determinism, perturbation-isolation, causal-integrity, and action-surface
properties that the harness build must satisfy.  Local CPU-only (CUDA hidden,
single thread, no screen capture / HID / network / background jobs).

The P1-P6 design is the one described by the committed T1-2 report
(``brain/docs/runs/2026-08-27-t12-pacman-harness-gates.md``) and the
``pacman_harness`` module.  A later, divergent reconstruction ("construction",
``pacman_harness_v1``) was retired to ``brain/scratch/retired-...`` because its
P3/P4 rescaled the control period off 60 Hz (54/66 Hz), violating the frozen
v3 section-4.3 native-control gate.  This module deliberately asserts that
P4/P5 keep the 60 Hz floor-schedule control period.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

import unittest
from hashlib import sha256

from irene_brain.environments.pacman_harness import (
    EMBODIED_BLOCK_END,
    EMBODIED_BLOCK_START,
    FAMILIES,
    P6_ALTERNATE_RULE,
    PACMAN_RENDER_CONTRACT,
    PARTITION_ORDER,
    SEALED_RANGES,
    TICK_PERIOD_NS,
    TIMING_GATES,
    Controller,
    PacmanHarness,
    action_class_of,
    all_harness_seeds,
    assert_seed_partition_disjointness,
    check_timing_gates,
    control_for_class,
    family,
    nearest_rank_quantile,
    p1_palette_offset,
    p1_remap_frame,
    p2_frame_dropped,
    p4_world_held,
    p5_world_extra_step,
    p6_family_kwargs,
    period_ns,
    partition_seeds,
    scheduled_start_ns,
)
from irene_brain.types import GenericControl
from irene_brain.v2.embodied_interface import (
    LOCOMOTION,
    EmbodiedInterfaceRecord,
    translate_record,
)

WASD_KEYS = {0x1A, 0x04, 0x16, 0x07}  # W A S D
FLOOR_PERIODS = {16_666_666, 16_666_667}  # the 30-ns-offset floor schedule


def _idle() -> Controller:
    return Controller(name="idle", fn=lambda cycle, frame: 0)


def _seed(fam: int = 1, part: str = "TEST") -> int:
    return partition_seeds(fam, part)[0]


def _run(fam: int, pert: str = "clean", *, cycles: int = 60, seed=None):
    h = PacmanHarness(fam, perturbation=pert, max_cycles=cycles)
    return h.run_episode(_seed(fam) if seed is None else seed, _idle(), partition="TEST")


class FrozenIdentityTests(unittest.TestCase):
    def test_family_table_is_frozen(self):
        self.assertEqual(len(FAMILIES), 5)
        expected = {
            "F0_calm": (1, 3, "shy", False, 24, 1, 4000),
            "F1_standard": (3, 2, "mixed", False, 16, 1, 6000),
            "F2_pressured": (4, 2, "direct", False, 12, 1, 6000),
            "F3_fast": (4, 1, "mixed", True, 16, 1, 8000),
            "F4_open_slow": (2, 3, "ambush", False, 40, 2, 8000),
        }
        for i, fam in enumerate(FAMILIES):
            self.assertEqual(
                (
                    fam.ghost_count,
                    fam.ghost_period,
                    fam.ghost_rule,
                    fam.ghost_elroy,
                    fam.extra_loops,
                    fam.player_period,
                    fam.max_ticks,
                ),
                expected[fam.name],
            )
            self.assertEqual(family(i), fam)

    def test_families_use_clean_mechanics_only(self):
        # Delay/sticky belong to the perturbation axes, not the difficulty axis.
        for fam in FAMILIES:
            kw = fam.env_kwargs()
            self.assertEqual(kw["input_delay_ticks"], 0)
            self.assertFalse(kw["sticky_direction"])
            self.assertEqual(kw["tick_period_ns"], TICK_PERIOD_NS)

    def test_perturbation_ids_are_frozen(self):
        from irene_brain.environments.pacman_harness import PERTURBATION_IDS

        self.assertEqual(
            PERTURBATION_IDS,
            (
                "clean",
                "P1_palette_shift",
                "P2_frame_loss_10pct",
                "P3_obs_delay_1",
                "P4_speed_90pct",
                "P5_speed_110pct",
                "P6_enemy_policy_change",
            ),
        )

    def test_p6_alternate_rule_is_frozen(self):
        # Committed section-2 P6 row: F0->direct F1->direct F2->shy F3->ambush
        # F4->direct.
        self.assertEqual(
            P6_ALTERNATE_RULE, ("direct", "direct", "shy", "ambush", "direct")
        )
        for i in range(len(FAMILIES)):
            self.assertEqual(p6_family_kwargs(i)["ghost_rule"], P6_ALTERNATE_RULE[i])

    def test_seed_partition_is_disjoint_and_in_block(self):
        assert_seed_partition_disjointness()  # raises on any collision
        seeds = all_harness_seeds()
        self.assertEqual(len(seeds), (40 + 100 + 100 + 524) * 5)
        for s in seeds:
            self.assertGreaterEqual(s, EMBODIED_BLOCK_START)
            self.assertLess(s, EMBODIED_BLOCK_END)
        # The fresh block sits above every sealed/retired range in the repo.
        for lo, hi in SEALED_RANGES:
            self.assertGreater(EMBODIED_BLOCK_START, hi - 1 if hi > lo else lo)

    def test_partition_counts_per_family(self):
        for fam in range(len(FAMILIES)):
            self.assertEqual(len(partition_seeds(fam, "TEST")), 40)
            self.assertEqual(len(partition_seeds(fam, "CAL")), 100)
            self.assertEqual(len(partition_seeds(fam, "DEV")), 100)
            self.assertEqual(len(partition_seeds(fam, "TRAIN")), 524)

    def test_timing_gate_constants_are_verbatim(self):
        # v3 section 4.3, Pac-Man-like column, verbatim.
        self.assertEqual(TIMING_GATES["median_l_plus_ns"], 8_000_000)
        self.assertEqual(TIMING_GATES["p95_l_plus_ns"], 13_000_000)
        self.assertEqual(TIMING_GATES["p99_l_plus_ns"], 16_000_000)
        self.assertEqual(TIMING_GATES["deadline_miss_rate"], 0.001)
        self.assertEqual(TIMING_GATES["p99_j_plus_ns"], 2_000_000)
        self.assertEqual(TIMING_GATES["max_consecutive_misses"], 0)
        # A clean offline replay (0-ns latency) clears every gate.
        clean_summary = {
            "median_l_plus_ns": 0,
            "p95_l_plus_ns": 0,
            "p99_l_plus_ns": 0,
            "miss_rate": 0.0,
            "p99_j_plus_ns": 0,
            "max_consecutive_misses": 0,
        }
        self.assertTrue(check_timing_gates(clean_summary)["passed"])


class ClockScheduleTests(unittest.TestCase):
    def test_scheduled_start_matches_floor_formula(self):
        for n in (0, 1, 2, 3, 99, 100, 101, 5000, 5999):
            self.assertEqual(
                scheduled_start_ns(n), (n * 1_000_000_000 + 30) // 60
            )
            self.assertIn(period_ns(n), FLOOR_PERIODS)

    def test_no_pause_schedule_in_episode_records(self):
        rec = _run(0, "clean", cycles=40)
        for i, r in enumerate(rec.records):
            self.assertEqual(r.scheduled_ns, scheduled_start_ns(i))
            self.assertEqual(r.deadline_ns, scheduled_start_ns(i + 1))
            self.assertIn(r.period_ns, FLOOR_PERIODS)
            self.assertFalse(r.is_miss)
        # No deadline miss, ever, in a clean offline replay.
        self.assertEqual(sum(1 for r in rec.records if r.is_miss), 0)

    def test_nearest_rank_quantile(self):
        vals = [1, 2, 3, 4, 5]
        self.assertEqual(nearest_rank_quantile(vals, 0.50), 3)
        self.assertEqual(nearest_rank_quantile(vals, 0.99), 5)
        self.assertEqual(nearest_rank_quantile(vals, 0.01), 1)


class DeterminismTests(unittest.TestCase):
    def _canonical(self, rec):
        payload = [
            [
                r.cycle_id,
                r.frame_id,
                r.action_id,
                r.scheduled_ns,
                r.period_ns,
                r.world_steps_this_cycle,
                r.action_class,
                r.frame_sha256,
                r.delivered_frame_sha256,
                round(r.reward, 10),
            ]
            for r in rec.records
        ]
        payload += [
            rec.pellets_eaten,
            rec.times_caught,
            rec.cleared,
            round(rec.total_reward, 10),
            rec.frame_sha256_stream_sha256,
            rec.env_state_hash_final,
        ]
        import json

        return sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def test_two_runs_byte_match(self):
        a = _run(1, "clean", cycles=80, seed=_seed())
        b = _run(1, "clean", cycles=80, seed=_seed())
        self.assertEqual(self._canonical(a), self._canonical(b))
        self.assertEqual(a.pellets_eaten, b.pellets_eaten)
        self.assertEqual(a.times_caught, b.times_caught)

    def test_different_seeds_differ(self):
        a = _run(1, "clean", cycles=80, seed=partition_seeds(1, "TRAIN")[0])
        b = _run(1, "clean", cycles=80, seed=partition_seeds(1, "TRAIN")[1])
        self.assertNotEqual(self._canonical(a), self._canonical(b))


class PerturbationIsolationTests(unittest.TestCase):
    def test_p1_shifts_pixels_not_geometry(self):
        clean = _run(0, "clean", cycles=30)
        p1 = _run(0, "P1_palette_shift", cycles=30)
        # Same world trajectory (palette is render-only)...
        self.assertEqual(clean.env_state_hash_final, p1.env_state_hash_final)
        self.assertEqual(clean.pellets_eaten, p1.pellets_eaten)
        # ...but a different rendered pixel stream.
        self.assertNotEqual(
            clean.frame_sha256_stream_sha256, p1.frame_sha256_stream_sha256
        )
        # The offset is process-stable (zlib.crc32), nonzero, and deterministic.
        self.assertGreater(p1_palette_offset(12345), 0)
        self.assertEqual(p1_palette_offset(12345), p1_palette_offset(12345))

    def test_p2_exact_drop_count_and_reuse(self):
        rec = _run(2, "P2_frame_loss_10pct", cycles=50)
        holds = [r.cycle_id for r in rec.records if r.is_hold]
        self.assertEqual(holds, [10, 20, 30, 40])  # frames 10,20,30,40 dropped
        # The helper encodes the frozen rule: g>=1 and g % 10 == 0.
        self.assertFalse(p2_frame_dropped(0))
        self.assertTrue(p2_frame_dropped(10))
        self.assertFalse(p2_frame_dropped(11))

    def test_p3_exact_one_cycle_shift(self):
        hc = PacmanHarness(1, max_cycles=40)
        rc = hc.run_episode(_seed(1), _idle(), partition="TEST")
        hd = PacmanHarness(1, perturbation="P3_obs_delay_1", max_cycles=40)
        rd = hd.run_episode(_seed(1), _idle(), partition="TEST")
        # Cycle 0 sees the reset frame; cycle n (n>=1) sees clean frame n-1.
        self.assertEqual(
            rd.records[0].delivered_frame_sha256, rc.records[0].frame_sha256
        )
        for n in range(1, len(rc.records)):
            self.assertEqual(
                rd.records[n].delivered_frame_sha256,
                rc.records[n - 1].frame_sha256,
            )

    def test_p4_world_step_accounting_90(self):
        rec = _run(1, "P4_speed_90pct", cycles=50)
        held = [r.cycle_id for r in rec.records if r.world_steps_this_cycle == 0]
        self.assertEqual(held, [0, 10, 20, 30, 40])  # world held on the 10-grid
        self.assertEqual(
            sum(r.world_steps_this_cycle for r in rec.records), 45
        )  # 50 - 5 held
        self.assertTrue(p4_world_held(0))
        self.assertTrue(p4_world_held(30))
        self.assertFalse(p4_world_held(31))

    def test_p5_world_step_accounting_110(self):
        rec = _run(1, "P5_speed_110pct", cycles=50)
        extra = [r.cycle_id for r in rec.records if r.world_steps_this_cycle == 2]
        self.assertEqual(extra, [0, 10, 20, 30, 40])  # extra step on the 10-grid
        self.assertEqual(
            sum(r.world_steps_this_cycle for r in rec.records), 55
        )  # 50 + 5 extra
        self.assertTrue(p5_world_extra_step(0))
        self.assertTrue(p5_world_extra_step(40))
        self.assertFalse(p5_world_extra_step(41))

    def test_p4_p5_keep_60hz_control_period(self):
        # The critical property that distinguishes the conforming P1-P6 design
        # from the retired reconstruction: P4/P5 perturb the world/agent
        # relative pace at the channel level but NEVER rescale the control
        # period.  Every record keeps the 60 Hz floor-schedule period.
        for pert in ("P4_speed_90pct", "P5_speed_110pct"):
            rec = _run(1, pert, cycles=30)
            for i, r in enumerate(rec.records):
                self.assertEqual(r.scheduled_ns, scheduled_start_ns(i))
                self.assertIn(r.period_ns, FLOOR_PERIODS)

    def test_p6_changes_dynamics_only(self):
        clean = _run(1, "clean", cycles=40)
        p6 = _run(1, "P6_enemy_policy_change", cycles=40)
        self.assertNotEqual(clean.env_state_hash_final, p6.env_state_hash_final)


class CausalBoundaryTests(unittest.TestCase):
    def test_observation_carries_no_privileged_fields(self):
        from irene_brain.environments.maze_chase import MazeChaseEnv
        from dataclasses import fields as dc_fields

        h = PacmanHarness(1)
        env = h.build_env(_seed(1))
        obs = env.current_observation
        obs_fields = set(f.name for f in dc_fields(type(obs)))
        privileged = {
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
        self.assertEqual(obs_fields & privileged, set())

    def test_control_arm_internal_change_no_pixel_change(self):
        h = PacmanHarness(1)
        env = h.build_env(_seed(1))
        frame_a = env.current_observation.rgb
        h2 = PacmanHarness(1)
        env2 = h2.build_env(_seed(1))
        env2._previous_key_mask = 0x0F  # noqa: SLF001 (control-arm internal)
        frame_b = env2.current_observation.rgb
        # Changing an internal, non-rendered state leaves the pixels identical:
        # the observation is a pure function of rendered pixels (C8).
        self.assertEqual(frame_a.pixels, frame_b.pixels)


class ActionSurfaceTests(unittest.TestCase):
    def test_all_locomotion_maps_to_wasd_allowlist(self):
        for loc in LOCOMOTION:
            keys = translate_record(EmbodiedInterfaceRecord(locomotion=loc))
            self.assertTrue(set(keys) <= WASD_KEYS, loc)

    def test_non_locomotion_fields_add_no_wasd_keys(self):
        record = EmbodiedInterfaceRecord(
            locomotion="NOOP",
            look_yaw_deg=30,
            look_pitch_deg=-30,
            cursor_dx_px=64,
            cursor_dy_px=-64,
            buttons=(True, True, True, True, True, False),
        )
        keys = translate_record(record)
        self.assertEqual(set(keys) & WASD_KEYS, set())

    def test_env_input_pipeline_masks_to_wasd(self):
        h = PacmanHarness(0)
        env = h.build_env(partition_seeds(0, "TEST")[0])
        for action_class in range(5):
            mask = env._mask_from_control(control_for_class(action_class))  # noqa: SLF001
            if action_class == 0:
                self.assertEqual(mask, 0)
            else:
                self.assertIn(mask, (0x01, 0x02, 0x04, 0x08))

    def test_action_class_roundtrip(self):
        for ac in range(5):
            self.assertEqual(action_class_of(ac), ac)


if __name__ == "__main__":
    unittest.main()
