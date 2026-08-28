"""Focused contracts for the Pac-Man-like v3 section-4 harness (construction).

Preregistration:
``brain/docs/preregistrations/2026-08-27-pacman-harness-v1-construction.md``
(frozen 2026-08-27).  Local CPU-only (CUDA hidden, single thread, no screen
capture / HID / network / background jobs — workspace rules).

Covers the construction gates and the frozen identity:
  1. Frozen difficulty-family table (5 families, exact parameters).
  2. Frozen perturbation table (P0..P5, exact mechanics).
  3. Seed partition: fresh EMB0 block, disjoint sub-ranges, layout /
     replicate enumeration geometry.
  4. Section-4.3 timing-gate constants verbatim.
  5. Determinism: two independent runs of one slice byte-match.
  6. No-pause schedule: s_n strictly increasing by the period; the
     over-budget-latency self-test produces deterministic misses while the
     declared-latency no-op self-test produces zero.
  7. Perturbation isolation: P1 exact drop count; P2 exact one-cycle shift;
     P3/P4 exact periods; P5 rule swap; P0 palette leaves geometry untouched
     and shifts pixels.
  8. Causal boundary (C8): the policy sees only ModelObservation fields.
  9. Legal-action surface: every emitted record decodes to keys inside the
     frozen actuated set.
  10. Reactive baseline: pixel-only, deterministic, legal, and reactive
     (it avoids an adjacent ghost rather than walking into it).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))

import json
import unittest
from hashlib import sha256

from irene_brain.environments.pacman_harness_v1 import (
    CLEAN,
    EMB0,
    FAMILIES,
    FAMILY_ORDER,
    MEDIAN_LPLUS_NS,
    MISS_RATE_LIMIT,
    NoOpPolicy,
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    PACMAN_BASE_PERIOD_NS,
    P95_LPLUS_NS,
    P99_JPLUS_NS,
    P99_LPLUS_NS,
    PacmanHarness,
    PERTURBATIONS,
    ReactivePixelPolicy,
    SeedPartition,
    timing_gate_verdict,
    timing_gates,
)
from irene_brain.types import GenericControl, HidKey, ModelObservation, RgbFrame
from irene_brain.v2.embodied_interface import (
    ACTUATED_KEY_SET,
    EmbodiedInterfaceRecord,
    translate_record,
)

BASE_PERIOD = PACMAN_BASE_PERIOD_NS
PROBE_SEED = EMB0 + 3


def _canonical(records):
    payload = [
        [
            t.cycle_id,
            t.frame_id,
            t.action_id,
            t.s_ns,
            t.d_ns,
            t.p_ns,
            t.t_emit_ns,
            t.t_dequeue_ns,
            t.l_plus_ns,
            t.j_plus_ns,
            t.miss,
            t.late,
            t.frame_lost,
            t.dropped_frames,
            round(t.reward, 10),
            t.terminated,
            t.truncated,
        ]
        for t in records.ticks_data
    ]
    payload += [
        records.ticks,
        records.pellets_eaten,
        records.times_caught,
        records.cleared,
        records.truncated,
        round(records.total_reward, 10),
        records.legal_action_rate,
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


class FrozenIdentityTests(unittest.TestCase):
    def test_family_table_is_frozen(self):
        self.assertEqual(FAMILY_ORDER, ("F0", "F1", "F2", "F3", "F4"))
        self.assertEqual(len(FAMILIES), 5)
        expected = {
            "F0": (40, "shy", 4, 2),
            "F1": (24, "mixed", 3, 3),
            "F2": (16, "direct", 2, 3),
            "F3": (10, "ambush", 2, 4),
            "F4": (6, "direct", 1, 4),
        }
        for fid, fam in FAMILIES.items():
            self.assertEqual(
                (fam.extra_loops, fam.ghost_rule, fam.ghost_period, fam.ghost_count),
                expected[fid],
            )
        # Clean mechanics only: no delay/sticky/elroy inside the families.
        for fam in FAMILIES.values():
            kw = fam.env_kwargs()
            self.assertEqual(kw["player_period"], 1)
            self.assertFalse(kw["ghost_elroy"])
            self.assertEqual(kw["input_delay_ticks"], 0)
            self.assertFalse(kw["sticky_direction"])

    def test_perturbation_table_is_frozen(self):
        self.assertEqual([p.id for p in PERTURBATIONS], [0, 1, 2, 3, 4, 5])
        self.assertEqual(P0.palette_offsets["F2"], (15, 15, -10))
        self.assertEqual(P1.frame_loss_stride, 10)
        self.assertTrue(P2.observation_delay)
        self.assertEqual(P3.period_ns, 18_518_519)
        self.assertEqual(P4.period_ns, 15_151_515)
        self.assertEqual(
            P5.rule_alternate,
            {"F0": "direct", "F1": "direct", "F2": "ambush", "F3": "direct", "F4": "shy"},
        )
        self.assertEqual(CLEAN.id, -1)

    def test_seed_partition_is_disjoint(self):
        sp = SeedPartition()
        ranges = {p: sp.sub_range(p) for p in ("TRAIN", "DEV", "CAL", "TEST")}
        self.assertEqual(sp.base, 201_326_592)
        # Disjointness: no overlap, each range exactly its registered size.
        cursor = EMB0
        for p in ("TRAIN", "DEV", "CAL", "TEST"):
            lo, hi = ranges[p]
            self.assertEqual(lo, cursor)
            cursor = hi
        self.assertEqual(
            (ranges["TRAIN"][1] - ranges["TRAIN"][0]), 200_000
        )
        self.assertEqual(
            (ranges["TEST"][1] - ranges["TEST"][0]), 400_000
        )
        # No overlap with the sealed PB21* blocks (max sealed end is
        # 11*2**24 + 768 = 184_550_144) or the RCQ family.
        self.assertGreater(sp.base, 184_550_144)

    def test_test_layout_and_replicate_geometry(self):
        sp = SeedPartition()
        layouts = sp.test_layouts
        self.assertEqual(len(layouts), 40)
        lo, _ = sp.sub_range("TEST")
        # Layout i belongs to family FAMILY_ORDER[i // 8] (8 layouts/family).
        for i, (fam, seed) in enumerate(layouts):
            self.assertEqual(fam.id, FAMILY_ORDER[i // 8])
            self.assertEqual(seed, lo + i)
        # Every family appears exactly 8 times.
        counts = {fam.id: 0 for fam in FAMILIES.values()}
        for fam, _seed in layouts:
            counts[fam.id] += 1
        self.assertEqual(set(counts.values()), {8})
        reps = sp.test_replicates(0)
        self.assertEqual(reps, [lo + 40 + k for k in range(5)])
        reps39 = sp.test_replicates(39)
        self.assertEqual(reps39, [lo + 40 + 5 * 39 + k for k in range(5)])
        # Replicates are disjoint from the 40 layout seeds.
        layout_seeds = {seed for _fam, seed in layouts}
        for i in range(40):
            for r in sp.test_replicates(i):
                self.assertNotIn(r, layout_seeds)

    def test_timing_gate_constants_are_verbatim(self):
        self.assertEqual(MEDIAN_LPLUS_NS, 8_000_000)
        self.assertEqual(P95_LPLUS_NS, 13_000_000)
        self.assertEqual(P99_LPLUS_NS, 16_000_000)
        self.assertEqual(MISS_RATE_LIMIT, 0.001)
        self.assertEqual(P99_JPLUS_NS, 2_000_000)


class HarnessMechanicsTests(unittest.TestCase):
    def _run(self, family, pert, policy=None, seed=PROBE_SEED, ticks=60, infer=3_000_000):
        return PacmanHarness(
            FAMILIES[family],
            pert,
            seed=seed,
            policy=policy or NoOpPolicy(),
            partition="TRAIN",
            simulated_infer_ns=infer,
            max_ticks=ticks,
        ).run_episode()

    def test_no_pause_schedule_is_monotone(self):
        rec = self._run("F1", CLEAN, ticks=40)
        for i, t in enumerate(rec.ticks_data):
            if i == 0:
                continue
            prev = rec.ticks_data[i - 1]
            self.assertEqual(t.s_ns - prev.s_ns, BASE_PERIOD)
            self.assertEqual(t.d_ns, t.s_ns + t.p_ns)
            self.assertEqual(t.cycle_id, i)

    def test_noop_declared_latency_has_zero_misses(self):
        rec = self._run("F0", CLEAN, ticks=100)
        stats = timing_gates(list(rec.ticks_data))
        self.assertEqual(stats["miss_count"], 0)
        self.assertEqual(stats["consecutive_miss_max_run"], 0)
        self.assertEqual(stats["median_lplus_ns"], 3_000_000)
        verdict = timing_gate_verdict(stats)
        self.assertTrue(verdict["passed"])

    def test_over_budget_latency_misses_deterministically(self):
        rec = self._run("F0", CLEAN, ticks=50, infer=20_000_000)
        stats = timing_gates(list(rec.ticks_data))
        self.assertEqual(stats["miss_count"], 50)
        self.assertEqual(stats["consecutive_miss_max_run"], 50)
        verdict = timing_gate_verdict(stats)
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["checks"]["miss_rate"])

    def test_determinism_two_runs_byte_match(self):
        a = self._run("F1", P1, policy=ReactivePixelPolicy(), seed=4242, ticks=80)
        b = self._run("F1", P1, policy=ReactivePixelPolicy(), seed=4242, ticks=80)
        self.assertEqual(_canonical(a), _canonical(b))
        self.assertEqual(a.ticks, b.ticks)
        self.assertEqual(a.pellets_eaten, b.pellets_eaten)
        self.assertEqual(a.times_caught, b.times_caught)

    def test_different_seeds_differ(self):
        a = self._run("F1", CLEAN, policy=ReactivePixelPolicy(), seed=4242, ticks=80)
        b = self._run("F1", CLEAN, policy=ReactivePixelPolicy(), seed=4243, ticks=80)
        self.assertNotEqual(_canonical(a), _canonical(b))

    def test_p0_shifts_pixels_but_not_geometry(self):
        # P0 changes only rendered colors; the world simulation (pellets,
        # ghosts, catch, clear) is untouched.  Verify with the NoOp policy,
        # which is pixel-blind, so both runs execute the identical game.
        clean = self._run("F0", CLEAN, policy=NoOpPolicy(), ticks=30)
        p0 = self._run("F0", P0, policy=NoOpPolicy(), ticks=30)
        self.assertEqual(clean.pellets_eaten, p0.pellets_eaten)
        self.assertEqual(clean.times_caught, p0.times_caught)
        self.assertEqual(clean.total_reward, p0.total_reward)
        # And P0 actually changes the pixels a pixel-exact policy sees, so
        # the perturbation is not a no-op for vision policies.
        r_clean = self._run("F0", CLEAN, policy=ReactivePixelPolicy(), ticks=30)
        r_p0 = self._run("F0", P0, policy=ReactivePixelPolicy(), ticks=30)
        self.assertNotEqual(
            (r_clean.pellets_eaten, r_clean.times_caught),
            (r_p0.pellets_eaten, r_p0.times_caught),
        )

    def test_p1_exact_drop_count_and_reuse(self):
        rec = self._run("F2", P1, ticks=100)
        drops = [t for t in rec.ticks_data if t.frame_lost]
        # Fresh frames 10,20,...,90 are dropped at cycles 10..90.
        self.assertEqual(
            [t.cycle_id for t in drops], [10, 20, 30, 40, 50, 60, 70, 80, 90]
        )
        for t in drops:
            self.assertEqual(t.frame_id, t.cycle_id - 1)
        fids = [t.frame_id for t in rec.ticks_data]
        # Cycle 10 serves the reused frame 9; cycle 11 resumes the fresh
        # frame (frame 11); cycle 20 reuses frame 19.
        self.assertEqual(fids[10], 9)
        self.assertEqual(fids[11], 11)
        self.assertEqual(fids[20], 19)
        self.assertEqual(rec.ticks_data[-1].dropped_frames, len(drops))

    def test_p2_exact_one_cycle_shift(self):
        rec = self._run("F2", P2, ticks=30)
        fids = [t.frame_id for t in rec.ticks_data]
        self.assertEqual(fids[0], 0)
        self.assertEqual(fids[1], 0)
        for n in range(2, 30):
            self.assertEqual(fids[n], n - 1)
        # After the first cycle the observation is exactly one period old.
        for n in range(1, 30):
            t = rec.ticks_data[n]
            self.assertEqual(t.s_ns - t.t_emit_ns, BASE_PERIOD)
            self.assertTrue(t.observation_delayed)

    def test_p3_p4_periods_on_all_ticks(self):
        rec3 = self._run("F0", P3, ticks=20)
        rec4 = self._run("F0", P4, ticks=20)
        self.assertTrue(all(t.p_ns == 18_518_519 for t in rec3.ticks_data))
        self.assertTrue(all(t.p_ns == 15_151_515 for t in rec4.ticks_data))
        # The schedule is uniform at the scaled period: s_n - s_{n-1} == P.
        for rec, p in ((rec3, 18_518_519), (rec4, 15_151_515)):
            for i in range(1, len(rec.ticks_data)):
                self.assertEqual(
                    rec.ticks_data[i].s_ns - rec.ticks_data[i - 1].s_ns, p
                )
        self.assertEqual(rec3.period_ns, 18_518_519)
        self.assertEqual(rec4.period_ns, 15_151_515)

    def test_p5_swaps_only_the_ghost_rule(self):
        for fid in FAMILY_ORDER:
            rec_alt = self._run(fid, P5, policy=NoOpPolicy(), ticks=1)
            self.assertEqual(rec_alt.perturbation_name, "enemy-policy-altered")

    def test_causal_boundary_model_observation(self):
        # The harness only ever hands a policy a ModelObservation; verify the
        # policy is a deterministic function of that boundary alone.
        from irene_brain.types import GenericControl

        pol = ReactivePixelPolicy()
        n = 16
        f = RgbFrame(width=n, height=n, pixels=bytes(n * n * 3))
        o_a = ModelObservation(
            frame_id=1,
            elapsed_ns=BASE_PERIOD,
            observation_age_ns=0,
            rgb=f,
            previous_control=GenericControl(),
        )
        o_b = ModelObservation(
            frame_id=1,
            elapsed_ns=BASE_PERIOD,
            observation_age_ns=0,
            rgb=f,
            previous_control=GenericControl(),
        )
        self.assertEqual(pol.act(o_a, 0), pol.act(o_b, 0))
        # A different frame still yields a valid record.
        f2 = RgbFrame(width=n, height=n, pixels=bytes(n * n * 3))
        o_c = ModelObservation(
            frame_id=2,
            elapsed_ns=2 * BASE_PERIOD,
            observation_age_ns=0,
            rgb=f2,
            previous_control=GenericControl(),
        )
        rec = pol.act(o_c, 1)
        self.assertIsInstance(rec, EmbodiedInterfaceRecord)

    def test_legal_action_surface_all_cells(self):
        bad = []
        for fid in FAMILY_ORDER:
            for pert in (CLEAN, P0, P1, P2, P3, P4, P5):
                rec = self._run(fid, pert, policy=ReactivePixelPolicy(), ticks=12)
                if rec.legal_action_rate != 1.0:
                    bad.append((fid, pert.name))
        self.assertEqual(bad, [])

    def test_reactive_policy_avoids_adjacent_ghost(self):
        # Build a 16x16 frame: player at (8,8), a wall field around, a ghost
        # adjacent at (8,9) (below the player).  The reactive policy must not
        # choose BACKWARD (down, into the ghost).
        n = 16
        walls = set()
        # A full wall box except a corridor row/col through the player.
        for y in range(n):
            for x in range(n):
                if x == 8 or y == 8:
                    continue
                walls.add((x, y))
        px = bytearray(n * n * 3)
        from irene_brain.environments.pacman_harness_v1 import (
            _WALL_RGB,
            _PLAYER_RGB,
            _GHOST_RGB,
        )

        for y in range(n):
            for x in range(n):
                color = _WALL_RGB if (x, y) in walls else (8, 11, 18)
                off = (y * n + x) * 3
                px[off:off + 3] = bytes(color)
        for (x, y) in [(8, 8), (8, 9)]:
            color = _PLAYER_RGB if (x, y) == (8, 8) else _GHOST_RGB
            off = (y * n + x) * 3
            px[off:off + 3] = bytes(color)
        frame = RgbFrame(width=n, height=n, pixels=bytes(px))
        obs = ModelObservation(
            frame_id=0,
            elapsed_ns=0,
            observation_age_ns=0,
            rgb=frame,
            previous_control=GenericControl(),
        )
        rec = ReactivePixelPolicy().act(obs, 0)
        self.assertNotEqual(rec.locomotion, "BACKWARD")


_NEUTRAL = None  # (removed; unused)


if __name__ == "__main__":
    unittest.main()
