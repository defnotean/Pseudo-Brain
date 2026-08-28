"""Pac-Man-like (maze_chase) embodied harness v1.

Implements the frozen harness contract of
``brain/docs/preregistrations/2026-08-27-pacman-harness-v1.md``:

- the no-pause 60 Hz clock/queue discipline (v3 section 4): exact
  scheduled starts ``s_n = t0 + floor((n * 1_000_000_000 + 30) / 60)``,
  deadlines ``d_n = s_(n+1)``, capacity-one latest-complete-frame queue,
  and conservative ``L_plus`` / ``J_plus`` latency and jitter;
- the five frozen difficulty families F0..F4 (prereg section 1);
- the six frozen perturbations P1..P6 (prereg section 2);
- the registered TRAIN/DEV/CAL/TEST seed partitions (prereg section 3);
- the H1-H7 harness-build acceptance gates (prereg section 6), run by
  ``brain/scripts/pacman_harness_gates.py``.

The harness wraps ``MazeChaseEnv`` (the original, rights-clean 60 Hz
Pac-Man-like world) and never hands the model any privileged state:
only the ``ModelObservation`` boundary crosses into the candidate
(constitution C8).  Deterministic offline replay is exact (manual clock);
wall-clock live timing (Q1, Tier-3) reuses the same record schema with
real ``time.perf_counter_ns`` receipts.

Determinism: the harness itself uses no randomness; all stochasticity is
the wrapped environment's seeded ``_SplitMix64`` stream plus the P1 byte
offset (CRC32-derived, process-stable).  Seed banks elsewhere in the
program use ``zlib.crc32``; never the process-salted builtin ``hash()``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from struct import Struct
from typing import Callable, Mapping, Sequence, Tuple

from .maze_chase import MazeChaseEnv
from ..types import GenericControl, RgbFrame

# --- Frozen constants (preregistration 2026-08-27-pacman-harness-v1.md) ---

TICK_PERIOD_NS = 16_666_667
EPOCH_OFFSET_NS = 30  # s_n = t0 + floor((n * 1e9 + 30) / 60)

EMBODIED_BLOCK_START = 201_326_592  # 12 * 2**24
EMBODIED_BLOCK_END = 201_347_072  # 12 * 2**24 + 20480

FAMILY_COUNT = 5
FAMILY_STRIDE = 4096
PARTITION_STRIDE = 1024
TEST_PER_FAMILY = 40
CAL_PER_FAMILY = 100
DEV_PER_FAMILY = 100
TRAIN_PER_FAMILY = 524

PARTITION_ORDER: tuple[str, ...] = ("TRAIN", "DEV", "CAL", "TEST")
_PARTITION_INDEX = {name: i for i, name in enumerate(PARTITION_ORDER)}

# Sealed/retired seed ranges already in the repo (verified 2026-08-27).
# No harness seed may collide with any of them (H5).
SEALED_RANGES: tuple[Tuple[int, int], ...] = (
    (1_048_576, 1_050_112),
    (2_097_152, 2_097_920),
    (3_145_728, 3_147_264),
    (4_194_304, 4_195_840),
    (33_554_432, 33_554_624),
    (100_663_296, 100_664_448),
    (117_440_512, 117_440_896),
    (134_217_728, 134_219_264),
    (150_994_944, 150_995_712),
    (167_772_160, 167_774_464),
    (184_549_376, 184_550_144),
)

# Core V2's five action classes: 0=idle, 1=W, 2=A, 3=S, 4=D.
_ACTION_KEYS: Tuple[int, ...] = (0x1A, 0x04, 0x16, 0x07)  # W A S D

# P1 palette entries, in the order the prereg names them:
# PLAYER, PELLET, GHOST, WALL, _BACKGROUND_EVEN, _BACKGROUND_ODD.
_PALETTE_RGB: Tuple[Tuple[int, int, int], ...] = (
    MazeChaseEnv.PLAYER_RGB,
    MazeChaseEnv.PELLET_RGB,
    (225, 55, 65),  # MazeChaseEnv._GHOST_COLOR
    MazeChaseEnv.WALL_RGB,
    (8, 11, 18),  # MazeChaseEnv._BACKGROUND_EVEN
    (10, 14, 22),  # MazeChaseEnv._BACKGROUND_ODD
)

_P1_DOMAIN = b"PACMANHARNESS.P1.PALETTE\x01"

# The complete, public render contract: the six colors MazeChaseEnv renders
# (16x16 grid, one color per cell).  Pixel-only gate checks (H3/H4/H6)
# classify frames using ONLY these values — the simulator's coordinates,
# pellets, ghosts, and reward never enter.
PACMAN_RENDER_CONTRACT: Mapping[str, tuple[int, int, int]] = {
    "player": MazeChaseEnv.PLAYER_RGB,
    "pellet": MazeChaseEnv.PELLET_RGB,
    "ghost": (225, 55, 65),
    "wall": MazeChaseEnv.WALL_RGB,
    "background_even": (8, 11, 18),
    "background_odd": (10, 14, 22),
    "player_ghost_overlap": (255, 255, 255),
}


# --- Seed partitions (preregistration section 3) ----------------------------


def partition_base(family: int, partition: str) -> int:
    """``base(f, p) = 201326592 + f * 4096 + pi * 1024`` (prereg section 3)."""
    if isinstance(family, bool) or not isinstance(family, int) or not (0 <= family < FAMILY_COUNT):
        raise ValueError(f"family must be in [0, {FAMILY_COUNT}), got {family}")
    if partition not in _PARTITION_INDEX:
        raise ValueError(f"partition must be one of {PARTITION_ORDER}, got {partition!r}")
    return EMBODIED_BLOCK_START + family * FAMILY_STRIDE + _PARTITION_INDEX[partition] * PARTITION_STRIDE


def partition_seeds(family: int, partition: str) -> tuple[int, ...]:
    """Registered seeds for one (family, partition).

    TEST  = base+0..base+39 (40 layouts);
    CAL   = base+100..base+199 (100);
    DEV   = base+400..base+499 (100);
    TRAIN = base+500..base+1023 (524).
    """
    base = partition_base(family, partition)
    spans = {
        "TEST": (0, TEST_PER_FAMILY),
        "CAL": (100, CAL_PER_FAMILY),
        "DEV": (400, DEV_PER_FAMILY),
        "TRAIN": (500, TRAIN_PER_FAMILY),
    }
    offset, count = spans[partition]
    return tuple(base + offset + i for i in range(count))


def all_harness_seeds() -> set[int]:
    return {
        seed
        for fam in range(FAMILY_COUNT)
        for part in PARTITION_ORDER
        for seed in partition_seeds(fam, part)
    }


def assert_seed_partition_disjointness() -> None:
    """H5: every (family, partition) seed set is pairwise disjoint and free
    of collisions with the repo's sealed/retired ranges."""
    seen: dict[int, tuple[int, str]] = {}
    for fam in range(FAMILY_COUNT):
        for part in PARTITION_ORDER:
            for seed in partition_seeds(fam, part):
                if seed in seen:
                    raise AssertionError(
                        f"seed {seed} appears in both {seen[seed]} and ({fam}, {part})"
                    )
                if not (EMBODIED_BLOCK_START <= seed < EMBODIED_BLOCK_END):
                    raise AssertionError(f"seed {seed} outside the reserved embodied block")
                for lo, hi in SEALED_RANGES:
                    if lo <= seed < hi:
                        raise AssertionError(f"seed {seed} collides with sealed range [{lo}, {hi})")
                seen[seed] = (fam, part)


# --- Difficulty families (preregistration section 1) -----------------------


@dataclass(frozen=True, slots=True)
class DifficultyFamily:
    """One frozen maze_chase constructor tuple (prereg section 1 table)."""

    name: str
    ghost_count: int
    ghost_period: int
    ghost_rule: str
    ghost_elroy: bool
    extra_loops: int
    player_period: int
    max_ticks: int

    def env_kwargs(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = dict(
            ghost_count=self.ghost_count,
            ghost_period=self.ghost_period,
            ghost_rule=self.ghost_rule,
            ghost_elroy=self.ghost_elroy,
            extra_loops=self.extra_loops,
            player_period=self.player_period,
            max_ticks=self.max_ticks,
            input_delay_ticks=0,
            sticky_direction=False,
            tick_period_ns=TICK_PERIOD_NS,
        )
        base.update(overrides)
        return base


FAMILIES: tuple[DifficultyFamily, ...] = (
    DifficultyFamily("F0_calm", 1, 3, "shy", False, 24, 1, 4000),
    DifficultyFamily("F1_standard", 3, 2, "mixed", False, 16, 1, 6000),
    DifficultyFamily("F2_pressured", 4, 2, "direct", False, 12, 1, 6000),
    DifficultyFamily("F3_fast", 4, 1, "mixed", True, 16, 1, 8000),
    DifficultyFamily("F4_open_slow", 2, 3, "ambush", False, 40, 2, 8000),
)

# P6 frozen alternate ghost rule per family (prereg section 2, P6 row).
P6_ALTERNATE_RULE: Tuple[str, ...] = ("direct", "direct", "shy", "ambush", "direct")


def family(index: int) -> DifficultyFamily:
    if isinstance(index, bool) or not isinstance(index, int) or not (0 <= index < len(FAMILIES)):
        raise ValueError(f"family index must be in [0, {len(FAMILIES)})")
    return FAMILIES[index]


# --- Perturbation rules (preregistration section 2) ------------------------

PERTURBATION_IDS: Tuple[str, ...] = (
    "clean",
    "P1_palette_shift",
    "P2_frame_loss_10pct",
    "P3_obs_delay_1",
    "P4_speed_90pct",
    "P5_speed_110pct",
    "P6_enemy_policy_change",
)


def p1_palette_offset(seed: int) -> int:
    """The P1 per-episode byte offset: CRC32-based, nonzero, deterministic.

    Nonzero so P1 always differs from clean (H4 requires a pixel diff).
    """
    import zlib

    digest = zlib.crc32(_P1_DOMAIN + seed.to_bytes(8, "big"))
    return 7 + digest % 235  # in [7, 241]


def p1_remap_frame(frame: RgbFrame, seed: int) -> RgbFrame:
    """Apply the P1 palette shift to one rendered frame.

    Exactly the six registered color values are remapped (mod-256 wrap);
    every other pixel triple is byte-identical.  Geometry (cell identity by
    position) is preserved exactly while every palette pixel changes.
    """
    offset = p1_palette_offset(seed)
    table = {}
    for r, g, b in _PALETTE_RGB:
        value = (r << 16) | (g << 8) | b
        table[value] = ((r + offset) % 256, (g + offset) % 256, (b + offset) % 256)
    out = bytearray(frame.pixels)
    for i in range(0, len(out), 3):
        value = (out[i] << 16) | (out[i + 1] << 8) | out[i + 2]
        shifted = table.get(value)
        if shifted is not None:
            out[i], out[i + 1], out[i + 2] = shifted
    return RgbFrame(width=frame.width, height=frame.height, pixels=bytes(out))


def p2_frame_dropped(frame_index: int) -> bool:
    """P2: frame g (g >= 1) is dropped iff g mod 10 == 0; the reset frame
    (g=0) is never dropped (prereg section 2, P2 row)."""
    return frame_index >= 1 and frame_index % 10 == 0


def p4_world_held(cycle_index: int) -> bool:
    """P4: at every cycle n with n mod 10 == 0 (including n = 0) the world
    is held (0.9x); exactly 10 of any 100 aligned cycles are held."""
    return cycle_index % 10 == 0


def p5_world_extra_step(cycle_index: int) -> bool:
    """P5: at every cycle n with n mod 10 == 0 (including n = 0) the world
    steps twice (1.1x); exactly 10 of any 100 aligned cycles get the extra
    step."""
    return cycle_index % 10 == 0


def p6_family_kwargs(family_index: int) -> dict[str, object]:
    """P6: the frozen alternate ghost rule for the family (prereg P6 row)."""
    return {"ghost_rule": P6_ALTERNATE_RULE[family_index]}


# --- Clock contract (preregistration section 4, verbatim v3 section 4.2) ---


# --- Frozen Pac-Man-like timing gates (v3 section 4.3, verbatim) -----------
# The Tier-3 live/scored battery evaluates ``EpisodeResult.timing_summary``
# against these limits.  Native control rate stays 60 Hz under every
# registered perturbation (P4/P5 perturb the world/agent relative pace at
# the channel level; they do not rescale the control period).
TIMING_GATES: Mapping[str, object] = {
    "median_l_plus_ns": 8_000_000,        # <= 8.00 ms
    "p95_l_plus_ns": 13_000_000,          # <= 13.00 ms
    "p99_l_plus_ns": 16_000_000,          # <= 16.00 ms
    "deadline_miss_rate": 0.001,          # <= 0.1%
    "p99_j_plus_ns": 2_000_000,           # <= 2.00 ms
    "max_consecutive_misses": 0,          # zero
}


def check_timing_gates(summary: Mapping[str, object]) -> dict[str, object]:
    """Per-gate pass/fail of one ``timing_summary`` dict against
    ``TIMING_GATES``.  Returns ``{"passed": bool, "gates": {...}}``."""
    gates: dict[str, object] = {
        "median_l_plus": summary["median_l_plus_ns"] <= TIMING_GATES["median_l_plus_ns"],  # noqa: F401
        "p95_l_plus": summary["p95_l_plus_ns"] <= TIMING_GATES["p95_l_plus_ns"],
        "p99_l_plus": summary["p99_l_plus_ns"] <= TIMING_GATES["p99_l_plus_ns"],
        "deadline_miss_rate": summary["miss_rate"] <= TIMING_GATES["deadline_miss_rate"],
        "p99_j_plus": summary["p99_j_plus_ns"] <= TIMING_GATES["p99_j_plus_ns"],
        "max_consecutive_misses": summary["max_consecutive_misses"]
        <= TIMING_GATES["max_consecutive_misses"],
    }
    return {"passed": all(gates.values()), "gates": gates}


def scheduled_start_ns(n: int, t0: int = 0) -> int:
    """``s_n = t0 + floor((n * 1_000_000_000 + 30) / 60)`` ns."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    return t0 + (n * 1_000_000_000 + EPOCH_OFFSET_NS) // 60


def period_ns(n: int, t0: int = 0) -> int:
    """``P_n = s_(n+1) - s_n``: the cycle's integer-nanosecond period."""
    return scheduled_start_ns(n + 1, t0) - scheduled_start_ns(n, t0)


def nearest_rank_quantile(sorted_values: Sequence[int], probability: float) -> int:
    """Nearest-rank quantile over a pre-sorted non-decreasing sequence.

    rank = ceil(p * N); return values[rank - 1].  v3 section 4.2: "Quantiles
    use the nearest-rank definition over all eligible ticks."
    """
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    n = len(sorted_values)
    if n == 0:
        raise ValueError("empty sequence")
    import math

    rank = math.ceil(probability * n)
    rank = min(max(rank, 1), n)
    return sorted_values[rank - 1]


# --- Records -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CycleRecord:
    """One native cycle's clock, queue, and latency evidence (v3 section 4).

    In deterministic offline replay all timestamps equal the scheduled start
    (inference is declared instant, 0 ns bookkeeping cost); in live Q1 mode
    the same schema carries real ``time.perf_counter_ns`` receipts, so
    ``L_plus``/``J_plus``/miss definitions are identical across modes.
    Epsilon terms are 0 (single host clock domain).
    """

    cycle_id: int
    frame_id: int
    action_id: int
    scheduled_ns: int
    deadline_ns: int
    period_ns: int
    t_emit: int
    t_enqueue: int
    t_dequeue: int
    t_infer_start: int
    t_infer_end: int
    t_dispatch: int
    t_accept: int
    world_tick_before: int
    world_tick_after: int
    world_steps_this_cycle: int
    queue_event: str  # "dequeue" | "hold"
    is_miss: bool
    is_hold: bool
    action_class: int  # 0..4 = idle/W/A/S/D applied this cycle
    frame_sha256: str  # the frame rendered this cycle (post-perturbation)
    delivered_frame_sha256: str  # the frame the controller saw
    reward: float
    events: tuple[str, ...]
    terminated: bool
    truncated: bool

    @property
    def l_plus_ns(self) -> int:
        """Conservative authoritative latency (v3 section 4.1)."""
        return self.t_accept - self.t_emit

    def j_plus_ns(self, previous: "CycleRecord") -> int:
        """Per-cycle schedule jitter (v3 section 4.2)."""
        return abs((self.t_accept - previous.t_accept) - (self.scheduled_ns - previous.scheduled_ns))


# Snapshot header: mirrors MazeChaseEnv._SNAPSHOT_HEADER; fields needed for
# the public (no-underscore) episode metrics: pellets bytes, times_caught,
# cleared.  Offsets: magic(4)+version(2)+grid(2)+max_ticks(4)+tick_period(8)
# +seed(8)+rng(8)+tick(8)+px(4)+py(4)+prev_mask(4)+cleared(4)+pellets_eaten(4)
# +times_caught(4) ... (see maze_chase.py SNAPSHOT_HEADER order).
_SNAPSHOT_HEADER = Struct("<4sHHIQQQQBBBBIIBBBbbBBBBBB")
_PELLET_BYTES = 256 // 8


def _episode_metrics_from_snapshot(snapshot: bytes) -> tuple[int, int, bool]:
    """Pellets eaten / times caught / cleared, read from the public snapshot
    header (no private attribute access).

    The pack order in ``MazeChaseEnv.snapshot()`` is:
    magic, version, grid_size, max_ticks, tick_period_ns, episode_seed,
    rng_state, tick, player_x, player_y, previous_key_mask, cleared,
    pellets_eaten, times_caught, ...
    """
    payload = snapshot[:-32]
    fields = list(_SNAPSHOT_HEADER.unpack_from(payload))
    cleared = bool(fields[11])
    pellets_eaten = int(fields[12])
    times_caught = int(fields[13])
    return pellets_eaten, times_caught, cleared


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Full evidence for one deterministic offline episode."""

    family_index: int
    family_name: str
    seed: int
    partition: str
    perturbation: str
    records: tuple[CycleRecord, ...]
    frame_sha256_stream_sha256: str
    env_state_hash_final: str
    pellets_eaten: int
    times_caught: int
    cleared: bool
    total_reward: float
    elapsed_cycles: int

    def timing_summary(self) -> dict[str, int | float | str]:
        """Nearest-rank quantiles over eligible cycles (jitter starts at
        cycle 1: the first action has no jitter value)."""
        l_plus = sorted(r.l_plus_ns for r in self.records)
        j_plus = sorted(
            self.records[i].j_plus_ns(self.records[i - 1]) for i in range(1, len(self.records))
        )
        misses = sum(1 for r in self.records if r.is_miss)
        return {
            "cycles": len(self.records),
            "median_l_plus_ns": nearest_rank_quantile(l_plus, 0.50),
            "p95_l_plus_ns": nearest_rank_quantile(l_plus, 0.95),
            "p99_l_plus_ns": nearest_rank_quantile(l_plus, 0.99),
            "median_j_plus_ns": (nearest_rank_quantile(j_plus, 0.50) if j_plus else 0),
            "p99_j_plus_ns": (nearest_rank_quantile(j_plus, 0.99) if j_plus else 0),
            "miss_count": misses,
            "miss_rate": misses / len(self.records),
            "max_consecutive_misses": _max_consecutive(self.records, lambda r: r.is_miss),
        }


def _max_consecutive(records: Sequence[CycleRecord], predicate: Callable[[CycleRecord], bool]) -> int:
    best = 0
    run = 0
    for record in records:
        if predicate(record):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


# --- The harness ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Controller:
    """Deterministic control source: action class per cycle.

    ``fn(cycle_index, delivered_frame)`` returns an int in {0..4}
    (idle/W/A/S/D) or a ``GenericControl``.  H1/H2 offline replay uses pure
    logic on the delivered frame; the candidate model (T2-1+) plugs in here
    with its real inference call and live timing receipts.
    """

    name: str
    fn: object  # callable(cycle_index: int, frame: RgbFrame) -> int | GenericControl


def action_class_of(value: object) -> int:
    if isinstance(value, GenericControl):
        from ..v2.trajectory_objective import control_action_class

        return control_action_class(value)
    if isinstance(value, int) and 0 <= value <= 4:
        return value
    raise ValueError(f"controller must yield action class 0..4 or GenericControl, got {value!r}")


def control_for_class(action_class: int) -> GenericControl:
    if action_class == 0:
        return GenericControl(keys_down=())
    return GenericControl(keys_down=(_ACTION_KEYS[action_class - 1],))


class PacmanHarness:
    """No-pause 60 Hz clock/queue harness around MazeChaseEnv.

    Per cycle n (n >= 0):

    1. the clock is at ``s_n``; the world is at tick n (or n-9/... under
       P4/P5 accounting — see world_steps_this_cycle);
    2. P4 may HOLD the world (no step, the action for this cycle is not
       applied); P5 may mark an EXTRA step;
    3. otherwise the controller's action for cycle n is applied to the
       world (once, or twice under P5's extra step);
    4. the frame rendered by that step is the cycle's frame; the P1
       palette shift, if enabled, remaps its colors;
    5. the observation channel delivers the frame to the controller for
       the NEXT cycle through the capacity-one queue: P2 drops g mod 10 == 0
       frames (retaining the prior), P3 delays delivery by one cycle;
    6. the cycle record carries the full clock/queue/latency evidence.

    Cycle 0 applies the controller's first action to the reset frame, so the
    agent acts on the reset frame (consistent with P3's "the agent's first
    decision sees the reset frame" rule: the delivered frame for cycle 0 is
    the reset frame).
    """

    def __init__(
        self,
        family_index: int,
        *,
        perturbation: str = "clean",
        t0_ns: int = 0,
        max_cycles: int | None = None,
    ) -> None:
        if perturbation not in PERTURBATION_IDS:
            raise ValueError(f"perturbation must be one of {PERTURBATION_IDS}")
        fam = family(family_index)
        kwargs = fam.env_kwargs()
        if perturbation == "P6_enemy_policy_change":
            kwargs.update(p6_family_kwargs(family_index))
        if isinstance(t0_ns, bool) or not isinstance(t0_ns, int) or t0_ns < 0:
            raise ValueError("t0_ns must be a non-negative integer")
        if max_cycles is not None and (isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles < 1):
            raise ValueError("max_cycles must be a positive integer or None")
        self.family_index = family_index
        self.family_name = fam.name
        self.perturbation = perturbation
        self.t0_ns = t0_ns
        self.max_cycles = max_cycles
        self._env_factory_kwargs = kwargs
        self._default_max_cycles = int(kwargs["max_ticks"])

    def build_env(self, seed: int) -> MazeChaseEnv:
        env = MazeChaseEnv(**self._env_factory_kwargs)  # type: ignore[arg-type]
        env.reset(seed)
        return env

    def run_episode(
        self,
        seed: int,
        controller: Controller,
        *,
        partition: str = "TEST",
    ) -> EpisodeResult:
        """Run one deterministic offline episode and return full evidence.

        Channel semantics (frozen, prereg section 2):

        - The frame captured at cycle n is the world state at tick n,
          rendered *before* this cycle's action is applied (P1 remaps its
          colors in place; geometry is unchanged).
        - clean: the controller sees the frame captured at the same cycle.
        - P3: the delivered observation is delayed by exactly one cycle
          (the controller sees the frame captured at cycle n-1; cycle 0's
          frame is the reset frame, so the agent's first decision sees the
          reset frame).
        - P2: frame n is not delivered iff n >= 1 and n mod 10 == 0; the
          capacity-one queue retains the prior delivered frame.
        - P4: at cycles n >= 1 with n mod 10 == 0 the world is held (the
          action for that cycle is not applied) and the queue retains the
          prior delivered frame.
        - P5: at cycles n >= 1 with n mod 10 == 0 the action is applied to
          two consecutive world ticks; the queue's latest-complete-frame
          semantics delivers the newest.

        In deterministic offline replay the clock is manual and exact:
        every timestamp equals the scheduled start (inference is declared
        instant), so ``L_plus`` is 0 and no deadline can be missed; the
        live Q1 re-run fills the same schema with real receipts.
        """
        env = self.build_env(seed)
        max_cycles = self.max_cycles if self.max_cycles is not None else self._default_max_cycles
        p1 = self.perturbation == "P1_palette_shift"
        p2 = self.perturbation == "P2_frame_loss_10pct"
        p3 = self.perturbation == "P3_obs_delay_1"
        p4 = self.perturbation == "P4_speed_90pct"
        p5 = self.perturbation == "P5_speed_110pct"

        records: list[CycleRecord] = []
        frame_stream = hashlib.sha256(b"PACMANHARNESS.FRAMESTREAM\x01")
        total_reward = 0.0

        # Capacity-one queue: the last delivered frame (None until first).
        last_delivered: RgbFrame | None = None

        for cycle in range(max_cycles):
            s_n = scheduled_start_ns(cycle, self.t0_ns)
            deadline = scheduled_start_ns(cycle + 1, self.t0_ns)

            # 1. The frame captured at this cycle (world state before step).
            obs_now = env.current_observation
            frame_now = p1_remap_frame(obs_now.rgb, seed) if p1 else obs_now.rgb

            # 2. Channel: what is delivered at this cycle.
            dropped = p2 and p2_frame_dropped(cycle)
            held = p4 and p4_world_held(cycle)
            if (dropped or held) and last_delivered is not None:
                queue_event = "hold"
                delivered = last_delivered
            else:
                queue_event = "dequeue"
                delivered = frame_now
            last_delivered = delivered

            # 3. P3: one-cycle delay; cycle 0 sees the reset frame.
            if p3 and cycle > 0:
                controller_frame = last_delivered_pre_p3 if cycle > 0 else frame_now
            else:
                controller_frame = delivered
            last_delivered_pre_p3 = delivered

            # 4. The controller's action for this cycle.
            action_class = action_class_of(controller.fn(cycle, controller_frame))
            control = control_for_class(action_class)

            world_before = int(obs_now.frame_id)

            reward = 0.0
            events: tuple[str, ...] = ()
            terminated = False
            truncated = False
            world_steps = 0

            if held:
                # P4: world held; the action is not applied; no step.
                outcome_obs = env.current_observation
            else:
                outcome = env.step(control)
                reward += outcome.reward
                events = outcome.events
                terminated = outcome.terminated
                truncated = outcome.truncated
                outcome_obs = outcome.observation
                world_steps = 1
                if p5 and p5_world_extra_step(cycle) and not terminated and not truncated:
                    # P5: apply the same action to a second world tick.
                    outcome = env.step(control)
                    reward += outcome.reward
                    events = events + outcome.events
                    terminated = terminated or outcome.terminated
                    truncated = truncated or outcome.truncated
                    outcome_obs = outcome.observation
                    world_steps = 2
                # (if the first step already terminated/truncated, the extra
                # step is not attempted; the env would refuse it anyway.)

            total_reward += reward
            world_after = int(outcome_obs.frame_id)

            frame_stream.update(frame_now.sha256.encode("ascii"))

            is_hold = queue_event == "hold"
            records.append(
                CycleRecord(
                    cycle_id=cycle,
                    frame_id=cycle,
                    action_id=cycle,
                    scheduled_ns=s_n,
                    deadline_ns=deadline,
                    period_ns=deadline - s_n,
                    t_emit=s_n,
                    t_enqueue=s_n,
                    t_dequeue=s_n,
                    t_infer_start=s_n,
                    t_infer_end=s_n,
                    t_dispatch=s_n,
                    t_accept=s_n,
                    world_tick_before=world_before,
                    world_tick_after=world_after,
                    world_steps_this_cycle=world_steps,
                    queue_event=queue_event,
                    is_miss=False,  # s_n-based acceptance <= deadline in replay
                    is_hold=is_hold,
                    action_class=action_class,
                    frame_sha256=frame_now.sha256,
                    delivered_frame_sha256=controller_frame.sha256,
                    reward=reward,
                    events=tuple(events),
                    terminated=terminated,
                    truncated=truncated,
                )
            )

            if terminated or truncated:
                break

        pellets_eaten, times_caught, cleared = _episode_metrics_from_snapshot(env.snapshot())
        return EpisodeResult(
            family_index=self.family_index,
            family_name=self.family_name,
            seed=seed,
            partition=partition,
            perturbation=self.perturbation,
            records=tuple(records),
            frame_sha256_stream_sha256=frame_stream.hexdigest(),
            env_state_hash_final=env.state_hash(),
            pellets_eaten=pellets_eaten,
            times_caught=times_caught,
            cleared=cleared,
            total_reward=total_reward,
            elapsed_cycles=len(records),
        )


__all__ = [
    "TICK_PERIOD_NS",
    "EMBODIED_BLOCK_START",
    "EMBODIED_BLOCK_END",
    "FAMILY_COUNT",
    "FAMILY_STRIDE",
    "PARTITION_STRIDE",
    "PARTITION_ORDER",
    "SEALED_RANGES",
    "partition_base",
    "partition_seeds",
    "all_harness_seeds",
    "assert_seed_partition_disjointness",
    "DifficultyFamily",
    "FAMILIES",
    "P6_ALTERNATE_RULE",
    "family",
    "PERTURBATION_IDS",
    "PACMAN_RENDER_CONTRACT",
    "p1_palette_offset",
    "p1_remap_frame",
    "p2_frame_dropped",
    "p4_world_held",
    "p5_world_extra_step",
    "p6_family_kwargs",
    "scheduled_start_ns",
    "period_ns",
    "nearest_rank_quantile",
    "CycleRecord",
    "EpisodeResult",
    "Controller",
    "action_class_of",
    "control_for_class",
    "PacmanHarness",
]
