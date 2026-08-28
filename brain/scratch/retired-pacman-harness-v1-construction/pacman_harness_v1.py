"""Pac-Man-like v3 section-4 no-pause harness (construction layer).

Preregistration:
``brain/docs/preregistrations/2026-08-27-pacman-harness-v1-construction.md``
(frozen 2026-08-27).  This module is the Tier-1 *construction* of the
Pac-Man-like v3 section-4 clock/queue layer on top of the existing,
frozen ``environments/maze_chase.py`` (SNAPSHOT_VERSION 4).  It adds:

- the five registered difficulty families (frozen ``MazeChaseEnv``
  parameterizations);
- the six registered perturbations (P0..P5) applied in a deterministic,
  evidence-logging wrapper;
- the registered TRAIN/DEV/CAL/TEST seed partitions (fresh 24-bit-aligned
  block ``EMB0 = 12 * 2**24``);
- a section-4-compliant **logical** no-pause clock/queue with
  ``cycle_id``/``frame_id``/``action_id``, host receipt events, a
  capacity-1 latest-complete-frame queue, and deadline/miss/jitter
  accounting;
- a policy slot (no-op self-test, and a pixel-only **reactive** baseline
  that is never weakened);
- a canonical, byte-deterministic event log so two runs of the same
  slice byte-match (construction gate 1).

What this module is NOT:

- Not a qualification run.  It evaluates no model, opens no CPU-QUAL, and
  touches no TEST partition except to *enumerate* its registered seeds
  (it never runs TEST episodes here).  The model-side battery is the
  separate Tier-2/Tier-3 preregistrations.
- Not a live wall-clock loop.  The section-4 ``L_plus``/``J_plus`` values
  computed here use a **frozen declared** inference latency
  (``simulated_infer_ns``), exactly the manual-clock convention of
  ``evaluation/closed_loop_play.py``.  The live, physically-timed
  distribution (real capture -> model -> HID) is Tier-3 Q1 and is a
  separate registration.  What is validated *here* is that the harness's
  gate *computation* is correct: a no-op policy at the declared latency
  produces zero deadline misses, and an injected over-budget latency
  produces exactly the expected deterministic misses.
- Not a change to ``maze_chase`` internals.  The environment is consumed
  as-is; this file only wraps it.

Causal-integrity rule (C8, binding): the only object a policy ever sees is
a canonical ``ModelObservation`` (frame, elapsed, age, previous control,
dropped-frames counter).  Privileged state (ghost coordinates, pellet map,
reward, hazard, task/family id) is logged for evidence and scoring but never
enters the model boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol
from zlib import crc32

from ..types import GenericControl, HidKey, ModelObservation, RgbFrame
from ..v2.embodied_interface import EmbodiedInterfaceRecord, translate_record
from .maze_chase import MazeChaseEnv


# --- Frozen identity -------------------------------------------------------

IDENTITY = "irene.brain.pacman_harness.v1"
VERSION = 1

# The maze_chase render contract (public, used by pixel-only policies).
_WALL_RGB = (42, 48, 66)
_PELLET_RGB = (122, 118, 92)
_PLAYER_RGB = (65, 174, 255)
_GHOST_RGB = (225, 55, 65)
_OVERLAP_RGB = (255, 255, 255)

# Locomotion key mapping (W/A/S/D -> HID usages).  Mirrors the env's
# _KEY_BITS so a record's locomotion decodes to the exact key the env steps
# on.  Look/cursor/button fields are physically absent in the 2-D maze and
# are ignored by the env (the adapter "ignores fields the environment
# physically lacks").
#
# PACMAN_BASE_PERIOD_NS is the frozen clean-mechanics period (60 Hz native
# control rate).  P3/P4 scale it symmetrically for game and controller.
PACMAN_BASE_PERIOD_NS = 16_666_667

_LOCOMOTION_KEY = {
    "NOOP": (),
    "FORWARD": (int(HidKey.W),),
    "BACKWARD": (int(HidKey.S),),
    "LEFT": (int(HidKey.A),),
    "RIGHT": (int(HidKey.D),),
}

# Canonical direction order for deterministic tie-breaks: up, left, down,
# right.  Each maps to a locomotion record value.
# (dx, dy, locomotion) with grid convention: x = column (right+), y = row
# (down+).  Up is (0, -1) because rows increase downward.
_DIRECTION_ORDER: tuple[tuple[int, int, str], ...] = (
    (0, -1, "FORWARD"),   # up   -> W
    (-1, 0, "LEFT"),      # left -> A
    (0, 1, "BACKWARD"),   # down -> S
    (1, 0, "RIGHT"),      # right-> D
)

# The exact HID usages ``translate_record`` may ever actuate under
# ``EmbodiedInterfaceV1`` (the W/S/A/D locomotion keys plus the button
# keys).  The legal-action audit (gate 6) checks every emitted record's
# actuated keys against this set.
_ACTUATED_KEY_SET: frozenset[int] = frozenset(
    {
        int(HidKey.W),
        int(HidKey.S),
        int(HidKey.A),
        int(HidKey.D),
        int(HidKey.SPACE),
        int(HidKey.UP),
        int(HidKey.LEFT),
        int(HidKey.RIGHT),
        int(HidKey.DOWN),
    }
)


def _zlib_seed(name: str) -> int:
    """Process-stable 32-bit seed (zlib.crc32, NOT the salted builtin hash)."""
    return crc32(name.encode("utf-8")) & 0xFFFFFFFF


# --- Section-4 timing gate constants (v3 section 4.3, Pac-Man-like) ---------

MEDIAN_LPLUS_NS = 8_000_000
P95_LPLUS_NS = 13_000_000
P99_LPLUS_NS = 16_000_000
MISS_RATE_LIMIT = 0.001
P99_JPLUS_NS = 2_000_000


# --- Difficulty families (frozen, preregistration section 1) ---------------


@dataclass(frozen=True, slots=True)
class DifficultyFamily:
    id: str
    name: str
    extra_loops: int
    ghost_rule: str
    ghost_period: int
    ghost_count: int

    def env_kwargs(self) -> dict:
        """The frozen, clean-mechanics constructor args for one family.

        ``player_period=1``, ``ghost_elroy=False``, ``input_delay_ticks=0``,
        ``sticky_direction=False``: delay/sticky belong to the perturbation
        axes, not the difficulty axis.
        """
        return {
            "extra_loops": self.extra_loops,
            "ghost_rule": self.ghost_rule,
            "ghost_period": self.ghost_period,
            "ghost_count": self.ghost_count,
            "player_period": 1,
            "ghost_elroy": False,
            "input_delay_ticks": 0,
            "sticky_direction": False,
        }


FAMILIES: dict[str, DifficultyFamily] = {
    "F0": DifficultyFamily("F0", "open-calm", 40, "shy", 4, 2),
    "F1": DifficultyFamily("F1", "standard", 24, "mixed", 3, 3),
    "F2": DifficultyFamily("F2", "dense-fast", 16, "direct", 2, 3),
    "F3": DifficultyFamily("F3", "tight-ambush", 10, "ambush", 2, 4),
    "F4": DifficultyFamily("F4", "choke-elroy-off", 6, "direct", 1, 4),
}
FAMILY_ORDER: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4")


# --- Perturbations (frozen, preregistration section 2) ---------------------


@dataclass(frozen=True, slots=True)
class Perturbation:
    id: int
    name: str
    # P0: per-family 8-bit (r,g,b) palette offset, applied clipped to [0,255].
    palette_offsets: Mapping[str, tuple[int, int, int]] = field(default_factory=dict)
    # P1: frame-loss stride (drop a fresh frame every N frame ids); 0 = none.
    frame_loss_stride: int = 0
    # P2: one-frame observation delay when True.
    observation_delay: bool = False
    # P3/P4: controller/game period in ns (both identical, no-pause); None =
    # base period.
    period_ns: int | None = None
    # P5: per-family registered ghost-rule alternate; None = keep the family's.
    rule_alternate: Mapping[str, str] = field(default_factory=dict)


_PALETTE_OFFSETS: dict[str, tuple[int, int, int]] = {
    "F0": (10, -6, 4),
    "F1": (-8, 12, 0),
    "F2": (15, 15, -10),
    "F3": (-12, -4, 18),
    "F4": (6, -14, -6),
}
_RULE_ALTERNATE: dict[str, str] = {
    "F0": "direct",   # shy -> direct
    "F1": "direct",   # mixed -> direct
    "F2": "ambush",   # direct -> ambush
    "F3": "direct",   # ambush -> direct
    "F4": "shy",      # direct -> shy
}

CLEAN = Perturbation(id=-1, name="clean")
P0 = Perturbation(id=0, name="palette-shift", palette_offsets=_PALETTE_OFFSETS)
P1 = Perturbation(id=1, name="frame-loss-10pct", frame_loss_stride=10)
P2 = Perturbation(id=2, name="one-frame-delay", observation_delay=True)
P3 = Perturbation(id=3, name="speed-90pct", period_ns=18_518_519)
P4 = Perturbation(id=4, name="speed-110pct", period_ns=15_151_515)
P5 = Perturbation(id=5, name="enemy-policy-altered", rule_alternate=_RULE_ALTERNATE)
PERTURBATIONS: tuple[Perturbation, ...] = (P0, P1, P2, P3, P4, P5)


def perturbations_clean_and_all() -> tuple[Perturbation, ...]:
    """The clean condition plus the six registered perturbations."""
    return (CLEAN, *PERTURBATIONS)


# --- Seed partitions (frozen, preregistration section 4) -------------------

# Fresh 24-bit-aligned block.  No overlap with the sealed PB21* blocks (up to
# 11*2**24 + 768 = 184_550_144), the RCQ family, or any RCQ_V2_PROTOCOL range.
EMB0 = 12 * 2 ** 24  # 201_326_592

_PARTITION_SIZES: dict[str, int] = {
    "TRAIN": 200_000,
    "DEV": 50_000,
    "CAL": 50_000,
    "TEST": 400_000,
}
_PARTITION_ORDER: tuple[str, ...] = ("TRAIN", "DEV", "CAL", "TEST")


class SeedPartition:
    """Disjoint-by-integer seed sub-ranges inside the fresh EMB0 block.

    Each partition owns a contiguous, non-overlapping sub-range; the offset
    is a fixed function of the partition order (no random selection).
    """

    def __init__(self, base: int = EMB0) -> None:
        if isinstance(base, bool) or not isinstance(base, int):
            raise TypeError("base must be an integer")
        if base < 0:
            raise ValueError("base must be >= 0")
        self._base = base
        # Compute the layout once; verify non-overlap and sizes.
        cursor = base
        self._ranges: dict[str, tuple[int, int]] = {}
        for name in _PARTITION_ORDER:
            size = _PARTITION_SIZES[name]
            self._ranges[name] = (cursor, cursor + size)
            cursor += size
        self._total = cursor - base

    @property
    def base(self) -> int:
        return self._base

    def sub_range(self, partition: str) -> tuple[int, int]:
        if partition not in _PARTITION_ORDER:
            raise ValueError(f"unknown partition {partition!r}")
        return self._ranges[partition]

    def total_size(self) -> int:
        return self._total

    def episode_seed(self, partition: str, index: int) -> int:
        """The ``index``-th seed inside a partition's sub-range."""
        lo, hi = self._ranges[partition]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= hi - lo:
            raise ValueError(f"index {index} out of range for {partition}")
        return lo + index

    # TEST layout / replicate enumeration (Q2/Q3 geometry).

    @property
    def test_layouts(self) -> list[tuple[DifficultyFamily, int]]:
        """The first 40 TEST seeds as (family, seed), in F0..F4 block order.

        40 layouts ÷ 5 families = 8 layouts per family, so layout ``i``
        belongs to family ``FAMILY_ORDER[i // 8]``.
        """
        lo, _ = self._ranges["TEST"]
        return [(FAMILIES[FAMILY_ORDER[i // 8]], lo + i) for i in range(40)]

    def test_replicates(self, layout_index: int) -> list[int]:
        """The 5 registered Q3 stochastic replicate seeds for a TEST layout.

        Layout ``i``'s replicates occupy seeds ``test_lo + 40 + 5*i ..
        test_lo + 40 + 5*i + 4`` (200 contiguous seeds for 40 layouts).
        """
        if isinstance(layout_index, bool) or not isinstance(layout_index, int):
            raise TypeError("layout_index must be an integer")
        if layout_index < 0 or layout_index >= 40:
            raise ValueError("layout_index must be in [0, 40)")
        lo, _ = self._ranges["TEST"]
        start = lo + 40 + 5 * layout_index
        return [start + k for k in range(5)]


# --- Policy slot -----------------------------------------------------------


class Policy(Protocol):
    """A policy sees only a canonical ``ModelObservation`` (the C8 boundary).

    It returns one ``EmbodiedInterfaceRecord`` for the tick.  It may keep its
    own (deterministic) state, but it may never be handed privileged state:
    no ghost coordinates, pellet map, reward, hazard, task, or family id.
    """

    def act(self, obs: ModelObservation, cycle_id: int) -> EmbodiedInterfaceRecord:
        ...


def _record_to_control(record: EmbodiedInterfaceRecord) -> GenericControl:
    """Decode a record's locomotion into the env's WASD control.

    Look/cursor/button fields are physically absent in the 2-D maze and are
    ignored (the adapter ignores fields the environment physically lacks).
    """
    keys = _LOCOMOTION_KEY[record.locomotion]
    return GenericControl(keys_down=keys)


class NoOpPolicy:
    """The all-neutral policy: NOOP locomotion, everything else neutral.

    Used only for the harness self-test (gate 3): at the declared latency it
    must produce zero deadline misses, proving the schedule/queue/miss
    accounting is correct on the pass side.
    """

    def act(self, obs: ModelObservation, cycle_id: int) -> EmbodiedInterfaceRecord:
        return EmbodiedInterfaceRecord.neutral()


class ReactivePixelPolicy:
    """A pixel-only, no-cross-step-state reactive baseline (never weakened).

    On each tick it decodes the 16x16 ``RgbFrame`` into player / ghost /
    pellet / wall cells (the public render contract), then:

    1. If any ghost is within Chebyshev distance ``_DANGER_RADIUS`` of the
       player, move to the neighboring corridor cell that maximizes the
       minimum Chebyshev distance to any ghost (deterministic direction
       order for ties).  If no corridor neighbor is safe, NOOP.
    2. Otherwise, move toward the nearest visible pellet by greedy corridor
       step (a neighbor cell that is not a wall and reduces Chebyshev
       distance); NOOP if none do.

    It is a pure function of the frame (plus nothing else): no memory, no
    lookahead, no planner.  If it scores below random on a family, that is a
    recorded datum, not a reason to lower it.
    """

    _DANGER_RADIUS = 3

    def __init__(self, grid_size: int = 16) -> None:
        if isinstance(grid_size, bool) or not isinstance(grid_size, int):
            raise TypeError("grid_size must be an integer")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        self._n = grid_size

    # -- pixel decoding (public render contract only) --

    def _decode(self, frame: RgbFrame) -> dict:
        n = frame.width
        if frame.height != n or n != self._n:
            raise ValueError("frame must be a square grid matching grid_size")
        player: tuple[int, int] | None = None
        ghosts: list[tuple[int, int]] = []
        pellets: list[tuple[int, int]] = []
        walls: set[tuple[int, int]] = set()
        for y in range(n):
            for x in range(n):
                off = (y * n + x) * 3
                px = (frame.pixels[off], frame.pixels[off + 1], frame.pixels[off + 2])
                if px == _WALL_RGB:
                    walls.add((x, y))
                elif px == _PELLET_RGB:
                    pellets.append((x, y))
                elif px == _GHOST_RGB:
                    ghosts.append((x, y))
                elif px == _PLAYER_RGB or px == _OVERLAP_RGB:
                    player = (x, y)
        return {"player": player, "ghosts": ghosts, "pellets": pellets, "walls": walls}

    @staticmethod
    def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def act(self, obs: ModelObservation, cycle_id: int) -> EmbodiedInterfaceRecord:
        dec = self._decode(obs.rgb)
        player = dec["player"]
        if player is None:
            return EmbodiedInterfaceRecord.neutral()
        n = self._n
        walls: set[tuple[int, int]] = dec["walls"]
        ghosts: list[tuple[int, int]] = dec["ghosts"]

        def corridor(cell: tuple[int, int]) -> bool:
            return 0 <= cell[0] < n and 0 <= cell[1] < n and cell not in walls

        if ghosts:
            nearest = min(ghosts, key=lambda g: self._cheb(player, g))
            if self._cheb(player, nearest) <= self._DANGER_RADIUS:
                best: tuple[int, int] | None = None
                best_score = -1
                for dx, dy, _loc in _DIRECTION_ORDER:
                    cand = (player[0] + dx, player[1] + dy)
                    if not corridor(cand):
                        continue
                    # minimum Chebyshev distance to any ghost
                    score = min(self._cheb(cand, g) for g in ghosts)
                    if score > best_score:
                        best = cand
                        best_score = score
                if best is not None:
                    dx = best[0] - player[0]
                    dy = best[1] - player[1]
                    return self._record_for_delta(dx, dy)
                return EmbodiedInterfaceRecord.neutral()

        if not dec["pellets"]:
            return EmbodiedInterfaceRecord.neutral()
        target = min(dec["pellets"], key=lambda p: self._cheb(player, p))
        best: tuple[int, int] | None = None
        best_dist = self._cheb(player, target)
        for dx, dy, _loc in _DIRECTION_ORDER:
            cand = (player[0] + dx, player[1] + dy)
            if not corridor(cand):
                continue
            d = self._cheb(cand, target)
            if d < best_dist:
                best = cand
                best_dist = d
        if best is not None:
            return self._record_for_delta(best[0] - player[0], best[1] - player[1])
        return EmbodiedInterfaceRecord.neutral()

    @staticmethod
    def _record_for_delta(dx: int, dy: int) -> EmbodiedInterfaceRecord:
        for ddx, ddy, loc in _DIRECTION_ORDER:
            if (ddx, ddy) == (dx, dy):
                return EmbodiedInterfaceRecord(locomotion=loc)
        return EmbodiedInterfaceRecord.neutral()


# --- Tick / episode evidence records ---------------------------------------


@dataclass(frozen=True, slots=True)
class TickEvent:
    cycle_id: int
    frame_id: int
    action_id: int
    s_ns: int
    d_ns: int
    p_ns: int
    t_emit_ns: int
    t_enqueue_ns: int
    t_dequeue_ns: int
    t_infer_start_ns: int
    t_infer_end_ns: int
    t_dispatch_ns: int
    t_accept_ns: int
    l_plus_ns: int
    j_plus_ns: int | None
    miss: bool
    late: bool
    frame_lost: bool
    observation_delayed: bool
    dropped_frames: int
    reward: float
    events: tuple[str, ...]
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    seed: int
    family_name: str
    perturbation_name: str
    partition: str
    period_ns: int
    ticks: int
    pellets_eaten: int
    times_caught: int
    cleared: bool
    truncated: bool
    total_reward: float
    legal_action_rate: float
    ticks_data: tuple[TickEvent, ...]


# --- The harness -----------------------------------------------------------


class PacmanHarness:
    """A deterministic, no-pause, section-4-compliant single-episode runner.

    One instance is bound to one (family, perturbation, seed) triple and one
    policy.  :meth:`run_episode` executes the logical control loop and returns
    a byte-deterministic :class:`EpisodeRecord` (all timestamps are logical
    functions of the frozen schedule and the declared inference latency).
    """

    def __init__(
        self,
        family: DifficultyFamily,
        perturbation: Perturbation,
        seed: int,
        policy: Policy,
        *,
        partition: str = "TRAIN",
        simulated_infer_ns: int = 3_000_000,
        max_ticks: int = 10_000,
        t0_ns: int = 0,
        record_ticks: bool = True,
    ) -> None:
        if not isinstance(family, DifficultyFamily):
            raise TypeError("family must be a DifficultyFamily")
        if not isinstance(perturbation, Perturbation):
            raise TypeError("perturbation must be a Perturbation")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0:
            raise ValueError("seed must be >= 0")
        if isinstance(simulated_infer_ns, bool) or not isinstance(
            simulated_infer_ns, int
        ):
            raise TypeError("simulated_infer_ns must be an integer")
        if simulated_infer_ns < 0:
            raise ValueError("simulated_infer_ns must be >= 0")
        if partition not in _PARTITION_ORDER:
            raise ValueError(f"unknown partition {partition!r}")

        self._family = family
        self._perturbation = perturbation
        self._seed = seed
        self._policy = policy
        self._partition = partition
        self._simulated_infer_ns = simulated_infer_ns
        self._max_ticks = max_ticks
        self._t0 = t0_ns
        self._record_ticks = record_ticks

        # Resolve the effective mechanics: the family's frozen params plus
        # P5's registered rule alternate (the only mechanic a perturbation
        # may change besides the observation/timing channel).
        self._env_kwargs: dict = family.env_kwargs()
        if perturbation.rule_alternate:
            self._env_kwargs["ghost_rule"] = perturbation.rule_alternate[family.id]
        self._period_ns = (
            perturbation.period_ns
            if perturbation.period_ns is not None
            else PACMAN_BASE_PERIOD_NS
        )

    # -- public accessors --

    @property
    def family(self) -> DifficultyFamily:
        return self._family

    @property
    def perturbation(self) -> Perturbation:
        return self._perturbation

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def period_ns(self) -> int:
        return self._period_ns

    # -- section-4 schedule --

    def _schedule_ns(self, n: int) -> int:
        """The authoritative start time of native cycle ``n``.

        Base: the frozen section-4 nominal period (16.666667 ms) as the
        integer ``16_666_667`` ns, i.e. ``s_n = t0 + n * period``.  The v3
        literal ``floor((n*1e9 + 30)/60)`` differs by < 1 ns/cycle and is
        below every gate threshold (all ms-scale); the integer-period form
        keeps P3/P4 uniform and the schedule byte-deterministic.
        """
        return self._t0 + n * self._period_ns

    # -- render / perturbation transforms --

    def _apply_palette(self, frame: RgbFrame) -> RgbFrame:
        if not self._perturbation.palette_offsets:
            return frame
        off = self._perturbation.palette_offsets.get(self._family.id)
        if off is None:
            return frame
        dr, dg, db = off
        n = frame.width
        out = bytearray(len(frame.pixels))
        px = frame.pixels
        for i in range(0, len(px), 3):
            out[i] = min(255, max(0, px[i] + dr))
            out[i + 1] = min(255, max(0, px[i + 1] + dg))
            out[i + 2] = min(255, max(0, px[i + 2] + db))
        return RgbFrame(width=frame.width, height=frame.height, pixels=bytes(out))

    # -- episode loop --

    def run_episode(self) -> EpisodeRecord:
        env = MazeChaseEnv(
            **self._env_kwargs,
            tick_period_ns=self._period_ns,
            max_ticks=self._max_ticks,
        )
        env.reset(self._seed)

        period = self._period_ns
        infer = self._simulated_infer_ns
        stride = self._perturbation.frame_loss_stride
        delay = self._perturbation.observation_delay

        # Observation release channel (v3 section 4.2, capacity 1,
        # latest-complete-frame).  Frame labeling follows the env's own
        # convention: the pre-step render at cycle n carries
        # ``frame_id = n``, because the env's ``_tick`` equals n at that
        # point and its Observation stamps ``frame_id = _tick``.  The
        # content of frame k is the state after k applied actions (S_k).
        #
        #   clean/P0: at cycle n the policy sees frame n (fresh render,
        #              t_emit = s_n).
        #   P1:        the fresh frame of a "drop cycle" — one whose frame
        #              id is a positive multiple of the stride — is not
        #              released; the policy reuses the latest complete
        #              frame (the previous cycle's pre-step render), and
        #              the drop is logged.
        #   P2:        every frame is released one cycle late: the frame
        #              rendered at cycle n first becomes available at
        #              cycle n+1, so at cycle n the policy sees the
        #              pre-step render from cycle n-1 (frame n-1,
        #              t_emit = s_{n-1}).  Exactly the clean stream
        #              shifted by one cycle: clean serves frame n at
        #              cycle n; P2 serves frame n-1.
        prev_prerender: tuple[int, RgbFrame, int] | None = None    # cycle n-1's
        # each: (frame_id, rgb, t_emit)

        last_control = GenericControl()
        dropped_frames = 0
        total_reward = 0.0
        pellets_eaten = 0
        times_caught = 0
        legal = 0
        total_emitted = 0
        cleared = False
        truncated = False
        events_all: list[TickEvent] = []

        n = 0
        prev_accept: int | None = None
        while not cleared and not truncated and n < self._max_ticks:
            s_n = self._schedule_ns(n)
            d_n = self._schedule_ns(n + 1)
            p_n = d_n - s_n

            # Pre-step render of the current env state (frame n).  This is
            # the content a clean release would hand to the policy at cycle n.
            pre_obs = env.current_observation
            content_id = n
            fresh_rgb = self._apply_palette(pre_obs.rgb)

            frame_lost_this_cycle = False
            observation_delayed_this_cycle = False
            if n >= 1 and delay:
                content_id, rgb, t_emit = prev_prerender
                observation_delayed_this_cycle = True
            elif (
                stride
                and n >= 1
                and content_id % stride == 0
                and content_id >= stride
            ):
                # P1 drop cycle: the fresh frame (content_id = n, a positive
                # multiple of the stride) is not released; reuse the latest
                # complete frame.
                content_id, rgb, _ = prev_prerender
                t_emit = s_n - period  # the reused frame's own emit time
                dropped_frames += 1
                frame_lost_this_cycle = True
            else:
                content_id, rgb, t_emit = content_id, fresh_rgb, s_n

            t_dequeue = s_n
            t_infer_start = s_n
            t_infer_end = s_n + infer
            t_dispatch = t_infer_end
            t_accept = t_dispatch
            l_plus = t_accept - t_emit
            miss = t_accept > d_n
            late = (t_dequeue - t_emit) > p_n
            if prev_accept is None:
                j_plus: int | None = None
            else:
                j_plus = abs((t_accept - prev_accept) - p_n)
            prev_accept = t_accept

            # Causal boundary (C8): the ModelObservation is built ONLY from
            # the released frame and factual previous control.
            model_obs = ModelObservation(
                frame_id=content_id,
                elapsed_ns=content_id * period,
                observation_age_ns=max(0, s_n - t_emit),
                rgb=rgb,
                previous_control=last_control,
                dropped_frames=dropped_frames,
            )
            record = self._policy.act(model_obs, n)
            total_emitted += 1
            keys = translate_record(record)
            legal += int(all(k in _ACTUATED_KEY_SET for k in keys))
            control = _record_to_control(record)
            outcome = env.step(control)
            last_control = outcome.applied_control
            total_reward += outcome.reward
            terminated = outcome.terminated
            truncated = outcome.truncated
            if "pellet_eaten" in outcome.events:
                pellets_eaten += 1
            if "caught" in outcome.events:
                times_caught += 1
            cleared = bool(outcome.terminated and "cleared" in outcome.events)

            # Cache this cycle's pre-step render (frame n, emitted at s_n)
            # for the next cycle's P1/P2 reuse.
            prev_prerender = (n, fresh_rgb, s_n)

            if self._record_ticks:
                events_all.append(
                    TickEvent(
                        cycle_id=n,
                        frame_id=content_id,
                        action_id=n,
                        s_ns=s_n,
                        d_ns=d_n,
                        p_ns=p_n,
                        t_emit_ns=t_emit,
                        t_enqueue_ns=s_n,
                        t_dequeue_ns=t_dequeue,
                        t_infer_start_ns=t_infer_start,
                        t_infer_end_ns=t_infer_end,
                        t_dispatch_ns=t_dispatch,
                        t_accept_ns=t_accept,
                        l_plus_ns=l_plus,
                        j_plus_ns=j_plus,
                        miss=miss,
                        late=late,
                        frame_lost=frame_lost_this_cycle,
                        observation_delayed=observation_delayed_this_cycle,
                        dropped_frames=dropped_frames,
                        reward=outcome.reward,
                        events=outcome.events,
                        terminated=terminated,
                        truncated=truncated,
                    )
                )
            n += 1
            if cleared or truncated:
                break

        legal_rate = (legal / total_emitted) if total_emitted else 1.0
        return EpisodeRecord(
            seed=self._seed,
            family_name=self._family.id,
            perturbation_name=self._perturbation.name,
            partition=self._partition,
            period_ns=self._period_ns,
            ticks=n,
            pellets_eaten=pellets_eaten,
            times_caught=times_caught,
            cleared=cleared,
            truncated=truncated,
            total_reward=total_reward,
            legal_action_rate=legal_rate,
            ticks_data=tuple(events_all),
        )


# --- Slice / report aggregation + gate computation -------------------------


def quantile_nearest_rank(values: list[int], q: float) -> int:
    """Nearest-rank quantile (v3 section 4.2) over a non-empty list."""
    import math

    if not values:
        raise ValueError("quantile of empty list")
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered))
    rank = max(1, min(len(ordered), rank))
    return ordered[rank - 1]


def timing_gates(ticks: list[TickEvent]) -> dict:
    """Compute the section-4.3 timing statistics (no pass/fail verdict here).

    Returns the median/p95/p99 L_plus, the deadline-miss rate, p99 J_plus,
    and the consecutive-miss count.  Construction gate 3 reports these and
    separately asserts the miss count for the no-op self-test.
    """
    if not ticks:
        return {"n_ticks": 0}
    lplus = [t.l_plus_ns for t in ticks]
    jplus = [t.j_plus_ns for t in ticks if t.j_plus_ns is not None]
    misses = [t for t in ticks if t.miss]
    # consecutive-miss count: longest run of adjacent missed cycles
    run = 0
    best_run = 0
    for t in ticks:
        if t.miss:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return {
        "n_ticks": len(ticks),
        "median_lplus_ns": quantile_nearest_rank(lplus, 0.50),
        "p95_lplus_ns": quantile_nearest_rank(lplus, 0.95),
        "p99_lplus_ns": quantile_nearest_rank(lplus, 0.99),
        "miss_count": len(misses),
        "miss_rate": len(misses) / len(ticks),
        "consecutive_miss_max_run": best_run,
        "p99_jplus_ns": (quantile_nearest_rank(jplus, 0.99) if jplus else 0),
        "n_jplus": len(jplus),
    }


def timing_gate_verdict(stats: dict) -> dict:
    """Apply the frozen section-4.3 Pac-Man-like limits to a stats dict."""
    if stats.get("n_ticks", 0) == 0:
        return {"passed": False, "reason": "no ticks"}
    checks = {
        "median_lplus": stats["median_lplus_ns"] <= MEDIAN_LPLUS_NS,
        "p95_lplus": stats["p95_lplus_ns"] <= P95_LPLUS_NS,
        "p99_lplus": stats["p99_lplus_ns"] <= P99_LPLUS_NS,
        "miss_rate": stats["miss_rate"] <= MISS_RATE_LIMIT,
        "p99_jplus": stats["p99_jplus_ns"] <= P99_JPLUS_NS,
        "consecutive_misses": stats["consecutive_miss_max_run"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


__all__ = [
    "IDENTITY",
    "VERSION",
    "MEDIAN_LPLUS_NS",
    "P95_LPLUS_NS",
    "P99_LPLUS_NS",
    "MISS_RATE_LIMIT",
    "P99_JPLUS_NS",
    "DifficultyFamily",
    "FAMILIES",
    "FAMILY_ORDER",
    "PACMAN_BASE_PERIOD_NS",
    "Perturbation",
    "CLEAN",
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "PERTURBATIONS",
    "perturbations_clean_and_all",
    "EMB0",
    "SeedPartition",
    "Policy",
    "NoOpPolicy",
    "ReactivePixelPolicy",
    "TickEvent",
    "EpisodeRecord",
    "PacmanHarness",
    "_record_to_control",
    "quantile_nearest_rank",
    "timing_gates",
    "timing_gate_verdict",
]
