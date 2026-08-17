"""A deterministic pursuit world: collect targets while chasers close in.

This is the first moving-shapes successor on the environment ladder
(ROADMAP_TO_PACMAN.md §5, PLAN.md §20): pursuit/evasion. The player collects
relocating targets for +1 while one or more deterministic greedy pursuers
chase it; contact costs -1 and respawns the player. Pursuers are deliberately
procedural — this world trains and measures escape-under-pressure behavior,
not opponent modeling.

The implementation follows ``moving_shapes.py`` exactly: no ``random``, no
third-party packages, an explicit integer PRNG, swept-path same-time contact
tests, and a canonical checksummed snapshot so equal snapshots always produce
equal futures.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from struct import Struct
from typing import cast

from ..types import GenericControl, HidKey, Observation, RgbFrame, StepOutcome
from .moving_shapes import _SplitMix64, _paths_collide_at_same_time

_UINT64_MASK = (1 << 64) - 1


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


class PursuitEnv:
    """A deterministic 16x16 pursuit world controlled by W/A/S/D HID keys.

    The blue player moves one cell per call. Red pursuers each move one cell
    greedily toward the player every ``pursuer_period`` ticks (larger axis
    first, horizontal on ties). Touching or crossing a pursuer emits a
    ``caught`` event, costs one reward, and respawns the player on an empty
    cell; touching the yellow target relocates it and scores one reward.
    Simulator labels are returned only in :class:`StepOutcome`, never in
    :class:`Observation`.
    """

    GRID_SIZE = 16
    SNAPSHOT_VERSION = 1
    DEFAULT_TICK_PERIOD_NS = 16_666_667
    MAX_PURSUER_PERIOD = 64

    # Public render-contract colors used by pixel-only procedural teachers.
    # Dataset generators may inspect these visible pixels, but must never route
    # simulator coordinates or other privileged state into a model input.
    PLAYER_RGB = (65, 174, 255)
    TARGET_RGB = (250, 206, 55)
    PLAYER_PURSUER_OVERLAP_RGB = (255, 255, 255)

    _SNAPSHOT_MAGIC = b"IBPS"
    _SNAPSHOT_HEADER = Struct("<4sHHIQQQQBBBBBIIBB")
    _SNAPSHOT_PURSUER = Struct("<BB")
    _SNAPSHOT_DIGEST_BYTES = 32

    _KEY_BITS = {
        int(HidKey.W): 1 << 0,
        int(HidKey.A): 1 << 1,
        int(HidKey.S): 1 << 2,
        int(HidKey.D): 1 << 3,
    }
    _SUPPORTED_KEY_MASK = (1 << 4) - 1

    _BACKGROUND_EVEN = (8, 11, 18)
    _BACKGROUND_ODD = (10, 14, 22)
    _TARGET_COLOR = TARGET_RGB
    _PURSUER_COLOR = (225, 55, 65)
    _PLAYER_COLOR = PLAYER_RGB
    _OVERLAP_COLOR = PLAYER_PURSUER_OVERLAP_RGB

    def __init__(
        self,
        *,
        pursuer_count: int = 2,
        pursuer_period: int = 2,
        tick_period_ns: int = DEFAULT_TICK_PERIOD_NS,
        max_ticks: int = 10_000,
    ) -> None:
        if isinstance(pursuer_count, bool) or not isinstance(pursuer_count, int):
            raise TypeError("pursuer_count must be an integer")
        if pursuer_count < 1 or pursuer_count > self.GRID_SIZE * self.GRID_SIZE - 2:
            raise ValueError("pursuer_count does not fit in the logical grid")
        if isinstance(pursuer_period, bool) or not isinstance(pursuer_period, int):
            raise TypeError("pursuer_period must be an integer")
        if pursuer_period < 1 or pursuer_period > self.MAX_PURSUER_PERIOD:
            raise ValueError(
                f"pursuer_period must be in [1, {self.MAX_PURSUER_PERIOD}]"
            )
        if isinstance(tick_period_ns, bool) or not isinstance(tick_period_ns, int):
            raise TypeError("tick_period_ns must be an integer")
        if tick_period_ns <= 0 or tick_period_ns > _UINT64_MASK:
            raise ValueError("tick_period_ns must be in [1, 2**64 - 1]")
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int):
            raise TypeError("max_ticks must be an integer")
        if max_ticks <= 0 or max_ticks > 0xFFFFFFFF:
            raise ValueError("max_ticks must be in [1, 2**32 - 1]")

        self._pursuer_count = pursuer_count
        self._pursuer_period = pursuer_period
        self._tick_period_ns = tick_period_ns
        self._max_ticks = max_ticks
        self._episode_seed = 0
        self._rng = _SplitMix64(0)
        self._tick = 0
        self._player_x = 0
        self._player_y = 0
        self._target_x = 0
        self._target_y = 0
        self._previous_key_mask = 0
        self._targets_collected = 0
        self._times_caught = 0
        self._pursuers: list[_Pursuer] = []
        self.reset(0)

    @property
    def tick_period_ns(self) -> int:
        return self._tick_period_ns

    @property
    def current_observation(self) -> Observation:
        """Build the observation from public pixels and generic timing/control."""
        return Observation(
            frame_id=self._tick,
            capture_tick=self._tick,
            elapsed_ns=self._tick * self._tick_period_ns,
            rgb=self._render(),
            previous_control=self._control_from_mask(self._previous_key_mask),
            audio_pcm_s16le=None,
            text_inputs=(),
        )

    def reset(self, seed: int) -> Observation:
        """Reset to the unique initial state selected by a 64-bit integer seed."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0 or seed > _UINT64_MASK:
            raise ValueError("seed must be in [0, 2**64 - 1]")

        self._episode_seed = seed
        self._rng = _SplitMix64(seed)
        self._tick = 0
        self._player_x = self.GRID_SIZE // 2
        self._player_y = self.GRID_SIZE // 2
        self._previous_key_mask = 0
        self._targets_collected = 0
        self._times_caught = 0
        self._pursuers = []

        occupied = {(self._player_x, self._player_y)}
        self._target_x, self._target_y = self._sample_empty_cell(occupied)
        occupied.add((self._target_x, self._target_y))

        for _ in range(self._pursuer_count):
            # Rejection sampling keeps chasers from spawning on top of the
            # player while staying fully deterministic for a fixed seed.
            while True:
                x, y = self._sample_empty_cell(occupied)
                distance = abs(x - self._player_x) + abs(y - self._player_y)
                if distance >= 4:
                    break
            occupied.add((x, y))
            self._pursuers.append(_Pursuer(x=x, y=y))

        return self.current_observation

    def step(self, control: GenericControl) -> StepOutcome:
        """Apply one control state and advance exactly one logical tick."""
        if not isinstance(control, GenericControl):
            raise TypeError("control must be a GenericControl")
        if self._tick >= self._max_ticks:
            raise RuntimeError("the lifetime is truncated; call reset or restore before stepping")

        applied_mask = self._mask_from_control(control)
        applied_control = self._control_from_mask(applied_mask)
        old_player = (self._player_x, self._player_y)

        horizontal = int(bool(applied_mask & self._KEY_BITS[int(HidKey.D)])) - int(
            bool(applied_mask & self._KEY_BITS[int(HidKey.A)])
        )
        vertical = int(bool(applied_mask & self._KEY_BITS[int(HidKey.S)])) - int(
            bool(applied_mask & self._KEY_BITS[int(HidKey.W)])
        )
        self._player_x = min(self.GRID_SIZE - 1, max(0, self._player_x + horizontal))
        self._player_y = min(self.GRID_SIZE - 1, max(0, self._player_y + vertical))
        new_player = (self._player_x, self._player_y)

        pursuer_paths: list[tuple[tuple[int, int], tuple[int, int]]] = []
        pursuers_move = self._tick % self._pursuer_period == 0
        for pursuer in self._pursuers:
            before = (pursuer.x, pursuer.y)
            if pursuers_move:
                dx = _sign(self._player_x - pursuer.x)
                dy = _sign(self._player_y - pursuer.y)
                # Larger axis first; horizontal wins ties.
                if dx != 0 and abs(self._player_x - pursuer.x) >= abs(
                    self._player_y - pursuer.y
                ):
                    pursuer.x += dx
                elif dy != 0:
                    pursuer.y += dy
            pursuer_paths.append((before, (pursuer.x, pursuer.y)))

        events: list[str] = []
        reward = 0.0
        caught = any(
            _paths_collide_at_same_time(old_player, new_player, before, after)
            for before, after in pursuer_paths
        )
        if caught:
            self._times_caught += 1
            reward -= 1.0
            events.append("caught")
            occupied = {
                (self._target_x, self._target_y),
                *((pursuer.x, pursuer.y) for pursuer in self._pursuers),
            }
            self._player_x, self._player_y = self._sample_empty_cell(occupied)
        elif new_player == (self._target_x, self._target_y):
            self._targets_collected += 1
            reward += 1.0
            events.append("target_collected")
            occupied = {
                new_player,
                *((pursuer.x, pursuer.y) for pursuer in self._pursuers),
            }
            self._target_x, self._target_y = self._sample_empty_cell(occupied)

        self._tick += 1
        self._previous_key_mask = applied_mask
        return StepOutcome(
            observation=self.current_observation,
            requested_control=control,
            applied_control=applied_control,
            reward=reward,
            events=tuple(events),
            terminated=False,
            truncated=self._tick >= self._max_ticks,
        )

    def snapshot(self) -> bytes:
        """Return a canonical, checksummed, version-1 snapshot."""
        payload = bytearray(
            self._SNAPSHOT_HEADER.pack(
                self._SNAPSHOT_MAGIC,
                self.SNAPSHOT_VERSION,
                self.GRID_SIZE,
                self._max_ticks,
                self._tick_period_ns,
                self._episode_seed,
                self._rng.state,
                self._tick,
                self._player_x,
                self._player_y,
                self._target_x,
                self._target_y,
                self._previous_key_mask,
                self._targets_collected,
                self._times_caught,
                self._pursuer_count,
                self._pursuer_period,
            )
        )
        for pursuer in self._pursuers:
            payload.extend(self._SNAPSHOT_PURSUER.pack(pursuer.x, pursuer.y))
        payload.extend(sha256(payload).digest())
        return bytes(payload)

    def restore(self, snapshot: bytes) -> Observation:
        """Atomically restore a validated snapshot produced by this version."""
        if not isinstance(snapshot, bytes):
            raise TypeError("snapshot must be bytes")
        minimum_length = self._SNAPSHOT_HEADER.size + self._SNAPSHOT_DIGEST_BYTES
        if len(snapshot) < minimum_length:
            raise ValueError("snapshot is too short")

        payload = snapshot[: -self._SNAPSHOT_DIGEST_BYTES]
        supplied_digest = snapshot[-self._SNAPSHOT_DIGEST_BYTES :]
        if not compare_digest(sha256(payload).digest(), supplied_digest):
            raise ValueError("snapshot checksum mismatch")

        header = cast(
            tuple[
                bytes, int, int, int, int, int, int, int,
                int, int, int, int, int, int, int, int, int,
            ],
            self._SNAPSHOT_HEADER.unpack_from(payload),
        )
        (
            magic,
            version,
            grid_size,
            max_ticks,
            tick_period_ns,
            episode_seed,
            rng_state,
            tick,
            player_x,
            player_y,
            target_x,
            target_y,
            previous_key_mask,
            targets_collected,
            times_caught,
            pursuer_count,
            pursuer_period,
        ) = header
        if magic != self._SNAPSHOT_MAGIC:
            raise ValueError("snapshot magic mismatch")
        if version != self.SNAPSHOT_VERSION:
            raise ValueError("snapshot version mismatch")
        if grid_size != self.GRID_SIZE:
            raise ValueError("snapshot grid size mismatch")
        if pursuer_count != self._pursuer_count:
            raise ValueError("snapshot pursuer count mismatch")
        if pursuer_period != self._pursuer_period:
            raise ValueError("snapshot pursuer period mismatch")
        if max_ticks != self._max_ticks:
            raise ValueError("snapshot max_ticks mismatch")
        if tick_period_ns != self._tick_period_ns:
            raise ValueError("snapshot tick period mismatch")
        if len(payload) != self._SNAPSHOT_HEADER.size + pursuer_count * self._SNAPSHOT_PURSUER.size:
            raise ValueError("snapshot length mismatch")
        if tick > max_ticks:
            raise ValueError("snapshot tick exceeds max_ticks")
        if previous_key_mask & ~self._SUPPORTED_KEY_MASK:
            raise ValueError("snapshot contains unsupported key bits")

        coordinates = ((player_x, player_y), (target_x, target_y))
        if any(
            x >= self.GRID_SIZE or y >= self.GRID_SIZE
            for x, y in coordinates
        ):
            raise ValueError("snapshot contains an out-of-bounds coordinate")

        pursuers: list[_Pursuer] = []
        offset = self._SNAPSHOT_HEADER.size
        for _ in range(pursuer_count):
            x, y = cast(
                tuple[int, int],
                self._SNAPSHOT_PURSUER.unpack_from(payload, offset),
            )
            offset += self._SNAPSHOT_PURSUER.size
            if x >= self.GRID_SIZE or y >= self.GRID_SIZE:
                raise ValueError("snapshot contains an out-of-bounds pursuer")
            pursuers.append(_Pursuer(x=x, y=y))

        # Assign only after every byte has been validated, keeping failed restores atomic.
        self._episode_seed = episode_seed
        self._rng = _SplitMix64(rng_state)
        self._tick = tick
        self._player_x = player_x
        self._player_y = player_y
        self._target_x = target_x
        self._target_y = target_y
        self._previous_key_mask = previous_key_mask
        self._targets_collected = targets_collected
        self._times_caught = times_caught
        self._pursuers = pursuers
        return self.current_observation

    def state_hash(self) -> str:
        """Return a SHA-256 digest over the complete canonical state bytes."""
        return sha256(self.snapshot()).hexdigest()

    def _sample_empty_cell(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        available = self.GRID_SIZE * self.GRID_SIZE - len(occupied)
        if available <= 0:
            raise RuntimeError("no empty logical cell remains")
        selected = self._rng.randbelow(available)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) in occupied:
                    continue
                if selected == 0:
                    return x, y
                selected -= 1
        raise AssertionError("empty-cell selection fell outside the grid")

    def _render(self) -> RgbFrame:
        pixels = bytearray(self.GRID_SIZE * self.GRID_SIZE * 3)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self._BACKGROUND_EVEN if (x + y) % 2 == 0 else self._BACKGROUND_ODD
                self._paint(pixels, x, y, color)

        self._paint(pixels, self._target_x, self._target_y, self._TARGET_COLOR)
        for pursuer in self._pursuers:
            self._paint(pixels, pursuer.x, pursuer.y, self._PURSUER_COLOR)

        player_color = self._PLAYER_COLOR
        if any(
            pursuer.x == self._player_x and pursuer.y == self._player_y
            for pursuer in self._pursuers
        ):
            player_color = self._OVERLAP_COLOR
        self._paint(pixels, self._player_x, self._player_y, player_color)
        return RgbFrame(width=self.GRID_SIZE, height=self.GRID_SIZE, pixels=bytes(pixels))

    def _paint(self, pixels: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
        offset = (y * self.GRID_SIZE + x) * 3
        pixels[offset : offset + 3] = bytes(color)

    @classmethod
    def _mask_from_control(cls, control: GenericControl) -> int:
        mask = 0
        for key in control.keys_down:
            mask |= cls._KEY_BITS.get(key, 0)
        return mask

    @classmethod
    def _control_from_mask(cls, mask: int) -> GenericControl:
        keys = tuple(key for key, bit in cls._KEY_BITS.items() if mask & bit)
        return GenericControl(keys_down=keys)


@dataclass(slots=True)
class _Pursuer:
    x: int
    y: int


__all__ = ["PursuitEnv"]
