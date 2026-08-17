"""A deterministic occlusion world: moving shapes under fog of war.

This is the third moving-shapes successor on the environment ladder
(ROADMAP_TO_PACMAN.md §5, PLAN.md §20): occlusion. The mechanics are exactly
the moving-shapes world — bouncing hazards, relocating targets, one-cell
W/A/S/D movement — but the rendered frame only reveals the Chebyshev
``view_radius`` neighborhood around the player. Everything beyond it is fog,
including hazards that were visible a tick ago. Remembering what can no
longer be seen is the skill this world isolates, and it is the direct
environment-level test of the persistent thought-state thesis: a reactive
policy is structurally blind here.

The implementation follows ``moving_shapes.py``: no ``random``, no
third-party packages, an explicit integer PRNG, swept-path same-time contact
tests, and a canonical checksummed snapshot whose identity includes the view
radius.
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


@dataclass(slots=True)
class _Hazard:
    x: int
    y: int
    vx: int
    vy: int


class OcclusionEnv:
    """Moving-shapes mechanics with a fogged 16x16 frame and W/A/S/D control.

    The blue player moves one cell per call. Red hazards move diagonally and
    bounce from the world boundaries; the yellow target relocates when
    collected. Only the Chebyshev ``view_radius`` neighborhood of the player
    is rendered; every other cell — including hazards and the target — is
    fog. The mechanics are unaffected by visibility: contact is judged on the
    full simulator state with the shared swept-path same-time test, so an
    unseen hazard still collides. Simulator labels are returned only in
    :class:`StepOutcome`, never in :class:`Observation`.
    """

    GRID_SIZE = 16
    SNAPSHOT_VERSION = 1
    DEFAULT_TICK_PERIOD_NS = 16_666_667
    MAX_VIEW_RADIUS = 8

    # Public render-contract colors used by pixel-only procedural teachers.
    # Dataset generators may inspect these visible pixels, but must never route
    # simulator coordinates or other privileged state into a model input.
    PLAYER_RGB = (65, 174, 255)
    TARGET_RGB = (250, 206, 55)
    FOG_RGB = (2, 3, 5)
    PLAYER_HAZARD_OVERLAP_RGB = (255, 255, 255)

    _SNAPSHOT_MAGIC = b"IBOS"
    _SNAPSHOT_HEADER = Struct("<4sHHIQQQQBBBBBIIBBB")
    _SNAPSHOT_HAZARD = Struct("<BBbb")
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
    _HAZARD_COLOR = (225, 55, 65)
    _PLAYER_COLOR = PLAYER_RGB
    _FOG_COLOR = FOG_RGB
    _OVERLAP_COLOR = PLAYER_HAZARD_OVERLAP_RGB

    def __init__(
        self,
        *,
        hazard_count: int = 3,
        view_radius: int = 4,
        tick_period_ns: int = DEFAULT_TICK_PERIOD_NS,
        max_ticks: int = 10_000,
    ) -> None:
        if isinstance(hazard_count, bool) or not isinstance(hazard_count, int):
            raise TypeError("hazard_count must be an integer")
        if hazard_count < 1 or hazard_count > self.GRID_SIZE * self.GRID_SIZE - 2:
            raise ValueError("hazard_count does not fit in the logical grid")
        if isinstance(view_radius, bool) or not isinstance(view_radius, int):
            raise TypeError("view_radius must be an integer")
        if view_radius < 1 or view_radius > self.MAX_VIEW_RADIUS:
            raise ValueError(
                f"view_radius must be in [1, {self.MAX_VIEW_RADIUS}]"
            )
        if isinstance(tick_period_ns, bool) or not isinstance(tick_period_ns, int):
            raise TypeError("tick_period_ns must be an integer")
        if tick_period_ns <= 0 or tick_period_ns > _UINT64_MASK:
            raise ValueError("tick_period_ns must be in [1, 2**64 - 1]")
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int):
            raise TypeError("max_ticks must be an integer")
        if max_ticks <= 0 or max_ticks > 0xFFFFFFFF:
            raise ValueError("max_ticks must be in [1, 2**32 - 1]")

        self._hazard_count = hazard_count
        self._view_radius = view_radius
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
        self._collisions = 0
        self._hazards: list[_Hazard] = []
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
        self._collisions = 0
        self._hazards = []

        occupied = {(self._player_x, self._player_y)}
        self._target_x, self._target_y = self._sample_empty_cell(occupied)
        occupied.add((self._target_x, self._target_y))

        for _ in range(self._hazard_count):
            x, y = self._sample_empty_cell(occupied)
            occupied.add((x, y))
            vx = -1 if self._rng.next_u64() & 1 else 1
            vy = -1 if self._rng.next_u64() & 1 else 1
            self._hazards.append(_Hazard(x=x, y=y, vx=vx, vy=vy))

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

        hazard_paths: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for hazard in self._hazards:
            before = (hazard.x, hazard.y)
            next_x = hazard.x + hazard.vx
            next_y = hazard.y + hazard.vy
            if next_x < 0 or next_x >= self.GRID_SIZE:
                hazard.vx = -hazard.vx
                next_x = hazard.x + hazard.vx
            if next_y < 0 or next_y >= self.GRID_SIZE:
                hazard.vy = -hazard.vy
                next_y = hazard.y + hazard.vy
            hazard.x = next_x
            hazard.y = next_y
            hazard_paths.append((before, (hazard.x, hazard.y)))

        events: list[str] = []
        reward = 0.0
        collided = any(
            _paths_collide_at_same_time(old_player, new_player, before, after)
            for before, after in hazard_paths
        )
        if collided:
            self._collisions += 1
            reward -= 1.0
            events.append("collision")

        if new_player == (self._target_x, self._target_y):
            self._targets_collected += 1
            reward += 1.0
            events.append("target_collected")
            occupied = {new_player, *((hazard.x, hazard.y) for hazard in self._hazards)}
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
                self._collisions,
                self._hazard_count,
                self._view_radius,
                0,  # reserved
            )
        )
        for hazard in self._hazards:
            payload.extend(
                self._SNAPSHOT_HAZARD.pack(hazard.x, hazard.y, hazard.vx, hazard.vy)
            )
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
                int, int, int, int, int, int, int, int, int, int,
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
            collisions,
            hazard_count,
            view_radius,
            reserved,
        ) = header
        if magic != self._SNAPSHOT_MAGIC:
            raise ValueError("snapshot magic mismatch")
        if version != self.SNAPSHOT_VERSION:
            raise ValueError("snapshot version mismatch")
        if grid_size != self.GRID_SIZE:
            raise ValueError("snapshot grid size mismatch")
        if hazard_count != self._hazard_count:
            raise ValueError("snapshot hazard count mismatch")
        if view_radius != self._view_radius:
            raise ValueError("snapshot view radius mismatch")
        if reserved != 0:
            raise ValueError("snapshot reserved byte must be zero")
        if max_ticks != self._max_ticks:
            raise ValueError("snapshot max_ticks mismatch")
        if tick_period_ns != self._tick_period_ns:
            raise ValueError("snapshot tick period mismatch")
        if len(payload) != self._SNAPSHOT_HEADER.size + hazard_count * self._SNAPSHOT_HAZARD.size:
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

        hazards: list[_Hazard] = []
        offset = self._SNAPSHOT_HEADER.size
        for _ in range(hazard_count):
            x, y, vx, vy = cast(
                tuple[int, int, int, int],
                self._SNAPSHOT_HAZARD.unpack_from(payload, offset),
            )
            offset += self._SNAPSHOT_HAZARD.size
            if x >= self.GRID_SIZE or y >= self.GRID_SIZE:
                raise ValueError("snapshot contains an out-of-bounds hazard")
            if vx not in (-1, 1) or vy not in (-1, 1):
                raise ValueError("snapshot hazard velocities must be -1 or 1")
            hazards.append(_Hazard(x=x, y=y, vx=vx, vy=vy))

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
        self._collisions = collisions
        self._hazards = hazards
        return self.current_observation

    def state_hash(self) -> str:
        """Return a SHA-256 digest over the complete canonical state bytes."""
        return sha256(self.snapshot()).hexdigest()

    def _visible(self, x: int, y: int) -> bool:
        return (
            abs(x - self._player_x) <= self._view_radius
            and abs(y - self._player_y) <= self._view_radius
        )

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
                if not self._visible(x, y):
                    color = self._FOG_COLOR
                else:
                    color = (
                        self._BACKGROUND_EVEN if (x + y) % 2 == 0 else self._BACKGROUND_ODD
                    )
                self._paint(pixels, x, y, color)

        if self._visible(self._target_x, self._target_y):
            self._paint(pixels, self._target_x, self._target_y, self._TARGET_COLOR)
        for hazard in self._hazards:
            if self._visible(hazard.x, hazard.y):
                self._paint(pixels, hazard.x, hazard.y, self._HAZARD_COLOR)

        player_color = self._PLAYER_COLOR
        if any(
            hazard.x == self._player_x and hazard.y == self._player_y
            for hazard in self._hazards
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


__all__ = ["OcclusionEnv"]
