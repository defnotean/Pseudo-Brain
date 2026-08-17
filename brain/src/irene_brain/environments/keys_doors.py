"""A deterministic keys-and-doors maze world: multi-step planning.

This is the fourth moving-shapes successor on the environment ladder
(ROADMAP_TO_PACMAN.md §5, PLAN.md §20): keys/doors. The world is a perfect
maze carved deterministically from the episode seed. The target sits behind
a door on the unique corridor path from the player; the door refuses
movement until the player picks up the key placed on its own side of the
maze. Get key, open door, reach target — an ordered three-step plan with no
chaser pressure, isolating pure sequential planning.

Whether the player holds the key is deliberately not rendered. The frame
shows the key cell until it is picked up and the door cell until it opens;
the model must remember having collected the key. Simulator labels are
returned only in :class:`StepOutcome`, never in :class:`Observation`.

The implementation follows the ladder pattern: no ``random``, an explicit
integer PRNG, and a canonical checksummed snapshot covering the key/door
state.
"""

from __future__ import annotations

from hashlib import sha256
from hmac import compare_digest
from struct import Struct
from typing import cast

from ..types import GenericControl, HidKey, Observation, RgbFrame, StepOutcome
from .junction import _DIRECTIONS, _bfs_distances, _carve_maze
from .moving_shapes import _SplitMix64

_UINT64_MASK = (1 << 64) - 1
_ABSENT = 255


class KeysDoorsEnv:
    """A deterministic 16x16 maze with one key, one door, and one target.

    The blue player moves one corridor cell per call; walls and the closed
    door refuse movement. Walking onto the cyan key collects it
    (``key_collected``); walking into the orange door while holding the key
    opens it permanently (``door_opened``); reaching the yellow target scores
    one reward and relocates it (``target_collected``).
    """

    GRID_SIZE = 16
    SNAPSHOT_VERSION = 1
    DEFAULT_TICK_PERIOD_NS = 16_666_667

    # Public render-contract colors used by pixel-only procedural teachers.
    # Dataset generators may inspect these visible pixels, but must never route
    # simulator coordinates or other privileged state into a model input.
    PLAYER_RGB = (65, 174, 255)
    TARGET_RGB = (250, 206, 55)
    KEY_RGB = (90, 220, 230)
    DOOR_RGB = (170, 110, 40)
    WALL_RGB = (42, 48, 66)

    _SNAPSHOT_MAGIC = b"IBKS"
    _SNAPSHOT_HEADER = Struct("<4sHHIQQQQBBBBBIBBBBBB")
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
    _KEY_COLOR = KEY_RGB
    _DOOR_COLOR = DOOR_RGB
    _PLAYER_COLOR = PLAYER_RGB
    _WALL_COLOR = WALL_RGB

    def __init__(
        self,
        *,
        tick_period_ns: int = DEFAULT_TICK_PERIOD_NS,
        max_ticks: int = 10_000,
    ) -> None:
        if isinstance(tick_period_ns, bool) or not isinstance(tick_period_ns, int):
            raise TypeError("tick_period_ns must be an integer")
        if tick_period_ns <= 0 or tick_period_ns > _UINT64_MASK:
            raise ValueError("tick_period_ns must be in [1, 2**64 - 1]")
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int):
            raise TypeError("max_ticks must be an integer")
        if max_ticks <= 0 or max_ticks > 0xFFFFFFFF:
            raise ValueError("max_ticks must be in [1, 2**32 - 1]")

        self._tick_period_ns = tick_period_ns
        self._max_ticks = max_ticks
        self._episode_seed = 0
        self._rng = _SplitMix64(0)
        self._maze: frozenset[tuple[int, int]] = frozenset()
        self._tick = 0
        self._player_x = 1
        self._player_y = 1
        self._target_x = 0
        self._target_y = 0
        self._key_x = _ABSENT
        self._key_y = _ABSENT
        self._door_x = 0
        self._door_y = 0
        self._has_key = 0
        self._door_open = 0
        self._previous_key_mask = 0
        self._targets_collected = 0
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
        self._maze = _carve_maze(seed, self.GRID_SIZE)
        self._rng = _SplitMix64(seed)
        self._tick = 0
        self._player_x = 1
        self._player_y = 1
        self._has_key = 0
        self._door_open = 0
        self._previous_key_mask = 0
        self._targets_collected = 0

        player = (self._player_x, self._player_y)
        distances = _bfs_distances(player, self._maze)
        farthest_distance = max(distances.values())
        farthest = sorted(
            (cell for cell, d in distances.items() if d == farthest_distance),
            key=lambda cell: (cell[1], cell[0]),
        )
        self._target_x, self._target_y = farthest[0]
        target = (self._target_x, self._target_y)

        # The maze is perfect, so the player→target path is unique: walk
        # downhill on the target's distance field.
        from_target = _bfs_distances(target, self._maze)
        path = [player]
        while path[-1] != target:
            cell = path[-1]
            here = from_target[cell]
            for dx, dy in _DIRECTIONS:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if from_target.get(neighbor) == here - 1:
                    path.append(neighbor)
                    break
        # The door sits at the path midpoint, never on an endpoint.
        door_index = min(max(1, len(path) // 2), len(path) - 2)
        self._door_x, self._door_y = path[door_index]

        # The key goes to the farthest cell reachable with the door closed.
        blocked = self._maze - {(self._door_x, self._door_y)}
        reachable = _bfs_distances(player, blocked)
        if target in reachable:
            raise RuntimeError("door placement did not separate the target")
        key_candidates = sorted(
            (
                cell
                for cell in reachable
                if cell not in (player, target) and cell != (self._door_x, self._door_y)
            ),
            key=lambda cell: (-reachable[cell], cell[1], cell[0]),
        )
        if not key_candidates:
            raise RuntimeError("no reachable key cell remains")
        self._key_x, self._key_y = key_candidates[0]

        return self.current_observation

    def step(self, control: GenericControl) -> StepOutcome:
        """Apply one control state and advance exactly one logical tick."""
        if not isinstance(control, GenericControl):
            raise TypeError("control must be a GenericControl")
        if self._tick >= self._max_ticks:
            raise RuntimeError("the lifetime is truncated; call reset or restore before stepping")

        applied_mask = self._mask_from_control(control)
        applied_control = self._control_from_mask(applied_mask)

        horizontal = int(bool(applied_mask & self._KEY_BITS[int(HidKey.D)])) - int(
            bool(applied_mask & self._KEY_BITS[int(HidKey.A)])
        )
        vertical = int(bool(applied_mask & self._KEY_BITS[int(HidKey.S)])) - int(
            bool(applied_mask & self._KEY_BITS[int(HidKey.W)])
        )
        candidate = (
            min(self.GRID_SIZE - 1, max(0, self._player_x + horizontal)),
            min(self.GRID_SIZE - 1, max(0, self._player_y + vertical)),
        )

        events: list[str] = []
        reward = 0.0
        if candidate in self._maze:
            if candidate == (self._door_x, self._door_y) and not self._door_open:
                if self._has_key:
                    self._door_open = 1
                    events.append("door_opened")
                    self._player_x, self._player_y = candidate
                # Without the key the door refuses movement, like a wall.
            else:
                self._player_x, self._player_y = candidate

        player = (self._player_x, self._player_y)
        if not self._has_key and player == (self._key_x, self._key_y):
            self._has_key = 1
            events.append("key_collected")
            self._key_x = _ABSENT
            self._key_y = _ABSENT

        if player == (self._target_x, self._target_y):
            self._targets_collected += 1
            reward += 1.0
            events.append("target_collected")
            self._target_x, self._target_y = self._sample_empty_cell({player})

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
                self._key_x,
                self._key_y,
                self._door_x,
                self._door_y,
                self._has_key,
                self._door_open,
            )
        )
        payload.extend(sha256(payload).digest())
        return bytes(payload)

    def restore(self, snapshot: bytes) -> Observation:
        """Atomically restore a validated snapshot produced by this version."""
        if not isinstance(snapshot, bytes):
            raise TypeError("snapshot must be bytes")
        expected_length = self._SNAPSHOT_HEADER.size + self._SNAPSHOT_DIGEST_BYTES
        if len(snapshot) != expected_length:
            raise ValueError("snapshot length mismatch")

        payload = snapshot[: -self._SNAPSHOT_DIGEST_BYTES]
        supplied_digest = snapshot[-self._SNAPSHOT_DIGEST_BYTES :]
        if not compare_digest(sha256(payload).digest(), supplied_digest):
            raise ValueError("snapshot checksum mismatch")

        header = cast(
            tuple[
                bytes, int, int, int, int, int, int, int,
                int, int, int, int, int, int, int, int, int, int, int, int,
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
            key_x,
            key_y,
            door_x,
            door_y,
            has_key,
            door_open,
        ) = header
        if magic != self._SNAPSHOT_MAGIC:
            raise ValueError("snapshot magic mismatch")
        if version != self.SNAPSHOT_VERSION:
            raise ValueError("snapshot version mismatch")
        if grid_size != self.GRID_SIZE:
            raise ValueError("snapshot grid size mismatch")
        if max_ticks != self._max_ticks:
            raise ValueError("snapshot max_ticks mismatch")
        if tick_period_ns != self._tick_period_ns:
            raise ValueError("snapshot tick period mismatch")
        if tick > max_ticks:
            raise ValueError("snapshot tick exceeds max_ticks")
        if previous_key_mask & ~self._SUPPORTED_KEY_MASK:
            raise ValueError("snapshot contains unsupported key bits")
        if has_key not in (0, 1) or door_open not in (0, 1):
            raise ValueError("snapshot flags must be 0 or 1")

        maze = _carve_maze(episode_seed, self.GRID_SIZE)
        if (player_x, player_y) not in maze or (target_x, target_y) not in maze:
            raise ValueError("snapshot contains an off-corridor coordinate")
        if (door_x, door_y) not in maze:
            raise ValueError("snapshot door is off-corridor")
        if has_key and (key_x, key_y) != (_ABSENT, _ABSENT):
            raise ValueError("snapshot holds the key but keeps it placed")
        if not has_key:
            if (key_x, key_y) not in maze:
                raise ValueError("snapshot key is off-corridor")
        if door_open and not has_key:
            raise ValueError("snapshot has an open door without the key")

        # Assign only after every byte has been validated, keeping failed restores atomic.
        self._episode_seed = episode_seed
        self._maze = maze
        self._rng = _SplitMix64(rng_state)
        self._tick = tick
        self._player_x = player_x
        self._player_y = player_y
        self._target_x = target_x
        self._target_y = target_y
        self._key_x = key_x
        self._key_y = key_y
        self._door_x = door_x
        self._door_y = door_y
        self._has_key = has_key
        self._door_open = door_open
        self._previous_key_mask = previous_key_mask
        self._targets_collected = targets_collected
        return self.current_observation

    def state_hash(self) -> str:
        """Return a SHA-256 digest over the complete canonical state bytes."""
        return sha256(self.snapshot()).hexdigest()

    def _sample_empty_cell(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        reserved = set(occupied)
        reserved.add((self._door_x, self._door_y))
        if not self._has_key:
            reserved.add((self._key_x, self._key_y))
        candidates = sorted(
            (cell for cell in self._maze if cell not in reserved),
            key=lambda cell: (cell[1], cell[0]),
        )
        if not candidates:
            raise RuntimeError("no empty corridor cell remains")
        return candidates[self._rng.randbelow(len(candidates))]

    def _render(self) -> RgbFrame:
        pixels = bytearray(self.GRID_SIZE * self.GRID_SIZE * 3)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) in self._maze:
                    color = (
                        self._BACKGROUND_EVEN if (x + y) % 2 == 0 else self._BACKGROUND_ODD
                    )
                else:
                    color = self._WALL_COLOR
                self._paint(pixels, x, y, color)

        if not self._door_open:
            self._paint(pixels, self._door_x, self._door_y, self._DOOR_COLOR)
        if not self._has_key:
            self._paint(pixels, self._key_x, self._key_y, self._KEY_COLOR)
        self._paint(pixels, self._target_x, self._target_y, self._TARGET_COLOR)
        self._paint(pixels, self._player_x, self._player_y, self._PLAYER_COLOR)
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


__all__ = ["KeysDoorsEnv"]
