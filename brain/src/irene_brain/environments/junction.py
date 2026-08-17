"""A deterministic maze world: junction choice under pressure.

This is the second moving-shapes successor on the environment ladder
(ROADMAP_TO_PACMAN.md §5, PLAN.md §20): junction choice. The world is a
perfect maze carved deterministically from the episode seed. The player
collects relocating targets for +1 while one or more chasers follow exact
BFS shortest paths through the corridors; contact costs -1 and respawns the
player. Deciding correctly at branch points — not reaction speed in open
space — is the skill this world isolates, and it is the Pac-Man skill.

The implementation follows ``moving_shapes.py`` and ``pursuit.py``: no
``random``, no third-party packages, an explicit integer PRNG, swept-path
same-time contact tests, and a canonical checksummed snapshot. The maze is a
pure function of the episode seed (pinned by the snapshot version), so
snapshots stay self-contained: restore regenerates the maze from the stored
seed and validates everything else.
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

# Fixed direction order for every deterministic tie-break: W, A, S, D.
_DIRECTIONS = ((0, -1), (-1, 0), (0, 1), (1, 0))


def _carve_maze(seed: int, grid_size: int) -> frozenset[tuple[int, int]]:
    """Carve a perfect maze with a seeded iterative recursive backtracker.

    Rooms live at odd coordinates; every other cell is wall unless carved.
    The result is a connected, cycle-free corridor set — a pure function of
    ``seed`` and ``grid_size``.
    """

    rooms_per_side = grid_size // 2
    rng = _SplitMix64(seed ^ 0x4D415A45)  # "MAZE" domain separator
    carved: set[tuple[int, int]] = set()
    start = (1, 1)
    carved.add(start)
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        cx, cy = stack[-1]
        neighbors = [
            (cx + dx, cy + dy)
            for dx, dy in _DIRECTIONS
            if 0 <= cx + dx < rooms_per_side
            and 0 <= cy + dy < rooms_per_side
            and (cx + dx, cy + dy) not in visited
        ]
        if not neighbors:
            stack.pop()
            continue
        nx, ny = neighbors[rng.randbelow(len(neighbors))]
        visited.add((nx, ny))
        # Knock the wall between the two rooms.
        carved.add((2 * cx + 1 + (nx - cx), 2 * cy + 1 + (ny - cy)))
        carved.add((2 * nx + 1, 2 * ny + 1))
        stack.append((nx, ny))
    return frozenset(carved)


def _bfs_distances(
    origin: tuple[int, int], carved: frozenset[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    """Exact corridor distances from ``origin`` in fixed expansion order."""

    distances = {origin: 0}
    queue = [origin]
    for cell in queue:
        for dx, dy in _DIRECTIONS:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if neighbor in carved and neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances


class JunctionEnv:
    """A deterministic 16x16 maze world controlled by W/A/S/D HID keys.

    The blue player moves one corridor cell per call; walls refuse movement.
    Red chasers each move one cell along the exact BFS shortest path to the
    player every ``chaser_period`` ticks. Touching or crossing a chaser emits
    a ``caught`` event, costs one reward, and respawns the player on an empty
    corridor cell; touching the yellow target relocates it and scores one
    reward. Simulator labels are returned only in :class:`StepOutcome`, never
    in :class:`Observation`.
    """

    GRID_SIZE = 16
    SNAPSHOT_VERSION = 1
    DEFAULT_TICK_PERIOD_NS = 16_666_667
    MAX_CHASER_PERIOD = 64

    # Public render-contract colors used by pixel-only procedural teachers.
    # Dataset generators may inspect these visible pixels, but must never route
    # simulator coordinates or other privileged state into a model input.
    PLAYER_RGB = (65, 174, 255)
    TARGET_RGB = (250, 206, 55)
    WALL_RGB = (42, 48, 66)
    PLAYER_CHASER_OVERLAP_RGB = (255, 255, 255)

    _SNAPSHOT_MAGIC = b"IBJS"
    _SNAPSHOT_HEADER = Struct("<4sHHIQQQQBBBBBIIBB")
    _SNAPSHOT_CHASER = Struct("<BB")
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
    _CHASER_COLOR = (225, 55, 65)
    _PLAYER_COLOR = PLAYER_RGB
    _WALL_COLOR = WALL_RGB
    _OVERLAP_COLOR = PLAYER_CHASER_OVERLAP_RGB

    def __init__(
        self,
        *,
        chaser_count: int = 1,
        chaser_period: int = 2,
        tick_period_ns: int = DEFAULT_TICK_PERIOD_NS,
        max_ticks: int = 10_000,
    ) -> None:
        if isinstance(chaser_count, bool) or not isinstance(chaser_count, int):
            raise TypeError("chaser_count must be an integer")
        if chaser_count < 1 or chaser_count > 8:
            raise ValueError("chaser_count must be in [1, 8]")
        if isinstance(chaser_period, bool) or not isinstance(chaser_period, int):
            raise TypeError("chaser_period must be an integer")
        if chaser_period < 1 or chaser_period > self.MAX_CHASER_PERIOD:
            raise ValueError(
                f"chaser_period must be in [1, {self.MAX_CHASER_PERIOD}]"
            )
        if isinstance(tick_period_ns, bool) or not isinstance(tick_period_ns, int):
            raise TypeError("tick_period_ns must be an integer")
        if tick_period_ns <= 0 or tick_period_ns > _UINT64_MASK:
            raise ValueError("tick_period_ns must be in [1, 2**64 - 1]")
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int):
            raise TypeError("max_ticks must be an integer")
        if max_ticks <= 0 or max_ticks > 0xFFFFFFFF:
            raise ValueError("max_ticks must be in [1, 2**32 - 1]")

        self._chaser_count = chaser_count
        self._chaser_period = chaser_period
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
        self._previous_key_mask = 0
        self._targets_collected = 0
        self._times_caught = 0
        self._chasers: list[_Chaser] = []
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
        self._previous_key_mask = 0
        self._targets_collected = 0
        self._times_caught = 0
        self._chasers = []

        player = (self._player_x, self._player_y)
        distances = _bfs_distances(player, self._maze)
        farthest_distance = max(distances.values())
        farthest = [
            cell for cell, distance in distances.items() if distance == farthest_distance
        ]
        # Scan order (y, then x) makes the farthest-cell choice deterministic.
        self._target_x, self._target_y = sorted(
            farthest, key=lambda cell: (cell[1], cell[0])
        )[0]

        occupied = {player, (self._target_x, self._target_y)}
        minimum = min(10, farthest_distance)
        distant = sorted(
            (
                cell
                for cell, distance in distances.items()
                if distance >= minimum and cell not in occupied
            ),
            key=lambda cell: (cell[1], cell[0]),
        )
        for _ in range(self._chaser_count):
            if distant:
                x, y = distant[self._rng.randbelow(len(distant))]
            else:
                x, y = self._sample_empty_cell(occupied)
            occupied.add((x, y))
            distant = [cell for cell in distant if cell != (x, y)]
            self._chasers.append(_Chaser(x=x, y=y))

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
        candidate = (
            min(self.GRID_SIZE - 1, max(0, self._player_x + horizontal)),
            min(self.GRID_SIZE - 1, max(0, self._player_y + vertical)),
        )
        # Walls refuse the whole move; there is no sliding along them.
        if candidate in self._maze:
            self._player_x, self._player_y = candidate
        new_player = (self._player_x, self._player_y)

        chaser_paths: list[tuple[tuple[int, int], tuple[int, int]]] = []
        chasers_move = self._tick % self._chaser_period == 0
        if chasers_move:
            distances = _bfs_distances(new_player, self._maze)
        for chaser in self._chasers:
            before = (chaser.x, chaser.y)
            if chasers_move:
                here = distances.get(before)
                if here is not None and here > 0:
                    for dx, dy in _DIRECTIONS:
                        neighbor = (chaser.x + dx, chaser.y + dy)
                        if distances.get(neighbor) == here - 1:
                            chaser.x, chaser.y = neighbor
                            break
            chaser_paths.append((before, (chaser.x, chaser.y)))

        events: list[str] = []
        reward = 0.0
        caught = any(
            _paths_collide_at_same_time(old_player, new_player, before, after)
            for before, after in chaser_paths
        )
        if caught:
            self._times_caught += 1
            reward -= 1.0
            events.append("caught")
            occupied = {
                (self._target_x, self._target_y),
                *((chaser.x, chaser.y) for chaser in self._chasers),
            }
            self._player_x, self._player_y = self._sample_empty_cell(occupied)
        elif new_player == (self._target_x, self._target_y):
            self._targets_collected += 1
            reward += 1.0
            events.append("target_collected")
            occupied = {
                new_player,
                *((chaser.x, chaser.y) for chaser in self._chasers),
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
                self._chaser_count,
                self._chaser_period,
            )
        )
        for chaser in self._chasers:
            payload.extend(self._SNAPSHOT_CHASER.pack(chaser.x, chaser.y))
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
            chaser_count,
            chaser_period,
        ) = header
        if magic != self._SNAPSHOT_MAGIC:
            raise ValueError("snapshot magic mismatch")
        if version != self.SNAPSHOT_VERSION:
            raise ValueError("snapshot version mismatch")
        if grid_size != self.GRID_SIZE:
            raise ValueError("snapshot grid size mismatch")
        if chaser_count != self._chaser_count:
            raise ValueError("snapshot chaser count mismatch")
        if chaser_period != self._chaser_period:
            raise ValueError("snapshot chaser period mismatch")
        if max_ticks != self._max_ticks:
            raise ValueError("snapshot max_ticks mismatch")
        if tick_period_ns != self._tick_period_ns:
            raise ValueError("snapshot tick period mismatch")
        if len(payload) != self._SNAPSHOT_HEADER.size + chaser_count * self._SNAPSHOT_CHASER.size:
            raise ValueError("snapshot length mismatch")
        if tick > max_ticks:
            raise ValueError("snapshot tick exceeds max_ticks")
        if previous_key_mask & ~self._SUPPORTED_KEY_MASK:
            raise ValueError("snapshot contains unsupported key bits")

        # The maze is a pure function of the episode seed under this snapshot
        # version, so restoring it needs no bytes of its own.
        maze = _carve_maze(episode_seed, self.GRID_SIZE)
        coordinates = ((player_x, player_y), (target_x, target_y))
        if any(cell not in maze for cell in coordinates):
            raise ValueError("snapshot contains an off-corridor coordinate")

        chasers: list[_Chaser] = []
        offset = self._SNAPSHOT_HEADER.size
        for _ in range(chaser_count):
            x, y = cast(
                tuple[int, int],
                self._SNAPSHOT_CHASER.unpack_from(payload, offset),
            )
            offset += self._SNAPSHOT_CHASER.size
            if (x, y) not in maze:
                raise ValueError("snapshot contains an off-corridor chaser")
            chasers.append(_Chaser(x=x, y=y))

        # Assign only after every byte has been validated, keeping failed restores atomic.
        self._episode_seed = episode_seed
        self._maze = maze
        self._rng = _SplitMix64(rng_state)
        self._tick = tick
        self._player_x = player_x
        self._player_y = player_y
        self._target_x = target_x
        self._target_y = target_y
        self._previous_key_mask = previous_key_mask
        self._targets_collected = targets_collected
        self._times_caught = times_caught
        self._chasers = chasers
        return self.current_observation

    def state_hash(self) -> str:
        """Return a SHA-256 digest over the complete canonical state bytes."""
        return sha256(self.snapshot()).hexdigest()

    def _sample_empty_cell(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        candidates = sorted(
            (cell for cell in self._maze if cell not in occupied),
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

        self._paint(pixels, self._target_x, self._target_y, self._TARGET_COLOR)
        for chaser in self._chasers:
            self._paint(pixels, chaser.x, chaser.y, self._CHASER_COLOR)

        player_color = self._PLAYER_COLOR
        if any(
            chaser.x == self._player_x and chaser.y == self._player_y
            for chaser in self._chasers
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
class _Chaser:
    x: int
    y: int


__all__ = ["JunctionEnv"]
