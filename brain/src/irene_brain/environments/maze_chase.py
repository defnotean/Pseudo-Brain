"""maze_chase: the original Pac-Man-like environment (Phase 4 target).

An original, rights-clean chase-and-clear world: a perfect maze full of
pellets, one player, and deterministic ghosts that follow configurable
pursuit rules (direct chase, four-cell ambush, shy retreat, or a cyclical
mix). Eat every pellet to clear the maze (``cleared``, terminated); ghost
contact costs a life penalty and respawns the player (``caught``). No Namco
code, assets, names, or layouts — the maze is carved from the episode seed
by the same generator as the junction and keys/doors worlds, and every
behavior is defined in this file.

This is the culmination of the in-repo ladder (ROADMAP_TO_PACMAN.md §5):
moving shapes taught approach/avoidance, pursuit taught escape, junction
taught branch decisions, occlusion taught memory, keys/doors taught
sequencing. maze_chase demands all of them at once, at 60 Hz, under the same
render contract, control surface, and canonical snapshot rigor.
"""

from __future__ import annotations

from hashlib import sha256
from hmac import compare_digest
from struct import Struct
from typing import cast

from ..types import GenericControl, HidKey, Observation, RgbFrame, StepOutcome
from .junction import _DIRECTIONS, _bfs_distances, _carve_maze
from .moving_shapes import _SplitMix64, _paths_collide_at_same_time

_UINT64_MASK = (1 << 64) - 1

_PELLET_REWARD = 1.0
_CAUGHT_PENALTY = -10.0
_CLEARED_BONUS = 10.0

# Ghost pursuit rules. "direct" chases the player's cell; "ambush" targets
# four cells ahead of the player's last movement; "shy" chases while far and
# retreats while close; "mixed" assigns direct/ambush/shy cyclically by
# ghost index.
_GHOST_RULES = {"direct": 0, "ambush": 1, "shy": 2, "mixed": 3}
_AMBUSH_LEAD = 4
_SHY_DISTANCE = 8


def _carve_maze_with_loops(
    seed: int, grid_size: int, extra_loops: int
) -> frozenset[tuple[int, int]]:
    """The shared perfect maze plus ``extra_loops`` knocked walls.

    A perfect maze is a tree: a pursuing ghost is a hard wall, and no route
    around it ever exists. Real chase mazes have loops, so knocking a wall
    between two already-connected cells creates exactly one independent
    cycle. Loop carving is a pure function of the seed under its own domain
    separator, so snapshots still regenerate the maze by version rule.
    """

    maze = set(_carve_maze(seed, grid_size))
    rng = _SplitMix64(seed ^ 0x4C4F4F50)  # "LOOP" domain separator
    for _ in range(extra_loops):
        candidates = []
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                if (x, y) in maze:
                    continue
                horizontal = (x - 1, y) in maze and (x + 1, y) in maze
                vertical = (x, y - 1) in maze and (x, y + 1) in maze
                if horizontal or vertical:
                    candidates.append((x, y))
        if not candidates:
            break
        maze.add(candidates[rng.randbelow(len(candidates))])
    return frozenset(maze)


class MazeChaseEnv:
    """A deterministic 16x16 pellet maze with BFS ghosts and W/A/S/D control.

    Every corridor cell except the player spawn holds a pellet. The blue
    player moves one cell per call (walls refuse movement), eats pellets for
    +1 each, and clears the maze — ``cleared`` plus a terminated outcome —
    when the last pellet is eaten. Red ghosts each move one cell every
    ``ghost_period`` ticks under the configured ``ghost_rule``: ``direct``
    follows the exact BFS shortest path to the player, ``ambush`` targets
    the corridor cell four steps ahead of the player's last pressed
    direction, ``shy`` chases while far and retreats while close, and
    ``mixed`` assigns the three cyclically by ghost index. Contact under the
    shared swept-path same-time test emits ``caught``, costs 10 reward, and
    respawns the player at the spawn cell (or a sampled empty cell if a
    ghost occupies it). The speed-curve variant axis adds ``player_period``
    (the player acts once every N ticks; input is sampled on move ticks) and
    ``ghost_elroy`` (ghosts move one tick faster, minimum period 1, once at
    least half of the pellets are eaten). Simulator labels are returned only
    in :class:`StepOutcome`, never in :class:`Observation`.
    """

    GRID_SIZE = 16
    SNAPSHOT_VERSION = 3
    DEFAULT_TICK_PERIOD_NS = 16_666_667
    MAX_GHOST_PERIOD = 64
    MAX_PLAYER_PERIOD = 64
    MAX_EXTRA_LOOPS = 64

    # Public render-contract colors used by pixel-only procedural teachers.
    # Dataset generators may inspect these visible pixels, but must never route
    # simulator coordinates or other privileged state into a model input.
    PLAYER_RGB = (65, 174, 255)
    PELLET_RGB = (122, 118, 92)
    WALL_RGB = (42, 48, 66)
    PLAYER_GHOST_OVERLAP_RGB = (255, 255, 255)

    _SNAPSHOT_MAGIC = b"IBMC"
    _SNAPSHOT_HEADER = Struct("<4sHHIQQQQBBBBIIBBBbbBBB")
    _SNAPSHOT_GHOST = Struct("<BB")
    _SNAPSHOT_PELLET_BYTES = (GRID_SIZE * GRID_SIZE) // 8
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
    _PELLET_COLOR = PELLET_RGB
    _GHOST_COLOR = (225, 55, 65)
    _PLAYER_COLOR = PLAYER_RGB
    _WALL_COLOR = WALL_RGB
    _OVERLAP_COLOR = PLAYER_GHOST_OVERLAP_RGB

    def __init__(
        self,
        *,
        ghost_count: int = 3,
        ghost_period: int = 2,
        player_period: int = 1,
        extra_loops: int = 16,
        ghost_rule: str = "direct",
        ghost_elroy: bool = False,
        tick_period_ns: int = DEFAULT_TICK_PERIOD_NS,
        max_ticks: int = 10_000,
    ) -> None:
        if isinstance(ghost_count, bool) or not isinstance(ghost_count, int):
            raise TypeError("ghost_count must be an integer")
        if ghost_count < 1 or ghost_count > 8:
            raise ValueError("ghost_count must be in [1, 8]")
        if isinstance(ghost_period, bool) or not isinstance(ghost_period, int):
            raise TypeError("ghost_period must be an integer")
        if ghost_period < 1 or ghost_period > self.MAX_GHOST_PERIOD:
            raise ValueError(
                f"ghost_period must be in [1, {self.MAX_GHOST_PERIOD}]"
            )
        if isinstance(player_period, bool) or not isinstance(player_period, int):
            raise TypeError("player_period must be an integer")
        if player_period < 1 or player_period > self.MAX_PLAYER_PERIOD:
            raise ValueError(
                f"player_period must be in [1, {self.MAX_PLAYER_PERIOD}]"
            )
        if isinstance(extra_loops, bool) or not isinstance(extra_loops, int):
            raise TypeError("extra_loops must be an integer")
        if extra_loops < 0 or extra_loops > self.MAX_EXTRA_LOOPS:
            raise ValueError(
                f"extra_loops must be in [0, {self.MAX_EXTRA_LOOPS}]"
            )
        if not isinstance(ghost_rule, str) or ghost_rule not in _GHOST_RULES:
            raise ValueError(f"ghost_rule must be one of {sorted(_GHOST_RULES)}")
        if not isinstance(ghost_elroy, bool):
            raise TypeError("ghost_elroy must be a boolean")
        if isinstance(tick_period_ns, bool) or not isinstance(tick_period_ns, int):
            raise TypeError("tick_period_ns must be an integer")
        if tick_period_ns <= 0 or tick_period_ns > _UINT64_MASK:
            raise ValueError("tick_period_ns must be in [1, 2**64 - 1]")
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int):
            raise TypeError("max_ticks must be an integer")
        if max_ticks <= 0 or max_ticks > 0xFFFFFFFF:
            raise ValueError("max_ticks must be in [1, 2**32 - 1]")

        self._ghost_count = ghost_count
        self._ghost_period = ghost_period
        self._player_period = player_period
        self._extra_loops = extra_loops
        self._ghost_rule = ghost_rule
        self._ghost_elroy = ghost_elroy
        self._tick_period_ns = tick_period_ns
        self._max_ticks = max_ticks
        self._episode_seed = 0
        self._rng = _SplitMix64(0)
        self._maze: frozenset[tuple[int, int]] = frozenset()
        self._pellets = bytearray(self._SNAPSHOT_PELLET_BYTES)
        self._pellets_remaining = 0
        self._tick = 0
        self._player_x = 1
        self._player_y = 1
        self._player_dx = 0
        self._player_dy = -1
        self._previous_key_mask = 0
        self._pellets_eaten = 0
        self._times_caught = 0
        self._cleared = 0
        self._ghosts: list[tuple[int, int]] = []
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

    def _pellet_index(self, x: int, y: int) -> tuple[int, int]:
        cell = y * self.GRID_SIZE + x
        return cell // 8, cell % 8

    def _has_pellet(self, x: int, y: int) -> bool:
        byte, bit = self._pellet_index(x, y)
        return bool(self._pellets[byte] & (1 << bit))

    def _eat_pellet(self, x: int, y: int) -> None:
        byte, bit = self._pellet_index(x, y)
        self._pellets[byte] &= ~(1 << bit) & 0xFF
        self._pellets_remaining -= 1

    def reset(self, seed: int) -> Observation:
        """Reset to the unique initial state selected by a 64-bit integer seed."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if seed < 0 or seed > _UINT64_MASK:
            raise ValueError("seed must be in [0, 2**64 - 1]")

        self._episode_seed = seed
        self._maze = _carve_maze_with_loops(seed, self.GRID_SIZE, self._extra_loops)
        self._rng = _SplitMix64(seed)
        self._tick = 0
        self._player_x = 1
        self._player_y = 1
        self._player_dx = 0
        self._player_dy = -1
        self._previous_key_mask = 0
        self._pellets_eaten = 0
        self._times_caught = 0
        self._cleared = 0
        self._ghosts = []

        self._pellets = bytearray(self._SNAPSHOT_PELLET_BYTES)
        self._pellets_remaining = 0
        for x, y in self._maze:
            if (x, y) == (self._player_x, self._player_y):
                continue
            byte, bit = self._pellet_index(x, y)
            self._pellets[byte] |= 1 << bit
            self._pellets_remaining += 1

        # Ghosts spawn at distinct far corridor cells, chosen deterministically.
        player = (self._player_x, self._player_y)
        distances = _bfs_distances(player, self._maze)
        farthest_distance = max(distances.values())
        minimum = min(10, farthest_distance)
        candidates = sorted(
            (
                cell
                for cell, distance in distances.items()
                if distance >= minimum and cell != player
            ),
            key=lambda cell: (cell[1], cell[0]),
        )
        for _ in range(self._ghost_count):
            if candidates:
                cell = candidates[self._rng.randbelow(len(candidates))]
                candidates.remove(cell)
            else:
                cell = self._sample_empty_cell({player, *self._ghosts})
            self._ghosts.append(cell)

        return self.current_observation

    def step(self, control: GenericControl) -> StepOutcome:
        """Apply one control state and advance exactly one logical tick."""
        if not isinstance(control, GenericControl):
            raise TypeError("control must be a GenericControl")
        if self._cleared:
            raise RuntimeError("the maze is cleared; call reset or restore before stepping")
        if self._tick >= self._max_ticks:
            raise RuntimeError("the lifetime is truncated; call reset or restore before stepping")

        applied_mask = self._mask_from_control(control)
        applied_control = self._control_from_mask(applied_mask)
        old_player = (self._player_x, self._player_y)

        # The player acts once every player_period ticks; control input is
        # sampled on move ticks only, so a direction pressed and released
        # between move ticks is ignored entirely (the speed curve slows the
        # player's whole decision loop, not just its motion).
        player_moves = self._tick % self._player_period == 0
        if player_moves:
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
            if candidate in self._maze:
                self._player_x, self._player_y = candidate
            # Facing tracks control intent, not achieved motion: a wall-refused
            # press still turns the player, matching how the ambush rule reads
            # the player's last input rather than simulator state.
            if horizontal or vertical:
                self._player_dx, self._player_dy = horizontal, vertical
        new_player = (self._player_x, self._player_y)

        events: list[str] = []
        reward = 0.0
        if self._has_pellet(self._player_x, self._player_y):
            self._eat_pellet(self._player_x, self._player_y)
            self._pellets_eaten += 1
            reward += _PELLET_REWARD
            events.append("pellet_eaten")

        ghost_paths: list[tuple[tuple[int, int], tuple[int, int]]] = []
        ghosts_move = self._tick % self._effective_ghost_period() == 0
        player_field = _bfs_distances(new_player, self._maze) if ghosts_move else {}
        moved_ghosts: list[tuple[int, int]] = []
        for index, ghost in enumerate(self._ghosts):
            before = ghost
            after = ghost
            if ghosts_move:
                rule = self._rule_for_ghost(index)
                if rule == "shy":
                    after = self._shy_step(before, player_field)
                else:
                    if rule == "ambush":
                        target = self._ambush_target(new_player)
                        field = _bfs_distances(target, self._maze)
                    else:
                        field = player_field
                    after = self._step_downhill(before, field)
            ghost_paths.append((before, after))
            moved_ghosts.append(after)
        self._ghosts = moved_ghosts

        caught = any(
            _paths_collide_at_same_time(old_player, new_player, before, after)
            for before, after in ghost_paths
        )
        terminated = False
        if caught:
            self._times_caught += 1
            reward += _CAUGHT_PENALTY
            events.append("caught")
            ghost_cells = set(self._ghosts)
            if (1, 1) not in ghost_cells:
                self._player_x, self._player_y = 1, 1
            else:
                self._player_x, self._player_y = self._sample_empty_cell(ghost_cells)

        if self._pellets_remaining == 0:
            self._cleared = 1
            reward += _CLEARED_BONUS
            events.append("cleared")
            terminated = True

        self._tick += 1
        self._previous_key_mask = applied_mask
        return StepOutcome(
            observation=self.current_observation,
            requested_control=control,
            applied_control=applied_control,
            reward=reward,
            events=tuple(events),
            terminated=terminated,
            truncated=self._tick >= self._max_ticks,
        )

    def snapshot(self) -> bytes:
        """Return a canonical, checksummed, version-3 snapshot."""
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
                self._previous_key_mask,
                self._cleared,
                self._pellets_eaten,
                self._times_caught,
                self._ghost_count,
                self._ghost_period,
                self._extra_loops,
                self._player_dx,
                self._player_dy,
                _GHOST_RULES[self._ghost_rule],
                self._player_period,
                int(self._ghost_elroy),
            )
        )
        payload.extend(self._pellets)
        for ghost in self._ghosts:
            payload.extend(self._SNAPSHOT_GHOST.pack(*ghost))
        payload.extend(sha256(payload).digest())
        return bytes(payload)

    def restore(self, snapshot: bytes) -> Observation:
        """Atomically restore a validated snapshot produced by this version."""
        if not isinstance(snapshot, bytes):
            raise TypeError("snapshot must be bytes")
        minimum_length = (
            self._SNAPSHOT_HEADER.size
            + self._SNAPSHOT_PELLET_BYTES
            + self._SNAPSHOT_DIGEST_BYTES
        )
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
                int, int, int, int, int,
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
            previous_key_mask,
            cleared,
            pellets_eaten,
            times_caught,
            ghost_count,
            ghost_period,
            extra_loops,
            player_dx,
            player_dy,
            ghost_rule_code,
            player_period,
            ghost_elroy,
        ) = header
        if magic != self._SNAPSHOT_MAGIC:
            raise ValueError("snapshot magic mismatch")
        if version != self.SNAPSHOT_VERSION:
            raise ValueError("snapshot version mismatch")
        if grid_size != self.GRID_SIZE:
            raise ValueError("snapshot grid size mismatch")
        if ghost_count != self._ghost_count:
            raise ValueError("snapshot ghost count mismatch")
        if ghost_period != self._ghost_period:
            raise ValueError("snapshot ghost period mismatch")
        if extra_loops != self._extra_loops:
            raise ValueError("snapshot extra-loops mismatch")
        if max_ticks != self._max_ticks:
            raise ValueError("snapshot max_ticks mismatch")
        if tick_period_ns != self._tick_period_ns:
            raise ValueError("snapshot tick period mismatch")
        expected = (
            self._SNAPSHOT_HEADER.size
            + self._SNAPSHOT_PELLET_BYTES
            + ghost_count * self._SNAPSHOT_GHOST.size
        )
        if len(payload) != expected:
            raise ValueError("snapshot length mismatch")
        if tick > max_ticks:
            raise ValueError("snapshot tick exceeds max_ticks")
        if previous_key_mask & ~self._SUPPORTED_KEY_MASK:
            raise ValueError("snapshot contains unsupported key bits")
        if cleared not in (0, 1):
            raise ValueError("snapshot cleared flag must be 0 or 1")
        if player_dx not in (-1, 0, 1) or player_dy not in (-1, 0, 1):
            raise ValueError("snapshot facing components must be in {-1, 0, 1}")
        if player_dx == 0 and player_dy == 0:
            raise ValueError("snapshot facing must be a nonzero direction")
        if ghost_rule_code not in _GHOST_RULES.values():
            raise ValueError("snapshot ghost rule is unknown")
        if ghost_rule_code != _GHOST_RULES[self._ghost_rule]:
            raise ValueError("snapshot ghost rule mismatch")
        if player_period != self._player_period:
            raise ValueError("snapshot player period mismatch")
        if ghost_elroy not in (0, 1):
            raise ValueError("snapshot elroy flag must be 0 or 1")
        if bool(ghost_elroy) != self._ghost_elroy:
            raise ValueError("snapshot elroy flag mismatch")

        maze = _carve_maze_with_loops(episode_seed, self.GRID_SIZE, extra_loops)
        if (player_x, player_y) not in maze:
            raise ValueError("snapshot contains an off-corridor player")

        pellets = bytearray(
            payload[
                self._SNAPSHOT_HEADER.size : self._SNAPSHOT_HEADER.size
                + self._SNAPSHOT_PELLET_BYTES
            ]
        )
        pellets_remaining = 0
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                byte, bit = self._pellet_index(x, y)
                present = bool(pellets[byte] & (1 << bit))
                if present and (x, y) not in maze:
                    raise ValueError("snapshot has a pellet inside a wall")
                if present and (x, y) == (1, 1):
                    raise ValueError("snapshot has a pellet on the spawn cell")
                pellets_remaining += int(present)
        if pellets_eaten + pellets_remaining != len(maze) - 1:
            raise ValueError("snapshot pellet accounting is inconsistent")

        ghosts: list[tuple[int, int]] = []
        offset = self._SNAPSHOT_HEADER.size + self._SNAPSHOT_PELLET_BYTES
        for _ in range(ghost_count):
            x, y = cast(
                tuple[int, int], self._SNAPSHOT_GHOST.unpack_from(payload, offset)
            )
            offset += self._SNAPSHOT_GHOST.size
            if (x, y) not in maze:
                raise ValueError("snapshot contains an off-corridor ghost")
            ghosts.append((x, y))

        # Assign only after every byte has been validated, keeping failed restores atomic.
        self._episode_seed = episode_seed
        self._maze = maze
        self._pellets = pellets
        self._pellets_remaining = pellets_remaining
        self._rng = _SplitMix64(rng_state)
        self._tick = tick
        self._player_x = player_x
        self._player_y = player_y
        self._player_dx = player_dx
        self._player_dy = player_dy
        self._previous_key_mask = previous_key_mask
        self._cleared = cleared
        self._pellets_eaten = pellets_eaten
        self._times_caught = times_caught
        self._ghosts = ghosts
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

    def _rule_for_ghost(self, index: int) -> str:
        """Resolve the pursuit rule for one ghost under the configured rule."""
        if self._ghost_rule == "mixed":
            return ("direct", "ambush", "shy")[index % 3]
        return self._ghost_rule

    def _effective_ghost_period(self) -> int:
        """Return the current ghost period under the elroy speed curve.

        With ``ghost_elroy`` enabled, ghosts move one tick faster (minimum
        period 1) once at least half of the pellets have been eaten. The
        curve derives from ``pellets_remaining`` and the regenerated maze, so
        snapshots need no extra state beyond the config flag.
        """
        if self._ghost_elroy and self._pellets_remaining * 2 <= len(self._maze) - 1:
            return max(1, self._ghost_period - 1)
        return self._ghost_period

    def _ambush_target(self, player: tuple[int, int]) -> tuple[int, int]:
        """Return the corridor cell four steps ahead of the player's facing.

        The raw lead cell is clamped to the grid; if it lands inside a wall
        the nearest corridor cell by Manhattan distance is used instead, with
        row-major scan order breaking ties deterministically.
        """
        raw = (
            min(self.GRID_SIZE - 1, max(0, player[0] + _AMBUSH_LEAD * self._player_dx)),
            min(self.GRID_SIZE - 1, max(0, player[1] + _AMBUSH_LEAD * self._player_dy)),
        )
        if raw in self._maze:
            return raw
        best = player
        best_distance: int | None = None
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) not in self._maze:
                    continue
                distance = abs(x - raw[0]) + abs(y - raw[1])
                if best_distance is None or distance < best_distance:
                    best = (x, y)
                    best_distance = distance
        return best

    def _shy_step(
        self, ghost: tuple[int, int], player_field: dict[tuple[int, int], int]
    ) -> tuple[int, int]:
        """Chase while farther than ``_SHY_DISTANCE``; otherwise retreat uphill."""
        here = player_field.get(ghost)
        if here is None:
            return ghost
        if here > _SHY_DISTANCE:
            return self._step_downhill(ghost, player_field)
        for dx, dy in _DIRECTIONS:
            neighbor = (ghost[0] + dx, ghost[1] + dy)
            if player_field.get(neighbor) == here + 1:
                return neighbor
        return ghost

    @staticmethod
    def _step_downhill(
        cell: tuple[int, int], distances: dict[tuple[int, int], int]
    ) -> tuple[int, int]:
        """Move one cell along the exact BFS shortest path toward the field origin."""
        here = distances.get(cell)
        if here is None or here == 0:
            return cell
        for dx, dy in _DIRECTIONS:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if distances.get(neighbor) == here - 1:
                return neighbor
        return cell

    def _render(self) -> RgbFrame:
        pixels = bytearray(self.GRID_SIZE * self.GRID_SIZE * 3)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) not in self._maze:
                    color = self._WALL_COLOR
                elif self._has_pellet(x, y):
                    color = self._PELLET_COLOR
                else:
                    color = (
                        self._BACKGROUND_EVEN if (x + y) % 2 == 0 else self._BACKGROUND_ODD
                    )
                self._paint(pixels, x, y, color)

        for ghost in self._ghosts:
            self._paint(pixels, ghost[0], ghost[1], self._GHOST_COLOR)

        player_color = self._PLAYER_COLOR
        if (self._player_x, self._player_y) in set(self._ghosts):
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


__all__ = ["MazeChaseEnv"]
