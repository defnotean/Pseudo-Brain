"""PLAN.md §28 diagnostic policies for the closed-loop moving-shapes world.

The baseline suite requires every phase to report the same evaluator rows for
non-model reference policies: random and no-op policies (§28.1), a simple
scripted diagnostic policy (§28.2), and a privileged-state oracle for
diagnosis only (§28.15). This module implements those four decision sources
behind one tiny contract so they run through
:func:`irene_brain.evaluation.closed_loop_play.run_policy_closed_loop_episode`
with exactly the same clock, deadline, and rejection mechanics as a model.

Policy contract:

- ``identity`` — a stable dotted identifier recorded in the evidence report.
- ``uses_privileged_state`` — True only for the oracle. Privileged policies
  receive the live environment through ``bind(environment)`` after reset and
  may read simulator internals. Everything they produce is diagnosis-only
  evidence and must never be routed into a model input or training target.
- ``reset(episode_seed)`` — reseed per episode so multi-seed campaigns stay
  deterministic and independent.
- ``act(observation)`` — return the next :class:`GenericControl`.

The scripted chaser is deliberately pixel-only: it reads the public
render-contract colors (``PLAYER_RGB`` / ``TARGET_RGB``) from the canonical
observation frame and never touches simulator state, demonstrating that a
procedural teacher needs nothing beyond what a model would see.
"""

from __future__ import annotations

from typing import Sequence

from ..environments.junction import JunctionEnv, _DIRECTIONS, _bfs_distances
from ..environments.keys_doors import KeysDoorsEnv
from ..environments.maze_chase import MazeChaseEnv
from ..environments.moving_shapes import (
    MovingShapesEnv,
    _SplitMix64,
    _paths_collide_at_same_time,
)
from ..types import GenericControl, Observation, RgbFrame
from .closed_loop_play import (
    ClosedLoopPlayConfig,
    ClosedLoopPlayReport,
    run_policy_closed_loop_episode,
)

# HID usage identifiers, bit-packed exactly like MovingShapesEnv._KEY_BITS.
_KEY_W = 26
_KEY_A = 4
_KEY_S = 22
_KEY_D = 7
_KEY_FOR_DELTA = {(0, -1): _KEY_W, (-1, 0): _KEY_A, (0, 1): _KEY_S, (1, 0): _KEY_D}
# Movement-mask bits, matching _movement_control and MazeChaseEnv._KEY_BITS.
_MASK_FOR_DELTA = {(0, -1): 1, (-1, 0): 2, (0, 1): 4, (1, 0): 8}


def _movement_control(mask: int) -> GenericControl:
    keys = []
    if mask & 1:
        keys.append(_KEY_W)
    if mask & 2:
        keys.append(_KEY_A)
    if mask & 4:
        keys.append(_KEY_S)
    if mask & 8:
        keys.append(_KEY_D)
    return GenericControl(keys_down=tuple(keys))


class NoOpPolicy:
    """§28.1 no-op floor: never presses anything."""

    __slots__ = ()
    identity = "diagnostic.noop.v1"
    uses_privileged_state = False

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")

    def act(self, observation: Observation) -> GenericControl:
        return GenericControl()


class RandomMovementPolicy:
    """§28.1 random floor: a uniform W/A/S/D subset per decision."""

    __slots__ = ("_rng",)
    identity = "diagnostic.random_movement.v1"
    uses_privileged_state = False

    def __init__(self) -> None:
        self._rng = _SplitMix64(0)

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        self._rng = _SplitMix64(episode_seed ^ 0x5D1A60571C001)

    def act(self, observation: Observation) -> GenericControl:
        return _movement_control(self._rng.randbelow(16))


class ScriptedTargetChasePolicy:
    """§28.2 scripted diagnostic: pixel-only greedy chase of the target.

    Locates the player and target cells by the public render-contract colors
    and presses the one or two movement keys that reduce the Manhattan
    distance. Hazards are ignored on purpose: this is the simplest teacher
    that solves the task's approach component, and its collision count is
    part of the diagnostic signal.
    """

    __slots__ = ()
    identity = "diagnostic.scripted_chase.v1"
    uses_privileged_state = False

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")

    def act(self, observation: Observation) -> GenericControl:
        frame = observation.rgb
        grid = MovingShapesEnv.GRID_SIZE
        if frame.width != grid or frame.height != grid:
            raise ValueError("scripted chaser requires the canonical grid frame")
        player: tuple[int, int] | None = None
        target: tuple[int, int] | None = None
        pixels = frame.pixels
        for y in range(grid):
            for x in range(grid):
                offset = (y * grid + x) * 3
                color = (
                    pixels[offset],
                    pixels[offset + 1],
                    pixels[offset + 2],
                )
                if color == MovingShapesEnv.TARGET_RGB:
                    target = (x, y)
                elif color in (
                    MovingShapesEnv.PLAYER_RGB,
                    MovingShapesEnv.PLAYER_HAZARD_OVERLAP_RGB,
                ):
                    player = (x, y)
        if player is None:
            raise RuntimeError("scripted chaser could not locate the player")
        if target is None:
            # The player is standing on the target cell (the player is painted
            # last, so the target is occluded for exactly this observation).
            # Hold still; the relocated target is visible again next tick.
            return GenericControl()
        keys = []
        if target[1] < player[1]:
            keys.append(_KEY_W)
        elif target[1] > player[1]:
            keys.append(_KEY_S)
        if target[0] < player[0]:
            keys.append(_KEY_A)
        elif target[0] > player[0]:
            keys.append(_KEY_D)
        return GenericControl(keys_down=tuple(keys))


class OraclePolicy:
    """§28.15 privileged-state oracle, for diagnosis only.

    Reads simulator internals (player, target, hazard positions and
    velocities), predicts the hazards' next cells with the exact bounce rule,
    and takes the safe move that minimizes Manhattan distance to the target.
    "Safe" reuses the environment's own swept-path collision test, so the
    oracle's collision count measures only unavoidable timing traps, never
    prediction error. Its outputs must never be routed into a model input or
    training target.
    """

    __slots__ = ("_environment",)
    identity = "diagnostic.oracle_privileged.v1"
    uses_privileged_state = True

    def __init__(self) -> None:
        self._environment: MovingShapesEnv | None = None

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        self._environment = None

    def bind(self, environment: MovingShapesEnv) -> None:
        if not isinstance(environment, MovingShapesEnv):
            raise TypeError("oracle can only bind a MovingShapesEnv")
        self._environment = environment

    def act(self, observation: Observation) -> GenericControl:
        env = self._environment
        if env is None:
            raise RuntimeError("oracle act() before bind()")
        grid = MovingShapesEnv.GRID_SIZE
        player = (env._player_x, env._player_y)
        target = (env._target_x, env._target_y)
        hazard_paths = []
        for hazard in env._hazards:
            before = (hazard.x, hazard.y)
            next_x, vx = hazard.x + hazard.vx, hazard.vx
            if next_x < 0 or next_x >= grid:
                vx, next_x = -vx, hazard.x - hazard.vx
            next_y, vy = hazard.y + hazard.vy, hazard.vy
            if next_y < 0 or next_y >= grid:
                vy, next_y = -vy, hazard.y - hazard.vy
            hazard_paths.append((before, (next_x, next_y)))

        # Deterministic preference order: straight moves, stay, diagonals.
        candidates = (
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (0, 0),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        )

        def distance(delta: tuple[int, int]) -> int:
            new_x = min(grid - 1, max(0, player[0] + delta[0]))
            new_y = min(grid - 1, max(0, player[1] + delta[1]))
            return abs(new_x - target[0]) + abs(new_y - target[1])

        def safe(delta: tuple[int, int]) -> bool:
            new_x = min(grid - 1, max(0, player[0] + delta[0]))
            new_y = min(grid - 1, max(0, player[1] + delta[1]))
            after = (new_x, new_y)
            return not any(
                _paths_collide_at_same_time(player, after, before, hazard_after)
                for before, hazard_after in hazard_paths
            )

        safe_candidates = [delta for delta in candidates if safe(delta)]
        pool = safe_candidates if safe_candidates else list(candidates)
        chosen = min(pool, key=lambda delta: (distance(delta), candidates.index(delta)))
        keys = []
        if chosen[1] != 0:
            keys.append(_KEY_FOR_DELTA[(0, chosen[1])])
        if chosen[0] != 0:
            keys.append(_KEY_FOR_DELTA[(chosen[0], 0)])
        return GenericControl(keys_down=tuple(keys))


def _parse_maze_chase_frame(
    frame: RgbFrame, *, caller: str
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    tuple[int, int],
    set[tuple[int, int]],
]:
    """Parse the canonical maze_chase grid frame into visible cell sets.

    Returns ``(walkable, pellets, player, ghosts)``. Every color read here is
    part of the public render contract — ``WALL_RGB`` / ``PELLET_RGB`` /
    ``PLAYER_RGB`` and the visible ghost red — so a pixel-only policy sees
    exactly what a model would see and never touches simulator state.
    """

    grid = MazeChaseEnv.GRID_SIZE
    if frame.width != grid or frame.height != grid:
        raise ValueError(f"{caller} requires the canonical grid frame")
    # The ghost red is a visible pixel color; it is read from the frame,
    # never from simulator state.
    ghost_color = MazeChaseEnv._GHOST_COLOR
    pixels = frame.pixels
    player: tuple[int, int] | None = None
    walkable: set[tuple[int, int]] = set()
    pellets: set[tuple[int, int]] = set()
    ghosts: set[tuple[int, int]] = set()
    for y in range(grid):
        for x in range(grid):
            offset = (y * grid + x) * 3
            color = (pixels[offset], pixels[offset + 1], pixels[offset + 2])
            if color == MazeChaseEnv.WALL_RGB:
                continue
            cell = (x, y)
            walkable.add(cell)
            if color == MazeChaseEnv.PELLET_RGB:
                pellets.add(cell)
            elif color in (
                MazeChaseEnv.PLAYER_RGB,
                MazeChaseEnv.PLAYER_GHOST_OVERLAP_RGB,
            ):
                player = cell
            elif color == ghost_color:
                ghosts.add(cell)
    if player is None:
        raise RuntimeError(f"{caller} could not locate the player")
    return walkable, pellets, player, ghosts


class ScriptedPelletTeacherPolicy:
    """Pixel-only greedy pellet teacher for the maze_chase world.

    Reads only the rendered frame: walls and pellets via the public
    ``WALL_RGB`` / ``PELLET_RGB`` contract, the player via ``PLAYER_RGB``,
    and ghosts as the visible red cells. It walks the shortest safe path to
    the nearest visible pellet, treating cells within one BFS step of a ghost
    as blocked and falling back to the plain shortest path when avoidance
    makes every pellet unreachable. On worlds without pellets it holds
    still, so its matrix rows there are honest zeros.
    """

    __slots__ = ()
    identity = "diagnostic.scripted_pellet_teacher.v1"
    uses_privileged_state = False

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")

    def act(self, observation: Observation) -> GenericControl:
        walkable, pellets, player, ghosts = _parse_maze_chase_frame(
            observation.rgb, caller="pellet teacher"
        )
        if not pellets:
            return GenericControl()

        threatened: set[tuple[int, int]] = set()
        for ghost in ghosts:
            for cell, distance in _bfs_distances(ghost, frozenset(walkable)).items():
                if distance <= 1:
                    threatened.add(cell)

        frozen_walkable = frozenset(walkable)

        def nearest_goal(blocked: set[tuple[int, int]]) -> tuple[int, int] | None:
            open_cells = frozen_walkable - blocked
            if player not in open_cells:
                return None
            best = None
            for cell, distance in _bfs_distances(player, open_cells).items():
                if distance == 0 or cell not in pellets:
                    continue
                key = (distance, cell[1], cell[0])
                if best is None or key < best:
                    best = key
            return None if best is None else (best[2], best[1])

        goal = nearest_goal(threatened)
        if goal is None:
            goal = nearest_goal(set())
        if goal is None:
            return GenericControl()

        from_goal = _bfs_distances(goal, frozen_walkable)
        here = from_goal[player]
        fallback: int | None = None
        for (dx, dy), key in (
            ((0, -1), _KEY_W),
            ((-1, 0), _KEY_A),
            ((0, 1), _KEY_S),
            ((1, 0), _KEY_D),
        ):
            neighbor = (player[0] + dx, player[1] + dy)
            if from_goal.get(neighbor) == here - 1:
                if neighbor in threatened:
                    fallback = key
                    continue
                return GenericControl(keys_down=(key,))
        if fallback is not None:
            return GenericControl(keys_down=(fallback,))
        return GenericControl()


class ScriptedMazeChasePlannerPolicy:
    """Pixel-only lookahead planner for the canonical maze_chase world.

    Where the pellet teacher is greedy — nearest pellet, one step of ghost
    avoidance — the planner simulates the world's published mechanics several
    ticks ahead: ghosts step one cell toward the player along the exact BFS
    shortest path every ``ghost_period`` ticks, and contact is checked with
    the same swept-path rule the simulator uses. It reconstructs the shortest
    path to each of the ``candidate_pellets`` nearest visible pellets,
    simulates walking each path end to end (capped at ``horizon`` steps), and
    commits to the first path it can walk without being caught. If no pellet
    path is safe it falls back to the one-tick move that survives and
    maximizes the post-move distance to the nearest ghost.

    The planner reads nothing but the canonical observation frame: walls,
    pellets, the player, and the visible ghost cells all come from pixels,
    never from simulator state. Two honest pixel-vision caveats: the
    player/ghost overlap pixel briefly hides a ghost standing on the player's
    own cell, and two ghosts stacked on one cell render as one. Because the
    planner re-plans from fresh pixels every tick, a hidden ghost re-enters
    the plan as soon as it separates.

    Actuation awareness is configurable so the same planner can run matched
    to a variant slot's published mechanics:

    - ``input_delay_ticks`` — the planner tracks its own submitted presses
      (its own outputs, not simulator state) as a model of the world's delay
      FIFO, simulates the forced prefix those queued presses determine, and
      aims each new press at the tick it will actually apply.
    - ``player_period`` — the simulation gates player movement (and input
      sampling) to every Nth tick, like the world's speed curve.
    - ``ghost_elroy`` — the simulation shortens the ghost period by one once
      the visible pellet count says at least half the pellets are eaten, the
      same derivation the world uses. Pellets hidden under ghosts make the
      visible count an undercount, so the simulated speed-up can arrive
      slightly early — conservative, never optimistic.

    Remaining blind spots, by construction: ghost rules other than direct
    pursuit are not modeled (ambush/shy/mixed degrade to optimistic
    direct-pursuit guesses), and when the forced prefix is already fatal the
    post-catch respawn cell is sampled from hidden RNG, so the fallback plan
    past that point is approximate.
    """

    __slots__ = (
        "_ghost_period",
        "_candidate_pellets",
        "_horizon",
        "_input_delay_ticks",
        "_player_period",
        "_ghost_elroy",
        "_issued",
    )
    identity = "diagnostic.scripted_maze_chase_planner.v1"
    uses_privileged_state = False

    def __init__(
        self,
        *,
        ghost_period: int = 2,
        candidate_pellets: int = 6,
        horizon: int = 24,
        input_delay_ticks: int = 0,
        player_period: int = 1,
        ghost_elroy: bool = False,
    ) -> None:
        for value, name, low, high in (
            (ghost_period, "ghost_period", 1, 64),
            (candidate_pellets, "candidate_pellets", 1, 32),
            (horizon, "horizon", 1, 256),
            (input_delay_ticks, "input_delay_ticks", 0, 16),
            (player_period, "player_period", 1, 64),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < low or value > high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if not isinstance(ghost_elroy, bool):
            raise TypeError("ghost_elroy must be a boolean")
        self._ghost_period = ghost_period
        self._candidate_pellets = candidate_pellets
        self._horizon = horizon
        self._input_delay_ticks = input_delay_ticks
        self._player_period = player_period
        self._ghost_elroy = ghost_elroy
        self._issued: list[int] = [0] * input_delay_ticks

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        # The world starts its delay FIFO as all zeros; mirror that.
        self._issued = [0] * self._input_delay_ticks

    def act(self, observation: Observation) -> GenericControl:
        walkable, pellets, player, ghosts = _parse_maze_chase_frame(
            observation.rgb, caller="maze chase planner"
        )
        if not pellets:
            return self._commit(0)
        frozen_walkable = frozenset(walkable)
        # frame_id counts completed ticks, so the upcoming decision executes
        # on simulator tick ``start_tick`` and ghosts move on ticks divisible
        # by the period — the same condition the simulator evaluates.
        start_tick = observation.frame_id

        # Forced prefix: the next ``input_delay_ticks`` applied masks are
        # already sitting in the world's FIFO — they are this policy's own
        # earlier presses, so the policy knows them exactly. Simulate those
        # ticks first; the press chosen now applies at ``plan_tick``.
        sim_pellets = set(pellets)
        prefix_player = player
        prefix_ghosts = sorted(ghosts)
        prefix_caught = False
        for offset, queued_mask in enumerate(self._issued):
            prefix_player, prefix_ghosts, caught = self._step_sim(
                prefix_player,
                prefix_ghosts,
                sim_pellets,
                queued_mask,
                start_tick + offset,
                frozen_walkable,
            )
            prefix_caught = prefix_caught or caught
        plan_tick = start_tick + len(self._issued)

        mask = 0
        if not prefix_caught:
            mask = self._plan_from(
                prefix_player,
                prefix_ghosts,
                sim_pellets,
                frozen_walkable,
                plan_tick,
            )
        if mask == 0 and not prefix_caught:
            mask = self._survival_mask(
                prefix_player,
                prefix_ghosts,
                sim_pellets,
                frozen_walkable,
                plan_tick,
            )
        return self._commit(mask)

    def _commit(self, mask: int) -> GenericControl:
        """Record the submitted mask in the FIFO model and build the control."""
        if self._input_delay_ticks:
            self._issued = self._issued[1:] + [mask]
        if mask == 0:
            return GenericControl()
        keys = []
        for delta, bit in _MASK_FOR_DELTA.items():
            if mask & bit:
                keys.append(_KEY_FOR_DELTA[delta])
        return GenericControl(keys_down=tuple(keys))

    def _effective_period(self, visible_pellets: int, walkable_cells: int) -> int:
        """Ghost period under the elroy curve, derived from visible pixels.

        The world speeds ghosts up once ``pellets_remaining * 2 <= cells - 1``;
        both quantities are visible (pellet pixels, walkable cells), and pellet
        pixels hidden under ghosts only make the simulated speed-up early —
        conservative, never optimistic.
        """
        if self._ghost_elroy and visible_pellets * 2 <= walkable_cells - 1:
            return max(1, self._ghost_period - 1)
        return self._ghost_period

    def _step_sim(
        self,
        sim_player: tuple[int, int],
        sim_ghosts: list[tuple[int, int]],
        sim_pellets: set[tuple[int, int]],
        mask: int,
        tick: int,
        walkable: frozenset[tuple[int, int]],
    ) -> tuple[tuple[int, int], list[tuple[int, int]], bool]:
        """Advance one simulated tick; return (player, ghosts, caught)."""
        new_player = sim_player
        if tick % self._player_period == 0:
            delta = (
                int(bool(mask & 8)) - int(bool(mask & 2)),
                int(bool(mask & 4)) - int(bool(mask & 1)),
            )
            candidate = (sim_player[0] + delta[0], sim_player[1] + delta[1])
            if candidate in walkable:
                new_player = candidate
        sim_pellets.discard(new_player)
        period = self._effective_period(len(sim_pellets), len(walkable))
        ghosts_move = tick % period == 0
        # Every direct-pursuit ghost chases the same post-move player
        # cell, so one shared field serves them all — the same field the
        # simulator computes once per move tick.
        field = _bfs_distances(new_player, walkable) if ghosts_move else None
        moved: list[tuple[int, int]] = []
        caught = False
        for ghost in sim_ghosts:
            after = (
                MazeChaseEnv._step_downhill(ghost, field) if ghosts_move else ghost
            )
            if _paths_collide_at_same_time(sim_player, new_player, ghost, after):
                caught = True
            moved.append(after)
        return new_player, moved, caught

    def _plan_from(
        self,
        player: tuple[int, int],
        ghosts: list[tuple[int, int]],
        sim_pellets: set[tuple[int, int]],
        walkable: frozenset[tuple[int, int]],
        plan_tick: int,
    ) -> int:
        """Find a safe pellet path from ``plan_tick``; return its first mask."""
        from_player = _bfs_distances(player, walkable)
        ranked = sorted(
            (from_player[cell], cell[1], cell[0])
            for cell in sim_pellets
            if cell in from_player and from_player[cell] > 0
        )[: self._candidate_pellets]
        for distance, goal_y, goal_x in ranked:
            if distance > self._horizon:
                continue
            path = _downhill_path(from_player, player, (goal_x, goal_y))
            if not self._path_is_caught(
                player, ghosts, set(sim_pellets), path, walkable, plan_tick
            ):
                delta = (path[0][0] - player[0], path[0][1] - player[1])
                return _MASK_FOR_DELTA[delta]
        return 0

    def _path_is_caught(
        self,
        player: tuple[int, int],
        ghosts: list[tuple[int, int]],
        sim_pellets: set[tuple[int, int]],
        path: list[tuple[int, int]],
        walkable: frozenset[tuple[int, int]],
        start_tick: int,
    ) -> bool:
        """Simulate walking ``path``; return True if the player is caught.

        The player takes one path step per move tick (ticks divisible by
        ``player_period``) and holds still otherwise; presses landing on
        non-move ticks are discarded by the world, exactly as simulated.
        """
        sim_player = player
        sim_ghosts = list(ghosts)
        step_index = 0
        tick = start_tick
        while step_index < len(path):
            mask = 0
            if tick % self._player_period == 0:
                step_cell = path[step_index]
                mask = _MASK_FOR_DELTA[
                    (step_cell[0] - sim_player[0], step_cell[1] - sim_player[1])
                ]
                step_index += 1
            sim_player, sim_ghosts, caught = self._step_sim(
                sim_player, sim_ghosts, sim_pellets, mask, tick, walkable
            )
            if caught:
                return True
            tick += 1
        return False

    def _survival_mask(
        self,
        player: tuple[int, int],
        ghosts: list[tuple[int, int]],
        sim_pellets: set[tuple[int, int]],
        walkable: frozenset[tuple[int, int]],
        plan_tick: int,
    ) -> int:
        """One-tick fallback: survive, then maximize distance to the ghosts.

        Options scan in the fixed ``_DIRECTIONS`` order with stay last; a
        strict ``>`` comparison keeps the earliest option on ties, so the
        fallback is deterministic. A press landing on a non-move tick is
        discarded by the world, so the fallback then holds still.
        """
        if plan_tick % self._player_period != 0:
            return 0
        best_score: int | None = None
        best_mask = 0
        options: list[tuple[tuple[int, int], int]] = [
            (delta, _MASK_FOR_DELTA[delta]) for delta in _DIRECTIONS
        ]
        options.append(((0, 0), 0))
        for (dx, dy), mask in options:
            candidate = (player[0] + dx, player[1] + dy)
            new_player = candidate if candidate in walkable else player
            option_pellets = set(sim_pellets)
            new_player, moved, caught = self._step_sim(
                player,
                list(ghosts),
                option_pellets,
                mask,
                plan_tick,
                walkable,
            )
            if caught:
                continue
            field = _bfs_distances(new_player, walkable)
            nearest: int | None = None
            for ghost in moved:
                distance = field.get(ghost)
                # A ghost walled off from the player is infinitely far away.
                score = distance if distance is not None else 1 << 20
                nearest = score if nearest is None else min(nearest, score)
            survivor_score = nearest if nearest is not None else 1 << 20
            if best_score is None or survivor_score > best_score:
                best_score = survivor_score
                best_mask = mask
        return best_mask


def _downhill_path(
    field: dict[tuple[int, int], int],
    player: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Reconstruct the shortest player-to-goal path from a player-anchored BFS.

    Walks downhill from the goal in the fixed ``_DIRECTIONS`` order — the
    same tie-break the simulator's own ghost stepping uses — then reverses,
    so the result lists the cells the player would enter, in order.
    """

    path: list[tuple[int, int]] = []
    cell = goal
    while cell != player:
        path.append(cell)
        here = field[cell]
        for dx, dy in _DIRECTIONS:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if field.get(neighbor) == here - 1:
                cell = neighbor
                break
        else:  # pragma: no cover - BFS fields always admit a downhill step
            raise RuntimeError("maze chase planner lost the downhill path")
    path.reverse()
    return path


def _parse_keys_doors_frame(
    frame: RgbFrame, *, caller: str
) -> tuple[
    set[tuple[int, int]],
    tuple[int, int],
    tuple[int, int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
]:
    """Parse the canonical keys_doors grid frame into visible cell sets.

    Returns ``(walkable, player, key, door, target)``; the key, door, and
    target cells are ``None`` when not rendered. Every color read here is
    part of the public render contract — ``WALL_RGB`` / ``PLAYER_RGB`` /
    ``KEY_RGB`` / ``DOOR_RGB`` / ``TARGET_RGB`` — so a pixel-only policy sees
    exactly what a model would see and never touches simulator state. The
    world renders the key only while uncollected and the door only while
    closed, so their disappearance across frames is the episodic-memory
    signal, exactly as designed.
    """

    grid = KeysDoorsEnv.GRID_SIZE
    if frame.width != grid or frame.height != grid:
        raise ValueError(f"{caller} requires the canonical grid frame")
    pixels = frame.pixels
    player: tuple[int, int] | None = None
    key: tuple[int, int] | None = None
    door: tuple[int, int] | None = None
    target: tuple[int, int] | None = None
    walkable: set[tuple[int, int]] = set()
    for y in range(grid):
        for x in range(grid):
            offset = (y * grid + x) * 3
            color = (pixels[offset], pixels[offset + 1], pixels[offset + 2])
            if color == KeysDoorsEnv.WALL_RGB:
                continue
            cell = (x, y)
            walkable.add(cell)
            if color == KeysDoorsEnv.PLAYER_RGB:
                player = cell
            elif color == KeysDoorsEnv.KEY_RGB:
                key = cell
            elif color == KeysDoorsEnv.DOOR_RGB:
                door = cell
            elif color == KeysDoorsEnv.TARGET_RGB:
                target = cell
    if player is None:
        raise RuntimeError(f"{caller} could not locate the player")
    return walkable, player, key, door, target


class ScriptedKeysDoorsSolver:
    """Pixel-only key→door→target solver for the keys_doors world.

    The world isolates ordered planning plus one piece of unrendered
    episodic memory: whether the player holds the key is never painted. The
    solver derives that memory the only honest way a model could — the key
    pixel is visible while uncollected and disappears exactly on collection,
    and the door pixel is visible while closed and disappears exactly when
    opened. Once both have been seen, their absence in later frames *is* the
    remembered state; no simulator state is ever read.

    The plan re-derives from fresh pixels every tick:

    1. Key not yet collected (seen before, still visible) → walk the BFS
       shortest path to the key, treating the closed door cell as blocked
       (the world places the key on the player side by construction).
    2. Key collected (seen before, now absent) and door still closed → walk
       to the door cell; stepping into it while holding the key opens it.
    3. Door open (seen before, now absent) → walk the BFS shortest path to
       the target; the open door cell renders as an ordinary corridor.

    There are no movers in this world, so no lookahead is needed; the only
    stochasticity is the target relocation, which is visible in the very
    next frame and absorbed by re-planning. On worlds without key or door
    pixels the solver never activates and holds still, so its matrix rows
    there are honest zeros — the same contract the pellet teacher keeps.
    """

    __slots__ = ("_activated", "_key_seen", "_door_seen")
    identity = "diagnostic.scripted_keys_doors_solver.v1"
    uses_privileged_state = False

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        self._activated = False
        self._key_seen = False
        self._door_seen = False

    def act(self, observation: Observation) -> GenericControl:
        walkable, player, key, door, target = _parse_keys_doors_frame(
            observation.rgb, caller="keys_doors solver"
        )
        if key is not None:
            self._key_seen = True
        if door is not None:
            self._door_seen = True
        self._activated = self._activated or self._key_seen or self._door_seen
        if not self._activated:
            return GenericControl()

        key_held = self._key_seen and key is None
        door_open = self._door_seen and door is None
        if not key_held:
            if key is None:  # pragma: no cover - an uncollected key renders
                return GenericControl()
            goal = key
            open_cells = walkable - ({door} if door is not None else set())
        elif not door_open:
            if door is None:  # pragma: no cover - a closed door renders
                return GenericControl()
            goal = door
            open_cells = walkable
        else:
            if target is None:
                # The player is standing on the just-collected target cell;
                # the relocated target is visible again next tick.
                return GenericControl()
            goal = target
            open_cells = walkable

        field = _bfs_distances(player, frozenset(open_cells))
        if goal not in field:
            return GenericControl()
        path = _downhill_path(field, player, goal)
        delta = (path[0][0] - player[0], path[0][1] - player[1])
        return GenericControl(keys_down=(_KEY_FOR_DELTA[delta],))


def _parse_junction_frame(
    frame: RgbFrame, *, caller: str
) -> tuple[
    set[tuple[int, int]],
    tuple[int, int],
    tuple[int, int] | None,
    set[tuple[int, int]],
    bool,
]:
    """Parse the canonical junction grid frame into visible cell sets.

    Returns ``(walkable, player, target, chasers, saw_walls)``. Every color
    read here is part of the public render contract — ``WALL_RGB`` /
    ``PLAYER_RGB`` / ``TARGET_RGB`` and the visible chaser red — so a
    pixel-only policy sees exactly what a model would see and never touches
    simulator state. ``saw_walls`` lets the caller distinguish the maze
    worlds from the open-field worlds, whose palettes overlap otherwise.
    """

    grid = JunctionEnv.GRID_SIZE
    if frame.width != grid or frame.height != grid:
        raise ValueError(f"{caller} requires the canonical grid frame")
    # The chaser red is a visible pixel color; it is read from the frame,
    # never from simulator state.
    chaser_color = JunctionEnv._CHASER_COLOR
    pixels = frame.pixels
    player: tuple[int, int] | None = None
    target: tuple[int, int] | None = None
    walkable: set[tuple[int, int]] = set()
    chasers: set[tuple[int, int]] = set()
    saw_walls = False
    for y in range(grid):
        for x in range(grid):
            offset = (y * grid + x) * 3
            color = (pixels[offset], pixels[offset + 1], pixels[offset + 2])
            if color == JunctionEnv.WALL_RGB:
                saw_walls = True
                continue
            cell = (x, y)
            walkable.add(cell)
            if color in (
                JunctionEnv.PLAYER_RGB,
                JunctionEnv.PLAYER_CHASER_OVERLAP_RGB,
            ):
                player = cell
            elif color == JunctionEnv.TARGET_RGB:
                target = cell
            elif color == chaser_color:
                chasers.add(cell)
    if player is None:
        raise RuntimeError(f"{caller} could not locate the player")
    return walkable, player, target, chasers, saw_walls


class ScriptedJunctionSolver:
    """Pixel-only chaser-aware solver for the junction world.

    Junction isolates branch-point decisions under pursuit on a perfect
    maze: the corridor path between any two cells is unique, one chaser
    walks the exact BFS shortest path to the player every ``chaser_period``
    ticks, and contact costs a reward plus a hidden-RNG respawn. The solver
    reads nothing but the canonical observation frame — walls, the player,
    the yellow target, and the visible chaser-red cells — and simulates the
    world's published mechanics before committing:

    - It reconstructs the (unique) BFS shortest path to the visible target
      and simulates walking it one cell per tick while every visible chaser
      steps downhill on the post-move player field with the same
      ``_DIRECTIONS`` tie-break and swept-path contact rule the simulator
      uses. It commits to the path's first step only if the whole walk
      survives.
    - If the walk is caught, a one-tick survival fallback picks the move
      (four directions, then stay) that is not caught and maximizes the
      post-move BFS distance to the nearest chaser, ties broken by fixed
      scan order.

    Two structural facts shape the frontier, and the docstring records them
    so the run record can point here: on a perfect maze a chaser sitting on
    the unique player→target path stays on it while chasing, so a guarded
    target cannot be reached without a catch; and because the player moves
    every tick while the chaser moves every ``chaser_period`` ticks, an
    unguarded target can usually be outrun to. The solver never walks into a
    chaser on purpose: when the target is guarded it kites, accepting that
    a cornered catch costs one reward and re-rolls the geometry through
    the respawn.

    Honest pixel-vision caveats, matching the maze planner's: the
    player/chaser overlap pixel briefly hides a chaser standing on the
    player's own cell, and stacked chasers render as one. Re-planning from
    fresh pixels every tick bounds the damage; the post-catch respawn cell
    is sampled from hidden RNG, so the plan past a forced catch is simply
    re-derived from the next frame. On worlds without the wall + target +
    chaser pixel combination the solver never activates and holds still,
    so its matrix rows there are honest zeros.
    """

    __slots__ = ("_chaser_period", "_activated")
    identity = "diagnostic.scripted_junction_solver.v1"
    uses_privileged_state = False

    def __init__(self, *, chaser_period: int = 2) -> None:
        if isinstance(chaser_period, bool) or not isinstance(chaser_period, int):
            raise TypeError("chaser_period must be an integer")
        if chaser_period < 1 or chaser_period > JunctionEnv.MAX_CHASER_PERIOD:
            raise ValueError(
                f"chaser_period must be in [1, {JunctionEnv.MAX_CHASER_PERIOD}]"
            )
        self._chaser_period = chaser_period
        self._activated = False

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        self._activated = False

    def act(self, observation: Observation) -> GenericControl:
        walkable, player, target, chasers, saw_walls = _parse_junction_frame(
            observation.rgb, caller="junction solver"
        )
        if target is not None and chasers and saw_walls:
            # Only the junction world renders walls, a yellow target, and
            # chaser red together: the open-field worlds have no walls, the
            # maze_chase world has pellets instead of a target, and
            # keys_doors has no red.
            self._activated = True
        if not self._activated or target is None:
            return GenericControl()

        frozen_walkable = frozenset(walkable)
        # frame_id counts completed ticks, so the upcoming decision executes
        # on simulator tick ``start_tick`` and chasers move on ticks
        # divisible by the period — the condition the simulator evaluates.
        start_tick = observation.frame_id
        chaser_list = sorted(chasers)

        field = _bfs_distances(player, frozen_walkable)
        if target in field and field[target] > 0:
            path = _downhill_path(field, player, target)
            if not self._path_is_caught(
                player, chaser_list, path, frozen_walkable, start_tick
            ):
                delta = (path[0][0] - player[0], path[0][1] - player[1])
                return GenericControl(keys_down=(_KEY_FOR_DELTA[delta],))
        return self._survival(player, chaser_list, frozen_walkable, start_tick)

    def _step_sim(
        self,
        sim_player: tuple[int, int],
        sim_chasers: list[tuple[int, int]],
        mask: int,
        tick: int,
        walkable: frozenset[tuple[int, int]],
    ) -> tuple[tuple[int, int], list[tuple[int, int]], bool]:
        """Advance one simulated tick; return (player, chasers, caught)."""
        delta = (
            int(bool(mask & 8)) - int(bool(mask & 2)),
            int(bool(mask & 4)) - int(bool(mask & 1)),
        )
        candidate = (sim_player[0] + delta[0], sim_player[1] + delta[1])
        # Walls refuse the whole move; there is no sliding along them.
        new_player = candidate if candidate in walkable else sim_player
        chasers_move = tick % self._chaser_period == 0
        # Every chaser chases the same post-move player cell, so one shared
        # field serves them all — the same field the simulator computes once
        # per move tick.
        field = _bfs_distances(new_player, walkable) if chasers_move else None
        moved: list[tuple[int, int]] = []
        caught = False
        for chaser in sim_chasers:
            after = (
                MazeChaseEnv._step_downhill(chaser, field) if chasers_move else chaser
            )
            if _paths_collide_at_same_time(sim_player, new_player, chaser, after):
                caught = True
            moved.append(after)
        return new_player, moved, caught

    def _path_is_caught(
        self,
        player: tuple[int, int],
        chasers: list[tuple[int, int]],
        path: list[tuple[int, int]],
        walkable: frozenset[tuple[int, int]],
        start_tick: int,
    ) -> bool:
        """Simulate walking ``path`` one cell per tick; True if caught."""
        sim_player = player
        sim_chasers = list(chasers)
        for index, step_cell in enumerate(path):
            mask = _MASK_FOR_DELTA[
                (step_cell[0] - sim_player[0], step_cell[1] - sim_player[1])
            ]
            sim_player, sim_chasers, caught = self._step_sim(
                sim_player, sim_chasers, mask, start_tick + index, walkable
            )
            if caught:
                return True
        return False

    def _survival(
        self,
        player: tuple[int, int],
        chasers: list[tuple[int, int]],
        walkable: frozenset[tuple[int, int]],
        start_tick: int,
    ) -> GenericControl:
        """One-tick fallback: survive, then maximize distance to the chasers.

        Options scan in the fixed ``_DIRECTIONS`` order with stay last; a
        strict ``>`` comparison keeps the earliest option on ties, so the
        fallback is deterministic.
        """
        if not chasers:
            return GenericControl()
        best_score: int | None = None
        best_delta: tuple[int, int] = (0, 0)
        options: list[tuple[int, int]] = list(_DIRECTIONS) + [(0, 0)]
        for dx, dy in options:
            mask = _MASK_FOR_DELTA.get((dx, dy), 0)
            new_player, moved, caught = self._step_sim(
                player, list(chasers), mask, start_tick, walkable
            )
            if caught:
                continue
            field = _bfs_distances(new_player, walkable)
            nearest: int | None = None
            for chaser in moved:
                distance = field.get(chaser)
                # A chaser walled off from the player is infinitely far away.
                score = distance if distance is not None else 1 << 20
                nearest = score if nearest is None else min(nearest, score)
            survivor_score = nearest if nearest is not None else 1 << 20
            if best_score is None or survivor_score > best_score:
                best_score = survivor_score
                best_delta = (dx, dy)
        if best_delta == (0, 0):
            return GenericControl()
        return GenericControl(keys_down=(_KEY_FOR_DELTA[best_delta],))


def default_diagnostic_policies() -> tuple[object, ...]:
    """Return the §28 diagnostic policies in canonical suite order."""

    return (
        NoOpPolicy(),
        RandomMovementPolicy(),
        ScriptedTargetChasePolicy(),
        ScriptedPelletTeacherPolicy(),
        OraclePolicy(),
    )


def evaluate_diagnostic_policy_suite(
    policies: Sequence[object],
    *,
    config: ClosedLoopPlayConfig,
) -> tuple[ClosedLoopPlayReport, ...]:
    """Run each policy over every registered seed and return one report each.

    Reports share the exact ``ClosedLoopPlayReport`` schema the model
    evaluator emits, so diagnostic rows and model rows compare column for
    column. The oracle row is diagnosis-only evidence per PLAN.md §28.15.
    """

    if not isinstance(config, ClosedLoopPlayConfig):
        raise ValueError("config must be a ClosedLoopPlayConfig")
    if not policies:
        raise ValueError("policies cannot be empty")
    identities = [getattr(policy, "identity", None) for policy in policies]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("every policy must define a non-empty identity")
    if len(set(identities)) != len(identities):
        raise ValueError("policy identities must be unique")
    reports = []
    for policy, identity in zip(policies, identities):
        episodes = tuple(
            run_policy_closed_loop_episode(policy, seed=seed, config=config)
            for seed in config.episode_seeds
        )
        reports.append(
            ClosedLoopPlayReport(
                schema_version=1,
                config=config,
                model_description=f"diagnostic policy {identity}",
                episodes=episodes,
            )
        )
    return tuple(reports)


__all__ = [
    "NoOpPolicy",
    "OraclePolicy",
    "RandomMovementPolicy",
    "ScriptedJunctionSolver",
    "ScriptedKeysDoorsSolver",
    "ScriptedPelletTeacherPolicy",
    "ScriptedTargetChasePolicy",
    "default_diagnostic_policies",
    "evaluate_diagnostic_policy_suite",
]
