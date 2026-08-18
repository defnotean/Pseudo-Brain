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

from ..environments.junction import _DIRECTIONS, _bfs_distances
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
    simulates walking each path end to end (capped at ``horizon`` ticks), and
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

    The simulation assumes the canonical matrix slot's published mechanics
    (direct pursuit, one shared player-anchored ghost field, fixed period).
    On other ghost rules or speed curves the lookahead degrades to an
    optimistic direct-pursuit guess; the planner never reads which rule the
    simulator is actually running.
    """

    __slots__ = ("_ghost_period", "_candidate_pellets", "_horizon")
    identity = "diagnostic.scripted_maze_chase_planner.v1"
    uses_privileged_state = False

    def __init__(
        self,
        *,
        ghost_period: int = 2,
        candidate_pellets: int = 6,
        horizon: int = 24,
    ) -> None:
        for value, name, low, high in (
            (ghost_period, "ghost_period", 1, 64),
            (candidate_pellets, "candidate_pellets", 1, 32),
            (horizon, "horizon", 1, 256),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < low or value > high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        self._ghost_period = ghost_period
        self._candidate_pellets = candidate_pellets
        self._horizon = horizon

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")

    def act(self, observation: Observation) -> GenericControl:
        walkable, pellets, player, ghosts = _parse_maze_chase_frame(
            observation.rgb, caller="maze chase planner"
        )
        if not pellets:
            return GenericControl()
        frozen_walkable = frozenset(walkable)
        # frame_id counts completed ticks, so the upcoming decision executes
        # on simulator tick ``start_tick`` and ghosts move on ticks divisible
        # by the period — the same condition the simulator evaluates.
        start_tick = observation.frame_id
        from_player = _bfs_distances(player, frozen_walkable)

        ranked = sorted(
            (from_player[cell], cell[1], cell[0])
            for cell in pellets
            if cell in from_player and from_player[cell] > 0
        )[: self._candidate_pellets]
        for distance, goal_y, goal_x in ranked:
            if distance > self._horizon:
                continue
            path = _downhill_path(from_player, player, (goal_x, goal_y))
            if not self._path_is_caught(
                player, ghosts, path, frozen_walkable, start_tick
            ):
                delta = (path[0][0] - player[0], path[0][1] - player[1])
                return GenericControl(keys_down=(_KEY_FOR_DELTA[delta],))
        return self._survival_move(player, ghosts, frozen_walkable, start_tick)

    def _path_is_caught(
        self,
        player: tuple[int, int],
        ghosts: set[tuple[int, int]],
        path: list[tuple[int, int]],
        walkable: frozenset[tuple[int, int]],
        start_tick: int,
    ) -> bool:
        """Simulate walking ``path``; return True if the player is caught."""
        sim_player = player
        sim_ghosts = sorted(ghosts)
        for index, step_cell in enumerate(path):
            ghosts_move = (start_tick + index) % self._ghost_period == 0
            # Every direct-pursuit ghost chases the same post-move player
            # cell, so one shared field serves them all — the same field the
            # simulator computes once per move tick.
            field = (
                _bfs_distances(step_cell, walkable) if ghosts_move else None
            )
            moved: list[tuple[int, int]] = []
            for ghost in sim_ghosts:
                after = (
                    MazeChaseEnv._step_downhill(ghost, field)
                    if ghosts_move
                    else ghost
                )
                if _paths_collide_at_same_time(sim_player, step_cell, ghost, after):
                    return True
                moved.append(after)
            sim_player = step_cell
            sim_ghosts = moved
        return False

    def _survival_move(
        self,
        player: tuple[int, int],
        ghosts: set[tuple[int, int]],
        walkable: frozenset[tuple[int, int]],
        start_tick: int,
    ) -> GenericControl:
        """One-tick fallback: survive, then maximize distance to the ghosts.

        Options scan in the fixed ``_DIRECTIONS`` order with stay last; a
        strict ``>`` comparison keeps the earliest option on ties, so the
        fallback is deterministic. A wall-refused press leaves the player in
        place, exactly like the simulator.
        """
        ghosts_move = start_tick % self._ghost_period == 0
        ordered_ghosts = sorted(ghosts)
        best_score: int | None = None
        best_key: int | None = None
        options: list[tuple[tuple[int, int], int | None]] = [
            (delta, _KEY_FOR_DELTA[delta]) for delta in _DIRECTIONS
        ]
        options.append(((0, 0), None))
        for (dx, dy), key in options:
            candidate = (player[0] + dx, player[1] + dy)
            new_player = candidate if candidate in walkable else player
            field = _bfs_distances(new_player, walkable)
            caught = False
            nearest: int | None = None
            for ghost in ordered_ghosts:
                after = (
                    MazeChaseEnv._step_downhill(ghost, field)
                    if ghosts_move
                    else ghost
                )
                if _paths_collide_at_same_time(player, new_player, ghost, after):
                    caught = True
                    break
                distance = field.get(after)
                # A ghost walled off from the player is infinitely far away.
                score = distance if distance is not None else 1 << 20
                nearest = score if nearest is None else min(nearest, score)
            if caught:
                continue
            survivor_score = nearest if nearest is not None else 1 << 20
            if best_score is None or survivor_score > best_score:
                best_score = survivor_score
                best_key = key
        if best_key is None:
            return GenericControl()
        return GenericControl(keys_down=(best_key,))


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
    "ScriptedPelletTeacherPolicy",
    "ScriptedTargetChasePolicy",
    "default_diagnostic_policies",
    "evaluate_diagnostic_policy_suite",
]
