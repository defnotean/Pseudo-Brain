"""Lazy deterministic multi-scenario curriculum supervision for sensorimotor recovery.

This module implements curriculum datasets for the maze_chase ladder world across
five canonical sensorimotor recovery scenarios:
1. Scenario A (Wall Collisions & Unsticking): Places the agent against walls, corners,
   or dead-ends with wall-facing orientation, teaching 90° and 180° turn-aways.
2. Scenario B (Junction Decisions): Places the agent at crossroads and branching
   intersections to teach path selection toward open pellets.
3. Scenario C (Hazard Evasion): Spawns the agent in close proximity (2-3 steps) to
   approaching ghosts to teach vector repulsion and evasive maneuvers.
4. Scenario D (Motor Babbling / Control Discovery): Generates exploratory keypress
   sequences (10-frame snippets) for sensorimotor contingency discovery and forward
   model self-identification.
5. Scenario E (Standard Maze Navigation): Standard full-maze rollouts under the
   expert lookahead planner.

The dataset can generate homogeneous single-scenario batches or balanced round-robin
mixed batches across all 5 scenarios.

Split namespaces use disjoint 62-bit tags via :func:`split_episode_seed`.
Transitions and sequences strictly conform to :class:`MovingShapesTransition` and
:class:`MovingShapesSequence` with canonical :class:`ModelObservation` inputs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from json import dumps
from math import gcd
from struct import pack
from typing import Any, overload

from ..environments.junction import _DIRECTIONS, _bfs_distances
from ..environments.maze_chase import (
    _GHOST_RULES,
    MazeChaseEnv,
    _carve_maze_with_loops,
)
from ..environments.moving_shapes import _SplitMix64
from ..types import GenericControl, HidKey, ModelObservation, Observation
from .moving_shapes_dataset import (
    DatasetSplit,
    MovingShapesSequence,
    MovingShapesTransition,
    _TransitionWithoutValue,
    _integer,
    _number,
    _split,
    split_episode_seed,
    split_for_episode_seed,
)

_SEED_NAMESPACE_SIZE = 1 << 62
_DATASET_SCHEMA_VERSION = 1
_GENERATOR_ID_PREFIX = "irene.curriculum_recovery"
_LICENSE_RECORD_ID = "original-project-content"

_KEY_W = int(HidKey.W)
_KEY_A = int(HidKey.A)
_KEY_S = int(HidKey.S)
_KEY_D = int(HidKey.D)

_KEY_FOR_DELTA = {(0, -1): _KEY_W, (-1, 0): _KEY_A, (0, 1): _KEY_S, (1, 0): _KEY_D}
_MASK_FOR_DELTA = {(0, -1): 1, (-1, 0): 2, (0, 1): 4, (1, 0): 8}

# Aliases matching the dataset family pattern
CurriculumTransition = MovingShapesTransition
CurriculumSequence = MovingShapesSequence


class CurriculumScenario(str, Enum):
    """The canonical curriculum recovery scenarios."""

    WALL_UNSTICKING = "wall_unsticking"
    JUNCTION_DECISIONS = "junction_decisions"
    HAZARD_EVASION = "hazard_evasion"
    MOTOR_BABBLING = "motor_babbling"
    STANDARD_NAVIGATION = "standard_navigation"
    MIXED = "mixed"


CURRICULUM_SCENARIOS = (
    CurriculumScenario.WALL_UNSTICKING,
    CurriculumScenario.JUNCTION_DECISIONS,
    CurriculumScenario.HAZARD_EVASION,
    CurriculumScenario.MOTOR_BABBLING,
    CurriculumScenario.STANDARD_NAVIGATION,
)

_SCENARIO_ALIASES: dict[str, CurriculumScenario] = {
    "wall_unsticking": CurriculumScenario.WALL_UNSTICKING,
    "scenario_a": CurriculumScenario.WALL_UNSTICKING,
    "unsticking": CurriculumScenario.WALL_UNSTICKING,
    "wall": CurriculumScenario.WALL_UNSTICKING,
    "walls": CurriculumScenario.WALL_UNSTICKING,
    "junction_decisions": CurriculumScenario.JUNCTION_DECISIONS,
    "scenario_b": CurriculumScenario.JUNCTION_DECISIONS,
    "junction": CurriculumScenario.JUNCTION_DECISIONS,
    "junctions": CurriculumScenario.JUNCTION_DECISIONS,
    "hazard_evasion": CurriculumScenario.HAZARD_EVASION,
    "scenario_c": CurriculumScenario.HAZARD_EVASION,
    "evasion": CurriculumScenario.HAZARD_EVASION,
    "hazard": CurriculumScenario.HAZARD_EVASION,
    "ghost_evasion": CurriculumScenario.HAZARD_EVASION,
    "motor_babbling": CurriculumScenario.MOTOR_BABBLING,
    "scenario_d": CurriculumScenario.MOTOR_BABBLING,
    "babbling": CurriculumScenario.MOTOR_BABBLING,
    "control_discovery": CurriculumScenario.MOTOR_BABBLING,
    "standard_navigation": CurriculumScenario.STANDARD_NAVIGATION,
    "scenario_e": CurriculumScenario.STANDARD_NAVIGATION,
    "standard": CurriculumScenario.STANDARD_NAVIGATION,
    "navigation": CurriculumScenario.STANDARD_NAVIGATION,
    "maze_navigation": CurriculumScenario.STANDARD_NAVIGATION,
    "mixed": CurriculumScenario.MIXED,
    "all": CurriculumScenario.MIXED,
    "balanced": CurriculumScenario.MIXED,
}


def _scenario(value: CurriculumScenario | str) -> CurriculumScenario:
    if isinstance(value, CurriculumScenario):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"scenario must be a CurriculumScenario or string, got {type(value).__name__}"
        )
    normalized = value.lower().strip()
    if normalized in _SCENARIO_ALIASES:
        return _SCENARIO_ALIASES[normalized]
    raise ValueError(
        f"scenario '{value}' is unknown; must be one of {sorted(set(_SCENARIO_ALIASES))}"
    )


@dataclass(frozen=True, slots=True)
class CurriculumDatasetConfig:
    """Identity and generation bounds for a lazy multi-scenario curriculum dataset.

    Defaults describe 8,192 sequences of 32 transitions each (262,144 transitions)
    balanced across all 5 recovery scenarios.
    """

    split: DatasetSplit = DatasetSplit.TRAIN
    scenario: CurriculumScenario = CurriculumScenario.MIXED
    sequence_count: int = 8_192
    sequence_length: int = 32
    seed_offset: int = 0
    ghost_count: int = 3
    ghost_period: int = 2
    player_period: int = 1
    extra_loops: int = 16
    ghost_rule: str = "direct"
    ghost_elroy: bool = False
    input_delay_ticks: int = 0
    sticky_direction: bool = False
    tick_period_ns: int = MazeChaseEnv.DEFAULT_TICK_PERIOD_NS
    discount: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", _split(self.split))
        object.__setattr__(self, "scenario", _scenario(self.scenario))
        count = _integer(
            self.sequence_count,
            name="sequence_count",
            minimum=1,
            maximum=_SEED_NAMESPACE_SIZE,
        )
        length = _integer(
            self.sequence_length,
            name="sequence_length",
            minimum=1,
            maximum=0xFFFFFFFF,
        )
        offset = _integer(
            self.seed_offset,
            name="seed_offset",
            maximum=_SEED_NAMESPACE_SIZE - 1,
        )
        if offset + count > _SEED_NAMESPACE_SIZE:
            raise ValueError("seed_offset + sequence_count exceeds the split namespace")
        _integer(self.ghost_count, name="ghost_count", minimum=1, maximum=8)
        _integer(
            self.ghost_period,
            name="ghost_period",
            minimum=1,
            maximum=MazeChaseEnv.MAX_GHOST_PERIOD,
        )
        _integer(
            self.player_period,
            name="player_period",
            minimum=1,
            maximum=MazeChaseEnv.MAX_PLAYER_PERIOD,
        )
        _integer(
            self.extra_loops,
            name="extra_loops",
            minimum=0,
            maximum=MazeChaseEnv.MAX_EXTRA_LOOPS,
        )
        if not isinstance(self.ghost_rule, str) or self.ghost_rule not in _GHOST_RULES:
            raise ValueError(f"ghost_rule must be one of {sorted(_GHOST_RULES)}")
        if not isinstance(self.ghost_elroy, bool):
            raise ValueError("ghost_elroy must be a boolean")
        _integer(
            self.input_delay_ticks,
            name="input_delay_ticks",
            minimum=0,
            maximum=MazeChaseEnv.MAX_INPUT_DELAY,
        )
        if not isinstance(self.sticky_direction, bool):
            raise ValueError("sticky_direction must be a boolean")
        _integer(self.tick_period_ns, name="tick_period_ns", minimum=1)
        discount = _number(self.discount, name="discount")
        if discount < 0.0 or discount > 1.0:
            raise ValueError("discount must be in [0, 1]")

        object.__setattr__(self, "sequence_count", count)
        object.__setattr__(self, "sequence_length", length)
        object.__setattr__(self, "seed_offset", offset)
        object.__setattr__(self, "discount", discount)

    @property
    def generator_id(self) -> str:
        return f"{_GENERATOR_ID_PREFIX}.{self.scenario.value}.v1"

    @property
    def total_transitions(self) -> int:
        return self.sequence_count * self.sequence_length

    def manifest_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe identity of this virtual dataset."""
        return {
            "schema_version": _DATASET_SCHEMA_VERSION,
            "generator_id": self.generator_id,
            "origin": "in-repository deterministic procedural environment",
            "license_record_id": _LICENSE_RECORD_ID,
            "environment_family": "maze_chase",
            "dataset_family": "curriculum_recovery",
            "scenario": self.scenario.value,
            "split": self.split.value,
            "seed_namespace_tag": {
                DatasetSplit.TRAIN: 0,
                DatasetSplit.VALIDATION: 1,
                DatasetSplit.TEST: 2,
            }[self.split],
            "seed_offset": self.seed_offset,
            "sequence_count": self.sequence_count,
            "sequence_length": self.sequence_length,
            "total_transitions": self.total_transitions,
            "ghost_count": self.ghost_count,
            "ghost_period": self.ghost_period,
            "player_period": self.player_period,
            "extra_loops": self.extra_loops,
            "ghost_rule": self.ghost_rule,
            "ghost_elroy": self.ghost_elroy,
            "input_delay_ticks": self.input_delay_ticks,
            "sticky_direction": self.sticky_direction,
            "tick_period_ns": self.tick_period_ns,
            "discount_hex": self.discount.hex(),
            "teacher_identity": "diagnostic.scripted_maze_chase_planner.v1",
            "teacher_inputs": "visible_rgb_only",
            "world_target": "next_model_observation",
            "value_target": "zero_bootstrapped_discounted_sequence_return",
        }


def curriculum_dataset_manifest_sha256(config: CurriculumDatasetConfig) -> str:
    """Hash a configuration using a process-independent canonical manifest."""
    if not isinstance(config, CurriculumDatasetConfig):
        raise ValueError("config must be a CurriculumDatasetConfig")
    encoded = dumps(
        config.manifest_dict(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRCURRICULUM\x01" + encoded).hexdigest()


def _relocate_player_and_ghosts(
    env: MazeChaseEnv,
    *,
    player_pos: tuple[int, int],
    player_facing: tuple[int, int] = (0, -1),
    previous_key_mask: int = 0,
    ghost_positions: Sequence[tuple[int, int]] | None = None,
) -> Observation:
    """Relocate player and ghosts within the carved maze and update pellets/facing."""
    maze = env._maze
    if player_pos not in maze:
        raise ValueError(f"player position {player_pos} is not in maze")

    # Restore pellet at spawn (1, 1) if (1, 1) is in maze and player is elsewhere
    if (1, 1) in maze and (1, 1) != player_pos:
        byte, bit = env._pellet_index(1, 1)
        if not (env._pellets[byte] & (1 << bit)):
            env._pellets[byte] |= 1 << bit
            env._pellets_remaining += 1

    # Remove pellet at player's new position
    byte, bit = env._pellet_index(player_pos[0], player_pos[1])
    if env._pellets[byte] & (1 << bit):
        env._pellets[byte] &= ~(1 << bit) & 0xFF
        env._pellets_remaining -= 1

    env._player_x, env._player_y = player_pos
    env._player_dx, env._player_dy = player_facing
    env._previous_key_mask = previous_key_mask
    if ghost_positions is not None:
        for g in ghost_positions:
            if g not in maze:
                raise ValueError(f"ghost position {g} is not in maze")
        env._ghosts = list(ghost_positions)

    return env.current_observation


def _setup_wall_unsticking(
    env: MazeChaseEnv,
    episode_seed: int,
    config: CurriculumDatasetConfig,
) -> Observation:
    """Scenario A: Place player against walls/corners, facing wall to prompt unsticking."""
    maze = env._maze
    rng = _SplitMix64(episode_seed ^ 0x57414C4C)  # "WALL" domain separator

    dead_ends: list[tuple[tuple[int, int], tuple[int, int]]] = []
    corners: list[tuple[tuple[int, int], tuple[int, int]]] = []
    straights: list[tuple[tuple[int, int], tuple[int, int]]] = []
    all_wall_cells: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for y in range(1, env.GRID_SIZE - 1):
        for x in range(1, env.GRID_SIZE - 1):
            cell = (x, y)
            if cell not in maze:
                continue
            open_neighbors = [
                (x + dx, y + dy) for dx, dy in _DIRECTIONS if (x + dx, y + dy) in maze
            ]
            wall_dirs = [
                (dx, dy) for dx, dy in _DIRECTIONS if (x + dx, y + dy) not in maze
            ]
            if not wall_dirs or not open_neighbors:
                continue
            for wdir in wall_dirs:
                entry = (cell, wdir)
                all_wall_cells.append(entry)
                if len(open_neighbors) == 1:
                    dead_ends.append(entry)
                elif len(open_neighbors) == 2:
                    n1, n2 = open_neighbors[0], open_neighbors[1]
                    if n1[0] != n2[0] and n1[1] != n2[1]:
                        corners.append(entry)
                    else:
                        straights.append(entry)

    dead_ends.sort(key=lambda item: (item[0][1], item[0][0], item[1][1], item[1][0]))
    corners.sort(key=lambda item: (item[0][1], item[0][0], item[1][1], item[1][0]))
    straights.sort(key=lambda item: (item[0][1], item[0][0], item[1][1], item[1][0]))
    all_wall_cells.sort(key=lambda item: (item[0][1], item[0][0], item[1][1], item[1][0]))

    # Prioritize dead-ends, then corners, then straights
    pool = dead_ends if dead_ends else (corners if corners else (straights if straights else all_wall_cells))
    chosen_cell, wall_dir = pool[rng.randbelow(len(pool))]

    wall_mask = _MASK_FOR_DELTA.get(wall_dir, 0)

    # Place ghosts far away (distance >= 6) so unsticking happens cleanly
    distances = _bfs_distances(chosen_cell, maze)
    ghost_candidates = sorted(
        (cell for cell, dist in distances.items() if dist >= 6 and cell != chosen_cell),
        key=lambda cell: (cell[1], cell[0]),
    )
    if len(ghost_candidates) < config.ghost_count:
        ghost_candidates = sorted(
            (cell for cell, dist in distances.items() if dist >= 4 and cell != chosen_cell),
            key=lambda cell: (cell[1], cell[0]),
        )
    ghosts: list[tuple[int, int]] = []
    available = list(ghost_candidates)
    for _ in range(config.ghost_count):
        if available:
            g = available[rng.randbelow(len(available))]
            available.remove(g)
            ghosts.append(g)
        else:
            ghosts.append(env._sample_empty_cell({chosen_cell, *ghosts}))

    return _relocate_player_and_ghosts(
        env,
        player_pos=chosen_cell,
        player_facing=wall_dir,
        previous_key_mask=wall_mask,
        ghost_positions=ghosts,
    )


def _setup_junction_decisions(
    env: MazeChaseEnv,
    episode_seed: int,
    config: CurriculumDatasetConfig,
) -> Observation:
    """Scenario B: Place player at crossroads/intersections (cells with >= 3 open neighbors)."""
    maze = env._maze
    rng = _SplitMix64(episode_seed ^ 0x4A554E43)  # "JUNC" domain separator

    junctions: list[tuple[int, int]] = []
    for y in range(1, env.GRID_SIZE - 1):
        for x in range(1, env.GRID_SIZE - 1):
            cell = (x, y)
            if cell not in maze:
                continue
            open_neighbors = [
                (x + dx, y + dy) for dx, dy in _DIRECTIONS if (x + dx, y + dy) in maze
            ]
            if len(open_neighbors) >= 3:
                junctions.append(cell)

    if not junctions:
        # Fallback to any corridor cell with >= 2 neighbors
        junctions = sorted(
            (
                (x, y)
                for y in range(1, env.GRID_SIZE - 1)
                for x in range(1, env.GRID_SIZE - 1)
                if (x, y) in maze
                and len(
                    [
                        (x + dx, y + dy)
                        for dx, dy in _DIRECTIONS
                        if (x + dx, y + dy) in maze
                    ]
                )
                >= 2
            ),
            key=lambda cell: (cell[1], cell[0]),
        )
    else:
        junctions.sort(key=lambda cell: (cell[1], cell[0]))

    chosen_junction = junctions[rng.randbelow(len(junctions))]

    # Ghosts at far distance (>= 6)
    distances = _bfs_distances(chosen_junction, maze)
    ghost_candidates = sorted(
        (cell for cell, dist in distances.items() if dist >= 6 and cell != chosen_junction),
        key=lambda cell: (cell[1], cell[0]),
    )
    if len(ghost_candidates) < config.ghost_count:
        ghost_candidates = sorted(
            (cell for cell, dist in distances.items() if dist >= 4 and cell != chosen_junction),
            key=lambda cell: (cell[1], cell[0]),
        )
    ghosts: list[tuple[int, int]] = []
    available = list(ghost_candidates)
    for _ in range(config.ghost_count):
        if available:
            g = available[rng.randbelow(len(available))]
            available.remove(g)
            ghosts.append(g)
        else:
            ghosts.append(env._sample_empty_cell({chosen_junction, *ghosts}))

    return _relocate_player_and_ghosts(
        env,
        player_pos=chosen_junction,
        player_facing=(0, -1),
        previous_key_mask=0,
        ghost_positions=ghosts,
    )


def _setup_hazard_evasion(
    env: MazeChaseEnv,
    episode_seed: int,
    config: CurriculumDatasetConfig,
) -> Observation:
    """Scenario C: Place player near moving ghosts (distance 2-3) with valid escape path."""
    maze = env._maze
    rng = _SplitMix64(episode_seed ^ 0x45564153)  # "EVAS" domain separator

    corridor_cells = sorted(maze, key=lambda c: (c[1], c[0]))
    evasion_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for pcell in corridor_cells:
        distances = _bfs_distances(pcell, maze)
        open_neighbors = [
            (pcell[0] + dx, pcell[1] + dy)
            for dx, dy in _DIRECTIONS
            if (pcell[0] + dx, pcell[1] + dy) in maze
        ]
        for gcell, dist in distances.items():
            if 2 <= dist <= 3:
                g_distances = _bfs_distances(gcell, maze)
                # Player has at least one neighbor that doesn't step into the ghost
                has_escape = any(
                    g_distances.get(neighbor, 0) >= dist for neighbor in open_neighbors
                )
                if has_escape:
                    evasion_pairs.append((pcell, gcell))

    if not evasion_pairs:
        # Fallback: pairs with distance in [2, 4]
        for pcell in corridor_cells:
            distances = _bfs_distances(pcell, maze)
            for gcell, dist in distances.items():
                if 2 <= dist <= 4:
                    evasion_pairs.append((pcell, gcell))

    evasion_pairs.sort(key=lambda pair: (pair[0][1], pair[0][0], pair[1][1], pair[1][0]))
    player_cell, primary_ghost = evasion_pairs[rng.randbelow(len(evasion_pairs))]

    # Remaining ghosts placed further away (>= 6)
    distances = _bfs_distances(player_cell, maze)
    ghost_candidates = sorted(
        (
            cell
            for cell, dist in distances.items()
            if dist >= 6 and cell not in (player_cell, primary_ghost)
        ),
        key=lambda cell: (cell[1], cell[0]),
    )
    ghosts: list[tuple[int, int]] = [primary_ghost]
    available = list(ghost_candidates)
    for _ in range(config.ghost_count - 1):
        if available:
            g = available[rng.randbelow(len(available))]
            available.remove(g)
            ghosts.append(g)
        else:
            ghosts.append(env._sample_empty_cell({player_cell, *ghosts}))

    return _relocate_player_and_ghosts(
        env,
        player_pos=player_cell,
        player_facing=(0, -1),
        previous_key_mask=0,
        ghost_positions=ghosts,
    )


def _setup_motor_babbling(
    env: MazeChaseEnv,
    episode_seed: int,
    config: CurriculumDatasetConfig,
) -> tuple[Observation, _MotorBabblingPolicy]:
    """Scenario D: Setup motor babbling with exploratory keypresses for control discovery."""
    maze = env._maze
    rng = _SplitMix64(episode_seed ^ 0x42414242)  # "BABB" domain separator

    corridor_cells = sorted(maze, key=lambda c: (c[1], c[0]))
    player_cell = corridor_cells[rng.randbelow(len(corridor_cells))]

    # Ghosts placed far away (>= 8) so babbling is unobstructed
    distances = _bfs_distances(player_cell, maze)
    ghost_candidates = sorted(
        (cell for cell, dist in distances.items() if dist >= 8 and cell != player_cell),
        key=lambda cell: (cell[1], cell[0]),
    )
    if len(ghost_candidates) < config.ghost_count:
        ghost_candidates = sorted(
            (cell for cell, dist in distances.items() if dist >= 5 and cell != player_cell),
            key=lambda cell: (cell[1], cell[0]),
        )
    ghosts: list[tuple[int, int]] = []
    available = list(ghost_candidates)
    for _ in range(config.ghost_count):
        if available:
            g = available[rng.randbelow(len(available))]
            available.remove(g)
            ghosts.append(g)
        else:
            ghosts.append(env._sample_empty_cell({player_cell, *ghosts}))

    obs = _relocate_player_and_ghosts(
        env,
        player_pos=player_cell,
        player_facing=(0, -1),
        previous_key_mask=0,
        ghost_positions=ghosts,
    )
    policy = _MotorBabblingPolicy(rng)
    return obs, policy


class _MotorBabblingPolicy:
    """Exploratory policy generating single-key presses for sensorimotor discovery."""

    __slots__ = ("_rng",)

    def __init__(self, rng: _SplitMix64) -> None:
        self._rng = rng

    def act(self, observation: Observation) -> GenericControl:
        keys = (_KEY_W, _KEY_A, _KEY_S, _KEY_D)
        # Sample one of the 4 directional keys (or occasionally no-op)
        choice = self._rng.randbelow(5)
        if choice < 4:
            return GenericControl(keys_down=(keys[choice],))
        return GenericControl()


class CurriculumDataset(Sequence[CurriculumSequence]):
    """Random-access, on-the-fly multi-scenario dataset with deterministic epochs."""

    __slots__ = ("config", "manifest_sha256")

    def __init__(self, config: CurriculumDatasetConfig) -> None:
        if not isinstance(config, CurriculumDatasetConfig):
            raise ValueError("config must be a CurriculumDatasetConfig")
        self.config = config
        self.manifest_sha256 = curriculum_dataset_manifest_sha256(config)

    def __len__(self) -> int:
        return self.config.sequence_count

    def scenario_for_index(self, sequence_index: int) -> CurriculumScenario:
        """Return the active scenario for a specific sequence index."""
        if self.config.scenario != CurriculumScenario.MIXED:
            return self.config.scenario
        return CURRICULUM_SCENARIOS[sequence_index % len(CURRICULUM_SCENARIOS)]

    @overload
    def __getitem__(self, index: int) -> CurriculumSequence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CurriculumSequence, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> CurriculumSequence | tuple[CurriculumSequence, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer or slice")
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("dataset index out of range")
        return self._generate(normalized)

    def _generate(self, sequence_index: int) -> CurriculumSequence:
        from ..evaluation.diagnostic_policies import (
            ScriptedMazeChasePlannerPolicy,
        )

        config = self.config
        scenario = self.scenario_for_index(sequence_index)
        episode_seed = split_episode_seed(
            config.split,
            config.seed_offset + sequence_index,
        )

        rollout_ticks = config.sequence_length

        environment = MazeChaseEnv(
            ghost_count=config.ghost_count,
            ghost_period=config.ghost_period,
            player_period=config.player_period,
            extra_loops=config.extra_loops,
            ghost_rule=config.ghost_rule,
            ghost_elroy=config.ghost_elroy,
            input_delay_ticks=config.input_delay_ticks,
            sticky_direction=config.sticky_direction,
            tick_period_ns=config.tick_period_ns,
            max_ticks=rollout_ticks + 10,
        )
        environment.reset(episode_seed)

        babbling_policy: _MotorBabblingPolicy | None = None
        teacher: ScriptedMazeChasePlannerPolicy | None = None

        if scenario == CurriculumScenario.WALL_UNSTICKING:
            raw_observation = _setup_wall_unsticking(environment, episode_seed, config)
            teacher = ScriptedMazeChasePlannerPolicy(
                ghost_period=config.ghost_period,
                input_delay_ticks=config.input_delay_ticks,
                player_period=config.player_period,
                ghost_elroy=config.ghost_elroy,
            )
            teacher.reset(episode_seed)
        elif scenario == CurriculumScenario.JUNCTION_DECISIONS:
            raw_observation = _setup_junction_decisions(environment, episode_seed, config)
            teacher = ScriptedMazeChasePlannerPolicy(
                ghost_period=config.ghost_period,
                input_delay_ticks=config.input_delay_ticks,
                player_period=config.player_period,
                ghost_elroy=config.ghost_elroy,
            )
            teacher.reset(episode_seed)
        elif scenario == CurriculumScenario.HAZARD_EVASION:
            raw_observation = _setup_hazard_evasion(environment, episode_seed, config)
            teacher = ScriptedMazeChasePlannerPolicy(
                ghost_period=config.ghost_period,
                input_delay_ticks=config.input_delay_ticks,
                player_period=config.player_period,
                ghost_elroy=config.ghost_elroy,
            )
            teacher.reset(episode_seed)
        elif scenario == CurriculumScenario.MOTOR_BABBLING:
            raw_observation, babbling_policy = _setup_motor_babbling(
                environment, episode_seed, config
            )
        elif scenario == CurriculumScenario.STANDARD_NAVIGATION:
            raw_observation = environment.current_observation
            teacher = ScriptedMazeChasePlannerPolicy(
                ghost_period=config.ghost_period,
                input_delay_ticks=config.input_delay_ticks,
                player_period=config.player_period,
                ghost_elroy=config.ghost_elroy,
            )
            teacher.reset(episode_seed)
        else:
            raise RuntimeError(f"unhandled scenario: {scenario}")

        raw_transitions: list[_TransitionWithoutValue] = []

        for _ in range(rollout_ticks):
            observation = raw_observation.to_model_observation(observation_age_ns=0)
            if babbling_policy is not None:
                action_target = babbling_policy.act(raw_observation)
            elif teacher is not None:
                action_target = teacher.act(raw_observation)
            else:
                action_target = GenericControl()

            outcome = environment.step(action_target)
            next_observation = outcome.observation.to_model_observation(
                observation_age_ns=0
            )
            raw_transitions.append(
                _TransitionWithoutValue(
                    observation=observation,
                    applied_control=outcome.applied_control,
                    action_target=action_target,
                    next_observation_target=next_observation,
                    reward_target=outcome.reward,
                    event_targets=outcome.events,
                    terminated_target=outcome.terminated,
                    truncated_target=outcome.truncated,
                )
            )
            raw_observation = outcome.observation
            if outcome.terminated:
                break

        if raw_transitions:
            last = raw_transitions[-1]
            if not last.terminated_target and not last.truncated_target:
                raw_transitions[-1] = _TransitionWithoutValue(
                    observation=last.observation,
                    applied_control=last.applied_control,
                    action_target=last.action_target,
                    next_observation_target=last.next_observation_target,
                    reward_target=last.reward_target,
                    event_targets=last.event_targets,
                    terminated_target=False,
                    truncated_target=True,
                )

        running_return = 0.0
        reversed_transitions: list[CurriculumTransition] = []
        for raw in reversed(raw_transitions):
            done = raw.terminated_target or raw.truncated_target
            running_return = raw.reward_target + (
                0.0 if done else config.discount * running_return
            )
            reversed_transitions.append(
                CurriculumTransition(
                    observation=raw.observation,
                    applied_control=raw.applied_control,
                    action_target=raw.action_target,
                    next_observation_target=raw.next_observation_target,
                    reward_target=raw.reward_target,
                    value_target=running_return,
                    event_targets=raw.event_targets,
                    terminated_target=raw.terminated_target,
                    truncated_target=raw.truncated_target,
                )
            )

        return CurriculumSequence(
            split=config.split,
            sequence_index=sequence_index,
            episode_seed=episode_seed,
            discount=config.discount,
            manifest_sha256=self.manifest_sha256,
            transitions=tuple(reversed(reversed_transitions)),
        )

    def epoch_indices(self, *, epoch: int, shuffle: bool = True) -> tuple[int, ...]:
        """Return a deterministic bijection over sequence indices for an epoch."""
        epoch_number = _integer(epoch, name="epoch")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        count = len(self)
        if not shuffle or count == 1:
            return tuple(range(count))

        seed = sha256(
            b"IRCURRICULUMEPOCH\x01"
            + bytes.fromhex(self.manifest_sha256)
            + pack(">Q", epoch_number)
        ).digest()
        multiplier = int.from_bytes(seed[:8], "big") % count
        if multiplier == 0:
            multiplier = 1
        while gcd(multiplier, count) != 1:
            multiplier = (multiplier + 1) % count
            if multiplier == 0:
                multiplier = 1
        offset = int.from_bytes(seed[8:16], "big") % count
        return tuple((multiplier * index + offset) % count for index in range(count))

    def iter_epoch(
        self,
        *,
        epoch: int,
        shuffle: bool = True,
    ) -> Iterator[CurriculumSequence]:
        """Generate one deterministic epoch without retaining trajectories."""
        return (self[index] for index in self.epoch_indices(epoch=epoch, shuffle=shuffle))


# Backward compatibility and alternative naming
CurriculumSequenceDataset = CurriculumDataset

__all__ = [
    "CURRICULUM_SCENARIOS",
    "CurriculumDataset",
    "CurriculumDatasetConfig",
    "CurriculumScenario",
    "CurriculumSequence",
    "CurriculumSequenceDataset",
    "CurriculumTransition",
    "DatasetSplit",
    "curriculum_dataset_manifest_sha256",
]
