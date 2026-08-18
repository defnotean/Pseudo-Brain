"""Deterministic batching and the fixed 307-channel actuator target layout."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, gcd
from struct import pack
from typing import Iterator

from ..data import (
    DatasetSplit,
    MazeChaseDatasetConfig,
    MazeChaseSequenceDataset,
    MovingShapesDatasetConfig,
    MovingShapesSequence,
    MovingShapesSequenceDataset,
    SOLVER_WORLD_NAMES,
    SolverDatasetConfig,
    SolverSequenceDataset,
)
from ..environments.keys_doors import KeysDoorsEnv
from ..environments.maze_chase import _GHOST_RULES, MazeChaseEnv
from ..types import GenericControl
from .config import DatasetConfig


CONTROL_VECTOR_SIZE = 307
KEYBOARD_SLICE = slice(0, 256)
MOUSE_BUTTON_SLICE = slice(256, 264)
MOUSE_AXIS_SLICE = slice(264, 266)
SCROLL_INDEX = 266
GAMEPAD_BUTTON_SLICE = slice(267, 299)
GAMEPAD_AXIS_SLICE = slice(299, 307)
BUTTON_TARGET_INDICES = tuple(range(0, 264)) + tuple(range(267, 299))
CONTINUOUS_TARGET_INDICES = (264, 265, 266, *range(299, 307))
CONTROL_LAYOUT_ID = "generic-hid-307-v1"


def dataset_batch_source(config: DatasetConfig):
    """Resolve the deterministic batch source for a pinned dataset kind.

    ``maze_chase`` reuses the shared split-count fields and the canonical
    maze slot knobs. ``hazard_count`` stays on the schema for moving_shapes
    compatibility and is ignored for maze_chase.
    """

    if not isinstance(config, DatasetConfig):
        raise ValueError("config must be a DatasetConfig")
    if config.kind == "moving_shapes":
        return MovingShapesBatchSource(config)
    if config.kind == "maze_chase":
        return MazeChaseBatchSource(
            MazeChaseBatchConfig(
                train_sequences=config.train_sequences,
                validation_sequences=config.validation_sequences,
                test_sequences=config.test_sequences,
                sequence_length=config.sequence_length,
                burn_in_steps=config.burn_in_steps,
                seed_offset=config.seed_offset,
                tick_period_ns=config.tick_period_ns,
                discount=config.discount,
            )
        )
    raise ValueError(f"unsupported dataset.kind: {config.kind}")


def control_to_vector(control: GenericControl) -> tuple[float, ...]:
    """Encode only physical actuator state, excluding sequencing metadata."""

    if not isinstance(control, GenericControl):
        raise ValueError("control must be a GenericControl")
    result = [0.0] * CONTROL_VECTOR_SIZE
    for key in control.keys_down:
        result[key] = 1.0
    for button in control.mouse_buttons:
        result[MOUSE_BUTTON_SLICE.start + button] = 1.0
    result[MOUSE_AXIS_SLICE.start] = control.mouse_dx
    result[MOUSE_AXIS_SLICE.start + 1] = control.mouse_dy
    result[SCROLL_INDEX] = control.mouse_wheel
    for button in control.gamepad_buttons:
        result[GAMEPAD_BUTTON_SLICE.start + button] = 1.0
    axes = control.gamepad_axes if control.gamepad_axes else (0.0,) * 8
    result[GAMEPAD_AXIS_SLICE] = axes
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """Raw causal sequences; model inputs and labels remain separate objects."""

    split: str
    burn_in_steps: int
    sequences: tuple[MovingShapesSequence, ...]

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if type(self.burn_in_steps) is not int or self.burn_in_steps < 0:
            raise ValueError("burn_in_steps must be a nonnegative integer")
        if not isinstance(self.sequences, tuple) or not self.sequences:
            raise ValueError("sequences must be a non-empty tuple")
        expected_length = len(self.sequences[0].transitions)
        if self.burn_in_steps >= expected_length:
            raise ValueError("burn_in_steps must be smaller than sequence length")
        if any(len(sequence.transitions) != expected_length for sequence in self.sequences):
            raise ValueError("all sequences in a batch must have equal length")
        if any(sequence.split.value != self.split for sequence in self.sequences):
            raise ValueError("batch split must match every sequence split")

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def sequence_length(self) -> int:
        return len(self.sequences[0].transitions)

    @property
    def sample_count(self) -> int:
        return self.batch_size * (self.sequence_length - self.burn_in_steps)


class MovingShapesBatchSource:
    """Lazy split-namespaced trajectories with deterministic epoch ordering."""

    def __init__(self, config: DatasetConfig) -> None:
        if not isinstance(config, DatasetConfig):
            raise ValueError("config must be a DatasetConfig")
        self.config = config
        counts = {
            DatasetSplit.TRAIN: config.train_sequences,
            DatasetSplit.VALIDATION: config.validation_sequences,
            DatasetSplit.TEST: config.test_sequences,
        }
        self._datasets = {
            split: MovingShapesSequenceDataset(
                MovingShapesDatasetConfig(
                    split=split,
                    sequence_count=count,
                    sequence_length=config.sequence_length,
                    seed_offset=config.seed_offset,
                    hazard_count=config.hazard_count,
                    tick_period_ns=config.tick_period_ns,
                    discount=config.discount,
                )
            )
            for split, count in counts.items()
        }
        manifest = {
            "schema_version": 1,
            "batch_source": "moving_shapes_split_namespaces",
            "control_layout": CONTROL_LAYOUT_ID,
            "input_boundary": "ModelObservation-v1",
            "burn_in_steps": config.burn_in_steps,
            "splits": {
                split.value: dataset.manifest_sha256
                for split, dataset in sorted(
                    self._datasets.items(), key=lambda item: item[0].value
                )
            },
        }
        encoded = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._manifest_sha256 = sha256(b"IRTRAINBATCH\x01" + encoded).hexdigest()

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @staticmethod
    def _split(value: str) -> DatasetSplit:
        try:
            return DatasetSplit(value)
        except (TypeError, ValueError) as error:
            raise ValueError("split must be train, validation, or test") from error

    def batches_per_epoch(self, *, split: str, batch_size: int) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        dataset = self._datasets[self._split(split)]
        return ceil(len(dataset) / batch_size)

    def iter_batches(
        self,
        *,
        split: str,
        epoch: int,
        start_batch: int,
        batch_size: int,
        max_batches: int | None = None,
    ) -> Iterator[TrajectoryBatch]:
        partition = self._split(split)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if type(start_batch) is not int or start_batch < 0:
            raise ValueError("start_batch must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if max_batches is not None and (
            type(max_batches) is not int or max_batches < 1
        ):
            raise ValueError("max_batches must be a positive integer or None")

        dataset = self._datasets[partition]
        total_batches = self.batches_per_epoch(split=split, batch_size=batch_size)
        if start_batch > total_batches:
            raise ValueError("start_batch exceeds the number of batches")
        indices = dataset.epoch_indices(
            epoch=epoch,
            shuffle=partition is DatasetSplit.TRAIN,
        )
        emitted = 0
        for batch_index in range(start_batch, total_batches):
            if max_batches is not None and emitted >= max_batches:
                break
            start = batch_index * batch_size
            selected = indices[start : start + batch_size]
            yield TrajectoryBatch(
                split=partition.value,
                burn_in_steps=self.config.burn_in_steps,
                sequences=tuple(dataset[index] for index in selected),
            )
            emitted += 1


@dataclass(frozen=True, slots=True)
class MazeChaseBatchConfig:
    """Split counts and maze_chase world knobs for :class:`MazeChaseBatchSource`.

    Kept separate from the pinned ``DatasetConfig`` (whose ``kind`` only
    supports moving_shapes) so registered moving_shapes configuration hashes
    are untouched.
    """

    train_sequences: int
    validation_sequences: int
    test_sequences: int
    sequence_length: int
    burn_in_steps: int
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
        for value, name in (
            (self.train_sequences, "maze_chase.train_sequences"),
            (self.validation_sequences, "maze_chase.validation_sequences"),
            (self.test_sequences, "maze_chase.test_sequences"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for value, name, low, high in (
            (self.sequence_length, "maze_chase.sequence_length", 2, 2**32 - 1),
            (self.burn_in_steps, "maze_chase.burn_in_steps", 0, 2**32 - 1),
            (self.seed_offset, "maze_chase.seed_offset", 0, (1 << 62) - 1),
            (self.ghost_count, "maze_chase.ghost_count", 1, 8),
            (
                self.ghost_period,
                "maze_chase.ghost_period",
                1,
                MazeChaseEnv.MAX_GHOST_PERIOD,
            ),
            (
                self.player_period,
                "maze_chase.player_period",
                1,
                MazeChaseEnv.MAX_PLAYER_PERIOD,
            ),
            (
                self.extra_loops,
                "maze_chase.extra_loops",
                0,
                MazeChaseEnv.MAX_EXTRA_LOOPS,
            ),
            (
                self.input_delay_ticks,
                "maze_chase.input_delay_ticks",
                0,
                MazeChaseEnv.MAX_INPUT_DELAY,
            ),
            (self.tick_period_ns, "maze_chase.tick_period_ns", 1, 2**64 - 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < low or value > high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if self.burn_in_steps >= self.sequence_length:
            raise ValueError(
                "maze_chase.burn_in_steps must be smaller than sequence_length"
            )
        if not isinstance(self.ghost_rule, str) or self.ghost_rule not in _GHOST_RULES:
            raise ValueError(f"maze_chase.ghost_rule must be one of {sorted(_GHOST_RULES)}")
        if not isinstance(self.ghost_elroy, bool):
            raise ValueError("maze_chase.ghost_elroy must be a boolean")
        if not isinstance(self.sticky_direction, bool):
            raise ValueError("maze_chase.sticky_direction must be a boolean")
        if isinstance(self.discount, bool) or not isinstance(
            self.discount, (int, float)
        ):
            raise ValueError("maze_chase.discount must be a number")
        if not 0.0 <= float(self.discount) <= 1.0:
            raise ValueError("maze_chase.discount must be in [0, 1]")
        if self.seed_offset + max(
            self.train_sequences,
            self.validation_sequences,
            self.test_sequences,
        ) > (1 << 62):
            raise ValueError("maze_chase sequence range exceeds its split namespace")


class MazeChaseBatchSource:
    """Lazy split-namespaced maze_chase trajectories with deterministic epochs.

    Batches require equal-length sequences (:class:`TrajectoryBatch` fails
    closed otherwise). A maze_chase episode cannot terminate before every
    pellet is eaten — at least one tick per pellet — so sequences whose
    length stays below the maze's pellet count always truncate at the
    boundary and batch cleanly; the canonical slot clears in roughly 150–250
    ticks, well above the registered 128-tick training length.
    """

    def __init__(self, config: MazeChaseBatchConfig) -> None:
        if not isinstance(config, MazeChaseBatchConfig):
            raise ValueError("config must be a MazeChaseBatchConfig")
        self.config = config
        counts = {
            DatasetSplit.TRAIN: config.train_sequences,
            DatasetSplit.VALIDATION: config.validation_sequences,
            DatasetSplit.TEST: config.test_sequences,
        }
        self._datasets = {
            split: MazeChaseSequenceDataset(
                MazeChaseDatasetConfig(
                    split=split,
                    sequence_count=count,
                    sequence_length=config.sequence_length,
                    seed_offset=config.seed_offset,
                    ghost_count=config.ghost_count,
                    ghost_period=config.ghost_period,
                    player_period=config.player_period,
                    extra_loops=config.extra_loops,
                    ghost_rule=config.ghost_rule,
                    ghost_elroy=config.ghost_elroy,
                    input_delay_ticks=config.input_delay_ticks,
                    sticky_direction=config.sticky_direction,
                    tick_period_ns=config.tick_period_ns,
                    discount=config.discount,
                )
            )
            for split, count in counts.items()
        }
        manifest = {
            "schema_version": 1,
            "batch_source": "maze_chase_split_namespaces",
            "control_layout": CONTROL_LAYOUT_ID,
            "input_boundary": "ModelObservation-v1",
            "burn_in_steps": config.burn_in_steps,
            "splits": {
                split.value: dataset.manifest_sha256
                for split, dataset in sorted(
                    self._datasets.items(), key=lambda item: item[0].value
                )
            },
        }
        encoded = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._manifest_sha256 = sha256(b"IRTRAINBATCH\x01" + encoded).hexdigest()

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @staticmethod
    def _split(value: str) -> DatasetSplit:
        try:
            return DatasetSplit(value)
        except (TypeError, ValueError) as error:
            raise ValueError("split must be train, validation, or test") from error

    def batches_per_epoch(self, *, split: str, batch_size: int) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        dataset = self._datasets[self._split(split)]
        return ceil(len(dataset) / batch_size)

    def iter_batches(
        self,
        *,
        split: str,
        epoch: int,
        start_batch: int,
        batch_size: int,
        max_batches: int | None = None,
    ) -> Iterator[TrajectoryBatch]:
        partition = self._split(split)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if type(start_batch) is not int or start_batch < 0:
            raise ValueError("start_batch must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if max_batches is not None and (
            type(max_batches) is not int or max_batches < 1
        ):
            raise ValueError("max_batches must be a positive integer or None")

        dataset = self._datasets[partition]
        total_batches = self.batches_per_epoch(split=split, batch_size=batch_size)
        if start_batch > total_batches:
            raise ValueError("start_batch exceeds the number of batches")
        indices = dataset.epoch_indices(
            epoch=epoch,
            shuffle=partition is DatasetSplit.TRAIN,
        )
        emitted = 0
        for batch_index in range(start_batch, total_batches):
            if max_batches is not None and emitted >= max_batches:
                break
            start = batch_index * batch_size
            selected = indices[start : start + batch_size]
            yield TrajectoryBatch(
                split=partition.value,
                burn_in_steps=self.config.burn_in_steps,
                sequences=tuple(dataset[index] for index in selected),
            )
            emitted += 1


@dataclass(frozen=True, slots=True)
class SolverBatchConfig:
    """Split counts and world knob for :class:`SolverBatchSource`.

    Kept separate from the pinned ``DatasetConfig`` (whose ``kind`` only
    supports moving_shapes) so registered moving_shapes configuration hashes
    are untouched. The four solver worlds (keys_doors, junction, occlusion,
    pursuit) expose no additional generation knobs beyond the shared
    sequence/split/timing fields.
    """

    train_sequences: int
    validation_sequences: int
    test_sequences: int
    sequence_length: int
    burn_in_steps: int
    world: str = "keys_doors"
    seed_offset: int = 0
    tick_period_ns: int = KeysDoorsEnv.DEFAULT_TICK_PERIOD_NS
    discount: float = 0.99

    def __post_init__(self) -> None:
        for value, name in (
            (self.train_sequences, "solver.train_sequences"),
            (self.validation_sequences, "solver.validation_sequences"),
            (self.test_sequences, "solver.test_sequences"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for value, name, low, high in (
            (self.sequence_length, "solver.sequence_length", 2, 2**32 - 1),
            (self.burn_in_steps, "solver.burn_in_steps", 0, 2**32 - 1),
            (self.seed_offset, "solver.seed_offset", 0, (1 << 62) - 1),
            (self.tick_period_ns, "solver.tick_period_ns", 1, 2**64 - 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < low or value > high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if self.burn_in_steps >= self.sequence_length:
            raise ValueError(
                "solver.burn_in_steps must be smaller than sequence_length"
            )
        if not isinstance(self.world, str) or self.world not in SOLVER_WORLD_NAMES:
            raise ValueError(f"solver.world must be one of {sorted(SOLVER_WORLD_NAMES)}")
        if isinstance(self.discount, bool) or not isinstance(
            self.discount, (int, float)
        ):
            raise ValueError("solver.discount must be a number")
        if not 0.0 <= float(self.discount) <= 1.0:
            raise ValueError("solver.discount must be in [0, 1]")
        if self.seed_offset + max(
            self.train_sequences,
            self.validation_sequences,
            self.test_sequences,
        ) > (1 << 62):
            raise ValueError("solver sequence range exceeds its split namespace")


class SolverBatchSource:
    """Lazy split-namespaced solver trajectories with deterministic epochs.

    Batches require equal-length sequences (:class:`TrajectoryBatch` fails
    closed otherwise). All four solver worlds never terminate an episode
    (``terminated`` is always ``False``), so every sequence truncates at the
    length boundary and batches cleanly — even simpler than maze_chase, whose
    episodes end once every pellet is eaten.
    """

    def __init__(self, config: SolverBatchConfig) -> None:
        if not isinstance(config, SolverBatchConfig):
            raise ValueError("config must be a SolverBatchConfig")
        self.config = config
        counts = {
            DatasetSplit.TRAIN: config.train_sequences,
            DatasetSplit.VALIDATION: config.validation_sequences,
            DatasetSplit.TEST: config.test_sequences,
        }
        self._datasets = {
            split: SolverSequenceDataset(
                SolverDatasetConfig(
                    world=config.world,
                    split=split,
                    sequence_count=count,
                    sequence_length=config.sequence_length,
                    seed_offset=config.seed_offset,
                    tick_period_ns=config.tick_period_ns,
                    discount=config.discount,
                )
            )
            for split, count in counts.items()
        }
        manifest = {
            "schema_version": 1,
            "batch_source": "solver_split_namespaces",
            "world": config.world,
            "control_layout": CONTROL_LAYOUT_ID,
            "input_boundary": "ModelObservation-v1",
            "burn_in_steps": config.burn_in_steps,
            "splits": {
                split.value: dataset.manifest_sha256
                for split, dataset in sorted(
                    self._datasets.items(), key=lambda item: item[0].value
                )
            },
        }
        encoded = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._manifest_sha256 = sha256(b"IRTRAINBATCH\x01" + encoded).hexdigest()

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @staticmethod
    def _split(value: str) -> DatasetSplit:
        try:
            return DatasetSplit(value)
        except (TypeError, ValueError) as error:
            raise ValueError("split must be train, validation, or test") from error

    def batches_per_epoch(self, *, split: str, batch_size: int) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        dataset = self._datasets[self._split(split)]
        return ceil(len(dataset) / batch_size)

    def iter_batches(
        self,
        *,
        split: str,
        epoch: int,
        start_batch: int,
        batch_size: int,
        max_batches: int | None = None,
    ) -> Iterator[TrajectoryBatch]:
        partition = self._split(split)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if type(start_batch) is not int or start_batch < 0:
            raise ValueError("start_batch must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if max_batches is not None and (
            type(max_batches) is not int or max_batches < 1
        ):
            raise ValueError("max_batches must be a positive integer or None")

        dataset = self._datasets[partition]
        total_batches = self.batches_per_epoch(split=split, batch_size=batch_size)
        if start_batch > total_batches:
            raise ValueError("start_batch exceeds the number of batches")
        indices = dataset.epoch_indices(
            epoch=epoch,
            shuffle=partition is DatasetSplit.TRAIN,
        )
        emitted = 0
        for batch_index in range(start_batch, total_batches):
            if max_batches is not None and emitted >= max_batches:
                break
            start = batch_index * batch_size
            selected = indices[start : start + batch_size]
            yield TrajectoryBatch(
                split=partition.value,
                burn_in_steps=self.config.burn_in_steps,
                sequences=tuple(dataset[index] for index in selected),
            )
            emitted += 1


_MIXED_WORLD_ORDER = ("moving_shapes", "maze_chase")


@dataclass(frozen=True, slots=True)
class MixedWorldBatchConfig:
    """Equal-count mix of moving_shapes and maze_chase for the B3 generalist.

    Kept separate from the pinned ``DatasetConfig`` (whose ``kind`` only
    supports moving_shapes) so registered moving_shapes configuration hashes
    are untouched. Both member configs must share split counts,
    sequence_length, burn_in_steps, seed_offset, and discount; per-world
    timing knobs may differ. Campaign-scale materialization of either world
    remains accelerator-window work; this config only names the lazy mix.
    """

    moving_shapes: DatasetConfig
    maze_chase: MazeChaseBatchConfig

    def __post_init__(self) -> None:
        if not isinstance(self.moving_shapes, DatasetConfig):
            raise ValueError("mixed.moving_shapes must be a DatasetConfig")
        if not isinstance(self.maze_chase, MazeChaseBatchConfig):
            raise ValueError("mixed.maze_chase must be a MazeChaseBatchConfig")
        moving = self.moving_shapes
        maze = self.maze_chase
        for name in (
            "train_sequences",
            "validation_sequences",
            "test_sequences",
            "sequence_length",
            "burn_in_steps",
            "seed_offset",
        ):
            if getattr(moving, name) != getattr(maze, name):
                raise ValueError(f"mixed worlds must share {name}")
        if float(moving.discount) != float(maze.discount):
            raise ValueError("mixed worlds must share discount")


class MixedWorldBatchSource:
    """Lazy round-robin mix of moving_shapes and maze_chase trajectories.

    Unshuffled order is moving_shapes[0], maze_chase[0], moving_shapes[1],
    maze_chase[1], … so early batches already see both worlds. Train epochs
    apply the same affine bijection the single-world sources use, over the
    combined index space, keyed by this source's manifest. Batches require
    equal-length sequences (:class:`TrajectoryBatch` fails closed otherwise);
    the shared ``sequence_length`` makes mixed batches legal.

    This is the B3 generalist data identity. Specialists consume the member
    sources (:class:`MovingShapesBatchSource`, :class:`MazeChaseBatchSource`)
    under the same knobs. ``DatasetConfig.kind`` stays ``moving_shapes``.
    """

    def __init__(self, config: MixedWorldBatchConfig) -> None:
        if not isinstance(config, MixedWorldBatchConfig):
            raise ValueError("config must be a MixedWorldBatchConfig")
        self.config = config
        moving = config.moving_shapes
        maze = config.maze_chase
        counts = {
            DatasetSplit.TRAIN: moving.train_sequences,
            DatasetSplit.VALIDATION: moving.validation_sequences,
            DatasetSplit.TEST: moving.test_sequences,
        }
        self._moving = {
            split: MovingShapesSequenceDataset(
                MovingShapesDatasetConfig(
                    split=split,
                    sequence_count=count,
                    sequence_length=moving.sequence_length,
                    seed_offset=moving.seed_offset,
                    hazard_count=moving.hazard_count,
                    tick_period_ns=moving.tick_period_ns,
                    discount=moving.discount,
                )
            )
            for split, count in counts.items()
        }
        self._maze = {
            split: MazeChaseSequenceDataset(
                MazeChaseDatasetConfig(
                    split=split,
                    sequence_count=count,
                    sequence_length=maze.sequence_length,
                    seed_offset=maze.seed_offset,
                    ghost_count=maze.ghost_count,
                    ghost_period=maze.ghost_period,
                    player_period=maze.player_period,
                    extra_loops=maze.extra_loops,
                    ghost_rule=maze.ghost_rule,
                    ghost_elroy=maze.ghost_elroy,
                    input_delay_ticks=maze.input_delay_ticks,
                    sticky_direction=maze.sticky_direction,
                    tick_period_ns=maze.tick_period_ns,
                    discount=maze.discount,
                )
            )
            for split, count in counts.items()
        }
        manifest = {
            "schema_version": 1,
            "batch_source": "mixed_world_split_namespaces",
            "worlds": list(_MIXED_WORLD_ORDER),
            "interleave": "round_robin_moving_shapes_then_maze_chase",
            "control_layout": CONTROL_LAYOUT_ID,
            "input_boundary": "ModelObservation-v1",
            "burn_in_steps": moving.burn_in_steps,
            "splits": {
                split.value: {
                    "moving_shapes": self._moving[split].manifest_sha256,
                    "maze_chase": self._maze[split].manifest_sha256,
                }
                for split in sorted(counts, key=lambda item: item.value)
            },
        }
        encoded = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._manifest_sha256 = sha256(b"IRTRAINBATCH\x01" + encoded).hexdigest()

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @staticmethod
    def _split(value: str) -> DatasetSplit:
        try:
            return DatasetSplit(value)
        except (TypeError, ValueError) as error:
            raise ValueError("split must be train, validation, or test") from error

    def split_world_manifests(self, split: str) -> dict[str, str]:
        partition = self._split(split)
        return {
            "moving_shapes": self._moving[partition].manifest_sha256,
            "maze_chase": self._maze[partition].manifest_sha256,
        }

    def _epoch_tags(
        self, *, partition: DatasetSplit, epoch: int
    ) -> tuple[tuple[str, int], ...]:
        moving_count = len(self._moving[partition])
        maze_count = len(self._maze[partition])
        if moving_count != maze_count:
            raise ValueError("mixed worlds must contribute equally many sequences")
        total = moving_count + maze_count
        tags = tuple(
            (_MIXED_WORLD_ORDER[index % 2], index // 2) for index in range(total)
        )
        shuffle = partition is DatasetSplit.TRAIN
        if not shuffle or total <= 1:
            return tags
        seed = sha256(
            b"IRMIXEPOCH\x01"
            + bytes.fromhex(self.manifest_sha256)
            + pack(">Q", epoch)
        ).digest()
        multiplier = int.from_bytes(seed[:8], "big") % total
        if multiplier == 0:
            multiplier = 1
        while gcd(multiplier, total) != 1:
            multiplier = (multiplier + 1) % total
            if multiplier == 0:
                multiplier = 1
        offset = int.from_bytes(seed[8:16], "big") % total
        order = tuple((multiplier * index + offset) % total for index in range(total))
        return tuple(tags[position] for position in order)

    def _sequence(self, partition: DatasetSplit, world: str, index: int):
        if world == "moving_shapes":
            return self._moving[partition][index]
        if world == "maze_chase":
            return self._maze[partition][index]
        raise ValueError(f"unknown mixed world: {world}")

    def batches_per_epoch(self, *, split: str, batch_size: int) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        partition = self._split(split)
        total = len(self._moving[partition]) + len(self._maze[partition])
        return ceil(total / batch_size)

    def iter_batches(
        self,
        *,
        split: str,
        epoch: int,
        start_batch: int,
        batch_size: int,
        max_batches: int | None = None,
    ) -> Iterator[TrajectoryBatch]:
        partition = self._split(split)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if type(start_batch) is not int or start_batch < 0:
            raise ValueError("start_batch must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if max_batches is not None and (
            type(max_batches) is not int or max_batches < 1
        ):
            raise ValueError("max_batches must be a positive integer or None")

        total_batches = self.batches_per_epoch(split=split, batch_size=batch_size)
        if start_batch > total_batches:
            raise ValueError("start_batch exceeds the number of batches")
        tags = self._epoch_tags(partition=partition, epoch=epoch)
        emitted = 0
        for batch_index in range(start_batch, total_batches):
            if max_batches is not None and emitted >= max_batches:
                break
            start = batch_index * batch_size
            selected = tags[start : start + batch_size]
            yield TrajectoryBatch(
                split=partition.value,
                burn_in_steps=self.config.moving_shapes.burn_in_steps,
                sequences=tuple(
                    self._sequence(partition, world, index)
                    for world, index in selected
                ),
            )
            emitted += 1


__all__ = [
    "BUTTON_TARGET_INDICES",
    "CONTINUOUS_TARGET_INDICES",
    "CONTROL_LAYOUT_ID",
    "CONTROL_VECTOR_SIZE",
    "GAMEPAD_AXIS_SLICE",
    "GAMEPAD_BUTTON_SLICE",
    "KEYBOARD_SLICE",
    "MOUSE_AXIS_SLICE",
    "MOUSE_BUTTON_SLICE",
    "MazeChaseBatchConfig",
    "MazeChaseBatchSource",
    "MixedWorldBatchConfig",
    "MixedWorldBatchSource",
    "MovingShapesBatchSource",
    "SCROLL_INDEX",
    "SolverBatchConfig",
    "SolverBatchSource",
    "TrajectoryBatch",
    "control_to_vector",
    "dataset_batch_source",
]
