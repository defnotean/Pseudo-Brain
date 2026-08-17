"""Deterministic batching and the fixed 307-channel actuator target layout."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil
from typing import Iterator

from ..data import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequence,
    MovingShapesSequenceDataset,
)
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
    "MovingShapesBatchSource",
    "SCROLL_INDEX",
    "TrajectoryBatch",
    "control_to_vector",
]
