"""Lazy deterministic sensorimotor supervision from the in-repository world.

The generator intentionally has no filesystem, network, ML-framework, or
accelerator dependency.  A sequence is materialized only when indexed, making
the default million-transition pilot a compact virtual dataset rather than a
large local artifact.

Only :class:`~irene_brain.types.ModelObservation` values are model inputs.
World seeds, split identities, rewards, events, and future observations remain
sequence metadata or explicit supervision targets.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from json import dumps
from math import gcd, isfinite
from struct import pack
from typing import Any, overload

from ..environments import MovingShapesEnv
from ..types import GenericControl, HidKey, ModelObservation, RgbFrame, UINT64_MAX

_SEED_NAMESPACE_BITS = 62
_SEED_NAMESPACE_SIZE = 1 << _SEED_NAMESPACE_BITS
_DATASET_SCHEMA_VERSION = 1
_GENERATOR_ID = "irene.moving_shapes.pixel_teacher.v1"
_LICENSE_RECORD_ID = "original-project-content"
_RCQ_V2_SEALED_TEST_START = 3_145_728
_RCQ_V2_DONOR_TEST_START = 3_146_240
_RCQ_V2_SEALED_TEST_END = 3_147_264
_RCQ_V2_CLAIMED_TEST_CAPABILITY = object()


class DatasetSplit(str, Enum):
    """Closed set of dataset partitions with disjoint 64-bit seed namespaces."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_SPLIT_TAGS = {
    DatasetSplit.TRAIN: 0,
    DatasetSplit.VALIDATION: 1,
    DatasetSplit.TEST: 2,
}
_TAG_SPLITS = {tag: split for split, tag in _SPLIT_TAGS.items()}


def _overlaps_rcq_v2_sealed_test_range(config: MovingShapesDatasetConfig) -> bool:
    """Return target-blind range overlap without constructing a dataset."""

    if not isinstance(config, MovingShapesDatasetConfig):
        raise ValueError("config must be a MovingShapesDatasetConfig")
    end = config.seed_offset + config.sequence_count
    return (
        config.split is DatasetSplit.TEST
        and config.seed_offset < _RCQ_V2_SEALED_TEST_END
        and end > _RCQ_V2_SEALED_TEST_START
    )


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = UINT64_MAX,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _split(value: DatasetSplit | str) -> DatasetSplit:
    if isinstance(value, DatasetSplit):
        return value
    if not isinstance(value, str):
        raise ValueError("split must be train, validation, or test")
    try:
        return DatasetSplit(value)
    except ValueError as error:
        raise ValueError("split must be train, validation, or test") from error


def split_episode_seed(split: DatasetSplit | str, episode_index: int) -> int:
    """Map a split-local index into an exact, non-overlapping uint64 seed.

    The two most-significant bits identify the partition.  The remaining 62
    bits are the split-local episode index, so no hashing or probabilistic
    collision argument is involved.
    """

    partition = _split(split)
    local_index = _integer(
        episode_index,
        name="episode_index",
        maximum=_SEED_NAMESPACE_SIZE - 1,
    )
    return (_SPLIT_TAGS[partition] << _SEED_NAMESPACE_BITS) | local_index


def split_for_episode_seed(seed: int) -> DatasetSplit:
    """Return the namespace encoded by a generated seed.

    The fourth two-bit namespace is intentionally unassigned and rejected.
    """

    encoded = _integer(seed, name="seed")
    tag = encoded >> _SEED_NAMESPACE_BITS
    try:
        return _TAG_SPLITS[tag]
    except KeyError as error:
        raise ValueError("seed is outside the registered dataset split namespaces") from error


@dataclass(frozen=True, slots=True)
class MovingShapesDatasetConfig:
    """Identity and generation bounds for a lazy procedural dataset.

    Defaults describe 1,048,576 transitions, but construction performs no
    rollout work.  Callers can choose smaller counts for smoke runs.
    """

    split: DatasetSplit = DatasetSplit.TRAIN
    sequence_count: int = 8_192
    sequence_length: int = 128
    seed_offset: int = 0
    hazard_count: int = 3
    tick_period_ns: int = MovingShapesEnv.DEFAULT_TICK_PERIOD_NS
    discount: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", _split(self.split))
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
        _integer(
            self.hazard_count,
            name="hazard_count",
            minimum=1,
            maximum=MovingShapesEnv.GRID_SIZE * MovingShapesEnv.GRID_SIZE - 2,
        )
        _integer(
            self.tick_period_ns,
            name="tick_period_ns",
            minimum=1,
        )
        discount = _number(self.discount, name="discount")
        if discount < 0.0 or discount > 1.0:
            raise ValueError("discount must be in [0, 1]")
        object.__setattr__(self, "sequence_count", count)
        object.__setattr__(self, "sequence_length", length)
        object.__setattr__(self, "seed_offset", offset)
        object.__setattr__(self, "discount", discount)

    @property
    def total_transitions(self) -> int:
        return self.sequence_count * self.sequence_length

    def manifest_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe identity of this virtual dataset."""

        return {
            "schema_version": _DATASET_SCHEMA_VERSION,
            "generator_id": _GENERATOR_ID,
            "origin": "in-repository deterministic procedural environment",
            "license_record_id": _LICENSE_RECORD_ID,
            "environment_family": "moving_shapes",
            "split": self.split.value,
            "seed_namespace_tag": _SPLIT_TAGS[self.split],
            "seed_offset": self.seed_offset,
            "sequence_count": self.sequence_count,
            "sequence_length": self.sequence_length,
            "total_transitions": self.total_transitions,
            "hazard_count": self.hazard_count,
            "tick_period_ns": self.tick_period_ns,
            "discount_hex": self.discount.hex(),
            "teacher_inputs": "visible_rgb_only",
            "world_target": "next_model_observation",
            "value_target": "zero_bootstrapped_discounted_sequence_return",
        }


def dataset_manifest_sha256(config: MovingShapesDatasetConfig) -> str:
    """Hash a configuration using a process-independent canonical manifest."""

    if not isinstance(config, MovingShapesDatasetConfig):
        raise ValueError("config must be a MovingShapesDatasetConfig")
    encoded = dumps(
        config.manifest_dict(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRMSDATASET\x01" + encoded).hexdigest()


def _sha256_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


@dataclass(frozen=True, slots=True)
class MovingShapesTransition:
    """One causal training transition with inputs and labels kept distinct."""

    observation: ModelObservation
    applied_control: GenericControl
    action_target: GenericControl
    next_observation_target: ModelObservation
    reward_target: float
    value_target: float
    event_targets: tuple[str, ...]
    terminated_target: bool
    truncated_target: bool

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ModelObservation):
            raise ValueError("observation must be a ModelObservation")
        if not isinstance(self.applied_control, GenericControl):
            raise ValueError("applied_control must be a GenericControl")
        if not isinstance(self.action_target, GenericControl):
            raise ValueError("action_target must be a GenericControl")
        if not isinstance(self.next_observation_target, ModelObservation):
            raise ValueError("next_observation_target must be a ModelObservation")
        if self.next_observation_target.frame_id != self.observation.frame_id + 1:
            raise ValueError("world target must be the immediately following frame")
        if self.next_observation_target.elapsed_ns <= self.observation.elapsed_ns:
            raise ValueError("world target time must follow the input observation")
        if self.next_observation_target.previous_control != self.applied_control:
            raise ValueError("world target must carry the physically applied control")
        object.__setattr__(
            self,
            "reward_target",
            _number(self.reward_target, name="reward_target"),
        )
        object.__setattr__(
            self,
            "value_target",
            _number(self.value_target, name="value_target"),
        )
        if not isinstance(self.event_targets, tuple) or any(
            not isinstance(event, str) or not event for event in self.event_targets
        ):
            raise ValueError("event_targets must be a tuple of non-empty strings")
        if not isinstance(self.terminated_target, bool):
            raise ValueError("terminated_target must be a boolean")
        if not isinstance(self.truncated_target, bool):
            raise ValueError("truncated_target must be a boolean")

    @property
    def world_prediction_target(self) -> ModelObservation:
        """Descriptive alias used by trainers for the next observation label."""

        return self.next_observation_target


@dataclass(frozen=True, slots=True)
class MovingShapesSequence:
    """An immutable episode-length recurrent training sample."""

    split: DatasetSplit
    sequence_index: int
    episode_seed: int
    discount: float
    manifest_sha256: str
    transitions: tuple[MovingShapesTransition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", _split(self.split))
        _integer(self.sequence_index, name="sequence_index")
        _integer(self.episode_seed, name="episode_seed")
        if split_for_episode_seed(self.episode_seed) is not self.split:
            raise ValueError("episode_seed belongs to a different dataset split")
        discount = _number(self.discount, name="discount")
        if discount < 0.0 or discount > 1.0:
            raise ValueError("discount must be in [0, 1]")
        object.__setattr__(self, "discount", discount)
        _sha256_string(self.manifest_sha256, name="manifest_sha256")
        if not isinstance(self.transitions, tuple) or not self.transitions or any(
            not isinstance(transition, MovingShapesTransition)
            for transition in self.transitions
        ):
            raise ValueError("transitions must be a non-empty immutable tuple")

        for index, transition in enumerate(self.transitions):
            is_done = transition.terminated_target or transition.truncated_target
            if is_done and index != len(self.transitions) - 1:
                raise ValueError("a sequence cannot continue after a terminal transition")
            if index:
                prior = self.transitions[index - 1]
                if transition.observation != prior.next_observation_target:
                    raise ValueError("adjacent transitions must share the exact boundary frame")
        final = self.transitions[-1]
        if not (final.terminated_target or final.truncated_target):
            raise ValueError("a generated sequence must end at an explicit boundary")

        expected_return = 0.0
        for transition in reversed(self.transitions):
            done = transition.terminated_target or transition.truncated_target
            expected_return = transition.reward_target + (
                0.0 if done else self.discount * expected_return
            )
            if transition.value_target.hex() != expected_return.hex():
                raise ValueError("value_target does not match the discounted sequence return")

    @property
    def model_observations(self) -> tuple[ModelObservation, ...]:
        """Return only legal current-time model inputs, never sequence metadata."""

        return tuple(transition.observation for transition in self.transitions)

    @property
    def action_targets(self) -> tuple[GenericControl, ...]:
        return tuple(transition.action_target for transition in self.transitions)

    @property
    def world_prediction_targets(self) -> tuple[ModelObservation, ...]:
        return tuple(
            transition.next_observation_target for transition in self.transitions
        )

    @property
    def content_sha256(self) -> str:
        digest = sha256(b"IRMSSEQUENCE\x01")
        digest.update(bytes.fromhex(self.manifest_sha256))
        digest.update(pack(">Q", self.sequence_index))
        digest.update(pack(">Q", self.episode_seed))
        digest.update(pack(">d", self.discount))
        digest.update(pack(">I", len(self.transitions)))
        for transition in self.transitions:
            digest.update(bytes.fromhex(transition.observation.content_hash))
            for control in (transition.applied_control, transition.action_target):
                encoded = control.canonical_bytes()
                digest.update(pack(">I", len(encoded)))
                digest.update(encoded)
            digest.update(bytes.fromhex(transition.next_observation_target.content_hash))
            digest.update(pack(">d", transition.reward_target))
            digest.update(pack(">d", transition.value_target))
            digest.update(pack(">I", len(transition.event_targets)))
            for event in transition.event_targets:
                encoded_event = event.encode("utf-8")
                digest.update(pack(">I", len(encoded_event)))
                digest.update(encoded_event)
            digest.update(
                bytes((transition.terminated_target, transition.truncated_target))
            )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _TransitionWithoutValue:
    observation: ModelObservation
    applied_control: GenericControl
    action_target: GenericControl
    next_observation_target: ModelObservation
    reward_target: float
    event_targets: tuple[str, ...]
    terminated_target: bool
    truncated_target: bool


def _coordinates_with_color(
    frame: RgbFrame,
    colors: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    accepted = {bytes(color) for color in colors}
    result: list[tuple[int, int]] = []
    for pixel_index in range(frame.width * frame.height):
        offset = pixel_index * 3
        if frame.pixels[offset : offset + 3] in accepted:
            result.append((pixel_index % frame.width, pixel_index // frame.width))
    return tuple(result)


def _fallback_visible_action(observation: ModelObservation) -> GenericControl:
    """Choose a repeatable scan direction when a hazard occludes the target."""

    digest = sha256(
        b"IRMSOCCLUDED\x01"
        + bytes.fromhex(observation.rgb.sha256)
        + pack(">Q", observation.frame_id)
    ).digest()
    keys = (HidKey.W, HidKey.D, HidKey.S, HidKey.A)
    return GenericControl(keys_down=(int(keys[digest[0] % len(keys)]),))


def _pixel_teacher_action(observation: ModelObservation) -> GenericControl:
    """Greedily approach the visible target using visible RGB only."""

    player_positions = _coordinates_with_color(
        observation.rgb,
        (
            MovingShapesEnv.PLAYER_RGB,
            MovingShapesEnv.PLAYER_HAZARD_OVERLAP_RGB,
        ),
    )
    if len(player_positions) != 1:
        raise RuntimeError("moving-shapes frame does not contain exactly one visible player")
    target_positions = _coordinates_with_color(
        observation.rgb,
        (MovingShapesEnv.TARGET_RGB,),
    )
    if not target_positions:
        return _fallback_visible_action(observation)
    if len(target_positions) != 1:
        raise RuntimeError("moving-shapes frame contains multiple visible targets")

    player_x, player_y = player_positions[0]
    target_x, target_y = target_positions[0]
    keys: list[int] = []
    if target_x < player_x:
        keys.append(int(HidKey.A))
    elif target_x > player_x:
        keys.append(int(HidKey.D))
    if target_y < player_y:
        keys.append(int(HidKey.W))
    elif target_y > player_y:
        keys.append(int(HidKey.S))
    return GenericControl(keys_down=tuple(keys))


class MovingShapesSequenceDataset(Sequence[MovingShapesSequence]):
    """Random-access, on-the-fly dataset with deterministic epoch ordering."""

    __slots__ = ("config", "manifest_sha256")

    def __init__(
        self,
        config: MovingShapesDatasetConfig,
        *,
        _sealed_test_capability: object | None = None,
    ) -> None:
        if not isinstance(config, MovingShapesDatasetConfig):
            raise ValueError("config must be a MovingShapesDatasetConfig")
        if (
            _overlaps_rcq_v2_sealed_test_range(config)
            and _sealed_test_capability is not _RCQ_V2_CLAIMED_TEST_CAPABILITY
        ):
            raise PermissionError(
                "sealed RCQ-v2 TEST ranges require the post-claim dataset factory"
            )
        self.config = config
        self.manifest_sha256 = dataset_manifest_sha256(config)

    def __len__(self) -> int:
        return self.config.sequence_count

    @overload
    def __getitem__(self, index: int) -> MovingShapesSequence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[MovingShapesSequence, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> MovingShapesSequence | tuple[MovingShapesSequence, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer or slice")
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("dataset index out of range")
        return self._generate(normalized)

    def _generate(self, sequence_index: int) -> MovingShapesSequence:
        config = self.config
        episode_seed = split_episode_seed(
            config.split,
            config.seed_offset + sequence_index,
        )
        environment = MovingShapesEnv(
            hazard_count=config.hazard_count,
            tick_period_ns=config.tick_period_ns,
            max_ticks=config.sequence_length,
        )
        raw_observation = environment.reset(episode_seed)
        raw_transitions: list[_TransitionWithoutValue] = []

        for _ in range(config.sequence_length):
            observation = raw_observation.to_model_observation(observation_age_ns=0)
            action_target = _pixel_teacher_action(observation)
            outcome = environment.step(action_target)
            next_observation = outcome.observation.to_model_observation(
                observation_age_ns=0
            )
            raw_transitions.append(
                _TransitionWithoutValue(
                    observation=observation,
                    applied_control=outcome.applied_control,
                    action_target=outcome.requested_control,
                    next_observation_target=next_observation,
                    reward_target=outcome.reward,
                    event_targets=outcome.events,
                    terminated_target=outcome.terminated,
                    truncated_target=outcome.truncated,
                )
            )
            raw_observation = outcome.observation

        running_return = 0.0
        reversed_transitions: list[MovingShapesTransition] = []
        for raw in reversed(raw_transitions):
            done = raw.terminated_target or raw.truncated_target
            running_return = raw.reward_target + (
                0.0 if done else config.discount * running_return
            )
            reversed_transitions.append(
                MovingShapesTransition(
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

        return MovingShapesSequence(
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
            b"IRMSEPOCH\x01"
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
    ) -> Iterator[MovingShapesSequence]:
        """Generate one deterministic epoch without retaining trajectories."""

        return (self[index] for index in self.epoch_indices(epoch=epoch, shuffle=shuffle))


def _claimed_rcq_v2_test_dataset(
    config: MovingShapesDatasetConfig,
) -> MovingShapesSequenceDataset:
    """Construct one exact RCQ-v2 TEST slice after the trusted durable claim.

    This private capability prevents ordinary dataset/test code from
    accidentally opening the sealed recipient, donor, or guard namespace. The
    trusted evaluator remains responsible for verifying and publishing the
    canonical claim before it calls this factory.
    """

    if not isinstance(config, MovingShapesDatasetConfig):
        raise ValueError("config must be a MovingShapesDatasetConfig")
    if (
        config.split is not DatasetSplit.TEST
        or (config.seed_offset, config.sequence_count)
        not in {
            (_RCQ_V2_SEALED_TEST_START, 512),
            (_RCQ_V2_DONOR_TEST_START, 512),
        }
        or config.sequence_length != 8
        or config.hazard_count != 3
        or config.tick_period_ns != 16_666_667
        or config.discount != 0.99
    ):
        raise PermissionError(
            "the claimed RCQ-v2 factory accepts only an exact registered TEST slice"
        )
    return MovingShapesSequenceDataset(
        config,
        _sealed_test_capability=_RCQ_V2_CLAIMED_TEST_CAPABILITY,
    )


__all__ = [
    "DatasetSplit",
    "MovingShapesDatasetConfig",
    "MovingShapesSequence",
    "MovingShapesSequenceDataset",
    "MovingShapesTransition",
    "dataset_manifest_sha256",
    "split_episode_seed",
    "split_for_episode_seed",
]
