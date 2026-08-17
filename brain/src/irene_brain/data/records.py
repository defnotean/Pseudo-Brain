"""Immutable records for Pseudo-Brain lifetime and counterfactual datasets.

The serialized records intentionally contain more information than a deployed
model may consume. In particular, rewards, simulator snapshots, storage
locators, privileged state, and branch outcomes are not model input.
``StepRecord.materialize_model_observation`` resolves sensor content and builds
the same canonical boundary used by live inference.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping, TypeAlias, overload

from irene_brain.types import (
    GenericControl,
    MODEL_OBSERVATION_FIELDS as MODEL_OBSERVATION_FIELDS,
    ModelObservation,
    RgbFrame,
    UINT64_MAX,
)


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]


PRIVILEGED_STEP_FIELDS = frozenset(
    {
        "lifetime_id",
        "episode_id",
        "step",
        "environment_tick",
        "capture_time_qpc",
        "observation_ready_qpc",
        "action_requested_qpc",
        "action_applied_qpc",
        "rgb_ref",
        "audio_ref",
        "requested_control",
        "applied_control",
        "reward_raw",
        "events",
        "terminated",
        "truncated",
        "snapshot_hash",
        "privileged_state_ref",
        "metadata",
    }
)
"""Stored labels and identifiers that must never enter the model input view."""


_CONTROL_FIELDS = frozenset(
    {
        "keys_down",
        "mouse_dx",
        "mouse_dy",
        "mouse_buttons",
        "mouse_wheel",
        "gamepad_axes",
        "gamepad_buttons",
        "impulse_sequence",
        "intended_hold_ns",
    }
)


def _empty_json_object() -> JsonObject:
    return MappingProxyType({})


def _freeze_json(value: object, *, path: str) -> JsonValue:
    """Validate JSON data and return a recursively immutable canonical value."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} object keys must be strings")
        frozen: dict[str, JsonValue] = {}
        for key in sorted(value):
            frozen[key] = _freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{path} contains non-JSON value {type(value).__name__}")


def _freeze_json_object(value: object, *, name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    frozen = _freeze_json(value, path=name)
    if not isinstance(frozen, Mapping):  # Defensive; the input check makes this unreachable.
        raise ValueError(f"{name} must be a JSON object")
    return frozen


def _thaw_json(value: JsonValue) -> Any:
    """Return a detached, JSON-serializable copy with deterministic key order."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _validate_keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(sorted(missing))}")
    unknown = set(value).difference(required, optional)
    if unknown:
        raise ValueError(
            f"{name} contains unknown fields: {', '.join(sorted(unknown))}; "
            "put forward-compatible annotations under metadata"
        )


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_nonempty_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name=name)


@overload
def _sha256(
    value: object,
    *,
    name: str,
    optional: Literal[False] = False,
) -> str: ...


@overload
def _sha256(
    value: object,
    *,
    name: str,
    optional: Literal[True],
) -> str | None: ...


def _sha256(value: object, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    digest = _nonempty_string(value, name=name)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


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


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array of strings")
    result = tuple(
        _nonempty_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    return result


def _validate_control_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    _validate_keys(mapping, name=name, required=_CONTROL_FIELDS)

    for sequence_name in (
        "keys_down",
        "mouse_buttons",
        "gamepad_axes",
        "gamepad_buttons",
    ):
        sequence = mapping[sequence_name]
        if not isinstance(sequence, (list, tuple)):
            raise ValueError(f"{name}.{sequence_name} must be an array")

    for sequence_name in ("keys_down", "mouse_buttons", "gamepad_buttons"):
        for index, item in enumerate(mapping[sequence_name]):
            _integer(item, name=f"{name}.{sequence_name}[{index}]")

    for index, item in enumerate(mapping["gamepad_axes"]):
        _number(item, name=f"{name}.gamepad_axes[{index}]")
    for scalar_name in ("mouse_dx", "mouse_dy", "mouse_wheel"):
        _number(mapping[scalar_name], name=f"{name}.{scalar_name}")
    _integer(mapping["impulse_sequence"], name=f"{name}.impulse_sequence")
    _integer(mapping["intended_hold_ns"], name=f"{name}.intended_hold_ns")
    return mapping


def _control_from_dict(value: object, *, name: str) -> GenericControl:
    return GenericControl.from_dict(_validate_control_mapping(value, name=name))


def _validate_control(value: object, *, name: str) -> GenericControl:
    if not isinstance(value, GenericControl):
        raise ValueError(f"{name} must be GenericControl")
    if isinstance(value.intended_hold_ns, bool):
        raise ValueError(f"{name}.intended_hold_ns must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class LifetimeRecord:
    """Immutable manifest for a continuous learning lifetime."""

    SOURCE_TYPES: ClassVar[frozenset[str]] = frozenset({"human", "self_play", "branch"})
    SPLITS: ClassVar[frozenset[str]] = frozenset({"train", "validation", "test"})
    _REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "lifetime_id",
            "world_lineage_id",
            "environment_family",
            "environment_build_sha256",
            "container_digest",
            "generator_config",
            "seed_vector",
            "control_mapping_hash",
            "qpc_frequency_hz",
            "graphics_config",
            "source_type",
            "source_session_id",
            "license_record_id",
            "split",
        }
    )
    _OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"ancestor_lifetime_ids", "metadata", "player_pseudonym"}
    )

    lifetime_id: str
    world_lineage_id: str
    environment_family: str
    environment_build_sha256: str
    container_digest: str
    generator_config: JsonObject
    seed_vector: JsonObject
    control_mapping_hash: str
    qpc_frequency_hz: int
    graphics_config: JsonObject
    source_type: str
    source_session_id: str
    license_record_id: str
    split: str
    player_pseudonym: str | None = None
    ancestor_lifetime_ids: tuple[str, ...] = ()
    metadata: JsonObject = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        for name in (
            "lifetime_id",
            "world_lineage_id",
            "environment_family",
            "container_digest",
            "source_session_id",
            "license_record_id",
        ):
            _nonempty_string(getattr(self, name), name=name)
        _sha256(self.environment_build_sha256, name="environment_build_sha256")
        _sha256(self.control_mapping_hash, name="control_mapping_hash")
        _integer(self.qpc_frequency_hz, name="qpc_frequency_hz", minimum=1)
        _nonempty_string(self.source_type, name="source_type")
        _nonempty_string(self.split, name="split")
        if self.source_type not in self.SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(self.SOURCE_TYPES)}")
        if self.split not in self.SPLITS:
            raise ValueError(f"split must be one of {sorted(self.SPLITS)}")
        _optional_nonempty_string(self.player_pseudonym, name="player_pseudonym")
        if self.source_type == "human" and self.player_pseudonym is None:
            raise ValueError("human lifetimes require player_pseudonym")
        object.__setattr__(
            self,
            "ancestor_lifetime_ids",
            _string_tuple(
                self.ancestor_lifetime_ids,
                name="ancestor_lifetime_ids",
            ),
        )
        if self.lifetime_id in self.ancestor_lifetime_ids:
            raise ValueError("a lifetime cannot list itself as an ancestor")
        if len(self.ancestor_lifetime_ids) != len(set(self.ancestor_lifetime_ids)):
            raise ValueError("ancestor_lifetime_ids cannot contain duplicates")
        object.__setattr__(
            self,
            "generator_config",
            _freeze_json_object(self.generator_config, name="generator_config"),
        )
        object.__setattr__(
            self,
            "seed_vector",
            _freeze_json_object(self.seed_vector, name="seed_vector"),
        )
        object.__setattr__(
            self,
            "graphics_config",
            _freeze_json_object(self.graphics_config, name="graphics_config"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "lifetime_id": self.lifetime_id,
            "world_lineage_id": self.world_lineage_id,
            "environment_family": self.environment_family,
            "environment_build_sha256": self.environment_build_sha256,
            "container_digest": self.container_digest,
            "generator_config": _thaw_json(self.generator_config),
            "seed_vector": _thaw_json(self.seed_vector),
            "control_mapping_hash": self.control_mapping_hash,
            "qpc_frequency_hz": self.qpc_frequency_hz,
            "graphics_config": _thaw_json(self.graphics_config),
            "source_type": self.source_type,
            "source_session_id": self.source_session_id,
            "license_record_id": self.license_record_id,
            "split": self.split,
        }
        if self.player_pseudonym is not None:
            result["player_pseudonym"] = self.player_pseudonym
        if self.ancestor_lifetime_ids:
            result["ancestor_lifetime_ids"] = list(self.ancestor_lifetime_ids)
        if self.metadata:
            result["metadata"] = _thaw_json(self.metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LifetimeRecord:
        mapping = _require_mapping(value, name="LifetimeRecord")
        _validate_keys(
            mapping,
            name="LifetimeRecord",
            required=cls._REQUIRED_FIELDS,
            optional=cls._OPTIONAL_FIELDS,
        )
        return cls(
            lifetime_id=_nonempty_string(mapping["lifetime_id"], name="lifetime_id"),
            world_lineage_id=_nonempty_string(
                mapping["world_lineage_id"], name="world_lineage_id"
            ),
            environment_family=_nonempty_string(
                mapping["environment_family"], name="environment_family"
            ),
            environment_build_sha256=_sha256(
                mapping["environment_build_sha256"], name="environment_build_sha256"
            ),
            container_digest=_nonempty_string(
                mapping["container_digest"], name="container_digest"
            ),
            generator_config=_freeze_json_object(
                mapping["generator_config"], name="generator_config"
            ),
            seed_vector=_freeze_json_object(mapping["seed_vector"], name="seed_vector"),
            control_mapping_hash=_sha256(
                mapping["control_mapping_hash"], name="control_mapping_hash"
            ),
            qpc_frequency_hz=_integer(
                mapping["qpc_frequency_hz"],
                name="qpc_frequency_hz",
                minimum=1,
            ),
            graphics_config=_freeze_json_object(
                mapping["graphics_config"], name="graphics_config"
            ),
            source_type=_nonempty_string(mapping["source_type"], name="source_type"),
            source_session_id=_nonempty_string(
                mapping["source_session_id"],
                name="source_session_id",
            ),
            license_record_id=_nonempty_string(
                mapping["license_record_id"], name="license_record_id"
            ),
            split=_nonempty_string(mapping["split"], name="split"),
            player_pseudonym=_optional_nonempty_string(
                mapping.get("player_pseudonym"),
                name="player_pseudonym",
            ),
            ancestor_lifetime_ids=_string_tuple(
                mapping.get("ancestor_lifetime_ids", ()),
                name="ancestor_lifetime_ids",
            ),
            metadata=_freeze_json_object(mapping.get("metadata", {}), name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class SensorStorageLocator:
    """Non-model references used only to resolve stored sensor content."""

    rgb_ref: str
    audio_ref: str | None

    def __post_init__(self) -> None:
        _nonempty_string(self.rgb_ref, name="rgb_ref")
        _optional_nonempty_string(self.audio_ref, name="audio_ref")


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One timestamped observation/action transition in a lifetime."""

    _REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "lifetime_id",
            "episode_id",
            "step",
            "environment_tick",
            "frame_id",
            "capture_time_qpc",
            "observation_ready_qpc",
            "action_requested_qpc",
            "action_applied_qpc",
            "rgb_ref",
            "audio_ref",
            "requested_control",
            "applied_control",
            "reward_raw",
            "events",
            "terminated",
            "truncated",
            "dropped_frames",
            "snapshot_hash",
            "privileged_state_ref",
        }
    )
    _OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"metadata"})

    lifetime_id: str
    episode_id: str
    step: int
    environment_tick: int
    frame_id: int
    capture_time_qpc: int
    observation_ready_qpc: int
    action_requested_qpc: int
    action_applied_qpc: int
    rgb_ref: str
    audio_ref: str | None
    requested_control: GenericControl
    applied_control: GenericControl
    reward_raw: float
    events: tuple[str, ...] = ()
    terminated: bool = False
    truncated: bool = False
    dropped_frames: int = 0
    snapshot_hash: str | None = None
    privileged_state_ref: str | None = None
    metadata: JsonObject = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        for name in ("lifetime_id", "episode_id", "rgb_ref"):
            _nonempty_string(getattr(self, name), name=name)
        for name in (
            "step",
            "environment_tick",
            "frame_id",
            "capture_time_qpc",
            "observation_ready_qpc",
            "action_requested_qpc",
            "action_applied_qpc",
            "dropped_frames",
        ):
            _integer(getattr(self, name), name=name)
        if not (
            self.capture_time_qpc
            <= self.observation_ready_qpc
            <= self.action_requested_qpc
            <= self.action_applied_qpc
        ):
            raise ValueError(
                "QPC timestamps must satisfy capture <= observation ready "
                "<= action requested <= action applied"
            )
        _optional_nonempty_string(self.audio_ref, name="audio_ref")
        _validate_control(self.requested_control, name="requested_control")
        _validate_control(self.applied_control, name="applied_control")
        reward_raw = _number(self.reward_raw, name="reward_raw")
        object.__setattr__(self, "reward_raw", reward_raw)
        object.__setattr__(self, "events", _string_tuple(self.events, name="events"))
        _boolean(self.terminated, name="terminated")
        _boolean(self.truncated, name="truncated")
        _sha256(self.snapshot_hash, name="snapshot_hash", optional=True)
        _optional_nonempty_string(self.privileged_state_ref, name="privileged_state_ref")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "lifetime_id": self.lifetime_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "environment_tick": self.environment_tick,
            "frame_id": self.frame_id,
            "capture_time_qpc": self.capture_time_qpc,
            "observation_ready_qpc": self.observation_ready_qpc,
            "action_requested_qpc": self.action_requested_qpc,
            "action_applied_qpc": self.action_applied_qpc,
            "rgb_ref": self.rgb_ref,
            "audio_ref": self.audio_ref,
            "requested_control": self.requested_control.to_dict(),
            "applied_control": self.applied_control.to_dict(),
            "reward_raw": self.reward_raw,
            "events": list(self.events),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "dropped_frames": self.dropped_frames,
            "snapshot_hash": self.snapshot_hash,
            "privileged_state_ref": self.privileged_state_ref,
        }
        if self.metadata:
            result["metadata"] = _thaw_json(self.metadata)
        return result

    @property
    def sensor_storage_locator(self) -> SensorStorageLocator:
        """Return refs under a name that cannot be mistaken for model input."""

        return SensorStorageLocator(rgb_ref=self.rgb_ref, audio_ref=self.audio_ref)

    def materialize_model_observation(
        self,
        *,
        rgb_loader: Callable[[str], RgbFrame],
        audio_loader: Callable[[str], bytes],
        previous_control: GenericControl,
        lifetime_origin_qpc: int,
        qpc_frequency_hz: int,
    ) -> ModelObservation:
        """Resolve stored sensors into the canonical leakage-safe model input.

        ``requested_control`` and ``applied_control`` belong to this transition
        and are therefore labels at observation time.  A caller processing an
        ordered stream must supply the preceding transition's applied control.
        Absolute QPC values are converted to relative nanoseconds so machine
        uptime cannot become an accidental identity feature. The current step
        schema stores no provenance-labeled text, so recorded text_inputs are
        deliberately empty rather than reconstructed from privileged events.
        """

        if not callable(rgb_loader) or not callable(audio_loader):
            raise ValueError("rgb_loader and audio_loader must be callable")
        control = _validate_control(previous_control, name="previous_control")
        origin = _integer(lifetime_origin_qpc, name="lifetime_origin_qpc")
        frequency = _integer(qpc_frequency_hz, name="qpc_frequency_hz", minimum=1)
        if origin > self.capture_time_qpc:
            raise ValueError("lifetime_origin_qpc cannot follow capture_time_qpc")
        elapsed_ns = (
            (self.capture_time_qpc - origin) * 1_000_000_000 // frequency
        )
        observation_age_ns = (
            (self.observation_ready_qpc - self.capture_time_qpc)
            * 1_000_000_000
            // frequency
        )
        locator = self.sensor_storage_locator
        rgb = rgb_loader(locator.rgb_ref)
        audio = None if locator.audio_ref is None else audio_loader(locator.audio_ref)
        return ModelObservation(
            frame_id=self.frame_id,
            elapsed_ns=elapsed_ns,
            observation_age_ns=observation_age_ns,
            rgb=rgb,
            previous_control=control,
            audio_pcm_s16le=audio,
            text_inputs=(),
            dropped_frames=self.dropped_frames,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StepRecord:
        mapping = _require_mapping(value, name="StepRecord")
        _validate_keys(
            mapping,
            name="StepRecord",
            required=cls._REQUIRED_FIELDS,
            optional=cls._OPTIONAL_FIELDS,
        )
        return cls(
            lifetime_id=_nonempty_string(mapping["lifetime_id"], name="lifetime_id"),
            episode_id=_nonempty_string(mapping["episode_id"], name="episode_id"),
            step=_integer(mapping["step"], name="step"),
            environment_tick=_integer(mapping["environment_tick"], name="environment_tick"),
            frame_id=_integer(mapping["frame_id"], name="frame_id"),
            capture_time_qpc=_integer(mapping["capture_time_qpc"], name="capture_time_qpc"),
            observation_ready_qpc=_integer(
                mapping["observation_ready_qpc"], name="observation_ready_qpc"
            ),
            action_requested_qpc=_integer(
                mapping["action_requested_qpc"], name="action_requested_qpc"
            ),
            action_applied_qpc=_integer(
                mapping["action_applied_qpc"], name="action_applied_qpc"
            ),
            rgb_ref=_nonempty_string(mapping["rgb_ref"], name="rgb_ref"),
            audio_ref=_optional_nonempty_string(mapping["audio_ref"], name="audio_ref"),
            requested_control=_control_from_dict(
                mapping["requested_control"], name="requested_control"
            ),
            applied_control=_control_from_dict(
                mapping["applied_control"], name="applied_control"
            ),
            reward_raw=_number(mapping["reward_raw"], name="reward_raw"),
            events=_string_tuple(mapping["events"], name="events"),
            terminated=_boolean(mapping["terminated"], name="terminated"),
            truncated=_boolean(mapping["truncated"], name="truncated"),
            dropped_frames=_integer(mapping["dropped_frames"], name="dropped_frames"),
            snapshot_hash=_sha256(
                mapping["snapshot_hash"],
                name="snapshot_hash",
                optional=True,
            ),
            privileged_state_ref=_optional_nonempty_string(
                mapping["privileged_state_ref"], name="privileged_state_ref"
            ),
            metadata=_freeze_json_object(mapping.get("metadata", {}), name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class BranchRecord:
    """Verified counterfactual continuation from an exact simulator state."""

    _REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "branch_group_id",
            "root_snapshot_hash",
            "root_lifetime_id",
            "root_step",
            "branch_id",
            "rng_state_hash",
            "forced_control_sequence",
            "continuation_policy_id",
            "horizon_ticks",
            "future_frames_ref",
            "state_delta_ref",
            "events",
            "return",
            "replay_exact",
        }
    )
    _OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"metadata"})

    branch_group_id: str
    root_snapshot_hash: str
    root_lifetime_id: str
    root_step: int
    branch_id: int
    rng_state_hash: str
    forced_control_sequence: tuple[GenericControl, ...]
    continuation_policy_id: str
    horizon_ticks: int
    future_frames_ref: str
    state_delta_ref: str
    events: tuple[str, ...]
    return_value: float
    replay_exact: bool
    metadata: JsonObject = field(default_factory=_empty_json_object)

    def __post_init__(self) -> None:
        for name in (
            "branch_group_id",
            "root_lifetime_id",
            "continuation_policy_id",
            "future_frames_ref",
            "state_delta_ref",
        ):
            _nonempty_string(getattr(self, name), name=name)
        _sha256(self.root_snapshot_hash, name="root_snapshot_hash")
        _sha256(self.rng_state_hash, name="rng_state_hash")
        _integer(self.root_step, name="root_step")
        _integer(self.branch_id, name="branch_id")
        _integer(self.horizon_ticks, name="horizon_ticks", minimum=1)
        if not isinstance(self.forced_control_sequence, (list, tuple)):
            raise ValueError("forced_control_sequence must be an array of GenericControl")
        controls = tuple(
            _validate_control(control, name=f"forced_control_sequence[{index}]")
            for index, control in enumerate(self.forced_control_sequence)
        )
        if len(controls) > self.horizon_ticks:
            raise ValueError("forced_control_sequence cannot exceed horizon_ticks")
        object.__setattr__(self, "forced_control_sequence", controls)
        object.__setattr__(self, "events", _string_tuple(self.events, name="events"))
        return_value = _number(self.return_value, name="return_value")
        object.__setattr__(self, "return_value", return_value)
        _boolean(self.replay_exact, name="replay_exact")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_object(self.metadata, name="metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "branch_group_id": self.branch_group_id,
            "root_snapshot_hash": self.root_snapshot_hash,
            "root_lifetime_id": self.root_lifetime_id,
            "root_step": self.root_step,
            "branch_id": self.branch_id,
            "rng_state_hash": self.rng_state_hash,
            "forced_control_sequence": [
                control.to_dict() for control in self.forced_control_sequence
            ],
            "continuation_policy_id": self.continuation_policy_id,
            "horizon_ticks": self.horizon_ticks,
            "future_frames_ref": self.future_frames_ref,
            "state_delta_ref": self.state_delta_ref,
            "events": list(self.events),
            "return": self.return_value,
            "replay_exact": self.replay_exact,
        }
        if self.metadata:
            result["metadata"] = _thaw_json(self.metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BranchRecord:
        mapping = _require_mapping(value, name="BranchRecord")
        _validate_keys(
            mapping,
            name="BranchRecord",
            required=cls._REQUIRED_FIELDS,
            optional=cls._OPTIONAL_FIELDS,
        )
        raw_controls = mapping["forced_control_sequence"]
        if not isinstance(raw_controls, (list, tuple)):
            raise ValueError("forced_control_sequence must be an array")
        return cls(
            branch_group_id=_nonempty_string(
                mapping["branch_group_id"], name="branch_group_id"
            ),
            root_snapshot_hash=_sha256(
                mapping["root_snapshot_hash"], name="root_snapshot_hash"
            ),
            root_lifetime_id=_nonempty_string(
                mapping["root_lifetime_id"], name="root_lifetime_id"
            ),
            root_step=_integer(mapping["root_step"], name="root_step"),
            branch_id=_integer(mapping["branch_id"], name="branch_id"),
            rng_state_hash=_sha256(
                mapping["rng_state_hash"],
                name="rng_state_hash",
            ),
            forced_control_sequence=tuple(
                _control_from_dict(control, name=f"forced_control_sequence[{index}]")
                for index, control in enumerate(raw_controls)
            ),
            continuation_policy_id=_nonempty_string(
                mapping["continuation_policy_id"], name="continuation_policy_id"
            ),
            horizon_ticks=_integer(mapping["horizon_ticks"], name="horizon_ticks", minimum=1),
            future_frames_ref=_nonempty_string(
                mapping["future_frames_ref"], name="future_frames_ref"
            ),
            state_delta_ref=_nonempty_string(mapping["state_delta_ref"], name="state_delta_ref"),
            events=_string_tuple(mapping["events"], name="events"),
            return_value=_number(mapping["return"], name="return"),
            replay_exact=_boolean(mapping["replay_exact"], name="replay_exact"),
            metadata=_freeze_json_object(mapping.get("metadata", {}), name="metadata"),
        )


Record: TypeAlias = LifetimeRecord | StepRecord | BranchRecord
