"""Deterministic JSON Lines I/O for canonical Pseudo-Brain records."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar, overload

from irene_brain.runtime.policy import Capability, ResourcePolicy
from irene_brain.types import GenericControl, ModelObservation, RgbFrame

from .records import BranchRecord, LifetimeRecord, Record, StepRecord


RecordT = TypeVar("RecordT", LifetimeRecord, StepRecord, BranchRecord)
Pathish = str | PathLike[str]


class JsonlRecordError(ValueError):
    """A JSONL row could not be decoded as the requested canonical record."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def canonical_json(record: Record) -> str:
    """Serialize one record in its byte-stable, compact JSON representation."""

    if not isinstance(record, (LifetimeRecord, StepRecord, BranchRecord)):
        raise TypeError("record must be LifetimeRecord, StepRecord, or BranchRecord")
    return json.dumps(
        record.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def append_record(
    path: Pathish,
    record: Record,
    *,
    policy: ResourcePolicy,
) -> None:
    """Append exactly one canonical row, creating parent directories as needed."""

    policy.require(Capability.ARTIFACT_WRITE)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = canonical_json(record) + "\n"
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(row)


def append_jsonl(
    path: Pathish,
    record: Record,
    *,
    policy: ResourcePolicy,
) -> None:
    """Append one record; explicit storage-format spelling of append_record."""

    append_record(path, record, policy=policy)


@overload
def read_records(
    path: Pathish,
    record_type: type[LifetimeRecord],
) -> Iterator[LifetimeRecord]: ...


@overload
def read_records(path: Pathish, record_type: type[StepRecord]) -> Iterator[StepRecord]: ...


@overload
def read_records(path: Pathish, record_type: type[BranchRecord]) -> Iterator[BranchRecord]: ...


def read_records(path: Pathish, record_type: type[RecordT]) -> Iterator[RecordT]:
    """Yield validated records and attach source line information to failures."""

    if record_type not in (LifetimeRecord, StepRecord, BranchRecord):
        raise TypeError("record_type must be LifetimeRecord, StepRecord, or BranchRecord")
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise JsonlRecordError(f"{source}:{line_number}: blank JSONL rows are invalid")
            try:
                decoded: Any = json.loads(
                    line,
                    object_pairs_hook=_object_without_duplicate_keys,
                    parse_constant=_reject_nonstandard_number,
                )
            except json.JSONDecodeError as error:
                raise JsonlRecordError(
                    f"{source}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            except ValueError as error:
                raise JsonlRecordError(f"{source}:{line_number}: {error}") from error
            if not isinstance(decoded, dict):
                raise JsonlRecordError(
                    f"{source}:{line_number}: record must be a JSON object"
                )
            try:
                yield record_type.from_dict(decoded)
            except (TypeError, ValueError, OverflowError) as error:
                raise JsonlRecordError(f"{source}:{line_number}: {error}") from error


@overload
def read_jsonl(
    path: Pathish,
    record_type: type[LifetimeRecord],
) -> Iterator[LifetimeRecord]: ...


@overload
def read_jsonl(path: Pathish, record_type: type[StepRecord]) -> Iterator[StepRecord]: ...


@overload
def read_jsonl(path: Pathish, record_type: type[BranchRecord]) -> Iterator[BranchRecord]: ...


def read_jsonl(path: Pathish, record_type: type[RecordT]) -> Iterator[RecordT]:
    """Read records; explicit storage-format spelling of read_records."""

    yield from read_records(path, record_type)


def append_lifetime(
    path: Pathish,
    record: LifetimeRecord,
    *,
    policy: ResourcePolicy,
) -> None:
    append_record(path, record, policy=policy)


def read_lifetimes(path: Pathish) -> Iterator[LifetimeRecord]:
    yield from read_records(path, LifetimeRecord)


def append_step(
    path: Pathish,
    record: StepRecord,
    *,
    policy: ResourcePolicy,
) -> None:
    append_record(path, record, policy=policy)


def read_steps(path: Pathish) -> Iterator[StepRecord]:
    yield from read_records(path, StepRecord)


def append_branch(
    path: Pathish,
    record: BranchRecord,
    *,
    policy: ResourcePolicy,
) -> None:
    append_record(path, record, policy=policy)


def read_branches(path: Pathish) -> Iterator[BranchRecord]:
    yield from read_records(path, BranchRecord)


def read_step_observations(
    path: Pathish,
    *,
    lifetime_manifests: Mapping[str, LifetimeRecord],
    rgb_loader: Callable[[str], RgbFrame],
    audio_loader: Callable[[str], bytes],
) -> Iterator[ModelObservation]:
    """Yield resolved canonical observations from complete ordered episodes.

    The current transition's controls and all supervision remain hidden.  The
    previous applied control is carried forward per lifetime/episode, matching
    the live :class:`~irene_brain.types.ModelObservation` contract. Lifetime
    manifests are mandatory so each row uses its own recorded QPC frequency;
    unknown lifetimes fail closed.
    """

    if not isinstance(lifetime_manifests, Mapping):
        raise TypeError("lifetime_manifests must be a mapping")
    manifests: dict[str, LifetimeRecord] = {}
    for lifetime_id, manifest in lifetime_manifests.items():
        if not isinstance(lifetime_id, str) or not isinstance(manifest, LifetimeRecord):
            raise TypeError("lifetime_manifests must map strings to LifetimeRecord")
        if lifetime_id != manifest.lifetime_id:
            raise ValueError("lifetime manifest key must match manifest.lifetime_id")
        manifests[lifetime_id] = manifest
    if not callable(rgb_loader) or not callable(audio_loader):
        raise TypeError("rgb_loader and audio_loader must be callable")

    previous_controls: dict[tuple[str, str], GenericControl] = {}
    origins: dict[tuple[str, str], int] = {}
    previous_records: dict[tuple[str, str], StepRecord] = {}
    terminal_streams: set[tuple[str, str]] = set()
    for record in read_steps(path):
        manifest = manifests.get(record.lifetime_id)
        if manifest is None:
            raise JsonlRecordError(
                f"{Path(path)}: unknown lifetime_id {record.lifetime_id!r}"
            )
        stream_key = (record.lifetime_id, record.episode_id)
        previous_record = previous_records.get(stream_key)
        if previous_record is None:
            if record.step != 0:
                raise JsonlRecordError(
                    f"{Path(path)}: episode {stream_key!r} must begin at step 0"
                )
            previous_control = GenericControl.neutral()
            origin = record.capture_time_qpc
            origins[stream_key] = origin
        else:
            if stream_key in terminal_streams:
                raise JsonlRecordError(
                    f"{Path(path)}: row follows terminal episode {stream_key!r}"
                )
            if record.step != previous_record.step + 1:
                raise JsonlRecordError(
                    f"{Path(path)}: episode {stream_key!r} steps must be contiguous"
                )
            if record.environment_tick <= previous_record.environment_tick:
                raise JsonlRecordError(
                    f"{Path(path)}: environment_tick must increase in {stream_key!r}"
                )
            if record.frame_id <= previous_record.frame_id:
                raise JsonlRecordError(
                    f"{Path(path)}: frame_id must increase in {stream_key!r}"
                )
            for timestamp_name in (
                "capture_time_qpc",
                "observation_ready_qpc",
                "action_requested_qpc",
                "action_applied_qpc",
            ):
                if getattr(record, timestamp_name) <= getattr(previous_record, timestamp_name):
                    raise JsonlRecordError(
                        f"{Path(path)}: {timestamp_name} must increase in {stream_key!r}"
                    )
            previous_control = previous_controls[stream_key]
            origin = origins[stream_key]

        observation = record.materialize_model_observation(
            rgb_loader=rgb_loader,
            audio_loader=audio_loader,
            previous_control=previous_control,
            lifetime_origin_qpc=origin,
            qpc_frequency_hz=manifest.qpc_frequency_hz,
        )
        previous_controls[stream_key] = record.applied_control
        previous_records[stream_key] = record
        if record.terminated or record.truncated:
            terminal_streams.add(stream_key)
        yield observation
