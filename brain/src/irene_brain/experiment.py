"""Immutable, reproducible experiment manifests.

The manifest deliberately contains only JSON-native scalar metadata.  Large
artifacts are referred to by SHA-256 digest so a run record stays small and
can be compared without reading checkpoints or datasets.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .runtime.policy import Capability, ResourcePolicy


JsonScalar: TypeAlias = str | int | float | bool | None

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_METADATA_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_PRECISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.:/-]{0,63}\Z")
_STATUSES = frozenset(
    {
        "created",
        "pending",
        "running",
        "completed",
        "failed",
        "aborted",
        "cancelled",
    }
)

_FIELDS = frozenset(
    {
        "run_id",
        "schema_version",
        "created_at",
        "code_hash",
        "data_hash",
        "environment_hash",
        "checkpoint_hash",
        "seed",
        "config_hash",
        "hardware",
        "runtime",
        "precision",
        "allow_gpu",
        "allow_capture",
        "allow_hid_output",
        "allow_background_threads",
        "allow_network",
        "allow_subprocess",
        "allow_artifact_write",
        "cpu_threads",
        "status",
        "metrics",
    }
)


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _require_hash(value: object, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    digest = _require_string(value, name=name)
    if _HASH_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase, 64-character SHA-256 digest")
    return digest


def _canonical_timestamp(value: object) -> str:
    timestamp = _require_string(value, name="created_at")
    candidate = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a UTC offset")
    canonical = parsed.astimezone(timezone.utc)
    return canonical.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_scalar(value: object, *, name: str) -> JsonScalar:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
        return 0.0 if value == 0.0 else value
    raise TypeError(f"{name} must be a JSON scalar")


def _freeze_metadata(value: object, *, name: str) -> Mapping[str, JsonScalar]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not value:
        raise ValueError(f"{name} cannot be empty")

    validated: dict[str, JsonScalar] = {}
    for raw_key, raw_value in value.items():
        key = _require_string(raw_key, name=f"{name} key")
        if _METADATA_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid {name} key: {key!r}")
        validated[key] = _validate_scalar(raw_value, name=f"{name}.{key}")
    return MappingProxyType(dict(sorted(validated.items())))


def _freeze_metrics(value: object) -> Mapping[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError("metrics must be a mapping")

    validated: dict[str, int | float] = {}
    for raw_key, raw_value in value.items():
        key = _require_string(raw_key, name="metrics key")
        if _METADATA_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(f"invalid metrics key: {key!r}")
        if type(raw_value) not in {int, float}:
            raise TypeError(f"metrics.{key} must be a number, not a boolean or string")
        if type(raw_value) is float and not isfinite(raw_value):
            raise ValueError(f"metrics.{key} must be finite")
        validated[key] = 0.0 if raw_value == 0.0 else raw_value
    return MappingProxyType(dict(sorted(validated.items())))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """A complete, immutable identity and result record for one run.

    Hash fields contain lowercase hexadecimal SHA-256 digests. ``hardware``
    and ``runtime`` are flat maps of JSON scalars so their canonical encoding
    is portable and unambiguous. A fresh run may use ``None`` for
    ``checkpoint_hash``; all other content hashes are required.
    """

    run_id: str
    schema_version: int
    created_at: str
    code_hash: str
    data_hash: str
    environment_hash: str
    checkpoint_hash: str | None
    seed: int
    config_hash: str
    hardware: Mapping[str, JsonScalar]
    runtime: Mapping[str, JsonScalar]
    precision: str
    allow_gpu: bool
    allow_capture: bool
    allow_hid_output: bool
    allow_background_threads: bool
    allow_network: bool
    allow_subprocess: bool
    allow_artifact_write: bool
    cpu_threads: int
    status: str
    metrics: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        run_id = _require_string(self.run_id, name="run_id")
        if _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(
                "run_id must be 1-128 safe characters: letters, digits, '.', '_', or '-'"
            )
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be integer 1")

        object.__setattr__(self, "created_at", _canonical_timestamp(self.created_at))
        object.__setattr__(self, "code_hash", _require_hash(self.code_hash, name="code_hash"))
        object.__setattr__(self, "data_hash", _require_hash(self.data_hash, name="data_hash"))
        object.__setattr__(
            self,
            "environment_hash",
            _require_hash(self.environment_hash, name="environment_hash"),
        )
        object.__setattr__(
            self,
            "checkpoint_hash",
            _require_hash(self.checkpoint_hash, name="checkpoint_hash", optional=True),
        )
        object.__setattr__(
            self,
            "config_hash",
            _require_hash(self.config_hash, name="config_hash"),
        )

        if type(self.seed) is not int:
            raise TypeError("seed must be an integer, not a boolean or string")
        if self.seed < 0 or self.seed > (2**63 - 1):
            raise ValueError("seed must be between 0 and 2^63 - 1")

        object.__setattr__(self, "hardware", _freeze_metadata(self.hardware, name="hardware"))
        object.__setattr__(self, "runtime", _freeze_metadata(self.runtime, name="runtime"))

        precision = _require_string(self.precision, name="precision")
        if _PRECISION_PATTERN.fullmatch(precision) is None:
            raise ValueError("precision must be a non-empty, portable identifier")

        for name in (
            "allow_gpu",
            "allow_capture",
            "allow_hid_output",
            "allow_background_threads",
            "allow_network",
            "allow_subprocess",
            "allow_artifact_write",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if type(self.cpu_threads) is not int:
            raise TypeError("cpu_threads must be an integer")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be at least one")

        status = _require_string(self.status, name="status")
        if status not in _STATUSES:
            allowed = ", ".join(sorted(_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")

        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))

    @classmethod
    def now(cls) -> str:
        """Return a canonical UTC timestamp suitable for ``created_at``."""

        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-native dictionary in schema order."""

        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "code_hash": self.code_hash,
            "data_hash": self.data_hash,
            "environment_hash": self.environment_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "hardware": dict(self.hardware),
            "runtime": dict(self.runtime),
            "precision": self.precision,
            "allow_gpu": self.allow_gpu,
            "allow_capture": self.allow_capture,
            "allow_hid_output": self.allow_hid_output,
            "allow_background_threads": self.allow_background_threads,
            "allow_network": self.allow_network,
            "allow_subprocess": self.allow_subprocess,
            "allow_artifact_write": self.allow_artifact_write,
            "cpu_threads": self.cpu_threads,
            "status": self.status,
            "metrics": dict(self.metrics),
        }

    def canonical_json(self) -> str:
        """Return the deterministic UTF-8 JSON representation of this manifest."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    @property
    def manifest_hash(self) -> str:
        """SHA-256 of the exact bytes written by :meth:`save`."""

        return sha256(self.canonical_bytes).hexdigest()

    @property
    def resource_policy(self) -> ResourcePolicy:
        return ResourcePolicy(
            allow_gpu=self.allow_gpu,
            allow_capture=self.allow_capture,
            allow_hid_output=self.allow_hid_output,
            allow_background_threads=self.allow_background_threads,
            allow_network=self.allow_network,
            allow_subprocess=self.allow_subprocess,
            allow_artifact_write=self.allow_artifact_write,
            cpu_threads=self.cpu_threads,
        )

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        policy: ResourcePolicy,
    ) -> Path:
        """Atomically write the canonical manifest, replacing ``path`` if present."""

        policy.require(Capability.ARTIFACT_WRITE)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.canonical_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            finally:
                raise
        return target

    def save_atomic(
        self,
        path: str | os.PathLike[str],
        *,
        policy: ResourcePolicy,
    ) -> Path:
        """Explicit alias for :meth:`save`, whose writes are always atomic."""

        return self.save(path, policy=policy)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentManifest:
        """Validate and construct a manifest from an exact schema-shaped mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("manifest must be a mapping")
        invalid_keys = [key for key in value if type(key) is not str]
        if invalid_keys:
            raise TypeError("manifest field names must be strings")
        keys = set(value)
        missing = sorted(_FIELDS - keys)
        extra = sorted(keys - _FIELDS)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if extra:
                details.append(f"unknown fields: {', '.join(extra)}")
            raise ValueError("; ".join(details))
        return cls(
            run_id=value["run_id"],
            schema_version=value["schema_version"],
            created_at=value["created_at"],
            code_hash=value["code_hash"],
            data_hash=value["data_hash"],
            environment_hash=value["environment_hash"],
            checkpoint_hash=value["checkpoint_hash"],
            seed=value["seed"],
            config_hash=value["config_hash"],
            hardware=value["hardware"],
            runtime=value["runtime"],
            precision=value["precision"],
            allow_gpu=value["allow_gpu"],
            allow_capture=value["allow_capture"],
            allow_hid_output=value["allow_hid_output"],
            allow_background_threads=value["allow_background_threads"],
            allow_network=value["allow_network"],
            allow_subprocess=value["allow_subprocess"],
            allow_artifact_write=value["allow_artifact_write"],
            cpu_threads=value["cpu_threads"],
            status=value["status"],
            metrics=value["metrics"],
        )

    @classmethod
    def from_json(cls, value: str) -> ExperimentManifest:
        """Parse JSON while rejecting duplicate keys and non-finite numbers."""

        if type(value) is not str:
            raise TypeError("manifest JSON must be a string")
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
        if not isinstance(parsed, dict):
            raise TypeError("manifest JSON root must be an object")
        return cls.from_dict(parsed)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ExperimentManifest:
        """Read and strictly validate a UTF-8 manifest file."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["ExperimentManifest", "JsonScalar"]
