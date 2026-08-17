"""Atomic, integrity-checked optimizer-boundary checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping

from ..runtime.policy import Capability, ResourcePolicy


def _hash(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


@dataclass(frozen=True, slots=True)
class TrainerCursor:
    epoch: int = 0
    next_batch: int = 0
    optimizer_step: int = 0

    def __post_init__(self) -> None:
        for name in ("epoch", "next_batch", "optimizer_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "optimizer_step": self.optimizer_step,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrainerCursor:
        if not isinstance(value, Mapping) or set(value) != {
            "epoch",
            "next_batch",
            "optimizer_step",
        }:
            raise ValueError("checkpoint cursor has incompatible fields")
        return cls(
            epoch=value["epoch"],
            next_batch=value["next_batch"],
            optimizer_step=value["optimizer_step"],
        )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    cursor: TrainerCursor
    system_state: Mapping[str, object]
    rng_state: Mapping[str, object]
    checkpoint_sha256: str
    stage_state: Mapping[str, object] | None = None
    trainer_state: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cursor, TrainerCursor):
            raise ValueError("cursor must be a TrainerCursor")
        if not isinstance(self.system_state, Mapping):
            raise ValueError("system_state must be a mapping")
        if not isinstance(self.rng_state, Mapping):
            raise ValueError("rng_state must be a mapping")
        _hash(self.checkpoint_sha256, name="checkpoint_sha256")
        object.__setattr__(self, "system_state", MappingProxyType(dict(self.system_state)))
        object.__setattr__(self, "rng_state", MappingProxyType(dict(self.rng_state)))
        if self.stage_state is not None:
            normalized = _stage_state(self.stage_state)
            object.__setattr__(self, "stage_state", MappingProxyType(normalized))
        if self.trainer_state is not None:
            normalized_trainer = _trainer_state(self.trainer_state)
            object.__setattr__(
                self,
                "trainer_state",
                MappingProxyType(normalized_trainer),
            )


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    cursor: TrainerCursor,
    system_state: Mapping[str, object],
    rng_state: Mapping[str, object],
    config_sha256: str,
    data_sha256: str,
    code_sha256: str,
    runtime_fingerprint: Mapping[str, str | int | bool],
    policy: ResourcePolicy,
    stage_state: Mapping[str, object] | None = None,
    trainer_state: Mapping[str, object] | None = None,
) -> str:
    """Atomically save one trusted internal checkpoint without replacing a target."""

    if not isinstance(policy, ResourcePolicy):
        raise ValueError("policy must be a ResourcePolicy")
    policy.require(Capability.ARTIFACT_WRITE)
    if not isinstance(cursor, TrainerCursor):
        raise ValueError("cursor must be a TrainerCursor")
    if not isinstance(system_state, Mapping) or not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint states must be mappings")
    for name, value in (
        ("config_sha256", config_sha256),
        ("data_sha256", data_sha256),
        ("code_sha256", code_sha256),
    ):
        _hash(value, name=name)
    fingerprint = _fingerprint(runtime_fingerprint)
    if (stage_state is None) != (trainer_state is None):
        raise ValueError("staged checkpoints require both stage_state and trainer_state")
    payload: dict[str, object] = {
        "schema_version": 1 if stage_state is None else 2,
        "cursor": cursor.to_dict(),
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "code_sha256": code_sha256,
        "runtime_fingerprint": fingerprint,
        "system_state": dict(system_state),
        "rng_state": dict(rng_state),
    }
    if stage_state is not None:
        payload["stage_state"] = _stage_state(stage_state)
        payload["trainer_state"] = _trainer_state(trainer_state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        torch = _torch()
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Both paths share a directory, so linking publishes the completed inode
        # atomically while failing with FileExistsError if any writer already
        # owns the target name. Replacing here could destroy a resume boundary.
        os.link(temporary, target)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(target)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_config_sha256: str,
    expected_data_sha256: str,
    expected_code_sha256: str,
    expected_runtime_fingerprint: Mapping[str, str | int | bool],
    expected_checkpoint_sha256: str | None = None,
) -> LoadedCheckpoint:
    """Load with restricted unpickling and fail closed on any run mismatch."""

    for name, value in (
        ("expected_config_sha256", expected_config_sha256),
        ("expected_data_sha256", expected_data_sha256),
        ("expected_code_sha256", expected_code_sha256),
    ):
        _hash(value, name=name)
    target = Path(path)
    if expected_checkpoint_sha256 is not None:
        _hash(expected_checkpoint_sha256, name="expected_checkpoint_sha256")
    torch = _torch()
    try:
        before = os.lstat(target)
        _require_plain_checkpoint(before, name="checkpoint path")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as source, tempfile.SpooledTemporaryFile(
            max_size=64 * 1024 * 1024,
            mode="w+b",
        ) as snapshot:
            opened = os.fstat(source.fileno())
            _require_plain_checkpoint(opened, name="opened checkpoint")
            _require_same_checkpoint(before, opened, name="checkpoint changed while opening")
            digest = sha256()
            while block := source.read(1024 * 1024):
                digest.update(block)
                snapshot.write(block)
            after_read = os.fstat(source.fileno())
            _require_same_checkpoint(
                opened,
                after_read,
                name="checkpoint changed while being read",
                include_times=True,
            )
            after_path = os.lstat(target)
            _require_plain_checkpoint(after_path, name="checkpoint path after read")
            _require_same_checkpoint(
                opened,
                after_path,
                name="checkpoint path changed while being read",
            )
            actual_digest = digest.hexdigest()
            if (
                expected_checkpoint_sha256 is not None
                and actual_digest != expected_checkpoint_sha256
            ):
                raise ValueError(
                    "checkpoint file digest does not match the expected SHA-256"
                )
            snapshot.seek(0)
            payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "this PyTorch version lacks restricted weights_only checkpoint loading"
        ) from error
    except OSError as error:
        raise ValueError(f"checkpoint could not be opened as one stable regular file: {error}") from error
    common_required = {
        "schema_version",
        "cursor",
        "config_sha256",
        "data_sha256",
        "code_sha256",
        "runtime_fingerprint",
        "system_state",
        "rng_state",
    }
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint envelope has incompatible fields")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("unsupported checkpoint schema version")
    required = common_required | (
        {"stage_state", "trainer_state"} if schema_version == 2 else set()
    )
    if set(payload) != required:
        raise ValueError("checkpoint envelope has incompatible fields")
    comparisons = (
        ("configuration", payload["config_sha256"], expected_config_sha256),
        ("dataset", payload["data_sha256"], expected_data_sha256),
        ("source code", payload["code_sha256"], expected_code_sha256),
        (
            "runtime",
            _fingerprint(payload["runtime_fingerprint"]),
            _fingerprint(expected_runtime_fingerprint),
        ),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(f"checkpoint {name} identity is incompatible with this run")
    return LoadedCheckpoint(
        cursor=TrainerCursor.from_dict(payload["cursor"]),
        system_state=payload["system_state"],
        rng_state=payload["rng_state"],
        checkpoint_sha256=actual_digest,
        stage_state=(
            _stage_state(payload["stage_state"])
            if schema_version == 2
            else None
        ),
        trainer_state=(
            _trainer_state(payload["trainer_state"])
            if schema_version == 2
            else None
        ),
    )


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_plain_checkpoint(value: os.stat_result, *, name: str) -> None:
    reparse = getattr(value, "st_file_attributes", 0) & getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0,
    )
    if not stat.S_ISREG(value.st_mode) or reparse:
        raise ValueError(f"{name} must be a regular non-reparse file")
    if value.st_nlink != 1:
        raise ValueError(f"{name} must have exactly one hard link")


def _require_same_checkpoint(
    expected: os.stat_result,
    actual: os.stat_result,
    *,
    name: str,
    include_times: bool = False,
) -> None:
    fields = ["st_dev", "st_ino", "st_mode", "st_nlink", "st_size"]
    if include_times:
        fields.extend(("st_mtime_ns", "st_ctime_ns"))
    if any(getattr(expected, field) != getattr(actual, field) for field in fields):
        raise ValueError(name)


def source_tree_sha256(root: str | os.PathLike[str]) -> str:
    """Hash relative paths and bytes of all Python sources under ``root``."""

    base = Path(root).resolve()
    files = sorted(base.rglob("*.py"), key=lambda item: item.relative_to(base).as_posix())
    if not files:
        raise ValueError("source tree contains no Python files")
    digest = sha256(b"IRENESOURCE\x01")
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _fingerprint(value: object) -> dict[str, str | int | bool]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("runtime_fingerprint must be a non-empty mapping")
    result: dict[str, str | int | bool] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValueError("runtime fingerprint names must be non-empty strings")
        if type(item) not in {str, int, bool}:
            raise ValueError("runtime fingerprint values must be strings, integers, or booleans")
        result[key] = item
    return dict(sorted(result.items()))


def _stage_state(value: object) -> dict[str, object]:
    """Normalize a JSON-like staged-training envelope without coercion."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError("stage_state must be a non-empty mapping")

    def normalize(item: object, *, name: str) -> object:
        if item is None or type(item) in {str, int, bool}:
            return item
        if isinstance(item, (list, tuple)):
            return [
                normalize(child, name=f"{name}[{index}]")
                for index, child in enumerate(item)
            ]
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            for key, child in item.items():
                if type(key) is not str or not key:
                    raise ValueError(f"{name} keys must be non-empty strings")
                if key in result:
                    raise ValueError(f"{name} contains a duplicate key")
                result[key] = normalize(child, name=f"{name}.{key}")
            return dict(sorted(result.items()))
        raise ValueError(
            f"{name} values must be JSON-like strings, integers, booleans, or null"
        )

    return normalize(value, name="stage_state")  # type: ignore[return-value]


def _trainer_state(value: object) -> dict[str, object]:
    required = {
        "schema_version",
        "metrics_byte_length",
        "metrics_record_count",
        "metrics_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("checkpoint trainer_state has incompatible fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("checkpoint trainer_state schema_version must be integer 1")
    for name in ("metrics_byte_length", "metrics_record_count"):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError(f"checkpoint trainer_state {name} must be positive")
    _hash(value["metrics_sha256"], name="metrics_sha256")
    return dict(value)


def _torch() -> object:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to read or write training checkpoints") from error
    return torch


__all__ = [
    "LoadedCheckpoint",
    "TrainerCursor",
    "file_sha256",
    "load_checkpoint",
    "save_checkpoint",
    "source_tree_sha256",
]
