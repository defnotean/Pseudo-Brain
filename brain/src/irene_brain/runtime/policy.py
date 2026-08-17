"""Fail-closed resource capabilities for play-safe development."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping
import tomllib


class Capability(StrEnum):
    GPU = "gpu"
    CAPTURE = "capture"
    HID_OUTPUT = "hid_output"
    BACKGROUND_THREADS = "background_threads"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    ARTIFACT_WRITE = "artifact_write"


class ResourceDenied(RuntimeError):
    """Raised before a disabled resource is initialized."""


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Capabilities default to denied and must be deliberately armed."""

    allow_gpu: bool = False
    allow_capture: bool = False
    allow_hid_output: bool = False
    allow_background_threads: bool = False
    allow_network: bool = False
    allow_subprocess: bool = False
    allow_artifact_write: bool = False
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        bool_fields = (
            self.allow_gpu,
            self.allow_capture,
            self.allow_hid_output,
            self.allow_background_threads,
            self.allow_network,
            self.allow_subprocess,
            self.allow_artifact_write,
        )
        if any(not isinstance(value, bool) for value in bool_fields):
            raise ValueError("resource capability values must be booleans")
        if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int):
            raise ValueError("cpu_threads must be an integer")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be at least one")

    @classmethod
    def play_safe(cls) -> ResourcePolicy:
        return cls()

    @classmethod
    def from_toml(cls, path: str | Path) -> ResourcePolicy:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        experiment = _mapping(raw.get("experiment", {}), name="experiment")
        runtime = _mapping(raw.get("runtime", {}), name="runtime")
        device = experiment.get("device", "cpu")
        if not isinstance(device, str):
            raise ValueError("experiment.device must be a string")
        allow_gpu = _boolean(experiment.get("allow_gpu", False), name="allow_gpu")
        if device.casefold() != "cpu" and not allow_gpu:
            raise ValueError("a non-CPU device requires allow_gpu=true")
        allow_capture = _boolean(
            experiment.get("allow_capture", False),
            name="allow_capture",
        )
        allow_artifact_write = _boolean(
            runtime.get("write_artifacts", False),
            name="write_artifacts",
        )
        record_video = _boolean(
            runtime.get("record_video", False),
            name="record_video",
        )
        if record_video and not allow_capture:
            raise ValueError("record_video=true requires allow_capture=true")
        if record_video and not allow_artifact_write:
            raise ValueError("record_video=true requires write_artifacts=true")
        return cls(
            allow_gpu=allow_gpu,
            allow_capture=allow_capture,
            allow_hid_output=_boolean(
                experiment.get("allow_hid_output", False),
                name="allow_hid_output",
            ),
            allow_background_threads=_boolean(
                experiment.get("allow_background_threads", False),
                name="allow_background_threads",
            ),
            allow_network=_boolean(
                experiment.get("allow_network", False),
                name="allow_network",
            ),
            allow_subprocess=_boolean(
                experiment.get("allow_subprocess", False),
                name="allow_subprocess",
            ),
            allow_artifact_write=allow_artifact_write,
            cpu_threads=_integer(
                experiment.get("cpu_threads", 1),
                name="cpu_threads",
            ),
        )

    def allows(self, capability: Capability) -> bool:
        mapping = {
            Capability.GPU: self.allow_gpu,
            Capability.CAPTURE: self.allow_capture,
            Capability.HID_OUTPUT: self.allow_hid_output,
            Capability.BACKGROUND_THREADS: self.allow_background_threads,
            Capability.NETWORK: self.allow_network,
            Capability.SUBPROCESS: self.allow_subprocess,
            Capability.ARTIFACT_WRITE: self.allow_artifact_write,
        }
        return mapping[capability]

    def require(self, capability: Capability) -> None:
        if not self.allows(capability):
            raise ResourceDenied(
                f"{capability.value} is disabled by the active resource policy"
            )

    def process_environment(self) -> dict[str, str]:
        """Environment limits inherited by numerical libraries."""

        thread_count = str(self.cpu_threads)
        environment = {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
            "VECLIB_MAXIMUM_THREADS": thread_count,
            "BLIS_NUM_THREADS": thread_count,
            "RAYON_NUM_THREADS": thread_count,
            "TOKENIZERS_PARALLELISM": "false",
        }
        if not self.allow_gpu:
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
            environment["HIP_VISIBLE_DEVICES"] = "-1"
            environment["ROCR_VISIBLE_DEVICES"] = "-1"
        return environment


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a table")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
