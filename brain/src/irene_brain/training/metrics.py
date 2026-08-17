"""Canonical, capability-gated JSONL metric records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..runtime.policy import Capability, ResourcePolicy


@dataclass(frozen=True, slots=True)
class MetricRecord:
    step: int
    epoch: int
    split: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("metrics must be a non-empty mapping")
        values: dict[str, float] = {}
        for name, value in self.metrics.items():
            if type(name) is not str or not name:
                raise ValueError("metric names must be non-empty strings")
            if type(value) not in {int, float} or not isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
            values[name] = 0.0 if float(value) == 0.0 else float(value)
        object.__setattr__(self, "metrics", MappingProxyType(dict(sorted(values.items()))))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "step": self.step,
            "epoch": self.epoch,
            "split": self.split,
            "metrics": dict(self.metrics),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class JsonlMetricWriter:
    """Append and fsync each record; never opens a file before authorization."""

    def __init__(self, path: str | os.PathLike[str], *, policy: ResourcePolicy) -> None:
        if not isinstance(policy, ResourcePolicy):
            raise ValueError("policy must be a ResourcePolicy")
        policy.require(Capability.ARTIFACT_WRITE)
        self.path = Path(path)
        self.policy = policy

    def append(self, record: MetricRecord) -> None:
        if not isinstance(record, MetricRecord):
            raise ValueError("record must be a MetricRecord")
        self.policy.require(Capability.ARTIFACT_WRITE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (record.canonical_json + "\n").encode("utf-8")
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


__all__ = ["JsonlMetricWriter", "MetricRecord"]
