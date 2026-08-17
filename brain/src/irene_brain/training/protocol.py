"""Dependency-free interfaces between orchestration, data, and ML runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Iterator, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """Detached scalar results for one complete optimizer boundary."""

    loss: float
    metrics: Mapping[str, float]
    samples: int

    def __post_init__(self) -> None:
        if type(self.loss) not in {int, float} or not isfinite(float(self.loss)):
            raise ValueError("loss must be a finite number")
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("samples must be a positive integer")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics must be a mapping")
        frozen: dict[str, float] = {}
        for name, value in self.metrics.items():
            if type(name) is not str or not name:
                raise ValueError("metric names must be non-empty strings")
            if type(value) not in {int, float} or not isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
            frozen[name] = 0.0 if float(value) == 0.0 else float(value)
        object.__setattr__(self, "loss", float(self.loss))
        object.__setattr__(self, "metrics", MappingProxyType(dict(sorted(frozen.items()))))


@runtime_checkable
class DeterministicBatchSource(Protocol):
    """A finite source whose epoch order is a pure function of epoch number."""

    @property
    def manifest_sha256(self) -> str: ...

    def batches_per_epoch(self, *, split: str, batch_size: int) -> int: ...

    def iter_batches(
        self,
        *,
        split: str,
        epoch: int,
        start_batch: int,
        batch_size: int,
        max_batches: int | None = None,
    ) -> Iterator[object]: ...


@runtime_checkable
class TrainingSystem(Protocol):
    """One model/optimizer pair consumed by the generic trainer.

    ``train_optimizer_step`` owns gradient accumulation and must return only
    after ``optimizer.step`` and ``zero_grad`` have completed. That invariant
    makes every trainer checkpoint an optimizer-boundary checkpoint.
    """

    @property
    def runtime_fingerprint(self) -> Mapping[str, str | int | bool]: ...

    def train_optimizer_step(
        self,
        microbatches: Sequence[object],
    ) -> TrainingStepResult: ...

    def evaluate_batch(self, batch: object) -> TrainingStepResult: ...

    def checkpoint_state(self) -> Mapping[str, object]: ...

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None: ...

    def capture_rng_state(self) -> Mapping[str, object]: ...

    def restore_rng_state(self, state: Mapping[str, object]) -> None: ...


@runtime_checkable
class StagedTrainingSystem(TrainingSystem, Protocol):
    """Additional exact-resume contract used only by schema-3 training."""

    @property
    def active_stage_index(self) -> int: ...

    @property
    def optimizer_parameter_names(self) -> tuple[str, ...]: ...

    def transition_to_stage(
        self,
        stage_index: int,
        *,
        invariance_batches: Sequence[object],
        entry_gate_report_sha256: str,
        entry_gate_passed: bool,
        entry_gate_step: int,
    ) -> Mapping[str, object]: ...

    def prepare_stage_for_resume(self, stage_state: Mapping[str, object]) -> None: ...

    def stage_checkpoint_state(
        self,
        *,
        global_optimizer_step: int,
    ) -> Mapping[str, object]: ...

    def validate_stage_invariance(
        self,
        *,
        invariance_batches: Sequence[object],
    ) -> Mapping[str, object]: ...

    def bind_completion_gate(
        self,
        *,
        report_sha256: str,
        passed: bool,
        global_optimizer_step: int,
    ) -> None: ...

    def restore_model_for_evaluation(
        self,
        system_state: Mapping[str, object],
        stage_state: Mapping[str, object],
    ) -> None: ...


__all__ = [
    "DeterministicBatchSource",
    "TrainingStepResult",
    "StagedTrainingSystem",
    "TrainingSystem",
]
