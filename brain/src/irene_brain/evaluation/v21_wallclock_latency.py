"""Fail-closed wall-clock qualification for a V2.1i live policy checkpoint.

This module measures CPU execution only.  It does not open any dataset, train a
model, initialize CUDA, launch a GUI, or claim capture/HID/display latency.
Each measured tick records ``perf_counter_ns`` milestones around the exact
``OutcomeAwareCoreV2MazePolicy`` model call and in-process environment loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import BinaryIO, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn

from ..environments.maze_chase import MazeChaseEnv
from ..types import GenericControl, Observation, StepOutcome
from ..v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model, FeatureFlags
from ..v2.maze_policy import OutcomeAwareCoreV2MazePolicy


SCHEMA_VERSION = 1
QUALIFICATION_ID = "v21i-cpu-wallclock-live-policy-v1"
POLICY_IDENTITY = "irene.core_v2.outcome_aware_maze_policy.v1"
MEASUREMENT_KIND = "cpu_perf_counter_ns_in_process"
TICK_PERIOD_NS_60HZ = 16_666_667
MODEL_P99_LIMIT_NS = 8_000_000
LOOP_P99_LIMIT_NS = 12_000_000
LOOP_P999_LIMIT_NS = TICK_PERIOD_NS_60HZ
MAXIMUM_DEADLINE_MISS_RATE = 0.001
WARMUP_TICKS = 100
MEASURED_TICKS = 5_000
REPETITIONS = 3
EPISODE_MAX_TICKS = 512
EPISODE_SEED_BASE = 67_108_864
EPISODE_SEED_STRIDE = 1_048_576
MINIMUM_EPISODE_ROTATIONS = (WARMUP_TICKS + MEASURED_TICKS) // EPISODE_MAX_TICKS
DEVELOPMENT_ENVELOPE = "development_v1"
POST_DGX_ENVELOPE = "post_dgx_v1"
POST_DGX_MODE = "v21i_bounded_dgx_candidate_v1"
POST_DGX_CLASSIFICATION = "bounded_dgx_candidate_requires_post_training_cpu_qual"
POST_DGX_RELEASE_QUALIFICATION_ID = "v21i-post-dgx-cpu-qual-release-v1"
POST_DGX_RELEASE_STATUS = "passed"
REQUIRED_CPU_QUAL_MODEL_SEEDS = (44, 45, 46, 47, 48)

_DEVELOPMENT_CHECKPOINT_KEYS = {
    "calibration_installation_audit",
    "calibration_mode",
    "classification",
    "config",
    "dataset_partitions",
    "development_gate",
    "device",
    "expected_calibration_bindings",
    "finite_state_audit",
    "flags",
    "hazard_calibration",
    "mode",
    "model_seed",
    "model_state_dict",
    "schema_version",
    "source_bundle",
    "state_sha256",
    "threads",
    "training",
    "upstream_uncalibrated_checkpoint",
}
_POST_DGX_CHECKPOINT_KEYS = _DEVELOPMENT_CHECKPOINT_KEYS | {"training_runtime"}
_POST_DGX_TRAINING_RUNTIME_KEYS = {
    "accelerator",
    "authorization_receipt_sha256",
    "bounded_schedule_sha256",
    "run_artifact_sha256",
}
_POST_DGX_RELEASE_BODY_KEYS = {
    "authorization",
    "checkpoint",
    "classification",
    "cpu_qualification",
    "dgx_lineage",
    "post_dgx_verification",
}
_POST_DGX_RELEASE_CHECKPOINT_KEYS = {
    "checkpoint_sha256",
    "feature_flags_sha256",
    "hazard_calibration_sha256",
    "model_config_sha256",
    "model_seed",
    "source_bundle_sha256",
    "state_sha256",
}
_POST_DGX_CPU_QUAL_KEYS = {
    "all_candidates_passed",
    "evaluation_artifact_sha256",
    "observed_passes",
    "opening_receipt_sha256",
    "passed_model_seeds",
    "preregistration_sha256",
    "qualification_id",
    "required_model_seeds",
    "required_passes",
    "test_split_opened",
}
_POST_DGX_VERIFICATION_KEYS = {
    "all_metrics_finite",
    "artifact_sha256",
    "calibration_passed",
    "causal_shuffle_passed",
    "hazard_quality_passed",
    "noncollapse_passed",
    "preservation_passed",
    "qualification_id",
}

_SOURCE_BUNDLE_PIPELINE_FILES = (
    "brain/scripts/run_provenance.py",
    "brain/scripts/v21i_development_runner.py",
)
_EVALUATOR_BUNDLE_FILES = (
    "brain/scripts/v21i_wallclock_latency_qualification.py",
    "brain/src/irene_brain/environments/maze_chase.py",
    "brain/src/irene_brain/evaluation/v21_wallclock_latency.py",
    "brain/src/irene_brain/training/objective.py",
    "brain/src/irene_brain/types.py",
    "brain/src/irene_brain/v2/config.py",
    "brain/src/irene_brain/v2/core.py",
    "brain/src/irene_brain/v2/maze_policy.py",
    "brain/src/irene_brain/v2/outcome_model.py",
    "brain/src/irene_brain/v2/state.py",
)


class WallClockQualificationError(RuntimeError):
    """A checkpoint, runtime, measurement, or artifact invariant failed."""


class WallClockQualificationFailed(WallClockQualificationError):
    """The benchmark completed and published evidence, but a gate failed."""

    def __init__(self, report: "WallClockQualificationReport") -> None:
        super().__init__("wall-clock latency qualification failed")
        self.report = report


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WallClockQualificationError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = sha256(domain)
    digest.update(_canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    digest = sha256(b"IRV21ISTATE\x01")
    for name, value in sorted(state_dict.items()):
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise WallClockQualificationError("state_dict must map names to tensors")
        tensor = value.detach().to(device="cpu").contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(2, "big"))
        digest.update(encoded_dtype)
        digest.update(len(tensor.shape).to_bytes(2, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _bundle(
    project_root: Path,
    relative_paths: Sequence[str],
    *,
    domain: bytes,
) -> dict[str, object]:
    digest = sha256(domain)
    files: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        path = project_root / relative
        if not path.is_file():
            raise WallClockQualificationError(f"source bundle file is missing: {relative}")
        file_sha = _file_sha256(path)
        files[relative] = file_sha
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha))
    return {"schema_version": 1, "sha256": digest.hexdigest(), "files": files}


def _training_source_bundle(project_root: Path) -> dict[str, object]:
    package_root = project_root / "brain" / "src" / "irene_brain"
    package_files = (
        path.relative_to(project_root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file()
    )
    return _bundle(
        project_root,
        tuple({*_SOURCE_BUNDLE_PIPELINE_FILES, *package_files}),
        domain=b"IRV21IDEVSOURCE\x01",
    )


def _evaluator_bundle(project_root: Path) -> dict[str, object]:
    return _bundle(
        project_root,
        _EVALUATOR_BUNDLE_FILES,
        domain=b"IRV21IWALLCLOCKEVALUATOR\x01",
    )


@dataclass(frozen=True, slots=True)
class WallClockQualificationConfig:
    """Frozen production run shape and thresholds."""

    warmup_ticks: int = WARMUP_TICKS
    measured_ticks: int = MEASURED_TICKS
    repetitions: int = REPETITIONS
    tick_period_ns: int = TICK_PERIOD_NS_60HZ
    model_p99_limit_ns: int = MODEL_P99_LIMIT_NS
    loop_p99_limit_ns: int = LOOP_P99_LIMIT_NS
    loop_p999_limit_ns: int = LOOP_P999_LIMIT_NS
    maximum_deadline_miss_rate: float = MAXIMUM_DEADLINE_MISS_RATE
    episode_max_ticks: int = EPISODE_MAX_TICKS
    episode_seed_base: int = EPISODE_SEED_BASE
    episode_seed_stride: int = EPISODE_SEED_STRIDE

    def __post_init__(self) -> None:
        expected = (
            WARMUP_TICKS,
            MEASURED_TICKS,
            REPETITIONS,
            TICK_PERIOD_NS_60HZ,
            MODEL_P99_LIMIT_NS,
            LOOP_P99_LIMIT_NS,
            LOOP_P999_LIMIT_NS,
            MAXIMUM_DEADLINE_MISS_RATE,
            EPISODE_MAX_TICKS,
            EPISODE_SEED_BASE,
            EPISODE_SEED_STRIDE,
        )
        observed = (
            self.warmup_ticks,
            self.measured_ticks,
            self.repetitions,
            self.tick_period_ns,
            self.model_p99_limit_ns,
            self.loop_p99_limit_ns,
            self.loop_p999_limit_ns,
            self.maximum_deadline_miss_rate,
            self.episode_max_ticks,
            self.episode_seed_base,
            self.episode_seed_stride,
        )
        if observed != expected:
            raise WallClockQualificationError(
                "wall-clock qualification configuration is frozen"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _workload_record(config: WallClockQualificationConfig) -> dict[str, object]:
    return {
        "dataset_access": "none",
        "device": "cpu",
        "environment": {
            "extra_loops": 16,
            "ghost_count": 5,
            "ghost_elroy": False,
            "ghost_period": 1,
            "ghost_rule": "direct",
            "input_delay_ticks": 0,
            "max_ticks_per_episode": config.episode_max_ticks,
            "player_period": 1,
            "sticky_direction": False,
        },
        "episode_namespace": "LATENCY-WALLCLOCK (not a dataset split)",
        "episode_seed_base": config.episode_seed_base,
        "episode_seed_stride": config.episode_seed_stride,
        "measured_ticks_per_repetition": config.measured_ticks,
        "measurement_kind": MEASUREMENT_KIND,
        "policy_identity": POLICY_IDENTITY,
        "repetitions": config.repetitions,
        "terminal_handling": "reset environment and policy before loop-finished milestone",
        "tick_period_ns": config.tick_period_ns,
        "warmup_ticks_per_repetition": config.warmup_ticks,
    }


def wallclock_workload_sha256() -> str:
    """Return the immutable digest live qualification must bind."""

    return _json_sha256(
        _workload_record(WallClockQualificationConfig()),
        domain=b"IRV21IWALLCLOCKWORKLOAD\x01",
    )


def wallclock_config_sha256() -> str:
    """Return the digest of the frozen run shape and thresholds."""

    return _json_sha256(
        WallClockQualificationConfig().to_dict(),
        domain=b"IRV21IWALLCLOCKCONFIG\x01",
    )


def wallclock_training_source_bundle_sha256(
    project_root: str | Path | None = None,
) -> str:
    """Recompute the source digest the V2.1i checkpoint must carry."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    return str(_training_source_bundle(root)["sha256"])


def wallclock_evaluator_bundle_sha256(
    project_root: str | Path | None = None,
) -> str:
    """Recompute the exact wall-clock evaluator and policy source digest."""

    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    return str(_evaluator_bundle(root)["sha256"])


@dataclass(frozen=True, slots=True)
class TickMilestones:
    """Five ordered readings from the same monotonic wall clock."""

    loop_started_ns: int
    model_started_ns: int
    model_finished_ns: int
    policy_finished_ns: int
    loop_finished_ns: int

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(type(value) is not int or value < 0 for value in values):
            raise WallClockQualificationError(
                "perf_counter_ns milestones must be nonnegative integers"
            )
        if tuple(sorted(values)) != values:
            raise WallClockQualificationError("wall-clock milestones run backwards")

    @property
    def model_forward_ns(self) -> int:
        return self.model_finished_ns - self.model_started_ns

    @property
    def policy_ns(self) -> int:
        return self.policy_finished_ns - self.loop_started_ns

    @property
    def loop_ns(self) -> int:
        return self.loop_finished_ns - self.loop_started_ns

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.loop_started_ns,
            self.model_started_ns,
            self.model_finished_ns,
            self.policy_finished_ns,
            self.loop_finished_ns,
        )


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        raise WallClockQualificationError("latency distribution cannot be empty")
    if not 0.0 < probability <= 1.0:
        raise WallClockQualificationError("percentile probability must be in (0, 1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


@dataclass(frozen=True, slots=True)
class RepetitionEvidence:
    repetition: int
    warmup_ticks: int
    measured_ticks: int
    warmup_episode_rotations: int
    measured_episode_rotations: int
    episode_seeds: tuple[int, ...]
    milestones: tuple[TickMilestones, ...]

    def __post_init__(self) -> None:
        if type(self.repetition) is not int or self.repetition < 0:
            raise WallClockQualificationError("repetition must be nonnegative")
        if self.warmup_ticks != WARMUP_TICKS or self.measured_ticks != MEASURED_TICKS:
            raise WallClockQualificationError("repetition run shape drifted")
        if len(self.milestones) != self.measured_ticks:
            raise WallClockQualificationError("measured milestone count drifted")
        if any(
            self.milestones[index].loop_started_ns
            > self.milestones[index + 1].loop_started_ns
            for index in range(len(self.milestones) - 1)
        ):
            raise WallClockQualificationError("tick start clock runs backwards")
        if (
            type(self.warmup_episode_rotations) is not int
            or self.warmup_episode_rotations < 0
            or type(self.measured_episode_rotations) is not int
            or self.measured_episode_rotations < 0
        ):
            raise WallClockQualificationError("episode rotation counts are invalid")
        if not self.episode_seeds or any(
            type(seed) is not int or seed < 0 or seed > (1 << 64) - 1
            for seed in self.episode_seeds
        ):
            raise WallClockQualificationError("episode seed trace is invalid")
        expected_seed_count = (
            1 + self.warmup_episode_rotations + self.measured_episode_rotations
        )
        if len(self.episode_seeds) != expected_seed_count:
            raise WallClockQualificationError("episode rotation trace is incomplete")

    @property
    def model_forward_ns(self) -> tuple[int, ...]:
        return tuple(value.model_forward_ns for value in self.milestones)

    @property
    def loop_ns(self) -> tuple[int, ...]:
        return tuple(value.loop_ns for value in self.milestones)

    @property
    def model_p99_ns(self) -> int:
        return _nearest_rank(self.model_forward_ns, 0.99)

    @property
    def loop_p99_ns(self) -> int:
        return _nearest_rank(self.loop_ns, 0.99)

    @property
    def loop_p999_ns(self) -> int:
        return _nearest_rank(self.loop_ns, 0.999)

    @property
    def deadline_misses(self) -> int:
        return sum(value > TICK_PERIOD_NS_60HZ for value in self.loop_ns)

    @property
    def deadline_miss_rate(self) -> float:
        return self.deadline_misses / self.measured_ticks

    @property
    def episode_rotations(self) -> int:
        return self.warmup_episode_rotations + self.measured_episode_rotations

    @property
    def passed(self) -> bool:
        return all(
            (
                self.model_p99_ns <= MODEL_P99_LIMIT_NS,
                self.loop_p99_ns <= LOOP_P99_LIMIT_NS,
                self.loop_p999_ns <= LOOP_P999_LIMIT_NS,
                self.deadline_miss_rate < MAXIMUM_DEADLINE_MISS_RATE,
                self.episode_rotations >= MINIMUM_EPISODE_ROTATIONS,
            )
        )

    @property
    def milestone_trace_sha256(self) -> str:
        return _json_sha256(
            [list(value.as_tuple()) for value in self.milestones],
            domain=b"IRV21IWALLCLOCKTRACE\x01",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "deadline_miss_rate": self.deadline_miss_rate,
            "deadline_misses": self.deadline_misses,
            "episode_rotations": self.episode_rotations,
            "episode_seeds": list(self.episode_seeds),
            "loop_p99_ns": self.loop_p99_ns,
            "loop_p999_ns": self.loop_p999_ns,
            "measured_episode_rotations": self.measured_episode_rotations,
            "measured_ticks": self.measured_ticks,
            "milestone_columns": [
                "loop_started_ns",
                "model_started_ns",
                "model_finished_ns",
                "policy_finished_ns",
                "loop_finished_ns",
            ],
            "milestone_trace_sha256": self.milestone_trace_sha256,
            "milestones_ns": [list(value.as_tuple()) for value in self.milestones],
            "model_forward_p99_ns": self.model_p99_ns,
            "passed": self.passed,
            "repetition": self.repetition,
            "warmup_episode_rotations": self.warmup_episode_rotations,
            "warmup_ticks": self.warmup_ticks,
        }


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    checkpoint_envelope: str
    checkpoint_sha256: str
    state_sha256: str
    model_config_sha256: str
    feature_flags_sha256: str
    hazard_calibration_sha256: str
    training_source_bundle_sha256: str
    evaluator_bundle_sha256: str
    preregistration_sha256: str
    post_dgx_release_body_sha256: str | None
    post_dgx_release_file_sha256: str | None
    model_seed: int

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "state_sha256",
            "model_config_sha256",
            "feature_flags_sha256",
            "hazard_calibration_sha256",
            "training_source_bundle_sha256",
            "evaluator_bundle_sha256",
            "preregistration_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if type(self.model_seed) is not int or self.model_seed < 0:
            raise WallClockQualificationError("model_seed must be nonnegative")
        if self.checkpoint_envelope not in {DEVELOPMENT_ENVELOPE, POST_DGX_ENVELOPE}:
            raise WallClockQualificationError("checkpoint envelope is unsupported")
        release_digests = (
            self.post_dgx_release_body_sha256,
            self.post_dgx_release_file_sha256,
        )
        if self.checkpoint_envelope == DEVELOPMENT_ENVELOPE:
            if any(value is not None for value in release_digests):
                raise WallClockQualificationError(
                    "development checkpoint cannot carry a post-DGX release"
                )
        else:
            for name, value in zip(
                (
                    "post_dgx_release_body_sha256",
                    "post_dgx_release_file_sha256",
                ),
                release_digests,
            ):
                _require_sha256(value, name=name)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PostDgxReleaseReceipt:
    """Canonical external authorization for one exact schema-2 checkpoint."""

    path: Path
    payload: Mapping[str, object]
    body: Mapping[str, object]
    body_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.body_sha256, name="post-DGX release body sha256")
        _require_sha256(self.file_sha256, name="post-DGX release file sha256")


@dataclass(frozen=True, slots=True)
class LoadedV21ICheckpoint:
    """CPU-resident exact model plus its validated envelope lineage."""

    model: CoreV2Model
    envelope: str
    checkpoint_sha256: str
    state_sha256: str
    model_config_sha256: str
    feature_flags_sha256: str
    hazard_calibration_sha256: str
    source_bundle_sha256: str
    model_seed: int
    post_dgx_release: PostDgxReleaseReceipt | None
    training_runtime: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class WallClockQualificationReport:
    provenance: CheckpointProvenance
    config: WallClockQualificationConfig
    repetitions: tuple[RepetitionEvidence, ...]
    runtime: Mapping[str, object]
    model_state_unchanged: bool
    source_unchanged: bool

    def __post_init__(self) -> None:
        if len(self.repetitions) != REPETITIONS:
            raise WallClockQualificationError("exactly three repetitions are required")
        if tuple(value.repetition for value in self.repetitions) != tuple(range(REPETITIONS)):
            raise WallClockQualificationError("repetition sequence is invalid")
        if self.runtime.get("device") != "cpu":
            raise WallClockQualificationError("latency evidence must be CPU-only")
        if self.runtime.get("torch_threads") != 1:
            raise WallClockQualificationError("latency evidence requires one torch thread")
        if self.runtime.get("torch_interop_threads") != 1:
            raise WallClockQualificationError(
                "latency evidence requires one torch interop thread"
            )
        if self.runtime.get("clock") != "time.perf_counter_ns":
            raise WallClockQualificationError("latency evidence used the wrong clock")
        if self.runtime.get("clock_monotonic") is not True:
            raise WallClockQualificationError("latency clock must be monotonic")

    @property
    def passed(self) -> bool:
        return bool(
            self.model_state_unchanged
            and self.source_unchanged
            and all(value.passed for value in self.repetitions)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "gates": {
                "all_repetitions_pass": all(value.passed for value in self.repetitions),
                "loop_p99_limit_ns": LOOP_P99_LIMIT_NS,
                "loop_p999_limit_ns": LOOP_P999_LIMIT_NS,
                "maximum_deadline_miss_rate_exclusive": MAXIMUM_DEADLINE_MISS_RATE,
                "model_p99_limit_ns": MODEL_P99_LIMIT_NS,
                "model_state_unchanged": self.model_state_unchanged,
                "source_unchanged": self.source_unchanged,
            },
            "measurement_kind": MEASUREMENT_KIND,
            "passed": self.passed,
            "policy_identity": POLICY_IDENTITY,
            "provenance": self.provenance.to_dict(),
            "qualification_id": QUALIFICATION_ID,
            "repetitions": [value.to_dict() for value in self.repetitions],
            "runtime": dict(self.runtime),
            "schema_version": SCHEMA_VERSION,
            "scope": {
                "excluded": [
                    "screen capture",
                    "serialization or network transport",
                    "physical HID dispatch",
                    "display effect",
                ],
                "included": (
                    "RGB conversion, exact outcome-aware policy, one model forward, "
                    "in-process MazeChase step, and terminal episode rotation"
                ),
            },
        }

    def receipt(self) -> dict[str, object]:
        """Compact immutable binding consumed by the production live gate."""

        return {
            "checkpoint_envelope": self.provenance.checkpoint_envelope,
            "checkpoint_sha256": self.provenance.checkpoint_sha256,
            "evaluator_bundle_sha256": self.provenance.evaluator_bundle_sha256,
            "feature_flags_sha256": self.provenance.feature_flags_sha256,
            "hazard_calibration_sha256": self.provenance.hazard_calibration_sha256,
            "measurement_kind": MEASUREMENT_KIND,
            "model_config_sha256": self.provenance.model_config_sha256,
            "model_seed": self.provenance.model_seed,
            "model_state_unchanged": self.model_state_unchanged,
            "model_state_sha256": self.provenance.state_sha256,
            "overall_passed": self.passed,
            "policy_identity": POLICY_IDENTITY,
            "post_dgx_release_body_sha256": (
                self.provenance.post_dgx_release_body_sha256
            ),
            "post_dgx_release_file_sha256": (
                self.provenance.post_dgx_release_file_sha256
            ),
            "preregistration_sha256": self.provenance.preregistration_sha256,
            "qualification_config_sha256": wallclock_config_sha256(),
            "qualification_id": QUALIFICATION_ID,
            "repetitions": [
                {
                    "deadline_miss_rate": value.deadline_miss_rate,
                    "deadline_misses": value.deadline_misses,
                    "episode_rotations": value.episode_rotations,
                    "loop_p99_ns": value.loop_p99_ns,
                    "loop_p999_ns": value.loop_p999_ns,
                    "measured_ticks": value.measured_ticks,
                    "milestone_trace_sha256": value.milestone_trace_sha256,
                    "model_forward_p99_ns": value.model_p99_ns,
                    "passed": value.passed,
                    "repetition": value.repetition,
                    "warmup_ticks": value.warmup_ticks,
                }
                for value in self.repetitions
            ],
            "report_sha256": self.sha256,
            "runtime": {
                "clock": self.runtime["clock"],
                "clock_monotonic": self.runtime["clock_monotonic"],
                "device": self.runtime["device"],
                "torch_interop_threads": self.runtime["torch_interop_threads"],
                "torch_threads": self.runtime["torch_threads"],
            },
            "schema_version": SCHEMA_VERSION,
            "source_unchanged": self.source_unchanged,
            "thresholds": {
                "loop_p99_limit_ns": LOOP_P99_LIMIT_NS,
                "loop_p999_limit_ns": LOOP_P999_LIMIT_NS,
                "maximum_deadline_miss_rate_exclusive": MAXIMUM_DEADLINE_MISS_RATE,
                "model_forward_p99_limit_ns": MODEL_P99_LIMIT_NS,
            },
            "training_source_bundle_sha256": (
                self.provenance.training_source_bundle_sha256
            ),
            "workload_sha256": wallclock_workload_sha256(),
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.to_dict(), domain=b"IRV21IWALLCLOCKREPORT\x01")


class _Environment(Protocol):
    def reset(self, seed: int) -> Observation: ...

    def step(self, control: GenericControl) -> StepOutcome: ...


class _Policy(Protocol):
    def reset(self, seed: int) -> None: ...

    def act(self, observation: Observation) -> GenericControl: ...


Clock = Callable[[], int]
EnvironmentFactory = Callable[[], _Environment]


def _default_environment_factory(config: WallClockQualificationConfig) -> _Environment:
    return MazeChaseEnv(
        ghost_count=5,
        ghost_period=1,
        player_period=1,
        extra_loops=16,
        ghost_rule="direct",
        ghost_elroy=False,
        input_delay_ticks=0,
        sticky_direction=False,
        tick_period_ns=config.tick_period_ns,
        max_ticks=config.episode_max_ticks,
    )


def _run_repetition(
    policy: _Policy,
    model: nn.Module,
    *,
    repetition: int,
    config: WallClockQualificationConfig,
    clock: Clock = time.perf_counter_ns,
    environment_factory: EnvironmentFactory | None = None,
) -> RepetitionEvidence:
    """Measure one stream; exposed for focused tests, not protocol overrides."""

    if not callable(getattr(policy, "reset", None)) or not callable(
        getattr(policy, "act", None)
    ):
        raise WallClockQualificationError("policy must define reset() and act()")
    if type(repetition) is not int or repetition < 0:
        raise WallClockQualificationError("repetition must be nonnegative")
    environment = (
        environment_factory()
        if environment_factory is not None
        else _default_environment_factory(config)
    )
    next_seed = config.episode_seed_base + repetition * config.episode_seed_stride
    episode_seeds = [next_seed]
    observation = environment.reset(next_seed)
    policy.reset(next_seed)
    next_seed += 1
    active_model_start: list[int] = []
    active_model_finish: list[int] = []

    def before_forward(_module: nn.Module, _args: tuple[object, ...]) -> None:
        active_model_start.append(clock())

    def after_forward(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
    ) -> None:
        active_model_finish.append(clock())

    pre_handle = model.register_forward_pre_hook(before_forward)
    post_handle = model.register_forward_hook(after_forward)
    milestones: list[TickMilestones] = []
    warmup_rotations = 0
    measured_rotations = 0
    total_ticks = config.warmup_ticks + config.measured_ticks
    try:
        with torch.inference_mode():
            for tick in range(total_ticks):
                active_model_start.clear()
                active_model_finish.clear()
                loop_started = clock()
                control = policy.act(observation)
                policy_finished = clock()
                if len(active_model_start) != 1 or len(active_model_finish) != 1:
                    raise WallClockQualificationError(
                        "exactly one CoreV2 model forward is required per policy tick"
                    )
                outcome = environment.step(control)
                if not isinstance(outcome, StepOutcome):
                    raise WallClockQualificationError(
                        "environment.step() must return StepOutcome"
                    )
                observation = outcome.observation
                rotated = outcome.terminated or outcome.truncated
                if rotated:
                    if next_seed > (1 << 64) - 1:
                        raise WallClockQualificationError("episode seed overflow")
                    episode_seeds.append(next_seed)
                    observation = environment.reset(next_seed)
                    policy.reset(next_seed)
                    next_seed += 1
                    if tick < config.warmup_ticks:
                        warmup_rotations += 1
                    else:
                        measured_rotations += 1
                loop_finished = clock()
                sample = TickMilestones(
                    loop_started_ns=loop_started,
                    model_started_ns=active_model_start[0],
                    model_finished_ns=active_model_finish[0],
                    policy_finished_ns=policy_finished,
                    loop_finished_ns=loop_finished,
                )
                if tick >= config.warmup_ticks:
                    milestones.append(sample)
    finally:
        pre_handle.remove()
        post_handle.remove()

    return RepetitionEvidence(
        repetition=repetition,
        warmup_ticks=config.warmup_ticks,
        measured_ticks=config.measured_ticks,
        warmup_episode_rotations=warmup_rotations,
        measured_episode_rotations=measured_rotations,
        episode_seeds=tuple(episode_seeds),
        milestones=tuple(milestones),
    )


def _runtime_record() -> dict[str, object]:
    clock_info = time.get_clock_info("perf_counter")
    return {
        "clock": "time.perf_counter_ns",
        "clock_adjustable": clock_info.adjustable,
        "clock_monotonic": clock_info.monotonic,
        "clock_resolution_seconds": clock_info.resolution,
        "device": "cpu",
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_threads": torch.get_num_threads(),
    }


def _configure_one_thread_cpu() -> None:
    if torch.cuda.is_initialized():
        raise WallClockQualificationError(
            "CUDA was initialized before the CPU latency qualification"
        )
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise WallClockQualificationError(
                "cannot establish one torch interop thread in this process"
            ) from error
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise WallClockQualificationError("one-thread CPU setup failed")


def _expected_model_config() -> CoreV2Config:
    return CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
        hazard_parameterization="probability_sigmoid_v1",
        latent_comparison="cosine_distance_v1",
        reward_comparison="raw_mse_v0",
        outcome_action_conditioning="thought_only_v0",
        outcome_architecture="all_action_table_v1",
        hazard_outcome_path="dedicated_stopgrad_v1",
        reward_prediction="symlog_twohot_v1",
        prediction_error_fusion="latent_outcome_surprise_v1",
        next_weight=0.5,
        reward_weight=0.25,
        hazard_weight=1.0,
    )


def _strict_json_file(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file():
        raise WallClockQualificationError(f"{name} is missing")
    encoded = path.read_bytes()

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise WallClockQualificationError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                WallClockQualificationError(
                    f"{name} contains non-finite {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WallClockQualificationError(
            f"{name} must be strict UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise WallClockQualificationError(f"{name} root must be an object")
    if encoded != (_canonical_json(payload) + "\n").encode("utf-8"):
        raise WallClockQualificationError(
            f"{name} must be one canonical JSON line"
        )
    return payload, encoded


def load_post_dgx_release_receipt(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PostDgxReleaseReceipt:
    """Load the exact external release artifact without opening any dataset."""

    expected_file_sha256 = _require_sha256(
        expected_file_sha256, name="expected post-DGX release file sha256"
    )
    resolved = Path(path).expanduser().resolve()
    payload, encoded = _strict_json_file(resolved, name="post-DGX release receipt")
    file_sha = sha256(encoded).hexdigest()
    if file_sha != expected_file_sha256:
        raise WallClockQualificationError("post-DGX release file digest mismatch")
    if set(payload) != {
        "body",
        "body_sha256",
        "qualification_id",
        "schema_version",
        "status",
    }:
        raise WallClockQualificationError("post-DGX release envelope fields drifted")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise WallClockQualificationError("post-DGX release schema mismatch")
    if payload["qualification_id"] != POST_DGX_RELEASE_QUALIFICATION_ID:
        raise WallClockQualificationError("post-DGX release qualification ID mismatch")
    if payload["status"] != POST_DGX_RELEASE_STATUS:
        raise WallClockQualificationError("post-DGX release did not pass")
    body = payload["body"]
    if not isinstance(body, Mapping) or set(body) != _POST_DGX_RELEASE_BODY_KEYS:
        raise WallClockQualificationError("post-DGX release body fields drifted")
    body_sha = _json_sha256(body, domain=b"IRV21IPOSTDGXRELEASEBODY\x01")
    if payload["body_sha256"] != body_sha:
        raise WallClockQualificationError("post-DGX release body digest mismatch")
    return PostDgxReleaseReceipt(
        path=resolved,
        payload=payload,
        body=body,
        body_sha256=body_sha,
        file_sha256=file_sha,
    )


def validate_post_dgx_release_binding(
    release: PostDgxReleaseReceipt,
    *,
    checkpoint_sha256: str,
    state_sha256: str,
    model_config_sha256: str,
    feature_flags_sha256: str,
    hazard_calibration_sha256: str,
    source_bundle_sha256: str,
    model_seed: int,
    training_runtime: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind the release to one checkpoint and its complete DGX lineage."""

    if not isinstance(release, PostDgxReleaseReceipt):
        raise WallClockQualificationError("post-DGX release receipt is required")
    body = release.body
    if body.get("classification") != "post_dgx_cpu_qual_release_v1":
        raise WallClockQualificationError("post-DGX release classification mismatch")
    checkpoint = body.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != (
        _POST_DGX_RELEASE_CHECKPOINT_KEYS
    ):
        raise WallClockQualificationError("post-DGX checkpoint binding is incomplete")
    expected_checkpoint = {
        "checkpoint_sha256": _require_sha256(
            checkpoint_sha256, name="checkpoint_sha256"
        ),
        "feature_flags_sha256": _require_sha256(
            feature_flags_sha256, name="feature_flags_sha256"
        ),
        "hazard_calibration_sha256": _require_sha256(
            hazard_calibration_sha256, name="hazard_calibration_sha256"
        ),
        "model_config_sha256": _require_sha256(
            model_config_sha256, name="model_config_sha256"
        ),
        "model_seed": model_seed,
        "source_bundle_sha256": _require_sha256(
            source_bundle_sha256, name="source_bundle_sha256"
        ),
        "state_sha256": _require_sha256(state_sha256, name="state_sha256"),
    }
    if dict(checkpoint) != expected_checkpoint:
        raise WallClockQualificationError(
            "post-DGX release binds a different checkpoint"
        )
    lineage = body.get("dgx_lineage")
    if not isinstance(training_runtime, Mapping) or set(training_runtime) != (
        _POST_DGX_TRAINING_RUNTIME_KEYS
    ):
        raise WallClockQualificationError("DGX training runtime fields drifted")
    if training_runtime.get("accelerator") != "nvidia_dgx_spark":
        raise WallClockQualificationError("DGX training accelerator mismatch")
    for name in (
        "authorization_receipt_sha256",
        "bounded_schedule_sha256",
        "run_artifact_sha256",
    ):
        _require_sha256(training_runtime.get(name), name=name)
    if not isinstance(lineage, Mapping) or dict(lineage) != dict(training_runtime):
        raise WallClockQualificationError("post-DGX lineage differs from checkpoint")
    cpu_qual = body.get("cpu_qualification")
    if not isinstance(cpu_qual, Mapping) or set(cpu_qual) != _POST_DGX_CPU_QUAL_KEYS:
        raise WallClockQualificationError("CPU-QUAL release evidence is incomplete")
    if cpu_qual.get("qualification_id") != "v21i-cpu-qual-world-model-v1":
        raise WallClockQualificationError("CPU-QUAL qualification ID mismatch")
    for name in (
        "evaluation_artifact_sha256",
        "opening_receipt_sha256",
        "preregistration_sha256",
    ):
        _require_sha256(cpu_qual.get(name), name=name)
    required = list(REQUIRED_CPU_QUAL_MODEL_SEEDS)
    if (
        cpu_qual.get("required_model_seeds") != required
        or cpu_qual.get("passed_model_seeds") != required
        or cpu_qual.get("required_passes") != len(required)
        or cpu_qual.get("observed_passes") != len(required)
        or cpu_qual.get("all_candidates_passed") is not True
        or cpu_qual.get("test_split_opened") is not False
    ):
        raise WallClockQualificationError("CPU-QUAL did not pass the exact 5/5 cohort")
    if model_seed not in REQUIRED_CPU_QUAL_MODEL_SEEDS:
        raise WallClockQualificationError("released model seed is outside the CPU-QUAL cohort")
    post_dgx = body.get("post_dgx_verification")
    if not isinstance(post_dgx, Mapping) or set(post_dgx) != (
        _POST_DGX_VERIFICATION_KEYS
    ):
        raise WallClockQualificationError("post-DGX verification evidence is incomplete")
    if post_dgx.get("qualification_id") != "v21i-post-dgx-verification-v1":
        raise WallClockQualificationError("post-DGX verification ID mismatch")
    _require_sha256(post_dgx.get("artifact_sha256"), name="post-DGX artifact sha256")
    if any(
        post_dgx.get(name) is not True
        for name in (
            "all_metrics_finite",
            "calibration_passed",
            "causal_shuffle_passed",
            "hazard_quality_passed",
            "noncollapse_passed",
            "preservation_passed",
        )
    ):
        raise WallClockQualificationError("post-DGX verification gates did not all pass")
    authorization = body.get("authorization")
    if authorization != {
        "cpu_wallclock_allowed": True,
        "play_qualification_allowed": True,
        "test_split_access_allowed": False,
        "training_allowed": False,
    }:
        raise WallClockQualificationError("post-DGX authorization scope drifted")
    return body


def load_exact_v21i_checkpoint_cpu(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_source_bundle_sha256: str,
    expected_model_seed: int,
    project_root: Path,
    required_envelope: str,
    post_dgx_release: PostDgxReleaseReceipt | None = None,
) -> LoadedV21ICheckpoint:
    if required_envelope not in {DEVELOPMENT_ENVELOPE, POST_DGX_ENVELOPE}:
        raise WallClockQualificationError("required checkpoint envelope is unsupported")
    _configure_one_thread_cpu()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    )
    expected_source_bundle_sha256 = _require_sha256(
        expected_source_bundle_sha256, name="expected_source_bundle_sha256"
    )
    if type(expected_model_seed) is not int or expected_model_seed < 0:
        raise WallClockQualificationError("expected_model_seed must be nonnegative")
    actual_checkpoint_sha = _file_sha256(checkpoint_path)
    if actual_checkpoint_sha != expected_checkpoint_sha256:
        raise WallClockQualificationError("checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise WallClockQualificationError("checkpoint payload must be a dictionary")
    training_runtime: Mapping[str, object] | None = None
    if required_envelope == DEVELOPMENT_ENVELOPE:
        if set(payload) != _DEVELOPMENT_CHECKPOINT_KEYS:
            raise WallClockQualificationError("development checkpoint schema drifted")
        if payload.get("schema_version") != 1:
            raise WallClockQualificationError("development checkpoint schema mismatch")
        if payload.get("mode") != "v21i_development_only":
            raise WallClockQualificationError("checkpoint is not V2.1i development")
        if payload.get("classification") != "development_candidate_not_qualified":
            raise WallClockQualificationError("development classification is invalid")
        if payload.get("device") != "cpu" or payload.get("threads") != 1:
            raise WallClockQualificationError(
                "development checkpoint was not produced CPU-only/one-thread"
            )
        if post_dgx_release is not None:
            raise WallClockQualificationError(
                "development checkpoint cannot use a post-DGX release"
            )
    else:
        if set(payload) != _POST_DGX_CHECKPOINT_KEYS:
            raise WallClockQualificationError("post-DGX checkpoint schema drifted")
        if payload.get("schema_version") != 2:
            raise WallClockQualificationError("post-DGX checkpoint schema mismatch")
        if payload.get("mode") != POST_DGX_MODE:
            raise WallClockQualificationError("checkpoint is not bounded DGX V2.1i")
        if payload.get("classification") != POST_DGX_CLASSIFICATION:
            raise WallClockQualificationError("post-DGX classification is invalid")
        if payload.get("device") != "cuda" or payload.get("threads") != 1:
            raise WallClockQualificationError(
                "post-DGX training provenance must be CUDA/one-thread"
            )
        raw_training_runtime = payload.get("training_runtime")
        if not isinstance(raw_training_runtime, Mapping):
            raise WallClockQualificationError("post-DGX training runtime is missing")
        training_runtime = raw_training_runtime
        if post_dgx_release is None:
            raise WallClockQualificationError(
                "post-DGX checkpoint requires an external release receipt"
            )
    if payload.get("model_seed") != expected_model_seed:
        raise WallClockQualificationError("checkpoint model seed mismatch")
    if payload.get("calibration_mode") != "bias_only":
        raise WallClockQualificationError("checkpoint calibration mode must be bias_only")
    development_gate = payload.get("development_gate")
    if not isinstance(development_gate, Mapping) or development_gate.get("passed") is not True:
        raise WallClockQualificationError("checkpoint development gate did not pass")
    partitions = payload.get("dataset_partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != {
        "TRAIN-FIT",
        "TRAIN-CAL",
        "DEV",
    }:
        raise WallClockQualificationError(
            "checkpoint must contain only TRAIN-FIT/TRAIN-CAL/DEV partitions"
        )
    config_payload = payload.get("config")
    flags_payload = payload.get("flags")
    state_payload = payload.get("model_state_dict")
    calibration = payload.get("hazard_calibration")
    source_bundle = payload.get("source_bundle")
    if (
        not isinstance(config_payload, Mapping)
        or dict(config_payload) != asdict(_expected_model_config())
    ):
        raise WallClockQualificationError("checkpoint model configuration drifted")
    if (
        not isinstance(flags_payload, Mapping)
        or dict(flags_payload) != asdict(CONFIG_B_PREDICTIVE)
    ):
        raise WallClockQualificationError("checkpoint feature flags drifted")
    if not isinstance(state_payload, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in state_payload.items()
    ):
        raise WallClockQualificationError("checkpoint state_dict is invalid")
    if not isinstance(calibration, Mapping):
        raise WallClockQualificationError("checkpoint calibration is missing")
    if not isinstance(source_bundle, Mapping):
        raise WallClockQualificationError("checkpoint model configuration is incomplete")
    current_source_bundle = _training_source_bundle(project_root)
    if (
        dict(source_bundle) != current_source_bundle
        or current_source_bundle["sha256"] != expected_source_bundle_sha256
    ):
        raise WallClockQualificationError("current source differs from the checkpoint source")
    if calibration.get("fit_mode") != "bias_only" or calibration.get("accepted") is not True:
        raise WallClockQualificationError("accepted bias-only hazard calibration is required")
    scales = calibration.get("scale")
    biases = calibration.get("bias")
    if not isinstance(scales, list) or scales != [1.0] * 5:
        raise WallClockQualificationError("hazard calibration scales must remain identity")
    if not isinstance(biases, list) or len(biases) != 5 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in biases
    ):
        raise WallClockQualificationError("hazard calibration biases must be finite")
    config = CoreV2Config(**dict(config_payload))
    flags = FeatureFlags(**dict(flags_payload))
    model = CoreV2Model(config=config, flags=flags).to(torch.device("cpu"))
    model.load_state_dict(state_payload, strict=True)
    model.eval()
    if any(value.device.type != "cpu" for value in model.parameters()):
        raise WallClockQualificationError("checkpoint parameters are not CPU resident")
    if any(value.device.type != "cpu" for value in model.buffers()):
        raise WallClockQualificationError("checkpoint buffers are not CPU resident")
    if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise WallClockQualificationError("checkpoint parameters must be finite")
    if any(not bool(torch.isfinite(value).all()) for value in model.buffers()):
        raise WallClockQualificationError("checkpoint buffers must be finite")
    state_sha = _state_dict_sha256(model.state_dict())
    if payload.get("state_sha256") != state_sha:
        raise WallClockQualificationError("checkpoint state digest mismatch")
    if model.world_model is None or model.world_model.outcome_model is None:
        raise WallClockQualificationError("checkpoint outcome model is missing")
    scale_buffer = model.world_model.outcome_model.hazard_calibration_scale
    bias_buffer = model.world_model.outcome_model.hazard_calibration_bias
    if tuple(float(value) for value in scale_buffer.tolist()) != tuple(scales):
        raise WallClockQualificationError("calibration scale buffer metadata mismatch")
    if any(
        not math.isclose(float(buffer), float(metadata), rel_tol=0.0, abs_tol=1e-6)
        for buffer, metadata in zip(bias_buffer.tolist(), biases)
    ):
        raise WallClockQualificationError("calibration bias buffer metadata mismatch")
    model_config_sha = _json_sha256(
        dict(config_payload), domain=b"IRV21IMODELCONFIG\x01"
    )
    feature_flags_sha = _json_sha256(
        dict(flags_payload), domain=b"IRV21IFEATUREFLAGS\x01"
    )
    calibration_sha = _json_sha256(
        dict(calibration), domain=b"IRV21IHAZARDCALIBRATION\x01"
    )
    if required_envelope == POST_DGX_ENVELOPE:
        if training_runtime is None or post_dgx_release is None:
            raise AssertionError("post-DGX release precondition was lost")
        validate_post_dgx_release_binding(
            post_dgx_release,
            checkpoint_sha256=actual_checkpoint_sha,
            state_sha256=state_sha,
            model_config_sha256=model_config_sha,
            feature_flags_sha256=feature_flags_sha,
            hazard_calibration_sha256=calibration_sha,
            source_bundle_sha256=expected_source_bundle_sha256,
            model_seed=expected_model_seed,
            training_runtime=training_runtime,
        )
    return LoadedV21ICheckpoint(
        model=model,
        envelope=required_envelope,
        checkpoint_sha256=actual_checkpoint_sha,
        state_sha256=state_sha,
        model_config_sha256=model_config_sha,
        feature_flags_sha256=feature_flags_sha,
        hazard_calibration_sha256=calibration_sha,
        source_bundle_sha256=expected_source_bundle_sha256,
        model_seed=expected_model_seed,
        post_dgx_release=post_dgx_release,
        training_runtime=training_runtime,
    )


def _write_open_artifact(handle: BinaryIO, payload: object) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())


def validate_wallclock_receipt(
    receipt: Mapping[str, object],
    *,
    expected_checkpoint_sha256: str,
    expected_source_bundle_sha256: str,
    expected_preregistration_sha256: str | None = None,
    required_checkpoint_envelope: str | None = None,
    expected_post_dgx_release_receipt_sha256: str | None = None,
    project_root: str | Path | None = None,
) -> Mapping[str, object]:
    """Validate a compact receipt and recompute both live source bindings."""

    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    )
    expected_source_bundle_sha256 = _require_sha256(
        expected_source_bundle_sha256, name="expected_source_bundle_sha256"
    )
    if expected_preregistration_sha256 is not None:
        expected_preregistration_sha256 = _require_sha256(
            expected_preregistration_sha256,
            name="expected_preregistration_sha256",
        )
    if required_checkpoint_envelope is not None and required_checkpoint_envelope not in {
        DEVELOPMENT_ENVELOPE,
        POST_DGX_ENVELOPE,
    }:
        raise WallClockQualificationError("required checkpoint envelope is unsupported")
    if expected_post_dgx_release_receipt_sha256 is not None:
        expected_post_dgx_release_receipt_sha256 = _require_sha256(
            expected_post_dgx_release_receipt_sha256,
            name="expected_post_dgx_release_receipt_sha256",
        )
    required = {
        "checkpoint_envelope",
        "checkpoint_sha256",
        "evaluator_bundle_sha256",
        "feature_flags_sha256",
        "hazard_calibration_sha256",
        "measurement_kind",
        "model_config_sha256",
        "model_seed",
        "model_state_sha256",
        "model_state_unchanged",
        "overall_passed",
        "policy_identity",
        "post_dgx_release_body_sha256",
        "post_dgx_release_file_sha256",
        "preregistration_sha256",
        "qualification_config_sha256",
        "qualification_id",
        "repetitions",
        "report_sha256",
        "runtime",
        "schema_version",
        "source_unchanged",
        "thresholds",
        "training_source_bundle_sha256",
        "workload_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise WallClockQualificationError("wall-clock receipt fields are incomplete")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise WallClockQualificationError("wall-clock receipt schema mismatch")
    if receipt.get("qualification_id") != QUALIFICATION_ID:
        raise WallClockQualificationError("wall-clock qualification ID mismatch")
    if receipt.get("measurement_kind") != MEASUREMENT_KIND:
        raise WallClockQualificationError("wall-clock measurement kind mismatch")
    if receipt.get("policy_identity") != POLICY_IDENTITY:
        raise WallClockQualificationError("wall-clock policy identity mismatch")
    envelope = receipt.get("checkpoint_envelope")
    if envelope not in {DEVELOPMENT_ENVELOPE, POST_DGX_ENVELOPE}:
        raise WallClockQualificationError("wall-clock checkpoint envelope mismatch")
    if required_checkpoint_envelope is not None and envelope != (
        required_checkpoint_envelope
    ):
        raise WallClockQualificationError("wall-clock checkpoint envelope is not authorized")
    release_body_sha = receipt.get("post_dgx_release_body_sha256")
    release_file_sha = receipt.get("post_dgx_release_file_sha256")
    if envelope == DEVELOPMENT_ENVELOPE:
        if release_body_sha is not None or release_file_sha is not None:
            raise WallClockQualificationError(
                "development wall-clock receipt cannot carry post-DGX release"
            )
        if expected_post_dgx_release_receipt_sha256 is not None:
            raise WallClockQualificationError(
                "development wall-clock receipt cannot satisfy DGX release"
            )
    else:
        _require_sha256(release_body_sha, name="post_dgx_release_body_sha256")
        _require_sha256(release_file_sha, name="post_dgx_release_file_sha256")
        if (
            expected_post_dgx_release_receipt_sha256 is not None
            and release_file_sha != expected_post_dgx_release_receipt_sha256
        ):
            raise WallClockQualificationError("post-DGX release receipt mismatch")
    if receipt.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise WallClockQualificationError("wall-clock receipt checkpoint mismatch")
    if receipt.get("training_source_bundle_sha256") != expected_source_bundle_sha256:
        raise WallClockQualificationError("wall-clock receipt source mismatch")
    current_training_source_sha = wallclock_training_source_bundle_sha256(project_root)
    if current_training_source_sha != expected_source_bundle_sha256:
        raise WallClockQualificationError(
            "current training source differs from the wall-clock receipt"
        )
    current_evaluator_sha = wallclock_evaluator_bundle_sha256(project_root)
    if receipt.get("evaluator_bundle_sha256") != current_evaluator_sha:
        raise WallClockQualificationError(
            "current evaluator source differs from the wall-clock receipt"
        )
    if receipt.get("workload_sha256") != wallclock_workload_sha256():
        raise WallClockQualificationError("wall-clock workload digest mismatch")
    if receipt.get("qualification_config_sha256") != wallclock_config_sha256():
        raise WallClockQualificationError("wall-clock config digest mismatch")
    for digest_name in (
        "feature_flags_sha256",
        "hazard_calibration_sha256",
        "model_config_sha256",
        "model_state_sha256",
        "preregistration_sha256",
        "report_sha256",
    ):
        _require_sha256(receipt.get(digest_name), name=digest_name)
    if (
        expected_preregistration_sha256 is not None
        and receipt.get("preregistration_sha256")
        != expected_preregistration_sha256
    ):
        raise WallClockQualificationError("wall-clock preregistration mismatch")
    if type(receipt.get("model_seed")) is not int or receipt.get("model_seed") < 0:
        raise WallClockQualificationError("wall-clock model seed is invalid")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping) or runtime != {
        "clock": "time.perf_counter_ns",
        "clock_monotonic": True,
        "device": "cpu",
        "torch_interop_threads": 1,
        "torch_threads": 1,
    }:
        raise WallClockQualificationError("wall-clock CPU runtime receipt is invalid")
    thresholds = receipt.get("thresholds")
    if not isinstance(thresholds, Mapping) or thresholds != {
        "loop_p99_limit_ns": LOOP_P99_LIMIT_NS,
        "loop_p999_limit_ns": LOOP_P999_LIMIT_NS,
        "maximum_deadline_miss_rate_exclusive": MAXIMUM_DEADLINE_MISS_RATE,
        "model_forward_p99_limit_ns": MODEL_P99_LIMIT_NS,
    }:
        raise WallClockQualificationError("wall-clock thresholds drifted")
    repetitions = receipt.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != REPETITIONS:
        raise WallClockQualificationError("exactly three latency repetitions are required")
    repetition_fields = {
        "deadline_miss_rate",
        "deadline_misses",
        "episode_rotations",
        "loop_p99_ns",
        "loop_p999_ns",
        "measured_ticks",
        "milestone_trace_sha256",
        "model_forward_p99_ns",
        "passed",
        "repetition",
        "warmup_ticks",
    }
    for index, summary in enumerate(repetitions):
        if not isinstance(summary, Mapping) or set(summary) != repetition_fields:
            raise WallClockQualificationError("latency repetition receipt is malformed")
        if summary.get("repetition") != index:
            raise WallClockQualificationError("latency repetition order mismatch")
        if summary.get("warmup_ticks") != WARMUP_TICKS:
            raise WallClockQualificationError("latency warmup count mismatch")
        if summary.get("measured_ticks") != MEASURED_TICKS:
            raise WallClockQualificationError("latency measured count mismatch")
        _require_sha256(
            summary.get("milestone_trace_sha256"), name="milestone_trace_sha256"
        )
        model_p99 = summary.get("model_forward_p99_ns")
        loop_p99 = summary.get("loop_p99_ns")
        loop_p999 = summary.get("loop_p999_ns")
        miss_rate = summary.get("deadline_miss_rate")
        misses = summary.get("deadline_misses")
        episode_rotations = summary.get("episode_rotations")
        if (
            type(model_p99) is not int
            or model_p99 < 0
            or type(loop_p99) is not int
            or loop_p99 < 0
            or type(loop_p999) is not int
            or loop_p999 < 0
            or type(misses) is not int
            or misses < 0
            or type(episode_rotations) is not int
            or episode_rotations < MINIMUM_EPISODE_ROTATIONS
            or not isinstance(miss_rate, (int, float))
            or isinstance(miss_rate, bool)
            or not math.isfinite(float(miss_rate))
            or float(miss_rate) != misses / MEASURED_TICKS
            or model_p99 > MODEL_P99_LIMIT_NS
            or loop_p99 > LOOP_P99_LIMIT_NS
            or loop_p999 > LOOP_P999_LIMIT_NS
            or float(miss_rate) >= MAXIMUM_DEADLINE_MISS_RATE
            or summary.get("passed") is not True
        ):
            raise WallClockQualificationError("latency repetition failed a frozen gate")
    if (
        receipt.get("model_state_unchanged") is not True
        or receipt.get("source_unchanged") is not True
        or receipt.get("overall_passed") is not True
    ):
        raise WallClockQualificationError("wall-clock overall gate did not pass")
    return receipt


def validate_wallclock_artifact(
    artifact: Mapping[str, object],
    *,
    expected_checkpoint_sha256: str,
    expected_source_bundle_sha256: str,
    expected_preregistration_sha256: str | None = None,
    required_checkpoint_envelope: str | None = None,
    expected_post_dgx_release_receipt_sha256: str | None = None,
    project_root: str | Path | None = None,
) -> Mapping[str, object]:
    """Fail closed on any incomplete, altered, or failing latency receipt."""

    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    )
    expected_source_bundle_sha256 = _require_sha256(
        expected_source_bundle_sha256, name="expected_source_bundle_sha256"
    )
    if expected_preregistration_sha256 is not None:
        expected_preregistration_sha256 = _require_sha256(
            expected_preregistration_sha256,
            name="expected_preregistration_sha256",
        )
    if not isinstance(artifact, Mapping):
        raise WallClockQualificationError("wall-clock artifact must be a mapping")
    if set(artifact) != {
        "receipt",
        "receipt_sha256",
        "report",
        "report_sha256",
        "schema_version",
        "status",
    }:
        raise WallClockQualificationError("wall-clock artifact fields are incomplete")
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("status") != "passed":
        raise WallClockQualificationError("wall-clock artifact did not pass")
    report = artifact.get("report")
    receipt = artifact.get("receipt")
    if not isinstance(report, Mapping) or not isinstance(receipt, Mapping):
        raise WallClockQualificationError("wall-clock report or receipt is missing")
    report_sha = _json_sha256(report, domain=b"IRV21IWALLCLOCKREPORT\x01")
    if artifact.get("report_sha256") != report_sha:
        raise WallClockQualificationError("wall-clock report digest mismatch")
    if receipt.get("report_sha256") != report_sha:
        raise WallClockQualificationError("wall-clock receipt binds the wrong report")
    receipt_sha = _json_sha256(receipt, domain=b"IRV21IWALLCLOCKRECEIPT\x01")
    if artifact.get("receipt_sha256") != receipt_sha:
        raise WallClockQualificationError("wall-clock receipt digest mismatch")
    validate_wallclock_receipt(
        receipt,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_source_bundle_sha256=expected_source_bundle_sha256,
        expected_preregistration_sha256=expected_preregistration_sha256,
        required_checkpoint_envelope=required_checkpoint_envelope,
        expected_post_dgx_release_receipt_sha256=(
            expected_post_dgx_release_receipt_sha256
        ),
        project_root=project_root,
    )
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("qualification_id") != QUALIFICATION_ID
        or report.get("measurement_kind") != MEASUREMENT_KIND
        or report.get("policy_identity") != POLICY_IDENTITY
    ):
        raise WallClockQualificationError("wall-clock report identity drifted")
    report_provenance = report.get("provenance")
    if not isinstance(report_provenance, Mapping):
        raise WallClockQualificationError("wall-clock report provenance is missing")
    provenance_receipt_fields = {
        "checkpoint_envelope": "checkpoint_envelope",
        "checkpoint_sha256": "checkpoint_sha256",
        "evaluator_bundle_sha256": "evaluator_bundle_sha256",
        "feature_flags_sha256": "feature_flags_sha256",
        "hazard_calibration_sha256": "hazard_calibration_sha256",
        "model_config_sha256": "model_config_sha256",
        "model_seed": "model_seed",
        "post_dgx_release_body_sha256": "post_dgx_release_body_sha256",
        "post_dgx_release_file_sha256": "post_dgx_release_file_sha256",
        "preregistration_sha256": "preregistration_sha256",
        "state_sha256": "model_state_sha256",
        "training_source_bundle_sha256": "training_source_bundle_sha256",
    }
    if any(
        report_provenance.get(report_name) != receipt.get(receipt_name)
        for report_name, receipt_name in provenance_receipt_fields.items()
    ):
        raise WallClockQualificationError(
            "wall-clock report and receipt provenance differ"
        )
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise WallClockQualificationError("wall-clock receipt schema mismatch")
    if receipt.get("qualification_id") != QUALIFICATION_ID:
        raise WallClockQualificationError("wall-clock qualification ID mismatch")
    if receipt.get("measurement_kind") != MEASUREMENT_KIND:
        raise WallClockQualificationError("wall-clock measurement kind mismatch")
    if receipt.get("policy_identity") != POLICY_IDENTITY:
        raise WallClockQualificationError("wall-clock policy identity mismatch")
    if receipt.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise WallClockQualificationError("wall-clock receipt checkpoint mismatch")
    if receipt.get("training_source_bundle_sha256") != expected_source_bundle_sha256:
        raise WallClockQualificationError("wall-clock receipt source mismatch")
    if receipt.get("workload_sha256") != wallclock_workload_sha256():
        raise WallClockQualificationError("wall-clock workload digest mismatch")
    if receipt.get("qualification_config_sha256") != wallclock_config_sha256():
        raise WallClockQualificationError("wall-clock config digest mismatch")
    for digest_name in (
        "evaluator_bundle_sha256",
        "feature_flags_sha256",
        "hazard_calibration_sha256",
        "model_config_sha256",
        "model_state_sha256",
        "preregistration_sha256",
    ):
        _require_sha256(receipt.get(digest_name), name=digest_name)
    if (
        expected_preregistration_sha256 is not None
        and receipt.get("preregistration_sha256")
        != expected_preregistration_sha256
    ):
        raise WallClockQualificationError("wall-clock preregistration mismatch")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping) or runtime != {
        "clock": "time.perf_counter_ns",
        "clock_monotonic": True,
        "device": "cpu",
        "torch_interop_threads": 1,
        "torch_threads": 1,
    }:
        raise WallClockQualificationError("wall-clock CPU runtime receipt is invalid")
    thresholds = receipt.get("thresholds")
    if not isinstance(thresholds, Mapping) or thresholds != {
        "loop_p99_limit_ns": LOOP_P99_LIMIT_NS,
        "loop_p999_limit_ns": LOOP_P999_LIMIT_NS,
        "maximum_deadline_miss_rate_exclusive": MAXIMUM_DEADLINE_MISS_RATE,
        "model_forward_p99_limit_ns": MODEL_P99_LIMIT_NS,
    }:
        raise WallClockQualificationError("wall-clock thresholds drifted")
    config_payload = report.get("config")
    if not isinstance(config_payload, Mapping):
        raise WallClockQualificationError("wall-clock report config is missing")
    if _json_sha256(
        dict(config_payload), domain=b"IRV21IWALLCLOCKCONFIG\x01"
    ) != wallclock_config_sha256():
        raise WallClockQualificationError("wall-clock report config drifted")
    repetitions = receipt.get("repetitions")
    report_repetitions = report.get("repetitions")
    if (
        not isinstance(repetitions, list)
        or not isinstance(report_repetitions, list)
        or len(repetitions) != REPETITIONS
        or len(report_repetitions) != REPETITIONS
    ):
        raise WallClockQualificationError("exactly three latency repetitions are required")
    full_repetition_fields = {
        "deadline_miss_rate",
        "deadline_misses",
        "episode_rotations",
        "episode_seeds",
        "loop_p99_ns",
        "loop_p999_ns",
        "measured_episode_rotations",
        "measured_ticks",
        "milestone_columns",
        "milestone_trace_sha256",
        "milestones_ns",
        "model_forward_p99_ns",
        "passed",
        "repetition",
        "warmup_episode_rotations",
        "warmup_ticks",
    }
    milestone_columns = [
        "loop_started_ns",
        "model_started_ns",
        "model_finished_ns",
        "policy_finished_ns",
        "loop_finished_ns",
    ]
    for index, (summary, full) in enumerate(zip(repetitions, report_repetitions)):
        if not isinstance(summary, Mapping) or not isinstance(full, Mapping):
            raise WallClockQualificationError("latency repetition is malformed")
        if set(full) != full_repetition_fields:
            raise WallClockQualificationError("full latency repetition is incomplete")
        if summary.get("repetition") != index:
            raise WallClockQualificationError("latency repetition order mismatch")
        if summary.get("warmup_ticks") != WARMUP_TICKS:
            raise WallClockQualificationError("latency warmup count mismatch")
        if summary.get("measured_ticks") != MEASURED_TICKS:
            raise WallClockQualificationError("latency measured count mismatch")
        if summary.get("model_forward_p99_ns") != full.get("model_forward_p99_ns"):
            raise WallClockQualificationError("model-forward receipt mismatch")
        if summary.get("loop_p99_ns") != full.get("loop_p99_ns"):
            raise WallClockQualificationError("loop p99 receipt mismatch")
        if summary.get("loop_p999_ns") != full.get("loop_p999_ns"):
            raise WallClockQualificationError("loop p99.9 receipt mismatch")
        if summary.get("deadline_miss_rate") != full.get("deadline_miss_rate"):
            raise WallClockQualificationError("deadline miss receipt mismatch")
        if summary.get("milestone_trace_sha256") != full.get("milestone_trace_sha256"):
            raise WallClockQualificationError("milestone trace receipt mismatch")
        if summary.get("episode_rotations") != full.get("episode_rotations"):
            raise WallClockQualificationError("episode rotation receipt mismatch")
        rows = full.get("milestones_ns")
        if full.get("milestone_columns") != milestone_columns:
            raise WallClockQualificationError("milestone column contract drifted")
        if not isinstance(rows, list) or len(rows) != MEASURED_TICKS:
            raise WallClockQualificationError("raw milestone count mismatch")
        raw_samples: list[TickMilestones] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(milestone_columns):
                raise WallClockQualificationError("raw milestone row is malformed")
            raw_samples.append(TickMilestones(*row))
        if any(
            raw_samples[position].loop_started_ns
            > raw_samples[position + 1].loop_started_ns
            for position in range(len(raw_samples) - 1)
        ):
            raise WallClockQualificationError("raw tick start clock runs backwards")
        raw_trace_sha = _json_sha256(
            rows,
            domain=b"IRV21IWALLCLOCKTRACE\x01",
        )
        if raw_trace_sha != summary.get("milestone_trace_sha256"):
            raise WallClockQualificationError("raw milestone trace digest mismatch")
        raw_model = [sample.model_forward_ns for sample in raw_samples]
        raw_loop = [sample.loop_ns for sample in raw_samples]
        raw_model_p99 = _nearest_rank(raw_model, 0.99)
        raw_loop_p99 = _nearest_rank(raw_loop, 0.99)
        raw_loop_p999 = _nearest_rank(raw_loop, 0.999)
        raw_misses = sum(value > TICK_PERIOD_NS_60HZ for value in raw_loop)
        raw_miss_rate = raw_misses / MEASURED_TICKS
        if (
            raw_model_p99 != summary.get("model_forward_p99_ns")
            or raw_loop_p99 != summary.get("loop_p99_ns")
            or raw_loop_p999 != summary.get("loop_p999_ns")
            or raw_misses != summary.get("deadline_misses")
            or raw_miss_rate != summary.get("deadline_miss_rate")
        ):
            raise WallClockQualificationError(
                "latency summaries do not reproduce from raw milestones"
            )
        warmup_rotations = full.get("warmup_episode_rotations")
        measured_rotations = full.get("measured_episode_rotations")
        episode_seeds = full.get("episode_seeds")
        if (
            type(warmup_rotations) is not int
            or warmup_rotations < 0
            or type(measured_rotations) is not int
            or measured_rotations < 0
            or not isinstance(episode_seeds, list)
            or len(episode_seeds) != 1 + warmup_rotations + measured_rotations
            or any(type(seed) is not int or seed < 0 for seed in episode_seeds)
            or full.get("episode_rotations")
            != warmup_rotations + measured_rotations
        ):
            raise WallClockQualificationError("episode rotation trace is incomplete")
        _require_sha256(
            summary.get("milestone_trace_sha256"), name="milestone_trace_sha256"
        )
        model_p99 = summary.get("model_forward_p99_ns")
        loop_p99 = summary.get("loop_p99_ns")
        loop_p999 = summary.get("loop_p999_ns")
        miss_rate = summary.get("deadline_miss_rate")
        if (
            type(model_p99) is not int
            or type(loop_p99) is not int
            or type(loop_p999) is not int
            or not isinstance(miss_rate, (int, float))
            or isinstance(miss_rate, bool)
            or not math.isfinite(float(miss_rate))
            or model_p99 > MODEL_P99_LIMIT_NS
            or loop_p99 > LOOP_P99_LIMIT_NS
            or loop_p999 > LOOP_P999_LIMIT_NS
            or float(miss_rate) >= MAXIMUM_DEADLINE_MISS_RATE
            or summary.get("passed") is not True
        ):
            raise WallClockQualificationError("latency repetition failed a frozen gate")
    gates = report.get("gates")
    if (
        receipt.get("overall_passed") is not True
        or receipt.get("source_unchanged") is not True
        or report.get("passed") is not True
        or not isinstance(gates, Mapping)
        or gates.get("model_state_unchanged") is not True
        or gates.get("source_unchanged") is not True
        or gates.get("all_repetitions_pass") is not True
    ):
        raise WallClockQualificationError("wall-clock overall gate did not pass")
    return receipt


def qualify_v21i_checkpoint_wallclock_create_only(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_source_bundle_sha256: str,
    expected_model_seed: int,
    preregistration_sha256: str,
    output_path: str | Path,
    checkpoint_envelope: str = DEVELOPMENT_ENVELOPE,
    post_dgx_release_receipt_path: str | Path | None = None,
    expected_post_dgx_release_receipt_sha256: str | None = None,
) -> WallClockQualificationReport:
    """Measure the exact checkpoint/policy and publish one immutable artifact."""

    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    )
    expected_source_bundle_sha256 = _require_sha256(
        expected_source_bundle_sha256, name="expected_source_bundle_sha256"
    )
    preregistration_sha256 = _require_sha256(
        preregistration_sha256, name="preregistration_sha256"
    )
    if type(expected_model_seed) is not int or expected_model_seed < 0:
        raise WallClockQualificationError("expected_model_seed must be nonnegative")
    if checkpoint_envelope not in {DEVELOPMENT_ENVELOPE, POST_DGX_ENVELOPE}:
        raise WallClockQualificationError("checkpoint_envelope is unsupported")
    release_path = (
        None
        if post_dgx_release_receipt_path is None
        else Path(post_dgx_release_receipt_path).expanduser().resolve()
    )
    if checkpoint_envelope == DEVELOPMENT_ENVELOPE:
        if (
            release_path is not None
            or expected_post_dgx_release_receipt_sha256 is not None
        ):
            raise WallClockQualificationError(
                "development envelope cannot use a post-DGX release"
            )
    elif release_path is None or expected_post_dgx_release_receipt_sha256 is None:
        raise WallClockQualificationError(
            "post-DGX envelope requires a release path and expected digest"
        )
    if expected_post_dgx_release_receipt_sha256 is not None:
        expected_post_dgx_release_receipt_sha256 = _require_sha256(
            expected_post_dgx_release_receipt_sha256,
            name="expected_post_dgx_release_receipt_sha256",
        )
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise WallClockQualificationError("checkpoint does not exist")
    if not destination.parent.is_dir():
        raise WallClockQualificationError("artifact parent directory must already exist")
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as error:
        raise WallClockQualificationError(
            f"refusing to overwrite existing artifact: {destination}"
        ) from error

    request = {
        "checkpoint_path": str(checkpoint),
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "expected_model_seed": expected_model_seed,
        "expected_source_bundle_sha256": expected_source_bundle_sha256,
        "checkpoint_envelope": checkpoint_envelope,
        "expected_post_dgx_release_receipt_sha256": (
            expected_post_dgx_release_receipt_sha256
        ),
        "post_dgx_release_receipt_path": (
            None if release_path is None else str(release_path)
        ),
        "preregistration_sha256": preregistration_sha256,
    }
    with os.fdopen(descriptor, "wb") as handle:
        try:
            _configure_one_thread_cpu()
            config = WallClockQualificationConfig()
            project_root = Path(__file__).resolve().parents[4]
            post_dgx_release = (
                None
                if release_path is None
                else load_post_dgx_release_receipt(
                    release_path,
                    expected_file_sha256=str(
                        expected_post_dgx_release_receipt_sha256
                    ),
                )
            )
            loaded = load_exact_v21i_checkpoint_cpu(
                checkpoint,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_source_bundle_sha256=expected_source_bundle_sha256,
                expected_model_seed=expected_model_seed,
                project_root=project_root,
                required_envelope=checkpoint_envelope,
                post_dgx_release=post_dgx_release,
            )
            model = loaded.model
            initial_source = _training_source_bundle(project_root)
            evaluator_bundle = _evaluator_bundle(project_root)
            provenance = CheckpointProvenance(
                checkpoint_envelope=loaded.envelope,
                checkpoint_sha256=loaded.checkpoint_sha256,
                state_sha256=loaded.state_sha256,
                model_config_sha256=loaded.model_config_sha256,
                feature_flags_sha256=loaded.feature_flags_sha256,
                hazard_calibration_sha256=loaded.hazard_calibration_sha256,
                training_source_bundle_sha256=loaded.source_bundle_sha256,
                evaluator_bundle_sha256=str(evaluator_bundle["sha256"]),
                preregistration_sha256=preregistration_sha256,
                post_dgx_release_body_sha256=(
                    None
                    if post_dgx_release is None
                    else post_dgx_release.body_sha256
                ),
                post_dgx_release_file_sha256=(
                    None
                    if post_dgx_release is None
                    else post_dgx_release.file_sha256
                ),
                model_seed=loaded.model_seed,
            )
            policy = OutcomeAwareCoreV2MazePolicy(model)
            if policy.identity != POLICY_IDENTITY or policy.model is not model:
                raise WallClockQualificationError("exact outcome-aware policy binding failed")
            before_state_sha = _state_dict_sha256(model.state_dict())
            repetitions = tuple(
                _run_repetition(
                    policy,
                    model,
                    repetition=repetition,
                    config=config,
                )
                for repetition in range(config.repetitions)
            )
            after_state_sha = _state_dict_sha256(model.state_dict())
            final_source = _training_source_bundle(project_root)
            final_evaluator = _evaluator_bundle(project_root)
            report = WallClockQualificationReport(
                provenance=provenance,
                config=config,
                repetitions=repetitions,
                runtime=_runtime_record(),
                model_state_unchanged=(
                    before_state_sha == after_state_sha == provenance.state_sha256
                ),
                source_unchanged=(
                    initial_source == final_source
                    and final_source["sha256"]
                    == provenance.training_source_bundle_sha256
                    and final_evaluator["sha256"]
                    == provenance.evaluator_bundle_sha256
                ),
            )
            if torch.cuda.is_initialized():
                raise WallClockQualificationError(
                    "CUDA initialized during CPU latency qualification"
                )
            artifact = {
                "receipt": report.receipt(),
                "receipt_sha256": _json_sha256(
                    report.receipt(), domain=b"IRV21IWALLCLOCKRECEIPT\x01"
                ),
                "report": report.to_dict(),
                "report_sha256": report.sha256,
                "schema_version": SCHEMA_VERSION,
                "status": "passed" if report.passed else "failed",
            }
            _write_open_artifact(handle, artifact)
            if not report.passed:
                raise WallClockQualificationFailed(report)
            return report
        except WallClockQualificationFailed:
            raise
        except Exception as error:
            _write_open_artifact(
                handle,
                {
                    "failure": {
                        "message": str(error),
                        "type": type(error).__name__,
                    },
                    "qualification_id": QUALIFICATION_ID,
                    "request": request,
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                },
            )
            raise WallClockQualificationError(
                "wall-clock qualification aborted; failure evidence was written"
            ) from error


__all__ = [
    "DEVELOPMENT_ENVELOPE",
    "EPISODE_MAX_TICKS",
    "LOOP_P99_LIMIT_NS",
    "LOOP_P999_LIMIT_NS",
    "MAXIMUM_DEADLINE_MISS_RATE",
    "MEASURED_TICKS",
    "MODEL_P99_LIMIT_NS",
    "POST_DGX_ENVELOPE",
    "POST_DGX_RELEASE_QUALIFICATION_ID",
    "PostDgxReleaseReceipt",
    "QUALIFICATION_ID",
    "REPETITIONS",
    "RepetitionEvidence",
    "TICK_PERIOD_NS_60HZ",
    "TickMilestones",
    "WARMUP_TICKS",
    "WallClockQualificationConfig",
    "WallClockQualificationError",
    "WallClockQualificationFailed",
    "WallClockQualificationReport",
    "load_exact_v21i_checkpoint_cpu",
    "load_post_dgx_release_receipt",
    "qualify_v21i_checkpoint_wallclock_create_only",
    "validate_wallclock_artifact",
    "validate_wallclock_receipt",
    "validate_post_dgx_release_binding",
    "wallclock_config_sha256",
    "wallclock_evaluator_bundle_sha256",
    "wallclock_training_source_bundle_sha256",
    "wallclock_workload_sha256",
]
