"""Pure, fail-closed evidence for deadline-bound sensorimotor latency.

The module performs no capture, input injection, clock access, network access,
or accelerator work. It validates an already-recorded, preregistered attempt
stream. Failed attempts remain in the denominator instead of disappearing
from latency claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from hmac import compare_digest
from itertools import pairwise
from math import ceil, isfinite
from typing import Iterable, Mapping, cast


_SCHEMA_VERSION = 2
_MEASUREMENT_KINDS = frozenset({"simulated", "desktop", "physical"})
_ATTEMPT_STATUSES = (
    "capture_failed",
    "inference_timeout",
    "submission_failed",
    "submitted",
    "effect_observed",
    "effect_timeout",
)
_FAILURE_STATUSES = frozenset(
    {"capture_failed", "inference_timeout", "submission_failed", "effect_timeout"}
)
_EFFECT_STATUSES = frozenset({"effect_observed", "effect_timeout"})


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _payload_sha256(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, whitespace-normalized string")
    return value


def _digest(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if len(text) != 64 or text != text.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    return text


@dataclass(frozen=True, slots=True)
class LatencyRunRegistration:
    """Exact provenance and schedule that must be pinned before measurement.

    ``sha256()`` is intentionally not stored inside this object. The caller
    must supply the previously archived digest to :func:`summarize_latency`, so
    editing and re-hashing a registration after seeing results fails closed.
    """

    schema_version: int
    campaign_id: str
    run_id: str
    measurement_kind: str
    architecture_variant_id: str
    architecture_identity_sha256: str
    architecture_manifest_sha256: str
    checkpoint_sha256: str
    model_config_sha256: str
    evaluation_code_sha256: str
    workload_manifest_sha256: str
    measured_attempt_schedule_offsets_ns: tuple[int, ...]
    attempt_schedule_sha256: str = field(init=False)
    device_manifest_sha256: str
    runtime_manifest_sha256: str
    precision_mode: str
    environment_id: str
    clock_source: str
    clock_domain: str
    clock_synchronization: str
    capture_path: str
    capture_width_px: int
    capture_height_px: int
    action_transport: str
    process_priority: str
    warmup_attempt_count: int
    first_measured_attempt_id: int
    measured_attempt_count: int
    submit_deadline_ns: int
    effect_deadline_ns: int | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        for name in (
            "campaign_id",
            "run_id",
            "architecture_variant_id",
            "precision_mode",
            "environment_id",
            "clock_source",
            "clock_domain",
            "clock_synchronization",
            "capture_path",
            "action_transport",
            "process_priority",
        ):
            _text(getattr(self, name), name=name)
        if (
            type(self.measurement_kind) is not str
            or self.measurement_kind not in _MEASUREMENT_KINDS
        ):
            raise ValueError("measurement_kind must be simulated, desktop, or physical")
        for name in (
            "architecture_identity_sha256",
            "architecture_manifest_sha256",
            "checkpoint_sha256",
            "model_config_sha256",
            "evaluation_code_sha256",
            "workload_manifest_sha256",
            "device_manifest_sha256",
            "runtime_manifest_sha256",
        ):
            _digest(getattr(self, name), name=name)
        _positive_int(self.capture_width_px, name="capture_width_px")
        _positive_int(self.capture_height_px, name="capture_height_px")
        _nonnegative_int(self.warmup_attempt_count, name="warmup_attempt_count")
        _nonnegative_int(
            self.first_measured_attempt_id,
            name="first_measured_attempt_id",
        )
        _positive_int(self.measured_attempt_count, name="measured_attempt_count")
        offsets = self.measured_attempt_schedule_offsets_ns
        if type(offsets) is not tuple:
            raise ValueError("measured_attempt_schedule_offsets_ns must be a tuple")
        if len(offsets) != self.measured_attempt_count:
            raise ValueError(
                "measured_attempt_schedule_offsets_ns must contain one offset per "
                "measured attempt"
            )
        for index, offset in enumerate(offsets):
            _nonnegative_int(
                offset,
                name=f"measured_attempt_schedule_offsets_ns[{index}]",
            )
        if offsets[0] != 0:
            raise ValueError("the first measured attempt schedule offset must be zero")
        if any(current <= previous for previous, current in pairwise(offsets)):
            raise ValueError("measured attempt schedule offsets must be strictly increasing")
        schedule_payload = {
            "schema_version": _SCHEMA_VERSION,
            "warmup_attempt_count": self.warmup_attempt_count,
            "first_measured_attempt_id": self.first_measured_attempt_id,
            "measured_attempt_count": self.measured_attempt_count,
            "measured_attempt_schedule_offsets_ns": list(offsets),
        }
        object.__setattr__(
            self,
            "attempt_schedule_sha256",
            _payload_sha256(schedule_payload),
        )
        _positive_int(self.submit_deadline_ns, name="submit_deadline_ns")
        if self.measurement_kind == "physical":
            if self.effect_deadline_ns is None:
                raise ValueError("physical registration requires effect_deadline_ns")
            _positive_int(self.effect_deadline_ns, name="effect_deadline_ns")
        elif self.effect_deadline_ns is not None:
            raise ValueError("effect_deadline_ns is only valid for physical measurement")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "measurement_kind": self.measurement_kind,
            "architecture_variant_id": self.architecture_variant_id,
            "architecture_identity_sha256": self.architecture_identity_sha256,
            "architecture_manifest_sha256": self.architecture_manifest_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_config_sha256": self.model_config_sha256,
            "evaluation_code_sha256": self.evaluation_code_sha256,
            "workload_manifest_sha256": self.workload_manifest_sha256,
            "measured_attempt_schedule_offsets_ns": list(
                self.measured_attempt_schedule_offsets_ns
            ),
            "attempt_schedule_sha256": self.attempt_schedule_sha256,
            "device_manifest_sha256": self.device_manifest_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "precision_mode": self.precision_mode,
            "environment_id": self.environment_id,
            "clock_source": self.clock_source,
            "clock_domain": self.clock_domain,
            "clock_synchronization": self.clock_synchronization,
            "capture_path": self.capture_path,
            "capture_width_px": self.capture_width_px,
            "capture_height_px": self.capture_height_px,
            "action_transport": self.action_transport,
            "process_priority": self.process_priority,
            "warmup_attempt_count": self.warmup_attempt_count,
            "first_measured_attempt_id": self.first_measured_attempt_id,
            "measured_attempt_count": self.measured_attempt_count,
            "submit_deadline_ns": self.submit_deadline_ns,
            "effect_deadline_ns": self.effect_deadline_ns,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LatencyAttempt:
    """One registered measured attempt in one monotonic clock domain.

    ``capture_started_ns`` is mandatory even when capture fails. Therefore a
    desktop claim starts when capture work begins, not when a frame later
    becomes available. Optional milestones are governed by ``status``.
    """

    attempt_id: int
    status: str
    scheduled_ns: int
    capture_started_ns: int
    stimulus_presented_ns: int | None = None
    observation_ready_ns: int | None = None
    inference_started_ns: int | None = None
    inference_finished_ns: int | None = None
    control_submitted_ns: int | None = None
    visible_effect_ns: int | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.attempt_id, name="attempt_id")
        if type(self.status) is not str or self.status not in _ATTEMPT_STATUSES:
            raise ValueError(f"status must be one of {', '.join(_ATTEMPT_STATUSES)}")
        scheduled = _nonnegative_int(self.scheduled_ns, name="scheduled_ns")
        capture = _nonnegative_int(self.capture_started_ns, name="capture_started_ns")
        if capture < scheduled:
            raise ValueError("capture_started_ns cannot precede scheduled_ns")

        optional_names = (
            "stimulus_presented_ns",
            "observation_ready_ns",
            "inference_started_ns",
            "inference_finished_ns",
            "control_submitted_ns",
            "visible_effect_ns",
        )
        for name in optional_names:
            value = getattr(self, name)
            if value is not None:
                _nonnegative_int(value, name=name)
        if self.stimulus_presented_ns is not None:
            if self.stimulus_presented_ns < scheduled:
                raise ValueError("stimulus_presented_ns cannot precede scheduled_ns")

        required_by_status = {
            "capture_failed": (),
            "inference_timeout": ("observation_ready_ns", "inference_started_ns"),
            "submission_failed": (
                "observation_ready_ns",
                "inference_started_ns",
                "inference_finished_ns",
            ),
            "submitted": (
                "observation_ready_ns",
                "inference_started_ns",
                "inference_finished_ns",
                "control_submitted_ns",
            ),
            "effect_observed": (
                "observation_ready_ns",
                "inference_started_ns",
                "inference_finished_ns",
                "control_submitted_ns",
                "visible_effect_ns",
            ),
            "effect_timeout": (
                "observation_ready_ns",
                "inference_started_ns",
                "inference_finished_ns",
                "control_submitted_ns",
            ),
        }
        allowed_by_status = {
            "capture_failed": frozenset(),
            "inference_timeout": frozenset(
                {"observation_ready_ns", "inference_started_ns"}
            ),
            "submission_failed": frozenset(
                {
                    "observation_ready_ns",
                    "inference_started_ns",
                    "inference_finished_ns",
                }
            ),
            "submitted": frozenset(
                {
                    "observation_ready_ns",
                    "inference_started_ns",
                    "inference_finished_ns",
                    "control_submitted_ns",
                }
            ),
            "effect_observed": frozenset(
                {
                    "observation_ready_ns",
                    "inference_started_ns",
                    "inference_finished_ns",
                    "control_submitted_ns",
                    "visible_effect_ns",
                }
            ),
            "effect_timeout": frozenset(
                {
                    "observation_ready_ns",
                    "inference_started_ns",
                    "inference_finished_ns",
                    "control_submitted_ns",
                }
            ),
        }
        for name in required_by_status[self.status]:
            if getattr(self, name) is None:
                raise ValueError(f"{self.status} requires {name}")
        allowed = allowed_by_status[self.status]
        for name in optional_names[1:]:
            if getattr(self, name) is not None and name not in allowed:
                raise ValueError(f"{self.status} cannot include {name}")

        pipeline_names = (
            "capture_started_ns",
            "observation_ready_ns",
            "inference_started_ns",
            "inference_finished_ns",
            "control_submitted_ns",
            "visible_effect_ns",
        )
        present = tuple(
            getattr(self, name)
            for name in pipeline_names
            if getattr(self, name) is not None
        )
        if tuple(sorted(present)) != present:
            raise ValueError("attempt milestones must be monotonically nondecreasing")
        if self.status in _FAILURE_STATUSES:
            _text(self.failure_reason, name="failure_reason")
        elif self.failure_reason is not None:
            raise ValueError(f"{self.status} cannot include failure_reason")

    @property
    def capture_ns(self) -> int | None:
        if self.observation_ready_ns is None:
            return None
        return self.observation_ready_ns - self.capture_started_ns

    @property
    def queue_ns(self) -> int | None:
        if self.observation_ready_ns is None or self.inference_started_ns is None:
            return None
        return self.inference_started_ns - self.observation_ready_ns

    @property
    def inference_ns(self) -> int | None:
        if self.inference_started_ns is None or self.inference_finished_ns is None:
            return None
        return self.inference_finished_ns - self.inference_started_ns

    @property
    def submission_ns(self) -> int | None:
        if self.inference_finished_ns is None or self.control_submitted_ns is None:
            return None
        return self.control_submitted_ns - self.inference_finished_ns

    @property
    def capture_to_submit_ns(self) -> int | None:
        if self.control_submitted_ns is None:
            return None
        return self.control_submitted_ns - self.capture_started_ns

    @property
    def stimulus_to_effect_ns(self) -> int | None:
        if self.stimulus_presented_ns is None or self.visible_effect_ns is None:
            return None
        return self.visible_effect_ns - self.stimulus_presented_ns

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "status": self.status,
            "scheduled_ns": self.scheduled_ns,
            "stimulus_presented_ns": self.stimulus_presented_ns,
            "capture_started_ns": self.capture_started_ns,
            "observation_ready_ns": self.observation_ready_ns,
            "inference_started_ns": self.inference_started_ns,
            "inference_finished_ns": self.inference_finished_ns,
            "control_submitted_ns": self.control_submitted_ns,
            "visible_effect_ns": self.visible_effect_ns,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    sample_count: int
    minimum_ns: int
    p50_ns: int
    p95_ns: int
    p99_ns: int
    maximum_ns: int
    mean_ns: float

    def __post_init__(self) -> None:
        _positive_int(self.sample_count, name="sample_count")
        values = (
            self.minimum_ns,
            self.p50_ns,
            self.p95_ns,
            self.p99_ns,
            self.maximum_ns,
        )
        for index, value in enumerate(values):
            _nonnegative_int(value, name=f"distribution[{index}]")
        if tuple(sorted(values)) != values:
            raise ValueError("distribution quantiles must be nondecreasing")
        if type(self.mean_ns) is not float:
            raise ValueError("mean_ns must be finite")
        if not isfinite(self.mean_ns):
            raise ValueError("mean_ns must be finite")
        if self.mean_ns < self.minimum_ns or self.mean_ns > self.maximum_ns:
            raise ValueError("mean_ns must lie within the observed range")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "minimum_ns": self.minimum_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "maximum_ns": self.maximum_ns,
            "mean_ns": self.mean_ns,
        }


def _nearest_rank(values: tuple[int, ...], probability: float) -> int:
    if not values:
        raise ValueError("a latency distribution cannot be empty")
    if probability <= 0.0 or probability > 1.0:
        raise ValueError("probability must be in (0, 1]")
    return values[max(0, ceil(probability * len(values)) - 1)]


def _distribution(values: Iterable[int | None]) -> DistributionSummary | None:
    ordered = tuple(
        sorted(
            _nonnegative_int(value, name="latency")
            for value in values
            if value is not None
        )
    )
    if not ordered:
        return None
    return DistributionSummary(
        sample_count=len(ordered),
        minimum_ns=ordered[0],
        p50_ns=_nearest_rank(ordered, 0.50),
        p95_ns=_nearest_rank(ordered, 0.95),
        p99_ns=_nearest_rank(ordered, 0.99),
        maximum_ns=ordered[-1],
        mean_ns=sum(ordered) / len(ordered),
    )


def _summary_dict(summary: DistributionSummary | None) -> dict[str, object] | None:
    return None if summary is None else summary.to_dict()


def _strict_equal(actual: object, expected: object) -> bool:
    """Compare report fields without JSON's numeric type equivalence loopholes."""

    if type(actual) is not type(expected):
        return False
    if type(expected) is tuple:
        actual_tuple = cast(tuple[object, ...], actual)
        expected_tuple = cast(tuple[object, ...], expected)
        return len(actual_tuple) == len(expected_tuple) and all(
            _strict_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(
                actual_tuple,
                expected_tuple,
                strict=True,
            )
        )
    if type(expected) is DistributionSummary:
        actual_summary = cast(DistributionSummary, actual)
        expected_summary = cast(DistributionSummary, expected)
        return _canonical_json(actual_summary.to_dict()) == _canonical_json(
            expected_summary.to_dict()
        )
    if type(expected) is float:
        return _canonical_json(actual) == _canonical_json(expected)
    return actual == expected


def _validate_attempt_stream(
    registration: LatencyRunRegistration,
    attempts: tuple[LatencyAttempt, ...],
) -> None:
    if len(attempts) != registration.measured_attempt_count:
        raise ValueError(
            "attempt stream must contain every registered measured attempt exactly once"
        )
    expected_ids = tuple(
        range(
            registration.first_measured_attempt_id,
            registration.first_measured_attempt_id + registration.measured_attempt_count,
        )
    )
    actual_ids = tuple(attempt.attempt_id for attempt in attempts)
    if actual_ids != expected_ids:
        raise ValueError("attempt IDs must exactly match the registered contiguous order")
    schedule_origin_ns = attempts[0].scheduled_ns
    actual_offsets = tuple(
        attempt.scheduled_ns - schedule_origin_ns for attempt in attempts
    )
    if actual_offsets != registration.measured_attempt_schedule_offsets_ns:
        raise ValueError(
            "attempt scheduled_ns values must exactly match the registered relative schedule"
        )

    for attempt in attempts:
        if registration.measurement_kind == "physical":
            if attempt.stimulus_presented_ns is None:
                raise ValueError("every physical attempt requires stimulus_presented_ns")
            if attempt.status == "submitted":
                raise ValueError(
                    "physical submitted attempts require effect_observed or effect_timeout"
                )
            if attempt.observation_ready_ns is not None:
                if attempt.stimulus_presented_ns > attempt.observation_ready_ns:
                    raise ValueError(
                        "physical stimulus_presented_ns cannot follow observation readiness"
                    )
            if attempt.visible_effect_ns is not None:
                if attempt.visible_effect_ns < attempt.stimulus_presented_ns:
                    raise ValueError("visible effect cannot precede the physical stimulus")
        else:
            if attempt.stimulus_presented_ns is not None:
                raise ValueError(
                    "stimulus_presented_ns is reserved for physical measurement"
                )
            if attempt.status in _EFFECT_STATUSES or attempt.visible_effect_ns is not None:
                raise ValueError("effect outcomes are reserved for physical measurement")


def _derive(
    registration: LatencyRunRegistration,
    attempts: tuple[LatencyAttempt, ...],
) -> dict[str, object]:
    _validate_attempt_stream(registration, attempts)
    status_counts = tuple(
        (status, sum(attempt.status == status for attempt in attempts))
        for status in _ATTEMPT_STATUSES
    )
    submit_values = tuple(attempt.capture_to_submit_ns for attempt in attempts)
    submitted_attempt_count = sum(value is not None for value in submit_values)
    submit_deadline_misses = sum(
        value is None or value > registration.submit_deadline_ns
        for value in submit_values
    )

    effect_observed_sample_count = sum(
        attempt.status == "effect_observed" for attempt in attempts
    )
    effect_deadline_misses: int | None = None
    effect_deadline_miss_rate: float | None = None
    if registration.measurement_kind == "physical":
        assert registration.effect_deadline_ns is not None
        effect_deadline_misses = sum(
            (value := attempt.stimulus_to_effect_ns) is None
            or value > registration.effect_deadline_ns
            for attempt in attempts
        )
        effect_deadline_miss_rate = effect_deadline_misses / len(attempts)

    raw_trace_sha256 = _payload_sha256(
        {
            "schema_version": _SCHEMA_VERSION,
            "attempts": [attempt.to_dict() for attempt in attempts],
        }
    )
    return {
        "raw_trace_sha256": raw_trace_sha256,
        "attempt_count": len(attempts),
        "status_counts": status_counts,
        "submitted_attempt_count": submitted_attempt_count,
        "submit_deadline_misses": submit_deadline_misses,
        "submit_deadline_miss_rate": submit_deadline_misses / len(attempts),
        "effect_observed_sample_count": effect_observed_sample_count,
        "effect_deadline_misses": effect_deadline_misses,
        "effect_deadline_miss_rate": effect_deadline_miss_rate,
        "capture": _distribution(attempt.capture_ns for attempt in attempts),
        "queue": _distribution(attempt.queue_ns for attempt in attempts),
        "inference": _distribution(attempt.inference_ns for attempt in attempts),
        "submission": _distribution(attempt.submission_ns for attempt in attempts),
        "capture_to_submit": _distribution(submit_values),
        "stimulus_to_visible_effect": _distribution(
            attempt.stimulus_to_effect_ns for attempt in attempts
        ),
    }


def _report_payload(
    *,
    registration: LatencyRunRegistration,
    registration_sha256: str,
    attempts: tuple[LatencyAttempt, ...],
    derived: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "registration": registration.to_dict(),
        "registration_sha256": registration_sha256,
        "raw_trace_sha256": derived["raw_trace_sha256"],
        "attempts": [attempt.to_dict() for attempt in attempts],
        "attempt_count": derived["attempt_count"],
        "status_counts": dict(derived["status_counts"]),
        "submitted_attempt_count": derived["submitted_attempt_count"],
        "submit_deadline_misses": derived["submit_deadline_misses"],
        "submit_deadline_miss_rate": derived["submit_deadline_miss_rate"],
        "effect_observed_sample_count": derived["effect_observed_sample_count"],
        "effect_deadline_misses": derived["effect_deadline_misses"],
        "effect_deadline_miss_rate": derived["effect_deadline_miss_rate"],
        "capture": _summary_dict(derived["capture"]),
        "queue": _summary_dict(derived["queue"]),
        "inference": _summary_dict(derived["inference"]),
        "submission": _summary_dict(derived["submission"]),
        "capture_to_submit": _summary_dict(derived["capture_to_submit"]),
        "stimulus_to_visible_effect": _summary_dict(
            derived["stimulus_to_visible_effect"]
        ),
    }


@dataclass(frozen=True, slots=True)
class LatencyEvidence:
    """Self-consistent report containing its complete measured attempt stream."""

    schema_version: int
    registration: LatencyRunRegistration
    registration_sha256: str
    attempts: tuple[LatencyAttempt, ...]
    raw_trace_sha256: str
    attempt_count: int
    status_counts: tuple[tuple[str, int], ...]
    submitted_attempt_count: int
    submit_deadline_misses: int
    submit_deadline_miss_rate: float
    effect_observed_sample_count: int
    effect_deadline_misses: int | None
    effect_deadline_miss_rate: float | None
    capture: DistributionSummary | None
    queue: DistributionSummary | None
    inference: DistributionSummary | None
    submission: DistributionSummary | None
    capture_to_submit: DistributionSummary | None
    stimulus_to_visible_effect: DistributionSummary | None
    report_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        if type(self.registration) is not LatencyRunRegistration:
            raise ValueError("registration must be a LatencyRunRegistration")
        if type(self.attempts) is not tuple or not all(
            type(attempt) is LatencyAttempt for attempt in self.attempts
        ):
            raise ValueError("attempts must be a tuple of LatencyAttempt values")
        registration_digest = _digest(
            self.registration_sha256,
            name="registration_sha256",
        )
        if not compare_digest(registration_digest, self.registration.sha256()):
            raise ValueError("registration_sha256 does not match registration")
        _digest(self.raw_trace_sha256, name="raw_trace_sha256")
        _digest(self.report_sha256, name="report_sha256")

        derived = _derive(self.registration, self.attempts)
        for name, expected in derived.items():
            if not _strict_equal(getattr(self, name), expected):
                raise ValueError(f"{name} is inconsistent with the raw attempt stream")
        stored_derived = {name: getattr(self, name) for name in _DERIVED_FIELDS}
        payload = _report_payload(
            registration=self.registration,
            registration_sha256=self.registration_sha256,
            attempts=self.attempts,
            derived=stored_derived,
        )
        if not compare_digest(self.report_sha256, _payload_sha256(payload)):
            raise ValueError("report_sha256 does not match the canonical report")

    def to_dict(self) -> dict[str, object]:
        derived = {name: getattr(self, name) for name in _DERIVED_FIELDS}
        result = _report_payload(
            registration=self.registration,
            registration_sha256=self.registration_sha256,
            attempts=self.attempts,
            derived=derived,
        )
        result["report_sha256"] = self.report_sha256
        return result


_DERIVED_FIELDS = (
    "raw_trace_sha256",
    "attempt_count",
    "status_counts",
    "submitted_attempt_count",
    "submit_deadline_misses",
    "submit_deadline_miss_rate",
    "effect_observed_sample_count",
    "effect_deadline_misses",
    "effect_deadline_miss_rate",
    "capture",
    "queue",
    "inference",
    "submission",
    "capture_to_submit",
    "stimulus_to_visible_effect",
)


def summarize_latency(
    registration: LatencyRunRegistration,
    attempts: Iterable[LatencyAttempt],
    *,
    expected_registration_sha256: str,
) -> LatencyEvidence:
    """Validate a complete stream against an externally pinned registration."""

    if type(registration) is not LatencyRunRegistration:
        raise ValueError("registration must be a LatencyRunRegistration")
    pinned_digest = _digest(
        expected_registration_sha256,
        name="expected_registration_sha256",
    )
    actual_digest = registration.sha256()
    if not compare_digest(pinned_digest, actual_digest):
        raise ValueError("registration does not match the externally pinned digest")
    materialized = tuple(attempts)
    if not all(type(attempt) is LatencyAttempt for attempt in materialized):
        raise ValueError("attempts must contain only LatencyAttempt values")
    derived = _derive(registration, materialized)
    payload = _report_payload(
        registration=registration,
        registration_sha256=pinned_digest,
        attempts=materialized,
        derived=derived,
    )
    return LatencyEvidence(
        schema_version=_SCHEMA_VERSION,
        registration=registration,
        registration_sha256=pinned_digest,
        attempts=materialized,
        report_sha256=_payload_sha256(payload),
        **derived,
    )


__all__ = [
    "DistributionSummary",
    "LatencyAttempt",
    "LatencyEvidence",
    "LatencyRunRegistration",
    "summarize_latency",
]
