"""Closed-loop play latency evidence: extend latency records from benchmarks to play.

The closed-loop evaluator drives an in-repo world on a manual clock with a
declared simulated inference latency, so every decision has exact simulated
timing milestones. This module converts those milestones into the
:class:`~irene_brain.evaluation.latency_evidence.LatencyAttempt` stream that
:func:`~irene_brain.evaluation.latency_evidence.summarize_latency` validates
against an externally pinned registration — the same fail-closed evidence
contract as the benchmark harness, now covering play.

The decision schedule is analytic: decision *k* is scheduled at tick
``k * max(decision_interval_ns, inference_latency_ns)`` because the clock
advances by the inference latency and then jumps forward to the next
decision interval boundary. Warmup decisions shift attempt IDs, not the
relative offsets. Simulated capture and queueing are free, so the capture
and queue legs are exactly zero; a rejected envelope becomes a
``submission_failed`` attempt carrying the driver's reason string.
"""

from __future__ import annotations

from typing import Callable

from .closed_loop_play import (
    ClosedLoopEpisodeReport,
    ClosedLoopPlayConfig,
    DecisionTiming,
    _run_episode_core,
    control_audit_stats,
)
from .latency_evidence import (
    LatencyAttempt,
    LatencyEvidence,
    LatencyRunRegistration,
    summarize_latency,
)


def play_decision_schedule_offsets_ns(
    *,
    config: ClosedLoopPlayConfig,
    measured_attempt_count: int,
) -> tuple[int, ...]:
    """Return the analytic measured-attempt schedule for a play registration.

    The offsets are policy-independent and therefore pinnable before
    measurement: decision *k* starts at ``k * stride`` where the stride is
    the larger of the decision interval and the declared inference latency.
    """

    if not isinstance(config, ClosedLoopPlayConfig):
        raise ValueError("config must be a ClosedLoopPlayConfig")
    if (
        isinstance(measured_attempt_count, bool)
        or not isinstance(measured_attempt_count, int)
        or measured_attempt_count < 1
    ):
        raise ValueError("measured_attempt_count must be a positive integer")
    stride = max(config.decision_interval_ns, config.inference_latency_ns)
    return tuple(index * stride for index in range(measured_attempt_count))


def measure_policy_play_latency(
    policy: object,
    *,
    seed: int,
    config: ClosedLoopPlayConfig,
    registration: LatencyRunRegistration,
    expected_registration_sha256: str,
    environment_factory: Callable[[], object] | None = None,
) -> tuple[ClosedLoopEpisodeReport, LatencyEvidence]:
    """Play one episode and return its report plus validated latency evidence.

    The registration must be pinned before the run: its measurement kind
    must be ``simulated``, its first measured attempt ID must equal
    ``warmup_attempt_count + 1`` (decision sequence numbers start at one),
    and its schedule offsets must equal the analytic play schedule. The
    episode must survive every registered measured decision; a shorter
    episode is an error, never a truncated stream.
    """

    if not isinstance(registration, LatencyRunRegistration):
        raise ValueError("registration must be a LatencyRunRegistration")
    if registration.measurement_kind != "simulated":
        raise ValueError("play latency evidence requires simulated measurement")
    for attribute in ("reset", "act"):
        if not callable(getattr(policy, attribute, None)):
            raise TypeError(f"policy must define {attribute}()")
    warmup = registration.warmup_attempt_count
    measured = registration.measured_attempt_count
    if registration.first_measured_attempt_id != warmup + 1:
        raise ValueError(
            "first_measured_attempt_id must equal warmup_attempt_count + 1: "
            "play decision sequence numbers start at one"
        )
    expected_offsets = play_decision_schedule_offsets_ns(
        config=config,
        measured_attempt_count=measured,
    )
    if registration.measured_attempt_schedule_offsets_ns != expected_offsets:
        raise ValueError(
            "registration schedule does not match the analytic play schedule"
        )

    policy.reset(seed)

    def bind(environment: object) -> None:
        if getattr(policy, "uses_privileged_state", False):
            policy.bind(environment)

    def decide(
        observation: object, elapsed_seconds: float
    ) -> tuple[object, dict[str, int | float], None]:
        control = policy.act(observation)
        return control, control_audit_stats(control), None

    timings: list[DecisionTiming] = []

    def sink(timing: DecisionTiming) -> None:
        if warmup < timing.action_sequence <= warmup + measured:
            timings.append(timing)

    report = _run_episode_core(
        seed=seed,
        config=config,
        decide=decide,
        on_environment=bind,
        environment_factory=environment_factory,
        attempt_sink=sink,
    )
    if len(timings) != measured:
        raise ValueError(
            "the episode ended before the registered measured stream completed "
            f"({len(timings)} of {measured} attempts)"
        )

    attempts = tuple(
        LatencyAttempt(
            attempt_id=timing.action_sequence,
            status="submitted" if timing.submitted else "submission_failed",
            scheduled_ns=timing.scheduled_ns,
            capture_started_ns=timing.scheduled_ns,
            observation_ready_ns=timing.scheduled_ns,
            inference_started_ns=timing.scheduled_ns,
            inference_finished_ns=timing.inference_finished_ns,
            control_submitted_ns=(
                timing.inference_finished_ns if timing.submitted else None
            ),
            failure_reason=timing.failure_reason,
        )
        for timing in timings
    )
    evidence = summarize_latency(
        registration,
        attempts,
        expected_registration_sha256=expected_registration_sha256,
    )
    return report, evidence


__all__ = [
    "measure_policy_play_latency",
    "play_decision_schedule_offsets_ns",
]
