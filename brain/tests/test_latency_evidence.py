from __future__ import annotations

import json
import unittest
from dataclasses import replace
from hashlib import sha256

from irene_brain.evaluation.latency_evidence import (
    LatencyAttempt,
    LatencyRunRegistration,
    summarize_latency,
)


def _h(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


_ATTEMPT_INTERVAL_NS = 10_000


def _registration(
    *,
    count: int,
    first_id: int = 0,
    kind: str = "desktop",
    submit_deadline_ns: int = 1_000,
    effect_deadline_ns: int | None = None,
    schedule_offsets_ns: tuple[int, ...] | None = None,
) -> LatencyRunRegistration:
    if schedule_offsets_ns is None:
        schedule_offsets_ns = tuple(
            index * _ATTEMPT_INTERVAL_NS for index in range(count)
        )
    return LatencyRunRegistration(
        schema_version=2,
        campaign_id="latency-campaign-v2",
        run_id=f"{kind}-run-0",
        measurement_kind=kind,
        architecture_variant_id="multi-thought-reference",
        architecture_identity_sha256=_h("architecture-identity"),
        architecture_manifest_sha256=_h("architecture-manifest"),
        checkpoint_sha256=_h("checkpoint"),
        model_config_sha256=_h("model-config"),
        evaluation_code_sha256=_h("evaluation-code"),
        workload_manifest_sha256=_h("workload"),
        measured_attempt_schedule_offsets_ns=schedule_offsets_ns,
        device_manifest_sha256=_h("device-manifest"),
        runtime_manifest_sha256=_h("runtime-manifest"),
        precision_mode="fp32",
        environment_id="offline-fixture-v1",
        clock_source="manual-monotonic-ns",
        clock_domain="manual-clock-domain-0",
        clock_synchronization="single-process-same-clock",
        capture_path="fixture-array-to-tensor",
        capture_width_px=160,
        capture_height_px=120,
        action_transport="fixture-action-sink",
        process_priority="normal",
        warmup_attempt_count=10,
        first_measured_attempt_id=first_id,
        measured_attempt_count=count,
        submit_deadline_ns=submit_deadline_ns,
        effect_deadline_ns=effect_deadline_ns,
    )


def _submitted(attempt_id: int, total_ns: int, *, start_ns: int = 1_000) -> LatencyAttempt:
    offset_ns = attempt_id * _ATTEMPT_INTERVAL_NS
    capture_started_ns = offset_ns + start_ns
    return LatencyAttempt(
        attempt_id=attempt_id,
        status="submitted",
        scheduled_ns=offset_ns,
        capture_started_ns=capture_started_ns,
        observation_ready_ns=capture_started_ns + 100,
        inference_started_ns=capture_started_ns + 150,
        inference_finished_ns=capture_started_ns + total_ns - 50,
        control_submitted_ns=capture_started_ns + total_ns,
    )


def _submission_failed(attempt_id: int, *, start_ns: int = 1_000) -> LatencyAttempt:
    offset_ns = attempt_id * _ATTEMPT_INTERVAL_NS
    capture_started_ns = offset_ns + start_ns
    return LatencyAttempt(
        attempt_id=attempt_id,
        status="submission_failed",
        scheduled_ns=offset_ns,
        capture_started_ns=capture_started_ns,
        observation_ready_ns=capture_started_ns + 100,
        inference_started_ns=capture_started_ns + 150,
        inference_finished_ns=capture_started_ns + 700,
        failure_reason="transport-error",
    )


def _inference_timeout(attempt_id: int, *, start_ns: int = 1_000) -> LatencyAttempt:
    offset_ns = attempt_id * _ATTEMPT_INTERVAL_NS
    capture_started_ns = offset_ns + start_ns
    return LatencyAttempt(
        attempt_id=attempt_id,
        status="inference_timeout",
        scheduled_ns=offset_ns,
        capture_started_ns=capture_started_ns,
        observation_ready_ns=capture_started_ns + 100,
        inference_started_ns=capture_started_ns + 150,
        failure_reason="deadline-expired",
    )


def _capture_failed(
    attempt_id: int,
    *,
    start_ns: int = 1_000,
    stimulus_ns: int | None = None,
) -> LatencyAttempt:
    offset_ns = attempt_id * _ATTEMPT_INTERVAL_NS
    return LatencyAttempt(
        attempt_id=attempt_id,
        status="capture_failed",
        scheduled_ns=offset_ns,
        stimulus_presented_ns=(
            None if stimulus_ns is None else offset_ns + stimulus_ns
        ),
        capture_started_ns=offset_ns + start_ns,
        failure_reason="capture-error",
    )


def _physical_attempt(
    attempt_id: int,
    *,
    status: str,
    stimulus_ns: int = 100,
    effect_ns: int | None = None,
) -> LatencyAttempt:
    offset_ns = attempt_id * _ATTEMPT_INTERVAL_NS
    return LatencyAttempt(
        attempt_id=attempt_id,
        status=status,
        scheduled_ns=offset_ns,
        stimulus_presented_ns=offset_ns + stimulus_ns,
        capture_started_ns=offset_ns + 120,
        observation_ready_ns=offset_ns + 200,
        inference_started_ns=offset_ns + 250,
        inference_finished_ns=offset_ns + 700,
        control_submitted_ns=offset_ns + 900,
        visible_effect_ns=None if effect_ns is None else offset_ns + effect_ns,
        failure_reason="effect-deadline-expired" if status == "effect_timeout" else None,
    )


def _summarize(
    registration: LatencyRunRegistration,
    attempts: tuple[LatencyAttempt, ...],
):
    return summarize_latency(
        registration,
        attempts,
        expected_registration_sha256=registration.sha256(),
    )


class LatencyEvidenceTests(unittest.TestCase):
    def test_registration_must_match_an_external_pin(self) -> None:
        registration = _registration(count=1)
        pinned = registration.sha256()
        altered = replace(registration, process_priority="high")
        with self.assertRaisesRegex(ValueError, "externally pinned"):
            summarize_latency(
                altered,
                (_submitted(0, 500),),
                expected_registration_sha256=pinned,
            )
        with self.assertRaisesRegex(ValueError, "externally pinned"):
            summarize_latency(
                registration,
                (_submitted(0, 500),),
                expected_registration_sha256=_h("different-pin"),
            )

    def test_attempt_ids_and_count_exactly_match_registered_schedule(self) -> None:
        registration = _registration(count=2, first_id=10)
        valid = (_submitted(10, 500), _submitted(11, 600))
        evidence = _summarize(registration, valid)
        self.assertEqual(evidence.attempt_count, 2)
        self.assertEqual(
            registration.measured_attempt_schedule_offsets_ns,
            (0, _ATTEMPT_INTERVAL_NS),
        )
        schedule_payload = {
            "schema_version": 2,
            "warmup_attempt_count": registration.warmup_attempt_count,
            "first_measured_attempt_id": registration.first_measured_attempt_id,
            "measured_attempt_count": registration.measured_attempt_count,
            "measured_attempt_schedule_offsets_ns": [0, _ATTEMPT_INTERVAL_NS],
        }
        canonical_schedule = json.dumps(
            schedule_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self.assertEqual(
            registration.attempt_schedule_sha256,
            sha256(canonical_schedule.encode("utf-8")).hexdigest(),
        )

        for invalid in (
            valid[:1],
            (valid[1], valid[0]),
            (valid[0], valid[0]),
        ):
            with self.subTest(ids=tuple(value.attempt_id for value in invalid)):
                with self.assertRaisesRegex(ValueError, "attempt"):
                    _summarize(registration, invalid)

        rescheduled = (
            valid[0],
            replace(valid[1], scheduled_ns=valid[1].scheduled_ns + 1),
        )
        with self.assertRaisesRegex(ValueError, "registered relative schedule"):
            _summarize(registration, rescheduled)
        altered_schedule = replace(
            registration,
            measured_attempt_schedule_offsets_ns=(0, _ATTEMPT_INTERVAL_NS + 1),
        )
        self.assertNotEqual(
            altered_schedule.attempt_schedule_sha256,
            registration.attempt_schedule_sha256,
        )
        self.assertNotEqual(altered_schedule.sha256(), registration.sha256())
        self.assertEqual(_summarize(altered_schedule, rescheduled).attempt_count, 2)

        with self.assertRaisesRegex(ValueError, "one offset per measured attempt"):
            _registration(count=2, schedule_offsets_ns=(0,))
        with self.assertRaisesRegex(ValueError, "first measured attempt"):
            _registration(count=1, schedule_offsets_ns=(1,))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _registration(count=2, schedule_offsets_ns=(0, 0))

    def test_statuses_enforce_consistent_optional_milestones(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires control_submitted_ns"):
            LatencyAttempt(
                attempt_id=0,
                status="submitted",
                scheduled_ns=0,
                capture_started_ns=1,
                observation_ready_ns=2,
                inference_started_ns=3,
                inference_finished_ns=4,
            )
        with self.assertRaisesRegex(ValueError, "cannot include observation_ready_ns"):
            LatencyAttempt(
                attempt_id=0,
                status="capture_failed",
                scheduled_ns=0,
                capture_started_ns=1,
                observation_ready_ns=2,
                failure_reason="capture-error",
            )
        with self.assertRaisesRegex(ValueError, "failure_reason"):
            LatencyAttempt(
                attempt_id=0,
                status="inference_timeout",
                scheduled_ns=0,
                capture_started_ns=1,
                observation_ready_ns=2,
                inference_started_ns=3,
            )

    def test_missing_submissions_are_deadline_misses_and_not_dropped(self) -> None:
        registration = _registration(count=4, submit_deadline_ns=1_000)
        attempts = (
            _submitted(0, 1_000),
            _submission_failed(1),
            _inference_timeout(2),
            _capture_failed(3),
        )
        evidence = _summarize(registration, attempts)
        self.assertEqual(evidence.submitted_attempt_count, 1)
        self.assertEqual(evidence.submit_deadline_misses, 3)
        self.assertEqual(evidence.submit_deadline_miss_rate, 0.75)
        self.assertEqual(evidence.capture_to_submit.sample_count, 1)
        self.assertEqual(dict(evidence.status_counts)["capture_failed"], 1)
        self.assertEqual(dict(evidence.status_counts)["inference_timeout"], 1)
        self.assertEqual(dict(evidence.status_counts)["submission_failed"], 1)

    def test_capture_start_is_desktop_latency_origin_and_tail_is_nearest_rank(self) -> None:
        registration = _registration(count=100, submit_deadline_ns=395)
        attempts = tuple(
            _submitted(index, value) for index, value in enumerate(range(301, 401))
        )
        evidence = _summarize(registration, attempts)
        self.assertEqual(evidence.capture_to_submit.p50_ns, 350)
        self.assertEqual(evidence.capture_to_submit.p95_ns, 395)
        self.assertEqual(evidence.capture_to_submit.p99_ns, 399)
        self.assertEqual(evidence.capture_to_submit.maximum_ns, 400)
        self.assertEqual(evidence.submit_deadline_misses, 5)

        capture_inclusive = _registration(count=1, submit_deadline_ns=400)
        attempt = LatencyAttempt(
            attempt_id=0,
            status="submitted",
            scheduled_ns=90,
            capture_started_ns=100,
            observation_ready_ns=300,
            inference_started_ns=350,
            inference_finished_ns=550,
            control_submitted_ns=600,
        )
        inclusive_evidence = _summarize(capture_inclusive, (attempt,))
        self.assertEqual(inclusive_evidence.capture.minimum_ns, 200)
        self.assertEqual(inclusive_evidence.capture_to_submit.minimum_ns, 500)
        self.assertEqual(inclusive_evidence.submit_deadline_misses, 1)

    def test_physical_scope_requires_complete_effect_outcomes(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires effect_deadline_ns"):
            _registration(count=1, kind="physical")

        registration = _registration(
            count=4,
            kind="physical",
            submit_deadline_ns=1_000,
            effect_deadline_ns=1_000,
        )
        attempts = (
            _physical_attempt(0, status="effect_observed", effect_ns=1_100),
            _physical_attempt(1, status="effect_observed", effect_ns=1_101),
            _physical_attempt(2, status="effect_timeout"),
            _capture_failed(3, start_ns=120, stimulus_ns=100),
        )
        evidence = _summarize(registration, attempts)
        self.assertEqual(evidence.effect_observed_sample_count, 2)
        self.assertEqual(evidence.effect_deadline_misses, 3)
        self.assertEqual(evidence.effect_deadline_miss_rate, 0.75)
        self.assertEqual(evidence.submit_deadline_misses, 1)
        self.assertEqual(evidence.stimulus_to_visible_effect.sample_count, 2)

        submitted_without_effect_outcome = replace(
            _physical_attempt(0, status="effect_observed", effect_ns=1_100),
            status="submitted",
            visible_effect_ns=None,
        )
        one_registration = _registration(
            count=1,
            kind="physical",
            effect_deadline_ns=1_000,
        )
        with self.assertRaisesRegex(ValueError, "effect_observed or effect_timeout"):
            _summarize(one_registration, (submitted_without_effect_outcome,))

        missing_stimulus = replace(
            _physical_attempt(0, status="effect_timeout"),
            stimulus_presented_ns=None,
        )
        with self.assertRaisesRegex(ValueError, "stimulus_presented_ns"):
            _summarize(one_registration, (missing_stimulus,))

    def test_nonphysical_scope_rejects_effect_records(self) -> None:
        registration = _registration(count=1)
        effect_attempt = _physical_attempt(
            0,
            status="effect_observed",
            effect_ns=1_100,
        )
        with self.assertRaisesRegex(ValueError, "physical measurement"):
            _summarize(registration, (effect_attempt,))

    def test_raw_trace_report_hashes_and_evidence_invariants_fail_closed(self) -> None:
        registration = _registration(count=1)
        evidence = _summarize(registration, (_submitted(0, 500),))
        record = evidence.to_dict()
        report_digest = record.pop("report_sha256")
        canonical_report = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self.assertEqual(report_digest, sha256(canonical_report.encode()).hexdigest())
        raw_payload = {
            "schema_version": 2,
            "attempts": [attempt.to_dict() for attempt in evidence.attempts],
        }
        canonical_raw = json.dumps(
            raw_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self.assertEqual(
            evidence.raw_trace_sha256,
            sha256(canonical_raw.encode()).hexdigest(),
        )

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            replace(evidence, submit_deadline_misses=1)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            replace(evidence, raw_trace_sha256=_h("forged-raw-trace"))
        with self.assertRaisesRegex(ValueError, "report_sha256"):
            replace(evidence, report_sha256=_h("forged-report"))
        with self.assertRaisesRegex(ValueError, "attempt_count.*inconsistent"):
            replace(evidence, attempt_count=1.0)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            replace(evidence, schema_version=2.0)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            replace(registration, schema_version=2.0)
        with self.assertRaisesRegex(ValueError, "attempt_id"):
            replace(evidence.attempts[0], attempt_id=0.0)
        with self.assertRaisesRegex(ValueError, "submit_deadline_ns"):
            replace(registration, submit_deadline_ns=1_000.0)


if __name__ == "__main__":
    unittest.main()
