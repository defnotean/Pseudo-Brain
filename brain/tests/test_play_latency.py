from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.diagnostic_policies import NoOpPolicy
from irene_brain.evaluation.latency_evidence import LatencyRunRegistration
from irene_brain.evaluation.play_latency import (
    measure_policy_play_latency,
    play_decision_schedule_offsets_ns,
)

_TICK = 16_666_667


def _h(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _registration(
    *,
    config: ClosedLoopPlayConfig,
    warmup: int,
    measured: int,
    submit_deadline_ns: int,
    first_id: int | None = None,
    offsets: tuple[int, ...] | None = None,
) -> LatencyRunRegistration:
    return LatencyRunRegistration(
        schema_version=2,
        campaign_id="play-latency-test",
        run_id="simulated-play-run-0",
        measurement_kind="simulated",
        architecture_variant_id="diagnostic.noop.v1",
        architecture_identity_sha256=_h("noop-identity"),
        architecture_manifest_sha256=_h("manifest"),
        checkpoint_sha256=_h("no-checkpoint"),
        model_config_sha256=_h("no-model-config"),
        evaluation_code_sha256=_h("evaluation-code"),
        workload_manifest_sha256=_h("workload"),
        measured_attempt_schedule_offsets_ns=(
            offsets
            if offsets is not None
            else play_decision_schedule_offsets_ns(
                config=config, measured_attempt_count=measured
            )
        ),
        device_manifest_sha256=_h("manual-clock-device"),
        runtime_manifest_sha256=_h("runtime"),
        precision_mode="fp32",
        environment_id="moving_shapes.v1",
        clock_source="manual-monotonic-ns",
        clock_domain="manual-clock-domain-0",
        clock_synchronization="single-process-same-clock",
        capture_path="simulated-observation",
        capture_width_px=64,
        capture_height_px=64,
        action_transport="simulated-envelope",
        process_priority="normal",
        warmup_attempt_count=warmup,
        first_measured_attempt_id=(
            warmup + 1 if first_id is None else first_id
        ),
        measured_attempt_count=measured,
        submit_deadline_ns=submit_deadline_ns,
    )


class PlayScheduleTests(unittest.TestCase):
    def test_offsets_follow_the_larger_of_interval_and_latency(self) -> None:
        interval_bound = ClosedLoopPlayConfig(
            episode_seeds=(1,), inference_latency_ns=4_000_000
        )
        self.assertEqual(
            play_decision_schedule_offsets_ns(
                config=interval_bound, measured_attempt_count=3
            ),
            (0, _TICK, 2 * _TICK),
        )
        latency_bound = ClosedLoopPlayConfig(
            episode_seeds=(1,), inference_latency_ns=2 * _TICK
        )
        self.assertEqual(
            play_decision_schedule_offsets_ns(
                config=latency_bound, measured_attempt_count=3
            ),
            (0, 2 * _TICK, 4 * _TICK),
        )

    def test_schedule_validation(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(1,))
        with self.assertRaises(ValueError):
            play_decision_schedule_offsets_ns(
                config=config, measured_attempt_count=0
            )
        with self.assertRaises(ValueError):
            play_decision_schedule_offsets_ns(
                config=config, measured_attempt_count=True  # type: ignore[arg-type]
            )


class PlayLatencyEvidenceTests(unittest.TestCase):
    def test_clean_run_produces_validated_zero_queue_evidence(self) -> None:
        config = ClosedLoopPlayConfig(
            episode_seeds=(7,),
            max_ticks=100,
            inference_latency_ns=4_000_000,
        )
        registration = _registration(
            config=config, warmup=2, measured=10, submit_deadline_ns=5_000_000
        )
        report, evidence = measure_policy_play_latency(
            NoOpPolicy(),
            seed=7,
            config=config,
            registration=registration,
            expected_registration_sha256=registration.sha256(),
        )
        self.assertEqual(report.episode_seed, 7)
        self.assertEqual(evidence.attempt_count, 10)
        self.assertEqual(evidence.submitted_attempt_count, 10)
        self.assertEqual(evidence.submit_deadline_misses, 0)
        self.assertEqual(evidence.submit_deadline_miss_rate, 0.0)
        assert evidence.inference is not None
        self.assertEqual(evidence.inference.sample_count, 10)
        self.assertEqual(evidence.inference.p50_ns, 4_000_000)
        self.assertEqual(evidence.inference.p99_ns, 4_000_000)
        assert evidence.capture is not None
        self.assertEqual(evidence.capture.maximum_ns, 0)
        assert evidence.queue is not None
        self.assertEqual(evidence.queue.maximum_ns, 0)
        assert evidence.capture_to_submit is not None
        self.assertEqual(evidence.capture_to_submit.p50_ns, 4_000_000)
        # Attempt IDs follow the decision sequence numbers after warmup.
        self.assertEqual(
            tuple(attempt.attempt_id for attempt in evidence.attempts),
            tuple(range(3, 13)),
        )
        self.assertEqual(
            evidence.registration_sha256, registration.sha256()
        )

    def test_sink_does_not_change_the_episode(self) -> None:
        config = ClosedLoopPlayConfig(
            episode_seeds=(7,),
            max_ticks=100,
            inference_latency_ns=4_000_000,
        )
        plain = run_policy_closed_loop_episode(
            NoOpPolicy(), seed=7, config=config
        )
        registration = _registration(
            config=config, warmup=0, measured=5, submit_deadline_ns=5_000_000
        )
        measured, _evidence = measure_policy_play_latency(
            NoOpPolicy(),
            seed=7,
            config=config,
            registration=registration,
            expected_registration_sha256=registration.sha256(),
        )
        self.assertEqual(measured, plain)

    def test_stale_frames_become_submission_failures(self) -> None:
        config = ClosedLoopPlayConfig(
            episode_seeds=(7,),
            max_ticks=100,
            inference_latency_ns=2 * _TICK,
        )
        registration = _registration(
            config=config, warmup=0, measured=8, submit_deadline_ns=5_000_000
        )
        report, evidence = measure_policy_play_latency(
            NoOpPolicy(),
            seed=7,
            config=config,
            registration=registration,
            expected_registration_sha256=registration.sha256(),
        )
        self.assertEqual(evidence.submitted_attempt_count, 0)
        self.assertEqual(evidence.submit_deadline_miss_rate, 1.0)
        self.assertEqual(dict(evidence.status_counts)["submission_failed"], 8)
        for attempt in evidence.attempts:
            self.assertEqual(attempt.status, "submission_failed")
            self.assertEqual(
                attempt.failure_reason,
                "action does not target the newest observation",
            )
            assert attempt.inference_ns == 2 * _TICK
        self.assertGreaterEqual(report.decisions_rejected, 8)

    def test_registration_must_match_the_analytic_schedule(self) -> None:
        config = ClosedLoopPlayConfig(
            episode_seeds=(7,), max_ticks=100, inference_latency_ns=4_000_000
        )
        registration = _registration(
            config=config,
            warmup=0,
            measured=5,
            submit_deadline_ns=5_000_000,
            offsets=(0, _TICK, 2 * _TICK, 3 * _TICK, 4 * _TICK + 1),
        )
        with self.assertRaisesRegex(ValueError, "analytic play schedule"):
            measure_policy_play_latency(
                NoOpPolicy(),
                seed=7,
                config=config,
                registration=registration,
                expected_registration_sha256=registration.sha256(),
            )

    def test_first_measured_attempt_id_must_follow_warmup(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=100)
        registration = _registration(
            config=config,
            warmup=2,
            measured=5,
            submit_deadline_ns=5_000_000,
            first_id=1,
        )
        with self.assertRaisesRegex(ValueError, "warmup_attempt_count"):
            measure_policy_play_latency(
                NoOpPolicy(),
                seed=7,
                config=config,
                registration=registration,
                expected_registration_sha256=registration.sha256(),
            )

    def test_pinned_digest_mismatch_fails_closed(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=100)
        registration = _registration(
            config=config, warmup=0, measured=5, submit_deadline_ns=5_000_000
        )
        with self.assertRaisesRegex(ValueError, "pinned digest"):
            measure_policy_play_latency(
                NoOpPolicy(),
                seed=7,
                config=config,
                registration=registration,
                expected_registration_sha256=_h("wrong"),
            )

    def test_short_episode_cannot_satisfy_the_registration(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=6)
        registration = _registration(
            config=config, warmup=2, measured=10, submit_deadline_ns=5_000_000
        )
        with self.assertRaisesRegex(ValueError, "ended before"):
            measure_policy_play_latency(
                NoOpPolicy(),
                seed=7,
                config=config,
                registration=registration,
                expected_registration_sha256=registration.sha256(),
            )

    def test_non_simulated_registration_is_rejected(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=100)
        registration = _registration(
            config=config, warmup=0, measured=5, submit_deadline_ns=5_000_000
        )
        desktop_fields = registration.to_dict()
        desktop_fields.pop("attempt_schedule_sha256")
        desktop_fields["measurement_kind"] = "desktop"
        desktop_fields["measured_attempt_schedule_offsets_ns"] = tuple(
            registration.measured_attempt_schedule_offsets_ns
        )
        desktop = LatencyRunRegistration(**desktop_fields)
        with self.assertRaisesRegex(ValueError, "simulated"):
            measure_policy_play_latency(
                NoOpPolicy(),
                seed=7,
                config=config,
                registration=desktop,
                expected_registration_sha256=desktop.sha256(),
            )


if __name__ == "__main__":
    unittest.main()
