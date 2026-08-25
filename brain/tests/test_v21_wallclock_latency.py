from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.evaluation.v21_wallclock_latency import (  # noqa: E402
    LOOP_P999_LIMIT_NS,
    MEASURED_TICKS,
    POST_DGX_ENVELOPE,
    REPETITIONS,
    TICK_PERIOD_NS_60HZ,
    WARMUP_TICKS,
    CheckpointProvenance,
    RepetitionEvidence,
    TickMilestones,
    WallClockQualificationConfig,
    WallClockQualificationError,
    WallClockQualificationReport,
    _default_environment_factory,
    _json_sha256,
    _expected_model_config,
    _run_repetition,
    _state_dict_sha256,
    _training_source_bundle,
    load_exact_v21i_checkpoint_cpu,
    load_post_dgx_release_receipt,
    qualify_v21i_checkpoint_wallclock_create_only,
    validate_wallclock_artifact,
    validate_wallclock_receipt,
    validate_post_dgx_release_binding,
    wallclock_config_sha256,
    wallclock_evaluator_bundle_sha256,
    wallclock_training_source_bundle_sha256,
    wallclock_workload_sha256,
)
from irene_brain.types import (  # noqa: E402
    GenericControl,
    Observation,
    RgbFrame,
    StepOutcome,
)
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model  # noqa: E402


def _h(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _sample(index: int, *, loop_ns: int = 11_000_000) -> TickMilestones:
    start = index * 20_000_000
    return TickMilestones(
        loop_started_ns=start,
        model_started_ns=start + 1_000_000,
        model_finished_ns=start + 7_000_000,
        policy_finished_ns=start + 8_000_000,
        loop_finished_ns=start + loop_ns,
    )


def _repetition(index: int, *, deadline_misses: int = 0) -> RepetitionEvidence:
    fast_count = MEASURED_TICKS - deadline_misses
    samples = tuple(
        [*(_sample(tick) for tick in range(fast_count))]
        + [
            _sample(fast_count + offset, loop_ns=TICK_PERIOD_NS_60HZ + 1)
            for offset in range(deadline_misses)
        ]
    )
    return RepetitionEvidence(
        repetition=index,
        warmup_ticks=WARMUP_TICKS,
        measured_ticks=MEASURED_TICKS,
        warmup_episode_rotations=0,
        measured_episode_rotations=9,
        episode_seeds=tuple(
            67_108_864 + index * 1_048_576 + offset
            for offset in range(10)
        ),
        milestones=samples,
    )


def _provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        checkpoint_envelope="development_v1",
        checkpoint_sha256=_h("checkpoint"),
        state_sha256=_h("state"),
        model_config_sha256=_h("model-config"),
        feature_flags_sha256=_h("flags"),
        hazard_calibration_sha256=_h("calibration"),
        training_source_bundle_sha256=wallclock_training_source_bundle_sha256(ROOT.parent),
        evaluator_bundle_sha256=wallclock_evaluator_bundle_sha256(ROOT.parent),
        preregistration_sha256=_h("preregistration"),
        post_dgx_release_body_sha256=None,
        post_dgx_release_file_sha256=None,
        model_seed=44,
    )


def _report() -> WallClockQualificationReport:
    return WallClockQualificationReport(
        provenance=_provenance(),
        config=WallClockQualificationConfig(),
        repetitions=tuple(_repetition(index) for index in range(REPETITIONS)),
        runtime={
            "clock": "time.perf_counter_ns",
            "clock_monotonic": True,
            "device": "cpu",
            "torch_interop_threads": 1,
            "torch_threads": 1,
        },
        model_state_unchanged=True,
        source_unchanged=True,
    )


def _artifact(report: WallClockQualificationReport) -> dict[str, object]:
    receipt = report.receipt()
    return {
        "receipt": receipt,
        "receipt_sha256": _json_sha256(
            receipt, domain=b"IRV21IWALLCLOCKRECEIPT\x01"
        ),
        "report": report.to_dict(),
        "report_sha256": report.sha256,
        "schema_version": 1,
        "status": "passed",
    }


class FrozenContractTests(unittest.TestCase):
    def test_run_shape_thresholds_and_digests_are_frozen(self) -> None:
        config = WallClockQualificationConfig()
        self.assertEqual(
            (config.warmup_ticks, config.measured_ticks, config.repetitions),
            (100, 5_000, 3),
        )
        self.assertEqual(config.tick_period_ns, 16_666_667)
        self.assertEqual(config.model_p99_limit_ns, 8_000_000)
        self.assertEqual(config.loop_p99_limit_ns, 12_000_000)
        self.assertEqual(config.loop_p999_limit_ns, 16_666_667)
        self.assertEqual(config.maximum_deadline_miss_rate, 0.001)
        self.assertEqual(len(wallclock_config_sha256()), 64)
        self.assertEqual(len(wallclock_workload_sha256()), 64)
        environment = _default_environment_factory(config)
        self.assertEqual(environment._ghost_count, 5)
        self.assertEqual(environment._ghost_period, 1)
        self.assertEqual(environment._player_period, 1)
        self.assertEqual(environment._extra_loops, 16)
        self.assertEqual(environment._ghost_rule, "direct")
        self.assertFalse(environment._ghost_elroy)
        self.assertEqual(environment._input_delay_ticks, 0)
        self.assertFalse(environment._sticky_direction)
        self.assertEqual(environment._max_ticks, 512)
        with self.assertRaisesRegex(WallClockQualificationError, "frozen"):
            WallClockQualificationConfig(measured_ticks=4_999)

    def test_miss_rate_is_strictly_below_one_tenth_percent(self) -> None:
        passing = _repetition(0, deadline_misses=4)
        failing = _repetition(0, deadline_misses=5)
        self.assertEqual(passing.deadline_miss_rate, 0.0008)
        self.assertTrue(passing.passed)
        self.assertEqual(failing.deadline_miss_rate, 0.001)
        self.assertFalse(failing.passed)
        # Exactly five misses are rejected by the strict rate gate even though
        # nearest-rank p99.9 still lands on the last non-missing sample.
        self.assertLessEqual(failing.loop_p999_ns, LOOP_P999_LIMIT_NS)


class _CounterClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1
        return self.value


class _TinyModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1.0


def _observation(frame: int) -> Observation:
    return Observation(
        frame_id=frame,
        capture_tick=frame,
        elapsed_ns=frame,
        rgb=RgbFrame(width=1, height=1, pixels=b"\x01\x02\x03"),
        previous_control=GenericControl.neutral(),
    )


class _TinyPolicy:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.reset_seeds: list[int] = []

    def reset(self, seed: int) -> None:
        self.reset_seeds.append(seed)

    def act(self, observation: Observation) -> GenericControl:
        del observation
        self.model(torch.zeros(1))
        return GenericControl.neutral()


class _TwoTickEnvironment:
    def __init__(self) -> None:
        self.tick = 0
        self.frame = 0

    def reset(self, seed: int) -> Observation:
        del seed
        self.tick = 0
        self.frame += 1
        return _observation(self.frame)

    def step(self, control: object) -> StepOutcome:
        self.assert_control(control)
        self.tick += 1
        self.frame += 1
        terminal = self.tick == 2
        return StepOutcome(
            observation=_observation(self.frame),
            requested_control=control,
            applied_control=control,
            reward=0.0,
            terminated=terminal,
        )

    @staticmethod
    def assert_control(control: object) -> None:
        if not isinstance(control, GenericControl):
            raise AssertionError("expected GenericControl")


class MeasurementTests(unittest.TestCase):
    def test_exact_forward_milestones_and_terminal_rotation(self) -> None:
        model = _TinyModel()
        policy = _TinyPolicy(model)
        evidence = _run_repetition(
            policy,
            model,
            repetition=0,
            config=WallClockQualificationConfig(),
            clock=_CounterClock(),
            environment_factory=_TwoTickEnvironment,
        )
        self.assertEqual(len(evidence.milestones), 5_000)
        self.assertEqual(evidence.warmup_episode_rotations, 50)
        self.assertEqual(evidence.measured_episode_rotations, 2_500)
        self.assertEqual(len(evidence.episode_seeds), 2_551)
        self.assertEqual(len(policy.reset_seeds), 2_551)
        self.assertTrue(evidence.passed)
        first = evidence.milestones[0]
        self.assertEqual(first.model_forward_ns, 1)
        self.assertEqual(first.loop_ns, 4)

    def test_milestones_fail_closed_when_clock_runs_backwards(self) -> None:
        with self.assertRaisesRegex(WallClockQualificationError, "backwards"):
            TickMilestones(5, 4, 6, 7, 8)


class ReceiptTests(unittest.TestCase):
    @staticmethod
    def _post_dgx_body() -> dict[str, object]:
        return {
            "authorization": {
                "cpu_wallclock_allowed": True,
                "play_qualification_allowed": True,
                "test_split_access_allowed": False,
                "training_allowed": False,
            },
            "checkpoint": {
                "checkpoint_sha256": _h("dgx-checkpoint"),
                "feature_flags_sha256": _h("flags"),
                "hazard_calibration_sha256": _h("calibration"),
                "model_config_sha256": _h("model-config"),
                "model_seed": 44,
                "source_bundle_sha256": _h("source"),
                "state_sha256": _h("state"),
            },
            "classification": "post_dgx_cpu_qual_release_v1",
            "cpu_qualification": {
                "all_candidates_passed": True,
                "evaluation_artifact_sha256": _h("cpu-qual-evaluation"),
                "observed_passes": 5,
                "opening_receipt_sha256": _h("cpu-qual-opening"),
                "passed_model_seeds": [44, 45, 46, 47, 48],
                "preregistration_sha256": _h("cpu-qual-preregistration"),
                "qualification_id": "v21i-cpu-qual-world-model-v1",
                "required_model_seeds": [44, 45, 46, 47, 48],
                "required_passes": 5,
                "test_split_opened": False,
            },
            "dgx_lineage": {
                "accelerator": "nvidia_dgx_spark",
                "authorization_receipt_sha256": _h("dgx-authorization"),
                "bounded_schedule_sha256": _h("dgx-schedule"),
                "run_artifact_sha256": _h("dgx-run"),
            },
            "post_dgx_verification": {
                "all_metrics_finite": True,
                "artifact_sha256": _h("post-dgx-verification"),
                "calibration_passed": True,
                "causal_shuffle_passed": True,
                "hazard_quality_passed": True,
                "noncollapse_passed": True,
                "preservation_passed": True,
                "qualification_id": "v21i-post-dgx-verification-v1",
            },
        }

    def test_canonical_post_dgx_release_binds_exact_five_seed_lineage(self) -> None:
        body = self._post_dgx_body()
        body_sha = _json_sha256(
            body, domain=b"IRV21IPOSTDGXRELEASEBODY\x01"
        )
        payload = {
            "body": body,
            "body_sha256": body_sha,
            "qualification_id": "v21i-post-dgx-cpu-qual-release-v1",
            "schema_version": 1,
            "status": "passed",
        }
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "post-dgx-release.json"
            path.write_bytes(encoded)
            release = load_post_dgx_release_receipt(
                path,
                expected_file_sha256=sha256(encoded).hexdigest(),
            )
        checkpoint = body["checkpoint"]
        lineage = body["dgx_lineage"]
        validated = validate_post_dgx_release_binding(
            release,
            checkpoint_sha256=checkpoint["checkpoint_sha256"],
            state_sha256=checkpoint["state_sha256"],
            model_config_sha256=checkpoint["model_config_sha256"],
            feature_flags_sha256=checkpoint["feature_flags_sha256"],
            hazard_calibration_sha256=checkpoint["hazard_calibration_sha256"],
            source_bundle_sha256=checkpoint["source_bundle_sha256"],
            model_seed=checkpoint["model_seed"],
            training_runtime=lineage,
        )
        self.assertEqual(validated, body)

        release.body["cpu_qualification"]["test_split_opened"] = True
        with self.assertRaisesRegex(WallClockQualificationError, "5/5"):
            validate_post_dgx_release_binding(
                release,
                checkpoint_sha256=checkpoint["checkpoint_sha256"],
                state_sha256=checkpoint["state_sha256"],
                model_config_sha256=checkpoint["model_config_sha256"],
                feature_flags_sha256=checkpoint["feature_flags_sha256"],
                hazard_calibration_sha256=checkpoint[
                    "hazard_calibration_sha256"
                ],
                source_bundle_sha256=checkpoint["source_bundle_sha256"],
                model_seed=checkpoint["model_seed"],
                training_runtime=lineage,
            )

    def test_schema2_checkpoint_requires_and_loads_exact_release_on_cpu(self) -> None:
        config = _expected_model_config()
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        state = model.state_dict()
        state_sha = _state_dict_sha256(state)
        source_bundle = _training_source_bundle(ROOT.parent)
        calibration = {
            "accepted": True,
            "bias": [0.0] * 5,
            "fit_mode": "bias_only",
            "scale": [1.0] * 5,
        }
        training_runtime = self._post_dgx_body()["dgx_lineage"]
        payload = {
            "calibration_installation_audit": {},
            "calibration_mode": "bias_only",
            "classification": (
                "bounded_dgx_candidate_requires_post_training_cpu_qual"
            ),
            "config": asdict(config),
            "dataset_partitions": {
                "DEV": {},
                "TRAIN-CAL": {},
                "TRAIN-FIT": {},
            },
            "development_gate": {"passed": True},
            "device": "cuda",
            "expected_calibration_bindings": {},
            "finite_state_audit": {
                "all_buffers_finite": True,
                "all_parameters_finite": True,
                "buffer_tensor_count": len(tuple(model.named_buffers())),
                "parameter_tensor_count": len(tuple(model.named_parameters())),
                "state_sha256": state_sha,
            },
            "flags": asdict(CONFIG_B_PREDICTIVE),
            "hazard_calibration": calibration,
            "mode": "v21i_bounded_dgx_candidate_v1",
            "model_seed": 44,
            "model_state_dict": state,
            "schema_version": 2,
            "source_bundle": source_bundle,
            "state_sha256": state_sha,
            "threads": 1,
            "training": {},
            "training_runtime": training_runtime,
            "upstream_uncalibrated_checkpoint": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "post-dgx.pt"
            torch.save(payload, checkpoint_path)
            checkpoint_sha = sha256(checkpoint_path.read_bytes()).hexdigest()
            body = self._post_dgx_body()
            body["checkpoint"] = {
                "checkpoint_sha256": checkpoint_sha,
                "feature_flags_sha256": _json_sha256(
                    asdict(CONFIG_B_PREDICTIVE),
                    domain=b"IRV21IFEATUREFLAGS\x01",
                ),
                "hazard_calibration_sha256": _json_sha256(
                    calibration,
                    domain=b"IRV21IHAZARDCALIBRATION\x01",
                ),
                "model_config_sha256": _json_sha256(
                    asdict(config), domain=b"IRV21IMODELCONFIG\x01"
                ),
                "model_seed": 44,
                "source_bundle_sha256": source_bundle["sha256"],
                "state_sha256": state_sha,
            }
            body["dgx_lineage"] = training_runtime
            body_sha = _json_sha256(
                body, domain=b"IRV21IPOSTDGXRELEASEBODY\x01"
            )
            release_payload = {
                "body": body,
                "body_sha256": body_sha,
                "qualification_id": "v21i-post-dgx-cpu-qual-release-v1",
                "schema_version": 1,
                "status": "passed",
            }
            release_encoded = (
                json.dumps(
                    release_payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            release_path = root / "release.json"
            release_path.write_bytes(release_encoded)
            release = load_post_dgx_release_receipt(
                release_path,
                expected_file_sha256=sha256(release_encoded).hexdigest(),
            )
            with patch(
                "irene_brain.evaluation.v21_wallclock_latency."
                "_configure_one_thread_cpu"
            ):
                with self.assertRaisesRegex(
                    WallClockQualificationError, "requires an external release"
                ):
                    load_exact_v21i_checkpoint_cpu(
                        checkpoint_path,
                        project_root=ROOT.parent,
                        expected_checkpoint_sha256=checkpoint_sha,
                        expected_source_bundle_sha256=source_bundle["sha256"],
                        expected_model_seed=44,
                        required_envelope=POST_DGX_ENVELOPE,
                    )
                loaded = load_exact_v21i_checkpoint_cpu(
                    checkpoint_path,
                    project_root=ROOT.parent,
                    expected_checkpoint_sha256=checkpoint_sha,
                    expected_source_bundle_sha256=source_bundle["sha256"],
                    expected_model_seed=44,
                    required_envelope=POST_DGX_ENVELOPE,
                    post_dgx_release=release,
                )
        self.assertEqual(loaded.envelope, POST_DGX_ENVELOPE)
        self.assertEqual(loaded.checkpoint_sha256, checkpoint_sha)
        self.assertIs(loaded.post_dgx_release, release)
        self.assertTrue(all(value.device.type == "cpu" for value in loaded.model.parameters()))

    def test_complete_receipt_validates_and_self_rehash_does_not_hide_drift(self) -> None:
        report = _report()
        artifact = _artifact(report)
        receipt = validate_wallclock_artifact(
            artifact,
            expected_checkpoint_sha256=report.provenance.checkpoint_sha256,
            expected_source_bundle_sha256=(
                report.provenance.training_source_bundle_sha256
            ),
            expected_preregistration_sha256=(
                report.provenance.preregistration_sha256
            ),
        )
        self.assertTrue(receipt["overall_passed"])
        self.assertEqual(len(receipt["repetitions"]), 3)
        with self.assertRaisesRegex(WallClockQualificationError, "not authorized"):
            validate_wallclock_receipt(
                receipt,
                expected_checkpoint_sha256=report.provenance.checkpoint_sha256,
                expected_source_bundle_sha256=(
                    report.provenance.training_source_bundle_sha256
                ),
                required_checkpoint_envelope=POST_DGX_ENVELOPE,
            )
        self.assertIs(
            validate_wallclock_receipt(
                receipt,
                expected_checkpoint_sha256=report.provenance.checkpoint_sha256,
                expected_source_bundle_sha256=(
                    report.provenance.training_source_bundle_sha256
                ),
                expected_preregistration_sha256=(
                    report.provenance.preregistration_sha256
                ),
            ),
            receipt,
        )

        altered = json.loads(json.dumps(artifact))
        altered_receipt = altered["receipt"]
        altered_receipt["repetitions"][0]["measured_ticks"] = 4_999
        altered["receipt_sha256"] = _json_sha256(
            altered_receipt, domain=b"IRV21IWALLCLOCKRECEIPT\x01"
        )
        with self.assertRaisesRegex(WallClockQualificationError, "measured count"):
            validate_wallclock_artifact(
                altered,
                expected_checkpoint_sha256=report.provenance.checkpoint_sha256,
                expected_source_bundle_sha256=(
                    report.provenance.training_source_bundle_sha256
                ),
            )

        raw_altered = json.loads(json.dumps(artifact))
        raw_altered["report"]["repetitions"][0]["milestones_ns"][0][4] += 1
        raw_altered["report_sha256"] = _json_sha256(
            raw_altered["report"], domain=b"IRV21IWALLCLOCKREPORT\x01"
        )
        raw_altered["receipt"]["report_sha256"] = raw_altered["report_sha256"]
        raw_altered["receipt_sha256"] = _json_sha256(
            raw_altered["receipt"], domain=b"IRV21IWALLCLOCKRECEIPT\x01"
        )
        with self.assertRaisesRegex(WallClockQualificationError, "raw milestone"):
            validate_wallclock_artifact(
                raw_altered,
                expected_checkpoint_sha256=report.provenance.checkpoint_sha256,
                expected_source_bundle_sha256=(
                    report.provenance.training_source_bundle_sha256
                ),
            )

    def test_create_only_collision_blocks_before_checkpoint_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "not-a-checkpoint.pt"
            output = root / "latency.json"
            checkpoint.write_bytes(b"not a pickle")
            output.write_bytes(b"preserve me")
            with self.assertRaisesRegex(WallClockQualificationError, "overwrite"):
                qualify_v21i_checkpoint_wallclock_create_only(
                    checkpoint_path=checkpoint,
                    expected_checkpoint_sha256=_h("checkpoint"),
                    expected_source_bundle_sha256=_h("source"),
                    expected_model_seed=44,
                    preregistration_sha256=_h("preregistration"),
                    output_path=output,
                )
            self.assertEqual(output.read_bytes(), b"preserve me")


if __name__ == "__main__":
    unittest.main()
