from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.evaluation.closed_loop_play import (  # noqa: E402
    ClosedLoopEpisodeReport,
    DecisionTiming,
)
from irene_brain.evaluation.v21_live_play_qualification import (  # noqa: E402
    ACTION_IDS,
    EXACT_MAZE_ENVIRONMENT_ID,
    POST_DGX_CHECKPOINT_ENVELOPE,
    TICK_PERIOD_NS_60HZ,
    LiveEpisodeExecution,
    LivePlayProvenance,
    LivePlayQualificationConfig,
    LivePlayQualificationError,
    LivePlayQualificationFailed,
    SeededFiveActionRandomPolicy,
    StrictRgbOnlyPolicy,
    WallClockLatencyReceipt,
    _exact_maze_factory,
    _qualify_live_play_create_only_for_tests,
    exact_maze_environment_config_sha256,
    live_play_cohort_binding_sha256,
    live_play_workload_sha256,
    paired_clustered_bootstrap,
    qualify_live_play_create_only,
)
from irene_brain.evaluation.v21_wallclock_latency import (  # noqa: E402
    wallclock_workload_sha256,
)
from irene_brain.types import GenericControl, Observation, RgbFrame  # noqa: E402


def _h(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _observation(frame_id: int) -> Observation:
    color = bytes(((frame_id * 3) % 256, 17, 29))
    return Observation(
        frame_id=frame_id,
        capture_tick=frame_id * TICK_PERIOD_NS_60HZ,
        elapsed_ns=frame_id * TICK_PERIOD_NS_60HZ,
        rgb=RgbFrame(width=1, height=1, pixels=color),
        # This deliberately disagrees with the adapter's internal prior.  The
        # controller must never receive it.
        previous_control=GenericControl(keys_down=(7,)),
        audio_pcm_s16le=b"secret audio",
    )


class _RecordingController:
    def __init__(self, actions: tuple[int, ...] = (1, 2, 3, 4, 0)) -> None:
        self.actions = actions
        self.reset_seeds: list[int] = []
        self.calls: list[tuple[RgbFrame, int]] = []
        self.index = 0

    def reset(self, seed: int) -> None:
        self.reset_seeds.append(seed)
        self.calls.clear()
        self.index = 0

    def act_rgb(self, rgb: RgbFrame, previous_action_id: int) -> int:
        self.calls.append((rgb, previous_action_id))
        action = self.actions[self.index % len(self.actions)]
        self.index += 1
        return action


def _mask(control: GenericControl) -> int:
    result = 0
    for key, bit in ((26, 1), (4, 2), (22, 4), (7, 8)):
        if key in control.keys_down:
            result |= bit
    return result


def _clean_execution(
    policy: object,
    seed: int,
    closed_config: object,
    *,
    model_reward: float = 12.0,
    random_reward: float = 1.0,
    model_catches: int = 0,
    random_catches: int = 3,
    model_pellets: int = 6,
    random_pellets: int = 1,
    timing_offset_ns: int = 0,
) -> LiveEpisodeExecution:
    policy.reset(seed)
    masks: dict[int, int] = {}
    controls = []
    for frame_id in range(4):
        control = policy.act(_observation(frame_id))
        controls.append(control)
        mask = _mask(control)
        masks[mask] = masks.get(mask, 0) + 1
    is_model = isinstance(policy, StrictRgbOnlyPolicy)
    reward = model_reward if is_model else random_reward
    catches = model_catches if is_model else random_catches
    report = ClosedLoopEpisodeReport(
        episode_seed=seed,
        ticks_advanced=4,
        reward_sum=reward,
        targets_collected=0,
        pellets_eaten=model_pellets if is_model else random_pellets,
        collisions=catches,
        decisions_submitted=4,
        decisions_rejected=0,
        rejections_by_reason=(),
        observations_dropped=0,
        movement_mask_histogram=tuple(sorted(masks.items())),
        opposite_conflicts=0,
        non_movement_key_activations=0,
        continuous_outside_deadzone=0,
        continuous_max_abs=0.0,
        mean_value=0.0,
    )
    timings = tuple(
        DecisionTiming(
            action_sequence=index + 1,
            scheduled_ns=timing_offset_ns + index * TICK_PERIOD_NS_60HZ,
            inference_finished_ns=(
                timing_offset_ns
                + index * TICK_PERIOD_NS_60HZ
                + int(getattr(closed_config, "inference_latency_ns"))
            ),
            submitted=True,
            failure_reason=None,
        )
        for index in range(4)
    )
    step_digest = _h(
        f"{seed}:{getattr(policy, 'trajectory_input_action_sha256')}:{reward}:{catches}"
    )
    return LiveEpisodeExecution(
        report=report,
        timings=timings,
        environment_id=EXACT_MAZE_ENVIRONMENT_ID,
        environment_config_sha256=exact_maze_environment_config_sha256(
            int(getattr(closed_config, "max_ticks"))
        ),
        step_trajectory_sha256=step_digest,
        trajectory_step_count=4,
    )


def _config(**overrides: object) -> LivePlayQualificationConfig:
    values: dict[str, object] = {
        "episode_seeds": (101, 102, 103, 104),
        "cluster_ids": ("a", "b", "c", "d"),
        "max_ticks": 10,
        "model_reset_seed": 77,
        "random_baseline_seed": 88,
        "bootstrap_seed": 99,
        "bootstrap_resamples": 200,
    }
    values.update(overrides)
    return LivePlayQualificationConfig(**values)  # type: ignore[arg-type]


def _production_shape_config() -> LivePlayQualificationConfig:
    return LivePlayQualificationConfig(
        episode_seeds=tuple((3 << 62) | index for index in range(64)),
        cluster_ids=tuple(
            f"play-qual-environment-{index:02d}" for index in range(64)
        ),
        max_ticks=600,
        model_reset_seed=77,
        random_baseline_seed=2_126_202_681,
        bootstrap_seed=2_126_202_699,
        bootstrap_resamples=10_000,
        confidence_level=0.95,
    )


def _cohort_receipt(config: LivePlayQualificationConfig) -> dict[str, object]:
    return {
        "cluster_ids": list(config.cluster_ids),
        "cohort_id": "mock-live-cohort",
        "episode_seeds": list(config.episode_seeds),
        "max_ticks": config.max_ticks,
        "namespace": "LIVE-QUAL",
        "preregistration_sha256": _h("preregistration"),
        "random_baseline_streams": config.random_baseline_streams,
        "schema_version": 1,
        "sealed": True,
        "test_accessed": False,
    }


def _provenance(
    config: LivePlayQualificationConfig | None = None,
) -> LivePlayProvenance:
    bound_config = config or _config()
    return LivePlayProvenance(
        qualification_id="v21-live-play-test",
        model_identity="test.rgb-controller.v1",
        checkpoint_envelope=POST_DGX_CHECKPOINT_ENVELOPE,
        checkpoint_sha256=_h("checkpoint"),
        model_state_sha256=_h("model-state"),
        model_config_sha256=_h("model-config"),
        feature_flags_sha256=_h("feature-flags"),
        source_bundle_sha256=_h("source-bundle"),
        evaluator_bundle_sha256=_h("evaluator-bundle"),
        environment_source_sha256=_h("environment-source"),
        preregistration_sha256=_h("preregistration"),
        cohort_binding_sha256=live_play_cohort_binding_sha256(
            _cohort_receipt(bound_config)
        ),
        post_dgx_release_body_sha256=_h("post-dgx-release-body"),
        post_dgx_release_file_sha256=_h("post-dgx-release-file"),
    )


def _latency_receipt(config: LivePlayQualificationConfig) -> WallClockLatencyReceipt:
    provenance = _provenance(config)
    return WallClockLatencyReceipt(
        checkpoint_envelope=provenance.checkpoint_envelope,
        checkpoint_sha256=provenance.checkpoint_sha256,
        model_state_sha256=provenance.model_state_sha256,
        feature_flags_sha256=provenance.feature_flags_sha256,
        source_bundle_sha256=provenance.source_bundle_sha256,
        workload_sha256=live_play_workload_sha256(config),
        benchmark_workload_sha256=wallclock_workload_sha256(),
        validated_artifact_receipt_sha256=_h("validated-wallclock-receipt"),
        runtime_manifest_sha256=_h("runtime"),
        device_manifest_sha256=_h("device"),
        post_dgx_release_body_sha256=(
            provenance.post_dgx_release_body_sha256 or ""
        ),
        post_dgx_release_file_sha256=(
            provenance.post_dgx_release_file_sha256 or ""
        ),
        repetitions=3,
        warmup_per_repetition=100,
        measured_per_repetition=5_000,
        model_p99_ns=3_000_000,
        loop_p99_ns=5_000_000,
        loop_p999_ns=7_000_000,
        deadline_miss_rate=0.0,
        passed=True,
    )


class StrictRgbOnlyPolicyTests(unittest.TestCase):
    def test_only_rgb_and_internal_prior_cross_the_controller_boundary(self) -> None:
        controller = _RecordingController(actions=(1, 4, 0))
        policy = StrictRgbOnlyPolicy(controller, model_reset_seed=555)

        policy.reset(999_001)
        controls = [policy.act(_observation(index)) for index in range(3)]

        self.assertEqual(controller.reset_seeds, [555])
        self.assertEqual([prior for _rgb, prior in controller.calls], [0, 1, 4])
        self.assertTrue(all(isinstance(rgb, RgbFrame) for rgb, _ in controller.calls))
        self.assertEqual([control.keys_down for control in controls], [(26,), (7,), ()])
        self.assertEqual(policy.action_histogram, (1, 1, 0, 0, 1))
        self.assertEqual(len(policy.trajectory_input_action_sha256), 64)

        policy.reset(123_456)
        self.assertEqual(controller.reset_seeds, [555, 555])
        self.assertEqual(policy.action_count, 0)

    def test_non_five_action_output_fails_closed(self) -> None:
        controller = _RecordingController(actions=(5,))
        policy = StrictRgbOnlyPolicy(controller, model_reset_seed=1)
        policy.reset(2)
        with self.assertRaisesRegex(LivePlayQualificationError, "idle/W/A/S/D"):
            policy.act(_observation(0))


class RandomBaselineTests(unittest.TestCase):
    def test_seeded_policy_is_deterministic_and_uses_exactly_five_actions(self) -> None:
        def sequence() -> tuple[tuple[int, ...], ...]:
            policy = SeededFiveActionRandomPolicy(campaign_seed=901)
            policy.reset(42)
            return tuple(policy.act(_observation(index)).keys_down for index in range(500))

        first = sequence()
        second = sequence()
        self.assertEqual(first, second)
        observed = {
            (): 0,
            (26,): 1,
            (4,): 2,
            (22,): 3,
            (7,): 4,
        }
        self.assertEqual({observed[control] for control in first}, set(ACTION_IDS))
        self.assertTrue(all(control in observed for control in first))


class BootstrapTests(unittest.TestCase):
    def test_pairing_and_cluster_resampling_are_deterministic(self) -> None:
        first = paired_clustered_bootstrap(
            (3.0, 3.0, 5.0, 5.0),
            ("episode-a", "episode-a", "episode-b", "episode-b"),
            direction="model-minus-random",
            resamples=500,
            seed=17,
            confidence_level=0.95,
        )
        second = paired_clustered_bootstrap(
            (3.0, 3.0, 5.0, 5.0),
            ("episode-a", "episode-a", "episode-b", "episode-b"),
            direction="model-minus-random",
            resamples=500,
            seed=17,
            confidence_level=0.95,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.observed_mean_advantage, 4.0)
        self.assertGreater(first.lower_bound, 0.0)
        self.assertEqual(first.cluster_count, 2)


class ExactMazeBindingTests(unittest.TestCase):
    def test_factory_is_hard_bound_to_registered_maze_configuration(self) -> None:
        traced = _exact_maze_factory(600)
        environment = traced._environment
        self.assertEqual(environment._ghost_count, 5)
        self.assertEqual(environment._ghost_period, 1)
        self.assertEqual(environment._player_period, 1)
        self.assertEqual(environment._extra_loops, 16)
        self.assertEqual(environment._ghost_rule, "direct")
        self.assertFalse(environment._ghost_elroy)
        self.assertEqual(environment._input_delay_ticks, 0)
        self.assertFalse(environment._sticky_direction)
        self.assertEqual(environment._tick_period_ns, TICK_PERIOD_NS_60HZ)
        self.assertEqual(environment._max_ticks, 600)


class QualificationArtifactTests(unittest.TestCase):
    def test_public_surface_rejects_development_envelope_before_play(self) -> None:
        config = _production_shape_config()
        development = replace(
            _provenance(config),
            checkpoint_envelope="development_v1",
            post_dgx_release_body_sha256=None,
            post_dgx_release_file_sha256=None,
        )
        controller = _RecordingController()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "must-not-exist.json"
            with self.assertRaisesRegex(LivePlayQualificationError, "post-DGX"):
                qualify_live_play_create_only(
                    controller,
                    config=config,
                    provenance=development,
                    cohort_receipt=_cohort_receipt(config),
                    wall_clock_artifact={},
                    output_path=destination,
                )

            self.assertFalse(destination.exists())
            self.assertEqual(controller.reset_seeds, [])

    def test_production_surface_rejects_substituted_controller_before_play(self) -> None:
        config = _production_shape_config()
        provenance = _provenance(config)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "must-not-exist.json"
            with self.assertRaisesRegex(
                LivePlayQualificationError,
                "exact OutcomeAwareCoreV2MazePolicy",
            ):
                qualify_live_play_create_only(
                    _RecordingController(),
                    config=config,
                    provenance=provenance,
                    cohort_receipt=_cohort_receipt(config),
                    wall_clock_artifact={},
                    output_path=destination,
                )
            self.assertFalse(destination.exists())

    def test_passing_campaign_writes_complete_create_only_evidence(self) -> None:
        controller = _RecordingController()
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "qualification.json"
            report = _qualify_live_play_create_only_for_tests(
                controller,
                config=config,
                provenance=_provenance(config),
                cohort_receipt=_cohort_receipt(config),
                wall_clock_latency_receipt=_latency_receipt(config),
                output_path=destination,
                episode_runner=_clean_execution,
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertTrue(report.gates_passed)
        self.assertFalse(report.passed)
        self.assertEqual(payload["status"], "test_only")
        self.assertFalse(payload["production_eligible"])
        self.assertEqual(payload["report_sha256"], report.sha256)
        self.assertEqual(controller.reset_seeds, [77] * 8)
        self.assertTrue(
            all(episode.timing.passed for episode in report.model_episodes)
        )
        self.assertTrue(
            all(episode.control_validity_passed for episode in report.baseline_episodes)
        )
        self.assertEqual(len(report.baseline_episodes), 32)
        self.assertTrue(
            all(episode.deterministic_repeat_match for episode in report.model_episodes)
        )
        self.assertTrue(all(episode.cleared for episode in report.model_episodes))
        self.assertEqual(
            {len(episode.trajectory_sha256) for episode in report.model_episodes},
            {64},
        )
        self.assertGreater(report.reward_bootstrap.lower_bound, 0.0)
        self.assertGreater(report.catch_bootstrap.lower_bound, 0.0)

    def test_failed_gate_is_written_then_raised(self) -> None:
        def losing_runner(policy: object, seed: int, config: object) -> LiveEpisodeExecution:
            return _clean_execution(
                policy,
                seed,
                config,
                model_reward=-2.0,
                random_reward=2.0,
                model_catches=4,
                random_catches=1,
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "failed.json"
            with self.assertRaises(LivePlayQualificationFailed) as captured:
                _qualify_live_play_create_only_for_tests(
                    _RecordingController(),
                    config=_config(),
                    provenance=_provenance(),
                    cohort_receipt=_cohort_receipt(_config()),
                    wall_clock_latency_receipt=_latency_receipt(_config()),
                    output_path=destination,
                    episode_runner=losing_runner,
                )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertFalse(captured.exception.report.passed)
        self.assertEqual(payload["status"], "test_only")
        self.assertFalse(payload["report"]["passed"])

    def test_zero_model_pellets_is_an_explicit_hard_failure(self) -> None:
        config = _config()

        def zero_pellet_runner(
            policy: object, seed: int, closed_config: object
        ) -> LiveEpisodeExecution:
            return _clean_execution(
                policy,
                seed,
                closed_config,
                model_pellets=0,
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "zero-pellets.json"
            with self.assertRaises(LivePlayQualificationFailed) as captured:
                _qualify_live_play_create_only_for_tests(
                    _RecordingController(),
                    config=config,
                    provenance=_provenance(config),
                    cohort_receipt=_cohort_receipt(config),
                    wall_clock_latency_receipt=_latency_receipt(config),
                    output_path=destination,
                    episode_runner=zero_pellet_runner,
                )

        pellet_gate = next(
            gate
            for gate in captured.exception.report.gates
            if gate.name == "model_pellets_earned"
        )
        self.assertFalse(pellet_gate.passed)
        self.assertEqual(pellet_gate.observed, 0)

    def test_timing_mismatch_is_a_hard_gate_failure(self) -> None:
        def bad_timing_runner(
            policy: object, seed: int, config: object
        ) -> LiveEpisodeExecution:
            return _clean_execution(
                policy,
                seed,
                config,
                timing_offset_ns=(1 if seed == 102 else 0),
            )

        # A constant offset is allowed because cadence, not an absolute clock
        # origin, is registered.  Mutate one decision to break the cadence.
        def broken_runner(
            policy: object, seed: int, config: object
        ) -> LiveEpisodeExecution:
            execution = bad_timing_runner(policy, seed, config)
            if seed != 102:
                return execution
            timings = list(execution.timings)
            timing = timings[2]
            timings[2] = DecisionTiming(
                action_sequence=timing.action_sequence,
                scheduled_ns=timing.scheduled_ns + 1,
                inference_finished_ns=timing.inference_finished_ns + 1,
                submitted=timing.submitted,
                failure_reason=timing.failure_reason,
            )
            return replace(execution, timings=tuple(timings))

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "timing-failed.json"
            with self.assertRaises(LivePlayQualificationFailed) as captured:
                _qualify_live_play_create_only_for_tests(
                    _RecordingController(),
                    config=_config(),
                    provenance=_provenance(),
                    cohort_receipt=_cohort_receipt(_config()),
                    wall_clock_latency_receipt=_latency_receipt(_config()),
                    output_path=destination,
                    episode_runner=broken_runner,
                )
        timing_gate = next(
            gate
            for gate in captured.exception.report.gates
            if gate.name == "simulated_control_schedule_integrity"
        )
        self.assertFalse(timing_gate.passed)

    def test_repeat_trajectory_mismatch_is_a_hard_gate_failure(self) -> None:
        config = _config()
        calls = 0

        def nondeterministic_runner(
            policy: object, seed: int, closed_config: object
        ) -> LiveEpisodeExecution:
            nonlocal calls
            execution = _clean_execution(policy, seed, closed_config)
            calls += 1
            if isinstance(policy, StrictRgbOnlyPolicy) and seed == 101 and calls == 2:
                return replace(execution, step_trajectory_sha256=_h("changed-step"))
            return execution

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "repeat-failed.json"
            with self.assertRaises(LivePlayQualificationFailed) as captured:
                _qualify_live_play_create_only_for_tests(
                    _RecordingController(),
                    config=config,
                    provenance=_provenance(config),
                    cohort_receipt=_cohort_receipt(config),
                    wall_clock_latency_receipt=_latency_receipt(config),
                    output_path=destination,
                    episode_runner=nondeterministic_runner,
                )

        repeat_gate = next(
            gate
            for gate in captured.exception.report.gates
            if gate.name == "deterministic_repeat_trajectory_equality"
        )
        self.assertFalse(repeat_gate.passed)

    def test_runner_exception_writes_failure_and_collision_blocks_before_reset(self) -> None:
        controller = _RecordingController(actions=(9,))
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "aborted.json"
            with self.assertRaisesRegex(
                LivePlayQualificationError, "failure evidence was written"
            ):
                _qualify_live_play_create_only_for_tests(
                    controller,
                    config=config,
                    provenance=_provenance(config),
                    cohort_receipt=_cohort_receipt(config),
                    wall_clock_latency_receipt=_latency_receipt(config),
                    output_path=destination,
                    episode_runner=_clean_execution,
                )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            reset_count = len(controller.reset_seeds)
            with self.assertRaisesRegex(
                LivePlayQualificationError, "refusing to overwrite"
            ):
                _qualify_live_play_create_only_for_tests(
                    controller,
                    config=config,
                    provenance=_provenance(config),
                    cohort_receipt=_cohort_receipt(config),
                    wall_clock_latency_receipt=_latency_receipt(config),
                    output_path=destination,
                    episode_runner=_clean_execution,
                )

        self.assertEqual(payload["status"], "test_only")
        self.assertEqual(payload["failure"]["type"], "LivePlayQualificationError")
        self.assertEqual(len(controller.reset_seeds), reset_count)

    def test_wrong_environment_identity_cannot_be_scored(self) -> None:
        config = _config()

        def wrong_world_runner(
            policy: object, seed: int, closed_config: object
        ) -> LiveEpisodeExecution:
            return replace(
                _clean_execution(policy, seed, closed_config),
                environment_id="moving_shapes.v1",
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "wrong-world.json"
            with self.assertRaisesRegex(
                LivePlayQualificationError, "failure evidence was written"
            ):
                _qualify_live_play_create_only_for_tests(
                    _RecordingController(),
                    config=config,
                    provenance=_provenance(config),
                    cohort_receipt=_cohort_receipt(config),
                    wall_clock_latency_receipt=_latency_receipt(config),
                    output_path=destination,
                    episode_runner=wrong_world_runner,
                )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "test_only")
        self.assertIn("exact maze", payload["failure"]["message"])

    def test_invalid_controller_still_leaves_a_complete_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "invalid-controller.json"
            with self.assertRaisesRegex(
                LivePlayQualificationError, "failure evidence was written"
            ):
                _qualify_live_play_create_only_for_tests(
                    object(),  # type: ignore[arg-type]
                    config=_config(),
                    provenance=_provenance(),
                    cohort_receipt=_cohort_receipt(_config()),
                    wall_clock_latency_receipt=_latency_receipt(_config()),
                    output_path=destination,
                    episode_runner=_clean_execution,
                )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "test_only")
        self.assertEqual(payload["failure"]["type"], "LivePlayQualificationError")
        self.assertIn("act_rgb", payload["failure"]["message"])

    def test_provenance_and_frequency_are_strict(self) -> None:
        with self.assertRaisesRegex(LivePlayQualificationError, "lowercase SHA-256"):
            LivePlayProvenance(
                qualification_id="bad",
                model_identity="bad",
                checkpoint_envelope=POST_DGX_CHECKPOINT_ENVELOPE,
                checkpoint_sha256="ABC",
                model_state_sha256=_h("state"),
                model_config_sha256=_h("1"),
                feature_flags_sha256=_h("flags"),
                source_bundle_sha256=_h("2"),
                evaluator_bundle_sha256=_h("3"),
                environment_source_sha256=_h("4"),
                preregistration_sha256=_h("5"),
                cohort_binding_sha256=_h("6"),
                post_dgx_release_body_sha256=_h("7"),
                post_dgx_release_file_sha256=_h("8"),
            )
        with self.assertRaisesRegex(LivePlayQualificationError, "60 Hz"):
            _config(tick_period_ns=TICK_PERIOD_NS_60HZ - 1)


if __name__ == "__main__":
    unittest.main()
