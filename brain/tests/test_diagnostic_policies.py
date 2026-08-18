from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.environments.moving_shapes import MovingShapesEnv
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    control_audit_stats,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.diagnostic_policies import (
    NoOpPolicy,
    OraclePolicy,
    RandomMovementPolicy,
    ScriptedKeysDoorsSolver,
    ScriptedMazeChasePlannerPolicy,
    ScriptedPelletTeacherPolicy,
    ScriptedTargetChasePolicy,
    default_diagnostic_policies,
    evaluate_diagnostic_policy_suite,
)
from irene_brain.types import GenericControl


def _config(**overrides: object) -> ClosedLoopPlayConfig:
    knobs: dict[str, object] = {
        "episode_seeds": (5, 9),
        "max_ticks": 60,
        "hazard_count": 3,
    }
    knobs.update(overrides)
    return ClosedLoopPlayConfig(**knobs)  # type: ignore[arg-type]


class PolicyContractTests(unittest.TestCase):
    def test_identities_are_unique_and_stable(self) -> None:
        identities = [policy.identity for policy in default_diagnostic_policies()]
        self.assertEqual(len(set(identities)), 5)
        self.assertEqual(
            identities,
            [
                "diagnostic.noop.v1",
                "diagnostic.random_movement.v1",
                "diagnostic.scripted_chase.v1",
                "diagnostic.scripted_pellet_teacher.v1",
                "diagnostic.oracle_privileged.v1",
            ],
        )

    def test_only_the_oracle_uses_privileged_state(self) -> None:
        flags = {
            policy.identity: policy.uses_privileged_state
            for policy in default_diagnostic_policies()
        }
        self.assertEqual(
            flags,
            {
                "diagnostic.noop.v1": False,
                "diagnostic.random_movement.v1": False,
                "diagnostic.scripted_chase.v1": False,
                "diagnostic.scripted_pellet_teacher.v1": False,
                "diagnostic.oracle_privileged.v1": True,
            },
        )

    def test_reset_rejects_non_integer_seeds(self) -> None:
        for policy in default_diagnostic_policies():
            with self.assertRaises(TypeError):
                policy.reset(True)  # type: ignore[arg-type]

    def test_oracle_requires_bind_and_a_moving_shapes_env(self) -> None:
        oracle = OraclePolicy()
        oracle.reset(1)
        with self.assertRaises(RuntimeError):
            oracle.act(MovingShapesEnv(hazard_count=1).reset(1))
        with self.assertRaises(TypeError):
            oracle.bind(object())  # type: ignore[arg-type]

    def test_runner_rejects_objects_without_the_contract(self) -> None:
        with self.assertRaises(TypeError):
            run_policy_closed_loop_episode(object(), seed=1, config=_config())


class ControlAuditStatsTests(unittest.TestCase):
    def test_neutral_control_is_all_zero(self) -> None:
        stats = control_audit_stats(GenericControl())
        self.assertEqual(stats["active_button_count"], 0)
        self.assertEqual(stats["movement_mask"], 0)
        self.assertEqual(stats["opposite_conflict"], 0)
        self.assertEqual(stats["continuous_max_abs"], 0.0)

    def test_movement_and_conflict_flags_match_decode_semantics(self) -> None:
        stats = control_audit_stats(GenericControl(keys_down=(22, 26, 40)))
        self.assertEqual(stats["movement_mask"], 0b0101)
        self.assertEqual(stats["opposite_conflict"], 1)
        self.assertEqual(stats["non_movement_key_count"], 1)

    def test_rejects_non_control(self) -> None:
        with self.assertRaises(TypeError):
            control_audit_stats(object())  # type: ignore[arg-type]


class DiagnosticEpisodeTests(unittest.TestCase):
    def test_noop_never_moves_and_never_collects(self) -> None:
        report = run_policy_closed_loop_episode(
            NoOpPolicy(), seed=5, config=_config()
        )
        self.assertEqual(report.ticks_advanced, 60)
        self.assertEqual(report.decisions_submitted, 60)
        self.assertEqual(report.decisions_rejected, 0)
        self.assertEqual(report.targets_collected, 0)
        self.assertEqual(report.movement_mask_histogram, ((0, 60),))

    def test_scripted_chaser_collects_targets_from_pixels_only(self) -> None:
        report = run_policy_closed_loop_episode(
            ScriptedTargetChasePolicy(), seed=5, config=_config()
        )
        self.assertGreaterEqual(report.targets_collected, 1)
        self.assertEqual(report.opposite_conflicts, 0)
        self.assertEqual(report.non_movement_key_activations, 0)

    def test_oracle_collects_without_collisions(self) -> None:
        total_targets = 0
        total_collisions = 0
        for seed in (5, 9):
            report = run_policy_closed_loop_episode(
                OraclePolicy(), seed=seed, config=_config()
            )
            total_targets += report.targets_collected
            total_collisions += report.collisions
        self.assertGreaterEqual(total_targets, 2)
        self.assertEqual(total_collisions, 0)

    def test_random_policy_is_deterministic_per_seed(self) -> None:
        first = run_policy_closed_loop_episode(
            RandomMovementPolicy(), seed=5, config=_config()
        )
        second = run_policy_closed_loop_episode(
            RandomMovementPolicy(), seed=5, config=_config()
        )
        self.assertEqual(first, second)
        self.assertGreater(len(first.movement_mask_histogram), 1)

    def test_episode_seed_changes_the_world_and_the_policy(self) -> None:
        first = run_policy_closed_loop_episode(
            RandomMovementPolicy(), seed=5, config=_config()
        )
        second = run_policy_closed_loop_episode(
            RandomMovementPolicy(), seed=6, config=_config()
        )
        self.assertNotEqual(first.movement_mask_histogram, second.movement_mask_histogram)


class DiagnosticSuiteTests(unittest.TestCase):
    def test_suite_returns_one_report_per_policy_in_order(self) -> None:
        config = _config(max_ticks=30)
        reports = evaluate_diagnostic_policy_suite(
            default_diagnostic_policies(), config=config
        )
        self.assertEqual(len(reports), 5)
        for report, policy in zip(reports, default_diagnostic_policies()):
            self.assertEqual(
                report.model_description, f"diagnostic policy {policy.identity}"
            )
            self.assertEqual(len(report.episodes), 2)

    def test_suite_is_deterministic(self) -> None:
        config = _config(max_ticks=30)
        first = evaluate_diagnostic_policy_suite(
            default_diagnostic_policies(), config=config
        )
        second = evaluate_diagnostic_policy_suite(
            default_diagnostic_policies(), config=config
        )
        self.assertEqual(
            [report.sha256 for report in first],
            [report.sha256 for report in second],
        )

    def test_suite_orders_the_floors_below_the_oracle(self) -> None:
        config = _config(max_ticks=120)
        reports = evaluate_diagnostic_policy_suite(
            default_diagnostic_policies(), config=config
        )
        totals = [report.to_dict()["totals"] for report in reports]
        noop_targets = totals[0]["targets_collected"]
        oracle_targets = totals[4]["targets_collected"]
        self.assertEqual(noop_targets, 0)
        self.assertGreater(oracle_targets, noop_targets)

    def test_suite_rejects_duplicate_identities_and_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_diagnostic_policy_suite(
                (NoOpPolicy(), NoOpPolicy()), config=_config()
            )
        with self.assertRaises(ValueError):
            evaluate_diagnostic_policy_suite((), config=_config())
        with self.assertRaises(ValueError):
            evaluate_diagnostic_policy_suite(
                default_diagnostic_policies(), config=object()  # type: ignore[arg-type]
            )


class MazeChasePlannerTests(unittest.TestCase):
    """The lookahead planner maps the scripted maze_chase frontier."""

    def _run(self, policy: object, seed: int, max_ticks: int = 240, **env_overrides: object):
        knobs: dict[str, object] = {
            "ghost_count": 3,
            "ghost_period": 2,
            "extra_loops": 16,
            "max_ticks": max_ticks,
        }
        knobs.update(env_overrides)
        return run_policy_closed_loop_episode(
            policy,
            seed=seed,
            config=_config(max_ticks=max_ticks),
            environment_factory=lambda: MazeChaseEnv(**knobs),  # type: ignore[arg-type]
        )

    def test_contract_flags_and_constructor_validation(self) -> None:
        planner = ScriptedMazeChasePlannerPolicy()
        self.assertEqual(planner.identity, "diagnostic.scripted_maze_chase_planner.v1")
        self.assertFalse(planner.uses_privileged_state)
        with self.assertRaises(TypeError):
            ScriptedMazeChasePlannerPolicy(ghost_period=True)
        with self.assertRaises(TypeError):
            ScriptedMazeChasePlannerPolicy(candidate_pellets=2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ScriptedMazeChasePlannerPolicy(ghost_period=0)
        with self.assertRaises(ValueError):
            ScriptedMazeChasePlannerPolicy(candidate_pellets=33)
        with self.assertRaises(ValueError):
            ScriptedMazeChasePlannerPolicy(horizon=257)
        with self.assertRaises(TypeError):
            ScriptedMazeChasePlannerPolicy(input_delay_ticks=True)
        with self.assertRaises(ValueError):
            ScriptedMazeChasePlannerPolicy(input_delay_ticks=17)
        with self.assertRaises(TypeError):
            ScriptedMazeChasePlannerPolicy(player_period=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ScriptedMazeChasePlannerPolicy(player_period=0)
        with self.assertRaises(TypeError):
            ScriptedMazeChasePlannerPolicy(ghost_elroy=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            planner.reset(True)  # type: ignore[arg-type]

    def test_requires_the_canonical_grid_frame(self) -> None:
        from irene_brain.types import Observation, RgbFrame

        planner = ScriptedMazeChasePlannerPolicy()
        planner.reset(1)
        wrong_size = Observation(
            frame_id=0,
            capture_tick=0,
            elapsed_ns=0,
            rgb=RgbFrame(width=8, height=8, pixels=bytes(8 * 8 * 3)),
            previous_control=GenericControl(),
            audio_pcm_s16le=None,
            text_inputs=(),
        )
        with self.assertRaises(ValueError):
            planner.act(wrong_size)

    def test_planner_is_deterministic_per_seed(self) -> None:
        first = self._run(ScriptedMazeChasePlannerPolicy(), seed=5)
        second = self._run(ScriptedMazeChasePlannerPolicy(), seed=5)
        self.assertEqual(first, second)

    def test_planner_outplays_the_greedy_teacher(self) -> None:
        planner_reward = 0.0
        teacher_reward = 0.0
        planner_collisions = 0
        teacher_collisions = 0
        for seed in (5, 9):
            planned = self._run(ScriptedMazeChasePlannerPolicy(), seed)
            taught = self._run(ScriptedPelletTeacherPolicy(), seed)
            planner_reward += planned.reward_sum
            teacher_reward += taught.reward_sum
            planner_collisions += planned.collisions
            teacher_collisions += taught.collisions
        self.assertGreater(planner_reward, teacher_reward)
        self.assertLess(planner_collisions, teacher_collisions)

    def test_explicit_canonical_knobs_match_the_defaults(self) -> None:
        plain = self._run(ScriptedMazeChasePlannerPolicy(), seed=5)
        explicit = self._run(
            ScriptedMazeChasePlannerPolicy(
                input_delay_ticks=0, player_period=1, ghost_elroy=False
            ),
            seed=5,
        )
        self.assertEqual(plain, explicit)

    def test_delay_compensation_recovers_the_delayed_slot(self) -> None:
        for seed in (5, 9):
            plain = self._run(
                ScriptedMazeChasePlannerPolicy(), seed, input_delay_ticks=2
            )
            compensated = self._run(
                ScriptedMazeChasePlannerPolicy(input_delay_ticks=2),
                seed,
                input_delay_ticks=2,
            )
            self.assertGreater(compensated.reward_sum, plain.reward_sum)
            self.assertLess(compensated.collisions, plain.collisions)

    def test_elroy_compensation_recovers_the_speed_curve(self) -> None:
        for seed in (5, 9):
            plain = self._run(
                ScriptedMazeChasePlannerPolicy(), seed, ghost_elroy=True
            )
            compensated = self._run(
                ScriptedMazeChasePlannerPolicy(ghost_elroy=True),
                seed,
                ghost_elroy=True,
            )
            self.assertLess(compensated.collisions, plain.collisions)

    def test_player_period_compensation_is_outcome_neutral(self) -> None:
        # Presses submitted on non-move ticks are discarded by the world, so
        # the plain planner's every-tick player model costs nothing on this
        # slot: compensation changes the plan, not the outcome.
        for seed in (5, 9):
            plain = self._run(
                ScriptedMazeChasePlannerPolicy(), seed, player_period=2
            )
            compensated = self._run(
                ScriptedMazeChasePlannerPolicy(player_period=2),
                seed,
                player_period=2,
            )
            self.assertEqual(compensated.reward_sum, plain.reward_sum)
            self.assertEqual(compensated.collisions, plain.collisions)


class KeysDoorsSolverTests(unittest.TestCase):
    """The key→door→target solver maps the scripted keys_doors frontier."""

    def _run(self, policy: object, seed: int, max_ticks: int = 240):
        return run_policy_closed_loop_episode(
            policy,
            seed=seed,
            config=_config(max_ticks=max_ticks),
            environment_factory=lambda: KeysDoorsEnv(max_ticks=max_ticks),
        )

    def test_contract_flags_and_reset_validation(self) -> None:
        solver = ScriptedKeysDoorsSolver()
        self.assertEqual(solver.identity, "diagnostic.scripted_keys_doors_solver.v1")
        self.assertFalse(solver.uses_privileged_state)
        with self.assertRaises(TypeError):
            solver.reset(True)  # type: ignore[arg-type]

    def test_requires_the_canonical_grid_frame(self) -> None:
        from irene_brain.types import Observation, RgbFrame

        solver = ScriptedKeysDoorsSolver()
        solver.reset(1)
        wrong_size = Observation(
            frame_id=0,
            capture_tick=0,
            elapsed_ns=0,
            rgb=RgbFrame(width=8, height=8, pixels=bytes(8 * 8 * 3)),
            previous_control=GenericControl(),
            audio_pcm_s16le=None,
            text_inputs=(),
        )
        with self.assertRaises(ValueError):
            solver.act(wrong_size)

    def test_solver_is_deterministic_per_seed(self) -> None:
        first = self._run(ScriptedKeysDoorsSolver(), seed=5)
        second = self._run(ScriptedKeysDoorsSolver(), seed=5)
        self.assertEqual(first, second)

    def test_solver_collects_targets_where_reactive_policies_score_zero(self) -> None:
        for seed in (5, 9):
            report = self._run(ScriptedKeysDoorsSolver(), seed)
            self.assertGreaterEqual(report.targets_collected, 1)
            self.assertGreaterEqual(report.reward_sum, 1.0)
            self.assertEqual(report.decisions_rejected, 0)

    def test_solver_holds_still_on_foreign_worlds(self) -> None:
        report = run_policy_closed_loop_episode(
            ScriptedKeysDoorsSolver(),
            seed=5,
            config=_config(max_ticks=60, hazard_count=1),
            environment_factory=lambda: MovingShapesEnv(hazard_count=1, max_ticks=60),
        )
        self.assertEqual(report.targets_collected, 0)
        self.assertEqual(report.movement_mask_histogram, ((0, 60),))


if __name__ == "__main__":
    unittest.main()
