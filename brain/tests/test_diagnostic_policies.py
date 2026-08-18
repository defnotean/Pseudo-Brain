from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


if __name__ == "__main__":
    unittest.main()
