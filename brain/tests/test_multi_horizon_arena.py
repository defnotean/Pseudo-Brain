"""Unit tests for Multi-Horizon Multi-Uncertainty Ambiguity Arena."""

import unittest
import torch

from irene_brain.environments.multi_horizon_ambiguity_arena import (
    evaluate_multi_horizon_ambiguity_arena,
    evaluate_delayed_resolution_benchmark,
    evaluate_nested_uncertainty_benchmark,
    evaluate_non_stationary_regime_shift_benchmark,
)
from irene_brain.model.thought_mediated_model import (
    ProposalGRUBaseline,
    build_resource_matched_thought_model,
)


class TestMultiHorizonAmbiguityArena(unittest.TestCase):
    def test_evaluate_multi_horizon_arena(self) -> None:
        model = build_resource_matched_thought_model(4)
        report = evaluate_multi_horizon_ambiguity_arena(model, num_episodes=5, base_seed=40000)
        self.assertEqual(report.num_episodes, 5)
        self.assertGreaterEqual(report.staggered_survival_rate_pct, 0.0)
        self.assertLessEqual(report.staggered_survival_rate_pct, 100.0)
        self.assertGreaterEqual(report.catastrophic_branch_avoidance_pct, 0.0)
        self.assertLessEqual(report.catastrophic_branch_avoidance_pct, 100.0)
        self.assertGreaterEqual(report.working_memory_retention_score, 0.0)
        self.assertLessEqual(report.working_memory_retention_score, 1.0)
        self.assertIsInstance(report.mean_multi_horizon_return, float)

    def test_evaluate_multi_horizon_arena_with_gru(self) -> None:
        gru_model = ProposalGRUBaseline()
        report = evaluate_multi_horizon_ambiguity_arena(gru_model, num_episodes=5, base_seed=40000)
        self.assertEqual(report.num_episodes, 5)
        self.assertGreaterEqual(report.staggered_survival_rate_pct, 0.0)
        self.assertIsInstance(report.mean_multi_horizon_return, float)

    def test_delayed_resolution_benchmark(self) -> None:
        model = build_resource_matched_thought_model(4)
        report = evaluate_delayed_resolution_benchmark(model, delay_ticks=10, num_episodes=5, base_seed=50000)
        self.assertEqual(report.num_episodes, 5)
        self.assertEqual(report.delay_ticks, 10)
        self.assertGreaterEqual(report.delayed_decision_acc_pct, 0.0)
        self.assertLessEqual(report.delayed_decision_acc_pct, 100.0)
        self.assertGreaterEqual(report.hypothesis_retention_fidelity, 0.0)
        self.assertLessEqual(report.hypothesis_retention_fidelity, 1.0)

    def test_nested_uncertainty_benchmark(self) -> None:
        model = build_resource_matched_thought_model(4)
        report = evaluate_nested_uncertainty_benchmark(model, num_episodes=5, base_seed=60000)
        self.assertEqual(report.num_episodes, 5)
        self.assertGreaterEqual(report.four_to_two_collapse_acc_pct, 0.0)
        self.assertGreaterEqual(report.two_to_one_collapse_acc_pct, 0.0)
        self.assertGreaterEqual(report.partial_collapse_entropy_bits, 0.0)
        self.assertGreaterEqual(report.compound_goal_acc_pct, 0.0)

    def test_non_stationary_shift_benchmark(self) -> None:
        model = build_resource_matched_thought_model(4)
        report = evaluate_non_stationary_regime_shift_benchmark(model, episode_length=20, shift_tick=10, base_seed=70000)
        self.assertGreaterEqual(report.pre_shift_accuracy_pct, 0.0)
        self.assertGreaterEqual(report.post_shift_accuracy_pct, 0.0)
        self.assertGreaterEqual(report.probability_recovery_time_ticks, 0.0)
        self.assertGreaterEqual(report.empirical_cross_entropy_loss, 0.0)

    def test_hostile_ablation_reset_every_frame(self) -> None:
        model = build_resource_matched_thought_model(4)
        report_normal = evaluate_delayed_resolution_benchmark(model, delay_ticks=10, num_episodes=5, reset_every_frame=False)
        report_reset = evaluate_delayed_resolution_benchmark(model, delay_ticks=10, num_episodes=5, reset_every_frame=True)
        self.assertIsInstance(report_normal.delayed_decision_acc_pct, float)
        self.assertIsInstance(report_reset.delayed_decision_acc_pct, float)


if __name__ == "__main__":
    unittest.main()

