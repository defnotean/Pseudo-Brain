"""Unit tests for Stochastic, Partially-Occluded & Information-Gathering Benchmark Suite."""

import unittest
import torch

from irene_brain.environments.stochastic_occluded_benchmark import (
    evaluate_sequential_probe_resolution,
    evaluate_stochastic_occluded_benchmark,
)
from irene_brain.model.thought_mediated_model import build_resource_matched_thought_model


class TestStochasticOccludedSuite(unittest.TestCase):
    def test_evaluate_stochastic_benchmark(self) -> None:
        model = build_resource_matched_thought_model(4)
        report = evaluate_stochastic_occluded_benchmark(model, num_eval_states=10)
        self.assertEqual(report.num_eval_states, 10)
        self.assertGreaterEqual(report.stochastic_decision_accuracy_pct, 0.0)
        self.assertLessEqual(report.stochastic_decision_accuracy_pct, 100.0)
        self.assertGreaterEqual(report.greedy_gamble_avoidance_pct, 0.0)

    def test_evaluate_sequential_probe_resolution(self) -> None:
        model = build_resource_matched_thought_model(4)
        report = evaluate_sequential_probe_resolution(model, num_episodes=5, base_seed=25000)
        self.assertEqual(report.num_episodes, 5)
        self.assertGreaterEqual(report.probe_at_t0_rate_pct, 0.0)
        self.assertLessEqual(report.probe_at_t0_rate_pct, 100.0)
        self.assertGreaterEqual(report.posterior_resolution_accuracy_pct, 0.0)
        self.assertLessEqual(report.blind_gamble_death_rate_pct, 100.0)
        self.assertIsInstance(report.mean_episode_return, float)


if __name__ == "__main__":
    unittest.main()
