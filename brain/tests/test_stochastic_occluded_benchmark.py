"""Unit tests for Stochastic, Partially-Occluded & Information-Gathering Benchmark Suite."""

import unittest
import torch

from irene_brain.environments.stochastic_occluded_benchmark import (
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


if __name__ == "__main__":
    unittest.main()
