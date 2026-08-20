"""Unit tests for Decision-Critical Forced-Choice Benchmark Suite."""

import unittest
import torch

from irene_brain.environments.decision_critical_suite import (
    evaluate_decision_critical_benchmark,
)
from irene_brain.model.thought_mediated_model import build_resource_matched_thought_model


class TestDecisionCriticalSuite(unittest.TestCase):
    def test_evaluate_decision_critical_benchmark(self) -> None:
        model = build_resource_matched_thought_model(4)
        report = evaluate_decision_critical_benchmark(model, num_decision_points=10)
        self.assertEqual(report.num_decision_points, 10)
        self.assertEqual(report.optimal_choices + report.suboptimal_choices, 10)
        self.assertGreaterEqual(report.decision_accuracy_pct, 0.0)
        self.assertLessEqual(report.decision_accuracy_pct, 100.0)
        self.assertGreaterEqual(report.total_decision_score, -10.0)
        self.assertLessEqual(report.total_decision_score, +10.0)


if __name__ == "__main__":
    unittest.main()
