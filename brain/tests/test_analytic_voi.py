"""Analytic verification of Expected Consequence Utility and Value of Information (VoI)."""

from __future__ import annotations

import unittest

from irene_brain.evaluation.belief_collapse_diagnostics import compute_analytic_voi_example


class TestAnalyticVoI(unittest.TestCase):
    def test_voi_calculation(self) -> None:
        q_right, q_wait = compute_analytic_voi_example()
        self.assertAlmostEqual(q_right, -5.5)
        self.assertAlmostEqual(q_wait, 1.8)
        self.assertGreater(q_wait, q_right)


if __name__ == "__main__":
    unittest.main()
