"""Focused tests for the Phase 2.5 remaining-gates CPU parse."""

from __future__ import annotations

import unittest

from irene_brain.evaluation.phase25_remaining_gates import (
    CAMPAIGN_JSON,
    GATE6_JSON,
    analyze_remaining_gates,
    load_json,
    measure_permutation_action_change,
)


class TestPhase25RemainingGates(unittest.TestCase):
    def test_committed_campaign_gate_table(self) -> None:
        campaign = load_json(CAMPAIGN_JSON)
        gate6 = load_json(GATE6_JSON)
        rows = analyze_remaining_gates(campaign, gate6, permutation={"argmax_change_pct": 0.0, "action_dist_l1_pct": 0.0})
        by_id = {row.gate_id: row for row in rows}

        self.assertEqual(by_id["Gate 1"].verdict, "UNKNOWN")
        self.assertIn("1418.75%", by_id["Gate 1"].measured)
        self.assertIn("-32.0", by_id["Gate 1"].measured)
        self.assertIn("-486.0", by_id["Gate 1"].measured)

        self.assertEqual(by_id["Gate 2"].verdict, "FAIL")
        self.assertIn("98.0/98.0/32.0/-98.0/-98.0", by_id["Gate 2"].measured)

        self.assertEqual(by_id["Gate 3"].verdict, "UNKNOWN")
        self.assertEqual(by_id["Gate 3"].measured, "not measured")

        self.assertEqual(by_id["Gate 4"].verdict, "PASS")
        self.assertIn("0.00%", by_id["Gate 4"].measured)

        self.assertEqual(by_id["Gate 5"].verdict, "FAIL")
        self.assertIn("0.00%", by_id["Gate 5"].measured)

        self.assertEqual(by_id["Gate 6"].verdict, "FAIL")
        self.assertIn("-8.04%", by_id["Gate 6"].measured)
        self.assertIn("-27.758", by_id["Gate 6"].measured)
        self.assertIn("-25.694", by_id["Gate 6"].measured)

        self.assertEqual(by_id["Gate 7"].verdict, "UNKNOWN")
        self.assertEqual(by_id["Gate 7"].measured, "not measured")

        self.assertEqual(by_id["Gate 8"].verdict, "UNKNOWN")
        self.assertEqual(by_id["Gate 8"].measured, "not measured")

    def test_untrained_permutation_action_change_is_zero(self) -> None:
        result = measure_permutation_action_change(seed=0, batches=4, thoughtlets=8, core_width=16)
        self.assertEqual(result["argmax_change_pct"], 0.0)
        self.assertLessEqual(result["action_dist_l1_pct"], 1e-3)


if __name__ == "__main__":
    unittest.main()
