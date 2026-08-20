"""Tests for Gate 6 FAIL diagnosis (JSON parse + cheap untrained probes)."""

from __future__ import annotations

import unittest

from irene_brain.evaluation.gate6_fail_diagnosis import (
    FROZEN_GRU_IQM,
    FROZEN_K_CURVE,
    FROZEN_KNOCKOUT_PCT,
    FROZEN_PB_IQM,
    FROZEN_RELATIVE_PCT,
    FROZEN_SEED45_KNOCKOUT_PCT,
    diagnose,
    k_curve,
    leave_one_out_iqm,
    seed_table,
)
from irene_brain.evaluation.phase25_remaining_gates import CAMPAIGN_JSON, GATE6_JSON, load_json


class TestGate6FailDiagnosis(unittest.TestCase):
    def test_frozen_gate6_numbers_are_not_rewritten(self) -> None:
        gate6 = load_json(GATE6_JSON)
        self.assertAlmostEqual(FROZEN_PB_IQM, -27.758, places=3)
        self.assertAlmostEqual(FROZEN_GRU_IQM, -25.694, places=3)
        self.assertAlmostEqual(FROZEN_RELATIVE_PCT, -8.04, places=2)
        self.assertAlmostEqual(FROZEN_KNOCKOUT_PCT, 1343.0, places=0)
        self.assertEqual(FROZEN_SEED45_KNOCKOUT_PCT, 0.0)
        self.assertAlmostEqual(float(gate6["pb_iqm_return"]), FROZEN_PB_IQM, places=3)
        self.assertAlmostEqual(float(gate6["gru_iqm_return"]), FROZEN_GRU_IQM, places=3)
        self.assertEqual(gate6["gate6_status"], "FAIL")
        self.assertFalse(gate6["architecture_superiority_claimed"])

    def test_seed_45_is_zero_knockout_and_gru_outlier(self) -> None:
        gate6 = load_json(GATE6_JSON)
        rows = {row.seed: row for row in seed_table(gate6)}
        seed45 = rows[45]
        self.assertEqual(seed45.knockout_pct, 0.0)
        self.assertEqual(seed45.family_b_normal, -21.0)
        self.assertEqual(seed45.family_b_knockout, -21.0)
        self.assertLess(seed45.pb_iqm, seed45.gru_iqm)
        self.assertLess(seed45.pb_family_e, -700.0)
        self.assertGreater(seed45.gru_family_e, -50.0)

    def test_k_curve_is_anti_monotonic(self) -> None:
        campaign = load_json(CAMPAIGN_JSON)
        self.assertEqual(k_curve(campaign), FROZEN_K_CURVE)

    def test_leave_one_out_without_seed_45_does_not_rewrite_full_iqm(self) -> None:
        gate6 = load_json(GATE6_JSON)
        loo = leave_one_out_iqm(gate6, 45)
        self.assertAlmostEqual(loo["pb_iqm"], -28.04, places=2)
        self.assertAlmostEqual(loo["gru_iqm"], -28.48, places=2)
        self.assertGreater(loo["relative_pct"], 0.0)
        self.assertNotAlmostEqual(loo["pb_iqm"], FROZEN_PB_IQM, places=3)

    def test_diagnose_payload_names_the_new_probe(self) -> None:
        payload = diagnose()
        self.assertEqual(payload["probe_id"], "why-gru-wins-despite-knockout-v1")
        self.assertEqual(payload["untrained_actuator"]["permutation_invariance"]["argmax_change_pct"], 0.0)
        swap = payload["untrained_actuator"]["register_swap_when_registers_differ"]
        self.assertGreaterEqual(swap["argmax_change_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
