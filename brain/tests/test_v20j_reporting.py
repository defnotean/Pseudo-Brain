"""Regression contracts for V2.0j atomic progress reporting."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage_v20j_accumulated_macro_policy import summarize


def _record(lift: float, status: str = "completed") -> dict[str, object]:
    return {"status": status, "final_eval": {"mean_lift": lift}}


class V20JReportingContract(unittest.TestCase):
    def test_partial_seed_set_is_reported_as_in_progress(self) -> None:
        summary = summarize([_record(-0.02), _record(-0.04)], expected_runs=4)

        self.assertEqual(summary["status"], "in_progress")
        self.assertEqual(len(summary["runs"]), 2)

    def test_full_seed_set_is_reported_as_completed(self) -> None:
        summary = summarize(
            [_record(-0.02), _record(-0.04), _record(-0.03), _record(-0.05)],
            expected_runs=4,
        )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(summary["runs"]), 4)

    def test_nonfinite_run_still_reports_training_failure(self) -> None:
        summary = summarize(
            [_record(-0.02), _record(0.0, status="nonfinite")],
            expected_runs=4,
        )

        self.assertEqual(summary["status"], "training_failure")


if __name__ == "__main__":
    unittest.main()
