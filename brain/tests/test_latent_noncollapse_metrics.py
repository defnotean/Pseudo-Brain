"""Synthetic CPU-only tests for latent non-collapse qualification metrics."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from irene_brain.evaluation.latent_noncollapse_metrics import (
    LatentNonCollapseThresholds,
    evaluate_latent_noncollapse,
)


def _thresholds() -> LatentNonCollapseThresholds:
    return LatentNonCollapseThresholds(
        minimum_samples=4,
        minimum_nonzero_variance_fraction=0.75,
        minimum_covariance_effective_rank=2.5,
        minimum_mean_pairwise_cosine_distance=0.1,
    )


class LatentNonCollapseMetricsTests(unittest.TestCase):
    def test_isotropic_population_passes_with_expected_effective_rank(self) -> None:
        latents = np.concatenate((np.eye(4), -np.eye(4)), axis=0)

        report = evaluate_latent_noncollapse(latents, thresholds=_thresholds())

        self.assertTrue(report.computable)
        self.assertTrue(report.passed)
        self.assertTrue(report.variance_is_finite)
        self.assertTrue(report.has_nonzero_variance)
        self.assertEqual(report.failure_reasons, ())
        self.assertEqual(report.nonzero_variance_dimensions, 4)
        self.assertAlmostEqual(report.nonzero_variance_fraction or 0.0, 1.0, places=12)
        self.assertAlmostEqual(report.covariance_effective_rank or 0.0, 4.0, places=12)
        self.assertAlmostEqual(
            report.mean_pairwise_cosine_distance or 0.0,
            8.0 / 7.0,
            places=12,
        )

    def test_rank_one_population_reports_rank_and_fails_rank_gate(self) -> None:
        latents = np.asarray(
            [
                [-2.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        report = evaluate_latent_noncollapse(latents, thresholds=_thresholds())

        self.assertTrue(report.computable)
        self.assertFalse(report.passed)
        self.assertTrue(report.variance_is_finite)
        self.assertTrue(report.has_nonzero_variance)
        self.assertAlmostEqual(report.covariance_effective_rank or 0.0, 1.0, places=12)
        self.assertEqual(report.nonzero_variance_dimensions, 1)
        self.assertTrue(
            any(
                reason.startswith("covariance_effective_rank_below_minimum")
                for reason in report.failure_reasons
            )
        )

    def test_constant_population_is_a_clean_failed_report_not_nan(self) -> None:
        latents = np.full((8, 4), 3.0, dtype=np.float64)

        report = evaluate_latent_noncollapse(latents, thresholds=_thresholds())

        self.assertTrue(report.computable)
        self.assertFalse(report.passed)
        self.assertTrue(report.variance_is_finite)
        self.assertFalse(report.has_nonzero_variance)
        self.assertEqual(report.mean_feature_variance, 0.0)
        self.assertEqual(report.covariance_effective_rank, 0.0)
        self.assertEqual(report.mean_pairwise_cosine_distance, 0.0)
        self.assertIn("zero_total_variance", report.failure_reasons)
        self.assertFalse(
            any(
                math_value != math_value
                for math_value in (
                    report.mean_feature_variance,
                    report.covariance_effective_rank,
                    report.mean_pairwise_cosine_distance,
                )
                if math_value is not None
            )
        )

    def test_too_few_samples_returns_explicit_noncomputable_report(self) -> None:
        report = evaluate_latent_noncollapse(
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            thresholds=_thresholds(),
        )

        self.assertFalse(report.computable)
        self.assertFalse(report.passed)
        self.assertFalse(report.variance_is_finite)
        self.assertFalse(report.has_nonzero_variance)
        self.assertIsNone(report.mean_feature_variance)
        self.assertIsNone(report.covariance_effective_rank)
        self.assertIsNone(report.mean_pairwise_cosine_distance)
        self.assertEqual(report.failure_reasons, ("insufficient_samples:1<4",))

    def test_nonfinite_and_zero_norm_inputs_fail_without_nan_metrics(self) -> None:
        nonfinite = evaluate_latent_noncollapse(
            np.asarray([[1.0, 2.0], [3.0, np.inf], [4.0, 5.0], [6.0, 7.0]]),
            thresholds=_thresholds(),
        )
        zero_norm = evaluate_latent_noncollapse(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            thresholds=_thresholds(),
        )

        self.assertFalse(nonfinite.computable)
        self.assertEqual(nonfinite.failure_reasons, ("non_finite_latents",))
        self.assertIsNone(nonfinite.covariance_effective_rank)
        self.assertFalse(zero_norm.computable)
        self.assertIn("zero_norm_latent", zero_norm.failure_reasons)
        self.assertIsNone(zero_norm.mean_pairwise_cosine_distance)

    def test_finite_inputs_with_overflowing_variance_fail_cleanly(self) -> None:
        report = evaluate_latent_noncollapse(
            np.asarray(
                [[1.0e308, 1.0], [-1.0e308, 2.0], [1.0e308, 3.0], [-1.0e308, 4.0]]
            ),
            thresholds=_thresholds(),
        )

        self.assertTrue(report.all_finite)
        self.assertFalse(report.variance_is_finite)
        self.assertFalse(report.computable)
        self.assertFalse(report.passed)
        self.assertIn("non_finite_variance", report.failure_reasons)
        self.assertIsNone(report.mean_feature_variance)

    def test_results_and_json_projection_are_deterministic(self) -> None:
        latents = np.asarray(
            [[0.2, -0.7, 1.1], [1.4, 0.3, -0.2], [-0.4, 1.2, 0.8], [0.9, -1.0, 0.5]],
            dtype=np.float64,
        )

        first = evaluate_latent_noncollapse(latents, thresholds=_thresholds())
        second = evaluate_latent_noncollapse(latents.copy(), thresholds=_thresholds())

        self.assertEqual(first, second)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertIsInstance(first.as_dict()["failure_reasons"], list)

    def test_rejects_invalid_shapes_and_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            evaluate_latent_noncollapse(np.ones(4), thresholds=_thresholds())
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            evaluate_latent_noncollapse(np.empty((4, 0)), thresholds=_thresholds())
        with self.assertRaisesRegex(ValueError, "minimum_samples"):
            LatentNonCollapseThresholds(
                minimum_samples=1,
                minimum_nonzero_variance_fraction=0.5,
                minimum_covariance_effective_rank=2.0,
                minimum_mean_pairwise_cosine_distance=0.1,
            )


if __name__ == "__main__":
    unittest.main()
