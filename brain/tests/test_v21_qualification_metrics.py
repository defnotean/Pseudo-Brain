"""Synthetic, CPU-only tests for the V2.1 qualification statistics."""
from __future__ import annotations

import unittest

import numpy as np

from irene_brain.evaluation.v21_qualification_metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_DERANGEMENT_REPETITIONS,
    QualificationMetricsError,
    all_action_probability_metrics,
    binary_probability_metrics,
    clustered_improvement_bootstrap,
    evaluate_cross_episode_derangements,
    evaluate_train_fitted_baselines,
    fit_train_baselines,
    fixed_cross_episode_derangement_indices,
    fixed_cross_episode_derangements,
    gather_factual_rows,
)


class BinaryQualificationMetricTests(unittest.TestCase):
    def test_known_binary_metrics_include_equal_mass_ece_and_signed_bias(self) -> None:
        targets = np.asarray([0, 1, 0, 1], dtype=np.float64)
        probabilities = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float64)

        metrics = binary_probability_metrics(targets, probabilities, ece_bins=2)

        expected_bce = -np.mean(
            targets * np.log(probabilities)
            + (1.0 - targets) * np.log1p(-probabilities)
        )
        self.assertEqual(metrics.count, 4)
        self.assertEqual(metrics.positives, 2)
        self.assertAlmostEqual(metrics.bce, float(expected_bce))
        self.assertAlmostEqual(metrics.brier, 0.025)
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(metrics.pr_auc, 1.0)
        self.assertAlmostEqual(metrics.ece_equal_mass, 0.15)
        self.assertAlmostEqual(metrics.calibration_bias, 0.0)

    def test_tied_constant_score_has_chance_auc_and_prevalence_average_precision(self) -> None:
        metrics = binary_probability_metrics(
            [0, 1, 1, 0, 1],
            [0.4, 0.4, 0.4, 0.4, 0.4],
        )

        self.assertEqual(metrics.roc_auc, 0.5)
        self.assertAlmostEqual(metrics.pr_auc, 0.6)
        self.assertAlmostEqual(metrics.calibration_bias, -0.2)

    def test_undefined_auc_and_nonprobabilities_fail_closed(self) -> None:
        with self.assertRaisesRegex(QualificationMetricsError, "positive and one negative"):
            binary_probability_metrics([0, 0], [0.1, 0.2])
        with self.assertRaisesRegex(QualificationMetricsError, r"\[0, 1\]"):
            binary_probability_metrics([0, 1], [0.1, 1.1])


class ActionAndBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action_ids = (10, 20)
        self.train_all_targets = np.asarray(
            [
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 0],
            ],
            dtype=np.float64,
        )
        self.train_factual_actions = np.asarray([10, 20, 10, 20, 10, 20, 10, 20])
        self.train_factual_targets = np.asarray([0, 1, 1, 1, 0, 0, 0, 1])

    def test_all_action_report_preserves_semantic_action_identity(self) -> None:
        probabilities = np.where(self.train_all_targets == 1.0, 0.8, 0.2)
        report = all_action_probability_metrics(
            self.train_all_targets,
            probabilities,
            action_ids=self.action_ids,
            ece_bins=4,
        )

        self.assertEqual(report.action_ids, self.action_ids)
        self.assertEqual(tuple(entry.action_id for entry in report.per_action), self.action_ids)
        self.assertEqual(report.aggregate.roc_auc, 1.0)
        self.assertEqual(report.per_action[0].metrics.roc_auc, 1.0)
        self.assertEqual(report.per_action[1].metrics.roc_auc, 1.0)

    def test_train_priors_are_frozen_separately_for_branch_and_factual_data(self) -> None:
        baselines = fit_train_baselines(
            self.train_all_targets,
            self.train_factual_actions,
            self.train_factual_targets,
            action_ids=self.action_ids,
        )

        self.assertEqual(baselines.all_action.probabilities, (0.25, 0.625))
        self.assertEqual(baselines.all_action.sample_counts, (8, 8))
        self.assertEqual(baselines.factual.probabilities, (0.25, 0.75))
        self.assertEqual(baselines.factual.sample_counts, (4, 4))

        evaluation_targets = np.asarray(
            [[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float64
        )
        evaluation_actions = np.asarray([10, 20, 10, 20])
        evaluation_factual_targets = np.asarray([0, 0, 1, 1])
        report = evaluate_train_fitted_baselines(
            baselines,
            evaluation_targets,
            evaluation_actions,
            evaluation_factual_targets,
            ece_bins=2,
        )

        self.assertEqual(report.all_action.aggregate.count, 8)
        self.assertEqual(report.factual.count, 4)
        expected_factual = np.asarray([0.25, 0.75, 0.25, 0.75])
        expected_brier = np.mean(np.square(expected_factual - evaluation_factual_targets))
        self.assertAlmostEqual(report.factual.brier, float(expected_brier))

    def test_factual_gather_uses_semantic_ids_not_column_numbers(self) -> None:
        probabilities = np.arange(16, dtype=np.float64).reshape(8, 2) / 16.0
        actions = np.asarray([20, 10, 20, 10, 20, 10, 20, 10])

        targets, gathered = gather_factual_rows(
            self.train_all_targets,
            probabilities,
            actions,
            action_ids=self.action_ids,
        )

        expected_columns = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
        expected = probabilities[np.arange(8), expected_columns]
        self.assertTrue(np.array_equal(gathered, expected))
        self.assertTrue(
            np.array_equal(targets, self.train_all_targets[np.arange(8), expected_columns])
        )

    def test_factual_train_baseline_requires_every_registered_action(self) -> None:
        with self.assertRaisesRegex(QualificationMetricsError, "no samples for action 20"):
            fit_train_baselines(
                self.train_all_targets,
                np.asarray([10, 10, 10]),
                np.asarray([0, 1, 0]),
                action_ids=self.action_ids,
            )


class CrossEpisodeDerangementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cluster_ids = np.asarray(
            ["episode-0", "episode-0", "episode-1", "episode-1", "episode-2", "episode-2"]
        )

    def test_twenty_fixed_derangements_are_permutations_and_always_cross_episode(self) -> None:
        first = fixed_cross_episode_derangement_indices(
            self.cluster_ids,
            action_count=3,
        )
        second = fixed_cross_episode_derangement_indices(
            self.cluster_ids,
            action_count=3,
        )

        self.assertEqual(first.shape, (DEFAULT_DERANGEMENT_REPETITIONS, 3, 6))
        self.assertTrue(np.array_equal(first, second))
        expected_rows = np.arange(6)
        for repetition in range(first.shape[0]):
            for action in range(first.shape[1]):
                donors = first[repetition, action]
                self.assertTrue(np.array_equal(np.sort(donors), expected_rows))
                self.assertTrue(
                    np.all(self.cluster_ids[donors] != self.cluster_ids[expected_rows])
                )

    def test_deranged_probabilities_follow_each_actions_own_donor_indices(self) -> None:
        probabilities = np.column_stack(
            (np.arange(6, dtype=np.float64) / 10.0, np.arange(6, dtype=np.float64) / 20.0)
        )
        indices = fixed_cross_episode_derangement_indices(
            self.cluster_ids,
            action_count=2,
            repetitions=2,
            seed=17,
        )
        shuffled = fixed_cross_episode_derangements(
            probabilities,
            self.cluster_ids,
            repetitions=2,
            seed=17,
        )

        for repetition in range(2):
            for action in range(2):
                self.assertTrue(
                    np.array_equal(
                        shuffled[repetition, :, action],
                        probabilities[indices[repetition, action], action],
                    )
                )

    def test_derangement_report_exposes_real_and_all_null_repetitions(self) -> None:
        root_count = 40
        clusters = np.asarray([f"episode-{index // 2}" for index in range(root_count)])
        targets = np.column_stack(
            (
                np.arange(root_count) % 2,
                (np.arange(root_count) // 2) % 2,
            )
        ).astype(np.float64)
        probabilities = np.where(targets == 1.0, 0.9, 0.1)

        report = evaluate_cross_episode_derangements(
            targets,
            probabilities,
            clusters,
            action_ids=(4, 9),
        )

        self.assertEqual(report.repetitions, DEFAULT_DERANGEMENT_REPETITIONS)
        self.assertEqual(len(report.shuffled), DEFAULT_DERANGEMENT_REPETITIONS)
        self.assertEqual(report.real.aggregate.roc_auc, 1.0)
        self.assertGreater(report.aggregate_roc_auc_drop, 0.2)
        self.assertEqual(tuple(entry.action_id for entry in report.per_action), (4, 9))

    def test_impossible_cross_episode_matching_fails_closed(self) -> None:
        with self.assertRaisesRegex(QualificationMetricsError, "more than half"):
            fixed_cross_episode_derangement_indices(
                ["dominant", "dominant", "dominant", "other"],
                action_count=1,
            )


class ClusteredBootstrapTests(unittest.TestCase):
    def test_default_ten_thousand_resamples_are_deterministic_and_positive(self) -> None:
        targets = np.asarray(
            [
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            dtype=np.float64,
        )
        model = np.where(targets == 1.0, 0.9, 0.1)
        baseline = np.full_like(model, 0.5)
        clusters = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])

        first = clustered_improvement_bootstrap(targets, model, baseline, clusters)
        second = clustered_improvement_bootstrap(targets, model, baseline, clusters)

        self.assertEqual(first, second)
        self.assertEqual(first.resamples, DEFAULT_BOOTSTRAP_RESAMPLES)
        self.assertEqual(first.cluster_count, 4)
        self.assertEqual(first.observation_count, 16)
        self.assertGreater(first.observed_bce_improvement, 0.0)
        self.assertGreater(first.bce_improvement_lower_bound, 0.0)
        self.assertGreater(first.observed_brier_improvement, 0.0)
        self.assertGreater(first.brier_improvement_lower_bound, 0.0)
        self.assertEqual(len(first.replicate_sha256), 64)

    def test_cluster_resampling_keeps_variable_sized_episode_rows_together(self) -> None:
        targets = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float64)
        model = np.asarray([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
        baseline = np.full(6, 0.5)
        clusters = np.asarray(["short", "long", "long", "long", "tail", "tail"])

        result = clustered_improvement_bootstrap(
            targets,
            model,
            baseline,
            clusters,
            resamples=1_000,
            seed=99,
        )

        self.assertEqual(result.cluster_count, 3)
        self.assertEqual(result.observation_count, 6)
        self.assertGreater(result.bce_improvement_lower_bound, 0.0)

    def test_bootstrap_requires_at_least_two_clusters(self) -> None:
        with self.assertRaisesRegex(QualificationMetricsError, "at least two clusters"):
            clustered_improvement_bootstrap(
                [0, 1],
                [0.2, 0.8],
                [0.5, 0.5],
                ["same", "same"],
            )


if __name__ == "__main__":
    unittest.main()
