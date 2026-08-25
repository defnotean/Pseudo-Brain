"""Focused frozen-contract tests for PB21J-PROD-BZ-v1."""
from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for value in (SRC, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import v21j_fresh_bz_production_form_diagnostic_v1 as probe


def _tape(roots: int = 3) -> probe.FreshLiveTape:
    updater = np.arange(roots * 366, dtype=np.float64).reshape(roots, 366)
    updater[:, 360:365] = np.eye(5, dtype=np.float64)[np.arange(roots) % 5]
    updater[:, 365] = 1.0
    targets = np.asarray(
        [[(root + action) % 2 for action in range(5)] for root in range(roots)],
        dtype=np.float64,
    )
    return probe.FreshLiveTape(
        beliefs=np.arange(roots * 120, dtype=np.float64).reshape(roots, 120),
        updater_inputs=updater,
        hazard_targets=targets,
        parent_raw_logits=np.zeros((roots, 5), dtype=np.float64),
        factual_actions=np.arange(roots, dtype=np.int64) % 5,
        episode_group_ids=tuple(f"X:episode:{index}" for index in range(roots)),
        root_state_ids=tuple(sha256(f"root:{index}".encode()).hexdigest() for index in range(roots)),
        prior_assignment_count=np.arange(roots, dtype=np.int64),
        prior_disagreement_count=np.arange(roots, dtype=np.int64),
    )


def _candidate_cells(count: int = 9) -> dict[str, dict[str, object]]:
    return {
        f"c{index}": {
            "NZ": {
                "passed": True,
                "checks": {"CAL_accepted": True, "other_deployability": True},
            },
            "NBZ": {
                "passed": True,
                "checks": {"CAL_accepted": True, "other_deployability": True},
            },
        }
        for index in range(count)
    }


class FreshBZProductionFormTests(unittest.TestCase):
    def test_frozen_six_arms_and_candidate_paths(self) -> None:
        self.assertEqual(probe.ARMS, ("N", "Z", "BZ", "NZ", "NBZ", "U0"))
        self.assertEqual(probe.TOTAL_HEADS, 54)
        self.assertEqual(probe.TOTAL_OPTIMIZER_STEPS, 221_184)
        decision = probe.decision_contract()
        self.assertEqual(decision["fixed_minimality_preference"], ["NZ", "NBZ"])
        self.assertFalse(decision["selection_by_observed_AUC"])
        self.assertEqual(decision["BZ_minus_Z_role"], "mechanistic_confirmation_only_cannot_nominate")

    def test_exact_fresh_ranges_are_disjoint_and_cpu_qual_count_is_192(self) -> None:
        observed = [
            (value.label, value.split.value, value.seed_offset, value.seed_offset + value.episodes)
            for value in probe.PARTITION_SPECS
        ]
        self.assertEqual(
            observed,
            [
                ("F0", "train", 100_663_296, 100_663_552),
                ("C0", "train", 100_663_552, 100_663_680),
                ("F1", "train", 100_663_680, 100_663_936),
                ("C1", "train", 100_663_936, 100_664_064),
                ("F2", "train", 100_664_064, 100_664_320),
                ("C2", "train", 100_664_320, 100_664_448),
                ("DEV", "validation", 117_440_512, 117_440_896),
            ],
        )
        audit = probe.assert_no_range_collisions()
        self.assertTrue(audit["passed"])
        cpu = next(value for value in audit["records"] if value["owner"] == "V2.1 CPU-QUAL reserved")
        self.assertEqual(cpu["local_stop_exclusive"] - cpu["local_start"], 192)

    def test_range_audit_rejects_wallclock_collision(self) -> None:
        records = probe.occupied_range_registry()
        records.append(dict(records[4], owner="collision"))
        with self.assertRaisesRegex(RuntimeError, "collide"):
            probe.assert_no_range_collisions(records)

    def test_three_tick_feature_layout_is_exact_and_detached(self) -> None:
        tape = _tape(3)
        before_belief = tape.beliefs.copy()
        before_updater = tape.updater_inputs.copy()
        expected_components = {
            "N": (slice(0, 120), tape.beliefs),
            "B": (slice(120, 240), tape.updater_inputs[:, 0:120]),
            "Z": (slice(240, 360), tape.updater_inputs[:, 120:240]),
            "E": (slice(360, 480), tape.updater_inputs[:, 240:360]),
            "A": (slice(480, 485), tape.updater_inputs[:, 360:365]),
        }
        active_symbols = {
            "N": {"N"},
            "Z": {"Z"},
            "BZ": {"B", "Z"},
            "NZ": {"N", "Z"},
            "NBZ": {"N", "B", "Z"},
            "U0": {"B", "Z", "E", "A"},
        }
        for arm in probe.ARMS:
            actual = probe.superset_features(tape, arm)
            self.assertEqual(actual.shape, (3, 485))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            for symbol, (slot, values) in expected_components.items():
                if symbol in active_symbols[arm]:
                    np.testing.assert_array_equal(actual[:, slot], values.astype(np.float32))
                else:
                    np.testing.assert_array_equal(actual[:, slot], 0.0)
                    self.assertFalse(np.signbit(actual[:, slot]).any())
        np.testing.assert_array_equal(tape.beliefs, before_belief)
        np.testing.assert_array_equal(tape.updater_inputs, before_updater)

    def test_pending_is_exact_positive_one_then_structurally_omitted(self) -> None:
        tape = _tape()
        tape.updater_inputs[0, 365] = 0.0
        with self.assertRaisesRegex(RuntimeError, "pending flag"):
            probe.superset_features(tape, "NBZ")
        self.assertEqual(probe.SUPERSET_WIDTH, 485)
        self.assertEqual(probe.feature_contract()["pending_prediction_flag"], "structurally_omitted")

    def test_equal_training_head_topology_and_layer_norm_are_frozen(self) -> None:
        head = probe.initialized_head(probe.INIT_SEEDS[0])
        self.assertEqual(sum(value.numel() for value in head.parameters()), 87_961)
        output = head(torch.zeros(2, 485))
        self.assertEqual(tuple(output.shape), (2, 5))
        source = inspect.getsource(probe.MaskedSupersetHazardHead.forward)
        self.assertEqual(source.count("eps=1.0e-5"), 2)

    def test_arms_within_cell_start_byte_identical(self) -> None:
        heads = [probe.initialized_head(41_042) for _ in probe.ARMS]
        states = [head.state_dict() for head in heads]
        for left, right in zip(states[:-1], states[1:], strict=True):
            for x, y in zip(left.values(), right.values(), strict=True):
                self.assertTrue(torch.equal(x, y))

    def test_pruning_maps_exact_tensors_and_minimal_parameter_counts(self) -> None:
        padded = probe.initialized_head(41_042)
        features = probe.superset_features(_tape(), "NZ")
        for arm in probe.ARMS:
            pruned, report = probe.prune_head(padded, arm)
            self.assertTrue(report["mapped_tensors_byte_exact"])
            self.assertEqual(
                sum(value.numel() for value in pruned.parameters()),
                probe.PRUNED_PARAMETER_COUNTS[arm],
            )
            selected = probe.compact_features(features, arm)
            self.assertEqual(selected.shape[1], len(probe.ACTIVE_COLUMNS[arm]))

    def test_pruned_logits_numerically_match_padded_batch_one(self) -> None:
        padded = probe.initialized_head(42_042)
        for arm in probe.ARMS:
            features = probe.superset_features(_tape(), arm)
            pruned, _ = probe.prune_head(padded, arm)
            report = probe.pruning_equivalence_audit(padded, pruned, features, arm)
            self.assertTrue(report["passed"])
            self.assertLessEqual(report["maximum_absolute_difference"], 1.0e-5)

    def test_schedule_and_calibration_contract_are_frozen(self) -> None:
        schedule = probe.schedule_record()
        self.assertEqual(schedule["head_count"], 54)
        self.assertEqual(schedule["steps_per_head"], 4096)
        self.assertEqual(schedule["total_optimizer_steps"], 221184)
        self.assertTrue(schedule["all_fits_and_calibrators_complete_before_DEV_construction"])
        self.assertFalse(schedule["retry_allowed"])
        calibration = probe.calibration_contract()
        self.assertEqual(calibration["fit_mode"], "per_action_affine")
        self.assertTrue(calibration["protocol_change_from_v21i_bias_only"])
        self.assertEqual(calibration["tolerance"], 1.0e-10)

    def test_source_order_publishes_attempt_before_any_partition_source(self) -> None:
        source = inspect.getsource(probe.run)
        self.assertLess(source.index("_publish_attempt"), source.index("_run_impl"))
        implementation = inspect.getsource(probe._run_impl)
        fit = implementation.index("partition_sources(FIT_COHORTS)")
        cal = implementation.index("partition_sources(CAL_COHORTS)")
        dev = implementation.index('partition_sources(("DEV",))')
        self.assertLess(fit, cal)
        self.assertLess(cal, dev)
        self.assertIn("every fit, prune, and", implementation)

    def test_collector_records_current_action_after_prior_action_was_consumed(self) -> None:
        source = inspect.getsource(probe.collect_fresh_live_tape)
        reference = source.index("strict.strict_live_reference_step")
        record_current = source.index("factual_actions.append")
        install = source.index("strict._install_applied_pending")
        update_prior = source.index("prior_action = applied")
        self.assertLess(reference, record_current)
        self.assertLess(record_current, install)
        self.assertLess(install, update_prior)
        self.assertIn("batch_size=1", source)

    def test_bootstrap_matrix_is_shared_int32(self) -> None:
        left = probe.bootstrap_index_matrix(resamples=17, seed=62042, groups=9)
        right = probe.bootstrap_index_matrix(resamples=17, seed=62042, groups=9)
        self.assertEqual(left.dtype, np.int32)
        np.testing.assert_array_equal(left, right)

    def test_canonical_random_index_fixtures_are_exact(self) -> None:
        sampled = probe.bootstrap_index_matrix()
        self.assertEqual(sampled.shape, (50_000, 384))
        self.assertEqual(probe._array_sha256(sampled), probe.EXACT_BOOTSTRAP_INDEX_SHA256)
        np.testing.assert_array_equal(sampled[0, :8], [248, 87, 94, 283, 176, 19, 135, 10])
        np.testing.assert_array_equal(sampled[-1, -8:], [259, 61, 48, 237, 276, 305, 173, 36])

        episode = probe._joint_derangement_indices()
        self.assertEqual(episode.shape, (20, 384))
        self.assertEqual(
            probe._array_sha256(episode),
            probe.EXACT_DERANGEMENT_EPISODE_INDEX_SHA256,
        )
        identity = np.arange(384)
        for mapping in episode:
            self.assertFalse(np.any(mapping == identity))
            np.testing.assert_array_equal(np.sort(mapping), identity)
        self.assertEqual(len({row.tobytes() for row in episode}), 20)

    def test_expanded_derangement_is_cross_episode_bijective_and_same_ordinal(self) -> None:
        rows = tuple(
            np.arange(index * 12, (index + 1) * 12, dtype=np.int64)
            for index in range(384)
        )
        root = probe._root_derangement_indices(rows, probe._joint_derangement_indices())
        self.assertEqual(root.shape, (20, 4608))
        self.assertEqual(
            probe._array_sha256(root),
            probe.EXACT_DERANGEMENT_ROOT_INDEX_SHA256,
        )
        identity = np.arange(4608)
        for mapping in root:
            np.testing.assert_array_equal(np.sort(mapping), identity)
            self.assertTrue(np.all(mapping // 12 != identity // 12))
            np.testing.assert_array_equal(mapping % 12, identity % 12)

    def test_score_auc_uses_unsquashed_logits_and_preserves_ties(self) -> None:
        targets = np.asarray([0.0, 1.0])
        logits = np.asarray([40.0, 41.0])
        self.assertEqual(probe._score_auc(targets, logits), 1.0)
        saturated = torch.sigmoid(torch.from_numpy(logits)).numpy()
        np.testing.assert_array_equal(saturated, [1.0, 1.0])
        self.assertEqual(probe._score_auc(targets, saturated), 0.5)
        self.assertEqual(
            probe._score_auc(
                np.asarray([0.0, 1.0, 0.0, 1.0]),
                np.asarray([0.0, 0.0, 1.0, 1.0]),
            ),
            0.5,
        )

    def test_delete_one_auc_fails_closed_on_degenerate_class_sample(self) -> None:
        targets = np.asarray([0.0, 0.0, 1.0, 1.0])
        scores = np.asarray([0.0, 0.1, 0.9, 1.0])
        with self.assertRaisesRegex(RuntimeError, "degenerate"):
            probe._leave_one_auc(
                targets,
                scores,
                (np.asarray([0, 1]), np.asarray([2, 3])),
            )

    def test_proper_score_bootstrap_reports_point_025_as_point_975_confidence(self) -> None:
        targets = np.asarray([0.0, 1.0, 0.0, 1.0])
        model = np.asarray([0.1, 0.9, 0.2, 0.8])
        baseline = np.full(4, 0.5)
        rows = tuple(np.asarray([index]) for index in range(4))
        sampled = np.asarray([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=np.int32)
        report = probe.decomposable_improvement_bootstrap(
            targets, model, baseline, rows, sampled
        )
        self.assertEqual(report["one_sided_alpha"], 0.025)
        self.assertEqual(report["one_sided_confidence_level"], 0.975)
        self.assertIn("BCE_one_sided_lower_bound", report)
        self.assertNotIn("BCE_one_sided_95_lower", report)
        clipped_model = np.clip(model, 1.0e-12, 1.0 - 1.0e-12)
        bce_model = -(
            targets * np.log(clipped_model)
            + (1.0 - targets) * np.log1p(-clipped_model)
        )
        bce_baseline = np.full(4, -np.log(0.5))
        episode_improvement = bce_baseline - bce_model
        manual_draws = episode_improvement[sampled].mean(axis=1)
        self.assertEqual(
            report["BCE_one_sided_lower_bound"],
            float(np.quantile(manual_draws, 0.025, method="linear")),
        )

    def test_pseudovalue_fixture_and_strict_point_guards(self) -> None:
        targets = np.tile(np.asarray([0.0, 1.0]), 4)
        groups = tuple(
            group for index in range(4) for group in (f"episode:{index}",) * 2
        )
        sampled = np.asarray(
            [[0, 1, 2, 3], [0, 0, 1, 1], [2, 2, 3, 3]],
            dtype=np.int32,
        )
        cells = {
            f"c{index}": {
                "NZ": targets.copy(),
                "N": np.zeros_like(targets),
            }
            for index in range(9)
        }
        comparison = (("NZ_minus_N", "NZ", "N", 0.0, 0.025, "NZ"),)
        with (
            patch.object(probe, "DEV_EPISODES", 4),
            patch.object(probe, "ROOTS_PER_EPISODE", 2),
            patch.object(probe, "PRIMARY_COMPARISONS", comparison),
        ):
            report = probe.primary_iut_report(targets, cells, groups, sampled)
        value = report["comparisons"]["NZ_minus_N"]
        self.assertEqual(value["grand_mean_full_delta"], 0.5)
        self.assertEqual(value["one_sided_lower_bound"], 0.5)
        self.assertEqual(value["one_sided_confidence_level"], 0.975)
        self.assertTrue(value["every_cell_point_guard"])
        self.assertTrue(value["passed"])

        equality = (("NZ_minus_N", "NZ", "N", 0.5, 0.025, "NZ"),)
        with (
            patch.object(probe, "DEV_EPISODES", 4),
            patch.object(probe, "ROOTS_PER_EPISODE", 2),
            patch.object(probe, "PRIMARY_COMPARISONS", equality),
        ):
            report = probe.primary_iut_report(targets, cells, groups, sampled)
        self.assertFalse(
            report["comparisons"]["NZ_minus_N"]["every_cell_point_guard"]
        )
        self.assertFalse(report["comparisons"]["NZ_minus_N"]["passed"])

        cells["c0"]["NZ"] = np.zeros_like(targets)
        with (
            patch.object(probe, "DEV_EPISODES", 4),
            patch.object(probe, "ROOTS_PER_EPISODE", 2),
            patch.object(probe, "PRIMARY_COMPARISONS", comparison),
        ):
            report = probe.primary_iut_report(targets, cells, groups, sampled)
        self.assertFalse(
            report["comparisons"]["NZ_minus_N"]["every_cell_point_guard"]
        )
        self.assertFalse(report["comparisons"]["NZ_minus_N"]["passed"])

    def test_fixed_minimality_selection_does_not_use_auc(self) -> None:
        cells = {f"c{i}": {arm: {"passed": True} for arm in ("Z", "BZ", "NZ", "NBZ", "U0")} for i in range(9)}
        candidates = _candidate_cells()
        inference = {"path_passed": {"NZ": True, "NBZ": True, "mechanistic": False}}
        result = probe.select_recipe(cells, candidates, inference)
        self.assertEqual(result["selected_arm"], "NZ")
        self.assertFalse(result["selection_by_observed_AUC"])
        for cell in candidates.values():
            cell["NZ"]["passed"] = False
        result = probe.select_recipe(cells, candidates, inference)
        self.assertEqual(result["selected_arm"], "NBZ")

    def test_path_specific_failures_do_not_block_the_other_path(self) -> None:
        cells = {
            f"c{i}": {
                arm: {"passed": True}
                for arm in ("Z", "BZ", "NZ", "NBZ", "U0")
            }
            for i in range(9)
        }
        candidates = _candidate_cells()
        inference = {
            "path_passed": {"NZ": True, "NBZ": True, "mechanistic": False}
        }
        cells["c0"]["NBZ"]["passed"] = False
        candidates["c0"]["NBZ"]["passed"] = False
        candidates["c0"]["NBZ"]["checks"]["CAL_accepted"] = False
        result = probe.select_recipe(cells, candidates, inference)
        self.assertEqual(result["selected_arm"], "NZ")
        self.assertTrue(result["eligible"]["NZ"])
        self.assertFalse(result["eligible"]["NBZ"])

        cells["c0"]["NBZ"]["passed"] = True
        candidates["c0"]["NBZ"]["passed"] = True
        candidates["c0"]["NBZ"]["checks"]["CAL_accepted"] = True
        candidates["c0"]["NZ"]["passed"] = False
        candidates["c0"]["NZ"]["checks"]["other_deployability"] = False
        result = probe.select_recipe(cells, candidates, inference)
        self.assertEqual(result["selected_arm"], "NBZ")

    def test_diagnosis_distinguishes_calibration_from_other_deployability(self) -> None:
        cells = {
            f"c{i}": {
                arm: {"passed": True}
                for arm in ("Z", "BZ", "NZ", "NBZ", "U0")
            }
            for i in range(9)
        }
        candidates = _candidate_cells()
        inference = {
            "path_passed": {"NZ": True, "NBZ": False, "mechanistic": False}
        }
        candidates["c0"]["NZ"]["passed"] = False
        candidates["c0"]["NZ"]["checks"]["other_deployability"] = False
        result = probe.select_recipe(cells, candidates, inference)
        self.assertEqual(
            result["diagnosis"],
            "representation_supported_deployability_unresolved",
        )
        candidates["c0"]["NZ"]["checks"]["CAL_accepted"] = False
        self.assertEqual(
            probe.select_recipe(cells, candidates, inference)["diagnosis"],
            "representation_supported_calibration_unresolved",
        )

    def test_real_parent_and_registration_payload_validate_before_registration(self) -> None:
        root = Path(probe.__file__).resolve().parents[2]
        parent = probe.validate_component_parent(root)
        self.assertEqual(
            parent["BZ_minus_U0_registered_lower"],
            -0.027505939925940276,
        )
        payload = probe.registration_payload(root)
        self.assertEqual(len(payload["partitions"]), 7)
        self.assertTrue(payload["occupied_range_audit"]["passed"])
        self.assertEqual(
            payload["component_parent"]["result_sha256"],
            probe.EXACT_COMPONENT_RESULT_SHA256,
        )

    def test_partition_source_rejects_observed_manifest_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "manifest"):
            probe.partition_sources(
                ("F0",),
                source_factory=lambda contract: SimpleNamespace(
                    contract=contract,
                    manifest_sha256="0" * 64,
                ),
            )

    def test_calibration_logits_are_frozen_at_deployment_batch_one(self) -> None:
        self.assertEqual(
            probe.calibration_contract()["scorer_prediction_batch_size"],
            1,
        )
        source = inspect.getsource(probe.fit_calibrator)
        self.assertIn("batch_size=1", source)

    def test_upstream_preflight_failure_cannot_consume_attempt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "upstream_result": root / "upstream.json",
                "registration": root / "registration.json",
                "result": root / "result.json",
                "attempt": root / "attempt.json",
            }
            paths["registration"].write_text("{}", encoding="utf-8")
            record = {
                "path": str(paths["registration"]),
                "sha256": "a" * 64,
                "payload": {"source_bundle": {"sha256": "b" * 64}},
            }
            with (
                patch.object(probe, "canonical_paths", return_value=paths),
                patch.object(probe, "validate_registration", return_value=record),
                patch.object(
                    probe.strict,
                    "load_exact_parent",
                    side_effect=ValueError("drifted parent"),
                ),
                patch.object(probe, "_publish_attempt") as publish,
            ):
                with self.assertRaisesRegex(ValueError, "drifted parent"):
                    probe.run(
                        upstream_result=paths["upstream_result"],
                        output=paths["result"],
                        registration=paths["registration"],
                    )
                publish.assert_not_called()
                self.assertFalse(paths["attempt"].exists())
                self.assertFalse(paths["result"].exists())

    def test_attempt_is_irreversible_and_caught_failure_is_recorded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "upstream_result": root / "upstream.json",
                "registration": root / "registration.json",
                "result": root / "result.json",
                "attempt": root / "attempt.json",
            }
            paths["registration"].write_text("{}", encoding="utf-8")
            record = {
                "path": str(paths["registration"]),
                "sha256": "a" * 64,
                "payload": {"source_bundle": {"sha256": "b" * 64}},
            }
            with (
                patch.object(probe, "canonical_paths", return_value=paths),
                patch.object(probe, "validate_registration", return_value=record),
                patch.object(
                    probe.strict,
                    "load_exact_parent",
                    return_value=(object(), {}, {}),
                ),
                patch.object(
                    probe,
                    "_run_impl",
                    side_effect=RuntimeError("post-attempt failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "post-attempt failure"):
                    probe.run(
                        upstream_result=paths["upstream_result"],
                        output=paths["result"],
                        registration=paths["registration"],
                    )
            self.assertTrue(paths["attempt"].is_file())
            failure = json.loads(paths["result"].read_text(encoding="utf-8"))
            self.assertEqual(
                failure["classification"],
                "diagnostic_run_failed_not_candidate",
            )
            self.assertFalse(failure["retry_allowed"])
            with patch.object(probe, "canonical_paths", return_value=paths):
                with self.assertRaisesRegex(FileExistsError, "retry"):
                    probe.run(
                        upstream_result=paths["upstream_result"],
                        output=paths["result"],
                        registration=paths["registration"],
                    )

    def test_primary_comparison_multiplicity_is_frozen(self) -> None:
        contract = probe.inference_contract()
        candidate = [value for value in contract["comparisons"] if value["path"] != "mechanistic"]
        mechanistic = [value for value in contract["comparisons"] if value["path"] == "mechanistic"]
        self.assertEqual(len(candidate), 6)
        self.assertTrue(all(value["one_sided_alpha"] == 0.025 for value in candidate))
        self.assertEqual([value["name"] for value in mechanistic], ["BZ_minus_Z"])
        self.assertEqual(mechanistic[0]["one_sided_alpha"], 0.05)

    def test_main_result_never_emits_checkpoint_or_candidate(self) -> None:
        source = inspect.getsource(probe._run_impl)
        self.assertIn('"candidate_publication_allowed": False', source)
        self.assertIn('"checkpoint_emitted": False', source)
        self.assertIn('"compact_head_trained_from_initialization_equivalence_claimed": False', source)


if __name__ == "__main__":
    unittest.main()
