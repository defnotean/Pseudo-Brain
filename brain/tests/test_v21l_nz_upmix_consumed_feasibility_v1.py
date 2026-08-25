"""Fast contracts for PB21L consumed-data NZ scorer-upmix feasibility."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "v21l_nz_upmix_consumed_feasibility_v1.py"
)
spec = importlib.util.spec_from_file_location("pb21l_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
pb21l = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21l
spec.loader.exec_module(pb21l)


class PB21LNZUpmixTests(unittest.TestCase):
    def test_permanent_nonqualifying_contract_and_parent_hashes(self) -> None:
        self.assertEqual(pb21l.CLASSIFICATION, "exploratory_consumed_data_nonqualifying")
        self.assertEqual(pb21l.IMPLEMENTATION_REVISION, 2)
        self.assertEqual(pb21l.SCORERS, ("BASE", "POOL-UPMIX", "BAL-UPMIX"))
        self.assertEqual(pb21l.FIT_COHORTS, ("F0", "F2"))
        self.assertEqual(pb21l.CAL_COHORTS, ("C0", "C2"))
        self.assertEqual(pb21l.INIT_SEEDS, (51_042, 53_042))
        self.assertEqual(
            pb21l.CELLS,
            (
                "F0/init-51042",
                "F0/init-53042",
                "F2/init-51042",
                "F2/init-53042",
            ),
        )
        self.assertEqual(
            pb21l.EXACT_PB21K_RESULT_SHA256,
            "083ac9c44d3327f4f806b1e4842ea3991ec54e5bab65992e7c5446ad09ca02d4",
        )
        self.assertEqual(
            pb21l.EXACT_PB21K_EVIDENCE_SHA256,
            "9d57bf4a6bd60121e8f11e2b9405c1a6d64a9346381503054d6ad9cd4ece385c",
        )
        decision = pb21l.decision_contract()
        self.assertIs(decision["nomination_allowed"], False)
        self.assertIs(decision["checkpoint_emitted"], False)
        self.assertIs(decision["candidate_publication_allowed"], False)
        self.assertIs(decision["qualification_claimed"], False)
        self.assertIs(decision["same_consumed_DEV_retry_allowed"], False)

    def test_exact_evidence_schema_binds_redteam_replay_fields(self) -> None:
        contract = pb21l.expected_evidence_contract()
        self.assertEqual(set(contract), pb21l.EVIDENCE_KEYS)
        self.assertEqual(
            contract["training_initial_state_sha256"],
            {"shape": [3, 4], "dtype": "|S64"},
        )
        self.assertEqual(
            contract["training_pass_permutation_sha256"]["shape"], [3, 4, 32]
        )
        self.assertEqual(contract["training_pass_number"]["dtype"], "<i2")
        self.assertEqual(contract["pruning_mapping_sha256"]["shape"], [3, 4])
        self.assertEqual(
            contract["factual_balance_weights_consumed_float32"]["dtype"], "<f4"
        )
        self.assertEqual(contract["fit_group_ordered_sha256"]["shape"], [2])
        self.assertEqual(contract["cal_group_ordered_sha256"]["shape"], [2])
        self.assertEqual(contract["fit_nz_feature_sha256"]["shape"], [2])
        self.assertEqual(contract["calibration_source_namespace"]["shape"], [3, 4])
        self.assertEqual(contract["pruning_active_columns"]["shape"], [240])
        self.assertEqual(contract["calibration_partition_algorithm"]["dtype"], "|S512")
        self.assertEqual(
            contract["bootstrap_indices"],
            {"shape": [50_000, 768], "dtype": "<i4"},
        )

    def test_factual_balance_weights_are_exact_full_fit_estimator(self) -> None:
        actions = np.concatenate(
            (
                np.full(400, 0),
                np.full(500, 1),
                np.full(600, 2),
                np.full(700, 3),
                np.full(872, 4),
            )
        ).astype(np.int64)
        counts = pb21l.factual_action_counts(actions)
        weights = pb21l.factual_balance_weights(actions)
        consumed = pb21l.consumed_factual_balance_weights(actions)
        self.assertEqual(counts.tolist(), [400, 500, 600, 700, 872])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=14)
        self.assertEqual(consumed.dtype, np.dtype("<f4"))
        self.assertTrue(np.array_equal(consumed, weights.astype(np.float32)))
        for action in range(5):
            self.assertAlmostEqual(
                float(weights[actions == action].sum()),
                pb21l.ROOTS_PER_FIT / 5.0,
                places=11,
            )

    def test_exact_cell_axes_reject_valid_shaped_substitution(self) -> None:
        arrays = {
            "scorer_ids": pb21l._byte_strings(pb21l.SCORERS, 12),
            "cell_ids": pb21l._byte_strings(pb21l.CELLS, 16),
            "cell_fit_index": np.asarray((0, 0, 1, 1), dtype=np.int16),
            "cell_init_seed": np.asarray(
                (51_042, 53_042, 51_042, 53_042), dtype=np.int32
            ),
        }
        pb21l.validate_evidence_axes(arrays)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["cell_fit_index"][:] = (0, 1, 0, 1)
        with self.assertRaisesRegex(ValueError, "axes"):
            pb21l.validate_evidence_axes(changed)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["cell_init_seed"][:] = (53_042, 51_042, 53_042, 51_042)
        with self.assertRaisesRegex(ValueError, "axes"):
            pb21l.validate_evidence_axes(changed)

    def test_publication_receipt_recomputes_full_envelope(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.bin"
            path.write_bytes(b"immutable-evidence")
            arrays = {"x": np.arange(4, dtype=np.int32)}
            manifest = pb21l._array_manifest(arrays)
            with patch.object(pb21l, "EVIDENCE_KEYS", frozenset({"x"})):
                publication = {
                    "sha256": pb21l._sha256_file(path),
                    "byte_length": path.stat().st_size,
                    "allowed_key_set_sha256": pb21l.evidence_key_set_sha256(),
                    "arrays": manifest,
                    "array_manifest_sha256": pb21l._manifest_sha256(manifest),
                }
                pb21l.validate_publication_receipt(path, publication, arrays)
                for field, bad_value in (
                    ("byte_length", path.stat().st_size + 1),
                    ("allowed_key_set_sha256", "0" * 64),
                    ("array_manifest_sha256", "0" * 64),
                ):
                    with self.subTest(field=field):
                        changed = dict(publication)
                        changed[field] = bad_value
                        with self.assertRaisesRegex(ValueError, "envelope"):
                            pb21l.validate_publication_receipt(path, changed, arrays)
                changed = dict(publication)
                changed["arrays"] = {
                    "x": {**manifest["x"], "sha256": "0" * 64}
                }
                with self.assertRaisesRegex(ValueError, "manifest envelope"):
                    pb21l.validate_publication_receipt(path, changed, arrays)

    def test_three_upmix_objectives_and_loss_only_selector(self) -> None:
        logits = torch.tensor(
            [[-1.0, 0.1, 0.2, 0.3, 0.4], [0.5, -0.6, 0.7, -0.8, 0.9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        actions = torch.tensor([1, 4], dtype=torch.long)
        weights = torch.tensor([0.5, 1.5], dtype=torch.float32)
        all_loss = F.binary_cross_entropy_with_logits(logits, targets)
        factual_terms = F.binary_cross_entropy_with_logits(
            logits[torch.arange(2), actions],
            targets[torch.arange(2), actions],
            reduction="none",
        )
        base, _ = pb21l.upmix_loss(logits, targets, actions, scorer=pb21l.SCORER_BASE)
        pool, _ = pb21l.upmix_loss(logits, targets, actions, scorer=pb21l.SCORER_POOL)
        bal, _ = pb21l.upmix_loss(
            logits,
            targets,
            actions,
            scorer=pb21l.SCORER_BAL,
            balanced_weights=weights,
        )
        self.assertTrue(torch.equal(base, all_loss))
        self.assertTrue(
            torch.allclose(pool, 0.5 * all_loss + 0.5 * factual_terms.mean())
        )
        self.assertTrue(
            torch.allclose(
                bal,
                0.5 * all_loss + 0.5 * torch.mean(factual_terms * weights),
            )
        )
        bal.backward()
        self.assertIsNotNone(logits.grad)

    def test_balanced_and_pooled_factual_terms_diverge_on_imbalanced_fit(self) -> None:
        actions = np.concatenate(
            (
                np.full(400, 0),
                np.full(500, 1),
                np.full(600, 2),
                np.full(700, 3),
                np.full(872, 4),
            )
        ).astype(np.int64)
        logits = torch.zeros((pb21l.ROOTS_PER_FIT, 5), dtype=torch.float32)
        targets = torch.zeros_like(logits)
        action_tensor = torch.from_numpy(actions)
        for action in range(5):
            rows = torch.from_numpy(np.flatnonzero(actions == action))
            logits[rows, action] = float(action) * 0.5
        weights = torch.from_numpy(pb21l.factual_balance_weights(actions)).to(torch.float32)
        _, pooled = pb21l.upmix_loss(
            logits, targets, action_tensor, scorer=pb21l.SCORER_POOL
        )
        _, balanced = pb21l.upmix_loss(
            logits,
            targets,
            action_tensor,
            scorer=pb21l.SCORER_BAL,
            balanced_weights=weights,
        )
        selected = F.binary_cross_entropy_with_logits(
            logits[torch.arange(len(logits)), action_tensor],
            targets[torch.arange(len(logits)), action_tensor],
            reduction="none",
        )
        expected_balanced = torch.stack(
            [selected[action_tensor == action].mean() for action in range(5)]
        ).mean()
        self.assertAlmostEqual(
            balanced["factual_component_BCE"], float(expected_balanced), places=6
        )
        self.assertNotAlmostEqual(
            pooled["factual_component_BCE"],
            balanced["factual_component_BCE"],
            places=4,
        )

    def test_base_training_delegates_to_exact_pb21j_recipe(self) -> None:
        sentinel = {"initial_state_sha256": "i", "final_state_sha256": "f"}
        called: dict[str, object] = {}

        def fake_train(head, features, targets, permutations):
            called.update(
                head=head,
                features=features,
                targets=targets,
                permutations=permutations,
            )
            return sentinel

        head = object()
        features = np.zeros((1, 1), dtype=np.float32)
        targets = np.zeros((1, 1), dtype=np.float64)
        permutations = (torch.tensor([0]),)
        with patch.object(pb21l.pb21j, "train_head", fake_train):
            report = pb21l.train_upmix_head(
                head,
                features,
                targets,
                np.zeros(1, dtype=np.int64),
                permutations,
                scorer=pb21l.SCORER_BASE,
            )
        self.assertIs(called["head"], head)
        self.assertIs(called["features"], features)
        self.assertIs(called["targets"], targets)
        self.assertIs(called["permutations"], permutations)
        self.assertEqual(report["initial_state_sha256"], "i")
        self.assertEqual(report["objective"], pb21l.objective_contract()[pb21l.SCORER_BASE])

    def test_additive_domain_partition_identity(self) -> None:
        rng = np.random.default_rng(8)
        targets = rng.integers(0, 2, size=(20, 5)).astype(np.float64)
        probabilities = rng.uniform(0.05, 0.95, size=(20, 5))
        actions = np.tile(np.arange(5, dtype=np.int64), 4)
        report = pb21l.mask_partition_identity(targets, probabilities, actions)
        self.assertIs(report["passed"], True)
        self.assertLessEqual(report["maximum_BCE_absolute_error"], 1.0e-12)
        self.assertLessEqual(report["maximum_Brier_absolute_error"], 1.0e-12)

    def test_ordered_source_lineage_and_fit_feature_tamper_rejection(self) -> None:
        fit = (("F0:e0", "F0:e1"), ("F2:e0", "F2:e1"))
        cal = (("C0:e0", "C0:e1"), ("C2:e0", "C2:e1"))
        fit_digest = tuple(pb21l.strict._ordered_sequence_digest(row) for row in fit)
        cal_digest = tuple(pb21l.strict._ordered_sequence_digest(row) for row in cal)
        feature_digest = ("1" * 64, "2" * 64)
        lineage = pb21l.ExpectedSourceLineage(
            fit_group_ids=fit,
            cal_group_ids=cal,
            fit_group_ordered_sha256=fit_digest,
            cal_group_ordered_sha256=cal_digest,
            fit_nz_feature_sha256=feature_digest,
        )
        arrays = {
            "fit_group_ids": np.stack([pb21l._byte_strings(row, 64) for row in fit]),
            "cal_group_ids": np.stack([pb21l._byte_strings(row, 64) for row in cal]),
            "fit_group_ordered_sha256": pb21l._byte_strings(fit_digest, 64),
            "cal_group_ordered_sha256": pb21l._byte_strings(cal_digest, 64),
            "fit_nz_feature_sha256": pb21l._byte_strings(feature_digest, 64),
        }
        pb21l.validate_source_lineage_arrays(arrays, lineage)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["fit_group_ids"][0] = changed["fit_group_ids"][0, ::-1]
        with self.assertRaisesRegex(ValueError, "source lineage"):
            pb21l.validate_source_lineage_arrays(changed, lineage)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["cal_group_ordered_sha256"][0] = b"0" * 64
        with self.assertRaisesRegex(ValueError, "source lineage"):
            pb21l.validate_source_lineage_arrays(changed, lineage)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["fit_nz_feature_sha256"][0] = b"0" * 64
        with self.assertRaisesRegex(ValueError, "NZ features"):
            pb21l.validate_source_lineage_arrays(changed, lineage)

    def test_lineage_snapshot_is_bound_to_prepublication_source_evidence(self) -> None:
        class Tape:
            def __init__(self, groups):
                self.episode_group_ids = groups

        fit_groups = {"F0": ("F0:a", "F0:b"), "F2": ("F2:a", "F2:b")}
        cal_groups = {"C0": ("C0:a", "C0:b"), "C2": ("C2:a", "C2:b")}
        features = {
            "F0": np.arange(6, dtype=np.float32).reshape(2, 3),
            "F2": np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        }
        source_evidence = {
            label: {
                "ordered_episode_sha256": pb21l.strict._ordered_sequence_digest(groups)
            }
            for label, groups in {**fit_groups, **cal_groups}.items()
        }
        for label in pb21l.FIT_COHORTS:
            source_evidence[label]["NZ_feature_sha256"] = pb21l._array_sha256(
                features[label]
            )
        with patch.object(pb21l, "ROOTS_PER_FIT", 2), patch.object(
            pb21l, "ROOTS_PER_CAL", 2
        ):
            lineage = pb21l.build_expected_source_lineage(
                fit_tapes={label: Tape(groups) for label, groups in fit_groups.items()},
                cal_tapes={label: Tape(groups) for label, groups in cal_groups.items()},
                fit_features=features,
                source_evidence=source_evidence,
            )
            self.assertEqual(lineage.fit_group_ids[0], fit_groups["F0"])
            changed = {
                label: dict(evidence) for label, evidence in source_evidence.items()
            }
            changed["C0"]["ordered_episode_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "prepublication CAL"):
                pb21l.build_expected_source_lineage(
                    fit_tapes={
                        label: Tape(groups) for label, groups in fit_groups.items()
                    },
                    cal_tapes={
                        label: Tape(groups) for label, groups in cal_groups.items()
                    },
                    fit_features=features,
                    source_evidence=changed,
                )

    def test_factual_perturbation_does_not_change_complement_losses(self) -> None:
        rng = np.random.default_rng(19)
        targets = rng.integers(0, 2, size=(25, 5)).astype(np.float64)
        actions = np.tile(np.arange(5, dtype=np.int64), 5)
        base = np.full((25, 5), 0.5, dtype=np.float64)
        changed = base.copy()
        rows = np.arange(len(actions))
        changed[rows, actions] = np.where(targets[rows, actions] == 1.0, 0.9, 0.1)
        base_factual = pb21l._per_root_domain_losses(
            targets, base, actions, "factual"
        )
        changed_factual = pb21l._per_root_domain_losses(
            targets, changed, actions, "factual"
        )
        base_complement = pb21l._per_root_domain_losses(
            targets, base, actions, "nonselected_complement"
        )
        changed_complement = pb21l._per_root_domain_losses(
            targets, changed, actions, "nonselected_complement"
        )
        self.assertFalse(np.array_equal(base_factual, changed_factual))
        self.assertTrue(np.array_equal(base_complement, changed_complement))
        complement_changed = base.copy()
        mask = np.arange(5)[None, :] != actions[:, None]
        complement_changed[mask] = np.where(targets[mask] == 1.0, 0.8, 0.2)
        reciprocal_factual = pb21l._per_root_domain_losses(
            targets, complement_changed, actions, "factual"
        )
        reciprocal_complement = pb21l._per_root_domain_losses(
            targets, complement_changed, actions, "nonselected_complement"
        )
        self.assertTrue(np.array_equal(base_factual, reciprocal_factual))
        self.assertFalse(np.array_equal(base_complement, reciprocal_complement))

    def test_domain_specific_bias_routing(self) -> None:
        class Metrics:
            def as_dict(self):
                return {
                    "ece_equal_mass": 0.01,
                    "calibration_bias": 0.06,
                    "pr_auc": 0.8,
                    "prevalence": 0.2,
                    "brier": 0.1,
                    "bce": 0.3,
                }

        actions = np.tile(np.arange(5, dtype=np.int64), 2)
        rows = np.arange(10, dtype=np.int64)[:, None]
        cols = np.arange(5, dtype=np.int64)[None, :]
        targets = ((rows + cols) % 2).astype(np.float64)
        probabilities = np.full((10, 5), 0.5, dtype=np.float64)
        priors = np.full(5, 0.5, dtype=np.float64)
        with patch.object(
            pb21l, "binary_probability_metrics", lambda *args, **kwargs: Metrics()
        ):
            factual = pb21l._domain_metric_report(
                targets, probabilities, actions, domain="factual", priors=priors
            )
            complement = pb21l._domain_metric_report(
                targets,
                probabilities,
                actions,
                domain="nonselected_complement",
                priors=priors,
            )
            all_action = pb21l._domain_metric_report(
                targets, probabilities, actions, domain="all_action", priors=priors
            )
        self.assertIs(factual["per_action"][0]["checks"]["absolute_bias"], True)
        self.assertEqual(factual["per_action"][0]["maximum_absolute_bias"], 0.075)
        self.assertNotIn("PR_prevalence_gain", factual["per_action"][0]["checks"])
        self.assertIs(complement["per_action"][0]["checks"]["absolute_bias"], False)
        self.assertEqual(complement["per_action"][0]["maximum_absolute_bias"], 0.05)
        self.assertIn("PR_prevalence_gain", complement["per_action"][0]["checks"])
        self.assertIs(all_action["per_action"][0]["checks"]["absolute_bias"], False)
        self.assertNotIn("both_classes", all_action["per_action"][0]["checks"])
        self.assertIs(factual["aggregate_checks"]["absolute_bias"], False)

    def test_registered_aa_provenance_rejects_self_supplied_fields(self) -> None:
        shape = (3, 4)
        checkpoint = np.asarray(
            [[f"{row + 1:x}" * 64 for _ in range(4)] for row in range(3)],
            dtype=object,
        )
        manifests = [f"{value:x}" * 64 for value in (4, 5, 6, 7)]
        bundles = [[f"{value:x}" * 64] * 4 for value in (8, 9, 10)]
        registered = {
            "source_namespace": pb21l.AA_SOURCE_NAMESPACE,
            "source_split": pb21l.AA_SOURCE_SPLIT,
            "source_partition": pb21l.AA_SOURCE_PARTITION,
            "partition_algorithm": "fixed-partition-v1",
            "dataset_manifest_sha256_by_cell": manifests,
            "source_bundle_sha256_by_scorer_and_cell": bundles,
        }
        arrays = {
            "calibration_source_namespace": np.full(
                shape, registered["source_namespace"], dtype="S64"
            ),
            "calibration_source_split": np.full(
                shape, registered["source_split"], dtype="S16"
            ),
            "calibration_source_partition": np.full(
                shape, registered["source_partition"], dtype="S16"
            ),
            "calibration_partition_algorithm": np.full(
                shape, registered["partition_algorithm"], dtype="S512"
            ),
            "calibration_dataset_manifest_sha256": np.asarray(
                [manifests] * 3, dtype="S64"
            ),
            "calibration_source_bundle_sha256": np.asarray(bundles, dtype="S64"),
            "calibration_upstream_checkpoint_sha256": np.asarray(
                checkpoint, dtype="S64"
            ),
        }
        pb21l.validate_aa_provenance_arrays(arrays, registered, checkpoint)
        tamper_cases = {
            "calibration_source_namespace": b"unregistered",
            "calibration_source_split": b"DEV",
            "calibration_source_partition": b"TRAIN-FIT",
            "calibration_partition_algorithm": b"other",
            "calibration_dataset_manifest_sha256": b"0" * 64,
            "calibration_source_bundle_sha256": b"0" * 64,
            "calibration_upstream_checkpoint_sha256": b"0" * 64,
        }
        for field, replacement in tamper_cases.items():
            with self.subTest(field=field):
                changed = {name: value.copy() for name, value in arrays.items()}
                changed[field][0, 0] = replacement
                with self.assertRaisesRegex(ValueError, "registered/frozen"):
                    pb21l.validate_aa_provenance_arrays(
                        changed, registered, checkpoint
                    )

    def test_consumed_float32_balance_weight_tamper_rejection(self) -> None:
        actions = np.concatenate(
            (
                np.full(400, 0),
                np.full(500, 1),
                np.full(600, 2),
                np.full(700, 3),
                np.full(872, 4),
            )
        ).astype(np.int64)
        fit_actions = np.stack((actions, np.roll(actions, 17)))
        cell_fit_index = np.asarray((0, 0, 1, 1), dtype=np.int16)
        theoretical = np.stack(
            [pb21l.factual_balance_weights(fit_actions[index])
             for index in cell_fit_index]
        )
        arrays = {
            "fit_actions": fit_actions,
            "cell_fit_index": cell_fit_index,
            "factual_action_counts": np.stack(
                [pb21l.factual_action_counts(fit_actions[index])
                 for index in cell_fit_index]
            ),
            "factual_balance_weights": theoretical,
            "factual_balance_weights_consumed_float32": theoretical.astype(np.float32),
        }
        pb21l.validate_factual_weight_arrays(arrays)
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["factual_balance_weights_consumed_float32"][0, 0] = np.nextafter(
            changed["factual_balance_weights_consumed_float32"][0, 0],
            np.float32(np.inf),
        )
        with self.assertRaisesRegex(ValueError, "consumed weight"):
            pb21l.validate_factual_weight_arrays(changed)

    def test_fixed_precedence_truth_table(self) -> None:
        cases = (
            (False, False, None),
            (True, False, "POOL-UPMIX"),
            (False, True, "BAL-UPMIX"),
            (True, True, "BAL-UPMIX"),
        )
        for pool, bal, expected in cases:
            with self.subTest(pool=pool, bal=bal):
                self.assertEqual(
                    pb21l.select_fresh_license(
                        {pb21l.SCORER_POOL: pool, pb21l.SCORER_BAL: bal}
                    ),
                    expected,
                )

    def test_direct_comparison_sign_strict_threshold_and_bad_cell(self) -> None:
        arrays = {
            "dev_targets": np.zeros((2, 5), dtype=np.float64),
            "dev_actions": np.asarray([0, 1], dtype=np.int64),
            "dev_cluster_ordinal": np.asarray([0, 1], dtype=np.int32),
            "bootstrap_indices": np.zeros((5, 2), dtype=np.int32),
            "dev_calibrated_probabilities": np.zeros((3, 4, 2, 5), dtype=np.float64),
        }
        for scorer in range(3):
            for cell in range(4):
                arrays["dev_calibrated_probabilities"][scorer, cell, 0, 0] = (
                    10 * scorer + cell
                )

        def fake_losses(targets, probabilities, actions, domain):
            marker = int(probabilities[0, 0])
            scorer, cell = divmod(marker, 10)
            base = np.ones((2, 2), dtype=np.float64)
            if scorer == 0:
                return base
            improvement = 0.02
            if cell == 3 and domain == "factual":
                improvement = 0.0
            if cell == 3 and domain == "all_action":
                improvement = -pb21l.AA_BCE_MARGIN
            return base - improvement

        def fake_bootstrap(episode, sampled):
            return np.repeat(episode.mean(axis=0, keepdims=True), len(sampled), axis=0)

        with patch.object(pb21l, "_per_root_domain_losses", fake_losses), patch.object(
            pb21l, "_episode_means", lambda values, ordinal: values
        ), patch.object(pb21l, "_bootstrap_means", fake_bootstrap):
            report = pb21l.direct_comparison_report(
                arrays, candidate=pb21l.SCORER_POOL
            )
        self.assertGreater(
            report["comparisons"]["factual"]["cell_point_deltas"]
            ["F0/init-51042"]["BCE"],
            0.0,
        )
        self.assertIs(
            report["comparisons"]["factual"]["checks"]["BCE"], False
        )
        self.assertIs(
            report["comparisons"]["all_action"]["checks"]["BCE"], False
        )
        self.assertIs(report["passed"], False)

    def test_candidate_gate_failure_routes_through_precedence(self) -> None:
        arrays = {
            "fit_targets": np.zeros((2, 1, 5)),
            "fit_actions": np.zeros((2, 1), dtype=np.int64),
            "cell_fit_index": np.asarray([0, 0, 1, 1]),
            "dev_targets": np.zeros((1, 5)),
            "dev_actions": np.zeros(1, dtype=np.int64),
            "dev_cluster_ordinal": np.zeros(1, dtype=np.int32),
            "dev_raw_logits": np.zeros((3, 4, 1, 5)),
            "dev_raw_probabilities": np.zeros((3, 4, 1, 5)),
            "dev_calibrated_logits": np.zeros((3, 4, 1, 5)),
            "dev_calibrated_probabilities": np.zeros((3, 4, 1, 5)),
            "bootstrap_indices": np.zeros((1, 1), dtype=np.int32),
            "episode_derangements": np.zeros((1, 1), dtype=np.int32),
            "aa_accepted": np.ones((3, 4), dtype=np.bool_),
        }
        for scorer in range(3):
            arrays["dev_raw_logits"][scorer, :, 0, 0] = scorer
        calls = {"index": 0}

        def fake_absolute(**kwargs):
            scorer = int(kwargs["raw_logits"][0, 0])
            calls["index"] += 1
            return {"passed": not (scorer == 2 and calls["index"] == 12)}

        def fake_direct(arrays, *, candidate):
            return {"passed": candidate == pb21l.SCORER_POOL}

        with patch.object(pb21l, "absolute_cell_gate", fake_absolute), patch.object(
            pb21l, "direct_comparison_report", fake_direct
        ):
            report = pb21l.evaluate_authoritative_evidence(arrays)
        self.assertEqual(report["selection"]["fresh_data_license"], pb21l.SCORER_POOL)
        self.assertIs(report["path_passed"][pb21l.SCORER_BAL], False)

    def test_frozen_shared_array_and_inference_hash_tamper_rejection(self) -> None:
        frozen = {
            "fit_targets": np.zeros((3, 1, 5)),
            "fit_actions": np.zeros((3, 1), dtype=np.int64),
            "fit_root_ids": np.zeros((3, 1), dtype="S64"),
            "cal_targets": np.zeros((3, 1, 5)),
            "cal_actions": np.zeros((3, 1), dtype=np.int64),
            "cal_root_ids": np.zeros((3, 1), dtype="S64"),
            "dev_targets": np.zeros((2, 5)),
            "dev_actions": np.zeros(2, dtype=np.int64),
            "dev_root_ids": np.zeros(2, dtype="S64"),
            "dev_cluster_ordinal": np.asarray([0, 1], dtype=np.int32),
            "dev_cluster_ids": np.zeros(2, dtype="S64"),
            "bootstrap_indices": np.asarray([[0, 1]], dtype=np.int32),
            "episode_derangements": np.asarray([[1, 0]], dtype=np.int32),
        }
        arrays = {
            "fit_targets": frozen["fit_targets"][[0, 2]].copy(),
            "fit_actions": frozen["fit_actions"][[0, 2]].copy(),
            "fit_root_ids": frozen["fit_root_ids"][[0, 2]].copy(),
            "cal_targets": frozen["cal_targets"][[0, 2]].copy(),
            "cal_actions": frozen["cal_actions"][[0, 2]].copy(),
            "cal_root_ids": frozen["cal_root_ids"][[0, 2]].copy(),
            **{name: frozen[name].copy() for name in (
                "dev_targets", "dev_actions", "dev_root_ids", "dev_cluster_ordinal",
                "dev_cluster_ids", "bootstrap_indices", "episode_derangements"
            )},
        }
        with patch.object(
            pb21l,
            "EXACT_PB21K_BOOTSTRAP_SHA256",
            pb21l._array_sha256(frozen["bootstrap_indices"]),
        ), patch.object(
            pb21l,
            "EXACT_PB21K_DERANGEMENT_SHA256",
            pb21l._array_sha256(frozen["episode_derangements"]),
        ):
            pb21l.validate_frozen_shared_arrays(arrays, frozen)
            changed = {name: value.copy() for name, value in arrays.items()}
            changed["dev_targets"][0, 0] = 1.0
            with self.assertRaisesRegex(ValueError, "dev_targets"):
                pb21l.validate_frozen_shared_arrays(changed, frozen)
            changed = {name: value.copy() for name, value in arrays.items()}
            changed["dev_actions"][0] = 1
            with self.assertRaisesRegex(ValueError, "dev_actions"):
                pb21l.validate_frozen_shared_arrays(changed, frozen)
            other_frozen = {name: value.copy() for name, value in frozen.items()}
            other_arrays = {name: value.copy() for name, value in arrays.items()}
            other_frozen["bootstrap_indices"][:] = 1
            other_arrays["bootstrap_indices"][:] = 1
            with self.assertRaisesRegex(ValueError, "valid-shaped"):
                pb21l.validate_frozen_shared_arrays(other_arrays, other_frozen)
            other_frozen = {name: value.copy() for name, value in frozen.items()}
            other_arrays = {name: value.copy() for name, value in arrays.items()}
            other_frozen["episode_derangements"][:] = 0
            other_arrays["episode_derangements"][:] = 0
            with self.assertRaisesRegex(ValueError, "valid-shaped"):
                pb21l.validate_frozen_shared_arrays(other_arrays, other_frozen)

    def test_transform_and_base_array_tamper_rejection(self) -> None:
        raw = np.zeros((3, 4, 2, 5), dtype=np.float64)
        scales = np.ones((3, 4, 5), dtype=np.float64)
        biases = np.zeros((3, 4, 5), dtype=np.float64)
        arrays = {
            "dev_raw_logits": raw.copy(),
            "dev_raw_probabilities": pb21l.pb21k._sigmoid(raw),
            "aa_scales": scales,
            "aa_biases": biases,
            "dev_calibrated_logits": raw.copy(),
            "dev_calibrated_probabilities": pb21l.pb21k._sigmoid(raw),
            "cal_raw_logits": np.zeros((3, 4, 1, 5)),
            "aa_accepted": np.ones((3, 4), dtype=np.bool_),
        }
        frozen = {
            "cal_nz_raw_logits": np.zeros((9, 1, 5)),
            "calibrator_scales": np.ones((2, 9, 5)),
            "calibrator_biases": np.zeros((2, 9, 5)),
            "calibrator_accepted": np.ones((2, 9), dtype=np.bool_),
            "dev_raw_logits": np.zeros((9, 5, 2, 5)),
            "dev_raw_probabilities": np.full((9, 5, 2, 5), 0.5),
            "nz_calibrated_logits": np.zeros((2, 9, 2, 5)),
            "nz_calibrated_probabilities": np.full((2, 9, 2, 5), 0.5),
        }
        pb21l.validate_probability_transforms(arrays)
        pb21l.validate_base_array_replay(arrays, frozen)
        tampered = {name: value.copy() for name, value in arrays.items()}
        tampered["dev_calibrated_probabilities"][0, 0, 0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "calibrated probabilities"):
            pb21l.validate_probability_transforms(tampered)
        tampered = {name: value.copy() for name, value in arrays.items()}
        tampered["dev_raw_logits"][0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "BASE"):
            pb21l.validate_base_array_replay(tampered, frozen)

    def test_register_refuses_if_lifecycle_already_started(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "registration": root / "registration.json",
                "attempt": root / "attempt.json",
                "evidence": root / "evidence.npz",
                "result": root / "result.json",
                "upstream_result": root / "upstream.json",
            }
            paths["attempt"].write_text("{}", encoding="utf-8")
            with patch.object(pb21l, "canonical_paths", lambda _: paths), patch.object(
                pb21l,
                "registration_payload",
                lambda _: self.fail("payload must not be built after attempt"),
            ):
                with self.assertRaisesRegex(FileExistsError, "retry is forbidden"):
                    pb21l.register(paths["registration"])

    def test_evidence_writer_is_deterministic_create_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {"x": np.arange(8, dtype=np.int32)}
            with patch.object(pb21l, "EVIDENCE_KEYS", frozenset({"x"})):
                first = pb21l.publish_evidence_create_only(
                    root / "first.npz", arrays, attempt_sha256="a" * 64
                )
                second = pb21l.publish_evidence_create_only(
                    root / "second.npz", arrays, attempt_sha256="a" * 64
                )
                self.assertEqual(first["sha256"], second["sha256"])
                with self.assertRaisesRegex(FileExistsError, "retry is forbidden"):
                    pb21l.publish_evidence_create_only(
                        root / "first.npz", arrays, attempt_sha256="a" * 64
                    )


if __name__ == "__main__":
    unittest.main()
