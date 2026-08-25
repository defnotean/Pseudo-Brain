"""Synthetic contract tests for the DEV-only V2.1i runner."""
from __future__ import annotations

import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v21i_development_runner import (
    ACTION_COUNT,
    CALIBRATION_MODE,
    DEV_EPISODES,
    DEV_OFFSET,
    TRAIN_CAL_EPISODES,
    TRAIN_CAL_OFFSET,
    TRAIN_FIT_EPISODES,
    TRAIN_FIT_OFFSET,
    TRAINING_EPOCHS,
    REQUIRED_MODEL_SEEDS,
    CollectedEvidence,
    PartitionContract,
    PartitionSource,
    _assert_publication_targets_available,
    _canonicalize_table,
    _publish_checkpoint_create_only,
    _publish_json_create_only,
    _run_with_failure_receipt,
    audit_model_finite,
    assert_evidence_partitions_disjoint,
    calibration_expected_bindings,
    development_partition_contracts,
    evaluate_development_metrics,
    fit_calibrator,
    installed_model_preservation_report,
    learning_from_initialization_report,
    model_config,
    refine_hazard_path,
    run,
    train_joint_model,
    v21i_development_gate,
    v21i_five_seed_cohort_gate,
)
from irene_brain.data import DatasetSplit
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model


class _NeverMaterializedDataset:
    def __init__(self, config) -> None:
        self.config = config
        self.manifest_sha256 = "1" * 64
        self.materializations = 0

    def epoch_indices(self, *, epoch: int, shuffle: bool = True):
        return tuple(range(self.config.sequence_count))

    def __getitem__(self, index: int):
        self.materializations += 1
        raise AssertionError("synthetic source must not materialize a sequence")


def _synthetic_evidence(name: str, roots: int = 150) -> CollectedEvidence:
    rows = np.arange(roots, dtype=np.int64)
    actions = np.arange(ACTION_COUNT, dtype=np.int64)
    targets = ((rows[:, None] // ACTION_COUNT + actions[None, :]) % 2).astype(
        np.float64
    )
    probabilities = np.where(targets == 1.0, 0.78, 0.22)
    # Synthetic raw path has a known positive calibration bias.  The baked
    # bias-only transform represented by ``probabilities`` removes it.
    logits = np.log(probabilities / (1.0 - probabilities)) + 0.35
    factual = rows % ACTION_COUNT
    episode_groups = tuple(f"{name}:episode:{row // 2}" for row in rows)
    sequence_groups = tuple(sorted(set(episode_groups)))
    rng = np.random.default_rng(21)
    beliefs = rng.normal(size=(roots, 12))
    return CollectedEvidence(
        raw_hazard_logits=logits,
        hazard_probabilities=probabilities,
        hazard_targets=targets,
        factual_actions=factual,
        episode_group_ids=episode_groups,
        root_state_ids=tuple(
            sha256(f"{name}:{row}".encode("ascii")).hexdigest() for row in rows
        ),
        beliefs=beliefs,
        next_latent_losses=np.full((roots, ACTION_COUNT), 0.20),
        reward_losses=np.full((roots, ACTION_COUNT), 0.30),
        sequence_group_ids=sequence_groups,
    )


def _passing_gate_inputs():
    def factual_support(partition: str, counts: list[int]):
        return {
            "partition": partition,
            "observations": sum(counts),
            "every_action_has_both_hazard_classes": True,
            "per_action": [
                {
                    "action_id": action,
                    "observations": observations,
                    "positives": 10,
                    "negatives": observations - 10,
                    "has_both_hazard_classes": True,
                }
                for action, observations in enumerate(counts)
            ],
        }

    model_actions = [
        {
            "action_id": action,
            "metrics": {
                "brier": 0.10,
                "roc_auc": 0.75,
                "pr_auc": 0.55,
                "prevalence": 0.30,
                "ece_equal_mass": 0.04,
                "calibration_bias": 0.02,
            },
        }
        for action in range(ACTION_COUNT)
    ]
    baseline_actions = [
        {"action_id": action, "metrics": {"brier": 0.20}}
        for action in range(ACTION_COUNT)
    ]
    partition_algorithm = (
        "fixed_contiguous_episode_ranges_v1:"
        "TRAIN-FIT=train[16777216,16777344);"
        "TRAIN-CAL=train[16777344,16777408);"
        "DEV=validation[25165824,25165888)"
    )
    expected_bindings = {
        "source_namespace": "maze_chase.v21i.train-only.v1",
        "source_split": "TRAIN",
        "source_partition": "TRAIN-CAL",
        "calibration_group_sha256": "9" * 64,
        "upstream_model_fit_group_sha256": "a" * 64,
        "upstream_checkpoint_sha256": "2" * 64,
        "dataset_manifest_sha256": "3" * 64,
        "source_bundle_sha256": "4" * 64,
        "partition_algorithm": partition_algorithm,
        "calibration_groups": TRAIN_CAL_EPISODES,
        "observations": 3_840,
        "per_action_observations": 768,
    }
    calibration = {
        "schema": "irene.hazard.per_action_affine.v1",
        "formula": (
            "calibrated_logit[action] = scale[action] * raw_logit + bias[action]"
        ),
        "action_count": ACTION_COUNT,
        "accepted": True,
        "fit_mode": "bias_only",
        "scale": [1.0] * ACTION_COUNT,
        "bias": [-0.1] * ACTION_COUNT,
        "fit_bce": {"before": 0.65, "after": 0.60},
        "fitted_on": {
            key: value
            for key, value in expected_bindings.items()
            if key != "per_action_observations"
        },
        "per_action": [
            {
                "action_id": action,
                "observations": 768,
                "positives": 384,
                "negatives": 384,
                "before_bce": 0.65,
                "after_bce": 0.60,
                "accepted": True,
            }
            for action in range(ACTION_COUNT)
        ],
    }
    metrics = {
        "dev": {
            "raw_all_action": {"aggregate": {"bce": 0.42}},
            "calibrated_all_action": {
                "aggregate": {
                    "bce": 0.40,
                    "brier": 0.10,
                    "roc_auc": 0.75,
                    "ece_equal_mass": 0.04,
                },
                "per_action": model_actions,
            },
            "calibrated_factual": {"brier": 0.10},
            "baseline_metrics": {
                "all_action": {
                    "aggregate": {"bce": 0.60, "brier": 0.20, "roc_auc": 0.50},
                    "per_action": baseline_actions,
                },
                "factual": {"bce": 0.50, "brier": 0.20},
            },
            "factual_baseline_eligibility": {
                "schema_version": 1,
                "action_ids": list(range(ACTION_COUNT)),
                "minimum_per_class_per_action": 1,
                "partitions": {
                    "TRAIN-FIT": factual_support(
                        "TRAIN-FIT",
                        [308, 307, 307, 307, 307],
                    ),
                    "DEV": factual_support("DEV", [154, 154, 154, 153, 153]),
                },
                "baseline_factual": {
                    "bce": 0.50,
                    "brier": 0.20,
                    "finite": True,
                    "strictly_positive": True,
                },
                "passed": True,
            },
            "causal_cross_episode_derangements": {
                "repetitions": 20,
                "shuffled": [{} for _ in range(20)],
                "median_shuffled_bce": 0.50,
                "aggregate_roc_auc_drop": 0.06,
                "per_action": [
                    {"action_id": action, "roc_auc_drop": 0.04}
                    for action in range(ACTION_COUNT)
                ],
            },
            "all_action_clustered_bootstrap": {
                "bce_improvement_lower_bound": 0.01,
                "brier_improvement_lower_bound": 0.01,
            },
            "factual_clustered_bootstrap": {
                "bce_improvement_lower_bound": 0.01,
                "brier_improvement_lower_bound": 0.01,
            },
            "belief_noncollapse": {"passed": True},
            "partition_disjointness": {"all_factual_roots_disjoint": True},
            "replay_identity": {
                "ordered_roots_equal": True,
                "hazard_labels_equal": True,
                "factual_actions_equal": True,
                "ordered_root_sha256": "8" * 64,
            },
            "state_immutability": {
                "initialized_reference_unchanged": True,
                "uncalibrated_parent_unchanged": True,
                "installed_model_unchanged": True,
            },
            "joint_training": {
                "source_partition": "TRAIN-FIT",
                "optimizer": "AdamW",
                "epochs": 3,
                "optimizer_steps": 48,
                "batch_size": 8,
                "sequence_count": 128,
                "sequence_length": 16,
                "burn_in_steps": 4,
                "learning_rate": 1.0e-4,
                "weight_decay": 1.0e-4,
                "clip_norm": 1.0,
                "prediction_error_intervention": "normal",
                "target_encoder_update": "after_every_optimizer_step",
                "sampling": "deterministic_episode_permutation_without_replacement",
                "wall_seconds": 1.0,
                "telemetry": [
                    {
                        "epoch": epoch,
                        "samples": 1_536,
                        "loss": 1.0,
                        "components": {"hazard": 0.5},
                        "maximum_preclip_gradient_norm": 0.5,
                    }
                    for epoch in range(1, 4)
                ],
            },
            "hazard_refinement": {
                "enabled": True,
                "source_partition": "TRAIN-FIT",
                "dev_used_for_stopping": False,
                "passes": 8,
                "root_batch_size": 24,
                "root_count": 1536,
                "optimizer_steps_per_pass": 64,
                "optimizer_steps": 512,
                "trainable_parameter_count": 44_161,
                "all_other_tensors_byte_equal": True,
                "sampling": (
                    "deterministic_complete_root_permutation_without_replacement"
                ),
            },
            "raw_semantic_hazard_logits": {
                "retained": True,
                "action_ids": list(range(ACTION_COUNT)),
                "shape": [768, ACTION_COUNT],
                "sha256": "7" * 64,
            },
            "learning_from_initialization": {
                "checks": {
                    "hazard_loss_ratio_at_most_0_90": True,
                    "next_loss_ratio_at_most_0_60": True,
                    "reward_loss_ratio_at_most_0_80": True,
                    "every_action_next_loss_ratio_at_most_0_90": True,
                    "every_action_reward_loss_ratio_at_most_0_90": True,
                }
            },
            "installed_model_preservation": {
                "checks": {
                    "decision_loss_ratio_at_most_1_01": True,
                    "next_loss_ratio_at_most_1_01": True,
                    "reward_loss_ratio_at_most_1_01": True,
                    "next_cosine_drop_at_most_0_01": True,
                    "reward_mae_ratio_at_most_1_02": True,
                }
            },
            "finite_state_audit": {
                "all_parameters_finite": True,
                "all_buffers_finite": True,
                "parameter_tensor_count": 10,
                "buffer_tensor_count": 5,
                "state_sha256": "6" * 64,
            },
        }
    }
    installation = {
        "allowed_tensor_names": [
            "world_model.outcome_model.hazard_calibration_bias",
            "world_model.outcome_model.hazard_calibration_scale",
        ],
        "changed_tensor_names": [
            "world_model.outcome_model.hazard_calibration_bias"
        ],
        "all_other_tensors_byte_equal": True,
        "state_sha256_before": "5" * 64,
        "state_sha256_after": "6" * 64,
    }
    return calibration, metrics, installation, expected_bindings


class V21IDevelopmentRunnerTests(unittest.TestCase):
    def test_only_exact_development_ranges_are_reserved(self) -> None:
        contracts = development_partition_contracts()
        self.assertEqual(set(contracts), {"TRAIN-FIT", "TRAIN-CAL", "DEV"})
        fit = contracts["TRAIN-FIT"].dataset_config
        cal = contracts["TRAIN-CAL"].dataset_config
        dev = contracts["DEV"].dataset_config
        self.assertEqual(
            (fit.split, fit.seed_offset, fit.sequence_count),
            (DatasetSplit.TRAIN, TRAIN_FIT_OFFSET, TRAIN_FIT_EPISODES),
        )
        self.assertEqual(
            (cal.split, cal.seed_offset, cal.sequence_count),
            (DatasetSplit.TRAIN, TRAIN_CAL_OFFSET, TRAIN_CAL_EPISODES),
        )
        self.assertEqual(
            (dev.split, dev.seed_offset, dev.sequence_count),
            (DatasetSplit.VALIDATION, DEV_OFFSET, DEV_EPISODES),
        )
        self.assertEqual(fit.seed_offset + fit.sequence_count, cal.seed_offset)
        self.assertTrue(
            all(
                contract.dataset_config.counterfactual_targets == "all_actions_v1"
                for contract in contracts.values()
            )
        )
        self.assertTrue(
            all(
                contract.dataset_config.behavior_policy
                == "balanced_intervention_v1"
                and contract.dataset_config.behavior_intervention_rate == 0.5
                for contract in contracts.values()
            )
        )

    def test_constructing_partition_sources_is_lazy_and_opens_no_dev_rows(self) -> None:
        dev = development_partition_contracts()["DEV"]
        source = PartitionSource(dev, dataset_factory=_NeverMaterializedDataset)
        self.assertEqual(source.manifest_sha256, "1" * 64)
        self.assertEqual(source.dataset.materializations, 0)

    def test_partition_contract_rejects_any_behavior_protocol_drift(self) -> None:
        invalid_behaviors = (
            ("teacher", 0.0),
            ("balanced_intervention_v1", np.nextafter(0.5, 0.0)),
            ("balanced_intervention_v1", np.nextafter(0.5, 1.0)),
            ("balanced_intervention_v1", 1.0),
        )
        for name, contract in development_partition_contracts().items():
            for behavior_policy, intervention_rate in invalid_behaviors:
                with self.subTest(
                    partition=name,
                    behavior_policy=behavior_policy,
                    intervention_rate=intervention_rate,
                ):
                    config = replace(
                        contract.dataset_config,
                        behavior_policy=behavior_policy,
                        behavior_intervention_rate=intervention_rate,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "balanced_intervention_v1|intervention rate 0.5",
                    ):
                        PartitionContract(name, config)

    def test_current_model_and_calibration_mode_are_explicit(self) -> None:
        config = model_config()
        self.assertEqual(CALIBRATION_MODE, "bias_only")
        self.assertEqual(config.outcome_architecture, "all_action_table_v1")
        self.assertEqual(config.hazard_outcome_path, "dedicated_stopgrad_v1")
        self.assertEqual(config.reward_prediction, "symlog_twohot_v1")
        self.assertEqual(config.prediction_error_fusion, "latent_outcome_surprise_v1")
        self.assertEqual(config.hazard_weight, 1.0)
        self.assertEqual(TRAINING_EPOCHS, 3)
        self.assertEqual(TRAIN_FIT_EPISODES // 8 * TRAINING_EPOCHS, 48)

    def test_joint_training_record_reads_burn_in_from_partition_contract(self) -> None:
        class SyntheticObjective:
            def __init__(self, model, **_kwargs) -> None:
                self.model = model

            def __call__(self, _batch):
                loss = self.model.weight.square().mean()
                return SimpleNamespace(
                    loss=loss,
                    samples=1,
                    components={"synthetic": loss},
                )

            def update_target_encoder(self) -> None:
                return None

        contract = SimpleNamespace(
            name="TRAIN-FIT",
            burn_in_steps=4,
            dataset_config=SimpleNamespace(
                sequence_count=128,
                sequence_length=16,
            ),
        )

        class SyntheticSource:
            def __init__(self) -> None:
                self.contract = contract

            def iter_all_action_batches(self, *, epoch, batch_size):
                self.last_request = (epoch, batch_size)
                yield object()

        model = torch.nn.Linear(1, 1, bias=False)
        with patch(
            "v21i_development_runner.AllActionV2TrajectoryObjective",
            SyntheticObjective,
        ):
            record = train_joint_model(model, SyntheticSource(), epochs=1)
        self.assertEqual(record["burn_in_steps"], 4)
        self.assertEqual(record["sequence_count"], 128)
        self.assertEqual(record["sequence_length"], 16)

    def test_canonicalize_uses_semantic_action_ids(self) -> None:
        ids = torch.tensor([[2, 0, 4, 1, 3]])
        values = torch.tensor([[[12.0], [10.0], [14.0], [11.0], [13.0]]])
        canonical = _canonicalize_table(values, ids)
        np.testing.assert_array_equal(canonical, [[10.0, 11.0, 12.0, 13.0, 14.0]])

    def test_calibration_and_metrics_use_only_synthetic_episode_groups(self) -> None:
        training = _synthetic_evidence("TRAIN-FIT")
        calibration = _synthetic_evidence("TRAIN-CAL")
        development = _synthetic_evidence("DEV")
        calibrator = fit_calibrator(
            calibration,
            training,
            upstream_checkpoint_sha256="2" * 64,
            dataset_manifest_sha256="3" * 64,
            source_bundle_sha256="4" * 64,
        )
        self.assertEqual(calibrator.source_split, "TRAIN")
        self.assertEqual(calibrator.source_partition, "TRAIN-CAL")
        self.assertEqual(calibrator.action_count, ACTION_COUNT)
        self.assertEqual(calibrator.fit_mode, "bias_only")
        self.assertTrue(calibrator.accepted)
        self.assertEqual(calibrator.upstream_checkpoint_sha256, "2" * 64)
        self.assertEqual(calibrator.dataset_manifest_sha256, "3" * 64)
        self.assertEqual(calibrator.source_bundle_sha256, "4" * 64)
        self.assertTrue(all(scale == 1.0 for scale in calibrator.scales))
        expected_bindings = calibration_expected_bindings(
            calibration,
            training,
            upstream_checkpoint_sha256="2" * 64,
            dataset_manifest_sha256="3" * 64,
            source_bundle_sha256="4" * 64,
        )
        exported_provenance = calibrator.export_parameters()["fitted_on"]
        for key, expected in expected_bindings.items():
            if key != "per_action_observations":
                self.assertEqual(exported_provenance[key], expected)
        report = evaluate_development_metrics(
            training,
            calibration,
            development,
            calibrator,
        )
        self.assertIn("calibration_fit_population", report)
        dev = report["dev"]
        self.assertEqual(len(dev["calibrated_all_action"]["per_action"]), ACTION_COUNT)
        self.assertEqual(
            dev["causal_cross_episode_derangements"]["repetitions"],
            20,
        )
        self.assertEqual(dev["belief_noncollapse"]["sample_count"], 150)
        eligibility = dev["factual_baseline_eligibility"]
        self.assertTrue(eligibility["passed"])
        for partition in ("TRAIN-FIT", "DEV"):
            self.assertTrue(
                eligibility["partitions"][partition][
                    "every_action_has_both_hazard_classes"
                ]
            )

    def test_artifact_and_checkpoint_publication_are_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            checkpoint = root / "candidate.pt"
            uncalibrated = root / "uncalibrated.pt"
            _assert_publication_targets_available(output, checkpoint, uncalibrated)
            _publish_json_create_only(output, {"mode": "synthetic"})
            _publish_checkpoint_create_only(
                checkpoint,
                {"state": {"weight": torch.tensor([1.0])}},
            )
            original_output = output.read_bytes()
            original_checkpoint = checkpoint.read_bytes()
            with self.assertRaises(FileExistsError):
                _assert_publication_targets_available(output, checkpoint, uncalibrated)
            with self.assertRaises(FileExistsError):
                _publish_json_create_only(output, {"mode": "replacement"})
            with self.assertRaises(FileExistsError):
                _publish_checkpoint_create_only(
                    checkpoint,
                    {"state": {"weight": torch.tensor([2.0])}},
                )
            self.assertEqual(output.read_bytes(), original_output)
            self.assertEqual(checkpoint.read_bytes(), original_checkpoint)

    def test_pure_development_gate_passes_only_complete_threshold_evidence(self) -> None:
        calibration, metrics, installation, bindings = _passing_gate_inputs()
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["checks"].values()))

        missing = copy.deepcopy(metrics)
        del missing["dev"]["all_action_clustered_bootstrap"]
        missing_gate = v21i_development_gate(
            calibration,
            missing,
            installation,
            bindings,
        )
        self.assertFalse(missing_gate["passed"])
        self.assertTrue(missing_gate["errors"])

        failed_auc = copy.deepcopy(metrics)
        failed_auc["dev"]["calibrated_all_action"]["aggregate"]["roc_auc"] = 0.64
        auc_gate = v21i_development_gate(
            calibration,
            failed_auc,
            installation,
            bindings,
        )
        self.assertFalse(auc_gate["passed"])
        self.assertFalse(auc_gate["checks"]["dev_aggregate_roc_auc"])

    def test_gate_rejects_calibration_that_worsens_parent_dev_bce(self) -> None:
        calibration, metrics, installation, bindings = _passing_gate_inputs()
        metrics["dev"]["raw_all_action"]["aggregate"]["bce"] = 0.39
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["checks"]["calibrated_dev_bce_not_worse_than_parent"]
        )

    def test_gate_requires_exact_calibration_run_bindings(self) -> None:
        replacements = {
            "source_namespace": "maze_chase.v21i.wrong.v1",
            "source_split": "VALIDATION",
            "source_partition": "TRAIN-FIT",
            "calibration_group_sha256": "b" * 64,
            "upstream_model_fit_group_sha256": "c" * 64,
            "upstream_checkpoint_sha256": "d" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "source_bundle_sha256": "f" * 64,
            "partition_algorithm": "wrong_partition_algorithm",
            "calibration_groups": 63,
            "observations": 3_839,
        }
        for key, replacement in replacements.items():
            with self.subTest(binding=key):
                calibration, metrics, installation, bindings = (
                    _passing_gate_inputs()
                )
                calibration["fitted_on"][key] = replacement
                gate = v21i_development_gate(
                    calibration,
                    metrics,
                    installation,
                    bindings,
                )
                self.assertFalse(gate["passed"])
                self.assertFalse(
                    gate["checks"]["calibration_provenance_complete"]
                )

    def test_gate_rejects_missing_nonfinite_and_inconsistent_calibrator_fields(self) -> None:
        calibration, metrics, installation, bindings = _passing_gate_inputs()
        del calibration["bias"]
        missing = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(missing["passed"])
        self.assertTrue(missing["errors"])

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        calibration["bias"][2] = float("nan")
        nonfinite = v21i_development_gate(
            calibration,
            metrics,
            installation,
            bindings,
        )
        self.assertFalse(nonfinite["passed"])
        self.assertTrue(nonfinite["errors"])

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        calibration["per_action"][0]["observations"] = 767
        inconsistent = v21i_development_gate(
            calibration,
            metrics,
            installation,
            bindings,
        )
        self.assertFalse(inconsistent["passed"])
        self.assertFalse(
            inconsistent["checks"]["calibration_full_schema_and_fit_records"]
        )

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        calibration["schema"] = "wrong.schema"
        calibration["action_count"] = ACTION_COUNT - 1
        wrong_schema = v21i_development_gate(
            calibration,
            metrics,
            installation,
            bindings,
        )
        self.assertFalse(
            wrong_schema["checks"]["calibration_full_schema_and_fit_records"]
        )

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        calibration["scale"][1] = float("inf")
        nonfinite_scale = v21i_development_gate(
            calibration,
            metrics,
            installation,
            bindings,
        )
        self.assertFalse(nonfinite_scale["passed"])
        self.assertTrue(nonfinite_scale["errors"])

    def test_gate_rejects_joint_optimizer_step_or_schedule_drift(self) -> None:
        calibration, metrics, installation, bindings = _passing_gate_inputs()
        metrics["dev"]["joint_training"]["optimizer_steps"] = 47
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["joint_training_exact_schedule"])

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        metrics["dev"]["joint_training"]["telemetry"][1]["samples"] = 1_535
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(gate["checks"]["joint_training_exact_schedule"])

    def test_gate_rejects_nonfinite_state_audit(self) -> None:
        calibration, metrics, installation, bindings = _passing_gate_inputs()
        metrics["dev"]["finite_state_audit"]["all_buffers_finite"] = False
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["checks"]["all_model_parameters_and_buffers_finite"]
        )

    def test_gate_rejects_degenerate_or_malformed_factual_baseline_evidence(
        self,
    ) -> None:
        calibration, metrics, installation, bindings = _passing_gate_inputs()
        train_action = metrics["dev"]["factual_baseline_eligibility"][
            "partitions"
        ]["TRAIN-FIT"]["per_action"][1]
        train_action["positives"] = 0
        train_action["negatives"] = train_action["observations"]
        train_action["has_both_hazard_classes"] = False
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["checks"]["factual_baseline_eligible_and_supported"]
        )

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        dev_action = metrics["dev"]["factual_baseline_eligibility"][
            "partitions"
        ]["DEV"]["per_action"][3]
        dev_action["positives"] = dev_action["observations"]
        dev_action["negatives"] = 0
        dev_action["has_both_hazard_classes"] = False
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(
            gate["checks"]["factual_baseline_eligible_and_supported"]
        )

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        legacy = metrics["dev"]["factual_baseline_eligibility"]["partitions"][
            "TRAIN-FIT"
        ]["per_action"]
        for action, record in enumerate(legacy):
            record["positives"] = record["observations"] if action == 0 else 0
            record["negatives"] = record["observations"] - record["positives"]
            record["has_both_hazard_classes"] = False
        gate = v21i_development_gate(calibration, metrics, installation, bindings)
        self.assertFalse(
            gate["checks"]["factual_baseline_eligible_and_supported"]
        )

        for metric_name in ("bce", "brier"):
            with self.subTest(zero_baseline=metric_name):
                calibration, metrics, installation, bindings = (
                    _passing_gate_inputs()
                )
                baseline = metrics["dev"]["baseline_metrics"]["factual"]
                eligibility = metrics["dev"]["factual_baseline_eligibility"][
                    "baseline_factual"
                ]
                baseline[metric_name] = 0.0
                eligibility[metric_name] = 0.0
                gate = v21i_development_gate(
                    calibration,
                    metrics,
                    installation,
                    bindings,
                )
                self.assertFalse(
                    gate["checks"]["factual_baseline_eligible_and_supported"]
                )

        for metric_name in ("bce", "brier"):
            with self.subTest(nonfinite_baseline=metric_name):
                calibration, metrics, installation, bindings = (
                    _passing_gate_inputs()
                )
                baseline = metrics["dev"]["baseline_metrics"]["factual"]
                eligibility = metrics["dev"]["factual_baseline_eligibility"][
                    "baseline_factual"
                ]
                baseline[metric_name] = float("nan")
                eligibility[metric_name] = float("nan")
                gate = v21i_development_gate(
                    calibration,
                    metrics,
                    installation,
                    bindings,
                )
                self.assertFalse(gate["passed"])
                self.assertTrue(gate["errors"])

        calibration, metrics, installation, bindings = _passing_gate_inputs()
        del metrics["dev"]["factual_baseline_eligibility"]["partitions"]["DEV"][
            "per_action"
        ]
        malformed = v21i_development_gate(
            calibration,
            metrics,
            installation,
            bindings,
        )
        self.assertFalse(malformed["passed"])
        self.assertTrue(malformed["errors"])

    def test_five_seed_gate_rejects_missing_extra_or_one_failed_seed(self) -> None:
        passing = {seed: {"passed": True} for seed in REQUIRED_MODEL_SEEDS}
        self.assertTrue(v21i_five_seed_cohort_gate(passing)["passed"])

        one_failed = copy.deepcopy(passing)
        one_failed[REQUIRED_MODEL_SEEDS[-1]]["passed"] = False
        self.assertFalse(v21i_five_seed_cohort_gate(one_failed)["passed"])

        missing = dict(passing)
        missing.pop(REQUIRED_MODEL_SEEDS[0])
        self.assertFalse(v21i_five_seed_cohort_gate(missing)["passed"])

        extra = {**passing, 49: {"passed": True}}
        self.assertFalse(v21i_five_seed_cohort_gate(extra)["passed"])

    def test_learning_and_installation_preservation_gates_fail_independently(self) -> None:
        initial_evidence = _synthetic_evidence("INIT")
        final_evidence = replace(
            initial_evidence,
            next_latent_losses=initial_evidence.next_latent_losses * 0.50,
            reward_losses=initial_evidence.reward_losses * 0.70,
        )
        initial_objective = {
            "components": {"hazard": 1.0, "next_latent": 1.0, "reward": 1.0},
        }
        final_objective = {
            "components": {"hazard": 0.80, "next_latent": 0.50, "reward": 0.70},
        }
        learning = learning_from_initialization_report(
            initial_objective,
            final_objective,
            initial_evidence,
            final_evidence,
        )
        self.assertTrue(learning["passed"])
        failed_learning = learning_from_initialization_report(
            initial_objective,
            {"components": {"hazard": 0.80, "next_latent": 0.61, "reward": 0.70}},
            initial_evidence,
            final_evidence,
        )
        self.assertFalse(failed_learning["passed"])
        self.assertFalse(failed_learning["checks"]["next_loss_ratio_at_most_0_60"])

        before = {
            "components": {"decision": 1.0, "next_latent": 1.0, "reward": 1.0},
            "metrics": {"next_latent_cosine": 0.80, "reward_mae": 1.0},
        }
        after = {
            "components": {
                "decision": 1.005,
                "next_latent": 1.005,
                "reward": 1.005,
            },
            "metrics": {"next_latent_cosine": 0.795, "reward_mae": 1.01},
        }
        preservation = installed_model_preservation_report(before, after)
        self.assertTrue(preservation["passed"])
        failed_after = copy.deepcopy(after)
        failed_after["components"]["decision"] = 1.011
        failed_preservation = installed_model_preservation_report(before, failed_after)
        self.assertFalse(failed_preservation["passed"])
        self.assertFalse(
            failed_preservation["checks"]["decision_loss_ratio_at_most_1_01"]
        )

    def test_post_preflight_exception_always_writes_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "failed.json"
            checkpoint = root / "candidate.pt"
            parent = root / "parent.pt"

            def fail() -> None:
                _publish_checkpoint_create_only(parent, {"state": torch.tensor([1.0])})
                raise RuntimeError("synthetic failure after parent publication")

            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                _run_with_failure_receipt(
                    output=output,
                    checkpoint=checkpoint,
                    uncalibrated_checkpoint=parent,
                    operation=fail,
                )
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["classification"], "development_run_failed")
            self.assertFalse(receipt["qualification_claimed"])
            self.assertEqual(receipt["failure"]["type"], "RuntimeError")
            self.assertTrue(receipt["upstream_uncalibrated_checkpoint"]["published"])
            self.assertFalse(receipt["checkpoint"]["published"])

    def test_preflight_checkpoint_collision_writes_receipt_when_output_is_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "failed.json"
            checkpoint = root / "occupied.pt"
            parent = root / "parent.pt"
            checkpoint.write_bytes(b"existing checkpoint")

            with self.assertRaises(FileExistsError):
                run(
                    output=output,
                    checkpoint=checkpoint,
                    uncalibrated_checkpoint=parent,
                    model_seed=44,
                    epochs=TRAINING_EPOCHS,
                    calibration_mode=CALIBRATION_MODE,
                )
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["classification"], "development_run_failed")
            self.assertEqual(receipt["failure"]["type"], "FileExistsError")
            self.assertTrue(receipt["checkpoint"]["published"])
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")

    def test_preflight_collision_never_overwrites_occupied_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "occupied.json"
            checkpoint = root / "occupied.pt"
            parent = root / "parent.pt"
            original = b"existing output"
            output.write_bytes(original)
            checkpoint.write_bytes(b"existing checkpoint")

            with self.assertRaises(FileExistsError):
                run(
                    output=output,
                    checkpoint=checkpoint,
                    uncalibrated_checkpoint=parent,
                    model_seed=44,
                    epochs=TRAINING_EPOCHS,
                    calibration_mode=CALIBRATION_MODE,
                )
            self.assertEqual(output.read_bytes(), original)

    def test_model_finiteness_audit_rejects_parameter_and_buffer_nan(self) -> None:
        torch.manual_seed(44)
        model = CoreV2Model(config=model_config(), flags=CONFIG_B_PREDICTIVE)
        passing = audit_model_finite(model)
        self.assertTrue(passing["all_parameters_finite"])
        self.assertTrue(passing["all_buffers_finite"])

        with torch.no_grad():
            next(model.parameters()).flatten()[0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite model parameter"):
            audit_model_finite(model)

        torch.manual_seed(44)
        model = CoreV2Model(config=model_config(), flags=CONFIG_B_PREDICTIVE)
        floating_buffer = next(
            value for value in model.buffers() if value.is_floating_point()
        )
        with torch.no_grad():
            floating_buffer.flatten()[0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite model buffer"):
            audit_model_finite(model)

    def test_episode_root_disjointness_is_fail_closed(self) -> None:
        training = _synthetic_evidence("TRAIN-FIT")
        calibration = _synthetic_evidence("TRAIN-CAL")
        development = _synthetic_evidence("DEV")
        audit = assert_evidence_partitions_disjoint(
            {"TRAIN-FIT": training, "TRAIN-CAL": calibration, "DEV": development}
        )
        self.assertTrue(audit["all_factual_roots_disjoint"])
        overlapping = replace(
            calibration,
            root_state_ids=training.root_state_ids,
        )
        with self.assertRaisesRegex(RuntimeError, "share factual root"):
            assert_evidence_partitions_disjoint(
                {"TRAIN-FIT": training, "TRAIN-CAL": overlapping}
            )

    def test_hazard_refinement_is_exhaustive_whole_root_and_hazard_only(self) -> None:
        evidence = _synthetic_evidence("TRAIN-FIT", roots=48)
        evidence = replace(
            evidence,
            beliefs=np.random.default_rng(44).normal(size=(48, 120)),
        )
        torch.manual_seed(44)
        model = CoreV2Model(config=model_config(), flags=CONFIG_B_PREDICTIVE)
        audit = refine_hazard_path(model, evidence, seed=20_044)
        self.assertEqual(audit["passes"], 8)
        self.assertEqual(audit["root_batch_size"], 24)
        self.assertEqual(audit["optimizer_steps_per_pass"], 2)
        self.assertEqual(audit["optimizer_steps"], 16)
        self.assertEqual(audit["trainable_parameter_count"], 44_161)
        self.assertTrue(audit["all_other_tensors_byte_equal"])
        self.assertEqual(len(audit["pass_records"]), 8)
        self.assertTrue(all(record["roots"] == 48 for record in audit["pass_records"]))


if __name__ == "__main__":
    unittest.main()
