"""Synthetic contract tests for the V2.1i TRAIN-only continuation diagnostic."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v21i_frozen_representation_continuation_diagnostic_v1 import (
    ADDED_PASSES,
    CANDIDATE_PUBLICATION_ALLOWED,
    CLASSIFICATION,
    EXACT_TRAIN_CAL_MANIFEST_SHA256,
    EXACT_TRAIN_FIT_MANIFEST_SHA256,
    EXACT_V3_PARENT_SHA256,
    EXACT_V3_PARENT_STATE_SHA256,
    EXACT_V3_RESULT_SHA256,
    EXACT_V3_SOURCE_BUNDLE_SHA256,
    MILESTONES,
    MODE,
    PERMUTATION_SEED,
    TRAINABLE_PARAMETER_COUNT,
    _assert_targets_available,
    _checkpoint_paths,
    _diagnostic_source_bundle,
    _optimizer_state_sha256,
    _ordered_string_sequence_sha256,
    _run_with_failure_receipt,
    diagnostic_partition_contracts,
    diagnostic_partition_sources,
    deterministic_permutations,
    interpretation_report,
    preregistered_permutation_digests,
    register_preregistration,
    schedule_record,
    scope_limitation_record,
    select_hazard_path_parameters,
    validate_pass_chain,
    validate_preregistration_receipt,
    validate_upstream_result_payload,
)
from irene_brain.data import DatasetSplit
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model
from v21i_development_runner import model_config


def _upstream_fixture() -> dict[str, object]:
    partitions: dict[str, object] = {}
    for name, digest in (
        ("TRAIN-FIT", EXACT_TRAIN_FIT_MANIFEST_SHA256),
        ("TRAIN-CAL", EXACT_TRAIN_CAL_MANIFEST_SHA256),
        ("DEV", "f" * 64),
    ):
        partitions[name] = {
            "burn_in_steps": 4,
            "dataset_manifest_sha256": digest,
            "dataset_manifest": {
                "behavior_policy": "balanced_intervention_v1",
                "behavior_intervention_rate_hex": float(0.5).hex(),
                "counterfactual_targets": "all_actions_v1",
            },
        }
    return {
        "schema_version": 1,
        "mode": "v21i_development_only",
        "classification": "development_candidate_not_qualified",
        "qualification_claimed": False,
        "model_seed": 42,
        "epochs": 3,
        "source_bundle": {"sha256": EXACT_V3_SOURCE_BUNDLE_SHA256},
        "development_gate": {"passed": False},
        "namespace_policy": {
            "test_split_opened": False,
            "cpu_qual_opened": False,
            "dev_used_for_training_or_calibration": False,
        },
        "dataset_partitions": partitions,
        "upstream_uncalibrated_checkpoint": {
            "path": "exact-v3-parent.pt",
            "sha256": EXACT_V3_PARENT_SHA256,
            "state_sha256": EXACT_V3_PARENT_STATE_SHA256,
        },
    }


def _partition_metrics(
    *,
    bce: float,
    auc: float,
    action_aucs: list[float],
    shuffle_ratio: float,
) -> dict[str, object]:
    return {
        "all_action": {
            "aggregate": {"bce": bce, "roc_auc": auc},
            "per_action": [
                {"action_id": action, "metrics": {"roc_auc": action_auc}}
                for action, action_auc in enumerate(action_aucs)
            ],
        },
        "shuffle_to_model_bce_ratio": shuffle_ratio,
    }


def _milestone_pair(
    *,
    fit_bce: float,
    cal_bce: float,
    cal_auc: float,
    action_aucs: list[float],
    shuffle_ratio: float,
) -> dict[str, object]:
    return {
        "TRAIN-FIT": _partition_metrics(
            bce=fit_bce,
            auc=0.6,
            action_aucs=[0.6] * 5,
            shuffle_ratio=1.0,
        ),
        "TRAIN-CAL": _partition_metrics(
            bce=cal_bce,
            auc=cal_auc,
            action_aucs=action_aucs,
            shuffle_ratio=shuffle_ratio,
        ),
    }


class V21IFrozenContinuationDiagnosticTests(unittest.TestCase):
    def test_schedule_and_publication_boundary_are_frozen(self) -> None:
        schedule = schedule_record()
        self.assertEqual(MODE, "v21i_frozen_representation_continuation_diagnostic_v1")
        self.assertEqual(CLASSIFICATION, "diagnostic_not_candidate")
        self.assertFalse(CANDIDATE_PUBLICATION_ALLOWED)
        self.assertEqual(ADDED_PASSES, 16)
        self.assertEqual(MILESTONES, (0, 1, 2, 4, 8, 16))
        self.assertEqual(schedule["optimizer_steps"], 1_024)
        self.assertEqual(schedule["evaluation_partitions"], ["TRAIN-FIT", "TRAIN-CAL"])
        self.assertNotIn("DEV", schedule["evaluation_partitions"])
        self.assertFalse(schedule["early_stopping"])
        self.assertFalse(schedule["model_selection"])
        self.assertFalse(schedule["calibration_fit_or_install"])

    def test_live_signature_scope_limitation_is_explicit(self) -> None:
        limitation = scope_limitation_record()
        self.assertTrue(limitation["registered_recurrence_uses_prior_actual_reward"])
        self.assertTrue(limitation["registered_recurrence_uses_prior_actual_hazard"])
        self.assertFalse(limitation["live_signature_matched"])
        self.assertFalse(limitation["live_belief_sufficiency_claim_allowed"])
        self.assertEqual(limitation["live_qualification_implication"], "none")

    def test_exact_v3_result_binding_fails_closed(self) -> None:
        payload = _upstream_fixture()
        validated = validate_upstream_result_payload(
            payload,
            artifact_sha256=EXACT_V3_RESULT_SHA256,
        )
        self.assertEqual(validated["parent_sha256"], EXACT_V3_PARENT_SHA256)
        self.assertEqual(
            validated["train_fit_manifest_sha256"],
            EXACT_TRAIN_FIT_MANIFEST_SHA256,
        )

        with self.assertRaisesRegex(ValueError, "exact pinned v3"):
            validate_upstream_result_payload(payload, artifact_sha256="0" * 64)

        drift_cases = (
            ("parent", lambda value: value["upstream_uncalibrated_checkpoint"].update(sha256="0" * 64)),
            ("source", lambda value: value["source_bundle"].update(sha256="0" * 64)),
            ("gate", lambda value: value["development_gate"].update(passed=True)),
            (
                "behavior",
                lambda value: value["dataset_partitions"]["TRAIN-FIT"][
                    "dataset_manifest"
                ].update(behavior_policy="teacher"),
            ),
        )
        for name, mutate in drift_cases:
            with self.subTest(drift=name):
                drifted = copy.deepcopy(payload)
                mutate(drifted)
                with self.assertRaises(ValueError):
                    validate_upstream_result_payload(
                        drifted,
                        artifact_sha256=EXACT_V3_RESULT_SHA256,
                    )

    def test_only_exact_44161_hazard_path_parameters_are_trainable(self) -> None:
        torch.manual_seed(42)
        model = CoreV2Model(config=model_config(), flags=CONFIG_B_PREDICTIVE)
        parameters, names = select_hazard_path_parameters(model)
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 44_161)
        self.assertEqual(TRAINABLE_PARAMETER_COUNT, 44_161)
        self.assertTrue(names)
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(trainable, set(names))
        self.assertTrue(
            all(
                ".hazard_state_trunk." in name
                or ".hazard_action_embedding." in name
                or ".hazard_outcome_trunk." in name
                or ".hazard_head." in name
                for name in names
            )
        )

    def test_permutations_and_ordered_sequence_digests_are_deterministic(self) -> None:
        first = deterministic_permutations(31, passes=4, seed=PERMUTATION_SEED)
        second = deterministic_permutations(31, passes=4, seed=PERMUTATION_SEED)
        self.assertEqual(len(first), 4)
        for left, right in zip(first, second, strict=True):
            self.assertTrue(torch.equal(left, right))
            self.assertEqual(int(left.unique().numel()), 31)
        self.assertNotEqual(
            _ordered_string_sequence_sha256(("a", "b", "a")),
            _ordered_string_sequence_sha256(("a", "a", "b")),
        )
        self.assertNotEqual(
            _ordered_string_sequence_sha256(("a", "b", "a")),
            _ordered_string_sequence_sha256(("a", "b")),
        )

    def test_train_only_factory_never_constructs_dev_or_validation(self) -> None:
        config_calls: list[dict[str, object]] = []
        contract_calls: list[str] = []
        source_calls: list[str] = []

        def config_spy(**kwargs: object) -> object:
            self.assertIs(kwargs["split"], DatasetSplit.TRAIN)
            self.assertNotEqual(kwargs["split"], DatasetSplit.VALIDATION)
            config_calls.append(dict(kwargs))
            return SimpleNamespace(**kwargs)

        def contract_spy(name: str, config: object) -> object:
            self.assertNotEqual(name, "DEV")
            contract_calls.append(name)
            return SimpleNamespace(name=name, dataset_config=config)

        def source_spy(contract: object) -> object:
            name = contract.name
            self.assertNotEqual(name, "DEV")
            self.assertIs(contract.dataset_config.split, DatasetSplit.TRAIN)
            source_calls.append(name)
            return SimpleNamespace(contract=contract)

        sources = diagnostic_partition_sources(
            config_factory=config_spy,
            contract_factory=contract_spy,
            source_factory=source_spy,
        )
        self.assertEqual(list(sources), ["TRAIN-FIT", "TRAIN-CAL"])
        self.assertEqual(contract_calls, ["TRAIN-FIT", "TRAIN-CAL"])
        self.assertEqual(source_calls, ["TRAIN-FIT", "TRAIN-CAL"])
        self.assertEqual(len(config_calls), 2)
        self.assertEqual(
            [(call["sequence_count"], call["seed_offset"]) for call in config_calls],
            [(128, 16_777_216), (64, 16_777_344)],
        )
        self.assertTrue(
            all(call["behavior_policy"] == "balanced_intervention_v1" for call in config_calls)
        )
        self.assertTrue(
            all(call["behavior_intervention_rate"] == 0.5 for call in config_calls)
        )

        contracts = diagnostic_partition_contracts()
        self.assertEqual(set(contracts), {"TRAIN-FIT", "TRAIN-CAL"})
        self.assertTrue(
            all(
                contract.dataset_config.split is DatasetSplit.TRAIN
                for contract in contracts.values()
            )
        )

    def test_pass_chain_binds_parent_order_and_optimizer_steps(self) -> None:
        records: list[dict[str, object]] = []
        before = EXACT_V3_PARENT_STATE_SHA256
        optimizer_before = "a" * 64
        permutation_digests = preregistered_permutation_digests()
        for added_pass in range(1, 17):
            after = f"{added_pass:064x}"
            optimizer_after = f"{added_pass + 200:064x}"
            records.append(
                {
                    "added_pass": added_pass,
                    "total_refinement_pass": 8 + added_pass,
                    "roots": 1_536,
                    "optimizer_steps": 64,
                    "optimizer_step_start": (added_pass - 1) * 64,
                    "optimizer_step_end": added_pass * 64,
                    "state_sha256_before": before,
                    "state_sha256_after": after,
                    "optimizer_state_sha256_before": optimizer_before,
                    "optimizer_state_sha256_after": optimizer_after,
                    "root_permutation_sha256": permutation_digests[added_pass - 1],
                }
            )
            before = after
            optimizer_before = optimizer_after
        audit = validate_pass_chain(
            records,
            initial_optimizer_state_sha256="a" * 64,
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["optimizer_step_end"], 1_024)
        self.assertTrue(audit["exact_preregistered_permutations"])

        drifted = copy.deepcopy(records)
        drifted[8]["state_sha256_before"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "continuity"):
            validate_pass_chain(
                drifted,
                initial_optimizer_state_sha256="a" * 64,
            )

        reordered = copy.deepcopy(records)
        reordered[3], reordered[4] = reordered[4], reordered[3]
        with self.assertRaisesRegex(ValueError, "continuity"):
            validate_pass_chain(
                reordered,
                initial_optimizer_state_sha256="a" * 64,
            )

        wrong_permutation = copy.deepcopy(records)
        wrong_permutation[6]["root_permutation_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "permutation"):
            validate_pass_chain(
                wrong_permutation,
                initial_optimizer_state_sha256="a" * 64,
            )

        wrong_optimizer = copy.deepcopy(records)
        wrong_optimizer[6]["optimizer_state_sha256_before"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "continuity"):
            validate_pass_chain(
                wrong_optimizer,
                initial_optimizer_state_sha256="a" * 64,
            )

    def test_optimizer_state_digest_is_deterministic_and_tracks_updates(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-3, weight_decay=1.0e-4)
        initial = _optimizer_state_sha256(optimizer)
        self.assertEqual(initial, _optimizer_state_sha256(optimizer))
        parameter.grad = torch.tensor([0.5])
        optimizer.step()
        self.assertNotEqual(initial, _optimizer_state_sha256(optimizer))

    def test_preregistered_interpretation_has_three_fail_closed_outcomes(self) -> None:
        initial = _milestone_pair(
            fit_bce=0.50,
            cal_bce=0.50,
            cal_auc=0.60,
            action_aucs=[0.60] * 5,
            shuffle_ratio=1.00,
        )
        head = _milestone_pair(
            fit_bce=0.47,
            cal_bce=0.48,
            cal_auc=0.63,
            action_aucs=[0.62] * 5,
            shuffle_ratio=1.03,
        )
        self.assertEqual(
            interpretation_report(initial, head)["conclusion"],
            "hazard_path_undertraining_supported",
        )

        representation = _milestone_pair(
            fit_bce=0.47,
            cal_bce=0.495,
            cal_auc=0.605,
            action_aucs=[0.603] * 5,
            shuffle_ratio=1.005,
        )
        self.assertEqual(
            interpretation_report(initial, representation)["conclusion"],
            "frozen_belief_generalization_limitation_supported",
        )

        inconclusive = _milestone_pair(
            fit_bce=0.497,
            cal_bce=0.499,
            cal_auc=0.606,
            action_aucs=[0.604] * 5,
            shuffle_ratio=1.005,
        )
        report = interpretation_report(initial, inconclusive)
        self.assertEqual(report["conclusion"], "inconclusive")
        self.assertTrue(report["fit_plateau_below_0_005"])
        self.assertEqual(
            report["scope_limitation"]["live_qualification_implication"],
            "none",
        )

    def test_create_only_preflight_and_failure_receipt_never_publish_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "diagnostic.json"
            preregistration_receipt = root / "invalid-preregistration.json"
            preregistration_receipt.write_text("{}", encoding="utf-8")
            checkpoint_paths = _checkpoint_paths(root / "checkpoints")
            _assert_targets_available(output, checkpoint_paths)

            occupied = checkpoint_paths[0]
            occupied.parent.mkdir(parents=True)
            occupied.write_bytes(b"existing diagnostic checkpoint")
            with self.assertRaises(FileExistsError):
                _assert_targets_available(output, checkpoint_paths)

            def fail() -> None:
                raise RuntimeError("synthetic diagnostic failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic diagnostic"):
                _run_with_failure_receipt(
                    upstream_result=root / "upstream.json",
                    preregistration_receipt_path=preregistration_receipt,
                    output=output,
                    checkpoint_paths=checkpoint_paths,
                    operation=fail,
                )
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["classification"],
                "diagnostic_run_failed_not_candidate",
            )
            self.assertFalse(receipt["qualification_claimed"])
            self.assertFalse(receipt["candidate_publication_allowed"])
            self.assertFalse(receipt["candidate_checkpoint"]["published"])
            self.assertFalse(receipt["scope_limitation"]["live_signature_matched"])
            self.assertEqual(
                receipt["preregistration_receipt"]["path"],
                str(preregistration_receipt.resolve()),
            )
            self.assertTrue(receipt["preregistration_receipt"]["present"])
            self.assertEqual(len(receipt["preregistration_receipt"]["sha256"]), 64)
            self.assertFalse(receipt["preregistration_receipt"]["validated"])

    def test_create_only_preregistration_receipt_binds_source_and_rejects_drift(self) -> None:
        bundle = {
            "schema_version": 1,
            "sha256": "1" * 64,
            "production_v3_source_bundle": {"sha256": EXACT_V3_SOURCE_BUNDLE_SHA256},
            "diagnostic_files": {
                "brain/scripts/v21i_frozen_representation_continuation_diagnostic_v1.py": "2" * 64,
                "brain/docs/preregistrations/2026-08-24-v21i-frozen-representation-continuation-diagnostic-v1.md": "3" * 64,
                "brain/tests/test_v21i_frozen_representation_continuation_diagnostic_v1.py": "4" * 64,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "preregistration.json"
            factory = lambda _root: bundle
            receipt_sha = register_preregistration(
                receipt_path,
                project_root=root,
                source_bundle_factory=factory,
            )
            record = validate_preregistration_receipt(
                receipt_path,
                project_root=root,
                source_bundle_factory=factory,
            )
            self.assertEqual(record["sha256"], receipt_sha)
            self.assertEqual(record["source_bundle"], bundle)
            with self.assertRaises(FileExistsError):
                register_preregistration(
                    receipt_path,
                    project_root=root,
                    source_bundle_factory=factory,
                )

            drifted = copy.deepcopy(bundle)
            drifted["diagnostic_files"][
                "brain/tests/test_v21i_frozen_representation_continuation_diagnostic_v1.py"
            ] = "5" * 64
            with self.assertRaisesRegex(ValueError, "source drifted"):
                validate_preregistration_receipt(
                    receipt_path,
                    project_root=root,
                    source_bundle_factory=lambda _root: drifted,
                )

    def test_current_diagnostic_source_bundle_is_bound_to_v3_production(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        bundle = _diagnostic_source_bundle(project_root)
        self.assertEqual(
            bundle["production_v3_source_bundle"]["sha256"],
            EXACT_V3_SOURCE_BUNDLE_SHA256,
        )
        self.assertIn(
            "brain/scripts/v21i_frozen_representation_continuation_diagnostic_v1.py",
            bundle["diagnostic_files"],
        )
        self.assertIn(
            "brain/docs/preregistrations/"
            "2026-08-24-v21i-frozen-representation-continuation-diagnostic-v1.md",
            bundle["diagnostic_files"],
        )
        self.assertIn(
            "brain/tests/test_v21i_frozen_representation_continuation_diagnostic_v1.py",
            bundle["diagnostic_files"],
        )


if __name__ == "__main__":
    unittest.main()
