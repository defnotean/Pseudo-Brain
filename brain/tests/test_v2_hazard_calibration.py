"""CPU contracts for leakage-resistant per-action hazard calibration."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F

from irene_brain.v2.hazard_calibration import (
    PerActionAffineHazardCalibrator,
    PerActionCalibrationReport,
    TrainCalibrationProvenance,
    fit_train_only_per_action_affine,
)


def _underpredicted_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    logits: list[float] = []
    targets: list[float] = []
    actions: list[int] = []
    groups: list[str] = []
    action_biases = (-0.5, -0.2, 0.1, 0.4, 0.7)
    levels = torch.linspace(-2.5, 2.5, 21, dtype=torch.float64)
    repeats = 20
    for action_id, action_bias in enumerate(action_biases):
        for level_index, level in enumerate(levels.tolist()):
            true_probability = float(torch.sigmoid(torch.tensor(1.3 * level + action_bias)))
            positive_count = round(true_probability * repeats)
            for repeat in range(repeats):
                # Raw predictions preserve the signal but are shifted downward
                # and compressed, matching the seed-43 failure mode.
                logits.append(0.55 * level - 1.0)
                targets.append(float(repeat < positive_count))
                actions.append(action_id)
                groups.append(f"train-cal-{action_id}-{level_index}-{repeat}")
    return (
        torch.tensor(logits, dtype=torch.float64),
        torch.tensor(targets, dtype=torch.float64),
        torch.tensor(actions, dtype=torch.long),
        tuple(groups),
    )


class HazardCalibrationContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _provenance(self, groups: tuple[str, ...]) -> TrainCalibrationProvenance:
        return TrainCalibrationProvenance(
            source_namespace="maze-v21-train-seed-offset-8388608",
            source_split="TRAIN",
            source_partition="TRAIN-CAL",
            calibration_group_ids=groups,
            upstream_model_fit_group_ids=tuple(f"train-fit-{index}" for index in range(100)),
            upstream_checkpoint_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            source_bundle_sha256="c" * 64,
            partition_algorithm="manifest-hash-whole-episode-v1",
        )

    def test_corrects_underprediction_and_preserves_within_action_ranking(self) -> None:
        logits, targets, actions, groups = _underpredicted_fixture()
        calibrator = fit_train_only_per_action_affine(
            logits,
            targets,
            actions,
            provenance=self._provenance(groups),
            action_count=5,
        )
        calibrated = calibrator.transform(logits, actions)
        before_bce = F.binary_cross_entropy_with_logits(logits, targets)
        after_bce = F.binary_cross_entropy_with_logits(calibrated, targets)
        before_bias = (torch.sigmoid(logits) - targets).mean().abs()
        after_bias = (torch.sigmoid(calibrated) - targets).mean().abs()

        self.assertLess(float(after_bce), float(before_bce) - 0.05)
        self.assertLess(float(after_bias), float(before_bias) * 0.1)
        self.assertTrue(all(scale > 0.0 for scale in calibrator.scales))
        for action_id in range(5):
            mask = actions.eq(action_id)
            self.assertTrue(
                torch.equal(
                    torch.argsort(logits[mask], stable=True),
                    torch.argsort(calibrated[mask], stable=True),
                )
            )
            self.assertLessEqual(
                calibrator.per_action[action_id].after_bce,
                calibrator.per_action[action_id].before_bce,
            )

    def test_bias_only_mode_fixes_level_without_changing_any_action_scale(self) -> None:
        logits, targets, actions, groups = _underpredicted_fixture()
        calibrator = fit_train_only_per_action_affine(
            logits,
            targets,
            actions,
            provenance=self._provenance(groups),
            action_count=5,
            fit_mode="bias_only",
        )
        calibrated = calibrator.transform(logits, actions)

        self.assertEqual(calibrator.fit_mode, "bias_only")
        self.assertEqual(calibrator.scales, (1.0,) * 5)
        self.assertLess(
            float(F.binary_cross_entropy_with_logits(calibrated, targets)),
            float(F.binary_cross_entropy_with_logits(logits, targets)),
        )
        for action_id in range(5):
            mask = actions.eq(action_id)
            self.assertAlmostEqual(
                float(torch.sigmoid(calibrated[mask]).mean()),
                float(targets[mask].mean()),
                places=8,
            )

    def test_no_strict_train_cal_improvement_retains_identity(self) -> None:
        actions = torch.arange(5).repeat_interleave(20)
        targets = torch.tensor(([0.0] * 10 + [1.0] * 10) * 5, dtype=torch.float64)
        logits = torch.zeros(100, dtype=torch.float64)
        groups = tuple(f"train-cal-{index}" for index in range(100))
        calibrator = fit_train_only_per_action_affine(
            logits,
            targets,
            actions,
            provenance=self._provenance(groups),
            action_count=5,
            fit_mode="bias_only",
        )

        self.assertFalse(calibrator.accepted)
        self.assertEqual(calibrator.scales, (1.0,) * 5)
        self.assertEqual(calibrator.biases, (0.0,) * 5)
        self.assertEqual(calibrator.before_bce, calibrator.after_bce)
        self.assertTrue(all(not report.accepted for report in calibrator.per_action))

    def test_refuses_non_train_or_overlapping_fit_groups(self) -> None:
        logits, targets, actions, groups = _underpredicted_fixture()
        validation = TrainCalibrationProvenance(
            source_namespace="maze-v21-validation",
            source_split="VALIDATION",
            source_partition="TRAIN-CAL",
            calibration_group_ids=groups,
            upstream_model_fit_group_ids=("train-fit-0",),
            upstream_checkpoint_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            source_bundle_sha256="c" * 64,
            partition_algorithm="manifest-hash-whole-episode-v1",
        )
        with self.assertRaisesRegex(ValueError, "only consume the TRAIN split"):
            fit_train_only_per_action_affine(
                logits,
                targets,
                actions,
                provenance=validation,
                action_count=5,
            )

        overlap = TrainCalibrationProvenance(
            source_namespace="maze-v21-train",
            source_split="TRAIN",
            source_partition="TRAIN-CAL",
            calibration_group_ids=groups,
            upstream_model_fit_group_ids=(groups[0],),
            upstream_checkpoint_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            source_bundle_sha256="c" * 64,
            partition_algorithm="manifest-hash-whole-episode-v1",
        )
        with self.assertRaisesRegex(ValueError, "overlap upstream model-fit"):
            fit_train_only_per_action_affine(
                logits,
                targets,
                actions,
                provenance=overlap,
                action_count=5,
            )

    def test_fit_is_deterministic_and_export_round_trips(self) -> None:
        logits, targets, actions, groups = _underpredicted_fixture()
        provenance = self._provenance(groups)
        first = fit_train_only_per_action_affine(
            logits,
            targets,
            actions,
            provenance=provenance,
            action_count=5,
        )
        second = fit_train_only_per_action_affine(
            logits,
            targets,
            actions,
            provenance=provenance,
            action_count=5,
        )

        self.assertEqual(first.scales, second.scales)
        self.assertEqual(first.biases, second.biases)
        payload = first.export_parameters()
        restored = PerActionAffineHazardCalibrator.from_export_parameters(payload)
        self.assertEqual(first, restored)
        self.assertEqual(payload["fitted_on"]["source_split"], "TRAIN")
        self.assertNotIn(groups[0], str(payload))
        self.assertTrue(
            torch.equal(
                first.transform(logits, actions),
                restored.transform(logits, actions),
            )
        )

    def test_dense_action_table_uses_exportable_parameters(self) -> None:
        logits, targets, actions, groups = _underpredicted_fixture()
        calibrator = fit_train_only_per_action_affine(
            logits,
            targets,
            actions,
            provenance=self._provenance(groups),
            action_count=5,
        )
        table = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]], dtype=torch.float32)
        action_ids = torch.tensor([[4, 2, 0, 3, 1]])
        transformed = calibrator.transform_all_actions(table, action_ids)
        expected = (
            table * torch.tensor(calibrator.scales)[action_ids]
            + torch.tensor(calibrator.biases)[action_ids]
        )

        self.assertTrue(torch.allclose(transformed, expected))
        self.assertEqual(tuple(transformed.shape), (1, 5))
        head_native = table.unsqueeze(-1)
        self.assertTrue(
            torch.allclose(
                calibrator.transform_all_actions(head_native, action_ids),
                expected.unsqueeze(-1),
            )
        )
        with self.assertRaisesRegex(ValueError, "every semantic action once"):
            calibrator.transform_all_actions(
                table,
                torch.tensor([[4, 2, 0, 1, 1]]),
            )

    def test_nonfinite_evidence_and_fractional_action_ids_fail_closed(self) -> None:
        logits, targets, actions, groups = _underpredicted_fixture()
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            fit_train_only_per_action_affine(
                logits,
                targets,
                actions.to(torch.float64) + 0.9,
                provenance=self._provenance(groups),
                action_count=5,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            PerActionAffineHazardCalibrator(
                scales=(float("nan"),) * 5,
                biases=(0.0,) * 5,
                source_namespace="train-v21i",
                source_split="TRAIN",
                source_partition="TRAIN-CAL",
                calibration_group_digest="a" * 64,
                upstream_model_fit_group_digest="b" * 64,
                upstream_checkpoint_sha256="c" * 64,
                dataset_manifest_sha256="d" * 64,
                source_bundle_sha256="e" * 64,
                partition_algorithm="whole-episode-v1",
                calibration_groups=20,
                observations=100,
                before_bce=0.7,
                after_bce=0.6,
                per_action=tuple(
                    PerActionCalibrationReport(
                        action_id=action_id,
                        observations=20,
                        positives=10,
                        negatives=10,
                        before_bce=0.7,
                        after_bce=0.6,
                    )
                    for action_id in range(5)
                ),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_train_only_per_action_affine(
                logits,
                targets,
                actions,
                provenance=self._provenance(groups),
                action_count=5,
                l2_regularization=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
