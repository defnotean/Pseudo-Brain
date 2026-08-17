from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from irene_brain.training.batches import BUTTON_TARGET_INDICES, CONTROL_VECTOR_SIZE
from irene_brain.training.config import ObjectiveConfig, load_training_config
from irene_brain.types import HidKey


BRAIN_ROOT = Path(__file__).resolve().parents[1]
MOVEMENT_CONTROL_INDICES = (4, 7, 22, 26)

try:
    import torch
except ModuleNotFoundError:  # The stdlib-only play-safe runtime remains valid.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.training.objective import _structured_action_loss


class CalibratedActionConfigurationTests(unittest.TestCase):
    def test_support_mapping_and_every_toml_field_are_strict_and_hashed(self) -> None:
        self.assertEqual(
            tuple(int(key) for key in (HidKey.A, HidKey.D, HidKey.S, HidKey.W)),
            MOVEMENT_CONTROL_INDICES,
        )
        self.assertEqual(
            tuple(BUTTON_TARGET_INDICES.index(index) for index in MOVEMENT_CONTROL_INDICES),
            MOVEMENT_CONTROL_INDICES,
        )

        config_dir = BRAIN_ROOT / "configs" / "training"
        configs = {
            path.stem: load_training_config(path)
            for path in sorted(config_dir.glob("*.toml"))
        }
        self.assertEqual(
            configs["dgx-stagea-action-overfit"].objective.action_loss_kind,
            "sparse_hard_negative_v1",
        )
        self.assertEqual(
            configs["dgx-stagea-action-overfit-b"].objective.action_loss_kind,
            "support_aware_calibrated_v1",
        )
        a_record = configs["dgx-stagea-action-overfit"].to_dict()
        b_record = configs["dgx-stagea-action-overfit-b"].to_dict()
        b_record["run"]["name"] = a_record["run"]["name"]
        b_record["objective"]["action_loss_kind"] = a_record["objective"][
            "action_loss_kind"
        ]
        self.assertEqual(b_record, a_record)
        for name, config in configs.items():
            objective = config.objective
            self.assertEqual(
                objective.button_support_control_indices,
                MOVEMENT_CONTROL_INDICES,
                name,
            )
            self.assertEqual(objective.button_support_weight, 0.8, name)
            self.assertEqual(objective.button_background_weight, 0.2, name)
            self.assertEqual(objective.button_background_tail_mix, 0.9, name)
            self.assertEqual(
                objective.button_background_tail_temperature,
                0.1,
                name,
            )
            self.assertEqual(objective.continuous_action_weight, 0.25, name)
            self.assertEqual(
                config.to_dict()["objective"]["button_support_control_indices"],
                [4, 7, 22, 26],
                name,
            )

        smoke = configs["dgx-smoke"]
        changed = replace(
            smoke,
            objective=replace(
                smoke.objective,
                button_background_tail_temperature=0.2,
            ),
        )
        self.assertNotEqual(changed.config_sha256, smoke.config_sha256)
        self.assertIn('"button_background_tail_temperature":0.1', smoke.canonical_json)

    def test_objective_configuration_rejects_ambiguous_group_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            ObjectiveConfig(button_support_control_indices=(7, 4, 22, 26))
        with self.assertRaisesRegex(ValueError, "button channels"):
            ObjectiveConfig(button_support_control_indices=(4, 7, 22, 264))
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            ObjectiveConfig(button_support_weight=0.7)
        with self.assertRaisesRegex(ValueError, "positive"):
            ObjectiveConfig(button_background_tail_temperature=0.0)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in play-safe runtime")
class CalibratedActionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    @staticmethod
    def prediction(button_logits: object) -> SimpleNamespace:
        assert torch is not None
        batch_size = button_logits.shape[0]
        return SimpleNamespace(
            button_logits=button_logits,
            mouse_mean=torch.zeros(batch_size, 2),
            scroll_mean=torch.zeros(batch_size, 1),
            gamepad_axis_mean=torch.zeros(batch_size, 8),
        )

    @staticmethod
    def loss(
        button_logits: object,
        target: object,
        *,
        button_indices: object | None = None,
        loss_kind: str = "support_aware_calibrated_v1",
    ) -> object:
        assert torch is not None
        if button_indices is None:
            button_indices = torch.tensor(BUTTON_TARGET_INDICES, dtype=torch.long)
        return _structured_action_loss(
            CalibratedActionLossTests.prediction(button_logits),
            target,
            button_indices,
            loss_kind=loss_kind,
            support_control_indices=torch.tensor(
                MOVEMENT_CONTROL_INDICES,
                dtype=torch.long,
            ),
        )

    def test_zero_logits_have_calibrated_log_two_button_baseline(self) -> None:
        assert torch is not None
        target = torch.zeros(3, CONTROL_VECTOR_SIZE)
        logits = torch.zeros(3, len(BUTTON_TARGET_INDICES))
        loss = self.loss(logits, target)
        self.assertAlmostEqual(float(loss), math.log(2.0), places=6)

    def test_exact_beats_all_off_and_all_four_by_more_than_three(self) -> None:
        assert torch is not None
        target = torch.zeros(2, CONTROL_VECTOR_SIZE)
        target[0, [int(HidKey.W), int(HidKey.D)]] = 1.0
        target[1, [int(HidKey.A), int(HidKey.S)]] = 1.0
        exact = torch.full((2, len(BUTTON_TARGET_INDICES)), -8.0)
        exact[0, [int(HidKey.W), int(HidKey.D)]] = 8.0
        exact[1, [int(HidKey.A), int(HidKey.S)]] = 8.0
        all_off = torch.full_like(exact, -8.0)
        all_four = torch.full_like(exact, -8.0)
        all_four[:, list(MOVEMENT_CONTROL_INDICES)] = 8.0

        exact_loss = self.loss(exact, target)
        all_off_loss = self.loss(all_off, target)
        all_four_loss = self.loss(all_four, target)
        self.assertLess(float(exact_loss), 0.001)
        self.assertGreater(float(all_off_loss), float(exact_loss) + 3.0)
        self.assertGreater(float(all_four_loss), float(exact_loss) + 3.0)

    def test_reordered_mapping_gives_every_movement_logit_the_correct_gradient(self) -> None:
        assert torch is not None
        button_indices = torch.tensor(tuple(reversed(BUTTON_TARGET_INDICES)))
        target = torch.zeros(1, CONTROL_VECTOR_SIZE)
        target[0, [int(HidKey.A), int(HidKey.S)]] = 1.0
        logits = torch.zeros(1, len(BUTTON_TARGET_INDICES), requires_grad=True)
        self.loss(logits, target, button_indices=button_indices).backward()

        assert logits.grad is not None
        for control_index in MOVEMENT_CONTROL_INDICES:
            output_index = int((button_indices == control_index).nonzero()[0, 0])
            gradient = float(logits.grad[0, output_index])
            expected = -0.1 if target[0, control_index] > 0.5 else 0.1
            self.assertAlmostEqual(gradient, expected, places=6)

    def test_background_offender_is_nondiluting_and_ties_are_continuous(self) -> None:
        assert torch is not None
        target = torch.zeros(1, CONTROL_VECTOR_SIZE)

        full_logits = torch.full(
            (1, len(BUTTON_TARGET_INDICES)),
            -8.0,
            requires_grad=True,
        )
        with torch.no_grad():
            full_logits[0, 30] = 8.0
        self.loss(full_logits, target).backward()
        assert full_logits.grad is not None
        full_gradient = float(full_logits.grad[0, 30])

        small_indices = torch.tensor((*MOVEMENT_CONTROL_INDICES, 30), dtype=torch.long)
        small_logits = torch.full((1, 5), -8.0, requires_grad=True)
        with torch.no_grad():
            small_logits[0, 4] = 8.0
        self.loss(small_logits, target, button_indices=small_indices).backward()
        assert small_logits.grad is not None
        small_gradient = float(small_logits.grad[0, 4])
        self.assertGreater(full_gradient, 0.17)
        self.assertGreater(full_gradient / small_gradient, 0.85)

        tied = torch.full(
            (1, len(BUTTON_TARGET_INDICES)),
            -8.0,
            requires_grad=True,
        )
        with torch.no_grad():
            tied[0, 30:32] = 8.0
        self.loss(tied, target).backward()
        assert tied.grad is not None
        self.assertGreater(float(tied.grad[0, 30]), 0.08)
        self.assertAlmostEqual(
            float(tied.grad[0, 30]),
            float(tied.grad[0, 31]),
            places=7,
        )

        perturbed = torch.full(
            (1, len(BUTTON_TARGET_INDICES)),
            -8.0,
            requires_grad=True,
        )
        with torch.no_grad():
            perturbed[0, 30] = 8.0001
            perturbed[0, 31] = 8.0
        self.loss(perturbed, target).backward()
        assert perturbed.grad is not None
        self.assertLess(
            abs(float(perturbed.grad[0, 30] - tied.grad[0, 30])),
            0.001,
        )
        self.assertGreater(float(perturbed.grad[0, 31]), 0.08)

    def test_unexpected_positives_join_support_and_empty_background_is_finite(self) -> None:
        assert torch is not None
        unexpected_control_index = 42
        target = torch.zeros(1, CONTROL_VECTOR_SIZE)
        target[0, unexpected_control_index] = 1.0
        logits = torch.zeros(
            1,
            len(BUTTON_TARGET_INDICES),
            requires_grad=True,
        )
        self.loss(logits, target).backward()
        assert logits.grad is not None
        self.assertLess(float(logits.grad[0, unexpected_control_index]), 0.0)
        self.assertAlmostEqual(
            float(logits.grad[0, unexpected_control_index]),
            -0.08,
            places=6,
        )

        all_positive_target = torch.zeros(1, CONTROL_VECTOR_SIZE)
        all_positive_target[:, list(BUTTON_TARGET_INDICES)] = 1.0
        all_positive_logits = torch.full(
            (1, len(BUTTON_TARGET_INDICES)),
            8.0,
            requires_grad=True,
        )
        empty_background_loss = self.loss(all_positive_logits, all_positive_target)
        self.assertTrue(bool(torch.isfinite(empty_background_loss)))
        empty_background_loss.backward()
        assert all_positive_logits.grad is not None
        self.assertTrue(bool(torch.isfinite(all_positive_logits.grad).all()))
        self.assertTrue(bool((all_positive_logits.grad < 0.0).all()))

    def test_legacy_a_loss_remains_explicit_and_continuous_weight_is_quarter(self) -> None:
        assert torch is not None
        target = torch.zeros(1, CONTROL_VECTOR_SIZE)
        target[0, [int(HidKey.W), int(HidKey.D)]] = 1.0
        zero_logits = torch.zeros(1, len(BUTTON_TARGET_INDICES))
        calibrated = self.loss(zero_logits, target)
        legacy = self.loss(
            zero_logits,
            target,
            loss_kind="sparse_hard_negative_v1",
        )
        self.assertAlmostEqual(float(calibrated), math.log(2.0), places=6)
        self.assertAlmostEqual(float(legacy), 1.05 * math.log(2.0), places=6)

        exact_logits = torch.full_like(zero_logits, -8.0)
        exact_logits[0, [int(HidKey.W), int(HidKey.D)]] = 8.0
        baseline = self.loss(exact_logits, target)
        prediction = self.prediction(exact_logits)
        prediction.mouse_mean[0, 0] = 1.0
        with_mouse_error = _structured_action_loss(
            prediction,
            target,
            torch.tensor(BUTTON_TARGET_INDICES, dtype=torch.long),
            support_control_indices=torch.tensor(MOVEMENT_CONTROL_INDICES),
        )
        self.assertAlmostEqual(
            float(with_mouse_error - baseline),
            0.125,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
