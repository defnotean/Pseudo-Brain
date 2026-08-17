"""RCQ-v3 recipe options: deadzone hinge, structural squash, opposite-key penalty.

These tests pin three properties of the post-v2 recipe work:

1. Every historical configuration keeps its exact canonical identity while the
   new objective fields hold their defaults.
2. The deadzone hinge and opposite-key penalties are zero at quiescence,
   positive on the observed v2 failure modes, and gradient-connected.
3. The optional ``deadzone_tanh`` readout squash makes continuous deadzone
   violations structurally impossible without changing default behavior.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from irene_brain.training.batches import BUTTON_TARGET_INDICES
from irene_brain.training.config import ObjectiveConfig, load_training_config
from irene_brain.types import HidKey


BRAIN_ROOT = Path(__file__).resolve().parents[1]
RCQ_V2_CONFIG_SHA256 = (
    "b184361881b52189be78dce105ecc59ab52fa15551d63a2a7662a9dd06619366"
)

try:
    import torch
except ModuleNotFoundError:  # The stdlib-only play-safe runtime remains valid.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.model.actuator import (
        SQUASH_LIMIT,
        DirectActuatorReadout,
    )
    from irene_brain.model.spec import ActuatorQuerySpec
    from irene_brain.training.objective import _structured_action_loss


class RecipeFieldConfigurationTests(unittest.TestCase):
    def test_defaults_preserve_the_registered_rcq_v2_config_identity(self) -> None:
        config = load_training_config(
            BRAIN_ROOT / "configs" / "training" / "dgx-rcq-v2-reference.toml"
        )
        self.assertEqual(config.config_sha256, RCQ_V2_CONFIG_SHA256)
        canonical = config.canonical_json
        self.assertNotIn("continuous_deadzone_hinge_weight", canonical)
        self.assertNotIn("continuous_deadzone_hinge_margin", canonical)
        self.assertNotIn("opposite_key_pair_weight", canonical)
        self.assertNotIn("continuous_output_squash", canonical)
        for path in sorted((BRAIN_ROOT / "configs" / "training").glob("*.toml")):
            if path.stem == "dgx-rcq-v3-reference-candidate":
                continue
            loaded = load_training_config(path)
            self.assertEqual(
                loaded.objective.continuous_deadzone_hinge_weight, 0.0, path.name
            )
            self.assertEqual(
                loaded.objective.continuous_deadzone_hinge_margin, 0.04, path.name
            )
            self.assertEqual(loaded.objective.opposite_key_pair_weight, 0.0, path.name)
            self.assertEqual(loaded.objective.continuous_output_squash, "none", path.name)

    def test_v3_candidate_config_parses_with_its_review_values(self) -> None:
        candidate = load_training_config(
            BRAIN_ROOT / "configs" / "training" / "dgx-rcq-v3-reference-candidate.toml"
        )
        objective = candidate.objective
        self.assertEqual(objective.continuous_deadzone_hinge_weight, 0.5)
        self.assertEqual(objective.continuous_deadzone_hinge_margin, 0.04)
        self.assertEqual(objective.opposite_key_pair_weight, 0.25)
        self.assertEqual(objective.continuous_output_squash, "deadzone_tanh")
        self.assertNotEqual(candidate.config_sha256, RCQ_V2_CONFIG_SHA256)

    def test_non_default_recipe_fields_change_the_configuration_identity(self) -> None:
        config = load_training_config(
            BRAIN_ROOT / "configs" / "training" / "dgx-rcq-v2-reference.toml"
        )
        changed = replace(
            config,
            objective=replace(
                config.objective,
                continuous_deadzone_hinge_weight=0.5,
                opposite_key_pair_weight=0.2,
                continuous_output_squash="deadzone_tanh",
            ),
        )
        self.assertNotEqual(changed.config_sha256, RCQ_V2_CONFIG_SHA256)
        self.assertIn('"continuous_deadzone_hinge_weight":0.5', changed.canonical_json)
        self.assertIn('"opposite_key_pair_weight":0.2', changed.canonical_json)
        self.assertIn('"continuous_output_squash":"deadzone_tanh"', changed.canonical_json)

    def test_recipe_fields_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "deadzone"):
            ObjectiveConfig(continuous_deadzone_hinge_margin=0.05)
        with self.assertRaisesRegex(ValueError, "finite"):
            ObjectiveConfig(continuous_deadzone_hinge_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            ObjectiveConfig(opposite_key_pair_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "deadzone_tanh"):
            ObjectiveConfig(continuous_output_squash="clamp")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ObjectiveConfig(continuous_output_squash="")


@unittest.skipUnless(torch is not None, "PyTorch is not installed in play-safe runtime")
class RecipeLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    @staticmethod
    def _target(batch_size: int = 2) -> "torch.Tensor":
        assert torch is not None
        return torch.zeros(batch_size, 307)

    @staticmethod
    def _prediction(
        batch_size: int = 2,
        *,
        key_logits: dict[int, float] | None = None,
        continuous: float = 0.0,
    ) -> SimpleNamespace:
        assert torch is not None
        button_logits = torch.full((batch_size, len(BUTTON_TARGET_INDICES)), -8.0)
        for index, value in (key_logits or {}).items():
            button_logits[:, index] = value
        return SimpleNamespace(
            button_logits=button_logits,
            mouse_mean=torch.full((batch_size, 2), continuous),
            scroll_mean=torch.full((batch_size, 1), continuous),
            gamepad_axis_mean=torch.full((batch_size, 8), continuous),
        )

    def _loss(self, prediction: SimpleNamespace, **kwargs: object) -> "torch.Tensor":
        assert torch is not None
        target = self._target(prediction.button_logits.shape[0])
        button_indices = torch.tensor(BUTTON_TARGET_INDICES, dtype=torch.long)
        return _structured_action_loss(prediction, target, button_indices, **kwargs)

    def test_default_call_matches_explicit_zero_recipe_weights(self) -> None:
        prediction = self._prediction(continuous=0.2)
        baseline = self._loss(prediction)
        movement = torch.tensor(
            [int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)],
            dtype=torch.long,
        )
        explicit = self._loss(
            prediction,
            deadzone_hinge_weight=0.0,
            opposite_pair_weight=0.0,
            movement_key_indices=movement,
        )
        self.assertTrue(torch.equal(baseline, explicit))

    def test_deadzone_hinge_is_zero_inside_margin_and_positive_outside(self) -> None:
        quiet = self._loss(
            self._prediction(continuous=0.03),
            deadzone_hinge_weight=1.0,
        )
        loud = self._loss(
            self._prediction(continuous=0.2),
            deadzone_hinge_weight=1.0,
        )
        baseline = self._loss(self._prediction(continuous=0.2))
        self.assertTrue(torch.equal(quiet, self._loss(self._prediction(continuous=0.03))))
        self.assertGreater((loud - baseline).item(), 0.0)
        expected_hinge = (0.2 - 0.04) * 1.0
        self.assertAlmostEqual((loud - baseline).item(), expected_hinge, places=5)

    def test_deadzone_hinge_backpropagates_to_continuous_outputs(self) -> None:
        prediction = self._prediction(continuous=0.2)
        prediction.mouse_mean.requires_grad_(True)
        loss = self._loss(prediction, deadzone_hinge_weight=1.0)
        loss.backward()
        self.assertIsNotNone(prediction.mouse_mean.grad)
        self.assertGreater(prediction.mouse_mean.grad.abs().sum().item(), 0.0)

    def test_opposite_pair_penalty_targets_coactivation_only(self) -> None:
        movement = torch.tensor(
            [int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)],
            dtype=torch.long,
        )
        w_only = self._prediction(key_logits={int(HidKey.W): 8.0})
        both = self._prediction(
            key_logits={int(HidKey.W): 8.0, int(HidKey.S): 8.0}
        )
        loss_w_only = self._loss(
            w_only, opposite_pair_weight=1.0, movement_key_indices=movement
        )
        loss_both = self._loss(
            both, opposite_pair_weight=1.0, movement_key_indices=movement
        )
        base_w = self._loss(w_only)
        base_both = self._loss(both)
        self.assertAlmostEqual((loss_w_only - base_w).item(), 0.0, places=3)
        self.assertGreater((loss_both - base_both).item(), 0.9)

    def test_opposite_pair_penalty_requires_movement_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "movement_key_indices"):
            self._loss(self._prediction(), opposite_pair_weight=1.0)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in play-safe runtime")
class DeadzoneSquashReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    @staticmethod
    def _readout(squash: str) -> "DirectActuatorReadout":
        assert torch is not None
        return DirectActuatorReadout(
            width=8,
            heads=2,
            actuator=ActuatorQuerySpec(continuous_squash=squash),
            plan_steps=2,
        )

    @staticmethod
    def _state(batch_size: int = 2) -> dict[str, "torch.Tensor"]:
        assert torch is not None
        generator = torch.Generator().manual_seed(1702)
        return {
            "sensors": torch.randn(batch_size, 4, 8, generator=generator),
            "belief": torch.randn(batch_size, 2, 8, generator=generator),
            "thoughts": torch.randn(batch_size, 2, 3, 8, generator=generator),
            "working_memory": torch.randn(batch_size, 2, 8, generator=generator),
            "retrieved_memory": torch.randn(batch_size, 2, 2, 8, generator=generator),
            "goal_context": torch.randn(batch_size, 1, 8, generator=generator),
        }

    def test_squash_keeps_every_continuous_channel_inside_the_deadzone(self) -> None:
        assert torch is not None
        readout = self._readout("deadzone_tanh")
        with torch.no_grad():
            for parameter in readout.parameters():
                parameter.mul_(40.0)
            prediction = readout(**self._state())
        limit = SQUASH_LIMIT
        for name in ("mouse_mean", "scroll_mean", "gamepad_axis_mean"):
            self.assertLessEqual(getattr(prediction, name).abs().max().item(), limit, name)
        control = prediction.control
        continuous = torch.cat(
            (control[:, 264:267], control[:, 299:307]), dim=1
        )
        self.assertLessEqual(continuous.abs().max().item(), limit)
        self.assertLess(continuous.abs().max().item(), 0.05)
        bf16_readout = self._readout("deadzone_tanh").to(torch.bfloat16)
        with torch.no_grad():
            for parameter in bf16_readout.parameters():
                parameter.mul_(40.0)
            bf16_state = {
                name: value.to(torch.bfloat16)
                for name, value in self._state().items()
            }
            bf16_prediction = bf16_readout(**bf16_state)
        bf16_continuous = torch.cat(
            (
                bf16_prediction.control[:, 264:267],
                bf16_prediction.control[:, 299:307],
            ),
            dim=1,
        ).float()
        self.assertLess(bf16_continuous.abs().max().item(), 0.05)

    def test_default_readout_behavior_is_unchanged(self) -> None:
        assert torch is not None
        state = self._state()
        default = self._readout("none")
        with torch.no_grad():
            default_prediction = default(**state)
        raw_mouse = default_prediction.mouse_mean
        self.assertTrue(
            torch.equal(raw_mouse, default_prediction.control[:, 264:266])
        )
        self.assertTrue(
            torch.equal(
                default_prediction.gamepad_axis_mean,
                default_prediction.control[:, 299:307],
            )
        )

    def test_actuator_spec_validates_the_squash_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "deadzone_tanh"):
            ActuatorQuerySpec(continuous_squash="clamp")


if __name__ == "__main__":
    unittest.main()
