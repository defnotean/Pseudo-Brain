"""CPU contracts for the standalone vectorized Core V2 outcome model."""
from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F

from irene_brain.v2.config import CoreV2Config
from irene_brain.v2.hazard_calibration import (
    PerActionAffineHazardCalibrator,
    PerActionCalibrationReport,
)
from irene_brain.v2.losses import consequence_losses
from irene_brain.v2.outcome_model import ActionOutcomeTable, VectorizedOutcomeModelV2


class VectorizedOutcomeModelContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = CoreV2Config()
        self.model = VectorizedOutcomeModelV2(self.config)
        self.context = torch.randn(3, self.config.width)

    def test_exhaustive_table_shapes(self) -> None:
        table = self.model(self.context)

        self.assertEqual(tuple(table.action_ids.shape), (3, self.config.actions))
        self.assertEqual(
            tuple(table.predicted_next_latent.shape),
            (3, self.config.actions, self.config.width),
        )
        self.assertEqual(tuple(table.predicted_reward.shape), (3, self.config.actions, 1))
        self.assertEqual(tuple(table.predicted_hazard.shape), (3, self.config.actions, 1))

    def test_gather_is_exact_semantic_table_lookup(self) -> None:
        table = self.model(self.context)
        factual_actions = torch.tensor([4, 0, 2])
        factual = table.gather(factual_actions)
        batch = torch.arange(3)

        self.assertTrue(
            torch.equal(
                factual.predicted_next_latent,
                table.predicted_next_latent[batch, factual_actions],
            )
        )
        self.assertTrue(
            torch.equal(
                factual.predicted_reward,
                table.predicted_reward[batch, factual_actions],
            )
        )
        self.assertTrue(
            torch.equal(
                factual.predicted_hazard,
                table.predicted_hazard[batch, factual_actions],
            )
        )

    def test_reordered_columns_preserve_action_identity(self) -> None:
        canonical = self.model(self.context)
        order = torch.tensor([3, 1, 4, 0, 2])
        reordered = self.model(self.context, order)

        for action_id in range(self.config.actions):
            action = torch.full((3,), action_id, dtype=torch.long)
            canonical_factual = canonical.gather(action)
            reordered_factual = reordered.gather(action)
            self.assertTrue(
                torch.equal(
                    canonical_factual.predicted_next_latent,
                    reordered_factual.predicted_next_latent,
                )
            )
            self.assertTrue(
                torch.equal(
                    canonical_factual.predicted_reward,
                    reordered_factual.predicted_reward,
                )
            )
            self.assertTrue(
                torch.equal(
                    canonical_factual.predicted_hazard,
                    reordered_factual.predicted_hazard,
                )
            )

    def test_all_outcome_losses_reach_shared_and_action_parameters(self) -> None:
        table = self.model(self.context)
        loss = (
            table.predicted_next_latent.square().mean()
            + table.predicted_reward.square().mean()
            + table.predicted_hazard.square().mean()
        )
        loss.backward()

        parameters = (
            self.model.state_trunk[0].weight,
            self.model.action_embedding.weight,
            self.model.outcome_trunk[0].weight,
            self.model.next_latent_head.weight,
            self.model.reward_head.weight,
            self.model.hazard_head.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_hazard_outputs_are_probabilities(self) -> None:
        hazard = self.model(self.context).predicted_hazard

        self.assertTrue(bool(torch.isfinite(hazard).all()))
        self.assertTrue(bool((hazard >= 0.0).all()))
        self.assertTrue(bool((hazard <= 1.0).all()))

    def test_train_calibration_is_baked_by_semantic_action_id(self) -> None:
        config = CoreV2Config(
            outcome_architecture="all_action_table_v1",
            hazard_parameterization="probability_sigmoid_v1",
            hazard_outcome_path="dedicated_stopgrad_v1",
        )
        model = VectorizedOutcomeModelV2(config)
        context = torch.randn(2, config.width)
        ids = torch.tensor([[4, 2, 0, 3, 1], [1, 0, 4, 2, 3]])
        identity = model(context, ids)
        before_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        reports = tuple(
            PerActionCalibrationReport(
                action_id=action_id,
                observations=20,
                positives=10,
                negatives=10,
                before_bce=0.7,
                after_bce=0.6,
            )
            for action_id in range(config.actions)
        )
        calibrator = PerActionAffineHazardCalibrator(
            scales=(1.0, 1.1, 1.2, 1.3, 1.4),
            biases=(-0.2, -0.1, 0.0, 0.1, 0.2),
            source_namespace="train-v21i",
            source_split="TRAIN",
            source_partition="TRAIN-CAL",
            calibration_group_digest="a" * 64,
            upstream_model_fit_group_digest="b" * 64,
            upstream_checkpoint_sha256="c" * 64,
            dataset_manifest_sha256="d" * 64,
            source_bundle_sha256="e" * 64,
            partition_algorithm="manifest-hash-whole-episode-v1",
            calibration_groups=20,
            observations=100,
            before_bce=0.7,
            after_bce=0.6,
            per_action=reports,
        )
        model.install_hazard_calibration(calibrator)
        calibrated = model(context, ids)
        expected_scale = torch.tensor(calibrator.scales)[ids].unsqueeze(-1)
        expected_bias = torch.tensor(calibrator.biases)[ids].unsqueeze(-1)

        self.assertTrue(
            torch.equal(identity.raw_hazard_logits, calibrated.raw_hazard_logits)
        )
        self.assertTrue(
            torch.allclose(
                calibrated.predicted_hazard_logits,
                calibrated.raw_hazard_logits * expected_scale + expected_bias,
            )
        )
        self.assertTrue(
            torch.allclose(
                calibrated.predicted_hazard,
                torch.sigmoid(calibrated.predicted_hazard_logits),
            )
        )
        self.assertFalse(
            torch.equal(identity.predicted_hazard, calibrated.predicted_hazard)
        )
        self.assertFalse(model.hazard_calibration_scale.requires_grad)
        self.assertFalse(model.hazard_calibration_bias.requires_grad)
        changed = {
            name
            for name, value in model.state_dict().items()
            if not torch.equal(value, before_state[name])
        }
        self.assertEqual(
            changed,
            {"hazard_calibration_scale", "hazard_calibration_bias"},
        )
        restored = VectorizedOutcomeModelV2(config)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored_table = restored(context, ids)
        self.assertTrue(
            torch.equal(
                restored_table.predicted_hazard,
                calibrated.predicted_hazard,
            )
        )

    def test_non_exhaustive_or_duplicate_action_table_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "permutation"):
            self.model(self.context, torch.tensor([0, 1, 2, 3, 3]))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            self.model(self.context, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.9]))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            self.model(self.context).gather(torch.tensor([0.0, 1.0, 2.9]))

    def test_pre_calibration_checkpoint_loads_with_identity_buffers(self) -> None:
        legacy_state = {
            name: value
            for name, value in self.model.state_dict().items()
            if name
            not in {"hazard_calibration_scale", "hazard_calibration_bias"}
        }
        restored = VectorizedOutcomeModelV2(self.config)
        restored.load_state_dict(legacy_state, strict=True)

        self.assertTrue(torch.equal(restored.hazard_calibration_scale, torch.ones(5)))
        self.assertTrue(torch.equal(restored.hazard_calibration_bias, torch.zeros(5)))

        partial_state = dict(self.model.state_dict())
        partial_state.pop("hazard_calibration_bias")
        with self.assertRaisesRegex(RuntimeError, "hazard_calibration_bias"):
            VectorizedOutcomeModelV2(self.config).load_state_dict(
                partial_state,
                strict=True,
            )

    def test_all_action_losses_are_row_order_invariant_means(self) -> None:
        table = self.model(self.context)
        reward = torch.randn(3, self.config.actions)
        hazard = torch.randint(
            0,
            2,
            (3, self.config.actions, 1),
            dtype=torch.float32,
        )
        next_latent = torch.randn(
            3,
            self.config.actions,
            self.config.width,
        )

        def losses(
            selected_table: ActionOutcomeTable,
            selected_reward: torch.Tensor,
            selected_hazard: torch.Tensor,
            selected_next: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            step = SimpleNamespace(
                hypotheses=SimpleNamespace(),
                pending_prediction=selected_table.gather(
                    torch.tensor([0, 1, 2])
                ),
                outcome_table=selected_table,
            )
            return consequence_losses(
                step,
                actual_reward=torch.zeros(3, 1),
                actual_hazard=torch.zeros(3, 1),
                next_latent=torch.zeros(3, self.config.width),
                hazard_parameterization="probability_sigmoid_v1",
                latent_comparison="cosine_distance_v1",
                reward_comparison="raw_mse_v0",
                outcome_architecture="all_action_table_v1",
                all_action_reward=selected_reward,
                all_action_hazard=selected_hazard,
                all_action_next_latent=selected_next,
            )

        canonical = losses(table, reward, hazard, next_latent)
        self.assertTrue(
            torch.allclose(
                canonical["reward"],
                F.mse_loss(table.predicted_reward, reward.unsqueeze(-1)),
            )
        )
        self.assertTrue(
            torch.allclose(
                canonical["hazard"],
                F.binary_cross_entropy(table.predicted_hazard, hazard),
            )
        )
        self.assertTrue(
            torch.allclose(
                canonical["next_latent"],
                1.0
                - F.cosine_similarity(
                    table.predicted_next_latent,
                    next_latent,
                    dim=-1,
                ).mean(),
            )
        )

        order = torch.tensor([3, 1, 4, 0, 2])
        permuted = ActionOutcomeTable(
            action_ids=table.action_ids[:, order],
            predicted_next_latent=table.predicted_next_latent[:, order],
            predicted_reward=table.predicted_reward[:, order],
            predicted_reward_logits=None,
            predicted_hazard=table.predicted_hazard[:, order],
        )
        reordered = losses(
            permuted,
            reward[:, order],
            hazard[:, order],
            next_latent[:, order],
        )
        for name in canonical:
            self.assertTrue(torch.allclose(canonical[name], reordered[name]))

    def test_dedicated_hazard_path_is_nonlinear_and_context_gradient_isolated(self) -> None:
        config = CoreV2Config(
            outcome_architecture="all_action_table_v1",
            hazard_parameterization="probability_sigmoid_v1",
            hazard_outcome_path="dedicated_stopgrad_v1",
        )
        model = VectorizedOutcomeModelV2(config)
        context = torch.randn(3, config.width, requires_grad=True)
        table = model(context)
        table.predicted_hazard.mean().backward()

        self.assertIsNone(context.grad)
        self.assertIsNone(model.state_trunk[0].weight.grad)
        for parameter in (
            model.hazard_state_trunk[0].weight,
            model.hazard_action_embedding.weight,
            model.hazard_outcome_trunk[0].weight,
            model.hazard_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_dedicated_hazard_path_requires_all_action_architecture(self) -> None:
        with self.assertRaisesRegex(ValueError, "all_action_table_v1"):
            CoreV2Config(hazard_outcome_path="dedicated_stopgrad_v1")


if __name__ == "__main__":
    unittest.main()
