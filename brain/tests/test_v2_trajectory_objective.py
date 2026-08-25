"""CPU contracts for the causal Core V2 trajectory objective."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from irene_brain.training.batches import (
    AllActionTrajectoryBatchV1,
    MazeChaseBatchConfig,
    MazeChaseBatchSource,
    TrajectoryBatch,
)
from irene_brain.types import GenericControl, HidKey
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.trajectory_objective import (
    AllActionV2TrajectoryObjective,
    V2TrajectoryObjective,
    control_action_class,
    transition_hazard,
)


class V2TrajectoryObjectiveContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = CoreV2Config(
            braincell_dynamics="normalized_mixture_v1",
            belief_dynamics="convex_gated_v1",
            hazard_parameterization="probability_sigmoid_v1",
        )
        self.model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        self.objective = V2TrajectoryObjective(self.model, flags=CONFIG_B_PREDICTIVE)
        source = MazeChaseBatchSource(
            MazeChaseBatchConfig(
                train_sequences=2,
                validation_sequences=1,
                test_sequences=1,
                sequence_length=3,
                burn_in_steps=1,
                seed_offset=77,
                ghost_count=1,
            )
        )
        self.batch = next(
            source.iter_batches(
                split="train",
                epoch=0,
                start_batch=0,
                batch_size=2,
                max_batches=1,
            )
        )

    def test_control_and_hazard_targets_are_explicit(self) -> None:
        self.assertEqual(control_action_class(GenericControl.neutral()), 0)
        self.assertEqual(control_action_class(GenericControl(keys_down=(int(HidKey.W),))), 1)
        self.assertEqual(control_action_class(GenericControl(keys_down=(int(HidKey.D),))), 4)
        self.assertEqual(transition_hazard(("pellet_eaten",)), 0.0)
        self.assertEqual(transition_hazard(("caught",)), 1.0)
        self.assertEqual(transition_hazard(("collision",)), 1.0)

    def test_objective_aligns_all_four_learning_signals(self) -> None:
        result = self.objective(self.batch)

        self.assertEqual(result.samples, 4)
        self.assertEqual(set(result.components), {"decision", "reward", "hazard", "next_latent"})
        self.assertEqual(
            set(result.metrics),
            {
                "action_accuracy",
                "reward_mae",
                "hazard_brier",
                "next_latent_cosine",
                "turn_accuracy",
                "turn_samples",
                "hazard_positives",
                "hazard_negatives",
                "hazard_positive_probability",
                "hazard_negative_probability",
                "max_abs_belief",
                "max_abs_thought",
            },
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertLessEqual(float(result.metrics["max_abs_belief"]), 1.00001)
        self.assertLessEqual(float(result.metrics["max_abs_thought"]), 16.0)

    def test_feedback_ablation_keeps_the_same_trainable_architecture(self) -> None:
        zeroed = V2TrajectoryObjective(
            self.model,
            flags=CONFIG_B_PREDICTIVE,
            prediction_error_intervention="zero",
        )

        self.assertEqual(
            sum(parameter.numel() for parameter in self.objective.parameters()),
            sum(parameter.numel() for parameter in zeroed.parameters()),
        )

    def test_all_predictive_heads_receive_gradient(self) -> None:
        result = self.objective(self.batch)
        result.loss.backward()

        parameters = (
            self.model.consequence_model.action_head.weight,
            self.model.consequence_model.reward_head.weight,
            self.model.consequence_model.hazard_head.weight,
            self.model.world_model.transition[0].weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_target_encoder_moves_only_on_explicit_ema_update(self) -> None:
        target_parameter = next(self.model.world_model.target_encoder.parameters())
        before = target_parameter.detach().clone()
        with torch.no_grad():
            next(self.model.encoder.parameters()).add_(0.1)
        self.assertTrue(torch.equal(before, target_parameter))

        self.objective.update_target_encoder()

        self.assertFalse(torch.equal(before, target_parameter))

    def test_fixed_trajectory_learning_reduces_joint_loss(self) -> None:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=1e-3,
            weight_decay=1e-4,
        )
        losses = []
        for _ in range(12):
            torch.manual_seed(1234)
            result = self.objective(self.batch)
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                1.0,
            )
            optimizer.step()
            self.objective.update_target_encoder()
            losses.append(float(result.loss.detach()))

        self.assertLess(sum(losses[-3:]) / 3, (sum(losses[:3]) / 3) * 0.80)

    def test_all_action_batches_and_objective_fail_closed_across_boundary(self) -> None:
        source = MazeChaseBatchSource(
            MazeChaseBatchConfig(
                train_sequences=2,
                validation_sequences=1,
                test_sequences=1,
                sequence_length=3,
                burn_in_steps=1,
                seed_offset=81,
                ghost_count=1,
                counterfactual_targets="all_actions_v1",
            )
        )
        with self.assertRaisesRegex(RuntimeError, "iter_all_action_batches"):
            next(
                source.iter_batches(
                    split="train",
                    epoch=0,
                    start_batch=0,
                    batch_size=2,
                    max_batches=1,
                )
            )
        batch = next(
            source.iter_all_action_batches(
                split="train",
                epoch=0,
                start_batch=0,
                batch_size=2,
                max_batches=1,
            )
        )
        self.assertIsInstance(batch, AllActionTrajectoryBatchV1)
        with self.assertRaisesRegex(ValueError, "AllActionV2TrajectoryObjective"):
            self.objective(
                # A manually widened legacy batch must still be rejected.
                TrajectoryBatch(
                    split=batch.split,
                    burn_in_steps=batch.burn_in_steps,
                    sequences=batch.sequences,
                )
            )

        config = CoreV2Config(
            braincell_dynamics="normalized_mixture_v1",
            belief_dynamics="convex_gated_v1",
            hazard_parameterization="probability_sigmoid_v1",
            latent_comparison="cosine_distance_v1",
            outcome_architecture="all_action_table_v1",
            reward_prediction="symlog_twohot_v1",
        )
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        objective = AllActionV2TrajectoryObjective(
            model,
            flags=CONFIG_B_PREDICTIVE,
        )
        with self.assertRaisesRegex(ValueError, "AllActionTrajectoryBatchV1"):
            objective(self.batch)  # type: ignore[arg-type]

        result = objective(batch)
        # CE exposure remains one sample per factual state, not five branches.
        self.assertEqual(result.samples, 4)
        self.assertEqual(
            set(result.components),
            {"decision", "reward", "hazard", "next_latent"},
        )
        result.loss.backward()
        outcome_model = model.world_model.outcome_model
        self.assertIsNotNone(outcome_model)
        for parameter in (
            outcome_model.next_latent_head.weight,
            outcome_model.reward_head.weight,
            outcome_model.hazard_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
