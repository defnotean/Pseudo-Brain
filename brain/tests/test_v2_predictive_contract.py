"""Mandatory CPU contracts for Core V2 predictive supervision."""
from __future__ import annotations

from dataclasses import replace
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.losses import symlog, total_v2_loss
from irene_brain.v2.prediction_error import PredictionErrorV2
from irene_brain.v2.state import PendingPrediction


class V2PredictiveContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.config = CoreV2Config(
            braincell_dynamics="normalized_mixture_v1",
            belief_dynamics="convex_gated_v1",
            hazard_parameterization="probability_sigmoid_v1",
        )
        self.model = CoreV2Model(config=self.config, flags=CONFIG_B_PREDICTIVE)

    def _forward(self):
        observation = torch.rand(3, 3, 32, 32)
        state = self.model.init_state(3, torch.device("cpu"))
        output, state = self.model(observation, state)
        return output, state

    def test_output_prediction_keeps_gradient_but_recurrent_copy_is_detached(self) -> None:
        output, state = self._forward()

        self.assertIsNotNone(output.pending_prediction)
        self.assertIsNotNone(state.fast.pending_prediction)
        self.assertTrue(output.pending_prediction.predicted_next_latent.requires_grad)
        self.assertFalse(state.fast.pending_prediction.predicted_next_latent.requires_grad)

    def test_world_model_has_frozen_target_encoder(self) -> None:
        self.assertIsNotNone(self.model.world_model.target_encoder)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in self.model.world_model.target_encoder.parameters()
            )
        )
        target = self.model.world_model.compute_target(torch.rand(3, 3, 32, 32))
        self.assertEqual(tuple(target.shape), (3, self.config.width))
        self.assertFalse(target.requires_grad)

    def test_target_encoder_stays_eval_and_buffers_change_only_on_ema(self) -> None:
        target_encoder = self.model.world_model.target_encoder
        self.model.train()
        self.assertFalse(target_encoder.training)
        before = {
            name: buffer.detach().clone()
            for name, buffer in target_encoder.named_buffers()
        }

        self.model.world_model.compute_target(torch.rand(3, 3, 32, 32))
        after_forward = dict(target_encoder.named_buffers())
        for name, expected in before.items():
            self.assertTrue(torch.equal(expected, after_forward[name]))

        self.model.encoder.train()
        self.model.encoder(torch.rand(3, 3, 32, 32))
        source_buffers = dict(self.model.encoder.named_buffers())
        self.model.world_model.update_target_encoder(self.model.encoder)
        after_ema = dict(target_encoder.named_buffers())
        floating_changed = False
        for name, old in before.items():
            if old.is_floating_point() and not torch.equal(old, source_buffers[name]):
                floating_changed = floating_changed or not torch.equal(old, after_ema[name])
            elif not old.is_floating_point():
                self.assertTrue(torch.equal(after_ema[name], source_buffers[name]))
        self.assertTrue(floating_changed)
        self.assertFalse(target_encoder.training)

    def test_predictive_hazard_is_a_probability(self) -> None:
        output, _ = self._forward()

        self.assertTrue(torch.all(output.hypotheses.predicted_hazard >= 0.0))
        self.assertTrue(torch.all(output.hypotheses.predicted_hazard <= 1.0))

    def test_factual_world_prediction_is_action_conditioned(self) -> None:
        observation = torch.rand(3, 3, 32, 32)
        state_zero = self.model.init_state(3, torch.device("cpu"))
        torch.manual_seed(99)
        output_zero, _ = self.model(
            observation,
            state_zero,
            world_model_action=torch.zeros(3, dtype=torch.long),
        )
        state_one = self.model.init_state(3, torch.device("cpu"))
        torch.manual_seed(99)
        output_one, _ = self.model(
            observation,
            state_one,
            world_model_action=torch.ones(3, dtype=torch.long),
        )

        self.assertTrue(
            torch.equal(
                output_zero.decision.action_values,
                output_one.decision.action_values,
            )
        )
        self.assertFalse(
            torch.equal(
                output_zero.pending_prediction.predicted_next_latent,
                output_one.pending_prediction.predicted_next_latent,
            )
        )

    def test_factual_outcomes_are_action_conditioned_after_causal_decision(self) -> None:
        config = replace(
            self.config,
            outcome_action_conditioning="factual_or_proposal_v1",
        )
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        observation = torch.rand(3, 3, 32, 32)
        state_zero = model.init_state(3, torch.device("cpu"))
        torch.manual_seed(99)
        output_zero, _ = model(
            observation,
            state_zero,
            world_model_action=torch.zeros(3, dtype=torch.long),
        )
        state_one = model.init_state(3, torch.device("cpu"))
        torch.manual_seed(99)
        output_one, _ = model(
            observation,
            state_one,
            world_model_action=torch.ones(3, dtype=torch.long),
        )

        # The factual action is supplied only after the causal decision.
        self.assertTrue(
            torch.equal(
                output_zero.decision.action_values,
                output_one.decision.action_values,
            )
        )
        self.assertFalse(
            torch.equal(
                output_zero.hypotheses.predicted_reward,
                output_one.hypotheses.predicted_reward,
            )
        )
        self.assertFalse(
            torch.equal(
                output_zero.hypotheses.predicted_hazard,
                output_one.hypotheses.predicted_hazard,
            )
        )

    def test_all_action_table_is_factual_action_invariant_then_gathered(self) -> None:
        config = replace(
            self.config,
            outcome_architecture="all_action_table_v1",
            reward_prediction="symlog_twohot_v1",
            latent_comparison="cosine_distance_v1",
        )
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        observation = torch.rand(3, 3, 32, 32)
        state_zero = model.init_state(3, torch.device("cpu"))
        torch.manual_seed(99)
        output_zero, _ = model(
            observation,
            state_zero,
            world_model_action=torch.zeros(3, dtype=torch.long),
        )
        state_one = model.init_state(3, torch.device("cpu"))
        torch.manual_seed(99)
        output_one, _ = model(
            observation,
            state_one,
            world_model_action=torch.ones(3, dtype=torch.long),
        )

        self.assertTrue(
            torch.equal(
                output_zero.decision.action_values,
                output_one.decision.action_values,
            )
        )
        self.assertTrue(
            torch.equal(
                output_zero.outcome_table.predicted_next_latent,
                output_one.outcome_table.predicted_next_latent,
            )
        )
        self.assertTrue(
            torch.equal(
                output_zero.outcome_table.predicted_reward_logits,
                output_one.outcome_table.predicted_reward_logits,
            )
        )
        self.assertTrue(
            torch.equal(
                output_zero.pending_prediction.predicted_next_latent,
                output_zero.outcome_table.gather(torch.zeros(3, dtype=torch.long)).predicted_next_latent,
            )
        )
        self.assertFalse(
            torch.equal(
                output_zero.pending_prediction.predicted_next_latent,
                output_one.pending_prediction.predicted_next_latent,
            )
        )

    def test_twohot_factual_losses_reach_vectorized_outcome_model(self) -> None:
        config = replace(
            self.config,
            outcome_architecture="all_action_table_v1",
            reward_prediction="symlog_twohot_v1",
            latent_comparison="cosine_distance_v1",
        )
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        observation = torch.rand(3, 3, 32, 32)
        state = model.init_state(3, torch.device("cpu"))
        output, _ = model(
            observation,
            state,
            world_model_action=torch.tensor([0, 1, 2]),
        )
        _, components = total_v2_loss(
            output,
            config,
            CONFIG_B_PREDICTIVE,
            torch.tensor([1, 2, 3]),
            actual_reward=torch.tensor([[-10.0], [0.0], [1.0]]),
            actual_hazard=torch.tensor([[1.0], [0.0], [0.0]]),
            next_latent=torch.randn(3, config.width),
        )
        total = sum(components.values())
        total.backward()
        outcome = model.world_model.outcome_model
        for parameter in (
            outcome.state_trunk[0].weight,
            outcome.action_embedding.weight,
            outcome.outcome_trunk[0].weight,
            outcome.next_latent_head.weight,
            outcome.reward_head.weight,
            outcome.hazard_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_reward_and_hazard_errors_change_cognition_when_fused(self) -> None:
        config = replace(
            self.config,
            outcome_architecture="all_action_table_v1",
            reward_prediction="symlog_twohot_v1",
            latent_comparison="cosine_distance_v1",
            prediction_error_fusion="latent_outcome_surprise_v1",
        )
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE).eval()
        observation = torch.rand(2, 3, 32, 32)
        baseline = model.init_state(2, torch.device("cpu"))
        changed = model.init_state(2, torch.device("cpu"))
        shared_latent = torch.randn(2, config.width)
        baseline.fast.pending_prediction = PendingPrediction(
            predicted_next_latent=shared_latent,
            predicted_reward=torch.zeros(2, 1),
            predicted_hazard=torch.zeros(2, 1),
        )
        changed.fast.pending_prediction = PendingPrediction(
            predicted_next_latent=shared_latent.clone(),
            predicted_reward=torch.full((2, 1), 3.0),
            predicted_hazard=torch.ones(2, 1),
        )
        kwargs = {
            "actual_reward": torch.zeros(2, 1),
            "actual_hazard": torch.zeros(2, 1),
            "world_model_action": torch.zeros(2, dtype=torch.long),
        }
        torch.manual_seed(123)
        output_baseline, _ = model(observation, baseline, **kwargs)
        torch.manual_seed(123)
        output_changed, _ = model(observation, changed, **kwargs)

        self.assertTrue(
            torch.equal(
                output_baseline.prediction_error.latent_error,
                output_changed.prediction_error.latent_error,
            )
        )
        self.assertFalse(
            torch.equal(
                output_baseline.prediction_error.cognitive_error,
                output_changed.prediction_error.cognitive_error,
            )
        )
        self.assertFalse(torch.equal(output_baseline.belief, output_changed.belief))
        self.assertFalse(torch.equal(output_baseline.thoughts, output_changed.thoughts))
        self.assertFalse(
            torch.equal(
                output_baseline.decision.action_values,
                output_changed.decision.action_values,
            )
        )

    def test_next_latent_component_uses_configured_weight(self) -> None:
        output, _ = self._forward()
        target_action = torch.tensor([0, 1, 2])
        next_latent = torch.randn(3, self.config.width)

        total, components = total_v2_loss(
            output,
            self.config,
            CONFIG_B_PREDICTIVE,
            target_action,
            next_latent=next_latent,
        )

        expected = (
            self.config.decision_weight * components["decision"]
            + self.config.next_weight * components["next_latent"]
        )
        self.assertTrue(torch.allclose(total, expected))

    def test_next_latent_loss_reaches_world_model(self) -> None:
        output, _ = self._forward()
        target_action = torch.tensor([0, 1, 2])
        next_latent = torch.randn(3, self.config.width)
        total, _ = total_v2_loss(
            output,
            self.config,
            CONFIG_B_PREDICTIVE,
            target_action,
            next_latent=next_latent,
        )

        total.backward()
        gradient = self.model.world_model.transition[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_cosine_latent_loss_ignores_representation_scale(self) -> None:
        config = replace(self.config, latent_comparison="cosine_distance_v1")
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        observation = torch.rand(3, 3, 32, 32)
        state = model.init_state(3, torch.device("cpu"))
        output, _ = model(observation, state)
        target_action = torch.tensor([0, 1, 2])
        target = torch.randn(3, config.width)
        _, components = total_v2_loss(
            output,
            config,
            CONFIG_B_PREDICTIVE,
            target_action,
            next_latent=target,
        )
        _, scaled = total_v2_loss(
            output,
            config,
            CONFIG_B_PREDICTIVE,
            target_action,
            next_latent=target * 9.0,
        )

        self.assertTrue(torch.allclose(components["next_latent"], scaled["next_latent"]))

    def test_cosine_prediction_error_feedback_is_scale_invariant(self) -> None:
        config = replace(self.config, latent_comparison="cosine_distance_v1")
        module = PredictionErrorV2(config).eval()
        predicted = torch.randn(2, 3, config.width)
        actual = torch.randn(2, config.width)
        baseline = module(PendingPrediction(predicted_next_latent=predicted), actual)
        scaled = module(
            PendingPrediction(predicted_next_latent=predicted * 7.0),
            actual * 3.0,
        )

        self.assertTrue(
            torch.allclose(baseline.latent_error, scaled.latent_error, atol=1e-7, rtol=1e-7)
        )
        self.assertTrue(
            torch.allclose(baseline.surprise, scaled.surprise, atol=1e-7, rtol=1e-7)
        )

    def test_symlog_reward_loss_and_feedback_use_compressed_scale(self) -> None:
        config = replace(self.config, reward_comparison="symlog_mse_v1")
        model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
        observation = torch.rand(3, 3, 32, 32)
        state = model.init_state(3, torch.device("cpu"))
        output, _ = model(observation, state)
        rewards = torch.tensor([[-10.0], [0.0], [1.0]])
        _, components = total_v2_loss(
            output,
            config,
            CONFIG_B_PREDICTIVE,
            torch.tensor([0, 1, 2]),
            actual_reward=rewards,
        )
        expected = torch.nn.functional.mse_loss(
            symlog(output.hypotheses.predicted_reward.mean(dim=1)),
            symlog(rewards),
        )
        self.assertTrue(torch.allclose(components["reward"], expected))

        pending = PendingPrediction(
            predicted_next_latent=torch.randn(3, config.width),
            predicted_reward=torch.zeros(3, 1),
        )
        feedback = PredictionErrorV2(config)(
            pending,
            torch.randn(3, config.width),
            actual_reward=rewards,
        )
        self.assertTrue(torch.allclose(feedback.reward_error, symlog(rewards)))


if __name__ == "__main__":
    unittest.main()
