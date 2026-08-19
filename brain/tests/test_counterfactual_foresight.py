"""Unit tests for Action-Conditioned Counterfactual Foresight and Planner Integration."""

from __future__ import annotations

import unittest
import torch
import torch.nn.functional as F

from irene_brain.model.counterfactual_foresight import (
    ActionConditionedCounterfactualForesightHead,
    CounterfactualBranchOutput,
)
from irene_brain.model.intent import DirectionalAction
from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.cognitive_losses import CognitiveAuxiliaryLoss


class CounterfactualForesightTests(unittest.TestCase):
    def test_counterfactual_head_shapes_and_outputs(self) -> None:
        width = 32
        head = ActionConditionedCounterfactualForesightHead(
            width=width,
            num_actions=5,
            horizons=(1, 3, 5),
        )
        batch_size = 4
        thoughts = torch.randn(batch_size, 4, 2, width)

        # 1. Single action forward
        single_out = head.forward_single_action(thoughts.mean(dim=(1, 2)), 1)
        self.assertIsInstance(single_out, CounterfactualBranchOutput)
        self.assertEqual(single_out.action_index, 1)
        self.assertEqual(tuple(single_out.predicted_displacement.shape), (batch_size, 3, 2))
        self.assertEqual(tuple(single_out.hazard_probability.shape), (batch_size, 3, 1))
        self.assertEqual(tuple(single_out.predicted_escape_margin.shape), (batch_size, 3, 1))
        self.assertTrue(bool((single_out.hazard_probability >= 0.0).all()))
        self.assertTrue(bool((single_out.hazard_probability <= 1.0).all()))

        # 2. All actions parallel forward
        all_out = head.forward_all_actions(thoughts)
        self.assertEqual(len(all_out), 5)
        for a in range(5):
            self.assertIn(a, all_out)
            self.assertEqual(tuple(all_out[a].predicted_displacement.shape), (batch_size, 3, 2))

    def test_counterfactual_branch_differentiation(self) -> None:
        width = 32
        head = ActionConditionedCounterfactualForesightHead(
            width=width,
            num_actions=5,
            horizons=(1, 3, 5),
        )
        thoughts = torch.randn(1, 4, 2, width)
        all_out = head.forward_all_actions(thoughts)

        # Branch 1 (W) vs Branch 3 (S) must have distinct representations
        w_disp = all_out[1].predicted_displacement
        s_disp = all_out[3].predicted_displacement
        self.assertFalse(torch.allclose(w_disp, s_disp))

    def test_counterfactual_loss_gradient_flow(self) -> None:
        width = 32
        head = ActionConditionedCounterfactualForesightHead(
            width=width,
            num_actions=5,
            horizons=(1, 3, 5),
        )
        loss_fn = CognitiveAuxiliaryLoss(counterfactual_weight=1.0)
        batch_size = 2
        thoughts = torch.randn(batch_size, 4, 2, width, requires_grad=True)

        all_out = head.forward_all_actions(thoughts)

        class DummyDiagnostics:
            def __init__(self, summaries: torch.Tensor, cf_preds: dict) -> None:
                self.thought_summaries = summaries
                self.counterfactual_predictions = cf_preds
                self.thought_update_gates = None
                self.future_trajectory_predictions = None
                self.halting_probabilities = ()

        diag = DummyDiagnostics(thoughts.mean(dim=(1, 2)), all_out)
        executed_actions = torch.tensor([1, 4], dtype=torch.long)
        future_disp_targets = torch.randn(batch_size, 3, 2)
        actual_collisions = torch.tensor([[[0.0], [0.0], [0.0]], [[1.0], [1.0], [1.0]]])
        future_escape_targets = torch.tensor([[[2.5], [1.8], [0.5]], [[-1.0], [-2.0], [-3.0]]])

        loss_output = loss_fn(
            diagnostics=diag,
            future_disp_targets=future_disp_targets,
            future_escape_targets=future_escape_targets,
            executed_actions=executed_actions,
            actual_collisions=actual_collisions,
        )

        self.assertGreater(loss_output.total_loss.item(), 0.0)
        self.assertIsNotNone(loss_output.counterfactual_loss)
        loss_output.total_loss.backward()
        self.assertIsNotNone(thoughts.grad)
        self.assertTrue(bool(torch.isfinite(thoughts.grad).all()))

    def test_planner_counterfactual_hazard_pruning(self) -> None:
        config = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config=config, enable_adaptive_cognition=True)
        planner = LatentLookaheadPlanner(model=model, horizon=2, hazard_prune_threshold=0.5)

        state = model.initial_state(batch_size=1)
        plan_result = planner.plan(state=state)
        self.assertIn(
            plan_result.best_action,
            (
                DirectionalAction.NONE,
                DirectionalAction.W,
                DirectionalAction.A,
                DirectionalAction.S,
                DirectionalAction.D,
            ),
        )
        self.assertGreater(len(plan_result.all_branches), 0)


if __name__ == "__main__":
    unittest.main()
