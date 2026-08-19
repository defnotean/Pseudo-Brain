from __future__ import annotations
import unittest
import torch
from irene_brain.model.topological_goal import TopologicalGoalFieldHead, TopologicalGoalPrediction
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.lookahead_planner import LatentLookaheadPlanner

class TestTopologicalGoalHead(unittest.TestCase):
    def test_head_shapes_and_outputs(self) -> None:
        head = TopologicalGoalFieldHead(width=64, thoughtlets=4)
        # Test 3D input (B, K, D)
        thoughts_3d = torch.randn(2, 4, 64)
        pred = head(thoughts_3d)
        self.assertIsInstance(pred, TopologicalGoalPrediction)
        self.assertEqual(pred.pellet_cluster_vector.shape, (2, 2))
        self.assertEqual(pred.junction_exit_logits.shape, (2, 5))
        self.assertEqual(pred.corridor_depth_estimate.shape, (2, 1))
        
        # Test 4D input (B, K, R, D)
        thoughts_4d = torch.randn(2, 4, 3, 64)
        pred_4d = head(thoughts_4d)
        self.assertEqual(pred_4d.pellet_cluster_vector.shape, (2, 2))
        
        norms = torch.norm(pred.pellet_cluster_vector, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-4))
        self.assertTrue(torch.all(pred.corridor_depth_estimate >= 0.0))

    def test_gradient_flow(self) -> None:
        head = TopologicalGoalFieldHead(width=64, thoughtlets=4)
        thoughts = torch.randn(2, 4, 64, requires_grad=True)
        pred = head(thoughts)
        loss = pred.pellet_cluster_vector.sum() + pred.junction_exit_logits.sum() + pred.corridor_depth_estimate.sum()
        loss.backward()
        self.assertIsNotNone(thoughts.grad)
        self.assertTrue(torch.all(torch.isfinite(thoughts.grad)))

    def test_planner_topological_goal_routing_bonus(self) -> None:
        cfg = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config=cfg, enable_adaptive_cognition=True)
        planner = LatentLookaheadPlanner(model=model, horizon=2)
        
        state = model.initial_state(batch_size=1)
        plan_res = planner.plan(state=state)
        self.assertIsNotNone(plan_res.best_action)
        self.assertGreater(len(plan_res.all_branches), 0)

if __name__ == '__main__':
    unittest.main()
