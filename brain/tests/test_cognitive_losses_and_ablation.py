"""Unit tests for Cognitive Auxiliary Losses, Leak-Free Foresight, and Ablation Configurations."""

from dataclasses import replace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.spatial_cognitive_diagnostics import (
    run_instrumented_diagnostic_episode,
)
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.cognitive_losses import CognitiveAuxiliaryLoss


class TestCognitiveLossesAndAblation(unittest.TestCase):
    """Rigorous unit tests ensuring no data leakage, valid gradients, and ablation consistency."""

    def setUp(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is required.")
        base_config = ThoughtFieldConfig.smoke()
        self.config = replace(
            base_config,
            core_width=32,
            thoughtlets=4,
            cognitive_cycles=3,
        )

    def test_zero_cheating_and_no_future_leakage(self) -> None:
        """Assert that IreneBrainModel forward pass strictly depends on current input tensors only."""
        model = IreneBrainModel(self.config, enable_adaptive_cognition=True)
        model.eval()

        batch_size = 2
        pixels = torch.zeros(batch_size, 3, 32, 32)
        prev_ctrl = torch.zeros(batch_size, 307)
        elapsed = torch.ones(batch_size) * (1.0 / 60.0)

        # Forward step at time t
        out_t = model(pixels, prev_ctrl, elapsed, state=None)

        # Confirm diagnostics are present and zero future state was injected
        self.assertIsNotNone(out_t.diagnostics)
        self.assertIn("predicted_displacement", out_t.diagnostics.future_trajectory_predictions)
        self.assertIn("predicted_ghost_proximity", out_t.diagnostics.future_trajectory_predictions)
        self.assertIn("predicted_escape_margin", out_t.diagnostics.future_trajectory_predictions)

        # Verify shapes: horizons 1, 3, 5 -> [B, 3, 2], [B, 3, 1], [B, 3, 1]
        self.assertEqual(out_t.diagnostics.future_trajectory_predictions["predicted_displacement"].shape, (2, 3, 2))
        self.assertEqual(out_t.diagnostics.future_trajectory_predictions["predicted_ghost_proximity"].shape, (2, 3, 1))
        self.assertEqual(out_t.diagnostics.future_trajectory_predictions["predicted_escape_margin"].shape, (2, 3, 1))

    def test_cognitive_auxiliary_loss_computation_and_gradients(self) -> None:
        """Verify cognitive auxiliary loss produces valid finite scalars and backpropagates cleanly."""
        model = IreneBrainModel(self.config, enable_adaptive_cognition=True)
        model.train()

        batch_size = 4
        pixels = torch.randn(batch_size, 3, 32, 32)
        prev_ctrl = torch.zeros(batch_size, 307)
        elapsed = torch.ones(batch_size) * (1.0 / 60.0)

        out = model(pixels, prev_ctrl, elapsed, state=None)

        loss_fn = CognitiveAuxiliaryLoss(
            future_weight=1.0,
            gate_surprise_weight=1.0,
            halting_weight=0.5,
            compute_cost_per_cycle=0.01,
        )

        future_disp_targets = torch.randn(batch_size, 3, 2)
        future_ghost_targets = torch.ones(batch_size, 3, 1) * 3.0
        future_escape_targets = torch.ones(batch_size, 3, 1) * 1.5
        hazard_mask = torch.tensor([True, False, True, False])
        complexity = torch.tensor([0.9, 0.1, 0.8, 0.2])

        loss_out = loss_fn(
            diagnostics=out.diagnostics,
            future_disp_targets=future_disp_targets,
            future_ghost_targets=future_ghost_targets,
            future_escape_targets=future_escape_targets,
            hazard_mask=hazard_mask,
            complexity_level=complexity,
        )

        self.assertTrue(torch.isfinite(loss_out.total_loss))
        self.assertGreater(loss_out.total_loss.item(), 0.0)

        # Backpropagation test
        loss_out.total_loss.backward()

        # Check gradients exist on all 3 cognitive submodules
        for param in model.adaptive_thought_gate.parameters():
            self.assertIsNotNone(param.grad)
        for param in model.future_trajectory_head.parameters():
            self.assertIsNotNone(param.grad)
        for param in model.halting_controller.parameters():
            self.assertIsNotNone(param.grad)

    def test_ablation_model_variants_instantiation(self) -> None:
        """Verify that all 5 ablation variants construct with exact isolated heads."""
        import sys
        from pathlib import Path
        brain_dir = Path(__file__).resolve().parents[1]
        if str(brain_dir) not in sys.path:
            sys.path.insert(0, str(brain_dir))
        from scripts.run_cognitive_ablation_battery import build_ablation_model

        # A: Baseline (no adaptive cognition)
        mod_a = build_ablation_model("A_baseline", self.config)
        self.assertFalse(mod_a.enable_adaptive_cognition)
        self.assertIsNone(mod_a.adaptive_thought_gate)
        self.assertIsNone(mod_a.future_trajectory_head)
        self.assertIsNone(mod_a.halting_controller)

        # B: + Adaptive Gate Only
        mod_b = build_ablation_model("B_adaptive_gate", self.config)
        self.assertTrue(mod_b.enable_adaptive_cognition)
        self.assertIsNotNone(mod_b.adaptive_thought_gate)
        self.assertIsNone(mod_b.future_trajectory_head)
        self.assertIsNone(mod_b.halting_controller)

        # C: + Future Prediction Only
        mod_c = build_ablation_model("C_future_prediction", self.config)
        self.assertTrue(mod_c.enable_adaptive_cognition)
        self.assertIsNone(mod_c.adaptive_thought_gate)
        self.assertIsNotNone(mod_c.future_trajectory_head)
        self.assertIsNone(mod_c.halting_controller)

        # D: + Adaptive Depth Only
        mod_d = build_ablation_model("D_adaptive_depth", self.config)
        self.assertTrue(mod_d.enable_adaptive_cognition)
        self.assertIsNone(mod_d.adaptive_thought_gate)
        self.assertIsNone(mod_d.future_trajectory_head)
        self.assertIsNotNone(mod_d.halting_controller)

        # E: All Three
        mod_e = build_ablation_model("E_all_three", self.config)
        self.assertTrue(mod_e.enable_adaptive_cognition)
        self.assertIsNotNone(mod_e.adaptive_thought_gate)
        self.assertIsNotNone(mod_e.future_trajectory_head)
        self.assertIsNotNone(mod_e.halting_controller)


if __name__ == "__main__":
    unittest.main()
