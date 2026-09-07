r"""Unit and Regression Test Suite for Tier 2 (~50M Parameter) Architecture Configuration.
Verifies the Three Architectural Laws of Cognitive Scaling:
1. Law 1 (Slot Width Clamping):
   - Slot width bounded at W=64, K=128 slots.
   - Total recurrent state is K*W = 8,192 floats (32 KB state memory, avoiding the
     53k state explosion seen when W was 832).
2. Law 2 (Factorized Deep Projections):
   - Input and hidden projections factorized: W=64 -> rank r=32 -> proj_dim=4096.
   - Delivers the parametric capacity of Tier 2 ~50M without parameter or compute explosion.
3. Law 3 (Event-Driven Dynamic Sparsity):
   - Slot evaluation bypassed when Cognitive Input Gate salience <_ epsilon_dormant (0.05).
   - Zero-FLOP bypass on dormant slots while preserving state bitwise.
Also validates:
- First-class support for tier="tier2" across BrainCellCore, Tier2BrainModel, and make_tier_model.
- Parameter counting utility confirms parameters in ~35M - 50M range (~51M).
- Full forward and backward autograd gradient flow.
"""
from __future__ import annotations
import unittest
import torch
import torch.nn as nn
from irene_brain.model.brain_cell import (
    BrainCellCore,
    BrainCell,
    FactorizedLowRankProjection,
)
from irene_brain.model.torch_model import (
    Tier2BrainModel,
    IreneBrainModel,
    ThoughtFieldConfig,
    count_parameters,
    make_tier_model,
)
class TestTier2Architecture(unittest.TestCase):
    """Test suite for Tier 2 (~50M parameter) architecture specification and laws."""
    def test_tier2_brain_cell_core_instantiation(self) -> None:
        """Verify BrainCellCore first-class support for tier='tier2'."""
        core = BrainCellCore(tier="tier2")
        self.assertEqual(core.tier, "tier2")
        self.assertEqual(core.thought_size, 64)
        self.assertEqual(core.rank, 32)
        self.assertEqual(core.proj_dim, 4096)
        self.assertEqual(core.input_size, 4096)
        self.assertIsInstance(core.deep_proj, nn.Sequential)
        self.assertIsInstance(core.W_ir, FactorizedLowRankProjection)
        self.assertIsInstance(core.W_iz, FactorizedLowRankProjection)
        self.assertEqual(core.state_bytes(K=128), 32768)

    def test_tier2_model_instantiation(self) -> None:
        """Verify Tier2BrainModel and make_tier_model instantiation."""
        model = Tier2BrainModel(input_dim=64, K=128)
        self.assertEqual(model.tier, "tier2")
        self.assertEqual(model.K, 128)
        self.assertEqual(model.thought_size, 64)
        self.assertEqual(model.rank, 32)
        self.assertEqual(model.proj_dim, 4096)
        self.assertEqual(model.epsilon_dormant, 0.05)

        factory_model = make_tier_model(tier="tier2", input_dim=64, K=128)
        self.assertIsInstance(factory_model, Tier2BrainModel)
        self.assertEqual(factory_model.K, 128)
        self.assertEqual(factory_model.thought_size, 64)

    def test_tier2_parameter_count_specification(self) -> None:
        """Verify parameter counting utility confirms parameters in ~35M - 50M range."""
        model = Tier2BrainModel(input_dim=64, K=128)
        counts = count_parameters(model)
        total_p = counts["total"]
        trainable_p = counts["trainable"]
        self.assertEqual(total_p, trainable_p)
        self.assertGreaterEqual(total_p, 35_000_000)
        self.assertLessEqual(total_p, 52_000_000)

        core = BrainCellCore(tier="tier2")
        core_counts = count_parameters(core)
        self.assertGreaterEqual(core_counts["total"], 35_000_000)
        self.assertLessEqual(core_counts["total"], 52_000_000)

    def test_tier2_state_bytes_contract_32kb(self) -> None:
        """Verify Law 1: Slot Width Clamping maintains exactly 32 KB state memory."""
        K = 128
        W = 64
        bytes_per_float = 4
        expected_floats = K * W
        expected_bytes = expected_floats * bytes_per_float
        self.assertEqual(expected_floats, 8192)
        self.assertEqual(expected_bytes, 32768)

        model = Tier2BrainModel(K=K, thought_size=W)
        self.assertEqual(model.state_bytes(), 32768)

        core = BrainCellCore(tier="tier2")
        self.assertEqual(core.state_bytes(K=128), 32768)

        init_state = model.initial_state(batch_size=1)
        self.assertEqual(init_state.shape, (1, 128, 64))
        self.assertEqual(init_state.element_size() * init_state.numel(), 32768)

    def test_tier2_forward_and_backward_autograd_pass(self) -> None:
        """Verify forward evaluation and backward autograd gradient pass."""
        B, T, D = 2, 2, 64
        model = Tier2BrainModel(input_dim=D, K=128, num_deep_layers=2)
        x_seq = torch.randn(B, T, D, requires_grad=True)

        logits = model(x_seq)
        self.assertEqual(logits.shape, (B, T, model.num_values))
        self.assertTrue(torch.isfinite(logits).all())

        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(x_seq.grad)
        self.assertTrue(torch.isfinite(x_seq.grad).all())
        self.assertIsNotNone(model.input_proj.weight.grad)
        self.assertIsNotNone(model.brain_cell.W_ir.down.weight.grad)

    def test_tier2_conditional_recurrence_execution(self) -> None:
        """Verify Law 3: Event-Driven Synaptic Sparsity bypasses dormant slots with zero FLOPs."""
        B = 2
        K = 128
        W = 64
        core = BrainCellCore(tier="tier2", num_deep_layers=2)
        thoughts = torch.randn(B, K, W)
        x = torch.randn(B, K, 4096)

        salience = torch.full((B, K, 1), 0.01)
        salience[:, 5] = 0.90

        core.reset_telemetry()
        updated, active_mask = core.forward_conditional(
            thoughts=thoughts,
            x=x,
            salience=salience,
            epsilon_dormant=0.05,
        )

        self.assertEqual(int(active_mask.sum().item()), 1 * B)
        self.assertEqual(core.eval_count, 1 * B)

        for k in range(K):
            if k != 5:
                self.assertTrue(torch.equal(updated[:, k], thoughts[:, k]))
        self.assertFalse(torch.equal(updated[:, 5], thoughts[:, 5]))

    def test_tier2_thought_field_config_support(self) -> None:
        """Verify ThoughtFieldConfig.tier2() and IreneBrainModel support."""
        cfg = ThoughtFieldConfig.tier2()
        self.assertEqual(cfg.core_width, 64)
        self.assertEqual(cfg.thoughtlets, 128)
        self.assertEqual(cfg.registers_per_thoughtlet, 1)

        irene = IreneBrainModel(tier="tier2", plan_steps=1)
        self.assertEqual(irene.config.core_width, 64)
        self.assertEqual(irene.config.thoughtlets, 128)

if __name__ == '__main__':
    unittest.main()
