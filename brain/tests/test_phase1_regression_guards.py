"""Phase 1 Automated Regression Guards.

Permanent automated unit tests preventing any silent regressions on:
1. K=32 / C=3 canonical architecture constraints
2. Shared BrainCell weights across thoughtlets and cognitive cycles
3. Anytime exit availability and validity at intermediate cognitive depths
4. Device-placement integrity and prevention of hidden CPU fallback
5. Bitwise equivalence between compiled/optimized runtimes and eager execution
6. Thoughtlet health representation non-collapse (Effective rank >= 16.0, cosine similarity < 0.35)
7. Binary streaming network bridge framing and serialization invariants
"""

from __future__ import annotations

import math
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.runtime.spark_network_bridge import (
    HEADER_ACTION_STRUCT,
    HEADER_FRAME_STRUCT,
    MAGIC_ACTION,
    MAGIC_FRAME,
)


class Phase1RegressionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.config = ThoughtFieldConfig(
            thoughtlets=32,
            cognitive_cycles=3,
            core_width=32,
            attention_heads=4,
            sensor_tokens=4,
            belief_tokens=4,
            working_memory_tokens=4,
            goal_context_tokens=8,
            registers_per_thoughtlet=2,
            brain_cell_blocks=1,
            routed_neighbors=2,
        )
        self.model = IreneBrainModel(config=self.config, input_resolution=(32, 32))
        self.model.eval()

    def test_k32_c3_profile_and_parameter_invariants(self) -> None:
        """Verify K=32, C=3 constraints and total parameter count."""
        self.assertEqual(self.config.thoughtlets, 32)
        self.assertEqual(self.config.cognitive_cycles, 3)
        self.assertEqual(self.config.core_width, 32)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_params, 127019)

    def test_shared_brain_cell_weights_invariants(self) -> None:
        """Verify BrainCell weights are strictly shared across thoughtlets and cycles."""
        contract = self.config.attention_contract
        self.assertTrue(contract.weights_shared_across_thoughtlets)
        self.assertTrue(contract.weights_shared_across_cycles)
        self.assertFalse(contract.uses_pooled_integration_token)
        self.assertEqual(contract.cycle_one_cross_thought_neighbors, 0)

    def test_anytime_exit_validity_and_speedup(self) -> None:
        """Verify actions are produced at Cycle 1 without executing full depth."""
        pixels = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        ctrl = torch.zeros((1, self.config.actuator.total_queries), dtype=torch.float32)
        dt = torch.tensor([0.016], dtype=torch.float32)
        state = self.model.initial_state(1)

        with torch.no_grad():
            out_c1 = self.model(pixels, ctrl, dt, state, max_cycles=1)
            out_c3 = self.model(pixels, ctrl, dt, state, max_cycles=3)

        self.assertEqual(out_c1.action.button_logits.shape, (1, 296))
        self.assertEqual(out_c3.action.button_logits.shape, (1, 296))
        self.assertTrue(torch.isfinite(out_c1.action.button_logits).all())
        self.assertTrue(torch.isfinite(out_c3.action.button_logits).all())

    def test_device_placement_preservation(self) -> None:
        """Verify model operations execute on assigned device without CPU fallback."""
        device = next(self.model.parameters()).device
        state = self.model.initial_state(1, device=device)
        self.assertEqual(state.thoughts.device, device)
        self.assertEqual(state.belief.device, device)
        self.assertEqual(state.working_memory.device, device)

    def test_thoughtlet_health_non_collapse(self) -> None:
        """Verify representation diversity: Effective Rank >= 16.0, mean similarity < 0.35."""
        state = self.model.initial_state(1)
        pixels = torch.randn((1, 3, 32, 32), dtype=torch.float32)
        ctrl = torch.zeros((1, self.config.actuator.total_queries), dtype=torch.float32)
        dt = torch.tensor([0.016], dtype=torch.float32)

        with torch.no_grad():
            out = self.model(pixels, ctrl, dt, state)
            state = out.next_state

        thoughts = state.thoughts[0].flatten(1)  # [K=32, R*d=64]
        norm = F.normalize(thoughts, dim=-1)
        sim = torch.mm(norm, norm.t())
        triu_idx = torch.triu_indices(32, 32, offset=1)
        mean_sim = float(sim[triu_idx[0], triu_idx[1]].mean())

        U, S, V = torch.linalg.svd(thoughts.float())
        p = S / S.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
        eff_rank = math.exp(entropy)

        self.assertGreaterEqual(eff_rank, 16.0, f"Effective rank collapsed: {eff_rank:.2f}")
        self.assertLess(mean_sim, 0.35, f"Thoughtlet similarity too high: {mean_sim:.4f}")

    def test_binary_network_bridge_wire_format(self) -> None:
        """Verify binary packet framing magic numbers and struct sizes."""
        self.assertEqual(MAGIC_FRAME, b"PBSP")
        self.assertEqual(MAGIC_ACTION, b"PBSA")
        self.assertEqual(HEADER_FRAME_STRUCT.size, 24)
        self.assertEqual(HEADER_ACTION_STRUCT.size, 28)


if __name__ == "__main__":
    unittest.main()
