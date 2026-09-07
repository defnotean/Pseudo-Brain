"""Unit Test Suite for Multimodal Model on DirectML Hardware Acceleration.

Verifies:
1. MultimodalPseudoBrainModel resolution and instantiation on DirectML (AMD Radeon RX 9070 XT).
2. End-to-end forward execution (ConvEncoder + Recurrent Core + Language Readout).
3. Exact numerical equivalence between CPU and DirectML execution within tolerance.
4. Action logits querying, cross-modal binding, and streaming sessions on DirectML.
"""

from __future__ import annotations

import unittest
import torch

from irene_brain.device import (
    get_best_directml_device_index,
    get_device_telemetry,
    get_directml_device,
    is_directml_available,
    resolve_optimal_device,
)
from irene_brain.semantic.multimodal_model import (
    MultimodalCognitiveState,
    MultimodalPseudoBrain,
    MultimodalPseudoBrainModel,
    MultimodalStreamingSession,
)


class TestMultimodalDirectML(unittest.TestCase):
    """Test suite validating Multimodal model execution on DirectML accelerator."""

    def setUp(self):
        torch.manual_seed(42)
        self.dev, self.backend = resolve_optimal_device()

    def test_multimodal_device_resolution(self):
        """Verify device resolution selects DirectML when available on AMD platform."""
        if is_directml_available():
            self.assertEqual(self.backend, "directml")
            self.assertEqual(self.dev.type, "privateuseone")
            telemetry = get_device_telemetry()
            if any("9070" in d.get("name", "") for d in telemetry.get("directml_devices", [])):
                self.assertEqual(telemetry.get("directml_device_index"), 1)

    def test_multimodal_forward_on_resolved_device(self):
        """Verify end-to-end forward unroll on resolved device (ConvEncoder + Recurrence + Readout)."""
        model = MultimodalPseudoBrainModel(
            vocab_size=344,
            K=16,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
            visual_dim=192,
            num_visual_tokens=4,
        ).to(self.dev)
        model.eval()

        batch_size = 2
        img = torch.randn(batch_size, 3, 64, 64, device=self.dev)
        tok = torch.randint(0, 300, (batch_size, 6), device=self.dev)

        with torch.no_grad():
            logits, state = model(token_seq=tok, image_tensor=img)

        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (batch_size, 344))
        self.assertIsInstance(state, MultimodalCognitiveState)
        self.assertEqual(state.thoughts.shape, (batch_size, 16, 32))
        self.assertEqual(state.thoughts.device.type, self.dev.type)

    def test_multimodal_numerical_equivalence(self):
        """Verify numerical equivalence between CPU and DirectML outputs within tight tolerance."""
        if not is_directml_available():
            self.skipTest("DirectML not available in this environment.")

        # Create identical CPU and DML models
        torch.manual_seed(123)
        model_cpu = MultimodalPseudoBrainModel(
            vocab_size=344,
            K=16,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
            visual_dim=192,
            num_visual_tokens=4,
        )
        model_dml = MultimodalPseudoBrainModel(
            vocab_size=344,
            K=16,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
            visual_dim=192,
            num_visual_tokens=4,
        )
        model_dml.load_state_dict(model_cpu.state_dict())
        model_dml.to(self.dev)

        model_cpu.eval()
        model_dml.eval()

        # Generate inputs
        batch_size = 4
        img_cpu = torch.randn(batch_size, 3, 64, 64)
        tok_cpu = torch.randint(0, 300, (batch_size, 8))

        img_dml = img_cpu.clone().to(self.dev)
        tok_dml = tok_cpu.clone().to(self.dev)

        with torch.no_grad():
            logits_cpu, state_cpu = model_cpu(token_seq=tok_cpu, image_tensor=img_cpu)
            logits_dml, state_dml = model_dml(token_seq=tok_dml, image_tensor=img_dml)

        logits_dml_cpu = logits_dml.cpu()
        state_dml_cpu = state_dml.thoughts.cpu()

        max_logit_diff = float(torch.max(torch.abs(logits_cpu - logits_dml_cpu)).item())
        max_state_diff = float(torch.max(torch.abs(state_cpu.thoughts - state_dml_cpu)).item())

        self.assertLess(max_logit_diff, 1e-3, f"Logit diff too large: {max_logit_diff}")
        self.assertLess(max_state_diff, 1e-3, f"Thought slot state diff too large: {max_state_diff}")

    def test_multimodal_action_readout_on_device(self):
        """Verify POMDP action logit querying on the resolved device."""
        model = MultimodalPseudoBrainModel(
            vocab_size=344,
            K=8,
            thought_size=32,
            n_actions=5,
        ).to(self.dev)
        model.eval()

        img = torch.randn(2, 3, 64, 64, device=self.dev)
        with torch.no_grad():
            _, state = model(image_tensor=img)
            action_logits = model.get_action_logits(state, slot_idx=0)

        self.assertEqual(action_logits.shape, (2, 5))
        self.assertEqual(action_logits.device.type, self.dev.type)

    def test_multimodal_streaming_session_on_device(self):
        """Verify MultimodalStreamingSession execution on resolved device."""
        model = MultimodalPseudoBrainModel(
            vocab_size=344,
            K=8,
            thought_size=32,
        ).to(self.dev)

        session = MultimodalStreamingSession(model=model)
        pixels = torch.randn(1, 3, 64, 64, device=self.dev)

        # Ingest visual scene
        info = session.ingest_image(pixels, slot_idx=1)
        self.assertEqual(info["active_slot"], 1)

        # Ingest text command tokens
        for tid in [12, 45, 67]:
            logits, dt = session.step_token(tid, thread_id=1)
            self.assertEqual(logits.shape[-1], 344)

        telemetry = session.get_telemetry()
        self.assertIn("active_thread", telemetry)
        self.assertEqual(telemetry["active_thread"], 1)

        action_logits = session.get_action_logits(slot_idx=1)
        self.assertEqual(action_logits.shape[-1], 5)


if __name__ == "__main__":
    unittest.main()
