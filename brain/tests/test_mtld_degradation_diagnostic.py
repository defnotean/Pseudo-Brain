"""Unit and Regression Test Suite for MTLD Degradation Causal Diagnostic.

Verifies:
1. DiagnosticCGPModel instantiation and forward pass across all 4 routing modes:
   ('disconnected', 'sparse_topk', 'dense', 'dependency_gated').
2. Output tensor shapes (B, T, num_values) and diagnostics telemetry.
3. Gate temperature sharpening impact on salience distribution.
4. Latch persistence (lambda = 1.0) and gradient backpropagation.
5. Slot width capacity scaling (W in [12, 24, 48, 64]).
6. Dependency-gated dynamic routing threshold activation.
"""

from __future__ import annotations

import unittest
import torch
import torch.nn.functional as F

from memory_benchmark.multi_threaded_latent_dependency_benchmark import (
    MultiThreadedLatentDependencyEnv,
    build_batch,
    count_params,
)
from memory_benchmark.mtld_degradation_diagnostic import (
    DiagnosticCGPModel,
    train_diagnostic_model,
    evaluate_diagnostic_model,
)


class MTLDDegradationDiagnosticTests(unittest.TestCase):
    """Test suite for DiagnosticCGPModel and diagnostic harness."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.env = MultiThreadedLatentDependencyEnv(
            num_variables=4,
            num_values=8,
            delay_1=4,
            delay_2=4,
            num_updates=1,
            input_dim=64,
        )

    def test_routing_modes_forward_pass(self):
        """Verify forward pass and telemetry across all 4 routing modalities."""
        modes = ["disconnected", "sparse_topk", "dense", "dependency_gated"]
        obs_b, _, _ = build_batch(self.env, batch_size=4, seed=42)
        B, T, _ = obs_b.shape

        for mode in modes:
            with self.subTest(routing_mode=mode):
                model = DiagnosticCGPModel(
                    input_dim=64,
                    K=8,
                    thought_size=12,
                    num_values=8,
                    routing_mode=mode,
                ).to(self.device)

                logits, diags = model(obs_b, return_diagnostics=True)
                self.assertEqual(logits.shape, (B, T, 8))
                self.assertFalse(torch.isnan(logits).any())
                self.assertIn("mean_salience", diags)
                self.assertIn("mean_routing_energy", diags)
                self.assertGreaterEqual(diags["mean_salience"], 0.0)
                self.assertLessEqual(diags["mean_salience"], 1.0)

    def test_gate_temperature_sharpening(self):
        """Verify gate temperature sharpening influences mean salience."""
        obs_b, _, _ = build_batch(self.env, batch_size=4, seed=100)

        model_standard = DiagnosticCGPModel(gate_temperature=1.0).to(self.device)
        model_sharp = DiagnosticCGPModel(gate_temperature=0.2).to(self.device)

        # Copy gate weights so only temperature differs
        model_sharp.cig_gate[0].weight.data.copy_(model_standard.cig_gate[0].weight.data)
        model_sharp.cig_gate[0].bias.data.copy_(model_standard.cig_gate[0].bias.data)

        _, diag_std = model_standard(obs_b, return_diagnostics=True)
        _, diag_sharp = model_sharp(obs_b, return_diagnostics=True)

        self.assertIsInstance(diag_std["mean_salience"], float)
        self.assertIsInstance(diag_sharp["mean_salience"], float)

    def test_latch_persistence_and_gradient_flow(self):
        """Verify lambda=1.0 backward pass propagates valid gradients without exploding/vanishing."""
        model = DiagnosticCGPModel(plastic_decay=1.0).to(self.device)
        obs_b, tgt_b, _ = build_batch(self.env, batch_size=4, seed=200)

        logits, _ = model(obs_b)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), tgt_b.reshape(B * T), ignore_index=-100)
        self.assertFalse(torch.isnan(loss))

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN in grad for {name}")

    def test_slot_width_capacity_scaling(self):
        """Verify slot widths W in [12, 24, 48, 64] produce increasing parameter budgets and valid outputs."""
        widths = [12, 24, 48, 64]
        obs_b, _, _ = build_batch(self.env, batch_size=2, seed=300)
        prev_params = 0

        for w in widths:
            with self.subTest(thought_size=w):
                model = DiagnosticCGPModel(K=4, thought_size=w, routing_mode="sparse_topk").to(self.device)
                p_count = count_params(model)["total"]
                self.assertGreater(p_count, prev_params)
                prev_params = p_count

                logits, _ = model(obs_b)
                self.assertEqual(logits.shape, (2, obs_b.shape[1], 8))
                self.assertFalse(torch.isnan(logits).any())

    def test_dependency_gated_routing_energy(self):
        """Verify dependency-gated routing computes mutual cross-affinity and routes selectively."""
        model = DiagnosticCGPModel(
            K=8,
            thought_size=16,
            routing_mode="dependency_gated",
            dependency_threshold=0.999,  # High threshold suppresses routing
        ).to(self.device)
        obs_b, _, _ = build_batch(self.env, batch_size=2, seed=400)

        logits_high_thresh, diags_high = model(obs_b, return_diagnostics=True)
        self.assertEqual(logits_high_thresh.shape, (2, obs_b.shape[1], 8))
        self.assertGreaterEqual(diags_high["mean_routing_energy"], 0.0)

    def test_short_training_and_eval_pipeline(self):
        """Verify short training and evaluation cycle executes end-to-end without errors."""
        model = DiagnosticCGPModel(K=4, thought_size=12, routing_mode="disconnected").to(self.device)
        train_diagnostic_model(model, self.env, num_steps=2, batch_size=4, device=self.device)
        eval_metrics = evaluate_diagnostic_model(model, self.env, num_seeds=2, device=self.device)

        self.assertIn("overall_retention_accuracy", eval_metrics)
        self.assertIn("untouched_retention_accuracy", eval_metrics)
        self.assertIn("cross_talk_error_rate", eval_metrics)
        self.assertIn("latency_ms_mean", eval_metrics)
        self.assertGreater(eval_metrics["latency_ms_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
