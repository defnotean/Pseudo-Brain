"""Unit and Regression Test Suite for Cognitive Scaling Ladder Benchmark.

Verifies:
1. Tier factory construction for CGP, GRU, and GLRU architectures.
2. Monotonic parameter scaling across tiers (131k < 500k < 2M < 8M).
3. Forward pass tensor shapes (B, T, num_values) and numerical stability (no NaNs).
4. Training and evaluation harness integration smoke test.
"""

from __future__ import annotations

import unittest
import torch

from memory_benchmark.multi_threaded_latent_dependency_benchmark import (
    MultiThreadedLatentDependencyEnv,
    build_batch,
    count_params,
)
from memory_benchmark.cognitive_scaling_ladder_benchmark import (
    create_model_for_tier,
    train_scaling_model,
    evaluate_scaling_model,
)


class CognitiveScalingLadderTests(unittest.TestCase):
    """Validation suite for multi-scale model definitions and execution harness."""

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

    def test_model_construction_and_parameter_monotonicity(self):
        """Verify parameter scaling order across tiers for all architectures."""
        tiers = ["131k", "500k", "2M", "8M"]
        archs = ["cgp", "gru", "glru"]

        for arch in archs:
            prev_params = 0
            for t_name in tiers:
                with self.subTest(arch=arch, tier=t_name):
                    model = create_model_for_tier(arch, t_name, input_dim=64, num_values=8, K=4)
                    p_count = count_params(model)["total"]
                    self.assertGreater(
                        p_count,
                        prev_params,
                        f"Tier {t_name} did not have more parameters than previous tier for {arch}",
                    )
                    prev_params = p_count

    def test_forward_pass_numerical_stability(self):
        """Verify forward pass generates valid logits without NaNs for all model types."""
        obs_b, _, _ = build_batch(self.env, batch_size=2, seed=42)
        B, T, _ = obs_b.shape

        for arch in ["cgp", "gru", "glru"]:
            with self.subTest(arch=arch):
                model = create_model_for_tier(arch, "131k", input_dim=64, num_values=8, K=4)
                logits = model(obs_b)
                self.assertEqual(logits.shape, (B, T, 8))
                self.assertFalse(torch.isnan(logits).any())

    def test_smoke_training_and_eval_pipeline(self):
        """Verify fast end-to-end training and evaluation cycle."""
        model = create_model_for_tier("cgp", "131k", input_dim=64, num_values=8, K=4)
        train_scaling_model(model, self.env, num_steps=2, batch_size=4, device=self.device)
        eval_metrics = evaluate_scaling_model(model, self.env, num_seeds=2, device=self.device)

        self.assertIn("overall_retention_accuracy", eval_metrics)
        self.assertIn("untouched_retention_accuracy", eval_metrics)
        self.assertIn("cross_talk_error_rate", eval_metrics)
        self.assertIn("latency_ms_mean", eval_metrics)
        self.assertGreater(eval_metrics["latency_ms_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
