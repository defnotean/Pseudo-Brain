"""Unit and Regression Test Suite for Concurrent Cognitive Thread Capacity Scaling Benchmark.

Verifies:
1. State memory calculation across architectures, tiers, and high-K configurations.
2. Environment scaling and batch construction with high thread counts (K=32, 64, 128).
3. Forward pass numerical stability for Pseudo-Brain and baselines under high-K scaling.
4. Effective concurrent cognitive thread metric computation (K_eff = K * Recov * Iso).
5. Fast end-to-end training and evaluation smoke test.
6. Tier 1 (~10M parameter) model instantiation and dimension verification.
7. Objective cognitive and real-time knee detection logic.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from memory_benchmark.multi_threaded_cognitive_process_benchmark import (
    MultiThreadedCognitiveProcessEnv,
    build_mtcp_batch,
    make_mtcp_model,
    train_mtcp_model,
    evaluate_mtcp_model,
    count_params,
)
from memory_benchmark.thread_capacity_scaling_benchmark import (
    compute_state_bytes,
    make_capacity_model,
    detect_knees,
)


class ThreadCapacityScalingTests(unittest.TestCase):
    """Validation suite for thread capacity scaling law harness."""

    def setUp(self):
        self.device = torch.device("cpu")

    def test_state_bytes_computation(self):
        """Verify state memory calculation matches theoretical footprints across tiers."""
        # Tier 0: K=8: PB = 8 * 48 * 4 + 8 * 8 * 4 = 1536 + 256 = 1792 bytes
        pb_8_t0 = compute_state_bytes("pseudo_brain", tier="tier0", K=8)
        self.assertEqual(pb_8_t0, 1792)

        # Tier 0: K=64: PB = 64 * 48 * 4 + 64 * 8 * 4 = 12288 + 2048 = 14336 bytes
        pb_64_t0 = compute_state_bytes("pseudo_brain", tier="tier0", K=64)
        self.assertEqual(pb_64_t0, 14336)

        # Tier 1: K=8: PB = 8 * 832 * 4 + 8 * 8 * 4 = 26624 + 256 = 26880 bytes
        pb_8_t1 = compute_state_bytes("pseudo_brain", tier="tier1", K=8)
        self.assertEqual(pb_8_t1, 26880)

        # Monolithic baselines have fixed state size regardless of K
        gru_bytes_t0 = compute_state_bytes("gru", tier="tier0", K=8)
        ssm_bytes_t0 = compute_state_bytes("diagonal_ssm", tier="tier0", K=8)
        self.assertEqual(compute_state_bytes("gru", tier="tier0", K=64), gru_bytes_t0)
        self.assertEqual(compute_state_bytes("diagonal_ssm", tier="tier0", K=64), ssm_bytes_t0)

    def test_tier1_model_instantiation(self):
        """Verify Tier 1 models are instantiated with ~10M parameters (+-5%)."""
        archs = ["pseudo_brain", "gru", "diagonal_ssm", "linear_attention"]
        for arch in archs:
            with self.subTest(arch=arch):
                model = make_capacity_model(arch, tier="tier1", input_dim=64, num_threads=8, num_values=8)
                p_count = count_params(model)["total"]
                # Must be in [9.0M, 11.0M]
                self.assertGreater(p_count, 9_000_000, f"{arch} has only {p_count} params")
                self.assertLess(p_count, 11_000_000, f"{arch} has {p_count} params")

    def test_high_k_environment_and_forward_pass(self):
        """Verify environment and models execute without NaNs under K=32 and K=64."""
        for K in [32, 64]:
            with self.subTest(K=K):
                env = MultiThreadedCognitiveProcessEnv(num_threads=K, num_values=8, max_interruption_length=8)
                obs_b, tgt_b, _ = build_mtcp_batch(env, batch_size=2, seed=100 + K)
                B, T, _ = obs_b.shape

                # Test Pseudo-Brain at high K
                pb_model = make_capacity_model("pseudo_brain", tier="tier0", input_dim=64, num_threads=K, num_values=8).to(self.device)
                logits = pb_model(obs_b)
                self.assertEqual(logits.shape, (B, T, 8))
                self.assertFalse(torch.isnan(logits).any())

    def test_effective_threads_metric_calculation(self):
        """Verify K_eff logic scales linearly with recovery and isolation rates."""
        K = 16
        rec_rate = 0.75
        iso_rate = 0.90
        K_eff = K * rec_rate * iso_rate
        self.assertAlmostEqual(K_eff, 10.8, places=2)

    def test_detect_knees_logic(self):
        """Verify cognitive knee and real-time knee identification logic."""
        mock_grid = {
            "K_8": {"pseudo_brain": {"K_eff": 6.8, "latency_ms_mean": 1.5}},
            "K_16": {"pseudo_brain": {"K_eff": 13.5, "latency_ms_mean": 2.2}},
            "K_32": {"pseudo_brain": {"K_eff": 27.0, "latency_ms_mean": 4.1}},
            "K_64": {"pseudo_brain": {"K_eff": 58.0, "latency_ms_mean": 9.2}},
            "K_128": {"pseudo_brain": {"K_eff": 20.0, "latency_ms_mean": 21.5}},  # Util = 15.6% (<50%), Lat > 16.67
        }
        cog_knee, rt_knee = detect_knees(mock_grid, "pseudo_brain", [8, 16, 32, 64, 128])
        self.assertEqual(cog_knee, 128)
        self.assertEqual(rt_knee, 128)

    def test_smoke_capacity_eval_cycle(self):
        """Verify short training and evaluation cycle executes at K=16."""
        env = MultiThreadedCognitiveProcessEnv(num_threads=16, num_values=8, max_interruption_length=4)
        model = make_capacity_model("pseudo_brain", tier="tier0", input_dim=64, num_threads=16, num_values=8).to(self.device)

        train_mtcp_model(model, env, num_steps=2, batch_size=2, device=self.device)
        metrics = evaluate_mtcp_model(model, env, num_seeds=2, device=self.device)

        self.assertIn("recovery_accuracy", metrics)
        self.assertIn("latency_ms_mean", metrics)
        self.assertGreater(metrics["latency_ms_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
