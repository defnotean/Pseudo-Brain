"""Unit and Regression Test Suite for Multi-Threaded Cognitive Process Benchmark (MTCP-Bench).

Verifies:
1. MultiThreadedCognitiveProcessEnv episode sequencing:
   - Enrollment, pipeline execution, interruption/preemption bursts,
   - Pipeline resumption, cross-thread dependencies, delayed consequences,
   - Multi-type queries (recovery, dependency, interference, retention).
2. Tensor builder batch dimensions and target masking (-100 on non-queries).
3. All 4 model architectures instantiate, forward pass without NaNs, and produce valid shapes:
   - Pseudo-Brain CGP (with slot-targeted query readout),
   - Monolithic GRU,
   - Modern Diagonal SSM (GLRU),
   - Recurrent Linear Attention.
4. End-to-end training and evaluation smoke test with compound cognitive score computation.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn.functional as F

from memory_benchmark.multi_threaded_cognitive_process_benchmark import (
    MultiThreadedCognitiveProcessEnv,
    MTCPOp,
    build_mtcp_batch,
    make_mtcp_model,
    train_mtcp_model,
    evaluate_mtcp_model,
    count_params,
)


class MultiThreadedCognitiveProcessTests(unittest.TestCase):
    """Test suite for MTCP environment, architectures, and metrics."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.env = MultiThreadedCognitiveProcessEnv(
            num_threads=8,
            num_values=8,
            input_dim=64,
            max_interruption_length=10,
        )

    def test_environment_structure_and_ground_truth(self):
        """Verify episode generates structured multi-threaded workflow with valid ground truth."""
        rng = np.random.RandomState(42)
        ep = self.env.generate_episode(rng)

        # Episode must have non-zero length and advance through stages
        self.assertGreater(len(ep), 20)
        ops = [step.op_type for step in ep]
        self.assertIn(MTCPOp.DATA, ops)
        self.assertIn(MTCPOp.PIPELINE_FETCH, ops)
        self.assertIn(MTCPOp.PIPELINE_VERIFY, ops)
        self.assertIn(MTCPOp.DEPENDENT_STEP, ops)
        self.assertIn(MTCPOp.DELAYED_CONSEQUENCE, ops)
        self.assertIn(MTCPOp.QUERY, ops)

        # Query verification
        queries = [s for s in ep if s.is_query]
        self.assertGreaterEqual(len(queries), 4)
        query_types = {q.query_type for q in queries}
        self.assertIn("recovery", query_types)
        self.assertIn("dependency", query_types)
        self.assertIn("interference", query_types)
        self.assertIn("retention", query_types)

        for q in queries:
            self.assertTrue(0 <= q.target < 8)

    def test_batch_builder_shapes_and_masks(self):
        """Verify batch builder creates correct tensor dimensions and sets ignore index."""
        obs_b, tgt_b, eps = build_mtcp_batch(self.env, batch_size=4, seed=123)
        self.assertEqual(obs_b.shape[0], 4)
        self.assertEqual(obs_b.shape[2], 64)
        self.assertEqual(tgt_b.shape, (4, obs_b.shape[1]))

        # Check that query steps have valid targets and non-queries are -100
        for b in range(4):
            ep = eps[b]
            for t, step in enumerate(ep):
                if step.is_query:
                    self.assertEqual(tgt_b[b, t].item(), step.target)
                else:
                    self.assertEqual(tgt_b[b, t].item(), -100)

    def test_architectures_forward_pass(self):
        """Verify all 4 competitor architectures execute forward pass without NaNs."""
        archs = ["pseudo_brain", "gru", "diagonal_ssm", "linear_attention"]
        obs_b, _, _ = build_mtcp_batch(self.env, batch_size=2, seed=456)
        B, T, _ = obs_b.shape

        for arch in archs:
            with self.subTest(arch=arch):
                model = make_mtcp_model(arch, input_dim=64, num_threads=8, num_values=8).to(self.device)
                logits = model(obs_b)
                self.assertEqual(logits.shape, (B, T, 8))
                self.assertFalse(torch.isnan(logits).any(), f"NaN detected in {arch}")

    def test_parameter_counts_budget_matching(self):
        """Verify models are reasonably matched in capacity."""
        pb = count_params(make_mtcp_model("pseudo_brain", num_threads=8))["total"]
        gru = count_params(make_mtcp_model("gru", num_threads=8))["total"]
        ssm = count_params(make_mtcp_model("diagonal_ssm", num_threads=8))["total"]
        lin = count_params(make_mtcp_model("linear_attention", num_threads=8))["total"]

        # All models should be in the ~100k to ~350k range for fair comparison
        for name, p in [("pseudo_brain", pb), ("gru", gru), ("ssm", ssm), ("linear_attention", lin)]:
            self.assertGreater(p, 100_000, f"{name} param count too low: {p}")
            self.assertLess(p, 400_000, f"{name} param count too high: {p}")

    def test_end_to_end_train_and_eval_pipeline(self):
        """Verify short training and evaluation pipeline computes compound cognitive score."""
        model = make_mtcp_model("pseudo_brain", input_dim=64, num_threads=8, num_values=8).to(self.device)
        train_mtcp_model(model, self.env, num_steps=2, batch_size=4, device=self.device)
        metrics = evaluate_mtcp_model(model, self.env, num_seeds=2, device=self.device)

        self.assertIn("retention_accuracy", metrics)
        self.assertIn("interference_accuracy", metrics)
        self.assertIn("cross_talk_error", metrics)
        self.assertIn("recovery_accuracy", metrics)
        self.assertIn("dependency_accuracy", metrics)
        self.assertIn("latency_ms_mean", metrics)
        self.assertIn("compound_cognitive_score", metrics)
        self.assertGreaterEqual(metrics["compound_cognitive_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
