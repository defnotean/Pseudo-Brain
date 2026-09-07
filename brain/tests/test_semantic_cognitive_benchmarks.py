"""Tests for Native Semantic Cognitive Benchmark and Scaling Sweeps.

Validates that:
1. All 6 architectures run reliably, report finite losses, and produce valid metric bounds.
2. Exhaustive evaluation metrics (token accuracy, preemption recovery, thread isolation,
   cross-thread dependency, K_eff, latencies, state bytes, 60Hz compliance) adhere to strict bounds.
3. Concurrency and width scaling sweeps (K, W) execute stably and respect physical memory contracts.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import torch

from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    make_semantic_model,
)
from irene_brain.semantic.tokenizer import SemanticTokenizer
from semantic_benchmark.curriculum_datasets import (
    LanguageCurriculumGenerator,
    build_curriculum_batch,
)
from semantic_benchmark.language_cognitive_benchmark import (
    evaluate_semantic_model,
    run_language_cognitive_benchmark,
    run_scaling_sweep,
    train_semantic_model,
)


class SemanticCognitiveBenchmarkTests(unittest.TestCase):
    """Test suite for semantic cognitive benchmarks and scaling sweeps."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cpu")
        self.tokenizer = SemanticTokenizer(max_threads=8)
        self.generator = LanguageCurriculumGenerator(tokenizer=self.tokenizer, max_threads=8)

    def test_benchmark_metric_keys_and_bounds(self) -> None:
        """Verify that evaluate_semantic_model returns all required keys within valid theoretical bounds."""
        model = make_semantic_model("pseudo_brain", vocab_size=self.tokenizer.vocab_size, K=4, thought_size=16)
        metrics = evaluate_semantic_model(model, self.generator, num_eval_episodes=16, batch_size=8, device=self.device)

        required_keys = [
            "parameters",
            "token_accuracy",
            "copy_accuracy",
            "binding_accuracy",
            "delayed_recall_accuracy",
            "preemption_recovery_accuracy",
            "thread_isolation_accuracy",
            "cross_thread_dependency_accuracy",
            "effective_threads_Keff",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p90_ms",
            "latency_p99_ms",
            "recurrent_state_bytes",
            "under_16_67ms",
        ]
        for k in required_keys:
            self.assertIn(k, metrics, f"Missing required metric: {k}")

        accuracy_keys = [
            "token_accuracy",
            "copy_accuracy",
            "binding_accuracy",
            "delayed_recall_accuracy",
            "preemption_recovery_accuracy",
            "thread_isolation_accuracy",
            "cross_thread_dependency_accuracy",
        ]
        for ak in accuracy_keys:
            val = metrics[ak]
            self.assertGreaterEqual(val, 0.0, f"{ak} below 0.0: {val}")
            self.assertLessEqual(val, 100.0, f"{ak} above 100.0: {val}")

        self.assertGreaterEqual(metrics["effective_threads_Keff"], 0.0)
        self.assertLessEqual(metrics["effective_threads_Keff"], 4.0)

        self.assertGreater(metrics["latency_mean_ms"], 0.0)
        self.assertGreater(metrics["latency_p50_ms"], 0.0)
        self.assertGreaterEqual(metrics["latency_p90_ms"], metrics["latency_p50_ms"] * 0.9)
        self.assertGreaterEqual(metrics["latency_p99_ms"], metrics["latency_p90_ms"] * 0.9)
        self.assertIsInstance(metrics["under_16_67ms"], bool)

    def test_all_six_architectures_run_and_evaluate(self) -> None:
        """Verify all 6 architectures initialize, train for a step, and evaluate cleanly."""
        archs = [
            "pseudo_brain",
            "pseudo_brain_no_cgp",
            "pseudo_brain_no_threads",
            "gru",
            "ssm",
            "reactive",
        ]

        for arch in archs:
            model = make_semantic_model(
                arch,
                vocab_size=self.tokenizer.vocab_size,
                K=4,
                thought_size=16,
                embed_dim=32,
                proj_dim=64,
            )
            p_count = sum(p.numel() for p in model.parameters())
            self.assertGreater(p_count, 0, f"Architecture {arch} has 0 parameters")

            losses = train_semantic_model(model, self.generator, num_steps=2, batch_size=4, device=self.device)
            self.assertEqual(len(losses), 2)
            self.assertFalse(np.isnan(losses[-1]), f"NaN loss in {arch}")

            metrics = evaluate_semantic_model(model, self.generator, num_eval_episodes=8, batch_size=4, device=self.device)
            self.assertIn("token_accuracy", metrics)
            self.assertIn("under_16_67ms", metrics)
            self.assertTrue(metrics["under_16_67ms"], f"Architecture {arch} violated 60Hz real-time SLA")

    def test_scaling_sweep_execution_and_bounds(self) -> None:
        """Verify run_scaling_sweep executes a mini-grid and enforces valid data schemas."""
        K_vals = [2, 4]
        W_vals = [12, 24]
        sweep_data = run_scaling_sweep(
            K_values=K_vals,
            W_values=W_vals,
            num_train_steps=2,
            device_str="cpu",
        )

        self.assertIn("sweep_config", sweep_data)
        self.assertIn("grid", sweep_data)

        for K in K_vals:
            for W in W_vals:
                key = f"K{K}_W{W}"
                self.assertIn(key, sweep_data["grid"])
                cell = sweep_data["grid"][key]
                self.assertEqual(cell["K"], K)
                self.assertEqual(cell["W"], W)
                self.assertGreaterEqual(cell["effective_threads_Keff"], 0.0)
                self.assertLessEqual(cell["effective_threads_Keff"], float(K))
                self.assertTrue(cell["under_16_67ms"])

    def test_recurrent_state_memory_contract(self) -> None:
        """Verify theoretical state byte calculation adheres to the K*W*4 + K*V*4 contract."""
        K = 16
        W = 32
        V = self.tokenizer.vocab_size
        model = make_semantic_model("pseudo_brain", vocab_size=V, K=K, thought_size=W)
        metrics = evaluate_semantic_model(model, self.generator, num_eval_episodes=4, batch_size=4, device=self.device)

        expected_bytes = K * W * 4 + K * V * 4
        self.assertEqual(metrics["recurrent_state_bytes"], expected_bytes)

        gru_model = make_semantic_model("gru", vocab_size=V)
        gru_metrics = evaluate_semantic_model(gru_model, self.generator, num_eval_episodes=4, batch_size=4, device=self.device)
        self.assertEqual(gru_metrics["recurrent_state_bytes"], 256 * 4)

        reactive_model = make_semantic_model("reactive", vocab_size=V)
        react_metrics = evaluate_semantic_model(reactive_model, self.generator, num_eval_episodes=4, batch_size=4, device=self.device)
        self.assertEqual(react_metrics["recurrent_state_bytes"], 0)

    def test_monolithic_single_slot_thread_clamping(self) -> None:
        """Verify pseudo_brain_no_threads safely handles multi-thread token sequences without out-of-bounds index errors."""
        model = make_semantic_model("pseudo_brain_no_threads", vocab_size=self.tokenizer.vocab_size, K=8, thought_size=16)
        tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        threads = torch.tensor([[0, 3, 7, 2]], dtype=torch.long)
        logits = model(tokens, thread_seq=threads)
        self.assertEqual(logits.shape, (1, 4, self.tokenizer.vocab_size))


if __name__ == "__main__":
    unittest.main()
