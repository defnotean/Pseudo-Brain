"""Unit and regression test suite for Native Streaming Conversational CLI and Telemetry.

Verifies:
1. Multi-turn dialogue execution with zero token buffer replay.
2. Latent state persistence across preemption and cross-thread isolation.
3. Latency compliance under the 60 Hz frame budget (<16.67 ms).
4. Subsystem failure attribution health across all 6 cognitive layers.
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
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.tokenizer import SemanticTokenizer
from semantic_benchmark.curriculum_datasets import LanguageCurriculumGenerator
from semantic_benchmark.cli_chat import train_conversational_calibration

try:
    from experiments.benchmarks.streaming_conversational_benchmark import (
        evaluate_input_encoding,
        evaluate_semantic_representation,
        evaluate_thread_selection,
        evaluate_memory_persistence,
        evaluate_cross_thread_interference,
        evaluate_output_decoding,
    )
except ImportError:
    from benchmarks.streaming_conversational_benchmark import (
        evaluate_input_encoding,
        evaluate_semantic_representation,
        evaluate_thread_selection,
        evaluate_memory_persistence,
        evaluate_cross_thread_interference,
        evaluate_output_decoding,
    )


class StreamingConversationalCLITests(unittest.TestCase):
    """Validation test suite for Pseudo-Brain streaming conversational engine."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.threads = 16
        cls.tokenizer = SemanticTokenizer(max_threads=cls.threads)
        cls.model = make_semantic_model("pseudo_brain", vocab_size=cls.tokenizer.vocab_size, K=cls.threads, thought_size=32)
        
        # Fast calibration
        generator = LanguageCurriculumGenerator(tokenizer=cls.tokenizer, max_threads=cls.threads)
        train_conversational_calibration(cls.model, generator, num_steps=140, batch_size=4, lr=3e-3, device=cls.device)
        cls.model.eval()

    def setUp(self):
        self.session = StreamingCognitiveSession(model=self.model, tokenizer=self.tokenizer, device=self.device)

    def test_scripted_session_zero_buffer_replay(self):
        """Verify 6-turn scripted conversation succeeds from latent state without buffer replay."""
        turns = [
            # Turn 1: Enroll Fact on Thread 0
            {"thread_id": 0, "user_text": "[THREAD:0]Remember Alice likes coffee. [RESP]", "expected": "coffee"},
            # Turn 2: Enroll Fact on Thread 1
            {"thread_id": 1, "user_text": "[THREAD:1]Remember Bob likes tea. [RESP]", "expected": "tea"},
            # Turn 3: Query Thread 0
            {"thread_id": 0, "user_text": "[THREAD:0]What does Alice like? [RESP]", "expected": "coffee"},
            # Turn 4: Preemption - Long task on Thread 2
            {"thread_id": 2, "user_text": "[THREAD:2]Let's plan a trip to Tokyo. Step 1: book flights. Step 2: hotel. [RESP]", "expected": "explore tokyo"},
            # Turn 5: Query Thread 1 after preemption
            {"thread_id": 1, "user_text": "[THREAD:1]What does Bob like? [RESP]", "expected": "tea"},
            # Turn 6: Resume Thread 2
            {"thread_id": 2, "user_text": "[THREAD:2]Continue the Tokyo plan. Step 3: [RESP]", "expected": "explore tokyo"},
        ]

        prev_total_tokens = 0
        for t_idx, turn in enumerate(turns, 1):
            res = self.session.generate_response(
                prompt_text=turn["user_text"],
                thread_id=turn["thread_id"],
                max_new_tokens=16,
                temperature=0.0,
            )
            resp = res["response_text"].strip().lower()
            expected = turn["expected"].lower()

            # Verify response matches expected
            self.assertTrue(
                resp == expected or expected in resp,
                f"Turn {t_idx} failed: got '{resp}', expected '{expected}'"
            )

            # Verify active thread
            self.assertEqual(res["active_thread"], turn["thread_id"])

            # Verify zero token buffer replay: session ingested strictly this turn's prompt + generated tokens
            turn_tokens = len(res["prompt_latencies_ms"]) + len(res["generation_latencies_ms"])
            self.assertEqual(self.session.total_tokens_processed, prev_total_tokens + turn_tokens)
            prev_total_tokens = self.session.total_tokens_processed

    def test_latency_bounds_under_60hz(self):
        """Verify per-token latency satisfies 60 Hz real-time budget (<16.67 ms)."""
        # Ingest a sequence and generate
        res = self.session.generate_response(
            prompt_text="[THREAD:0]Remember Alice likes coffee. [RESP]",
            thread_id=0,
            max_new_tokens=16,
        )

        all_lats = res["prompt_latencies_ms"] + res["generation_latencies_ms"]
        mean_lat = float(np.mean(all_lats))
        p90_lat = float(np.percentile(all_lats, 90))
        p99_lat = float(np.percentile(all_lats, 99))

        # Must be well under 16.67 ms SLA
        self.assertLess(mean_lat, 5.0, f"Mean latency {mean_lat:.2f}ms exceeded 5ms bound")
        self.assertLess(p90_lat, 8.0, f"p90 latency {p90_lat:.2f}ms exceeded 8ms bound")
        self.assertLess(p99_lat, 16.67, f"p99 latency {p99_lat:.2f}ms exceeded 16.67ms bound")

        telem = self.session.get_telemetry()
        self.assertTrue(telem["latency_under_16_67ms"])

    def test_failure_attribution_subsystems(self):
        """Verify 0% failure rate across all 6 cognitive layers."""
        attr_encoding = evaluate_input_encoding(self.tokenizer)
        self.assertEqual(attr_encoding["status"], "PASS")
        self.assertEqual(attr_encoding["failures"], 0)

        attr_representation = evaluate_semantic_representation(self.model, self.device)
        self.assertEqual(attr_representation["status"], "PASS")
        self.assertEqual(attr_representation["failures"], 0)

        attr_thread = evaluate_thread_selection(self.model, self.tokenizer, self.device)
        self.assertEqual(attr_thread["status"], "PASS")
        self.assertEqual(attr_thread["failures"], 0)

        attr_persistence = evaluate_memory_persistence(self.model, self.tokenizer, self.device)
        self.assertEqual(attr_persistence["status"], "PASS")
        self.assertEqual(attr_persistence["failures"], 0)

        attr_interference = evaluate_cross_thread_interference(self.model, self.tokenizer, self.device)
        self.assertEqual(attr_interference["status"], "PASS")
        self.assertEqual(attr_interference["failures"], 0)

        attr_decoding = evaluate_output_decoding(self.session)
        self.assertEqual(attr_decoding["status"], "PASS")
        self.assertEqual(attr_decoding["failures"], 0)


if __name__ == "__main__":
    unittest.main()
