"""Unit and integration test suite for Hugging Face Conversational Scaling and Tier 2 DirectML Execution.

Verifies:
1. HFConversationalLoader ingestion and schema parsing (UltraChat & Alpaca).
2. Cognitive stream transformation with synthetic preemption and assistant response masking.
3. Tier 2 model specification, parameter count (>10M), and Law 1 32 KB state memory contract.
4. DirectML forward and backward autograd gradient flow.
5. Streaming multi-turn conversation with zero conversation token replay buffer.
6. 60 Hz latency budget compliance (<16.67 ms).
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from irene_brain.device import (
    get_directml_device,
    is_directml_available,
)
from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    make_semantic_model,
)
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.tokenizer import SemanticTokenizer
from semantic_benchmark.hf_dataset_loader import HFConversationalLoader
from semantic_benchmark.llm_data_transform import LLMDataTransformer, RawDialogueTurn, RawDialogue


class TestTier2ConversationalHF(unittest.TestCase):
    """Test suite for HF Conversational Scaling and Tier 2 DirectML execution."""

    @classmethod
    def setUpClass(cls):
        cls.dml_device = get_directml_device()
        cls.cpu_device = torch.device("cpu")
        cls.threads = 16
        cls.tokenizer = SemanticTokenizer(max_threads=cls.threads)

    def test_hf_loader_ingestion_and_fallback(self):
        """Verify HFConversationalLoader fetches or generates valid multi-turn dialogues."""
        loader = HFConversationalLoader()
        corpus = loader.fetch_combined_corpus(num_ultrachat=3, num_alpaca=3, use_cache=True)
        self.assertEqual(len(corpus), 6)
        for item in corpus:
            self.assertIn("messages", item)
            self.assertGreaterEqual(len(item["messages"]), 2)
            roles = [m["role"] for m in item["messages"]]
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)

    def test_cognitive_stream_transformation(self):
        """Verify transforming raw dialogues into multi-threaded cognitive streams with preemption."""
        loader = HFConversationalLoader()
        transformer = LLMDataTransformer(max_threads=self.threads, tokenizer=self.tokenizer)
        raw_dicts = loader.fetch_combined_corpus(num_ultrachat=2, num_alpaca=2, use_cache=True)
        raw_dialogues = transformer.parse_standard_dialogue_json(raw_dicts)
        self.assertGreaterEqual(len(raw_dialogues), 2)

        episodes = transformer.transform_corpus(
            raw_dialogues=raw_dialogues,
            num_episodes=4,
            threads_per_episode=2,
            seed=42,
            paradigm="model_d",
        )
        self.assertEqual(len(episodes), 4)
        for ep in episodes:
            self.assertIsNotNone(ep.tokens)
            self.assertIsNotNone(ep.targets)
            self.assertIsNotNone(ep.threads)
            self.assertEqual(len(ep.tokens), len(ep.targets))
            self.assertEqual(len(ep.tokens), len(ep.threads))
            # Verify target masking: at least one token is ignored (-100) and at least one is active
            self.assertTrue(any(t == -100 for t in ep.targets))
            self.assertTrue(any(t != -100 for t in ep.targets))

    def test_tier2_architecture_spec_and_state_bytes(self):
        """Verify Tier 2 model specification, parameters, and 32 KB state memory contract."""
        # Test full Tier 2 model (proj_dim=2048, rank=32, num_deep_layers=2)
        model = make_semantic_model(
            "tier2",
            vocab_size=self.tokenizer.vocab_size,
            K=16,
            proj_dim=2048,
            rank=32,
            num_deep_layers=2,
        )
        counts = model.count_parameters()
        self.assertGreaterEqual(counts["total"], 10_000_000)
        self.assertEqual(counts["total"], counts["trainable"])

        # Law 1 contract: state memory must be strictly <= 32 KB (at K=128 W=64 it is 32 KB, at K=16 W=64 it is 4 KB)
        state_bytes = model.state_bytes()
        self.assertLessEqual(state_bytes, 32768)
        self.assertEqual(state_bytes, 16 * 64 * 4)

    def test_tier2_forward_backward_pass(self):
        """Verify Tier 2 forward pass and backward autograd gradient calculation."""
        device = self.dml_device if self.dml_device is not None else self.cpu_device
        model = make_semantic_model(
            "tier2",
            vocab_size=self.tokenizer.vocab_size,
            K=16,
            proj_dim=512,  # Compact size for fast unit test
            rank=16,
            num_deep_layers=1,
        ).to(device)

        B, T = 2, 8
        tokens = torch.randint(0, self.tokenizer.vocab_size, (B, T), device=device)
        threads = torch.zeros(B, T, dtype=torch.long, device=device)

        out = model(tokens, thread_seq=threads)
        self.assertEqual(out.shape, (B, T, self.tokenizer.vocab_size))

        loss = out.sum()
        loss.backward()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_streaming_session_zero_buffer_replay(self):
        """Verify streaming generation executes with zero buffer replay and sub-16.67ms latency."""
        device = self.dml_device if self.dml_device is not None else self.cpu_device
        model = make_semantic_model(
            "tier2",
            vocab_size=self.tokenizer.vocab_size,
            K=16,
            proj_dim=512,
            rank=16,
            num_deep_layers=1,
        ).to(device)
        model.eval()

        session = StreamingCognitiveSession(model=model, tokenizer=self.tokenizer, device=device)

        # Turn 1: Enroll fact
        res1 = session.generate_response(
            prompt_text="[THREAD:0]Hello, what can you do? [RESP]",
            thread_id=0,
            max_new_tokens=8,
            temperature=0.0,
        )
        self.assertIn("response_text", res1)
        self.assertEqual(res1["active_thread"], 0)
        self.assertLess(res1["mean_step_latency_ms"], 16.67)

        # Turn 2: Switch thread with preemption
        prev_processed = session.total_tokens_processed
        res2 = session.generate_response(
            prompt_text="[THREAD:1]Urgent task on thread 1. [RESP]",
            thread_id=1,
            max_new_tokens=8,
            temperature=0.0,
        )
        self.assertEqual(res2["active_thread"], 1)

        # Verify zero buffer replay: strictly turn 2 tokens were ingested
        expected_turn2_tokens = len(res2["prompt_latencies_ms"]) + len(res2["generation_latencies_ms"])
        self.assertEqual(session.total_tokens_processed, prev_processed + expected_turn2_tokens)


if __name__ == "__main__":
    unittest.main()
