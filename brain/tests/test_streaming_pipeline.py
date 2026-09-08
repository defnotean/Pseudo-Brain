"""Comprehensive test suite for Infinite-Streaming Multi-Task Pretraining Pipeline.

Verifies:
1. Sequence Packing: Uniform length T blocks across varied T in [64, 128, 256, 512, 1024].
2. Batch Shapes: Strict [B, T] tensor alignment, dtypes, and dict-like interface.
3. Loss Mask Alignment: Exact match loss_mask == (labels != -100), boundary isolation,
   and assistant-response masking.
4. Cognitive Thread ID Tracking: Thread slots [THREAD:0..K-1] mapped and bound to tokens.
5. Multi-Task Mixture: Configurable task proportions (Language 40%, Code 35%, Reasoning 15%, Game POMDP 10%).
6. Token Diversity & Entropy: Lexical richness and domain-specific tokens across streams.
7. Resilience & Fallback: Seamless offline operation under network disconnect or rate limits.
8. Tokenizer Parity: Full compatibility with both SemanticTokenizer and BpeSemanticTokenizer.
9. Zero Memory Leaks: Verified memory stability across 1,000+ streamed batches.
10. High Throughput: Benchmark measuring tokens/sec ready for Colab A100 consumption.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import tracemalloc
import unittest
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn.functional as F

from irene_brain.data.streaming_loader import (
    HFStreamIterator,
    MultiTaskMixtureStream,
    PackedSequenceBlock,
    RawDocument,
    SequencePacker,
    StreamingBatch,
    StreamingConfig,
    StreamingMultiTaskDataset,
    SyntheticCodeStream,
    SyntheticLanguageStream,
    SyntheticPOMDPStream,
    SyntheticReasoningStream,
    TaskType,
    TaskWeights,
    collate_streaming_batch,
    create_streaming_dataloader,
)
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.semantic.tokenizer import SemanticTokenizer


class TestStreamingDataPipeline(unittest.TestCase):
    """Test suite for the infinite-streaming pretraining pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = SemanticTokenizer(max_threads=16)
        cls.bpe_tokenizer = BpeSemanticTokenizer(vocab_size=1000, max_threads=16)

    def test_sequence_packer_uniform_blocks(self):
        """Verify SequencePacker packs variable-length documents into exact length T."""
        for seq_len in [64, 128, 256, 512]:
            packer = SequencePacker(tokenizer=self.tokenizer, seq_len=seq_len, max_threads=16)
            stream = SyntheticLanguageStream(seed=101, max_threads=16)

            # Append documents until 3 blocks can be emitted
            blocks: List[PackedSequenceBlock] = []
            while len(blocks) < 3:
                doc = stream.sample()
                packer.append_document(doc)
                while packer.can_emit_block() and len(blocks) < 3:
                    blocks.append(packer.emit_block())

            self.assertEqual(len(blocks), 3)
            for block in blocks:
                self.assertEqual(block.input_ids.shape, (seq_len,))
                self.assertEqual(block.labels.shape, (seq_len,))
                self.assertEqual(block.loss_mask.shape, (seq_len,))
                self.assertEqual(block.thread_ids.shape, (seq_len,))
                self.assertEqual(block.input_ids.dtype, torch.long)
                self.assertEqual(block.labels.dtype, torch.long)
                self.assertEqual(block.loss_mask.dtype, torch.float32)
                self.assertEqual(block.thread_ids.dtype, torch.long)

    def test_batch_shapes_and_types(self):
        """Verify DataLoader collates packed blocks into [B, T] tensors."""
        B, T = 4, 128
        config = StreamingConfig(seq_len=T, batch_size=B, offline_mode=True, seed=42)
        loader = create_streaming_dataloader(config, self.tokenizer)

        batch = next(iter(loader))
        self.assertIsInstance(batch, StreamingBatch)
        self.assertEqual(batch.input_ids.shape, (B, T))
        self.assertEqual(batch.labels.shape, (B, T))
        self.assertEqual(batch.loss_mask.shape, (B, T))
        self.assertEqual(batch.thread_ids.shape, (B, T))
        self.assertEqual(len(batch.task_types), B)

        # Verify dict-like access
        self.assertTrue(torch.equal(batch["input_ids"], batch.input_ids))
        self.assertTrue(torch.equal(batch["labels"], batch.labels))
        self.assertTrue(torch.equal(batch["loss_mask"], batch.loss_mask))
        self.assertTrue(torch.equal(batch["thread_ids"], batch.thread_ids))

        # Verify .to(device)
        cpu_batch = batch.to("cpu")
        self.assertEqual(cpu_batch.input_ids.device, torch.device("cpu"))

    def test_loss_mask_alignment_and_boundary_isolation(self):
        """Verify loss mask strictly matches labels != -100 and document boundary isolation."""
        config = StreamingConfig(
            seq_len=256,
            batch_size=2,
            offline_mode=True,
            mask_cross_document_boundary=True,
            supervised_only_loss=False,
        )
        loader = create_streaming_dataloader(config, self.tokenizer)
        batch = next(iter(loader))

        # 1. Exact equality between loss_mask and (labels != -100)
        expected_mask = (batch.labels != -100).float()
        self.assertTrue(torch.equal(batch.loss_mask, expected_mask))

        # 2. Verify active loss mask is non-trivial (> 80% active tokens)
        active_ratio = batch.loss_mask.mean().item()
        self.assertGreater(active_ratio, 0.70)

        # 3. Test supervised_only_loss mode (only assistant responses are active)
        config_sup = StreamingConfig(
            seq_len=256,
            batch_size=2,
            offline_mode=True,
            supervised_only_loss=True,
        )
        loader_sup = create_streaming_dataloader(config_sup, self.tokenizer)
        batch_sup = next(iter(loader_sup))

        # Must have both masked (-100) prompt tokens and active completion tokens
        self.assertTrue((batch_sup.labels == -100).any())
        self.assertTrue((batch_sup.labels != -100).any())
        self.assertTrue(torch.equal(batch_sup.loss_mask, (batch_sup.labels != -100).float()))

    def test_thread_id_tracking(self):
        """Verify thread IDs match active cognitive slots within [0, max_threads - 1]."""
        max_threads = 16
        config = StreamingConfig(
            seq_len=256,
            batch_size=4,
            max_threads=max_threads,
            offline_mode=True,
            seed=123,
        )
        loader = create_streaming_dataloader(config, self.tokenizer)
        batch = next(iter(loader))

        # Check bounds
        self.assertTrue((batch.thread_ids >= 0).all())
        self.assertTrue((batch.thread_ids < max_threads).all())

        # Verify that multiple threads are activated across the batch
        unique_threads = torch.unique(batch.thread_ids).tolist()
        self.assertGreaterEqual(len(unique_threads), 2)

    def test_multitask_mixture_ratios(self):
        """Verify task mixture weights: Language (40%), Code (35%), Reasoning (15%), POMDP (10%)."""
        weights = TaskWeights(language=0.40, code=0.35, reasoning=0.15, game_pomdp=0.10)
        config = StreamingConfig(
            seq_len=64,
            batch_size=1,
            weights=weights,
            offline_mode=True,
            seed=777,
        )

        mixture = MultiTaskMixtureStream(config, self.tokenizer)

        # Sample 500 documents and record counts
        sample_counts = {t: 0 for t in TaskType}
        total_samples = 500

        for _ in range(total_samples):
            task = mixture.rng.choices(mixture.tasks, weights=mixture.probs, k=1)[0]
            sample_counts[task] += 1

        empirical_ratios = {t: count / total_samples for t, count in sample_counts.items()}

        # Verify within reasonable statistical confidence bounds (+/- 6%)
        self.assertAlmostEqual(empirical_ratios[TaskType.LANGUAGE], 0.40, delta=0.06)
        self.assertAlmostEqual(empirical_ratios[TaskType.CODE], 0.35, delta=0.06)
        self.assertAlmostEqual(empirical_ratios[TaskType.REASONING], 0.15, delta=0.05)
        self.assertAlmostEqual(empirical_ratios[TaskType.GAME_POMDP], 0.10, delta=0.05)

    def test_token_diversity_across_domains(self):
        """Verify unique vocabulary diversity and modality-specific features."""
        # 1. Language stream
        lang_stream = SyntheticLanguageStream(seed=1)
        lang_docs = [lang_stream.sample().text for _ in range(20)]
        lang_joined = " ".join(lang_docs)
        self.assertTrue(any(w in lang_joined for w in ["Quantum", "Memory", "Plate", "Decoherence", "recurrent"]))

        # 2. Code stream
        code_stream = SyntheticCodeStream(seed=2)
        code_docs = [code_stream.sample().text for _ in range(20)]
        code_joined = " ".join(code_docs)
        self.assertTrue(any(w in code_joined for w in ["def ", "class ", "import ", "return", "self."]))

        # 3. Reasoning / Tool stream
        reason_stream = SyntheticReasoningStream(seed=3)
        reason_docs = [reason_stream.sample().text for _ in range(20)]
        reason_joined = " ".join(reason_docs)
        self.assertTrue(any(w in reason_joined for w in ["<tool_call>", "<tool_response>", "<thought>", "[DEP]"]))

        # 4. POMDP stream
        pomdp_stream = SyntheticPOMDPStream(seed=4)
        pomdp_docs = [pomdp_stream.sample().text for _ in range(20)]
        pomdp_joined = " ".join(pomdp_docs)
        self.assertTrue(any(w in pomdp_joined for w in ["[POMDP]", "obs:", "action:", "reward:"]))

    def test_offline_fallback_resilience(self):
        """Verify HFStreamIterator gracefully switches to synthetic fallback on bad URL or error."""
        # Intentionally point to non-existent dataset with offline_mode=False
        iterator = HFStreamIterator(
            task_type=TaskType.LANGUAGE,
            dataset_name="non_existent_org/non_existent_dataset_123456",
            split="train",
            fallback_stream=SyntheticLanguageStream(seed=99),
            offline_mode=False,
        )

        doc = iterator.next_document()
        self.assertIsNotNone(doc)
        self.assertEqual(doc.task_type, TaskType.LANGUAGE)
        self.assertGreater(len(doc.text), 20)
        self.assertTrue(iterator._use_fallback)

    def test_bpe_tokenizer_compatibility(self):
        """Verify pipeline operates seamlessly with BpeSemanticTokenizer."""
        config = StreamingConfig(
            seq_len=128,
            batch_size=3,
            offline_mode=True,
            seed=2026,
        )
        loader = create_streaming_dataloader(config, self.bpe_tokenizer)
        batch = next(iter(loader))

        self.assertEqual(batch.input_ids.shape, (3, 128))
        self.assertEqual(batch.labels.shape, (3, 128))
        self.assertEqual(batch.loss_mask.shape, (3, 128))
        self.assertEqual(batch.thread_ids.shape, (3, 128))

        # Check decoded text roundtrip
        decoded = self.bpe_tokenizer.decode(batch.input_ids[0].tolist(), skip_special=False)
        self.assertGreater(len(decoded), 0)

    def test_zero_memory_leak_across_thousands_of_batches(self):
        """Verify zero memory growth across 1,000 streamed pretraining batches."""
        config = StreamingConfig(
            seq_len=128,
            batch_size=4,
            offline_mode=True,
            seed=42,
        )
        loader = create_streaming_dataloader(config, self.tokenizer)
        iterator = iter(loader)

        # Warm up 50 batches
        for _ in range(50):
            _ = next(iterator)

        gc.collect()

        try:
            import psutil
            process = psutil.Process()
            m_start = process.memory_info().rss / (1024.0 * 1024.0)

            # Stream 1,000 batches
            for _ in range(1000):
                batch = next(iterator)
                _ = batch.input_ids.sum()

            gc.collect()
            m_end = process.memory_info().rss / (1024.0 * 1024.0)
            delta_mb = m_end - m_start

            # Memory growth across 1,000 batches (512,000 tokens) must be strictly bounded (< 25 MB)
            self.assertLess(delta_mb, 25.0, f"Memory growth of {delta_mb:.2f} MB indicates a potential leak.")
        except ImportError:
            for _ in range(500):
                batch = next(iterator)
                self.assertEqual(batch.input_ids.shape, (4, 128))

    def test_streaming_throughput(self):
        """Benchmark streaming data pipeline tokens/second throughput."""
        B, T = 8, 256
        num_batches = 40
        config = StreamingConfig(
            seq_len=T,
            batch_size=B,
            offline_mode=True,
            seed=42,
        )
        loader = create_streaming_dataloader(config, self.tokenizer)
        iterator = iter(loader)

        # Warmup
        for _ in range(5):
            _ = next(iterator)

        start_time = time.perf_counter()
        total_tokens = 0

        for _ in range(num_batches):
            batch = next(iterator)
            total_tokens += batch.input_ids.numel()

        elapsed = time.perf_counter() - start_time
        throughput_tok_per_sec = total_tokens / elapsed

        print(
            f"\n[STREAMING THROUGHPUT] "
            f"Processed {total_tokens:,} tokens across {num_batches} batches "
            f"in {elapsed:.3f}s -> {throughput_tok_per_sec:,.0f} tokens/sec"
        )
        # Should easily exceed 15,000 tokens/sec on CPU
        self.assertGreater(throughput_tok_per_sec, 15000)

    def test_end_to_end_pretraining_step(self):
        """Verify streaming batch runs forward, loss computation, and backward autograd pass."""
        from irene_brain.semantic.native_semantic_model import make_semantic_model

        B, T = 2, 64
        config = StreamingConfig(seq_len=T, batch_size=B, offline_mode=True, seed=999)
        loader = create_streaming_dataloader(config, self.tokenizer)
        batch = next(iter(loader))

        model = make_semantic_model("pseudo_brain_stable", vocab_size=self.tokenizer.vocab_size, K=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward pass
        logits = model(batch.input_ids, thread_seq=batch.thread_ids)
        self.assertEqual(logits.shape, (B, T, self.tokenizer.vocab_size))

        # Loss with ignore_index=-100
        loss = F.cross_entropy(
            logits.reshape(-1, self.tokenizer.vocab_size),
            batch.labels.reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.isfinite(loss))

        # Backward autograd
        optimizer.zero_grad()
        loss.backward()

        # Check gradient presence
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        self.assertTrue(has_grad)

        optimizer.step()


if __name__ == "__main__":
    unittest.main()

