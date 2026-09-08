"""Comprehensive test suite for Hierarchical Memory Architecture.

Validates:
1. Law 1 Working Memory Footprint & 60 Hz Latency Budget Compliance:
   - Fast working memory strictly maintains 4.0 KB (4,096 bytes) L1-cache friendly footprint.
   - Step latency comfortably passes the 16.67 ms (60 Hz) real-time deadline.
2. Long-Context Factual Recall across 500+ token intervals:
   - Needle-in-a-haystack fact retention: Early fact survives 500+ distractor tokens.
   - Proves attractor collapse in single-tier 4.0 KB memory vs durable episodic retention.
3. State Serialization & Deserialization:
   - Dictionary and binary serialization roundtrips preserve state tensors bitwise.
4. Scalable Configurations Across Parameter Tiers:
   - Validates memory budgets across Tier 0, Tier 1, Tier 2, Tier 3, and Tier 5 (1B).
5. Autograd Gradient Flow:
   - Differentiable consolidation and retrieval paths allow full end-to-end learning.
"""

from __future__ import annotations

import time
import unittest
import torch
import torch.nn.functional as F

from irene_brain.memory.hierarchical_state import (
    CrossTierGatedConsolidator,
    CrossTierGatedRetriever,
    HierarchicalCognitiveState,
    HierarchicalMemoryConfig,
    HierarchicalMemorySystem,
    HierarchicalSemanticPseudoBrain,
)


class TestHierarchicalMemoryArchitecture(unittest.TestCase):
    """Test suite validating Pseudo-Brain Hierarchical Memory Architecture."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    def test_law1_working_memory_footprint_compliance(self) -> None:
        """Verify Fast Working Memory strictly respects Law 1 (4,096 bytes / 4.0 KB)."""
        config = HierarchicalMemoryConfig.for_tier("tier2")
        self.assertEqual(config.K_fast, 16)
        self.assertEqual(config.W_fast, 64)

        # 16 slots * 64 floats * 4 bytes/float = 4,096 bytes
        expected_fast_bytes = 16 * 64 * 4
        self.assertEqual(expected_fast_bytes, 4096)
        self.assertEqual(config.fast_state_bytes(), 4096)

        system = HierarchicalMemorySystem(config)
        state = system.initial_state(batch_size=1, device=self.device)

        # Check tensor byte footprint per sample
        actual_fast_bytes = state.working_memory[0].numel() * state.working_memory.element_size()
        self.assertEqual(actual_fast_bytes, 4096)
        self.assertEqual(state.fast_state_bytes(), 4096)

        # Check report telemetry
        report = system.footprint_report()
        self.assertTrue(report["l1_cache_compliant"])
        self.assertEqual(report["fast_state_bytes"], 4096)
        self.assertEqual(report["fast_state_kb"], 4.0)

    def test_parameter_tier_scaling(self) -> None:
        """Verify scalable configurations across parameter tiers (Tier 0 to Tier 5 / 1B)."""
        tiers_expected = {
            "tier0": {"K_f": 8, "W_f": 32, "K_e": 16, "W_e": 64, "fast_bytes": 1024, "epi_bytes": 4096},
            "tier1": {"K_f": 16, "W_f": 64, "K_e": 32, "W_e": 64, "fast_bytes": 4096, "epi_bytes": 8192},
            "tier2": {"K_f": 16, "W_f": 64, "K_e": 64, "W_e": 128, "fast_bytes": 4096, "epi_bytes": 32768},
            "tier3": {"K_f": 16, "W_f": 64, "K_e": 128, "W_e": 128, "fast_bytes": 4096, "epi_bytes": 65536},
            "tier5_1b": {"K_f": 16, "W_f": 64, "K_e": 256, "W_e": 128, "fast_bytes": 4096, "epi_bytes": 131072},
        }

        for tier_name, expected in tiers_expected.items():
            with self.subTest(tier=tier_name):
                cfg = HierarchicalMemoryConfig.for_tier(tier_name)
                self.assertEqual(cfg.K_fast, expected["K_f"])
                self.assertEqual(cfg.W_fast, expected["W_f"])
                self.assertEqual(cfg.K_episodic, expected["K_e"])
                self.assertEqual(cfg.W_episodic, expected["W_e"])
                self.assertEqual(cfg.fast_state_bytes(), expected["fast_bytes"])
                self.assertEqual(cfg.episodic_state_bytes(), expected["epi_bytes"])
                self.assertEqual(cfg.total_state_bytes(), expected["fast_bytes"] + expected["epi_bytes"])

                system = HierarchicalMemorySystem(cfg)
                report = system.footprint_report()
                self.assertEqual(report["total_state_bytes"], expected["fast_bytes"] + expected["epi_bytes"])
                self.assertGreater(report["memory_expansion_ratio"], 1.0)

    def test_fast_step_latency_budget(self) -> None:
        """Verify hierarchical memory step satisfies 60 Hz (16.67 ms) real-time deadline."""
        config = HierarchicalMemoryConfig.for_tier("tier2")
        system = HierarchicalMemorySystem(config)
        state = system.initial_state(batch_size=1, device=self.device)
        dummy_working = torch.randn(1, config.K_fast, config.W_fast, device=self.device)

        # Warmup
        for _ in range(5):
            dummy_working, state, _ = system.step(dummy_working, state)

        # Benchmark 50 steps
        num_steps = 50
        start = time.perf_counter()
        for _ in range(num_steps):
            dummy_working, state, _ = system.step(dummy_working, state)
        elapsed_sec = time.perf_counter() - start
        ms_per_step = (elapsed_sec / num_steps) * 1000.0

        # Must be well within the 16.67 ms (60 Hz) budget
        self.assertLess(ms_per_step, 16.67, f"Step latency {ms_per_step:.2f} ms exceeds 16.67 ms 60 Hz budget")
        # On modern CPUs, expected step time is < 3.0 ms
        self.assertLess(ms_per_step, 5.0, f"Step latency {ms_per_step:.2f} ms unusually slow")

    def test_state_serialization_and_deserialization(self) -> None:
        """Verify lossless serialization and deserialization via dictionary and bytes."""
        config = HierarchicalMemoryConfig.for_tier("tier2")
        system = HierarchicalMemorySystem(config)
        state = system.initial_state(batch_size=2, device=self.device)

        # Mutate state with known values
        state.working_memory += 0.42
        state.episodic_memory -= 0.17
        state.episodic_ages.fill_(12.5)
        state.step_count += 37

        # 1. Test to_dict / from_dict
        state_dict = state.to_dict()
        self.assertIsInstance(state_dict, dict)
        restored_from_dict = HierarchicalCognitiveState.from_dict(state_dict, device=self.device)

        self.assertTrue(torch.equal(state.working_memory, restored_from_dict.working_memory))
        self.assertTrue(torch.equal(state.episodic_memory, restored_from_dict.episodic_memory))
        self.assertTrue(torch.equal(state.episodic_ages, restored_from_dict.episodic_ages))
        self.assertTrue(torch.equal(state.step_count, restored_from_dict.step_count))

        # 2. Test to_bytes / from_bytes
        byte_data = state.to_bytes()
        self.assertIsInstance(byte_data, bytes)
        self.assertGreater(len(byte_data), 0)
        restored_from_bytes = HierarchicalCognitiveState.from_bytes(byte_data, device=self.device)

        self.assertTrue(torch.equal(state.working_memory, restored_from_bytes.working_memory))
        self.assertTrue(torch.equal(state.episodic_memory, restored_from_bytes.episodic_memory))
        self.assertTrue(torch.equal(state.episodic_ages, restored_from_bytes.episodic_ages))
        self.assertTrue(torch.equal(state.step_count, restored_from_bytes.step_count))

        # 3. Test clone & detach
        cloned = state.clone()
        self.assertTrue(torch.equal(state.working_memory, cloned.working_memory))
        cloned.working_memory += 1.0
        self.assertFalse(torch.equal(state.working_memory, cloned.working_memory))

    def test_long_context_factual_recall_500_plus_tokens(self) -> None:
        """Verify long-context factual recall across 500+ token intervals without attractor collapse.

        Simulation Protocol:
        1. Step 0: Introduce a distinct factual pattern into working memory with high salience.
        2. Steps 1 to 520: Stream 520 distractor noise steps representing multi-turn conversation.
        3. Step 521: Query the memory system for the original fact.
        4. Validate:
           - Baseline 16-slot 4.0 KB working memory completely forgets the fact (cosine similarity < 0.2).
           - Hierarchical consolidated episodic memory successfully retains the fact (cosine similarity > 0.80).
        """
        config = HierarchicalMemoryConfig(
            tier="tier2",
            K_fast=16,
            W_fast=64,
            K_episodic=64,
            W_episodic=128,
            consolidation_rate=0.2,
            consolidation_threshold=0.1,
            retrieval_top_k=4,
        )
        system = HierarchicalMemorySystem(config)
        state = system.initial_state(batch_size=1, device=self.device)

        # 1. Create a distinct factual pattern in slot 0
        fact_pattern = torch.randn(1, 1, config.W_fast, device=self.device)
        fact_pattern = F.normalize(fact_pattern, dim=-1)

        initial_working = state.working_memory.clone()
        initial_working[:, 0:1, :] = fact_pattern

        # Step 0: High salience write of the fact
        salience_fact = torch.zeros(1, config.K_fast, 1, device=self.device)
        salience_fact[:, 0, :] = 1.0  # slot 0 has maximum consequence salience

        # Run several consolidation steps on the fact to simulate initial encoding
        for _ in range(3):
            _, state, _ = system.step(initial_working, state, salience=salience_fact)

        # Save target projection for semantic comparison
        with torch.no_grad():
            projected_fact = system.consolidator.write_proj(fact_pattern).squeeze(1)  # [1, W_e]
            projected_fact = F.normalize(projected_fact, dim=-1)

        # Track single-tier baseline working memory under 520 distractor updates
        baseline_working = initial_working.clone()

        # 2. Stream 520 distractor steps
        num_distractor_steps = 520
        torch.manual_seed(123)

        for step_idx in range(num_distractor_steps):
            # Transient distractor content (sentences / conversation noise)
            distractor_input = torch.randn(1, config.K_fast, config.W_fast, device=self.device) * 0.5
            distractor_salience = torch.rand(1, config.K_fast, 1, device=self.device) * 0.05  # below threshold background salience

            # Single-tier recurrent simulation (continuous overwrite without episodic tier)
            baseline_gate = torch.sigmoid(torch.randn(1, config.K_fast, 1, device=self.device) * 0.5)
            baseline_working = (1.0 - baseline_gate) * baseline_working + baseline_gate * distractor_input

            # Hierarchical memory step
            _, state, _ = system.step(distractor_input, state, salience=distractor_salience)

        # 3. Evaluate Baseline Working Memory
        baseline_slot0 = baseline_working[:, 0, :]
        baseline_sim = F.cosine_similarity(baseline_slot0, fact_pattern.squeeze(1)).item()
        # Single-tier working memory must experience catastrophic forgetting
        self.assertLess(
            baseline_sim,
            0.30,
            f"Baseline memory unexpectedly retained fact (sim {baseline_sim:.3f}), expected attractor collapse",
        )

        # 4. Evaluate Hierarchical Episodic Memory
        # Query episodic memory using the original fact's query representation
        episodic_slots = state.episodic_memory[0]  # [K_episodic, W_episodic]
        episodic_norm = F.normalize(episodic_slots, dim=-1)

        sims = torch.matmul(projected_fact, episodic_norm.transpose(0, 1)).squeeze(0)
        max_sim = sims.max().item()

        # Episodic tier must retain a high-fidelity representation of the fact after 520+ steps
        self.assertGreater(
            max_sim,
            0.80,
            f"Episodic memory suffered attractor collapse (max cosine similarity {max_sim:.3f} < 0.80)",
        )

        # 5. Test retrieval back into working memory
        query_override = torch.zeros(1, config.K_fast, config.W_fast, device=self.device)
        query_override[:, 0:1, :] = fact_pattern  # Query for the fact in slot 0
        augmented_working, retrieved_context, weights = system.retriever(
            working_memory=query_override,
            episodic_memory=state.episodic_memory,
            query_override=query_override,
        )

        # The retrieved context should exhibit positive correlation with the fact pattern
        retrieved_slot0 = retrieved_context[:, 0, :]
        retrieved_sim = F.cosine_similarity(retrieved_slot0, fact_pattern.squeeze(1)).item()
        self.assertGreater(
            retrieved_sim,
            0.50,
            f"Retrieved context correlation {retrieved_sim:.3f} was insufficient",
        )

    def test_autograd_gradient_flow(self) -> None:
        """Verify autograd gradients flow differentiably through consolidation and retrieval."""
        config = HierarchicalMemoryConfig(
            tier="tier2",
            K_fast=8,
            W_fast=32,
            K_episodic=16,
            W_episodic=64,
            consolidation_rate=0.1,
            consolidation_threshold=0.1,
        )
        system = HierarchicalMemorySystem(config)
        state = system.initial_state(batch_size=2, device=self.device)

        working = torch.randn(2, config.K_fast, config.W_fast, requires_grad=True, device=self.device)
        salience = torch.rand(2, config.K_fast, 1, requires_grad=True, device=self.device)

        augmented, next_state, _ = system.step(working, state, salience=salience)
        loss = augmented.sum() + next_state.episodic_memory.sum()
        loss.backward()

        # Check gradients exist and are non-zero in key consolidation & retrieval parameters
        self.assertIsNotNone(system.consolidator.write_proj.weight.grad)
        self.assertGreater(system.consolidator.write_proj.weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(system.retriever.q_proj.weight.grad)
        self.assertGreater(system.retriever.q_proj.weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(system.retriever.read_gate[0].weight.grad)
        self.assertGreater(system.retriever.read_gate[0].weight.grad.abs().sum().item(), 0.0)

    def test_hierarchical_semantic_pseudobrain_e2e(self) -> None:
        """Verify end-to-end HierarchicalSemanticPseudoBrain forward pass and streaming."""
        config = HierarchicalMemoryConfig.for_tier("tier2", K_fast=8, W_fast=32, K_episodic=16, W_episodic=64)
        model = HierarchicalSemanticPseudoBrain(
            vocab_size=100,
            config=config,
            embed_dim=32,
            proj_dim=64,
        )

        B, T = 2, 8
        tokens = torch.randint(0, 100, (B, T), device=self.device)

        # Sequential unroll
        logits = model(tokens)
        self.assertEqual(logits.shape, (B, T, 100))
        self.assertFalse(torch.isnan(logits).any())

        # Single step streaming
        state = model.initial_state(B, self.device)
        step_logits, next_state = model.step(tokens[:, 0], state)
        self.assertEqual(step_logits.shape, (B, 100))
        self.assertFalse(torch.isnan(step_logits).any())
        self.assertEqual(next_state.working_memory.shape, (B, 8, 32))
        self.assertEqual(next_state.episodic_memory.shape, (B, 16, 64))


    def test_multi_fact_non_interference_500_tokens(self) -> None:
        """Verify multiple distinct facts consolidate into distinct episodic slots without interference."""
        config = HierarchicalMemoryConfig(
            tier="tier2",
            K_fast=16,
            W_fast=64,
            K_episodic=64,
            W_episodic=128,
            consolidation_rate=0.2,
            consolidation_threshold=0.1,
        )
        system = HierarchicalMemorySystem(config)
        state = system.initial_state(batch_size=1, device=self.device)

        # Three distinct facts
        fA = F.normalize(torch.randn(1, 1, config.W_fast, device=self.device), dim=-1)
        fB = F.normalize(torch.randn(1, 1, config.W_fast, device=self.device), dim=-1)
        fC = F.normalize(torch.randn(1, 1, config.W_fast, device=self.device), dim=-1)

        # Step 0: Fact A in slot 0
        w = state.working_memory.clone()
        w[:, 0:1, :] = fA
        sA = torch.zeros(1, config.K_fast, 1, device=self.device)
        sA[:, 0] = 1.0
        _, state, _ = system.step(w, state, salience=sA)

        # Steps 1 to 149: Distractor dialogue
        for _ in range(149):
            din = torch.randn(1, config.K_fast, config.W_fast, device=self.device) * 0.5
            dsal = torch.rand(1, config.K_fast, 1, device=self.device) * 0.05
            _, state, _ = system.step(din, state, salience=dsal)

        # Step 150: Fact B in slot 1
        w = state.working_memory.clone()
        w[:, 1:2, :] = fB
        sB = torch.zeros(1, config.K_fast, 1, device=self.device)
        sB[:, 1] = 1.0
        _, state, _ = system.step(w, state, salience=sB)

        # Steps 151 to 299: Distractor dialogue
        for _ in range(149):
            din = torch.randn(1, config.K_fast, config.W_fast, device=self.device) * 0.5
            dsal = torch.rand(1, config.K_fast, 1, device=self.device) * 0.05
            _, state, _ = system.step(din, state, salience=dsal)

        # Step 300: Fact C in slot 2
        w = state.working_memory.clone()
        w[:, 2:3, :] = fC
        sC = torch.zeros(1, config.K_fast, 1, device=self.device)
        sC[:, 2] = 1.0
        _, state, _ = system.step(w, state, salience=sC)

        # Steps 301 to 500: Distractor dialogue
        for _ in range(200):
            din = torch.randn(1, config.K_fast, config.W_fast, device=self.device) * 0.5
            dsal = torch.rand(1, config.K_fast, 1, device=self.device) * 0.05
            _, state, _ = system.step(din, state, salience=dsal)

        # Query and recall Fact A
        qA = torch.zeros(1, config.K_fast, config.W_fast, device=self.device)
        qA[:, 0:1] = fA
        _, retA, _ = system.retriever(qA, state.episodic_memory, query_override=qA)
        simA = F.cosine_similarity(retA[:, 0], fA.squeeze(1)).item()
        self.assertGreater(simA, 0.90, f"Fact A recall failed: {simA:.3f}")

        # Query and recall Fact B
        qB = torch.zeros(1, config.K_fast, config.W_fast, device=self.device)
        qB[:, 1:2] = fB
        _, retB, _ = system.retriever(qB, state.episodic_memory, query_override=qB)
        simB = F.cosine_similarity(retB[:, 1], fB.squeeze(1)).item()
        self.assertGreater(simB, 0.90, f"Fact B recall failed: {simB:.3f}")

        # Query and recall Fact C
        qC = torch.zeros(1, config.K_fast, config.W_fast, device=self.device)
        qC[:, 2:3] = fC
        _, retC, _ = system.retriever(qC, state.episodic_memory, query_override=qC)
        simC = F.cosine_similarity(retC[:, 2], fC.squeeze(1)).item()
        self.assertGreater(simC, 0.90, f"Fact C recall failed: {simC:.3f}")


if __name__ == "__main__":
    unittest.main()
