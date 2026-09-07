"""Comprehensive verification suite for Sparse Top-k Thoughtlet Routing.

Phase 2.6 Workstream E:
Validates:
1. Slot permutation equivariance: permuting slot indices strictly permutes output messages.
2. Sparsity contract: exactly top-k peer influences are routed; non-selected peers receive zero weight and zero gradient.
3. Determinism & gradient stability across K in [16, 32, 64].
4. Cycle-1 private computation contract (allow_routing=False).
5. Dense routing baseline comparison.
6. Integration with StructuredBrainBlock and BrainCell across scaling regimes.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from irene_brain.model.brain_cell import BrainCell, StructuredBrainBlock
    from irene_brain.model.sparse_thought_router import RoutingDiagnostics, SparseThoughtRouter


@unittest.skipUnless(torch is not None, "PyTorch required for sparse thought routing tests")
class SparseThoughtRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.manual_seed(42)

    def test_slot_permutation_equivariance(self) -> None:
        """Permuting input thoughtlet slots must identically permute output messages."""
        assert torch is not None

        for k in (2, 4):
            for thoughtlets in (16, 32, 64):
                with self.subTest(k=k, thoughtlets=thoughtlets):
                    width = 64
                    batch_size = 2
                    router = SparseThoughtRouter(width=width, routed_neighbors=k)

                    # Distinct per-slot features to avoid score ties
                    summaries = torch.randn(batch_size, thoughtlets, width)

                    # Baseline forward pass
                    baseline_messages, baseline_diag = router(summaries)

                    # Generate a random non-trivial permutation of slots
                    perm = torch.randperm(thoughtlets)
                    inv_perm = torch.empty_like(perm)
                    inv_perm[perm] = torch.arange(thoughtlets)

                    permuted_summaries = summaries[:, perm, :]
                    permuted_messages, permuted_diag = router(permuted_summaries)

                    # Check output message permutation equivariance:
                    # Message at permuted position i must equal baseline message at perm[i]
                    expected_permuted_messages = baseline_messages[:, perm, :]
                    max_diff = (permuted_messages - expected_permuted_messages).abs().max()
                    self.assertLess(
                        float(max_diff),
                        1e-5,
                        f"Failed equivariance at K={thoughtlets}, k={k} with diff {max_diff}",
                    )

                    # Check routing weights equivariance:
                    # The set of routed weights for slot i under permutation should match perm[i]
                    expected_permuted_weights = baseline_diag.weights[:, perm, :]
                    weight_diff = (permuted_diag.weights - expected_permuted_weights).abs().max()
                    self.assertLess(
                        float(weight_diff),
                        1e-5,
                        f"Weight mismatch under permutation at K={thoughtlets}, k={k}",
                    )

    def test_sparsity_contract_and_zero_interference(self) -> None:
        """Top-k selection routes exactly k peers; unselected peers receive zero gradient."""
        assert torch is not None

        batch_size = 2
        thoughtlets = 16
        width = 32
        k = 2

        router = SparseThoughtRouter(width=width, routed_neighbors=k)
        summaries = torch.randn(batch_size, thoughtlets, width)

        messages, diagnostics = router(summaries)

        # 1. Structural dimensions
        self.assertEqual(diagnostics.indices.shape, (batch_size, thoughtlets, k))
        self.assertEqual(diagnostics.weights.shape, (batch_size, thoughtlets, k))
        self.assertEqual(messages.shape, (batch_size, thoughtlets, width))

        # 2. Partition of unity (weights sum to 1.0 along top-k axis)
        weight_sums = diagnostics.weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))

        # 3. Bio-plausible self-exclusion contract: no slot may route to itself
        slot_indices = (
            torch.arange(thoughtlets, device=summaries.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, thoughtlets, k)
        )
        self.assertFalse(
            (diagnostics.indices == slot_indices).any().item(),
            "Violation of bio-plausible self-exclusion: a thoughtlet routed to itself!",
        )

        # 4. Rigorous Gradient Sparsity Isolation:
        # Evaluate loss ONLY on receiving slot 0 for batch item 0.
        # Check gradient on value representation V:
        # - Slot 0 (self): must have zero gradient.
        # - The k selected peers: must have non-zero gradient matching routing weights.
        # - All other K - 1 - k peers: must have IDENTICALLY ZERO gradient.
        x = torch.randn(1, thoughtlets, width, requires_grad=True)
        query = router.route_query(x)
        key = router.route_key(x)
        value = router.route_value(x)
        value.retain_grad()

        scores = torch.matmul(query, key.transpose(-1, -2)) / (width**0.5)
        diag_mask = torch.eye(thoughtlets, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(diag_mask, torch.finfo(scores.dtype).min)

        values, indices = torch.topk(scores, k=k, dim=-1)
        weights = torch.softmax(values, dim=-1)

        sparse_weights = torch.zeros(1, thoughtlets, thoughtlets, dtype=x.dtype)
        sparse_weights.scatter_(2, indices, weights)
        message = torch.bmm(sparse_weights, value)

        receiver_slot = 0
        loss = message[0, receiver_slot, :].sum()
        loss.backward()

        assert value.grad is not None
        v_grad = value.grad[0]  # [thoughtlets, width]

        selected_peers = set(indices[0, receiver_slot].tolist())
        self.assertEqual(len(selected_peers), k)
        self.assertNotIn(receiver_slot, selected_peers)

        for peer_idx in range(thoughtlets):
            peer_grad_norm = float(v_grad[peer_idx].abs().sum())
            if peer_idx in selected_peers:
                self.assertGreater(
                    peer_grad_norm,
                    1e-6,
                    f"Selected peer {peer_idx} unexpectedly received zero gradient",
                )
            else:
                self.assertEqual(
                    peer_grad_norm,
                    0.0,
                    f"Unselected peer {peer_idx} received non-zero gradient ({peer_grad_norm})! Sparsity contract violated!",
                )

    def test_determinism_and_gradient_stability(self) -> None:
        """Forward pass is strictly deterministic and gradients are stable across K in [16, 32, 64]."""
        assert torch is not None

        for thoughtlets in (16, 32, 64):
            for k in (2, 4):
                with self.subTest(thoughtlets=thoughtlets, k=k):
                    width = 48
                    batch_size = 2
                    router = SparseThoughtRouter(width=width, routed_neighbors=k)

                    inputs = torch.randn(batch_size, thoughtlets, width, requires_grad=True)

                    # Determinism test
                    out1, diag1 = router(inputs)
                    out2, diag2 = router(inputs)

                    self.assertTrue(torch.equal(out1, out2))
                    self.assertTrue(torch.equal(diag1.indices, diag2.indices))
                    self.assertTrue(torch.equal(diag1.weights, diag2.weights))

                    # Gradient stability test
                    loss = out1.square().mean()
                    loss.backward()

                    for name, param in router.named_parameters():
                        self.assertIsNotNone(param.grad, f"{name} grad is None")
                        assert param.grad is not None
                        self.assertTrue(
                            torch.isfinite(param.grad).all().item(),
                            f"Non-finite gradient in {name} at K={thoughtlets}, k={k}",
                        )
                        self.assertGreater(
                            float(param.grad.abs().sum()),
                            1e-6,
                            f"Vanishing gradient in {name} at K={thoughtlets}, k={k}",
                        )

                    assert inputs.grad is not None
                    self.assertTrue(torch.isfinite(inputs.grad).all().item())
                    self.assertGreater(float(inputs.grad.abs().sum()), 1e-6)

    def test_allow_routing_disabled_contract(self) -> None:
        """When allow_routing is False, messages must be zero and indices/weights empty."""
        assert torch is not None

        router = SparseThoughtRouter(width=32, routed_neighbors=2)
        summaries = torch.randn(3, 16, 32)

        messages, diag = router(summaries, allow_routing=False)

        self.assertTrue(torch.equal(messages, torch.zeros_like(summaries)))
        self.assertEqual(diag.indices.shape, (3, 16, 0))
        self.assertEqual(diag.weights.shape, (3, 16, 0))

    def test_dense_routing_mode(self) -> None:
        """Dense routing routes all K-1 peers with valid softmax weights."""
        assert torch is not None

        thoughtlets = 8
        width = 16
        router = SparseThoughtRouter(width=width, dense_routing=True)
        summaries = torch.randn(2, thoughtlets, width)

        messages, diag = router(summaries)

        self.assertEqual(diag.indices.shape, (2, thoughtlets, thoughtlets - 1))
        self.assertEqual(diag.weights.shape, (2, thoughtlets, thoughtlets - 1))
        self.assertEqual(messages.shape, (2, thoughtlets, width))

        # Weights along peer axis sum to 1.0
        weight_sums = diag.weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))

        # No self routing
        for i in range(thoughtlets):
            self.assertNotIn(i, diag.indices[0, i].tolist())

    def test_input_validation(self) -> None:
        """Validation guards reject invalid widths and neighbor counts."""
        assert torch is not None

        with self.assertRaises(ValueError):
            SparseThoughtRouter(width=0)
        with self.assertRaises(ValueError):
            SparseThoughtRouter(width=32, routed_neighbors=0)
        with self.assertRaises(ValueError):
            SparseThoughtRouter(width=True)  # type: ignore[arg-type]

        router = SparseThoughtRouter(width=32, routed_neighbors=2)
        with self.assertRaises(ValueError):
            router(torch.randn(2, 8, 64))  # width mismatch

    def test_brain_cell_integration_across_k_scaling(self) -> None:
        """StructuredBrainBlock and BrainCell integrate SparseThoughtRouter seamlessly at K=16, 32, 64."""
        assert torch is not None

        for thoughtlets in (16, 32, 64):
            for k in (2, 4):
                with self.subTest(thoughtlets=thoughtlets, k=k):
                    width = 32
                    heads = 2
                    blocks = 2
                    cell = BrainCell(
                        width=width,
                        heads=heads,
                        routed_neighbors=k,
                        blocks=blocks,
                    )

                    batch = 1
                    registers = 3
                    sensors = torch.randn(batch, 4, width)
                    action_time = torch.randn(batch, 2, width)
                    belief = torch.randn(batch, 4, width)
                    working_memory = torch.randn(batch, 2, width)
                    thoughts = torch.randn(batch, thoughtlets, registers, width)
                    goal_context = torch.randn(batch, 1, width)
                    retrieved_memory = torch.randn(batch, thoughtlets, 1, width)
                    elapsed = torch.tensor([[0.016]])

                    # Forward pass with allow_routing=True
                    b_out, m_out, t_out, routing = cell(
                        belief=belief,
                        working_memory=working_memory,
                        thoughts=thoughts,
                        sensors=sensors,
                        action_time_tokens=action_time,
                        goal_context=goal_context,
                        retrieved_memory=retrieved_memory,
                        elapsed_seconds=elapsed,
                        allow_routing=True,
                    )

                    self.assertEqual(len(routing), blocks)
                    for block_diag in routing:
                        self.assertEqual(block_diag.indices.shape, (batch, thoughtlets, k))
                        self.assertEqual(block_diag.weights.shape, (batch, thoughtlets, k))

                    self.assertEqual(t_out.shape, thoughts.shape)
                    self.assertEqual(b_out.shape, belief.shape)
                    self.assertEqual(m_out.shape, working_memory.shape)


if __name__ == "__main__":
    unittest.main()
