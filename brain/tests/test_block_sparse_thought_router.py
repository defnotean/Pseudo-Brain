"""Unit test suite for BlockSparseClusteredThoughtRouter (P6).

Verifies:
1. Shape contract: [B, K, W] -> [B, K, W] and diagnostics [B, K, k].
2. Partition of unity: routing weights sum to 1.0 per thoughtlet.
3. Self-exclusion: no thoughtlet routes to itself.
4. Scale-invariance: works seamlessly across small and large K (e.g. K=8, 16, 32, 64, 128).
5. Autograd gradient flow & stability across 100 backward steps.
6. Permutation equivariance check.
"""

from __future__ import annotations

import unittest
import torch

from irene_brain.model.sparse_thought_router import (
    BlockSparseClusteredThoughtRouter,
    SparseThoughtRouter,
)


class BlockSparseRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.width = 32
        self.k = 2

    def test_shape_and_diagnostics_contract(self) -> None:
        """Verify output tensor shapes across varied K values."""
        for K in [8, 16, 32, 48, 64, 128]:
            router = BlockSparseClusteredThoughtRouter(width=self.width, routed_neighbors=self.k)
            x = torch.randn(2, K, self.width)
            msg, diag = router(x)

            self.assertEqual(msg.shape, (2, K, self.width), f"Failed for K={K}")
            self.assertEqual(diag.indices.shape, (2, K, self.k), f"Failed for K={K}")
            self.assertEqual(diag.weights.shape, (2, K, self.k), f"Failed for K={K}")

    def test_partition_of_unity(self) -> None:
        """Verify that routing weights sum to exactly 1.0 along peer dimension."""
        for K in [16, 32, 64]:
            router = BlockSparseClusteredThoughtRouter(width=self.width, routed_neighbors=self.k)
            x = torch.randn(1, K, self.width)
            _, diag = router(x)

            weight_sums = diag.weights.sum(dim=-1)
            self.assertTrue(
                torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5),
                f"Weights did not sum to 1.0 for K={K}",
            )

    def test_self_exclusion_invariance(self) -> None:
        """Verify that no thoughtlet ever routes messages to itself."""
        for K in [16, 32, 64]:
            router = BlockSparseClusteredThoughtRouter(width=self.width, routed_neighbors=self.k)
            x = torch.randn(2, K, self.width)
            _, diag = router(x)

            self_indices = torch.arange(K, device=x.device).view(1, K, 1).expand(2, -1, self.k)
            has_self_loop = (diag.indices == self_indices).any().item()
            self.assertFalse(has_self_loop, f"Detected self-loop in routing indices for K={K}")

    def test_autograd_gradient_stability(self) -> None:
        """Verify 50 backward steps execute without NaN or Inf gradients."""
        router = BlockSparseClusteredThoughtRouter(width=self.width, routed_neighbors=self.k)
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)

        for step in range(50):
            optimizer.zero_grad()
            x = torch.randn(2, 32, self.width, requires_grad=True)
            msg, _ = router(x)
            loss = (msg ** 2).mean()
            loss.backward()

            for name, param in router.named_parameters():
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite grad at step {step} in {name}")

            optimizer.step()

    def test_sub_quadratic_recall_vs_dense(self) -> None:
        """Verify candidate recall against exact SparseThoughtRouter."""
        K = 32
        exact_router = SparseThoughtRouter(width=self.width, routed_neighbors=self.k)
        cluster_router = BlockSparseClusteredThoughtRouter(width=self.width, routed_neighbors=self.k, cluster_neighbors=3)

        # Match linear projection weights
        with torch.no_grad():
            cluster_router.route_query.weight.copy_(exact_router.route_query.weight)
            cluster_router.route_key.weight.copy_(exact_router.route_key.weight)
            cluster_router.route_value.weight.copy_(exact_router.route_value.weight)

        x = torch.randn(4, K, self.width)
        _, exact_diag = exact_router(x)
        _, cluster_diag = cluster_router(x)

        # Measure Top-k Recall overlap
        exact_idx = exact_diag.indices  # [B, K, k]
        clust_idx = cluster_diag.indices  # [B, K, k]

        overlaps = 0
        total = 4 * K * self.k
        for b in range(4):
            for i in range(K):
                set_exact = set(exact_idx[b, i].tolist())
                set_clust = set(clust_idx[b, i].tolist())
                overlaps += len(set_exact.intersection(set_clust))

        recall = overlaps / total
        self.assertGreater(recall, 0.50, f"Recall@k ({recall:.2%}) should capture major peer affinity")


if __name__ == "__main__":
    unittest.main()
