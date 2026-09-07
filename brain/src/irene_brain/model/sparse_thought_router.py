"""Bio-plausible Sparse Top-k Thoughtlet Routing (Phase 2.6 Workstream E).

In high-capacity parallel cognition (K thoughtlet slots), dense all-to-all
mixing creates quadratic communication overhead and cognitive interference.
Biological neural ensembles instead rely on sparse, selective inter-module
signaling where each computational unit keeps its internal dynamics private
and gathers messages only from a small subset of relevant peers.

This module provides ``SparseThoughtRouter``, an nn.Module that:
1. Emits query, key, and value representations per thoughtlet slot.
2. Computes pairwise peer affinities with self-connections masked out
   (preventing self-interference and narcissistic feedback).
3. Selectively routes messages from only the top-k most relevant peers
   (default k=2 or k=4).
4. Strictly preserves permutation equivariance across thoughtlet slots.
5. Ensures non-selected peers receive zero weight and zero gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class RoutingDiagnostics:
    """Diagnostic telemetry captured from a thoughtlet routing pass."""

    indices: Tensor
    weights: Tensor


class SparseThoughtRouter(nn.Module):
    """Top-k selective peer router across parallel thoughtlet slots.

    Parameters
    ----------
    width : int
        Dimensionality of the thoughtlet summary representation (W).
    routed_neighbors : int, default=2
        Number of top-k peer influences each thoughtlet gathers from (k).
    dense_routing : bool, default=False
        If True, executes unrestricted all-to-all softmax communication
        (used for matched ablation baselines).
    """

    def __init__(
        self,
        *,
        width: int,
        routed_neighbors: int = 2,
        dense_routing: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError(f"width must be a positive integer, got {width!r}")
        if (
            isinstance(routed_neighbors, bool)
            or not isinstance(routed_neighbors, int)
            or routed_neighbors < 1
        ):
            raise ValueError(
                f"routed_neighbors must be a positive integer, got {routed_neighbors!r}"
            )

        self.width = width
        self.routed_neighbors = routed_neighbors
        self.dense_routing = bool(dense_routing)

        # Projections: queries, keys, and values for thoughtlet message routing
        self.route_query = nn.Linear(width, width, bias=False)
        self.route_key = nn.Linear(width, width, bias=False)
        self.route_value = nn.Linear(width, width, bias=False)

    def forward(
        self,
        summaries: Tensor,
        *,
        allow_routing: bool = True,
    ) -> tuple[Tensor, RoutingDiagnostics]:
        """Route messages across thoughtlet slots.

        Parameters
        ----------
        summaries : Tensor
            Thoughtlet slot summaries of shape [batch, thoughtlets, width].
        allow_routing : bool, default=True
            Whether inter-slot communication is permitted (e.g. cognitive cycle > 0).
            When False, thoughts remain strictly private (empty diagnostics, zero messages).

        Returns
        -------
        tuple[Tensor, RoutingDiagnostics]
            - routed_message: Tensor of shape [batch, thoughtlets, width]
            - diagnostics: RoutingDiagnostics containing selected peer indices and weights
        """
        batch, thoughtlets, width = summaries.shape
        if width != self.width:
            raise ValueError(
                f"Expected summaries width {self.width}, got {width} (shape {summaries.shape})"
            )

        if not allow_routing:
            empty_indices = torch.empty(
                batch,
                thoughtlets,
                0,
                device=summaries.device,
                dtype=torch.long,
            )
            empty_weights = summaries.new_empty((batch, thoughtlets, 0))
            return torch.zeros_like(summaries), RoutingDiagnostics(
                empty_indices, empty_weights
            )

        # 1. Project thoughtlet representations into Query, Key, and Value spaces
        query = self.route_query(summaries)
        key = self.route_key(summaries)
        value = self.route_value(summaries)

        # 2. Compute scaled pairwise affinity scores: [B, K, K]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(width)

        # 3. Bio-plausible self-exclusion mask:
        # A thoughtlet's internal representation is already retained via its private
        # registers; routing is strictly peer-to-peer. Masking the diagonal prevents
        # self-interference and self-reinforcing echo loops.
        diagonal = torch.eye(thoughtlets, device=summaries.device, dtype=torch.bool)
        scores = scores.masked_fill(diagonal.unsqueeze(0), torch.finfo(scores.dtype).min)

        if self.dense_routing:
            # Unrestricted all-to-all communication (ablation baseline)
            base = torch.arange(thoughtlets, device=summaries.device)
            off_diagonal = base.unsqueeze(0).expand(thoughtlets, thoughtlets)
            off_diagonal = off_diagonal[~diagonal].reshape(thoughtlets, thoughtlets - 1)
            indices = off_diagonal.unsqueeze(0).expand(
                batch, thoughtlets, thoughtlets - 1
            )
            weights = torch.gather(torch.softmax(scores, dim=-1), 2, indices)

            sparse_weights = torch.zeros(
                batch,
                thoughtlets,
                thoughtlets,
                device=summaries.device,
                dtype=summaries.dtype,
            )
            sparse_weights.scatter_(2, indices, weights)
            message = torch.bmm(sparse_weights, value)
            return message, RoutingDiagnostics(indices, weights)

        # 4. Sparse Top-k selection:
        # Select exactly the top-k most relevant distinct peers for each slot
        k_effective = min(self.routed_neighbors, thoughtlets - 1)
        values, indices = torch.topk(scores, k=k_effective, dim=-1)

        # Softmax strictly over the top-k chosen candidates (partition of unity)
        weights = torch.softmax(values, dim=-1)

        # 5. Sparse message gathering via gather-multiply-sum.
        # Direct gather provides O(K * k * W) computation and strictly zero gradients
        # for unselected peers, completely eliminating dense [B, K, K] BLAS BMM allocations.
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, width)
        gathered_value = torch.gather(value.unsqueeze(1).expand(-1, thoughtlets, -1, -1), 2, idx_expanded)
        message = (weights.unsqueeze(-1) * gathered_value).sum(dim=2)

        return message, RoutingDiagnostics(indices, weights)


class BlockSparseClusteredThoughtRouter(nn.Module):
    """Hierarchical Sub-Quadratic Thoughtlet Router via Macro-Column Clustering.

    Complexity:
    - Affinity Time: O(B * K^1.5 * W) instead of O(B * K^2 * W)
    - Memory: O(B * K^1.5) instead of O(B * K^2)
    - Completely eliminates [B, K, K] dense tensor allocations.
    """

    def __init__(
        self,
        *,
        width: int,
        routed_neighbors: int = 2,
        clusters: int | None = None,
        cluster_neighbors: int = 2,
    ) -> None:
        super().__init__()
        self.width = width
        self.routed_neighbors = routed_neighbors
        self.clusters = clusters
        self.cluster_neighbors = cluster_neighbors

        self.route_query = nn.Linear(width, width, bias=False)
        self.route_key = nn.Linear(width, width, bias=False)
        self.route_value = nn.Linear(width, width, bias=False)

    def forward(self, summaries: Tensor) -> tuple[Tensor, RoutingDiagnostics]:
        batch, thoughtlets, width = summaries.shape
        if thoughtlets <= 1 or self.routed_neighbors <= 0:
            empty_indices = torch.empty(batch, thoughtlets, 0, device=summaries.device, dtype=torch.long)
            empty_weights = summaries.new_empty((batch, thoughtlets, 0))
            return torch.zeros_like(summaries), RoutingDiagnostics(empty_indices, empty_weights)

        # Fall back to exact router if K is small (K <= 16)
        if thoughtlets <= 16:
            router = SparseThoughtRouter(width=width, routed_neighbors=self.routed_neighbors).to(summaries.device)
            router.route_query = self.route_query
            router.route_key = self.route_key
            router.route_value = self.route_value
            return router(summaries)

        M = self.clusters or max(2, int(math.ceil(math.sqrt(thoughtlets))))
        C = int(math.ceil(thoughtlets / M))
        pad_len = M * C - thoughtlets

        if pad_len > 0:
            padded_summaries = F.pad(summaries, (0, 0, 0, pad_len))
        else:
            padded_summaries = summaries

        Q = self.route_query(padded_summaries)  # [B, M*C, W]
        K = self.route_key(padded_summaries)    # [B, M*C, W]
        V = self.route_value(padded_summaries)  # [B, M*C, W]

        # 1. Coarse Routing: compute cluster centroid keys [B, M, W]
        K_clustered = K.view(batch, M, C, width)
        cluster_centroids = K_clustered.mean(dim=2)  # [B, M, W]

        # Coarse affinities: [B, M*C, M] -> O(K * sqrt(K) * W)
        coarse_scores = torch.matmul(Q, cluster_centroids.transpose(-1, -2)) / math.sqrt(width)
        k_c = min(self.cluster_neighbors, M)
        top_clusters = torch.topk(coarse_scores, k=k_c, dim=-1).indices  # [B, M*C, k_c]

        # 2. Form candidate pool per slot:
        home_cluster = torch.arange(M * C, device=summaries.device).div(C, rounding_mode="floor")  # [M*C]
        home_cluster_expanded = home_cluster.unsqueeze(0).expand(batch, -1).unsqueeze(-1)  # [B, M*C, 1]
        all_selected_clusters = torch.cat([home_cluster_expanded, top_clusters], dim=-1)  # [B, M*C, 1 + k_c]

        cluster_keys = K.view(batch, M, C, width)
        cluster_values = V.view(batch, M, C, width)

        P_clusters = 1 + k_c
        b_idx = torch.arange(batch, device=summaries.device).view(batch, 1, 1).expand(-1, M * C, P_clusters)
        cand_K = cluster_keys[b_idx, all_selected_clusters].view(batch, M * C, P_clusters * C, width)
        cand_V = cluster_values[b_idx, all_selected_clusters].view(batch, M * C, P_clusters * C, width)

        # Global indices of candidates: [B, M*C, P_clusters * C]
        cluster_slot_base = (all_selected_clusters * C).unsqueeze(-1) + torch.arange(C, device=summaries.device).view(1, 1, 1, C)
        cand_global_indices = cluster_slot_base.view(batch, M * C, P_clusters * C)

        # 3. Fine Affinities: [B, M*C, P_clusters * C]
        fine_scores = (Q.unsqueeze(2) * cand_K).sum(dim=-1) / math.sqrt(width)

        # Mask self-connections and padding slots
        self_mask = cand_global_indices == torch.arange(M * C, device=summaries.device).view(1, -1, 1)
        pad_mask = cand_global_indices >= thoughtlets
        invalid_mask = self_mask | pad_mask
        fine_scores = fine_scores.masked_fill(invalid_mask, torch.finfo(fine_scores.dtype).min)

        # 4. Top-k selection over candidate pool:
        k_effective = min(self.routed_neighbors, thoughtlets - 1)
        fine_values, local_indices = torch.topk(fine_scores, k=k_effective, dim=-1)
        fine_weights = torch.softmax(fine_values, dim=-1)

        chosen_global_indices = torch.gather(cand_global_indices, 2, local_indices)

        local_idx_expanded = local_indices.unsqueeze(-1).expand(-1, -1, -1, width)
        chosen_values = torch.gather(cand_V, 2, local_idx_expanded)
        message_padded = (fine_weights.unsqueeze(-1) * chosen_values).sum(dim=2)

        message = message_padded[:, :thoughtlets, :]
        out_indices = chosen_global_indices[:, :thoughtlets, :]
        out_weights = fine_weights[:, :thoughtlets, :]

        return message, RoutingDiagnostics(out_indices, out_weights)


__all__ = [
    "BlockSparseClusteredThoughtRouter",
    "RoutingDiagnostics",
    "SparseThoughtRouter",
]
