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

        # 5. Sparse message gathering via scatter into sparse weight matrix + bmm.
        # This provides O(K * k) sparsity without memory-wasteful 4D tensor expansions,
        # guaranteeing that non-selected peers receive strictly zero gradient.
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
