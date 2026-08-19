from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

@dataclass(frozen=True, slots=True)
class TopologicalGoalPrediction:
    pellet_cluster_vector: Tensor  # (B, 2), normalized directional vector (dx, dy)
    junction_exit_logits: Tensor   # (B, 5), logits for 5-way cardinal exits (None, W, A, S, D)
    corridor_depth_estimate: Tensor  # (B, 1), estimated remaining open corridor depth in tiles

class TopologicalGoalFieldHead(nn.Module):
    def __init__(self, *, width: int, thoughtlets: int = 4) -> None:
        super().__init__()
        self.width = width
        self.thoughtlets = thoughtlets
        self.thought_attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, width // 4),
            nn.GELU(),
        )
        self.vector_head = nn.Linear(width // 4, 2)
        self.junction_head = nn.Linear(width // 4, 5)
        self.depth_head = nn.Linear(width // 4, 1)

    def forward(self, thoughts: Tensor) -> TopologicalGoalPrediction:
        if thoughts.ndim == 4:
            # Aggregate across registers: (B, K, R, D) -> (B, K, D)
            thoughts = thoughts.mean(dim=2)
        B, K, D = thoughts.shape
        attn_out, _ = self.thought_attention(thoughts, thoughts, thoughts)
        pooled = self.norm(attn_out.mean(dim=1))
        feat = self.mlp(pooled)
        raw_vec = self.vector_head(feat)
        norm_vec = F.normalize(raw_vec, p=2, dim=-1, eps=1e-6)
        junction_logits = self.junction_head(feat)
        depth = F.softplus(self.depth_head(feat))
        return TopologicalGoalPrediction(
            pellet_cluster_vector=norm_vec,
            junction_exit_logits=junction_logits,
            corridor_depth_estimate=depth,
        )
