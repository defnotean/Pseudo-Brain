"""Natural Language Goal Specification and Embedding for Pseudo-Brain Autonomous Agent."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


@dataclass
class GoalSpecification:
    """Represents a verifiable autonomous task."""
    goal_id: str
    text: str
    verification_fn: Optional[Callable[[], bool]] = None
    verification_command: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Evaluate if the task completion condition is met."""
        if self.verification_fn is not None:
            return bool(self.verification_fn())
        return False


class GoalEncoder(nn.Module):
    """Encodes arbitrary text goals into a fixed-dimensional continuous representation.

    Uses deterministic token-hash projections + a learned LayerNorm MLP to guarantee
    zero external dependencies and consistent embeddings across restarts.
    """

    def __init__(self, embedding_dim: int = 128, vocab_buckets: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_buckets = vocab_buckets
        self.table = nn.Embedding(vocab_buckets, 64)
        self.proj = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _hash_token(self, token: str) -> int:
        h = int(hashlib.md5(token.lower().encode("utf-8")).hexdigest()[:8], 16)
        return h % self.vocab_buckets

    def forward(self, text_or_tokens: str | List[str]) -> torch.Tensor:
        if isinstance(text_or_tokens, str):
            tokens = text_or_tokens.strip().split()
        else:
            tokens = text_or_tokens

        if not tokens:
            tokens = ["<empty>"]

        indices = torch.tensor([self._hash_token(t) for t in tokens], dtype=torch.long)
        emb = self.table(indices)  # (N, 64)
        pooled = emb.mean(dim=0, keepdim=True)  # (1, 64)
        return self.proj(pooled)  # (1, embedding_dim)
