"""Natural Language Goal Specification and Embedding for Pseudo-Brain Autonomous Agent."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


@dataclass
class GoalMilestone:
    """Endogenous hierarchical sub-goal / milestone.

    Represents an intermediate milestone that must be achieved and consolidated
    before branching into dependent tasks.
    """
    milestone_id: str
    description: str
    tool_name: Optional[str] = None
    tool_idx: Optional[int] = None
    verification_fn: Optional[Callable[[], bool]] = None
    prerequisites: List[str] = field(default_factory=list)
    reinforcement: float = 1.0  # Internal milestone reinforcement m_t > 0
    completed: bool = False
    step_completed: Optional[int] = None


@dataclass
class GoalSpecification:
    """Represents a verifiable autonomous task."""
    goal_id: str
    text: str
    verification_fn: Optional[Callable[[], bool]] = None
    verification_command: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    milestones: List[GoalMilestone] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Evaluate if the task completion condition is met."""
        if self.verification_fn is not None:
            return bool(self.verification_fn())
        if self.milestones and all(m.completed for m in self.milestones):
            return True
        return False

    def get_uncompleted_milestones(self) -> List[GoalMilestone]:
        """Return milestones that have not yet been completed."""
        return [m for m in self.milestones if not m.completed]

    def get_ready_milestones(self) -> List[GoalMilestone]:
        """Return uncompleted milestones whose prerequisites are all satisfied."""
        completed_ids = {m.milestone_id for m in self.milestones if m.completed}
        return [
            m for m in self.milestones
            if not m.completed and all(p in completed_ids for p in m.prerequisites)
        ]

    def check_milestone_completion(
        self,
        tool_name: Optional[str] = None,
        tool_idx: Optional[int] = None,
        tool_success: bool = True,
        step: Optional[int] = None,
    ) -> List[GoalMilestone]:
        """Check and mark newly completed milestones whose prerequisites are met."""
        if not tool_success:
            return []
        completed_ids = {m.milestone_id for m in self.milestones if m.completed}
        newly_completed: List[GoalMilestone] = []

        for m in self.milestones:
            if m.completed:
                continue
            if not all(p in completed_ids for p in m.prerequisites):
                continue

            matched = False
            if m.verification_fn is not None and m.verification_fn():
                matched = True
            elif tool_name is not None and m.tool_name == tool_name:
                matched = True
            elif tool_idx is not None and m.tool_idx == tool_idx:
                matched = True

            if matched:
                m.completed = True
                m.step_completed = step
                completed_ids.add(m.milestone_id)
                newly_completed.append(m)

        return newly_completed



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
