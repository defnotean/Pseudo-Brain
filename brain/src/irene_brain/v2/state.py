"""Core V2 Runtime State — explicit dataclasses for all timescales.

No hidden globals. All state transitions are explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch
from torch import Tensor


@dataclass
class PredictionErrorState:
    """Prediction error at time t — computed BEFORE new cognition.

    This is the "how wrong was I" signal from the previous step.
    """
    latent_error: Optional[Tensor] = None       # [B, W] or [B, K, W]
    cognitive_error: Optional[Tensor] = None    # fused [B, W] or [B, K, W]
    reward_error: Optional[Tensor] = None       # [B, 1] or [B, K, 1]
    hazard_error: Optional[Tensor] = None       # [B, 1] or [B, K, 1]
    confidence_error: Optional[Tensor] = None   # [B, 1] or [B, K, 1]
    surprise: Optional[Tensor] = None           # [B, 1] or [B, K, 1] or [B]

    # Intervention support
    def with_intervention(self, mode: Literal["normal", "zero", "scrambled", "stale", "oracle"]) -> "PredictionErrorState":
        if mode == "zero":
            return PredictionErrorState(
                latent_error=torch.zeros_like(self.latent_error) if self.latent_error is not None else None,
                cognitive_error=torch.zeros_like(self.cognitive_error) if self.cognitive_error is not None else None,
                reward_error=torch.zeros_like(self.reward_error) if self.reward_error is not None else None,
                hazard_error=torch.zeros_like(self.hazard_error) if self.hazard_error is not None else None,
                confidence_error=torch.zeros_like(self.confidence_error) if self.confidence_error is not None else None,
                surprise=torch.zeros_like(self.surprise) if self.surprise is not None else None,
            )
        elif mode == "scrambled":
            # Random permutation of error magnitudes within batch
            def scramble(t):
                if t is None:
                    return None
                idx = torch.randperm(t.shape[0], device=t.device)
                return t[idx]
            return PredictionErrorState(
                latent_error=scramble(self.latent_error),
                cognitive_error=scramble(self.cognitive_error),
                reward_error=scramble(self.reward_error),
                hazard_error=scramble(self.hazard_error),
                confidence_error=scramble(self.confidence_error),
                surprise=scramble(self.surprise),
            )
        elif mode == "stale":
            # Return zeros (simulating no prediction error signal)
            return self.with_intervention("zero")
        elif mode == "oracle":
            # In practice, this would be filled with ground-truth errors
            # For now, return as-is (caller must provide oracle)
            return self
        return self


@dataclass
class PendingPrediction:
    """Predictions made at t for t+1 — used to compute next-step prediction error.

    Saved after each cognitive cycle to be compared against reality.
    """
    predicted_next_latent: Optional[Tensor] = None    # [B, K, W] or [B, W]
    predicted_reward: Optional[Tensor] = None         # [B, K, 1] or [B, 1]
    predicted_reward_logits: Optional[Tensor] = None  # [B, bins]
    predicted_hazard: Optional[Tensor] = None         # [B, K, 1] or [B, 1]
    predicted_confidence: Optional[Tensor] = None     # [B, K, 1] or [B, 1]
    predicted_branch_logit: Optional[Tensor] = None   # [B, K, 1] or [B, 1]

    def detach_(self):
        """Stop-gradient all predictions so they don't chain into next-step gradients."""
        for attr in ("predicted_next_latent", "predicted_reward", "predicted_reward_logits", "predicted_hazard",
                     "predicted_confidence", "predicted_branch_logit"):
            v = getattr(self, attr)
            if v is not None:
                v.detach_()
        return self


@dataclass
class EpisodicMemoryEntry:
    """One entry in the episodic memory.

    All fields are tensors (can be batched for storage).
    """
    key: Tensor                          # [W] or [B, W] — query embedding
    value: Tensor                        # [W] or [B, W] — stored context
    action: Tensor                       # [1] or [B] — action taken
    predicted_consequence: dict          # dict of predicted outcomes
    observed_consequence: dict           # dict of actual outcomes
    prediction_error: dict               # dict of error magnitudes
    reward: Tensor                       # [1] or [B]
    hazard: Tensor                       # [1] or [B]
    confidence: Tensor                   # [1] or [B]
    trial_index: int
    timestamp: int


@dataclass
class EpisodicMemoryV2:
    """Bounded differentiable episodic memory.

    Capacity fixed at 256. Retrieval via learned similarity/attention.
    """
    capacity: int = 256
    retrieval_k: int = 8
    entries: list = field(default_factory=list)
    write_pointer: int = 0

    def __len__(self):
        return len(self.entries)

    def can_write(self) -> bool:
        return len(self.entries) < self.capacity

    def write(self, entry: EpisodicMemoryEntry, priority: float = 1.0):
        """Write entry. If full, replace lowest-priority entry (simplified)."""
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            # Replace oldest (simplest priority: overwrite by write_pointer)
            self.entries[self.write_pointer] = entry
            self.write_pointer = (self.write_pointer + 1) % self.capacity

    def query(self, query_vec: Tensor, k: Optional[int] = None) -> list[EpisodicMemoryEntry]:
        """Retrieve top-k entries by cosine similarity to query_vec.

        query_vec: [B, W] or [W]
        Returns list of entries (no gradients through retrieval indices).
        """
        if not self.entries:
            return []
        k = min(k or self.retrieval_k, len(self.entries))
        # Flatten query for similarity
        if query_vec.dim() == 2:
            q = query_vec  # [B, W]
        else:
            q = query_vec.unsqueeze(0)  # [1, W]

        # Stack entry keys: [N, W]
        keys = torch.stack([e.key.flatten() if e.key.dim() > 1 else e.key for e in self.entries])
        keys = keys.to(q.device)

        # Cosine similarity: [B, N]
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(keys, dim=-1)
        sim = q_norm @ k_norm.T  # [B, N]

        # Top-k indices per batch
        _, topk_idx = sim.topk(k, dim=-1)
        return [[self.entries[i.item()] for i in topk_idx[b]] for b in range(q.shape[0])]

    def clear(self):
        self.entries.clear()
        self.write_pointer = 0

    def with_intervention(self, mode: Literal["normal", "disable_read", "disable_write", "clear", "donor", "stale", "oracle"]):
        """Return intervention mode — actual logic in caller."""
        return mode


@dataclass
class SessionState:
    """Medium-timescale state — persists across trials within a session."""
    session_latent: Tensor                   # [B, W]
    episodic_memory: EpisodicMemoryV2
    trial_index: int = 0
    session_stats: dict = field(default_factory=dict)

    def reset_trial(self):
        """Called at trial boundary — increments trial counter, preserves memory."""
        self.trial_index += 1

    def reset_session(self):
        """Called at session boundary — clears everything except slow weights."""
        self.session_latent.zero_()
        self.episodic_memory.clear()
        self.trial_index = 0
        self.session_stats.clear()


@dataclass
class FastState:
    """Fast timescale — current trial state, reset at trial boundary."""
    belief: Tensor                           # [B, W]
    thoughts: Tensor                         # [B, K, W]
    prediction_error: Optional[PredictionErrorState] = None
    retrieved_memory: Optional[Tensor] = None  # [B, K, W] or [B, W]
    pending_prediction: Optional[PendingPrediction] = None

    def reset(self):
        """Reset at trial boundary (called by session.reset_trial)."""
        self.belief.zero_()
        self.thoughts.zero_()
        self.prediction_error = None
        self.retrieved_memory = None
        self.pending_prediction = None


@dataclass
class CoreV2State:
    """Complete V2 state — three timescales combined."""
    fast: FastState
    session: SessionState

    def reset_fast_state(self):
        self.fast.reset()

    def reset_episode_state(self):
        """Alias for reset_fast_state at trial boundary."""
        self.fast.reset()

    def reset_session_state(self):
        self.session.reset_session()

    def reset_all_state(self):
        self.reset_fast_state()
        self.reset_session_state()


# Import needed for EpisodicMemoryV2.query
import torch.nn.functional as F
