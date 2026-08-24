"""Session State V2 — medium-timescale persistent state across trials."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import CoreV2Config
from .state import SessionState, EpisodicMemoryV2 as EpisodicMemoryState
from .episodic_memory import EpisodicMemoryV2


class SessionStateV2(nn.Module):
    """Manages session-level state: session latent + episodic memory."""

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        self.session_latent_dim = config.session_latent_dim

        # Session latent update
        self.session_update = nn.Sequential(
            nn.Linear(W * 3, 128),
            nn.ReLU(),
            nn.Linear(128, W),
        )

        # Initialize episodic memory module
        self.episodic_memory = EpisodicMemoryV2(config)

    def init_session(self, batch_size: int, device: torch.device) -> SessionState:
        """Create fresh session state."""
        return SessionState(
            session_latent=torch.zeros(batch_size, self.session_latent_dim, device=device),
            episodic_memory=EpisodicMemoryState(
                capacity=self.config.episodic_capacity,
                retrieval_k=self.config.retrieval_k,
            ),
            trial_index=0,
            session_stats={},
        )

    def update_session(
        self,
        session_state: SessionState,
        belief: torch.Tensor,              # [B, W]
        prediction_error: Optional[object] = None,
        reward: Optional[torch.Tensor] = None,
        trial_summary: Optional[dict] = None,
    ) -> SessionState:
        """Update session latent at important events / trial end."""
        B, W = belief.shape
        device = belief.device

        # Build update signal
        signals = [belief, session_state.session_latent]
        if trial_summary and "final_belief" in trial_summary:
            signals.append(trial_summary["final_belief"])
        else:
            signals.append(torch.zeros_like(belief))

        update_input = torch.cat(signals, dim=-1)
        delta = self.session_update(update_input)
        session_state.session_latent = session_state.session_latent + 0.1 * delta

        if trial_summary:
            session_state.session_stats[f"trial_{session_state.trial_index}"] = trial_summary

        return session_state

    def end_trial(self, session_state: SessionState):
        """Called at trial boundary - increments trial counter."""
        session_state.reset_trial()

    def end_session(self, session_state: SessionState):
        """Called at session boundary - full reset."""
        session_state.reset_session()