"""Model architectures for Rapid Online Adaptation Benchmark.

Includes:
1. ReactiveModel: ConvEncoder -> FC (no recurrence, no memory)
2. PredictiveGRUModel: 384-d GRU with latent prediction and fast plasticity
3. PredictiveThoughtletModel: K=32 parallel thoughtlets with latent prediction and fast plasticity

Plasticity features:
- P_t: Fast associative state updated online without backprop
- Diagnostic stats: tracks ||P_t||, ||Delta P_t||, and correlation with prediction error
- Configurable decay (default 0.98 for inter-trial retention)
- Ablation flags:
  * use_plasticity=False: disables P_t entirely (returns 0)
  * ablate_surprise=True: forces surprise input e_t = 0
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from memory_benchmark.models import (
    ConvEncoder,
    _prev_onehot,
    BrainCell,
    N_FRAMES,
    ACTION_CLASSES,
    FEATURE_DIM,
)

LATENT_DIM = 128
SURPRISE_DIM = 16


class SurpriseEncoder(nn.Module):
    """Embeds scalar prediction error distance into a compact surprise representation."""

    def __init__(self, out_dim: int = SURPRISE_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        if error.dim() == 1:
            error = error.unsqueeze(-1)
        return self.fc(error)


class FutureLatentPredictor(nn.Module):
    """Predicts next latent state z_hat_{t+1} given recurrent state and chosen action."""

    def __init__(self, state_dim: int, latent_dim: int = LATENT_DIM, n_actions: int = ACTION_CLASSES):
        super().__init__()
        self.act_emb = nn.Embedding(n_actions, 16)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a_emb = self.act_emb(action.long())
        x = torch.cat([state, a_emb], dim=-1)
        return self.net(x)


class FastPlasticityModule(nn.Module):
    """Fast episodic state adaptation (P_t).

    Updates online during gameplay without backprop:
        P_{t+1} = gamma * P_t + eta * tanh(W [h_t, e_t])
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = ACTION_CLASSES,
        decay: float = 0.98,
        lr: float = 0.25,
    ):
        super().__init__()
        self.decay = decay
        self.lr = lr
        self.n_actions = n_actions
        self.modulator = nn.Linear(state_dim + SURPRISE_DIM, n_actions)
        self.surprise_gate = nn.Sequential(
            nn.Linear(SURPRISE_DIM, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.surprise_gate[0].bias, -2.0)
        self.scale = nn.Parameter(torch.tensor(1.5))

    def init_trace(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_actions, device=device)

    def update(
        self,
        P_t: torch.Tensor,
        state: torch.Tensor,
        surprise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (P_{t+1}, delta_P)."""
        gate = self.surprise_gate(surprise)
        delta = gate * torch.tanh(self.modulator(torch.cat([state, surprise], dim=-1)))
        P_next = self.decay * P_t + self.lr * delta
        return P_next, delta


# --- 1. Reactive Model Baseline ---

class ReactiveModel(nn.Module):
    """Purely reactive feedforward policy (no recurrence, no memory)."""

    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.head = nn.Linear(FEATURE_DIM, ACTION_CLASSES)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B = frames.size(0)
        x = frames.reshape(B, N_FRAMES * 3, 16, 16)
        if x.size(-2) != 32:
            x = F.interpolate(x, size=(32, 32), mode="nearest")
        feats = self.encoder(x)
        return self.head(feats)


# --- 2. Predictive GRU Model ---

class PredictiveGRUModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        latent_dim: int = LATENT_DIM,
        use_plasticity: bool = True,
        ablate_surprise: bool = False,
        plastic_decay: float = 0.98,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.use_plasticity = use_plasticity
        self.ablate_surprise = ablate_surprise

        self.encoder = ConvEncoder()
        self.latent_proj = nn.Linear(FEATURE_DIM, latent_dim)
        self.surprise_encoder = SurpriseEncoder(SURPRISE_DIM)

        in_size = latent_dim + ACTION_CLASSES + SURPRISE_DIM
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=hidden_size)
        self.action_head = nn.Linear(hidden_size, ACTION_CLASSES)
        self.predictor = FutureLatentPredictor(state_dim=hidden_size, latent_dim=latent_dim)
        self.plasticity = FastPlasticityModule(state_dim=hidden_size, decay=plastic_decay)

    def encode_observation(self, frames: torch.Tensor) -> torch.Tensor:
        B = frames.size(0)
        x = frames.reshape(B, N_FRAMES * 3, 16, 16)
        if x.size(-2) != 32:
            x = F.interpolate(x, size=(32, 32), mode="nearest")
        feats = self.encoder(x)
        return self.latent_proj(feats)

    def forward_step(
        self,
        z_t: torch.Tensor,
        prev_action: torch.Tensor,
        surprise_t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        P_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B = z_t.size(0)
        dev = z_t.device
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=dev)
        if self.use_plasticity and P_t is None:
            P_t = self.plasticity.init_trace(B, dev)

        # Surprise processing with optional ablation
        if self.ablate_surprise:
            eff_surprise = torch.zeros_like(surprise_t)
        else:
            eff_surprise = surprise_t

        e_t = self.surprise_encoder(eff_surprise)

        a_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x_in = torch.cat([z_t, a_onehot, e_t], dim=-1)

        h_new = self.gru(x_in, h)
        base_logits = self.action_head(h_new)

        delta_P = None
        if self.use_plasticity and P_t is not None:
            P_t, delta_P = self.plasticity.update(P_t, h_new, e_t)
            logits = base_logits + self.plasticity.scale * P_t
        else:
            logits = base_logits

        return logits, h_new, e_t, P_t, delta_P

    def predict_next_latent(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(h, action)


# --- 3. Predictive Thoughtlet Model ---

class PredictiveThoughtletModel(nn.Module):
    def __init__(
        self,
        K: int = 32,
        thought_size: int = 12,
        latent_dim: int = LATENT_DIM,
        use_plasticity: bool = True,
        ablate_surprise: bool = False,
        plastic_decay: float = 0.98,
        plastic_lr: float = 0.25,
    ):
        super().__init__()
        self.K = K
        self.thought_size = thought_size
        self.total_state = K * thought_size  # 384
        self.latent_dim = latent_dim
        self.use_plasticity = use_plasticity
        self.ablate_surprise = ablate_surprise

        self.encoder = ConvEncoder()
        self.latent_proj = nn.Linear(FEATURE_DIM, latent_dim)
        self.surprise_encoder = SurpriseEncoder(SURPRISE_DIM)

        in_size = latent_dim + ACTION_CLASSES + SURPRISE_DIM
        self.brain_cell = BrainCell(input_size=in_size, thought_size=thought_size)
        self.attention = nn.MultiheadAttention(embed_dim=thought_size, num_heads=4, batch_first=True)
        self.attn_proj = nn.Linear(thought_size, thought_size)

        self.action_head = nn.Linear(self.total_state, ACTION_CLASSES)
        self.predictor = FutureLatentPredictor(state_dim=self.total_state, latent_dim=latent_dim)
        self.plasticity = FastPlasticityModule(
            state_dim=self.total_state,
            decay=plastic_decay,
            lr=plastic_lr,
        )

    def encode_observation(self, frames: torch.Tensor) -> torch.Tensor:
        B = frames.size(0)
        x = frames.reshape(B, N_FRAMES * 3, 16, 16)
        if x.size(-2) != 32:
            x = F.interpolate(x, size=(32, 32), mode="nearest")
        feats = self.encoder(x)
        return self.latent_proj(feats)

    def forward_step(
        self,
        z_t: torch.Tensor,
        prev_action: torch.Tensor,
        surprise_t: torch.Tensor,
        thoughts: Optional[torch.Tensor] = None,
        P_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B = z_t.size(0)
        dev = z_t.device
        if thoughts is None:
            thoughts = torch.zeros(B, self.K, self.thought_size, device=dev)
        if self.use_plasticity and P_t is None:
            P_t = self.plasticity.init_trace(B, dev)

        if self.ablate_surprise:
            eff_surprise = torch.zeros_like(surprise_t)
        else:
            eff_surprise = surprise_t

        e_t = self.surprise_encoder(eff_surprise)
        a_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x_in = torch.cat([z_t, a_onehot, e_t], dim=-1)

        # Parallel thoughtlet update via BrainCell
        x_expanded = x_in.unsqueeze(1).expand(-1, self.K, -1)
        x_flat = x_expanded.reshape(B * self.K, -1)
        t_flat = thoughts.reshape(B * self.K, self.thought_size)
        new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

        # Cross-thoughtlet self-attention
        attn_out, _ = self.attention(new_t, new_t, new_t)
        thoughts_new = new_t + self.attn_proj(attn_out)

        flat_state = thoughts_new.reshape(B, -1)
        base_logits = self.action_head(flat_state)

        delta_P = None
        if self.use_plasticity and P_t is not None:
            P_t, delta_P = self.plasticity.update(P_t, flat_state, e_t)
            logits = base_logits + self.plasticity.scale * P_t
        else:
            logits = base_logits

        return logits, thoughts_new, e_t, P_t, delta_P

    def predict_next_latent(self, thoughts: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = thoughts.size(0)
        flat_state = thoughts.reshape(B, -1)
        return self.predictor(flat_state, action)


__all__ = [
    "SurpriseEncoder",
    "FutureLatentPredictor",
    "FastPlasticityModule",
    "ReactiveModel",
    "PredictiveGRUModel",
    "PredictiveThoughtletModel",
    "LATENT_DIM",
    "SURPRISE_DIM",
]
