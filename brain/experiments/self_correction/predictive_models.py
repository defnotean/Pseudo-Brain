"""Phase B, C, F: Predictive recurrent models with surprise feedback & fast plasticity.

Architectural flow:
1. Encode current observation to compact latent z_t in R^128.
2. Form recurrent input: [z_t, prev_action, surprise_embedding_t].
3. Update persistent recurrent state:
   - GRU: h_t = GRU(input, h_{t-1})
   - Thoughtlet: K=32 thoughtlets updated via BrainCell + self-attention.
4. Predict action distribution: logits_t = ActionHead(h_t).
5. Predict future latent: z_hat_{t+1} = Predictor(h_t, action_t).
6. Next tick: compute prediction error E_{t+1} = ||z_hat_{t+1} - z_{t+1}|| and embed to e_{t+1} in R^16.
7. Optional Fast Episodic Plasticity (Phase F):
   Fast episodic trace P_t updated by surprise, modulates action policy during episode,
   resets cleanly to 0 on episode reset (no gradient/backprop during gameplay).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]  # brain/
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from memory_benchmark.models import ConvEncoder, _prev_onehot, BrainCell, N_FRAMES, ACTION_CLASSES, FEATURE_DIM

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
        # error: [B, 1] or [B]
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
        # state: [B, state_dim], action: [B] int64
        a_emb = self.act_emb(action.long())
        x = torch.cat([state, a_emb], dim=-1)
        return self.net(x)


class FastPlasticityModule(nn.Module):
    """Phase F: Fast episodic state adaptation.

    Maintains a dynamic bias P_t modulated by prediction surprise.
    Zero backprop during gameplay; resets cleanly per episode.
    """

    def __init__(self, state_dim: int, n_actions: int = ACTION_CLASSES, decay: float = 0.90, lr: float = 0.2):
        super().__init__()
        self.decay = decay
        self.lr = lr
        self.modulator = nn.Linear(state_dim + SURPRISE_DIM, n_actions)

    def init_trace(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, ACTION_CLASSES, device=device)

    def update(
        self,
        P_t: torch.Tensor,
        state: torch.Tensor,
        surprise: torch.Tensor,
    ) -> torch.Tensor:
        # state: [B, state_dim], surprise: [B, SURPRISE_DIM]
        delta = torch.tanh(self.modulator(torch.cat([state, surprise], dim=-1)))
        return self.decay * P_t + self.lr * delta


# --- Predictive GRU Model ---

class PredictiveGRUModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        latent_dim: int = LATENT_DIM,
        use_plasticity: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.use_plasticity = use_plasticity

        self.encoder = ConvEncoder()
        self.latent_proj = nn.Linear(FEATURE_DIM, latent_dim)
        self.surprise_encoder = SurpriseEncoder(SURPRISE_DIM)

        in_size = latent_dim + ACTION_CLASSES + SURPRISE_DIM
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=hidden_size)

        self.action_head = nn.Linear(hidden_size, ACTION_CLASSES)
        self.predictor = FutureLatentPredictor(state_dim=hidden_size, latent_dim=latent_dim)

        if use_plasticity:
            self.plasticity = FastPlasticityModule(state_dim=hidden_size)
        else:
            self.plasticity = None

    def encode_observation(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, N_FRAMES, 3, 16, 16]
        B = frames.size(0)
        x = frames.reshape(B, N_FRAMES * 3, 16, 16)
        if x.size(-2) != 32:
            x = F.interpolate(x, size=(32, 32), mode="nearest")
        feats = self.encoder(x)
        return self.latent_proj(feats)  # [B, LATENT_DIM]

    def forward_step(
        self,
        z_t: torch.Tensor,                  # [B, LATENT_DIM]
        prev_action: torch.Tensor,          # [B]
        surprise_t: torch.Tensor,           # [B, 1] scalar prediction error
        h: Optional[torch.Tensor] = None,   # [B, 384]
        P_t: Optional[torch.Tensor] = None, # [B, ACTION_CLASSES]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = z_t.size(0)
        dev = z_t.device
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=dev)
        if self.use_plasticity and P_t is None:
            P_t = self.plasticity.init_trace(B, dev)

        # 1. Embed surprise
        e_t = self.surprise_encoder(surprise_t)  # [B, SURPRISE_DIM]

        # 2. Form recurrent input
        a_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x_in = torch.cat([z_t, a_onehot, e_t], dim=-1)

        # 3. Recurrent update
        h_new = self.gru(x_in, h)

        # 4. Action logits
        base_logits = self.action_head(h_new)
        if self.use_plasticity:
            P_t = self.plasticity.update(P_t, h_new, e_t)
            logits = base_logits + P_t
        else:
            logits = base_logits

        return logits, h_new, e_t, P_t

    def predict_next_latent(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(h, action)


# --- Predictive Thoughtlet Model ---

class PredictiveThoughtletModel(nn.Module):
    def __init__(
        self,
        K: int = 32,
        thought_size: int = 12,
        latent_dim: int = LATENT_DIM,
        use_plasticity: bool = False,
    ):
        super().__init__()
        self.K = K
        self.thought_size = thought_size
        self.total_state = K * thought_size  # 384
        self.latent_dim = latent_dim
        self.use_plasticity = use_plasticity

        self.encoder = ConvEncoder()
        self.latent_proj = nn.Linear(FEATURE_DIM, latent_dim)
        self.surprise_encoder = SurpriseEncoder(SURPRISE_DIM)

        in_size = latent_dim + ACTION_CLASSES + SURPRISE_DIM
        self.brain_cell = BrainCell(input_size=in_size, thought_size=thought_size)
        self.attention = nn.MultiheadAttention(embed_dim=thought_size, num_heads=4, batch_first=True)
        self.attn_proj = nn.Linear(thought_size, thought_size)

        self.action_head = nn.Linear(self.total_state, ACTION_CLASSES)
        self.predictor = FutureLatentPredictor(state_dim=self.total_state, latent_dim=latent_dim)

        if use_plasticity:
            self.plasticity = FastPlasticityModule(state_dim=self.total_state)
        else:
            self.plasticity = None

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
        thoughts: Optional[torch.Tensor] = None, # [B, K, thought_size]
        P_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = z_t.size(0)
        dev = z_t.device
        if thoughts is None:
            thoughts = torch.zeros(B, self.K, self.thought_size, device=dev)
        if self.use_plasticity and P_t is None:
            P_t = self.plasticity.init_trace(B, dev)

        e_t = self.surprise_encoder(surprise_t)
        a_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x_in = torch.cat([z_t, a_onehot, e_t], dim=-1)

        # Parallel thoughtlet update
        x_expanded = x_in.unsqueeze(1).expand(-1, self.K, -1)
        x_flat = x_expanded.reshape(B * self.K, -1)
        t_flat = thoughts.reshape(B * self.K, self.thought_size)
        new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

        attn_out, _ = self.attention(new_t, new_t, new_t)
        thoughts_new = new_t + self.attn_proj(attn_out)

        flat_state = thoughts_new.reshape(B, -1)
        base_logits = self.action_head(flat_state)

        if self.use_plasticity:
            P_t = self.plasticity.update(P_t, flat_state, e_t)
            logits = base_logits + P_t
        else:
            logits = base_logits

        return logits, thoughts_new, e_t, P_t

    def predict_next_latent(self, thoughts: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = thoughts.size(0)
        flat_state = thoughts.reshape(B, -1)
        return self.predictor(flat_state, action)
