"""Model architectures for the Memory Benchmark.

Standardized model definitions:
- ReactiveModel: ConvEncoder -> FC (no recurrence)
- GRUModel: ConvEncoder -> GRU (384-d recurrent state)
- ThoughtletModel: ConvEncoder -> 32 parallel thoughtlets (12-d each, shared BrainCell + attention)

Features:
- Native fast GRUCell on CPU with fallback for DML
- Identical ConvEncoder across all models (4 frames of 16x16 RGB + prev action)
- Output: 5 action logits (Idle, W, A, S, D)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FRAMES = 4
ACTION_CLASSES = 5
UPSAMPLE = 32
FEATURE_DIM = 48 * 4 * 4  # 768


class ConvEncoder(nn.Module):
    """Shared conv encoder: [B, 12, 32, 32] -> [B, 768]."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(N_FRAMES * 3, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(48)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(F.relu(self.bn1(self.conv1(x))))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        return h.reshape(h.size(0), -1)


def prepare_input(frames: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    B = frames.size(0)
    x = frames.reshape(B, N_FRAMES * 3, 16, 16)
    if x.size(-2) != UPSAMPLE:
        x = F.interpolate(x, size=(UPSAMPLE, UPSAMPLE), mode="nearest")
    return x


def _prev_onehot(prev_action: torch.Tensor, n_classes: int = ACTION_CLASSES) -> torch.Tensor:
    dev = prev_action.device
    if dev.type == "privateuseone":
        return F.one_hot(prev_action.cpu().long(), n_classes).float().to(dev)
    return F.one_hot(prev_action.long(), n_classes).float()


# --- Model A: Reactive baseline ---
class ReactiveModel(nn.Module):
    def __init__(self, n_actions: int = ACTION_CLASSES):
        super().__init__()
        self.encoder = ConvEncoder()
        self.fc1 = nn.Linear(FEATURE_DIM + n_actions, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(
        self,
        frames: torch.Tensor,
        prev_action: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)
        prev_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, prev_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits, None


# --- Model B: GRU Recurrent ---
class GRUModel(nn.Module):
    def __init__(self, n_actions: int = ACTION_CLASSES, hidden_size: int = 384):
        super().__init__()
        self.encoder = ConvEncoder()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size=FEATURE_DIM + n_actions, hidden_size=hidden_size)
        self.action_head = nn.Linear(hidden_size, n_actions)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        prev_action: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)
        prev_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, prev_onehot], dim=-1)

        if h is None:
            h = self.init_hidden(B, frames.device)

        h = self.gru(x, h)
        logits = self.action_head(h)
        return logits, h


# --- BrainCell for Thoughtlet ---
class BrainCell(nn.Module):
    def __init__(self, input_size: int, thought_size: int):
        super().__init__()
        self.W_ir = nn.Linear(input_size, thought_size)
        self.W_hr = nn.Linear(thought_size, thought_size)
        self.W_iz = nn.Linear(input_size, thought_size)
        self.W_hz = nn.Linear(thought_size, thought_size)
        self.W_in = nn.Linear(input_size, thought_size)
        self.W_hn = nn.Linear(thought_size, thought_size)

    def forward(self, thought: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(thought))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(thought))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(thought))
        return (1 - z) * n + z * thought


# --- Model C: Thoughtlet recurrent ---
class ThoughtletModel(nn.Module):
    def __init__(self, n_actions: int = ACTION_CLASSES, K: int = 32, thought_size: int = 12, cycles: int = 1):
        super().__init__()
        self.encoder = ConvEncoder()
        self.K = K
        self.thought_size = thought_size
        self.cycles = cycles
        self.n_actions = n_actions
        total_state = K * thought_size  # 384

        self.brain_cell = BrainCell(input_size=FEATURE_DIM + n_actions, thought_size=thought_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=thought_size, num_heads=4, batch_first=True
        )
        self.attn_proj = nn.Linear(thought_size, thought_size)
        self.action_head = nn.Linear(total_state, n_actions)

    def init_thoughts(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.K, self.thought_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,
        prev_action: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)
        prev_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, prev_onehot], dim=-1)

        if h is None:
            h = self.init_thoughts(B, frames.device)

        thoughts = h
        for _ in range(self.cycles):
            x_expanded = x.unsqueeze(1).expand(-1, self.K, -1)
            x_flat = x_expanded.reshape(B * self.K, -1)
            t_flat = thoughts.reshape(B * self.K, self.thought_size)
            new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

            attn_out, _ = self.attention(new_t, new_t, new_t)
            thoughts = new_t + self.attn_proj(attn_out)

        flat_thoughts = thoughts.reshape(B, -1)
        logits = self.action_head(flat_thoughts)
        return logits, thoughts


# --- Model D: Consequence-Gated Plastic Thoughtlet (CGP) ---
class SurpriseEncoder(nn.Module):
    """Embeds scalar prediction error distance into a compact surprise representation."""

    def __init__(self, out_dim: int = 16):
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


class FastPlasticityModule(nn.Module):
    """Fast episodic state adaptation (P_t).

    Updates online during gameplay without backprop:
        P_{t+1} = gamma * P_t + eta * gate * tanh(W [h_t, e_t])
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = ACTION_CLASSES,
        decay: float = 0.999,
        lr: float = 0.25,
    ):
        super().__init__()
        self.decay = decay
        self.lr = lr
        self.n_actions = n_actions
        self.modulator = nn.Linear(state_dim + 16, n_actions)
        self.surprise_gate = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.surprise_gate[0].bias, -1.5)
        self.scale = nn.Parameter(torch.tensor(1.5))

    def init_trace(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_actions, device=device)

    def update(
        self,
        P_t: torch.Tensor,
        state: torch.Tensor,
        surprise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = self.surprise_gate(surprise)
        delta = gate * torch.tanh(self.modulator(torch.cat([state, surprise], dim=-1)))
        P_next = self.decay * P_t + self.lr * delta
        return P_next, delta, gate


class PredictiveCGPThoughtletModel(nn.Module):
    """Consequence-Gated Plastic Thoughtlet for Sequential POMDP.

    Combines:
    1. Parallel K=32 thoughtlets with cross-attention
    2. Endogenous Cognitive Input Gating to prevent corridor state diffusion
    3. Milestone head for explicit latching of task stages (Key Collected, Door Open)
    4. Predictive latent dynamics generating endogenous surprise
    5. Fast Plasticity P_t with episodic retention
    """

    def __init__(
        self,
        n_actions: int = ACTION_CLASSES,
        K: int = 32,
        thought_size: int = 12,
        latent_dim: int = 128,
        surprise_dim: int = 16,
        plastic_decay: float = 0.999,
        plastic_lr: float = 0.25,
    ):
        super().__init__()
        self.K = K
        self.thought_size = thought_size
        self.total_state = K * thought_size  # 384
        self.latent_dim = latent_dim
        self.surprise_dim = surprise_dim
        self.n_actions = n_actions

        self.encoder = ConvEncoder()
        self.latent_proj = nn.Linear(FEATURE_DIM, latent_dim)
        self.surprise_encoder = SurpriseEncoder(surprise_dim)

        in_size = latent_dim + n_actions + surprise_dim
        self.brain_cell = BrainCell(input_size=in_size, thought_size=thought_size)
        self.attention = nn.MultiheadAttention(embed_dim=thought_size, num_heads=4, batch_first=True)
        self.attn_proj = nn.Linear(thought_size, thought_size)

        # Cognitive Input Gate: g_t in [0, 1]
        self.cognitive_gate = nn.Sequential(
            nn.Linear(in_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cognitive_gate[0].bias, 1.5)

        # Milestone Head: predicts [has_key, door_open] logits
        self.milestone_head = nn.Linear(self.total_state, 2)

        # Predictive Latent Predictor
        self.act_emb = nn.Embedding(n_actions, 16)
        self.latent_predictor = nn.Sequential(
            nn.Linear(self.total_state + 16, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Fast Episodic Plasticity Module
        self.plasticity = FastPlasticityModule(
            state_dim=self.total_state,
            n_actions=n_actions,
            decay=plastic_decay,
            lr=plastic_lr,
        )

        # Action Policy Head: combines thoughts, milestone beliefs, and plasticity
        self.action_head = nn.Linear(self.total_state + 2, n_actions)

    def init_hidden(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "thoughts": torch.zeros(batch_size, self.K, self.thought_size, device=device),
            "P_t": self.plasticity.init_trace(batch_size, device),
            "prev_m": torch.zeros(batch_size, 2, device=device),
            "prev_z_hat": torch.zeros(batch_size, self.latent_dim, device=device),
        }

    def forward(
        self,
        frames: torch.Tensor,
        prev_action: torch.Tensor,
        h: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = frames.size(0)
        dev = frames.device

        if h is None or not isinstance(h, dict):
            h = self.init_hidden(B, dev)

        thoughts = h["thoughts"]
        P_t = h["P_t"]
        prev_m = h["prev_m"]
        prev_z_hat = h["prev_z_hat"]

        # 1. Visual encode
        x_raw = prepare_input(frames, prev_action)
        feats = self.encoder(x_raw)
        z_t = self.latent_proj(feats)

        # 2. Consequence / Milestone Surprise
        m_logits = self.milestone_head(thoughts.reshape(B, -1))
        m_prob = torch.sigmoid(m_logits)
        m_diff = torch.abs(m_prob[:, 0] - prev_m[:, 0])
        z_err = torch.norm(z_t - prev_z_hat, dim=-1)
        surprise_val = z_err + 3.0 * m_diff
        surprise_emb = self.surprise_encoder(surprise_val.unsqueeze(-1))

        # 3. Input formatting
        prev_onehot = _prev_onehot(prev_action, self.n_actions)
        x_in = torch.cat([z_t, prev_onehot, surprise_emb], dim=-1)

        # 4. Cognitive Input Gating & BrainCell
        x_expanded = x_in.unsqueeze(1).expand(-1, self.K, -1)
        x_flat = x_expanded.reshape(B * self.K, -1)
        t_flat = thoughts.reshape(B * self.K, self.thought_size)
        new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

        attn_out, _ = self.attention(new_t, new_t, new_t)
        t_candidate = new_t + self.attn_proj(attn_out)

        salience = self.cognitive_gate(x_in).unsqueeze(1)
        thoughts_next = (1.0 - salience) * thoughts + salience * t_candidate

        # 5. Fast Plasticity Update
        flat_state = thoughts_next.reshape(B, -1)
        P_next, delta_P, gate = self.plasticity.update(P_t, flat_state, surprise_emb)

        # 6. Re-evaluate milestone belief from updated thoughts
        m_logits_post = self.milestone_head(flat_state)
        m_prob_post = torch.sigmoid(m_logits_post)

        # 7. Action Logits
        policy_in = torch.cat([flat_state, m_prob_post], dim=-1)
        base_logits = self.action_head(policy_in)
        logits = base_logits + self.plasticity.scale * P_next

        # 8. Predict next latent z_hat
        chosen_act = logits.argmax(dim=-1)
        act_emb = self.act_emb(chosen_act)
        pred_in = torch.cat([flat_state, act_emb], dim=-1)
        z_hat = self.latent_predictor(pred_in)

        new_h = {
            "thoughts": thoughts_next,
            "P_t": P_next,
            "prev_m": m_prob_post.detach(),
            "prev_z_hat": z_hat.detach(),
            "milestone_logits": m_logits_post,
        }
        return logits, new_h

    def forward_sequence(
        self,
        frames: torch.Tensor,
        prev_actions: torch.Tensor,
        actions: torch.Tensor,
        ss_rate: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = frames.shape[:2]
        dev = frames.device

        # Pre-encode all frames in parallel batch
        frames_flat = frames.reshape(B * T, N_FRAMES * 3, 16, 16)
        if frames_flat.size(-2) != UPSAMPLE:
            frames_flat = F.interpolate(frames_flat, size=(UPSAMPLE, UPSAMPLE), mode="nearest")
        feats = self.encoder(frames_flat)
        all_z = self.latent_proj(feats).reshape(B, T, self.latent_dim)

        thoughts = torch.zeros(B, self.K, self.thought_size, device=dev)
        P_t = self.plasticity.init_trace(B, dev)
        prev_m = torch.zeros(B, 2, device=dev)
        prev_z_hat = torch.zeros(B, self.latent_dim, device=dev)
        curr_prev_act = prev_actions[:, 0]

        logits_list = []
        m_logits_list = []
        z_hat_list = []

        for t in range(T):
            z_t = all_z[:, t]
            m_logits = self.milestone_head(thoughts.reshape(B, -1))
            m_prob = torch.sigmoid(m_logits)
            m_diff = torch.abs(m_prob[:, 0] - prev_m[:, 0])
            z_err = torch.norm(z_t - prev_z_hat, dim=-1)
            surprise_val = z_err + 3.0 * m_diff
            surprise_emb = self.surprise_encoder(surprise_val.unsqueeze(-1))

            prev_onehot = _prev_onehot(curr_prev_act, self.n_actions)
            x_in = torch.cat([z_t, prev_onehot, surprise_emb], dim=-1)

            x_expanded = x_in.unsqueeze(1).expand(-1, self.K, -1)
            x_flat = x_expanded.reshape(B * self.K, -1)
            t_flat = thoughts.reshape(B * self.K, self.thought_size)
            new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

            attn_out, _ = self.attention(new_t, new_t, new_t)
            t_candidate = new_t + self.attn_proj(attn_out)

            salience = self.cognitive_gate(x_in).unsqueeze(1)
            thoughts = (1.0 - salience) * thoughts + salience * t_candidate

            flat_state = thoughts.reshape(B, -1)
            P_t, _, _ = self.plasticity.update(P_t, flat_state, surprise_emb)

            m_logits_post = self.milestone_head(flat_state)
            m_prob_post = torch.sigmoid(m_logits_post)

            policy_in = torch.cat([flat_state, m_prob_post], dim=-1)
            base_logits = self.action_head(policy_in)
            logits = base_logits + self.plasticity.scale * P_t

            logits_list.append(logits)
            m_logits_list.append(m_logits_post)

            act_target = actions[:, t].long()
            act_emb = self.act_emb(act_target)
            pred_in = torch.cat([flat_state, act_emb], dim=-1)
            z_hat = self.latent_predictor(pred_in)
            z_hat_list.append(z_hat)

            prev_m = m_prob_post.detach()
            prev_z_hat = z_hat.detach()

            if ss_rate > 0.0:
                use_pred = torch.rand(B, device=dev) < ss_rate
                curr_prev_act = torch.where(use_pred, logits.detach().argmax(dim=-1), actions[:, t])
            else:
                curr_prev_act = actions[:, t]

        all_logits = torch.stack(logits_list, dim=1)        # [B, T, n_actions]
        all_m_logits = torch.stack(m_logits_list, dim=1)    # [B, T, 2]
        all_z_hat = torch.stack(z_hat_list, dim=1)          # [B, T, latent_dim]

        return all_logits, all_m_logits, all_z_hat, all_z


def make_model(model_type: str) -> nn.Module:
    if model_type == "reactive":
        return ReactiveModel()
    elif model_type == "gru":
        return GRUModel()
    elif model_type == "thoughtlet":
        return ThoughtletModel()
    elif model_type == "cgp_thoughtlet":
        return PredictiveCGPThoughtletModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def count_parameters(model: nn.Module) -> Dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
