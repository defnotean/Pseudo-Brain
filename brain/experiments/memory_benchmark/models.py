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
from typing import Optional, Tuple
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


def make_model(model_type: str) -> nn.Module:
    if model_type == "reactive":
        return ReactiveModel()
    elif model_type == "gru":
        return GRUModel()
    elif model_type == "thoughtlet":
        return ThoughtletModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def count_parameters(model: nn.Module) -> Dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
