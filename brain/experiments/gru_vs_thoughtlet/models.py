"""Clean GRU vs Thoughtlet benchmark models.

All models share the same conv encoder (matching ReactiveBaselineV1).
Only the recurrent pathway differs.
- Model A: Reactive baseline (no recurrence) — equivalent to ReactiveBaselineV1
- Model B: 384-d GRU recurrent
- Model C: 32 x 12-d thoughtlets = 384-d total, shared-weight BrainCell + attention

Input format: [B, N_FRAMES, 3, 16, 16] normalized to [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

N_FRAMES = 4
ACTION_CLASSES = 5
UPSAMPLE = 32
FEATURE_DIM = 48 * 4 * 4  # 768 — output of conv encoder before action head


# --- Shared conv encoder (matches ReactiveBaselineV1) ---

class ConvEncoder(nn.Module):
    """Shared conv encoder: [B, 12, 32, 32] → [B, 768]."""

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
        """x: [B, 12, 32, 32] → [B, 768]"""
        h = self.pool(F.relu(self.bn1(self.conv1(x))))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        return h.reshape(h.size(0), -1)


def prepare_input(frames: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Prepare encoder input from corpus format.

    frames: [B, N_FRAMES, 3, 16, 16] normalized to [0,1]
    prev_action: [B] int64
    Returns: [B, 12, 32, 32] (concatenated frames upsampled to 32x32)
    """
    B = frames.size(0)
    x = frames.reshape(B, N_FRAMES * 3, 16, 16)
    if x.size(-2) != UPSAMPLE:
        x = F.interpolate(x, size=(UPSAMPLE, UPSAMPLE), mode="nearest")
    return x


def _prev_onehot(prev_action: torch.Tensor, n_classes: int = ACTION_CLASSES) -> torch.Tensor:
    """One-hot encoding that works on DML, CUDA, and CPU.

    F.one_hot uses scatter which is not supported on DirectML privateuseone.
    Fallback to CPU then move back for DML devices.
    """
    dev = prev_action.device
    if dev.type == "privateuseone":
        # DML: do via CPU then move — tiny tensor, negligible cost
        return F.one_hot(prev_action.cpu().long(), n_classes).float().to(dev)
    return F.one_hot(prev_action.long(), n_classes).float()


# --- Model A: Reactive baseline (no recurrence) ---

class ReactiveModel(nn.Module):
    """Observation → encoder → action. No recurrence. Matches ReactiveBaselineV1."""

    def __init__(self, n_actions: int = ACTION_CLASSES):
        super().__init__()
        self.encoder = ConvEncoder()
        self.fc1 = nn.Linear(FEATURE_DIM + n_actions, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(
        self,
        frames: torch.Tensor,          # [B, N_FRAMES, 3, 16, 16]
        prev_action: torch.Tensor,     # [B] int64
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)  # [B, 768]

        prev_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, prev_onehot], dim=-1)  # [B, 773]
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits, None


class ManualGRUCell(nn.Module):
    """Manual GRUCell using only Linear + pointwise ops (DML-compatible).

    Mathematically identical to nn.GRUCell but avoids the fused _thnn_fused_gru_cell
    kernel which is not implemented for DML.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Update gate
        self.weight_ih_r = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.weight_hh_r = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.bias_ih_r = nn.Parameter(torch.zeros(hidden_size))
        self.bias_hh_r = nn.Parameter(torch.zeros(hidden_size))
        # Reset gate uses same weights split — simpler: use combined Linear like BrainCell
        # For exact GRUCell compat, use 3*hidden linear
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        return (1 - z) * n + z * h


# --- Model B: GRU recurrent ---

class GRUModel(nn.Module):
    """Observation → encoder → GRU → action. 384-d recurrent state."""

    def __init__(self, n_actions: int = ACTION_CLASSES, hidden_size: int = 384):
        super().__init__()
        self.encoder = ConvEncoder()
        self.hidden_size = hidden_size
        self.gru = ManualGRUCell(input_size=FEATURE_DIM + n_actions, hidden_size=hidden_size)
        self.action_head = nn.Linear(hidden_size, n_actions)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,                              # [B, N_FRAMES, 3, 16, 16]
        prev_action: torch.Tensor,                         # [B] int64
        h: Optional[torch.Tensor] = None,                  # [B, 384]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)  # [B, 768]

        prev_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, prev_onehot], dim=-1)  # [B, 773]

        if h is None:
            h = self.init_hidden(B, frames.device)

        h = self.gru(x, h)
        logits = self.action_head(h)
        return logits, h


# --- BrainCell: shared-weight thoughtlet update ---

class BrainCell(nn.Module):
    """Shared-weight GRU-like update for a single thoughtlet.

    Weights are shared across all K thoughtlets and all C cycles.
    """

    def __init__(self, input_size: int, thought_size: int):
        super().__init__()
        self.W_ir = nn.Linear(input_size, thought_size)
        self.W_hr = nn.Linear(thought_size, thought_size)
        self.W_iz = nn.Linear(input_size, thought_size)
        self.W_hz = nn.Linear(thought_size, thought_size)
        self.W_in = nn.Linear(input_size, thought_size)
        self.W_hn = nn.Linear(thought_size, thought_size)

    def forward(self, thought: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """thought: [B, W], x: [B, input_size] → [B, W]"""
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(thought))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(thought))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(thought))
        return (1 - z) * n + z * thought


# --- Model C: Thoughtlet recurrent ---

class ThoughtletModel(nn.Module):
    """Observation → encoder → K thoughtlets (shared BrainCell + attention) → aggregate → action.

    Total recurrent state: K * thought_size = 32 * 12 = 384 dimensions.
    """

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
        # PPO critic head (zero-init: BC checkpoints load with strict=False,
        # policy behavior bit-identical until PPO trains the head).
        self.value_head = nn.Linear(total_state, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def init_thoughts(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.K, self.thought_size, device=device)

    def forward(
        self,
        frames: torch.Tensor,                              # [B, N_FRAMES, 3, 16, 16]
        prev_action: torch.Tensor,                         # [B] int64
        h: Optional[torch.Tensor] = None,                  # [B, K, thought_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)  # [B, 768]

        prev_onehot = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, prev_onehot], dim=-1)  # [B, 773]

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

        aggregated = thoughts.reshape(B, -1)  # [B, 384]
        logits = self.action_head(aggregated)
        return logits, thoughts

    def actor_critic(self, frames, prev_action, h=None):
        """PPO rollout/update entry: (logits, h_next, value). Shares one
        encoder pass; value_head is zero-init so BC behavior is unchanged."""
        logits, thoughts = self.forward(frames, prev_action, h)
        value = self.value_head(thoughts.reshape(frames.size(0), -1)).squeeze(-1)
        return logits, thoughts, value


# --- Ablation variants (same I/O contract: (frames, prev_action, h) -> (logits, h)) ---
# Registered in the factory so ablations train under the IDENTICAL recipe
# (sequence windows, scheduled sampling, label smoothing) as base models.

INPUT_SIZE = FEATURE_DIM + ACTION_CLASSES  # 773


class ThoughtletNoPersistence(nn.Module):
    """Thoughtlets re-initialized to zeros every frame (no persistence)."""

    def __init__(self, K=32, thought_size=12, n_actions=ACTION_CLASSES, cycles=1):
        super().__init__()
        self.encoder = ConvEncoder()
        self.K = K
        self.thought_size = thought_size
        self.cycles = cycles
        self.brain_cell = BrainCell(input_size=INPUT_SIZE, thought_size=thought_size)
        self.attention = nn.MultiheadAttention(embed_dim=thought_size, num_heads=4, batch_first=True)
        self.attn_proj = nn.Linear(thought_size, thought_size)
        self.action_head = nn.Linear(K * thought_size, n_actions)

    def forward(self, frames, prev_action, h=None):
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)
        pa = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, pa], dim=-1)

        thoughts = torch.zeros(B, self.K, self.thought_size, device=frames.device)
        for _ in range(self.cycles):
            x_exp = x.unsqueeze(1).expand(-1, self.K, -1).reshape(B * self.K, -1)
            t_flat = thoughts.reshape(B * self.K, self.thought_size)
            new_t = self.brain_cell(t_flat, x_exp).reshape(B, self.K, self.thought_size)
            attn_out, _ = self.attention(new_t, new_t, new_t)
            thoughts = new_t + self.attn_proj(attn_out)

        return self.action_head(thoughts.reshape(B, -1)), thoughts


class ThoughtletNoAttention(nn.Module):
    """Thoughtlets with no inter-slot communication (isolated slots)."""

    def __init__(self, K=32, thought_size=12, n_actions=ACTION_CLASSES, cycles=1):
        super().__init__()
        self.encoder = ConvEncoder()
        self.K = K
        self.thought_size = thought_size
        self.cycles = cycles
        self.brain_cell = BrainCell(input_size=INPUT_SIZE, thought_size=thought_size)
        self.action_head = nn.Linear(K * thought_size, n_actions)

    def forward(self, frames, prev_action, h=None):
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)
        pa = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, pa], dim=-1)

        thoughts = h if h is not None else torch.zeros(B, self.K, self.thought_size, device=frames.device)
        for _ in range(self.cycles):
            x_exp = x.unsqueeze(1).expand(-1, self.K, -1).reshape(B * self.K, -1)
            t_flat = thoughts.reshape(B * self.K, self.thought_size)
            thoughts = self.brain_cell(t_flat, x_exp).reshape(B, self.K, self.thought_size)

        return self.action_head(thoughts.reshape(B, -1)), thoughts


class IndependentPerSlot(nn.Module):
    """Independent per-slot transitions (no weight sharing)."""

    def __init__(self, K=32, thought_size=12, n_actions=ACTION_CLASSES, cycles=1):
        super().__init__()
        self.encoder = ConvEncoder()
        self.K = K
        self.thought_size = thought_size
        self.cycles = cycles
        self.cells = nn.ModuleList([
            BrainCell(input_size=INPUT_SIZE, thought_size=thought_size)
            for _ in range(K)
        ])
        self.attention = nn.MultiheadAttention(embed_dim=thought_size, num_heads=4, batch_first=True)
        self.attn_proj = nn.Linear(thought_size, thought_size)
        self.action_head = nn.Linear(K * thought_size, n_actions)

    def forward(self, frames, prev_action, h=None):
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)
        pa = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, pa], dim=-1)

        thoughts = h if h is not None else torch.zeros(B, self.K, self.thought_size, device=frames.device)
        for _ in range(self.cycles):
            new_thoughts = []
            for k in range(self.K):
                t_k = thoughts[:, k, :]
                new_t_k = self.cells[k](t_k, x)
                new_thoughts.append(new_t_k)
            thoughts = torch.stack(new_thoughts, dim=1)
            attn_out, _ = self.attention(thoughts, thoughts, thoughts)
            thoughts = thoughts + self.attn_proj(attn_out)

        return self.action_head(thoughts.reshape(B, -1)), thoughts


class ThoughtletK1(nn.Module):
    """Single thoughtlet (K=1, thought_size=384)."""

    def __init__(self, n_actions=ACTION_CLASSES, cycles=1):
        super().__init__()
        self.encoder = ConvEncoder()
        self.K = 1
        self.thought_size = 384
        self.cycles = cycles
        self.brain_cell = BrainCell(input_size=INPUT_SIZE, thought_size=384)
        self.action_head = nn.Linear(384, n_actions)

    def forward(self, frames, prev_action, h=None):
        B = frames.size(0)
        x = prepare_input(frames, prev_action)
        features = self.encoder(x)
        pa = _prev_onehot(prev_action, ACTION_CLASSES)
        x = torch.cat([features, pa], dim=-1)

        thoughts = h if h is not None else torch.zeros(B, 1, 384, device=frames.device)
        for _ in range(self.cycles):
            t_flat = thoughts.reshape(B * 1, 384)
            thoughts = self.brain_cell(t_flat, x).reshape(B, 1, 384)

        return self.action_head(thoughts.reshape(B, -1)), thoughts


# K-sweep table: total recurrent state held at 384-d
K_SWEEP = {
    "k1": {"K": 1, "thought_size": 384},
    "k2": {"K": 2, "thought_size": 192},
    "k4": {"K": 4, "thought_size": 96},
    "k8": {"K": 8, "thought_size": 48},
    "k16": {"K": 16, "thought_size": 24},
    "k32": {"K": 32, "thought_size": 12},
}


# --- Factory ---

def make_model(model_type: str, **kwargs) -> nn.Module:
    if model_type == "reactive":
        return ReactiveModel(**kwargs)
    elif model_type == "gru":
        return GRUModel(**kwargs)
    elif model_type == "thoughtlet":
        return ThoughtletModel(**kwargs)
    elif model_type == "no_persistence":
        return ThoughtletNoPersistence(**kwargs)
    elif model_type == "no_attention":
        return ThoughtletNoAttention(**kwargs)
    elif model_type == "independent_slots":
        return IndependentPerSlot(**kwargs)
    elif model_type in K_SWEEP:
        cfg = dict(K_SWEEP[model_type])
        cfg.update(kwargs)
        if model_type == "k1":
            return ThoughtletK1(**{k: v for k, v in cfg.items() if k in ("cycles", "n_actions")})
        # K-sweep uses the FULL mechanism (shared cell + attention), K=1 trivially unattended
        return ThoughtletModel(n_actions=cfg.get("n_actions", ACTION_CLASSES),
                               K=cfg["K"], thought_size=cfg["thought_size"],
                               cycles=cfg.get("cycles", 1))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_component = {}
    for name, param in model.named_parameters():
        component = name.split(".")[0]
        by_component[component] = by_component.get(component, 0) + param.numel()
    return {"total": total, "trainable": trainable, "by_component": by_component}


if __name__ == "__main__":
    for mtype in ["reactive", "gru", "thoughtlet"]:
        m = make_model(mtype)
        info = count_parameters(m)
        print(f"{mtype:12s}  total={info['total']:>7d}  trainable={info['trainable']:>7d}  {info['by_component']}")
