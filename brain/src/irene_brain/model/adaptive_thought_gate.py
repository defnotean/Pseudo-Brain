"""Adaptive Thought-Update Gate, Future Trajectory Prediction, and Dynamic Halting Controller for Pseudo-Brain.

Core Mechanisms:
1. Adaptive "Change My Mind" Thought-Update Gate:
   - Evaluates incoming sensory/belief mismatch against current persistent thought representations.
   - If sensory prediction matches reality: maintains high temporal inertia (persistence ~0.99).
   - If unexpected collision or ghost hazard occurs: surges update gate alpha -> 0.8+, immediately rewriting obsolete thoughts.

2. Multi-Horizon Future Trajectory & Escape Margin Prediction:
   - Predicts multi-step future player displacement (dx, dy), ghost proximity, and escape margin (t_ghost_exit - t_player_exit) across horizons k in {1, 3, 5}.

3. Adaptive Cognitive Halting Controller:
   - Evaluates situation complexity to dynamically determine how many internal thought cycles C in [1, C_max] to execute per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ModuleNotFoundError:
    torch = None
    nn = object  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]


class AdaptiveThoughtUpdateGate(nn.Module):
    """Computes dynamic, surprise-gated thought renewal conditioned on sensory discrepancies."""

    def __init__(self, *, width: int, initial_bias: float = -2.0) -> None:
        super().__init__()
        self.width = width
        # Project pooled sensory-belief context to match thought width
        self.context_proj = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        # Gate network: inputs are [thought_summary, context_summary, discrepancy, danger_signal]
        self.gate_mlp = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        # Initialize bias so baseline update probability is small (~0.1) under quiet frames
        with torch.no_grad():
            if hasattr(self.gate_mlp[-1], "bias") and self.gate_mlp[-1].bias is not None:
                self.gate_mlp[-1].bias.fill_(initial_bias)

    def forward(
        self,
        *,
        thoughts: Tensor,  # [B, T, R, W]
        seeds: Tensor,     # [B, T, R, W]
        sensors: Tensor,   # [B, S, W]
        belief: Tensor,    # [B, Bel, W]
    ) -> tuple[Tensor, Tensor]:
        """Blend old thoughts and freshly seeded context using an adaptive surprise gate.

        Returns:
            refreshed_thoughts: [B, T, R, W]
            update_gates: [B, T] in [0, 1]
        """
        B, T, R, W = thoughts.shape
        thought_summary = thoughts.mean(dim=2)  # [B, T, W]
        
        # Aggregate sensory and belief evidence
        combined_context = torch.cat((sensors, belief), dim=1)  # [B, S+Bel, W]
        context_pooled = self.context_proj(combined_context.mean(dim=1))  # [B, W]
        context_expanded = context_pooled.unsqueeze(1).expand(B, T, W)  # [B, T, W]

        # Explicit discrepancy / prediction error feature
        discrepancy = torch.abs(thought_summary - context_expanded)  # [B, T, W]

        # Combine into gate input: [B, T, 3*W]
        gate_input = torch.cat((thought_summary, context_expanded, discrepancy), dim=-1)
        update_logits = self.gate_mlp(gate_input).squeeze(-1)  # [B, T]
        alpha = torch.sigmoid(update_logits)  # [B, T] in [0, 1]

        # Apply update gate across all register dimensions
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        refreshed_thoughts = (1.0 - alpha_expanded) * thoughts + alpha_expanded * seeds

        return refreshed_thoughts, alpha


class PredictiveFutureTrajectoryHead(nn.Module):
    """Predicts multi-horizon future states, ghost proximity, and escape margin from thought representations."""

    def __init__(self, *, width: int, horizons: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.width = width
        self.horizons = horizons
        self.num_horizons = len(horizons)

        self.shared_trunk = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        # Displacement prediction: (dx, dy) per horizon -> [B, num_horizons, 2]
        self.displacement_head = nn.Linear(width, self.num_horizons * 2)
        # Ghost proximity prediction: estimated Euclidean distance to nearest ghost -> [B, num_horizons, 1]
        self.ghost_proximity_head = nn.Linear(width, self.num_horizons)
        # Escape margin prediction: (t_ghost_reach - t_player_reach) to junction -> [B, num_horizons, 1]
        self.escape_margin_head = nn.Linear(width, self.num_horizons)

    def forward(self, thoughts: Tensor) -> dict[str, Tensor]:
        """Predict future geometry from unpooled thoughtlet representations.

        Args:
            thoughts: [B, T, R, W] or pooled [B, W]

        Returns:
            Dict containing predicted displacements, ghost proximity, and escape margins.
        """
        if thoughts.ndim == 4:
            thought_repr = thoughts.mean(dim=(1, 2))  # [B, W]
        elif thoughts.ndim == 3:
            thought_repr = thoughts.mean(dim=1)  # [B, W]
        else:
            thought_repr = thoughts

        B = thought_repr.shape[0]
        feat = self.shared_trunk(thought_repr)

        disp = self.displacement_head(feat).view(B, self.num_horizons, 2)
        ghost_prox = torch.relu(self.ghost_proximity_head(feat).view(B, self.num_horizons, 1))
        escape_marg = self.escape_margin_head(feat).view(B, self.num_horizons, 1)

        return {
            "predicted_displacement": disp,
            "predicted_ghost_proximity": ghost_prox,
            "predicted_escape_margin": escape_marg,
        }


class AdaptiveHaltingController(nn.Module):
    """Dynamic cognitive cycle halting controller to allocate computation adaptively."""

    def __init__(self, *, width: int, max_cycles: int = 4, threshold: float = 0.65) -> None:
        super().__init__()
        self.width = width
        self.max_cycles = max_cycles
        self.threshold = threshold

        self.halt_mlp = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, thoughts: Tensor, cycle_index: int) -> tuple[Tensor, bool]:
        """Evaluate whether to halt thinking after the current cognitive cycle.

        Args:
            thoughts: [B, T, R, W]
            cycle_index: 0-indexed current cycle.

        Returns:
            halt_probability: [B, 1]
            should_halt: bool (True if all batch elements exceed threshold or cycle is max)
        """
        thought_summary = thoughts.mean(dim=(1, 2))  # [B, W]
        logit = self.halt_mlp(thought_summary)
        halt_prob = torch.sigmoid(logit)

        if cycle_index + 1 >= self.max_cycles:
            return halt_prob, True

        # Halt if average batch confidence exceeds threshold
        mean_halt_conf = float(halt_prob.mean().item())
        should_halt = mean_halt_conf >= self.threshold
        return halt_prob, should_halt
