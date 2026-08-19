"""Action-Conditioned Counterfactual Foresight for Pseudo-Brain.

Thesis:
Rather than predicting a single passive trajectory ("what will happen in general"),
persistent parallel thoughtlets maintain and compare several small alternative futures:
"If I choose action A -> what happens at t+1, t+3, t+5?"
"If I choose action B -> what happens at t+1, t+3, t+5?"
"If I choose action C -> what happens at t+1, t+3, t+5?"

Predictions per candidate action a in {0=None, 1=Up, 2=Left, 3=Down, 4=Right}:
1. Counterfactual Displacement (dx_k(a), dy_k(a))
2. Counterfactual Ghost Proximity / Hazard Probability c_k(a)
3. Counterfactual Escape Margin E_k(a) = d(ghost, exit) - d(player, exit)
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


@dataclass(frozen=True, slots=True)
class CounterfactualBranchOutput:
    """Predicted future outcomes for a specific candidate action."""
    action_index: int
    predicted_displacement: Tensor  # [B, num_horizons, 2]
    hazard_probability: Tensor      # [B, num_horizons, 1] in [0, 1]
    predicted_escape_margin: Tensor # [B, num_horizons, 1]


class ActionConditionedCounterfactualForesightHead(nn.Module):
    """Evaluates multi-horizon counterfactual futures conditioned on candidate actions."""

    def __init__(
        self,
        *,
        width: int,
        num_actions: int = 5,
        horizons: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.width = width
        self.num_actions = num_actions
        self.horizons = horizons
        self.num_horizons = len(horizons)

        # Action embedding layer mapping discrete action index -> latent action token
        self.action_embedding = nn.Embedding(num_actions, width // 2)

        # Counterfactual reasoning trunk: fuses thought summary + candidate action embedding
        self.fusion_trunk = nn.Sequential(
            nn.Linear(width + (width // 2), width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )

        # 1. Action-conditioned displacement head: (dx, dy) per horizon
        self.displacement_head = nn.Linear(width, self.num_horizons * 2)

        # 2. Action-conditioned hazard / collision classifier: sigmoid logit per horizon
        self.hazard_head = nn.Linear(width, self.num_horizons)

        # 3. Action-conditioned escape margin head: continuous escape margin per horizon
        self.escape_margin_head = nn.Linear(width, self.num_horizons)

    def forward_single_action(
        self,
        thought_summary: Tensor,  # [B, W]
        action_indices: Tensor,    # [B] or scalar int
    ) -> CounterfactualBranchOutput:
        """Evaluate counterfactual outcome for a specific candidate action."""
        B = thought_summary.shape[0]
        if isinstance(action_indices, int):
            action_indices = torch.full((B,), action_indices, dtype=torch.long, device=thought_summary.device)
        elif action_indices.ndim == 0:
            action_indices = action_indices.expand(B)

        act_emb = self.action_embedding(action_indices)  # [B, W // 2]
        fused = torch.cat((thought_summary, act_emb), dim=-1)  # [B, W + W//2]
        feat = self.fusion_trunk(fused)  # [B, W]

        disp = self.displacement_head(feat).view(B, self.num_horizons, 2)
        hazard_logit = self.hazard_head(feat).view(B, self.num_horizons, 1)
        hazard_prob = torch.sigmoid(hazard_logit)
        escape_marg = self.escape_margin_head(feat).view(B, self.num_horizons, 1)

        action_val = int(action_indices[0].item()) if B > 0 else 0
        return CounterfactualBranchOutput(
            action_index=action_val,
            predicted_displacement=disp,
            hazard_probability=hazard_prob,
            predicted_escape_margin=escape_marg,
        )

    def forward_all_actions(
        self,
        thoughts: Tensor,  # [B, T, R, W] or [B, W]
    ) -> dict[int, CounterfactualBranchOutput]:
        """Evaluate all candidate actions simultaneously via batched tensor parallelism."""
        if thoughts.ndim == 4:
            thought_summary = thoughts.mean(dim=(1, 2))  # [B, W]
        elif thoughts.ndim == 3:
            thought_summary = thoughts.mean(dim=1)  # [B, W]
        else:
            thought_summary = thoughts

        B = thought_summary.shape[0]
        results = {}
        for a in range(self.num_actions):
            act_tensor = torch.full((B,), a, dtype=torch.long, device=thought_summary.device)
            results[a] = self.forward_single_action(thought_summary, act_tensor)
        return results


__all__ = [
    "ActionConditionedCounterfactualForesightHead",
    "CounterfactualBranchOutput",
]
