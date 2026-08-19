"""Auxiliary Supervised Objectives for Adaptive Thought Updates, Future Foresight, Dynamic Halting, and Counterfactual Foresight.

This module provides explicit loss functions to train:
1. Multi-horizon spatial foresight (predicting future agent displacement, ghost proximity, and escape margin).
2. Adaptive surprise-gated thought updates (forcing alpha -> 0.8+ on unexpected collisions and hazard proximity).
3. Context-aware cognitive halting (allocating thinking cycles with an explicit compute penalty).
4. Action-Conditioned Counterfactual Foresight (evaluating counterfactual branch outcomes conditioned on candidate actions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None
    nn = object  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class CognitiveLossOutput:
    """Breakdown of cognitive auxiliary losses."""

    total_loss: Tensor
    future_displacement_loss: Tensor
    future_ghost_loss: Tensor
    future_escape_margin_loss: Tensor
    adaptive_gate_loss: Tensor
    halting_loss: Tensor
    counterfactual_loss: Tensor = None


class CognitiveAuxiliaryLoss(nn.Module):
    """Computes supervised losses for internal thought adaptation, foresight, and counterfactual reasoning."""

    def __init__(
        self,
        *,
        future_weight: float = 1.0,
        gate_surprise_weight: float = 1.0,
        halting_weight: float = 0.5,
        counterfactual_weight: float = 1.0,
        compute_cost_per_cycle: float = 0.01,
    ) -> None:
        super().__init__()
        self.future_weight = future_weight
        self.gate_surprise_weight = gate_surprise_weight
        self.halting_weight = halting_weight
        self.counterfactual_weight = counterfactual_weight
        self.compute_cost_per_cycle = compute_cost_per_cycle

    def forward(
        self,
        *,
        diagnostics: Any,
        future_disp_targets: Tensor | None = None,     # [B, 3, 2]
        future_ghost_targets: Tensor | None = None,    # [B, 3, 1]
        future_escape_targets: Tensor | None = None,   # [B, 3, 1]
        hazard_mask: Tensor | None = None,             # [B] bool (True if wall bump or ghost near)
        complexity_level: Tensor | None = None,        # [B] in [0, 1] (0 = easy corridor, 1 = danger/dead-end)
        executed_actions: Tensor | None = None,        # [B] long action indices
        actual_collisions: Tensor | None = None,       # [B, 3, 1] float (1 if collision/catch at t+k, 0 otherwise)
    ) -> CognitiveLossOutput:
        """Compute the combined cognitive auxiliary loss.

        All arguments are strictly derived from ground-truth transition labels
        and evaluated strictly during loss backpropagation.
        """
        device = (
            diagnostics.thought_summaries.device
            if hasattr(diagnostics, "thought_summaries") and diagnostics.thought_summaries is not None
            else torch.device("cpu")
        )
        zero = torch.tensor(0.0, device=device)

        # 1. Future Foresight Loss
        future_disp_loss = zero
        future_ghost_loss = zero
        future_escape_loss = zero

        future_preds = getattr(diagnostics, "future_trajectory_predictions", None)
        if future_preds is not None and isinstance(future_preds, dict):
            if future_disp_targets is not None and "predicted_displacement" in future_preds:
                pred_disp = future_preds["predicted_displacement"]
                future_disp_loss = F.smooth_l1_loss(pred_disp, future_disp_targets)

            if future_ghost_targets is not None and "predicted_ghost_proximity" in future_preds:
                pred_ghost = future_preds["predicted_ghost_proximity"]
                future_ghost_loss = F.smooth_l1_loss(pred_ghost, future_ghost_targets)

            if future_escape_targets is not None and "predicted_escape_margin" in future_preds:
                pred_escape = future_preds["predicted_escape_margin"]
                future_escape_loss = F.smooth_l1_loss(pred_escape, future_escape_targets)

        # 2. Adaptive Gate Surprise Loss
        gate_loss = zero
        gates = getattr(diagnostics, "thought_update_gates", None)
        if gates is not None and hazard_mask is not None:
            # Target alpha: 0.05 for normal movement, 0.90 for hazard/wall bump
            target_alpha = torch.where(
                hazard_mask.unsqueeze(-1),
                torch.tensor(0.90, device=device),
                torch.tensor(0.05, device=device),
            ).expand_as(gates)
            # Rebalance hazard steps so rare collision/ghost events provide sufficient gradient mass
            step_weight = torch.where(
                hazard_mask.unsqueeze(-1),
                torch.tensor(10.0, device=device),
                torch.tensor(1.0, device=device),
            ).expand_as(gates)
            gate_loss = (step_weight * (gates - target_alpha).square()).mean()

        # 3. Adaptive Halting Loss with Compute Penalty
        halt_loss = zero
        halt_probs = getattr(diagnostics, "halting_probabilities", ())
        if halt_probs and complexity_level is not None:
            # In easy corridors (complexity ~ 0): target high halting probability at cycle 1 (exit early)
            # In complex/danger states (complexity ~ 1): target low halting probability early (exit later)
            cycle_losses = []
            for cycle_idx, halt_p in enumerate(halt_probs):
                if cycle_idx == 0:
                    # Early cycle: Halt if easy, continue if complex
                    target_halt = (1.0 - complexity_level).unsqueeze(-1)
                    cycle_losses.append(F.binary_cross_entropy(halt_p, target_halt))
                else:
                    # Later cycles: Add compute penalty encouraging closure
                    compute_penalty = halt_p.mean() * (self.compute_cost_per_cycle * (cycle_idx + 1))
                    cycle_losses.append(compute_penalty)
            if cycle_losses:
                halt_loss = torch.stack(cycle_losses).mean()

        # 4. Action-Conditioned Counterfactual Loss
        cf_loss = zero
        cf_preds = getattr(diagnostics, "counterfactual_predictions", None)
        if cf_preds is not None and isinstance(cf_preds, dict) and executed_actions is not None:
            cf_branch_losses = []
            B = executed_actions.shape[0]
            for i in range(B):
                act_idx = int(executed_actions[i].item())
                if act_idx in cf_preds:
                    branch_out = cf_preds[act_idx]
                    if future_disp_targets is not None:
                        disp_l = F.smooth_l1_loss(branch_out.predicted_displacement[i:i+1], future_disp_targets[i:i+1])
                        cf_branch_losses.append(disp_l)
                    if actual_collisions is not None:
                        haz_l = F.binary_cross_entropy(branch_out.hazard_probability[i:i+1], actual_collisions[i:i+1])
                        cf_branch_losses.append(haz_l)
                    if future_escape_targets is not None:
                        esc_l = F.smooth_l1_loss(branch_out.predicted_escape_margin[i:i+1], future_escape_targets[i:i+1])
                        cf_branch_losses.append(esc_l)
            if cf_branch_losses:
                cf_loss = torch.stack(cf_branch_losses).mean()

        total = (
            self.future_weight * (future_disp_loss + future_ghost_loss + future_escape_loss)
            + self.gate_surprise_weight * gate_loss
            + self.halting_weight * halt_loss
            + self.counterfactual_weight * cf_loss
        )

        return CognitiveLossOutput(
            total_loss=total,
            future_displacement_loss=future_disp_loss,
            future_ghost_loss=future_ghost_loss,
            future_escape_margin_loss=future_escape_loss,
            adaptive_gate_loss=gate_loss,
            halting_loss=halt_loss,
            counterfactual_loss=cf_loss,
        )


__all__ = [
    "CognitiveAuxiliaryLoss",
    "CognitiveLossOutput",
]
