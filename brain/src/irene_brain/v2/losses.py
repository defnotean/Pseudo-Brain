"""V2 training losses — deploy-aligned decision loss + auxiliary heads.

The core lesson of Phase 2.7: train the DEPLOYED output (action_dist from
the aggregator), not a proxy (per-slot action_logits). Auxiliary heads get
small weights; the decision path gets the big weight.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .config import CoreV2Config, FeatureFlags
from .core import CoreV2Model, StepOutput


def deployed_decision_loss(
    step: StepOutput,
    target_action: torch.Tensor,      # [B] long
) -> torch.Tensor:
    """Cross-entropy on the DEPLOYED action_dist — the actual decision path."""
    log_probs = F.log_softmax(step.decision.action_values, dim=-1)
    return F.nll_loss(log_probs, target_action)


def per_slot_decision_loss(
    step: StepOutput,
    target_action: torch.Tensor,      # [B] long
    replicate: bool = True,
) -> torch.Tensor:
    """Legacy V1-style loss on per-slot action_logits.

    Kept ONLY for A/B comparison in Stage V2.1. Not the deploy-aligned loss.
    """
    logits = step.hypotheses.action_logits  # [B, K, A]
    if replicate:
        tgt = target_action.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt)
    else:
        raise NotImplementedError("slot-specific targets not yet defined")


def consequence_losses(
    step: StepOutput,
    actual_reward: Optional[torch.Tensor] = None,   # [B, 1]
    actual_hazard: Optional[torch.Tensor] = None,   # [B, 1]
    next_latent: Optional[torch.Tensor] = None,     # [B, W]
) -> dict:
    """Supervise consequence heads where ground truth exists."""
    losses = {}
    h = step.hypotheses

    if actual_reward is not None:
        losses["reward"] = F.mse_loss(h.predicted_reward.mean(dim=1), actual_reward)
    if actual_hazard is not None:
        losses["hazard"] = F.mse_loss(h.predicted_hazard.mean(dim=1), actual_hazard)
    if next_latent is not None and step.pending_prediction is not None:
        pred = step.pending_prediction.predicted_next_latent
        if pred.dim() == 3:
            pred = pred.mean(dim=1)
        losses["next_latent"] = F.mse_loss(pred, next_latent.detach())

    return losses


def total_v2_loss(
    step: StepOutput,
    config: CoreV2Config,
    flags: FeatureFlags,
    target_action: torch.Tensor,
    actual_reward: Optional[torch.Tensor] = None,
    actual_hazard: Optional[torch.Tensor] = None,
    next_latent: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Weighted sum of all active losses. Returns (total, component_dict)."""
    components = {}
    total = torch.tensor(0.0, device=target_action.device)

    # Deployed decision loss — THE loss
    components["decision"] = deployed_decision_loss(step, target_action)
    total = total + config.decision_weight * components["decision"]

    if flags.consequence_learning:
        aux = consequence_losses(step, actual_reward, actual_hazard, next_latent)
        for name, val in aux.items():
            components[name] = val
            w = getattr(config, f"{name}_weight", 0.0)
            total = total + w * val

    return total, components