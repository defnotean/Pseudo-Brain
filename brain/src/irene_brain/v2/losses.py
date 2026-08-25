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
from .reward_distribution import SymlogTwoHotReward


def symlog(value: torch.Tensor) -> torch.Tensor:
    """Bi-symmetric logarithm that preserves sign and near-zero resolution."""
    return torch.sign(value) * torch.log1p(value.abs())


def deployed_decision_loss(
    step: StepOutput,
    target_action: torch.Tensor,      # [B] long
    class_weights: Optional[torch.Tensor] = None,
    weight_normalization: str = "batch_weight_sum_v0",
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy on the DEPLOYED action_dist — the actual decision path."""
    log_probs = F.log_softmax(step.decision.action_values, dim=-1)
    if weight_normalization == "batch_weight_sum_v0":
        if sample_weights is not None:
            per_sample = F.nll_loss(log_probs, target_action, reduction="none")
            effective = sample_weights
            if class_weights is not None:
                effective = effective * class_weights[target_action]
            return (per_sample * effective).sum() / effective.sum().clamp_min(1e-12)
        return F.nll_loss(log_probs, target_action, weight=class_weights)
    if weight_normalization == "fixed_exposure_mean_v1":
        per_sample = F.nll_loss(log_probs, target_action, reduction="none")
        effective = torch.ones_like(per_sample)
        if class_weights is not None:
            effective = effective * class_weights[target_action]
        if sample_weights is not None:
            effective = effective * sample_weights
        return (per_sample * effective).mean()
    raise ValueError(f"unknown deployed-loss weight normalization: {weight_normalization}")


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
    hazard_parameterization: str = "legacy_unbounded_v0",
    latent_comparison: str = "raw_mse_v0",
    reward_comparison: str = "raw_mse_v0",
    outcome_architecture: str = "legacy_hypothesis_world_v0",
    reward_prediction: str = "scalar_v0",
    reward_bins: int = 65,
    reward_symlog_min: float = -4.0,
    reward_symlog_max: float = 4.0,
    all_action_reward: Optional[torch.Tensor] = None,  # [B, A]
    all_action_hazard: Optional[torch.Tensor] = None,  # [B, A, 1]
    all_action_next_latent: Optional[torch.Tensor] = None,  # [B, A, W]
) -> dict:
    """Supervise consequence heads where ground truth exists."""
    losses = {}
    h = step.hypotheses
    factual = step.pending_prediction
    table = step.outcome_table
    use_all_actions = any(
        target is not None
        for target in (
            all_action_reward,
            all_action_hazard,
            all_action_next_latent,
        )
    )
    if use_all_actions:
        if outcome_architecture != "all_action_table_v1" or table is None:
            raise ValueError(
                "all-action targets require an all_action_table_v1 output"
            )
        if not all(
            target is not None
            for target in (
                all_action_reward,
                all_action_hazard,
                all_action_next_latent,
            )
        ):
            raise ValueError("all-action reward, hazard, and next latent are atomic")

    if actual_reward is not None:
        predicted_reward = table.predicted_reward if use_all_actions else (
            factual.predicted_reward
            if outcome_architecture == "all_action_table_v1" and factual is not None
            else h.predicted_reward
        )
        reward_target = (
            all_action_reward.unsqueeze(-1) if use_all_actions else actual_reward
        )
        if predicted_reward.dim() == 3:
            if not use_all_actions:
                predicted_reward = predicted_reward.mean(dim=1)
        if reward_prediction == "symlog_twohot_v1":
            reward_logits = (
                table.predicted_reward_logits
                if use_all_actions
                else None if factual is None else factual.predicted_reward_logits
            )
            if reward_logits is None:
                raise ValueError("two-hot reward prediction requires reward logits")
            codec = SymlogTwoHotReward(
                num_bins=reward_bins,
                symlog_min=reward_symlog_min,
                symlog_max=reward_symlog_max,
            ).to(device=reward_logits.device)
            losses["reward"] = codec.loss(
                reward_logits,
                all_action_reward if use_all_actions else actual_reward.squeeze(-1),
            )
        elif reward_comparison == "raw_mse_v0":
            losses["reward"] = F.mse_loss(predicted_reward, reward_target)
        elif reward_comparison == "symlog_mse_v1":
            losses["reward"] = F.mse_loss(
                symlog(predicted_reward),
                symlog(reward_target),
            )
        else:
            raise ValueError(f"unknown reward comparison: {reward_comparison}")
    if actual_hazard is not None:
        predicted_hazard = table.predicted_hazard if use_all_actions else (
            factual.predicted_hazard
            if outcome_architecture == "all_action_table_v1" and factual is not None
            else h.predicted_hazard
        )
        hazard_target = all_action_hazard if use_all_actions else actual_hazard
        if predicted_hazard.dim() == 3:
            if not use_all_actions:
                predicted_hazard = predicted_hazard.mean(dim=1)
        if hazard_parameterization == "probability_sigmoid_v1":
            losses["hazard"] = F.binary_cross_entropy(
                predicted_hazard,
                hazard_target,
            )
        elif hazard_parameterization == "legacy_unbounded_v0":
            losses["hazard"] = F.mse_loss(predicted_hazard, hazard_target)
        else:
            raise ValueError(f"unknown hazard parameterization: {hazard_parameterization}")
    if next_latent is not None and step.pending_prediction is not None:
        pred = table.predicted_next_latent if use_all_actions else (
            step.pending_prediction.predicted_next_latent
        )
        if pred.dim() == 3:
            if not use_all_actions:
                pred = pred.mean(dim=1)
        target = (
            all_action_next_latent if use_all_actions else next_latent
        ).detach()
        if latent_comparison == "raw_mse_v0":
            losses["next_latent"] = F.mse_loss(pred, target)
        elif latent_comparison == "cosine_distance_v1":
            losses["next_latent"] = 1.0 - F.cosine_similarity(
                pred,
                target,
                dim=-1,
            ).mean()
        else:
            raise ValueError(f"unknown latent comparison: {latent_comparison}")

    return losses


def total_v2_loss(
    step: StepOutput,
    config: CoreV2Config,
    flags: FeatureFlags,
    target_action: torch.Tensor,
    actual_reward: Optional[torch.Tensor] = None,
    actual_hazard: Optional[torch.Tensor] = None,
    next_latent: Optional[torch.Tensor] = None,
    decision_class_weights: Optional[torch.Tensor] = None,
    decision_weight_normalization: str = "batch_weight_sum_v0",
    decision_sample_weights: Optional[torch.Tensor] = None,
    all_action_reward: Optional[torch.Tensor] = None,
    all_action_hazard: Optional[torch.Tensor] = None,
    all_action_next_latent: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Weighted sum of all active losses. Returns (total, component_dict)."""
    components = {}
    total = torch.tensor(0.0, device=target_action.device)

    # Deployed decision loss — THE loss
    components["decision"] = deployed_decision_loss(
        step,
        target_action,
        class_weights=decision_class_weights,
        weight_normalization=decision_weight_normalization,
        sample_weights=decision_sample_weights,
    )
    total = total + config.decision_weight * components["decision"]

    if flags.consequence_learning:
        aux = consequence_losses(
            step,
            actual_reward,
            actual_hazard,
            next_latent,
            hazard_parameterization=config.hazard_parameterization,
            latent_comparison=config.latent_comparison,
            reward_comparison=config.reward_comparison,
            outcome_architecture=config.outcome_architecture,
            reward_prediction=config.reward_prediction,
            reward_bins=config.reward_bins,
            reward_symlog_min=config.reward_symlog_min,
            reward_symlog_max=config.reward_symlog_max,
            all_action_reward=all_action_reward,
            all_action_hazard=all_action_hazard,
            all_action_next_latent=all_action_next_latent,
        )
        auxiliary_weights = {
            "reward": config.reward_weight,
            "hazard": config.hazard_weight,
            "next_latent": config.next_weight,
        }
        for name, val in aux.items():
            components[name] = val
            total = total + auxiliary_weights[name] * val

    return total, components
