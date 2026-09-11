"""Vectorized one-step action-outcome prediction for Core V2.

This module deliberately contains no policy or decision logic.  It evaluates
one current state against an exhaustive, explicitly labelled action table and
can gather the one prediction associated with a factual applied action.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import CoreV2Config
from .hazard_calibration import PerActionAffineHazardCalibrator
from .reward_distribution import SymlogTwoHotReward
from .state import PendingPrediction


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class ActionOutcomeTable:
    """Predicted one-step outcomes for every action.

    ``action_ids`` labels the table columns explicitly.  A row may use any
    permutation of the action vocabulary, so consumers must select outcomes
    by action identity rather than assuming that a column number is an action.
    """

    action_ids: Tensor                 # [B, A], each row a permutation of 0..A-1
    predicted_next_latent: Tensor      # [B, A, W]
    predicted_reward: Tensor           # [B, A, 1], environment units
    predicted_reward_logits: Tensor | None  # [B, A, bins] in distributional mode
    predicted_hazard: Tensor           # [B, A, 1], probability in [0, 1]
    raw_hazard_logits: Tensor | None = None       # [B, A, 1], before calibration
    predicted_hazard_logits: Tensor | None = None  # [B, A, 1], after calibration

    def gather(self, action: Tensor) -> PendingPrediction:
        """Select the outcome for each factual action by semantic action ID."""
        if action.dim() != 1 or action.shape[0] != self.action_ids.shape[0]:
            raise ValueError("action must have shape [B]")
        if action.dtype not in _INTEGER_DTYPES:
            raise ValueError("action must use an integer dtype")
        action = action.to(device=self.action_ids.device, dtype=torch.long)
        matches = self.action_ids.eq(action.unsqueeze(1))
        if not bool(matches.any(dim=1).all()):
            raise ValueError("every factual action must occur in its outcome-table row")
        if not bool(matches.sum(dim=1).eq(1).all()):
            raise ValueError("every factual action must occur exactly once per row")
        column = matches.to(dtype=torch.long).argmax(dim=1)
        batch = torch.arange(self.action_ids.shape[0], device=self.action_ids.device)
        return PendingPrediction(
            predicted_next_latent=self.predicted_next_latent[batch, column],
            predicted_reward=self.predicted_reward[batch, column],
            predicted_reward_logits=(
                None
                if self.predicted_reward_logits is None
                else self.predicted_reward_logits[batch, column]
            ),
            predicted_hazard=self.predicted_hazard[batch, column],
        )


class VectorizedOutcomeModelV2(nn.Module):
    """Predict all discrete one-step action outcomes in one vectorized pass.

    The state trunk runs once per batch item.  Only the small action-conditioned
    fusion and output heads are broadcast over the action dimension.
    """

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        width = config.width
        hidden = config.world_model_hidden
        actions = config.actions

        self.state_trunk = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(actions, hidden)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.next_latent_head = nn.Linear(hidden, width)
        reward_outputs = (
            config.reward_bins
            if config.reward_prediction == "symlog_twohot_v1"
            else 1
        )
        self.reward_head = nn.Linear(hidden, reward_outputs)
        self.reward_distribution = (
            SymlogTwoHotReward(
                num_bins=config.reward_bins,
                symlog_min=config.reward_symlog_min,
                symlog_max=config.reward_symlog_max,
            )
            if config.reward_prediction == "symlog_twohot_v1"
            else None
        )
        self.hazard_head = nn.Linear(hidden, 1)
        # Identity-initialized, non-gradient parameters fitted only on the
        # disjoint TRAIN-CAL partition after the raw hazard path is frozen.
        self.register_buffer("hazard_calibration_scale", torch.ones(actions))
        self.register_buffer("hazard_calibration_bias", torch.zeros(actions))
        if config.hazard_outcome_path == "dedicated_stopgrad_v1":
            self.hazard_state_trunk = nn.Sequential(
                nn.Linear(width, hidden),
                nn.ReLU(),
            )
            self.hazard_action_embedding = nn.Embedding(actions, hidden)
            self.hazard_outcome_trunk = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
            )
        else:
            self.hazard_state_trunk = None
            self.hazard_action_embedding = None
            self.hazard_outcome_trunk = None

    def forward(
        self,
        context: Tensor,
        action_ids: Tensor | None = None,
    ) -> ActionOutcomeTable:
        """Return an exhaustive action table for ``context``.

        Args:
            context: Current causal state representation, shape ``[B, W]``.
            action_ids: Optional column labelling as ``[A]`` or ``[B, A]``.
                Every row must be a permutation of the complete action set.
        """
        if context.dim() != 2 or context.shape[-1] != self.config.width:
            raise ValueError(f"context must have shape [B, {self.config.width}]")
        batch_size = context.shape[0]
        ids = self._normalize_action_ids(action_ids, batch_size, context.device)
        # Keep numerical kernels in canonical action order. Reordering their
        # inputs can change vectorized floating-point results by a few ULPs.
        # Column layout is a presentation operation, applied after all heads.
        compute_ids = torch.arange(self.config.actions, device=context.device).expand(batch_size, -1)

        state = self.state_trunk(context)  # [B, H], computed once
        state = F.layer_norm(state, (state.shape[-1],))
        state = state.unsqueeze(1).expand(-1, self.config.actions, -1)
        action = self.action_embedding(compute_ids)
        action = F.layer_norm(action, (action.shape[-1],))
        conditioned = self.outcome_trunk(torch.cat((state, action), dim=-1))

        reward_output = self.reward_head(conditioned)
        reward_logits = None
        if self.reward_distribution is None:
            predicted_reward = reward_output
        else:
            reward_logits = reward_output
            predicted_reward = self.reward_distribution.decode(reward_logits).unsqueeze(-1)

        hazard_conditioned = conditioned
        if self.hazard_state_trunk is not None:
            hazard_state = self.hazard_state_trunk(context.detach())
            hazard_state = F.layer_norm(
                hazard_state,
                (hazard_state.shape[-1],),
            )
            hazard_state = hazard_state.unsqueeze(1).expand(
                -1,
                self.config.actions,
                -1,
            )
            hazard_action = self.hazard_action_embedding(compute_ids)
            hazard_action = F.layer_norm(
                hazard_action,
                (hazard_action.shape[-1],),
            )
            hazard_conditioned = self.hazard_outcome_trunk(
                torch.cat((hazard_state, hazard_action), dim=-1)
            )

        raw_hazard_logits = self.hazard_head(hazard_conditioned)
        calibration_scale = self.hazard_calibration_scale[compute_ids].unsqueeze(-1)
        calibration_bias = self.hazard_calibration_bias[compute_ids].unsqueeze(-1)
        calibrated_hazard_logits = (
            raw_hazard_logits * calibration_scale + calibration_bias
        )

        def arrange(value: Tensor | None) -> Tensor | None:
            if value is None or action_ids is None:
                return value
            return value.gather(1, ids.unsqueeze(-1).expand(-1, -1, value.shape[-1]))

        return ActionOutcomeTable(
            action_ids=ids,
            predicted_next_latent=arrange(self.next_latent_head(conditioned)),
            predicted_reward=arrange(predicted_reward),
            predicted_reward_logits=arrange(reward_logits),
            predicted_hazard=arrange(torch.sigmoid(calibrated_hazard_logits)),
            raw_hazard_logits=arrange(raw_hazard_logits),
            predicted_hazard_logits=arrange(calibrated_hazard_logits),
        )

    @torch.no_grad()
    def install_hazard_calibration(
        self,
        calibrator: PerActionAffineHazardCalibrator,
    ) -> None:
        """Bake a fitted TRAIN-CAL transform into the live outcome model."""
        if self.config.outcome_architecture != "all_action_table_v1":
            raise ValueError("hazard calibration requires all_action_table_v1")
        if self.config.hazard_outcome_path != "dedicated_stopgrad_v1":
            raise ValueError("hazard calibration requires dedicated_stopgrad_v1")
        if calibrator.action_count != self.config.actions:
            raise ValueError("hazard calibrator action count does not match model")
        scale = torch.as_tensor(
            calibrator.scales,
            device=self.hazard_calibration_scale.device,
            dtype=self.hazard_calibration_scale.dtype,
        )
        bias = torch.as_tensor(
            calibrator.biases,
            device=self.hazard_calibration_bias.device,
            dtype=self.hazard_calibration_bias.dtype,
        )
        if not bool(torch.isfinite(scale).all() and torch.isfinite(bias).all()):
            raise ValueError("hazard calibration parameters must be finite")
        if not bool(scale.gt(0.0).all()):
            raise ValueError("hazard calibration scales must be positive")
        self.hazard_calibration_scale.copy_(scale)
        self.hazard_calibration_bias.copy_(bias)

    def _normalize_action_ids(
        self,
        action_ids: Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        actions = self.config.actions
        if action_ids is None:
            ids = torch.arange(actions, device=device).unsqueeze(0)
            ids = ids.expand(batch_size, -1)
        else:
            if action_ids.dtype not in _INTEGER_DTYPES:
                raise ValueError("action_ids must use an integer dtype")
            ids = action_ids.to(device=device, dtype=torch.long)
            if ids.dim() == 1 and ids.shape[0] == actions:
                ids = ids.unsqueeze(0).expand(batch_size, -1)
            elif ids.dim() != 2 or ids.shape != (batch_size, actions):
                raise ValueError(f"action_ids must have shape [{actions}] or [B, {actions}]")

        canonical = torch.arange(actions, device=device).expand(batch_size, -1)
        if not bool(ids.sort(dim=1).values.eq(canonical).all()):
            raise ValueError("each action_ids row must be a permutation of every action")
        return ids

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Load pre-V2.1i checkpoints with an explicit identity calibrator.

        Their stale source bundles keep them ineligible for publication, but
        retaining read compatibility is useful for development comparisons.
        """
        calibration_names = (
            "hazard_calibration_scale",
            "hazard_calibration_bias",
        )
        calibration_present = tuple(
            prefix + name in state_dict for name in calibration_names
        )
        if not any(calibration_present):
            for name in calibration_names:
                state_dict[prefix + name] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


__all__ = ["ActionOutcomeTable", "VectorizedOutcomeModelV2"]
