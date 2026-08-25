"""Causal trajectory objective for the Core V2 predictive milestone.

The model sees only the current observation and already-applied control.  The
teacher action, transition reward/hazard, and next observation remain explicit
supervision targets.  Prediction-error inputs at tick ``t`` use only the
outcome of tick ``t - 1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from ..data import MazeChaseCounterfactualTransition
from ..training.batches import AllActionTrajectoryBatchV1, TrajectoryBatch
from ..training.objective import _rgb_tensor
from ..types import GenericControl, HidKey
from .config import FeatureFlags
from .core import CoreV2Model
from .losses import total_v2_loss


_ACTION_KEYS = (
    int(HidKey.W),
    int(HidKey.A),
    int(HidKey.S),
    int(HidKey.D),
)
_HAZARD_EVENTS = frozenset({"collision", "caught"})


@dataclass(frozen=True)
class V2TrajectoryLoss:
    loss: Tensor
    components: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]
    samples: int


def control_action_class(control: GenericControl) -> int:
    """Map neutral/W/A/S/D control to Core V2's five action classes."""
    if not isinstance(control, GenericControl):
        raise ValueError("control must be a GenericControl")
    active = [index for index, key in enumerate(_ACTION_KEYS, start=1) if key in control.keys_down]
    if len(active) > 1:
        raise ValueError("Core V2 trajectory actions must be exclusive W/A/S/D or idle")
    return active[0] if active else 0


def transition_hazard(events: tuple[str, ...]) -> float:
    """Return the observable binary physical-hazard target for one transition."""
    return float(any(event in _HAZARD_EVENTS for event in events))


class V2TrajectoryObjective(nn.Module):
    """Unroll Core V2 over a batch of equal-length causal trajectories."""

    def __init__(
        self,
        model: CoreV2Model,
        *,
        flags: FeatureFlags,
        resolution: tuple[int, int] = (32, 32),
        hold_action_weight: float = 0.1,
        prediction_error_intervention: str = "normal",
    ) -> None:
        super().__init__()
        if model.world_model is None or model.world_model.target_encoder is None:
            raise ValueError("V2 trajectory training requires a configured world model target encoder")
        if not flags.consequence_learning or not flags.world_model_learning:
            raise ValueError("V2 trajectory training requires consequence and world-model learning")
        if not 0.0 < hold_action_weight <= 1.0:
            raise ValueError("hold_action_weight must be in (0, 1]")
        if prediction_error_intervention not in {"normal", "zero"}:
            raise ValueError("prediction_error_intervention must be normal or zero")
        self.model = model
        self.flags = flags
        self.resolution = resolution
        self.hold_action_weight = hold_action_weight
        self.prediction_error_intervention = prediction_error_intervention

    def forward(self, batch: TrajectoryBatch) -> V2TrajectoryLoss:
        if not isinstance(batch, TrajectoryBatch):
            raise ValueError("batch must be a TrajectoryBatch")
        if any(
            isinstance(transition, MazeChaseCounterfactualTransition)
            for sequence in batch.sequences
            for transition in sequence.transitions
        ):
            raise ValueError(
                "counterfactual targets require AllActionV2TrajectoryObjective"
            )
        return self._forward_batch(batch)

    def _forward_batch(
        self,
        batch: TrajectoryBatch | AllActionTrajectoryBatchV1,
    ) -> V2TrajectoryLoss:
        device = next(self.model.parameters()).device
        batch_size = batch.batch_size
        state = self.model.init_state(batch_size, device)
        totals: dict[str, Tensor] = {}
        metric_totals: dict[str, Tensor] = {}
        total_loss = torch.zeros((), device=device)
        optimized_ticks = 0
        max_abs_belief = torch.zeros((), device=device)
        max_abs_thought = torch.zeros((), device=device)

        for tick in range(batch.sequence_length):
            transitions = tuple(sequence.transitions[tick] for sequence in batch.sequences)
            pixels = _rgb_tensor(
                tuple(transition.observation.rgb for transition in transitions),
                device=device,
                resolution=self.resolution,
            )
            next_pixels = _rgb_tensor(
                tuple(transition.next_observation_target.rgb for transition in transitions),
                device=device,
                resolution=self.resolution,
            )
            applied_actions = torch.tensor(
                [control_action_class(transition.applied_control) for transition in transitions],
                dtype=torch.long,
                device=device,
            )
            previous_actions = torch.tensor(
                [
                    control_action_class(transition.observation.previous_control)
                    for transition in transitions
                ],
                dtype=torch.long,
                device=device,
            )

            prior_reward = None
            prior_hazard = None
            if tick:
                prior = tuple(sequence.transitions[tick - 1] for sequence in batch.sequences)
                prior_reward = torch.tensor(
                    [[transition.reward_target] for transition in prior],
                    dtype=torch.float32,
                    device=device,
                )
                prior_hazard = torch.tensor(
                    [[transition_hazard(transition.event_targets)] for transition in prior],
                    dtype=torch.float32,
                    device=device,
                )

            output, state = self.model(
                pixels,
                state,
                prev_action=previous_actions,
                actual_reward=prior_reward,
                actual_hazard=prior_hazard,
                world_model_action=applied_actions,
                intervention_pe=self.prediction_error_intervention,
            )
            max_abs_belief = torch.maximum(
                max_abs_belief,
                output.belief.detach().abs().max(),
            )
            max_abs_thought = torch.maximum(
                max_abs_thought,
                output.thoughts.detach().abs().max(),
            )
            if tick < batch.burn_in_steps:
                continue

            target_action = torch.tensor(
                [control_action_class(transition.action_target) for transition in transitions],
                dtype=torch.long,
                device=device,
            )
            action_weights = torch.where(
                target_action == previous_actions,
                torch.full(
                    (batch_size,),
                    self.hold_action_weight,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.ones(batch_size, dtype=torch.float32, device=device),
            )
            reward_target = torch.tensor(
                [[transition.reward_target] for transition in transitions],
                dtype=torch.float32,
                device=device,
            )
            hazard_target = torch.tensor(
                [[transition_hazard(transition.event_targets)] for transition in transitions],
                dtype=torch.float32,
                device=device,
            )
            next_latent = self.model.world_model.compute_target(next_pixels)
            counterfactual_rows = tuple(
                transition.counterfactual_targets
                if isinstance(transition, MazeChaseCounterfactualTransition)
                else None
                for transition in transitions
            )
            has_counterfactual = tuple(row is not None for row in counterfactual_rows)
            if any(has_counterfactual) and not all(has_counterfactual):
                raise ValueError(
                    "a trajectory tick cannot mix factual-only and all-action samples"
                )
            all_action_reward = None
            all_action_hazard = None
            all_action_next_latent = None
            if all(has_counterfactual):
                if output.outcome_table is None:
                    raise ValueError(
                        "all-action dataset targets require an all-action outcome model"
                    )
                rows = tuple(row for row in counterfactual_rows if row is not None)
                actions = self.model.config.actions
                if actions != 5 or any(len(row) != actions for row in rows):
                    raise ValueError(
                        "counterfactual target vocabulary must match model actions"
                    )
                canonical_reward = torch.tensor(
                    [
                        [target.reward_target for target in row]
                        for row in rows
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                canonical_hazard = torch.tensor(
                    [
                        [
                            [transition_hazard(target.event_targets)]
                            for target in row
                        ]
                        for row in rows
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                branch_pixels = _rgb_tensor(
                    tuple(
                        target.next_observation_target.rgb
                        for row in rows
                        for target in row
                    ),
                    device=device,
                    resolution=self.resolution,
                )
                canonical_next = self.model.world_model.compute_target(
                    branch_pixels
                ).reshape(batch_size, actions, -1)
                # Targets are stored in semantic action order. Align them to
                # the table's explicit column labels instead of trusting a
                # positional convention.
                ids = output.outcome_table.action_ids
                all_action_reward = torch.gather(canonical_reward, 1, ids)
                all_action_hazard = torch.gather(
                    canonical_hazard,
                    1,
                    ids.unsqueeze(-1),
                )
                all_action_next_latent = torch.gather(
                    canonical_next,
                    1,
                    ids.unsqueeze(-1).expand(-1, -1, canonical_next.shape[-1]),
                )
            tick_loss, components = total_v2_loss(
                output,
                self.model.config,
                self.flags,
                target_action,
                actual_reward=reward_target,
                actual_hazard=hazard_target,
                next_latent=next_latent,
                decision_weight_normalization="fixed_exposure_mean_v1",
                decision_sample_weights=action_weights,
                all_action_reward=all_action_reward,
                all_action_hazard=all_action_hazard,
                all_action_next_latent=all_action_next_latent,
            )
            total_loss = total_loss + tick_loss
            for name, value in components.items():
                totals[name] = totals.get(name, torch.zeros((), device=device)) + value
            predicted_action = output.decision.action_values.argmax(dim=-1)
            factual = output.pending_prediction
            if (
                self.model.config.outcome_architecture == "all_action_table_v1"
                and factual is not None
            ):
                predicted_hazard = factual.predicted_hazard
                predicted_reward = factual.predicted_reward
            else:
                predicted_hazard = output.hypotheses.predicted_hazard.mean(dim=1)
                predicted_reward = output.hypotheses.predicted_reward.mean(dim=1)
            predicted_next = output.pending_prediction.predicted_next_latent
            if predicted_next.dim() == 3:
                predicted_next = predicted_next.mean(dim=1)
            tick_metrics = {
                "action_accuracy": (predicted_action == target_action).float().mean(),
                "reward_mae": (
                    predicted_reward - reward_target
                ).abs().mean(),
                "hazard_brier": (predicted_hazard - hazard_target).square().mean(),
                "next_latent_cosine": torch.nn.functional.cosine_similarity(
                    predicted_next,
                    next_latent,
                    dim=-1,
                ).mean(),
                "turn_correct": (
                    (predicted_action == target_action) & (target_action != previous_actions)
                ).float().sum(),
                "turn_samples": (target_action != previous_actions).float().sum(),
                "hazard_positives": hazard_target.sum(),
                "hazard_positive_probability_sum": (
                    predicted_hazard * hazard_target
                ).sum(),
                "hazard_negative_probability_sum": (
                    predicted_hazard * (1.0 - hazard_target)
                ).sum(),
                "hazard_negatives": (1.0 - hazard_target).sum(),
            }
            for name, value in tick_metrics.items():
                metric_totals[name] = (
                    metric_totals.get(name, torch.zeros((), device=device)) + value
                )
            optimized_ticks += 1

        if optimized_ticks < 1:
            raise RuntimeError("trajectory batch contains no optimized ticks")
        return V2TrajectoryLoss(
            loss=total_loss / optimized_ticks,
            components={name: value / optimized_ticks for name, value in totals.items()},
            metrics={
                **{
                    name: value / optimized_ticks
                    for name, value in metric_totals.items()
                    if name not in {
                        "turn_correct",
                        "turn_samples",
                        "hazard_positive_probability_sum",
                        "hazard_negative_probability_sum",
                    }
                },
                "turn_accuracy": metric_totals["turn_correct"]
                / metric_totals["turn_samples"].clamp_min(1.0),
                "turn_samples": metric_totals["turn_samples"],
                "hazard_positives": metric_totals["hazard_positives"],
                "hazard_negatives": metric_totals["hazard_negatives"],
                "hazard_positive_probability": (
                    metric_totals["hazard_positive_probability_sum"]
                    / metric_totals["hazard_positives"].clamp_min(1.0)
                ),
                "hazard_negative_probability": (
                    metric_totals["hazard_negative_probability_sum"]
                    / metric_totals["hazard_negatives"].clamp_min(1.0)
                ),
                "max_abs_belief": max_abs_belief,
                "max_abs_thought": max_abs_thought,
            },
            samples=batch_size * optimized_ticks,
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Advance the EMA target exactly once after an optimizer step."""
        self.model.world_model.update_target_encoder(self.model.encoder)


class AllActionV2TrajectoryObjective(V2TrajectoryObjective):
    """Fail-closed objective for exhaustive same-state action supervision."""

    def __init__(self, model: CoreV2Model, **kwargs: object) -> None:
        if model.config.outcome_architecture != "all_action_table_v1":
            raise ValueError(
                "all-action supervision requires all_action_table_v1"
            )
        super().__init__(model, **kwargs)

    def forward(self, batch: AllActionTrajectoryBatchV1) -> V2TrajectoryLoss:
        if not isinstance(batch, AllActionTrajectoryBatchV1):
            raise ValueError("batch must be an AllActionTrajectoryBatchV1")
        return self._forward_batch(batch)


__all__ = [
    "AllActionV2TrajectoryObjective",
    "V2TrajectoryLoss",
    "V2TrajectoryObjective",
    "control_action_class",
    "transition_hazard",
]
