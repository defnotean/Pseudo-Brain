"""Stage-A causal trajectory objective for the trainable thought field.

This module is intentionally not imported by :mod:`irene_brain.training` so
Phase-0 tooling does not import PyTorch merely by discovering the package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..model.torch_model import deterministic_thought_identity_codes
from ..types import HidKey, RgbFrame
from .batches import (
    BUTTON_TARGET_INDICES,
    CONTINUOUS_TARGET_INDICES,
    TrajectoryBatch,
    control_to_vector,
)


_MOVEMENT_KEYS = (
    ("w", int(HidKey.W)),
    ("a", int(HidKey.A)),
    ("s", int(HidKey.S)),
    ("d", int(HidKey.D)),
)

# Maze-chase exclusive-direction action loss. Softmax CE on W/A/S/D matches
# exclusive_argmax_wasd_v1 play decode (ranking, not independent logit > 0).
# Idle uses the same margin as closed-loop play: winner strictly below -4.0.
# Frozen RCQ-v2 / moving-shapes keep support_aware_calibrated_v1.
EXCLUSIVE_WASD_SOFTMAX_V1 = "exclusive_wasd_softmax_v1"
EXCLUSIVE_WASD_SOFTMAX_TURN_WEIGHTED_V1 = "exclusive_wasd_softmax_turn_weighted_v1"
EXCLUSIVE_WASD_SOFTMAX_IDLE_MARGIN = -4.0
EXCLUSIVE_WASD_HOLD_WEIGHT = 0.1
_ACTION_LOSS_KINDS = frozenset(
    {
        "sparse_hard_negative_v1",
        "support_aware_calibrated_v1",
        EXCLUSIVE_WASD_SOFTMAX_V1,
        EXCLUSIVE_WASD_SOFTMAX_TURN_WEIGHTED_V1,
    }
)
_EXCLUSIVE_WASD_LOSS_KINDS = frozenset(
    {
        EXCLUSIVE_WASD_SOFTMAX_V1,
        EXCLUSIVE_WASD_SOFTMAX_TURN_WEIGHTED_V1,
    }
)


def _multi_horizon_offsets(sequence_length: int, burn_in_steps: int) -> tuple[int, ...]:
    """Powers-of-two prediction offsets the recorded window fully supports.

    Offset 1 is always trained: at an optimized step ``t`` its target frame
    ``t + 1`` is the transition's own ``next_observation_target``, which
    exists even at the final step. Longer offsets ``k`` target frame
    ``t + k`` inside the recorded window, so they require
    ``k <= sequence_length - 1 - burn_in_steps`` to keep at least the first
    optimized step eligible. A length-8, burn-in-2 campaign window yields
    exactly ``(1, 2, 4)`` — the preregistered B1 multi-horizon world loss.
    Offsets beyond 4 are capped pending a future preregistration with
    longer registered windows, and a length-2 smoke window yields ``(1,)``,
    bit-identical to the original single-horizon world loss.
    """

    offsets: list[int] = [1]
    offset = 2
    while offset <= 4 and offset <= sequence_length - 1 - burn_in_steps:
        offsets.append(offset)
        offset *= 2
    return tuple(offsets)


def _even_slot_groups(thoughtlets: int, groups: int) -> tuple[tuple[int, ...], ...]:
    """Contiguous, disjoint, complete, even partition of slot indices."""

    base, remainder = divmod(thoughtlets, groups)
    result: list[tuple[int, ...]] = []
    start = 0
    for group_index in range(groups):
        size = base + (1 if group_index < remainder else 0)
        result.append(tuple(range(start, start + size)))
        start += size
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LossOutput:
    loss: Tensor
    metrics: Mapping[str, Tensor]
    samples: int


class ThoughtFieldObjective(nn.Module):
    """Unroll the same one-step streaming model used for live inference."""

    def __init__(
        self,
        model: nn.Module,
        *,
        action_loss_kind: str = "support_aware_calibrated_v1",
        button_support_control_indices: tuple[int, ...] = (4, 7, 22, 26),
        button_support_weight: float = 0.8,
        button_background_weight: float = 0.2,
        button_background_tail_mix: float = 0.9,
        button_background_tail_temperature: float = 0.1,
        continuous_action_weight: float = 0.25,
        action_weight: float = 1.0,
        value_weight: float = 0.1,
        world_weight: float = 0.1,
        diversity_weight: float = 0.05,
        deadzone_hinge_weight: float = 0.0,
        deadzone_hinge_margin: float = 0.04,
        opposite_pair_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        if action_loss_kind not in _ACTION_LOSS_KINDS:
            raise ValueError("unsupported action_loss_kind")
        self.action_loss_kind = action_loss_kind
        self.button_support_weight = _positive_weight(
            button_support_weight,
            "button_support_weight",
        )
        self.button_background_weight = _positive_weight(
            button_background_weight,
            "button_background_weight",
        )
        if abs(
            self.button_support_weight + self.button_background_weight - 1.0
        ) > 1e-12:
            raise ValueError("button support and background weights must sum to 1")
        self.button_background_tail_mix = _unit_interval(
            button_background_tail_mix,
            "button_background_tail_mix",
        )
        self.button_background_tail_temperature = _strictly_positive_weight(
            button_background_tail_temperature,
            "button_background_tail_temperature",
        )
        self.continuous_action_weight = _positive_weight(
            continuous_action_weight,
            "continuous_action_weight",
        )
        self.action_weight = _positive_weight(action_weight, "action_weight")
        self.value_weight = _positive_weight(value_weight, "value_weight")
        self.world_weight = _positive_weight(world_weight, "world_weight")
        self.diversity_weight = _positive_weight(diversity_weight, "diversity_weight")
        self.deadzone_hinge_weight = _positive_weight(
            deadzone_hinge_weight,
            "deadzone_hinge_weight",
        )
        if not (0.0 <= deadzone_hinge_margin < 0.05):
            raise ValueError("deadzone_hinge_margin must be inside the 0.05 deadzone")
        self.deadzone_hinge_margin = float(deadzone_hinge_margin)
        self.opposite_pair_weight = _positive_weight(
            opposite_pair_weight,
            "opposite_pair_weight",
        )
        self.register_buffer(
            "button_target_indices",
            torch.tensor(BUTTON_TARGET_INDICES, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "movement_key_indices",
            torch.tensor(tuple(index for _name, index in _MOVEMENT_KEYS), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "continuous_target_indices",
            torch.tensor(CONTINUOUS_TARGET_INDICES, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "button_support_control_indices",
            _validated_support_control_indices(button_support_control_indices),
            persistent=False,
        )

    def set_loss_weights(
        self,
        *,
        action_weight: float,
        value_weight: float,
        world_weight: float,
        diversity_weight: float,
    ) -> None:
        """Apply one preregistered stage's scalar objective weights."""

        normalized = {
            name: _positive_weight(value, name)
            for name, value in (
                ("action_weight", action_weight),
                ("value_weight", value_weight),
                ("world_weight", world_weight),
                ("diversity_weight", diversity_weight),
            )
        }
        if not any(value > 0.0 for value in normalized.values()):
            raise ValueError("at least one objective weight must be positive")
        for name, value in normalized.items():
            setattr(self, name, value)

    def forward(self, batch: object) -> LossOutput:
        if not isinstance(batch, TrajectoryBatch):
            raise ValueError("ThoughtFieldObjective requires a TrajectoryBatch")
        parameter = next(self.model.parameters())
        device = parameter.device
        total_action = parameter.new_zeros(())
        total_value = parameter.new_zeros(())
        total_diversity = parameter.new_zeros(())
        world_sums: dict[int, Tensor] = {}
        world_counts: dict[int, int] = {}
        correct_keys = parameter.new_zeros(())
        positive_key_recall_sum = parameter.new_zeros(())
        movement_metric_sums: dict[str, Tensor] = {}
        diagnostic_metric_sums: dict[str, Tensor] = {}
        exit_action_loss_sums: list[Tensor] | None = None
        exit_button_exact_sums: list[Tensor] | None = None
        exit_movement_exact_sums: list[Tensor] | None = None
        key_elements = 0
        optimized_steps = batch.sequence_length - batch.burn_in_steps
        state = None
        horizon_offsets = _multi_horizon_offsets(
            batch.sequence_length, batch.burn_in_steps
        )
        # The B1 fixed-horizon control statically partitions its slots across
        # the window's supported horizons; every other variant keeps the
        # flexible min-over-slots assignment per horizon.
        slot_groups: tuple[tuple[int, ...], ...] | None = None
        if getattr(self.model, "fixed_horizon_partition", False):
            slot_groups = _even_slot_groups(
                self.model.config.thoughtlets, len(horizon_offsets)
            )
        # The same deterministic Gaussian identities are supplied on every
        # recurrent timestep in both train and evaluation.  The model has the
        # identical default for direct/live calls, while this explicit tensor
        # makes sequence-level reuse unambiguous here.
        thought_noise = deterministic_eval_thought_noise(
            thoughtlets=self.model.config.thoughtlets,
            width=self.model.config.core_width,
            batch_size=batch.batch_size,
            device=device,
        )

        for time_index in range(batch.sequence_length):
            transitions = tuple(
                sequence.transitions[time_index] for sequence in batch.sequences
            )
            pixels = _rgb_tensor(
                tuple(transition.observation.rgb for transition in transitions),
                device=device,
                resolution=getattr(self.model, "input_resolution", None),
            )
            previous_control = torch.tensor(
                [
                    control_to_vector(transition.observation.previous_control)
                    for transition in transitions
                ],
                dtype=torch.float32,
                device=device,
            )
            delta_seconds = torch.tensor(
                [
                    0.0
                    if time_index == 0
                    else (
                        transition.observation.elapsed_ns
                        - batch.sequences[batch_index]
                        .transitions[time_index - 1]
                        .observation.elapsed_ns
                    )
                    / 1_000_000_000.0
                    for batch_index, transition in enumerate(transitions)
                ],
                dtype=torch.float32,
                device=device,
            )
            if time_index < batch.burn_in_steps:
                with torch.no_grad():
                    output = self.model(
                        pixels,
                        previous_control,
                        delta_seconds,
                        state,
                        thought_noise=thought_noise,
                    )
                state = output.next_state.detach()
                continue

            output = self.model(
                pixels,
                previous_control,
                delta_seconds,
                state,
                thought_noise=thought_noise,
            )
            state = output.next_state
            action_target = torch.tensor(
                [control_to_vector(transition.action_target) for transition in transitions],
                dtype=torch.float32,
                device=device,
            )
            exclusive_row_weights = None
            if self.action_loss_kind == EXCLUSIVE_WASD_SOFTMAX_TURN_WEIGHTED_V1:
                previous_action_target = torch.tensor(
                    [
                        control_to_vector(
                            batch.sequences[batch_index]
                            .transitions[time_index - 1]
                            .action_target
                        )
                        for batch_index, _transition in enumerate(transitions)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                exclusive_row_weights = _exclusive_wasd_turn_weights(
                    action_target.index_select(1, self.movement_key_indices),
                    previous_action_target.index_select(1, self.movement_key_indices),
                )
            key_target = action_target[:, :256] > 0.5
            value_target = torch.tensor(
                [transition.value_target for transition in transitions],
                dtype=torch.float32,
                device=device,
            )
            next_pixels = _rgb_tensor(
                tuple(
                    transition.next_observation_target.rgb
                    for transition in transitions
                ),
                device=device,
                resolution=getattr(self.model, "input_resolution", None),
            )

            exit_losses = tuple(
                _structured_action_loss(
                    prediction,
                    action_target,
                    self.button_target_indices,
                    loss_kind=self.action_loss_kind,
                    support_control_indices=self.button_support_control_indices,
                    support_weight=self.button_support_weight,
                    background_weight=self.button_background_weight,
                    background_tail_mix=self.button_background_tail_mix,
                    background_tail_temperature=self.button_background_tail_temperature,
                    continuous_weight=self.continuous_action_weight,
                    deadzone_hinge_weight=self.deadzone_hinge_weight,
                    deadzone_hinge_margin=self.deadzone_hinge_margin,
                    opposite_pair_weight=self.opposite_pair_weight,
                    movement_key_indices=self.movement_key_indices,
                    exclusive_row_weights=exclusive_row_weights,
                )
                for prediction in output.anytime_actions
            )
            if exit_action_loss_sums is None:
                exit_action_loss_sums = [parameter.new_zeros(()) for _ in exit_losses]
                exit_button_exact_sums = [parameter.new_zeros(()) for _ in exit_losses]
                exit_movement_exact_sums = [
                    parameter.new_zeros(()) for _ in exit_losses
                ]
            elif len(exit_losses) != len(exit_action_loss_sums):
                raise RuntimeError("the number of anytime exits changed within a trajectory")
            assert exit_button_exact_sums is not None
            assert exit_movement_exact_sums is not None
            for exit_index, (prediction, exit_loss) in enumerate(
                zip(output.anytime_actions, exit_losses)
            ):
                exit_action_loss_sums[exit_index] = (
                    exit_action_loss_sums[exit_index] + exit_loss
                )
                exit_button_exact_sums[exit_index] = (
                    exit_button_exact_sums[exit_index]
                    + _button_exact_set_count(
                        prediction.button_logits.float(),
                        action_target,
                        self.button_target_indices,
                    )
                )
                exit_movement_exact_sums[exit_index] = (
                    exit_movement_exact_sums[exit_index]
                    + _movement_exact_set_count(
                        prediction.button_logits[:, :256].float(),
                        key_target,
                        self.movement_key_indices,
                    )
                )
            action_loss = torch.stack(exit_losses).mean()
            value_loss = F.mse_loss(output.value.float(), value_target)
            horizon_losses, prediction_error = self._world_horizon_losses(
                output=output,
                batch=batch,
                time_index=time_index,
                next_pixels=next_pixels,
                horizon_offsets=horizon_offsets,
                slot_groups=slot_groups,
                device=device,
            )
            for horizon, horizon_loss in horizon_losses.items():
                if horizon in world_sums:
                    world_sums[horizon] = world_sums[horizon] + horizon_loss
                    world_counts[horizon] += 1
                else:
                    world_sums[horizon] = horizon_loss
                    world_counts[horizon] = 1
            similarity = output.diagnostics.thought_cosine_similarity.float()
            thoughtlets = similarity.shape[-1]
            if thoughtlets == 1:
                # A monolithic recurrent control has no slot pairs. The empty
                # pair set contributes the exact additive identity, not NaN.
                diversity_loss = similarity.new_zeros(())
            else:
                off_diagonal = ~torch.eye(
                    thoughtlets,
                    dtype=torch.bool,
                    device=similarity.device,
                )
                diversity_loss = similarity[..., off_diagonal].square().mean()
            final_key_logits = output.action.button_logits[:, :256].float()
            key_prediction = final_key_logits > 0.0
            correct_keys = correct_keys + (key_prediction == key_target).sum()
            positive_per_sample = key_target.sum(dim=1)
            true_positive_per_sample = (key_prediction & key_target).sum(dim=1)
            # Stage A always has one or two positive movement keys. Defining a
            # zero-positive sample as recall 1 keeps this a per-sample macro
            # metric whose accumulation remains sample-weighted and additive.
            positive_key_recall_sum = positive_key_recall_sum + torch.where(
                positive_per_sample > 0,
                true_positive_per_sample.float()
                / positive_per_sample.clamp_min(1).float(),
                torch.ones_like(positive_per_sample, dtype=torch.float32),
            ).sum()
            step_movement_metrics = _movement_metric_counts(
                final_key_logits,
                key_target,
                previous_control[:, :256],
                self.movement_key_indices,
            )
            for name, value in step_movement_metrics.items():
                movement_metric_sums[name] = movement_metric_sums.get(
                    name, parameter.new_zeros(())
                ) + value
            with torch.no_grad():
                step_quiescence_metrics = _full_action_quiescence_metric_counts(
                    output.action.button_logits.float(),
                    output.action.control.float(),
                    action_target,
                    self.button_target_indices,
                    self.continuous_target_indices,
                    self.button_support_control_indices,
                )
            for name, value in step_quiescence_metrics.items():
                movement_metric_sums[name] = movement_metric_sums.get(
                    name, parameter.new_zeros(())
                ) + value
            with torch.no_grad():
                step_logit_metrics = _final_action_logit_metric_counts(
                    final_key_logits,
                    key_target,
                    self.movement_key_indices,
                )
            for name, value in step_logit_metrics.items():
                movement_metric_sums[name] = movement_metric_sums.get(
                    name, parameter.new_zeros(())
                ) + value
            with torch.no_grad():
                step_diagnostic_metrics = _thought_diagnostic_metric_counts(
                    output,
                    prediction_error,
                    self.movement_key_indices,
                )
            for name, value in step_diagnostic_metrics.items():
                diagnostic_metric_sums[name] = diagnostic_metric_sums.get(
                    name, parameter.new_zeros(())
                ) + value
            key_elements += key_target.numel()

            total_action = total_action + action_loss
            total_value = total_value + value_loss
            total_diversity = total_diversity + diversity_loss

        action_mean = total_action / optimized_steps
        value_mean = total_value / optimized_steps
        # The world weight splits evenly across the window's supported
        # horizons; each horizon's mean is taken over its eligible steps.
        horizon_means = {
            horizon: world_sums[horizon] / world_counts[horizon]
            for horizon in horizon_offsets
        }
        world_mean = sum(horizon_means.values()) / len(horizon_offsets)
        world_metrics = {
            f"world_loss_h{horizon}": horizon_means[horizon]
            for horizon in horizon_offsets
        }
        diversity_mean = total_diversity / optimized_steps
        metric_samples = batch.sample_count
        movement_metrics = {
            name: value.float() / metric_samples
            for name, value in movement_metric_sums.items()
        }
        diagnostic_metrics = {
            name: value.float() / metric_samples
            for name, value in diagnostic_metric_sums.items()
        }
        if (
            exit_action_loss_sums is None
            or exit_button_exact_sums is None
            or exit_movement_exact_sums is None
        ):
            raise RuntimeError("the trajectory produced no optimized anytime exits")
        exit_metrics: dict[str, Tensor] = {}
        exit_sums = zip(
            exit_action_loss_sums,
            exit_button_exact_sums,
            exit_movement_exact_sums,
        )
        for exit_index, (
            exit_action_sum,
            button_exact_sum,
            movement_exact_sum,
        ) in enumerate(exit_sums):
            exit_metrics[f"exit_{exit_index}_action_loss"] = (
                exit_action_sum / optimized_steps
            )
            exit_metrics[f"exit_{exit_index}_button_exact_match"] = (
                button_exact_sum.float() / metric_samples
            )
            exit_metrics[f"exit_{exit_index}_movement_exact_match"] = (
                movement_exact_sum.float() / metric_samples
            )
        loss = (
            self.action_weight * action_mean
            + self.value_weight * value_mean
            + self.world_weight * world_mean
            + self.diversity_weight * diversity_mean
        )
        return LossOutput(
            loss=loss,
            metrics={
                "action_loss": action_mean,
                "value_loss": value_mean,
                "world_loss": world_mean,
                **world_metrics,
                "diversity_loss": diversity_mean,
                "key_accuracy": correct_keys.float() / key_elements,
                "positive_key_recall": positive_key_recall_sum.float()
                / metric_samples,
                "total_loss": loss,
                **movement_metrics,
                **diagnostic_metrics,
                **exit_metrics,
            },
            samples=metric_samples,
        )

    def _world_horizon_losses(
        self,
        *,
        output: object,
        batch: TrajectoryBatch,
        time_index: int,
        next_pixels: Tensor,
        horizon_offsets: tuple[int, ...],
        slot_groups: tuple[tuple[int, ...], ...] | None,
        device: torch.device,
    ) -> tuple[dict[int, Tensor], Tensor]:
        """Per-horizon world losses for one optimized step, plus the h1 error.

        Any thoughtlet may own each horizon's prediction — or, on the fixed
        horizon control, any thoughtlet inside the statically assigned
        group. A hard minimum avoids forcing all slots toward the same
        target vector. Subclasses with their own preregistered recipe
        family (the B2 world-model actor) override exactly this surface.
        """

        losses: dict[int, Tensor] = {}
        prediction_error: Tensor | None = None
        for horizon_index, horizon in enumerate(horizon_offsets):
            if horizon == 1:
                horizon_pixels = next_pixels
            else:
                target_index = time_index + horizon - 1
                if target_index >= batch.sequence_length:
                    continue
                horizon_pixels = _rgb_tensor(
                    tuple(
                        sequence.transitions[target_index]
                        .next_observation_target.rgb
                        for sequence in batch.sequences
                    ),
                    device=device,
                    resolution=getattr(self.model, "input_resolution", None),
                )
            with torch.no_grad():
                sensor_target = self.model.pixel_encoder(horizon_pixels).mean(dim=1)
            horizon_error = (
                output.world.future_embedding.float()
                - sensor_target.float().unsqueeze(1)
            ).square().mean(dim=-1)
            if horizon == 1:
                # Diagnostics keep the short-horizon error surface.
                prediction_error = horizon_error
            if slot_groups is None:
                horizon_loss = horizon_error.min(dim=1).values.mean()
            else:
                horizon_loss = (
                    horizon_error[:, list(slot_groups[horizon_index])]
                    .min(dim=1)
                    .values.mean()
                )
            losses[horizon] = horizon_loss
        if prediction_error is None:
            raise RuntimeError("the window must always support the h1 world loss")
        return losses, prediction_error


def _positive_weight(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not (0.0 <= result < float("inf")):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _strictly_positive_weight(value: object, name: str) -> float:
    result = _positive_weight(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _unit_interval(value: object, name: str) -> float:
    result = _positive_weight(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must be at most 1")
    return result


def _validated_support_control_indices(indices: object) -> Tensor:
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError("button_support_control_indices must be a non-empty sequence")
    normalized: list[int] = []
    for value in indices:
        if type(value) is not int:
            raise ValueError("button_support_control_indices must contain integers")
        if value not in BUTTON_TARGET_INDICES:
            raise ValueError("button_support_control_indices must select button channels")
        normalized.append(value)
    if tuple(normalized) != tuple(sorted(set(normalized))):
        raise ValueError("button_support_control_indices must be sorted and unique")
    return torch.tensor(normalized, dtype=torch.long)


def deterministic_eval_thought_noise(
    *,
    thoughtlets: int,
    width: int,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    """Return fixed distinct slot codes without advancing any global RNG."""

    dimensions = (thoughtlets, width, batch_size)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("thoughtlets, width, and batch_size must be positive integers")
    slot_codes = deterministic_thought_identity_codes(
        thoughtlets=thoughtlets,
        width=width,
    )
    return slot_codes.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)


def _normalized_gram_rank_proxy(gram: Tensor) -> Tensor:
    """Return a per-sample participation-rank fraction in ``[0, 1]``."""

    if gram.ndim != 3 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError("gram must have shape [batch, slots, slots]")
    slots = gram.shape[-1]
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    frobenius_squared = gram.square().sum(dim=(-2, -1))
    effective_rank = trace.square() / frobenius_squared.clamp_min(1e-12)
    return (effective_rank / slots).clamp(0.0, 1.0)


def _thought_diagnostic_metric_counts(
    output: object,
    prediction_error: Tensor,
    movement_key_indices: Tensor,
) -> dict[str, Tensor]:
    """Return additive, sample-level thought diagnostics for one timestep."""

    summaries = output.diagnostics.thought_summaries.float()
    batch, thoughtlets, _width = summaries.shape
    if tuple(prediction_error.shape) != (batch, thoughtlets):
        raise ValueError("prediction_error must have shape [batch, thoughtlets]")
    if tuple(movement_key_indices.shape) != (4,):
        raise ValueError("movement_key_indices must contain W, A, S, and D")

    summary_gram = output.diagnostics.thought_cosine_similarity.float()
    summary_rank = _normalized_gram_rank_proxy(summary_gram)

    registers = output.next_state.thoughts.float()
    full_vectors = F.normalize(registers.flatten(2), dim=-1, eps=1e-6)
    full_gram = torch.matmul(full_vectors, full_vectors.transpose(-1, -2))
    full_register_rank = _normalized_gram_rank_proxy(full_gram)

    private = registers - registers.mean(dim=1, keepdim=True)
    total_energy = registers.square().mean(dim=(1, 2, 3))
    private_energy = private.square().mean(dim=(1, 2, 3))
    slot_private_energy = torch.where(
        total_energy > 1e-12,
        private_energy / total_energy.clamp_min(1e-12),
        torch.zeros_like(total_energy),
    ).clamp(0.0, 1.0)

    thought_attention = output.diagnostics.actuator_thought_attention.float()
    movement_attention = thought_attention.index_select(1, movement_key_indices)
    attention_mass = movement_attention.sum(dim=-1, keepdim=True)
    normalized_attention = movement_attention / attention_mass.clamp_min(1e-12)
    effective_slots = normalized_attention.square().sum(dim=-1).clamp_min(
        1e-12
    ).reciprocal()
    effective_slots = torch.where(
        attention_mass.squeeze(-1) > 1e-12,
        effective_slots,
        torch.zeros_like(effective_slots),
    ).clamp(0.0, float(thoughtlets))
    movement_effective_slots = effective_slots.mean(dim=1)

    applied_expire = output.diagnostics.applied_expire_probability.float()
    if tuple(applied_expire.shape) != (batch, thoughtlets):
        raise ValueError(
            "applied_expire_probability must have shape [batch, thoughtlets]"
        )

    if thoughtlets >= 2:
        best_two = torch.topk(
            prediction_error.float(),
            k=2,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        raw_gap = (best_two[:, 1] - best_two[:, 0]).clamp_min(0.0)
        world_gap = raw_gap
        # Collapse diagnostics: mean off-diagonal summary cosine (the
        # thesis-scale alert watches for sustained values above 0.7) and
        # the fraction of full-register slot pairs that are near-duplicates.
        off_diagonal = ~torch.eye(
            thoughtlets,
            dtype=torch.bool,
            device=summary_gram.device,
        )
        pairwise_cosine_mean = summary_gram[..., off_diagonal].mean(dim=-1)
        duplicate_pair_fraction = (
            (full_gram[..., off_diagonal] > 0.98).float().mean(dim=-1)
        )
        # Movement-readout slot utilization: per-query attention entropy
        # normalized to [0, 1]; a collapsed readout concentrates on one slot.
        query_entropy = -(
            normalized_attention * normalized_attention.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(thoughtlets)
        slot_utilization_entropy = torch.where(
            attention_mass.squeeze(-1) > 1e-12,
            query_entropy,
            torch.zeros_like(query_entropy),
        ).clamp(0.0, 1.0).mean(dim=1)
    else:
        world_gap = prediction_error.new_zeros((batch,), dtype=torch.float32)
        pairwise_cosine_mean = prediction_error.new_zeros((batch,), dtype=torch.float32)
        duplicate_pair_fraction = prediction_error.new_zeros(
            (batch,), dtype=torch.float32
        )
        slot_utilization_entropy = prediction_error.new_zeros(
            (batch,), dtype=torch.float32
        )

    # These are diagnostics only. In particular, the best/second gap is not
    # added to the loss, so future-prediction divergence is never rewarded.
    return {
        "thought_summary_rank_proxy": summary_rank.sum(),
        "thought_full_register_rank_proxy": full_register_rank.sum(),
        "thought_slot_private_energy": slot_private_energy.sum(),
        "thought_pairwise_cosine_mean": pairwise_cosine_mean.sum(),
        "thought_duplicate_pair_fraction": duplicate_pair_fraction.sum(),
        "movement_query_effective_slot_count": movement_effective_slots.sum(),
        "movement_query_slot_entropy": slot_utilization_entropy.sum(),
        "mean_applied_expire_probability": applied_expire.mean(dim=1).sum(),
        "world_best_second_error_gap": world_gap.sum(),
    }


def _rgb_tensor(
    frames: tuple[RgbFrame, ...],
    *,
    device: torch.device,
    resolution: tuple[int, int] | None,
) -> Tensor:
    if not frames:
        raise ValueError("frames cannot be empty")
    first = frames[0]
    width = first.width
    height = first.height
    if any(frame.width != width or frame.height != height for frame in frames):
        raise ValueError("all RGB frames in a batch must have equal dimensions")
    buffer = bytearray().join(frame.pixels for frame in frames)
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).reshape(
        len(frames), height, width, 3
    )
    tensor = tensor.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    tensor = tensor.div_(255.0)
    if resolution is not None and tuple(tensor.shape[-2:]) != tuple(resolution):
        tensor = F.interpolate(tensor, size=resolution, mode="nearest")
    return tensor


def _structured_action_loss(
    prediction: object,
    target: Tensor,
    button_indices: Tensor,
    *,
    loss_kind: str = "support_aware_calibrated_v1",
    support_control_indices: Tensor | None = None,
    support_weight: float = 0.8,
    background_weight: float = 0.2,
    background_tail_mix: float = 0.9,
    background_tail_temperature: float = 0.1,
    continuous_weight: float = 0.25,
    deadzone_hinge_weight: float = 0.0,
    deadzone_hinge_margin: float = 0.04,
    opposite_pair_weight: float = 0.0,
    movement_key_indices: Tensor | None = None,
    exclusive_row_weights: Tensor | None = None,
) -> Tensor:
    button_target = target.index_select(1, button_indices)
    button_logits = prediction.button_logits.float()
    if tuple(button_logits.shape) != tuple(button_target.shape):
        raise ValueError("button logits must match the selected button target shape")
    if loss_kind == "sparse_hard_negative_v1":
        button_loss = _legacy_sparse_button_loss(button_logits, button_target)
    elif loss_kind == "support_aware_calibrated_v1":
        if support_control_indices is None:
            support_control_indices = button_indices.new_tensor((4, 7, 22, 26))
        button_loss = _support_aware_calibrated_button_loss(
            button_logits,
            button_target,
            button_indices,
            support_control_indices,
            support_weight=support_weight,
            background_weight=background_weight,
            background_tail_mix=background_tail_mix,
            background_tail_temperature=background_tail_temperature,
        )
    elif loss_kind in _EXCLUSIVE_WASD_LOSS_KINDS:
        if movement_key_indices is None:
            raise ValueError(
                "exclusive WASD softmax requires movement_key_indices in W, A, S, D order"
            )
        if (
            loss_kind == EXCLUSIVE_WASD_SOFTMAX_TURN_WEIGHTED_V1
            and exclusive_row_weights is None
        ):
            raise ValueError(
                "exclusive_wasd_softmax_turn_weighted_v1 requires exclusive_row_weights"
            )
        if loss_kind == EXCLUSIVE_WASD_SOFTMAX_V1 and exclusive_row_weights is not None:
            raise ValueError(
                "exclusive_wasd_softmax_v1 rejects exclusive_row_weights"
            )
        button_loss = _exclusive_wasd_softmax_button_loss(
            button_logits,
            button_target,
            button_indices,
            movement_key_indices,
            support_weight=support_weight,
            background_weight=background_weight,
            background_tail_mix=background_tail_mix,
            background_tail_temperature=background_tail_temperature,
            exclusive_row_weights=exclusive_row_weights,
        )
    else:
        raise ValueError(f"unsupported action loss kind: {loss_kind}")
    mouse_loss = F.mse_loss(prediction.mouse_mean.float(), target[:, 264:266])
    scroll_loss = F.mse_loss(prediction.scroll_mean.float(), target[:, 266:267])
    gamepad_axis_loss = F.mse_loss(
        prediction.gamepad_axis_mean.float(),
        target[:, 299:307],
    )
    loss = button_loss + continuous_weight * (
        mouse_loss + scroll_loss + gamepad_axis_loss
    )
    if deadzone_hinge_weight > 0.0:
        continuous_prediction = torch.cat(
            (
                prediction.mouse_mean.float(),
                prediction.scroll_mean.float(),
                prediction.gamepad_axis_mean.float(),
            ),
            dim=1,
        )
        hinge = (continuous_prediction.abs() - deadzone_hinge_margin).clamp_min(0.0)
        loss = loss + deadzone_hinge_weight * hinge.mean()
    if opposite_pair_weight > 0.0:
        if movement_key_indices is None or tuple(movement_key_indices.shape) != (4,):
            raise ValueError(
                "opposite_pair_weight requires movement_key_indices in W, A, S, D order"
            )
        movement_logits = button_logits[:, :256].index_select(1, movement_key_indices)
        probabilities = torch.sigmoid(movement_logits)
        # _MOVEMENT_KEYS order is W, A, S, D: opposite pairs are (W, S) and (A, D).
        coactivation = probabilities[:, 0] * probabilities[:, 2] + (
            probabilities[:, 1] * probabilities[:, 3]
        )
        loss = loss + opposite_pair_weight * coactivation.mean()
    return loss


def _legacy_sparse_button_loss(button_logits: Tensor, button_target: Tensor) -> Tensor:
    """The immutable A-run loss, retained only for an explicit A/B recipe."""

    element_loss = F.binary_cross_entropy_with_logits(
        button_logits,
        button_target,
        reduction="none",
    )
    positive = button_target > 0.5
    negative = ~positive
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    positive_loss = (element_loss * positive).sum(dim=1) / positive_count.clamp_min(1)
    negative_loss = (element_loss * negative).sum(dim=1) / negative_count.clamp_min(1)
    hard_negative_loss = element_loss.masked_fill(~negative, 0.0).amax(dim=1)
    return (
        0.5 * (positive_loss + hard_negative_loss) + 0.05 * negative_loss
    ).mean()


def _support_aware_calibrated_button_loss(
    button_logits: Tensor,
    button_target: Tensor,
    button_control_indices: Tensor,
    support_control_indices: Tensor,
    *,
    support_weight: float,
    background_weight: float,
    background_tail_mix: float,
    background_tail_temperature: float,
) -> Tensor:
    """Calibrated support BCE plus a smooth, nondiluting negative tail risk.

    ``support_control_indices`` use coordinates in the full 307-control vector.
    They are resolved through ``button_control_indices`` so this loss does not
    rely on a particular ordering of the 296 emitted button logits. Any target
    positive outside the configured support joins the support for that sample,
    which guarantees that the remaining background is always-negative.
    """

    if button_logits.ndim != 2 or tuple(button_target.shape) != tuple(
        button_logits.shape
    ):
        raise ValueError("button logits and targets must have equal rank-2 shapes")
    if button_control_indices.ndim != 1 or button_control_indices.numel() != (
        button_logits.shape[1]
    ):
        raise ValueError("button_control_indices must map every emitted button logit")
    if support_control_indices.ndim != 1 or support_control_indices.numel() == 0:
        raise ValueError("support_control_indices must be a non-empty vector")
    mapping = button_control_indices.unsqueeze(1) == support_control_indices.unsqueeze(0)

    configured_support = mapping.any(dim=1).unsqueeze(0)
    positive = button_target > 0.5
    support = configured_support | positive
    background = ~support

    support_bce = F.binary_cross_entropy_with_logits(
        button_logits,
        button_target,
        reduction="none",
    )
    support_loss = _masked_row_mean(support_bce, support)

    negative_loss = F.softplus(button_logits)
    background_mean = _masked_row_mean(negative_loss, background)
    background_tail = _centered_masked_logmeanexp(
        negative_loss,
        background,
        temperature=background_tail_temperature,
    )
    background_risk = (
        (1.0 - background_tail_mix) * background_mean
        + background_tail_mix * background_tail
    )
    return (
        support_weight * support_loss + background_weight * background_risk
    ).mean()


def _button_columns_for_controls(
    button_control_indices: Tensor,
    control_indices: Tensor,
) -> Tensor:
    """Map HID control indices onto emitted button-logit columns."""

    mapping = button_control_indices.unsqueeze(1) == control_indices.unsqueeze(0)
    if mapping.ndim != 2:
        raise ValueError("control mapping must be rank-2")
    if not bool(mapping.any(dim=0).all().item()):
        raise ValueError(
            "exclusive WASD softmax requires W, A, S, and D in emitted buttons"
        )
    if bool((mapping.sum(dim=0) != 1).any().item()):
        raise ValueError("each WASD control must map to exactly one emitted button")
    return mapping.to(dtype=torch.long).argmax(dim=0)


def _exclusive_wasd_row_loss(wasd_logits: Tensor, wasd_target: Tensor) -> Tensor:
    """Per-row exclusive WASD loss matching argmax play decode.

    Teacher maze-chase movement is one of W/A/S/D, or idle. Directed rows
    use softmax cross-entropy toward the unique teacher key (first-on in
    W/A/S/D order if several are labeled). Idle rows push the winning
    logit strictly below ``EXCLUSIVE_WASD_SOFTMAX_IDLE_MARGIN``, the same
    threshold closed-loop exclusive-argmax uses to stay idle.
    """

    if wasd_logits.ndim != 2 or tuple(wasd_logits.shape[1:]) != (4,):
        raise ValueError("WASD logits must have shape [batch, 4]")
    if tuple(wasd_target.shape) != tuple(wasd_logits.shape):
        raise ValueError("WASD targets must match WASD logits")
    positive = wasd_target > 0.5
    idle = ~positive.any(dim=1)
    class_ids = positive.to(dtype=torch.long).argmax(dim=1)
    directed = F.cross_entropy(wasd_logits, class_ids, reduction="none")
    idle_loss = F.softplus(
        wasd_logits.amax(dim=1) - EXCLUSIVE_WASD_SOFTMAX_IDLE_MARGIN
    )
    return torch.where(idle, idle_loss, directed)


def _exclusive_wasd_class_ids(wasd_target: Tensor) -> Tensor:
    if wasd_target.ndim != 2 or tuple(wasd_target.shape[1:]) != (4,):
        raise ValueError("WASD targets must have shape [batch, 4]")
    positive = wasd_target > 0.5
    idle = ~positive.any(dim=1)
    class_ids = positive.to(dtype=torch.long).argmax(dim=1)
    return torch.where(idle, torch.full_like(class_ids, -1), class_ids)


def _exclusive_wasd_turn_weights(current: Tensor, previous: Tensor) -> Tensor:
    """Weight teacher direction *changes* over corridor holds.

    Exclusive CE on mixed tiled windows still collapsed to a constant key
    because most ticks are holds. Down-weighting holds (0.1) relative to
    turns (1.0) is a teaching-signal change, not a coverage tweak.
    """

    if tuple(current.shape) != tuple(previous.shape):
        raise ValueError("turn weights need matching current and previous WASD targets")
    change = _exclusive_wasd_class_ids(current) != _exclusive_wasd_class_ids(previous)
    ones = torch.ones(current.shape[0], dtype=current.dtype, device=current.device)
    holds = torch.full(
        (current.shape[0],),
        EXCLUSIVE_WASD_HOLD_WEIGHT,
        dtype=current.dtype,
        device=current.device,
    )
    return torch.where(change, ones, holds)


def _exclusive_wasd_softmax_button_loss(
    button_logits: Tensor,
    button_target: Tensor,
    button_control_indices: Tensor,
    movement_key_indices: Tensor,
    *,
    support_weight: float,
    background_weight: float,
    background_tail_mix: float,
    background_tail_temperature: float,
    exclusive_row_weights: Tensor | None = None,
) -> Tensor:
    """Exclusive WASD softmax plus the frozen non-movement background tail.

    WASD is trained as one mutually exclusive direction, not four
    independent BCEs. Non-movement buttons keep the support-aware
    background risk so maze-chase cannot recapture RCQ-v2's calibrated
    multi-label recipe by renaming it.
    """

    if button_logits.ndim != 2 or tuple(button_target.shape) != tuple(
        button_logits.shape
    ):
        raise ValueError("button logits and targets must have equal rank-2 shapes")
    if movement_key_indices.ndim != 1 or tuple(movement_key_indices.shape) != (4,):
        raise ValueError(
            "exclusive WASD softmax requires movement_key_indices in W, A, S, D order"
        )
    columns = _button_columns_for_controls(
        button_control_indices, movement_key_indices
    )
    exclusive = _exclusive_wasd_row_loss(
        button_logits.index_select(1, columns),
        button_target.index_select(1, columns),
    )
    if exclusive_row_weights is not None:
        if tuple(exclusive_row_weights.shape) != tuple(exclusive.shape):
            raise ValueError("exclusive_row_weights must match the WASD batch")
        exclusive = exclusive * exclusive_row_weights

    wasd_emitted = torch.zeros(
        button_logits.shape[1],
        dtype=torch.bool,
        device=button_logits.device,
    )
    wasd_emitted[columns] = True
    background = ~wasd_emitted.unsqueeze(0)
    positive = button_target > 0.5
    outside_positive = positive & background
    support_bce = F.binary_cross_entropy_with_logits(
        button_logits,
        button_target,
        reduction="none",
    )
    outside_positive_loss = _masked_row_mean(support_bce, outside_positive)

    negative_loss = F.softplus(button_logits)
    background_negative = background & ~positive
    background_mean = _masked_row_mean(negative_loss, background_negative)
    background_tail = _centered_masked_logmeanexp(
        negative_loss,
        background_negative,
        temperature=background_tail_temperature,
    )
    background_risk = (
        (1.0 - background_tail_mix) * background_mean
        + background_tail_mix * background_tail
    )
    return (
        support_weight * (exclusive + outside_positive_loss)
        + background_weight * background_risk
    ).mean()


def _masked_row_mean(values: Tensor, mask: Tensor) -> Tensor:
    if tuple(values.shape) != tuple(mask.shape) or values.ndim != 2:
        raise ValueError("masked row mean requires equal rank-2 shapes")
    counts = mask.sum(dim=1)
    return (values * mask).sum(dim=1) / counts.clamp_min(1).to(values.dtype)


def _centered_masked_logmeanexp(
    values: Tensor,
    mask: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Return temperature-scaled log-mean-exp, with finite empty rows."""

    if tuple(values.shape) != tuple(mask.shape) or values.ndim != 2:
        raise ValueError("masked logmeanexp requires equal rank-2 shapes")
    if not (temperature > 0.0 and temperature < float("inf")):
        raise ValueError("temperature must be finite and positive")
    if values.shape[1] == 0:
        return values.sum(dim=1)

    counts = mask.sum(dim=1)
    has_values = counts > 0
    # An injected constant makes all-empty rows safe for logsumexp's backward
    # pass. The final where selects exact zero for those rows, so no dummy
    # gradient reaches the input.
    fallback = (~has_values).unsqueeze(1) & (
        torch.arange(values.shape[1], device=values.device).unsqueeze(0) == 0
    )
    scaled = values / temperature
    masked = torch.where(mask, scaled, torch.full_like(scaled, -torch.inf))
    masked = torch.where(fallback, torch.zeros_like(masked), masked)
    centered = temperature * (
        torch.logsumexp(masked, dim=1)
        - counts.clamp_min(1).to(values.dtype).log()
    )
    return torch.where(has_values, centered, torch.zeros_like(centered))


def _button_exact_set_count(
    button_logits: Tensor,
    target: Tensor,
    button_indices: Tensor,
) -> Tensor:
    """Count samples whose complete discrete button set is exactly correct."""

    batch_size = target.shape[0]
    if target.ndim != 2 or tuple(target.shape) != (batch_size, 307):
        raise ValueError("target must have shape [batch, 307]")
    if button_indices.ndim != 1:
        raise ValueError("button_indices must be one-dimensional")
    expected = (batch_size, button_indices.numel())
    if tuple(button_logits.shape) != expected:
        raise ValueError(f"button_logits must have shape {expected}")
    button_target = target.index_select(1, button_indices) > 0.5
    return ((button_logits > 0.0) == button_target).all(dim=1).sum()


def _full_action_quiescence_metric_counts(
    button_logits: Tensor,
    control: Tensor,
    target: Tensor,
    button_indices: Tensor,
    continuous_indices: Tensor,
    support_control_indices: Tensor,
) -> dict[str, Tensor]:
    """Count all off-support buttons and continuous deadzone violations."""

    batch_size = target.shape[0]
    if tuple(target.shape) != (batch_size, 307):
        raise ValueError("target must have shape [batch, 307]")
    if tuple(control.shape) != (batch_size, 307):
        raise ValueError("control must have shape [batch, 307]")
    if tuple(button_logits.shape) != (batch_size, button_indices.numel()):
        raise ValueError("button_logits do not match the packed button layout")
    if tuple(continuous_indices.shape) != (11,):
        raise ValueError("continuous_indices must select eleven controls")
    if tuple(support_control_indices.shape) != (4,):
        raise ValueError("support_control_indices must select W, A, S, and D")

    packed_support = (
        button_indices.unsqueeze(1) == support_control_indices.unsqueeze(0)
    ).any(dim=1)
    predicted_buttons = button_logits > 0.0
    target_buttons = target.index_select(1, button_indices) > 0.5
    outside = ~packed_support
    continuous_prediction = control.index_select(1, continuous_indices).float()
    continuous_target = target.index_select(1, continuous_indices).float()
    return {
        "full_button_exact_match": (
            predicted_buttons == target_buttons
        ).all(dim=1).sum(),
        "off_support_button_positive_count": predicted_buttons[:, outside].sum(),
        "off_support_button_target_active_count": target_buttons[:, outside].sum(),
        "continuous_target_nonzero_count": (
            continuous_target != 0.0
        ).sum(),
        "continuous_action_squared_magnitude": continuous_prediction.square().sum(),
        "continuous_action_outside_0_05_count": (
            continuous_prediction.abs() > 0.05
        ).sum(),
    }


def _movement_exact_set_count(
    key_logits: Tensor,
    key_target: Tensor,
    movement_key_indices: Tensor,
) -> Tensor:
    """Count samples whose complete W/A/S/D set is exactly correct."""

    batch_size = key_logits.shape[0]
    if tuple(key_logits.shape) != (batch_size, 256):
        raise ValueError("key_logits must have shape [batch, 256]")
    if tuple(key_target.shape) != (batch_size, 256):
        raise ValueError("key_target must have shape [batch, 256]")
    if tuple(movement_key_indices.shape) != (4,):
        raise ValueError("movement_key_indices must contain W, A, S, and D")
    prediction = (key_logits > 0.0).index_select(1, movement_key_indices)
    target = key_target.to(dtype=torch.bool).index_select(1, movement_key_indices)
    return (prediction == target).all(dim=1).sum()


def _final_action_logit_metric_counts(
    key_logits: Tensor,
    key_target: Tensor,
    movement_key_indices: Tensor,
) -> dict[str, Tensor]:
    """Return additive per-sample logit diagnostics for the final exit.

    Means are macro means: each sample contributes one value. A sample with no
    positive key contributes zero to ``final_positive_key_logit_mean``. The
    inactive-movement statistics likewise contribute zero if all four movement
    keys are active, avoiding empty reductions while preserving strict additive
    aggregation across microbatches and evaluation batches.
    """

    batch_size = key_logits.shape[0]
    if tuple(key_logits.shape) != (batch_size, 256):
        raise ValueError("key_logits must have shape [batch, 256]")
    if tuple(key_target.shape) != (batch_size, 256):
        raise ValueError("key_target must have shape [batch, 256]")
    if tuple(movement_key_indices.shape) != (4,):
        raise ValueError("movement_key_indices must contain W, A, S, and D")

    target = key_target.to(dtype=torch.bool)
    positive_count = target.sum(dim=1)
    positive_mean = key_logits.masked_fill(~target, 0.0).sum(
        dim=1
    ) / positive_count.clamp_min(1)
    positive_mean = torch.where(
        positive_count > 0,
        positive_mean,
        torch.zeros_like(positive_mean),
    )

    movement_logits = key_logits.index_select(1, movement_key_indices)
    movement_target = target.index_select(1, movement_key_indices)
    inactive_movement = ~movement_target
    inactive_count = inactive_movement.sum(dim=1)
    inactive_mean = movement_logits.masked_fill(~inactive_movement, 0.0).sum(
        dim=1
    ) / inactive_count.clamp_min(1)
    inactive_max = movement_logits.masked_fill(
        ~inactive_movement,
        -torch.inf,
    ).amax(dim=1)
    no_inactive = inactive_count == 0
    inactive_mean = torch.where(
        no_inactive,
        torch.zeros_like(inactive_mean),
        inactive_mean,
    )
    inactive_max = torch.where(
        no_inactive,
        torch.zeros_like(inactive_max),
        inactive_max,
    )

    unique_teacher = movement_target.sum(dim=1) == 1
    teacher_logit = (movement_logits * movement_target.to(dtype=movement_logits.dtype)).sum(
        dim=1
    )
    other_max = movement_logits.masked_fill(movement_target, -torch.inf).amax(dim=1)
    teacher_gap = teacher_logit - other_max
    teacher_gap = torch.where(
        unique_teacher,
        teacher_gap,
        torch.zeros_like(teacher_gap),
    )

    outside_movement = torch.ones(256, dtype=torch.bool, device=key_logits.device)
    outside_movement[movement_key_indices] = False
    non_movement_max = key_logits[:, outside_movement].amax(dim=1)
    return {
        "final_positive_key_logit_mean": positive_mean.sum(),
        "final_inactive_movement_key_logit_max": inactive_max.sum(),
        "final_inactive_movement_key_logit_mean": inactive_mean.sum(),
        "final_non_movement_key_logit_max": non_movement_max.sum(),
        "final_teacher_movement_logit_gap": teacher_gap.sum(),
    }


def _movement_metric_counts(
    key_logits: Tensor,
    key_target: Tensor,
    previous_keyboard: Tensor,
    movement_key_indices: Tensor,
) -> dict[str, Tensor]:
    """Return additive diagnostic counts for a batch of movement decisions.

    Every value is a count, never a batch-local conditional ratio. The caller
    divides by the total number of decisions, so sample-weighted accumulation
    across time, gradient-accumulation microbatches, and validation batches is
    exact. Conditional changed-action accuracy is recovered after aggregation
    as ``changed_exact_matches_per_sample / changed_samples_per_sample``.
    Per-direction precision and recall are likewise derived from the emitted
    true-positive, predicted-positive, and target-positive rates.
    """

    batch_size = key_logits.shape[0]
    if tuple(key_logits.shape) != (batch_size, 256):
        raise ValueError("key_logits must have shape [batch, 256]")
    if tuple(key_target.shape) != (batch_size, 256):
        raise ValueError("key_target must have shape [batch, 256]")
    if tuple(previous_keyboard.shape) != (batch_size, 256):
        raise ValueError("previous_keyboard must have shape [batch, 256]")
    if tuple(movement_key_indices.shape) != (4,):
        raise ValueError("movement_key_indices must contain W, A, S, and D")

    prediction = key_logits > 0.0
    target = key_target.to(dtype=torch.bool)
    previous = previous_keyboard > 0.5
    movement_prediction = prediction.index_select(1, movement_key_indices)
    movement_target = target.index_select(1, movement_key_indices)
    movement_previous = previous.index_select(1, movement_key_indices)
    exact = (movement_prediction == movement_target).all(dim=1)
    previous_exact = (movement_previous == movement_target).all(dim=1)
    changed = ~previous_exact

    outside_movement = torch.ones(256, dtype=torch.bool, device=key_logits.device)
    outside_movement[movement_key_indices] = False
    movement_false_positive = movement_prediction & ~movement_target
    outside_false_positive = prediction[:, outside_movement] & ~target[:, outside_movement]
    opposite_conflict = (
        (movement_prediction[:, 0] & movement_prediction[:, 2])
        | (movement_prediction[:, 1] & movement_prediction[:, 3])
    )

    wasd_logits = key_logits.index_select(1, movement_key_indices)
    winner = wasd_logits.argmax(dim=1)
    winner_logit = wasd_logits.gather(1, winner.unsqueeze(1)).squeeze(1)
    idle_pred = winner_logit < EXCLUSIVE_WASD_SOFTMAX_IDLE_MARGIN
    exclusive_pred = F.one_hot(winner, 4).to(dtype=torch.bool) & ~idle_pred.unsqueeze(1)
    exclusive_match = (exclusive_pred == movement_target).all(dim=1)

    counts: dict[str, Tensor] = {
        "movement_exact_match": exact.sum(),
        "movement_predicted_active_count": movement_prediction.sum(),
        "movement_target_active_count": movement_target.sum(),
        "movement_false_positive_count": movement_false_positive.sum(),
        "non_movement_key_false_positive_count": outside_false_positive.sum(),
        "movement_opposite_conflict_rate": opposite_conflict.sum(),
        "previous_control_movement_exact_match": previous_exact.sum(),
        "movement_changed_samples_per_sample": changed.sum(),
        "movement_changed_exact_matches_per_sample": (changed & exact).sum(),
        "all_off_movement_exact_match": (~movement_target).all(dim=1).sum(),
        "all_four_movement_exact_match": movement_target.all(dim=1).sum(),
        "movement_exclusive_argmax_match": exclusive_match.sum(),
    }
    for column, (name, _key_index) in enumerate(_MOVEMENT_KEYS):
        direction_prediction = movement_prediction[:, column]
        direction_target = movement_target[:, column]
        counts[f"movement_{name}_true_positive_rate"] = (
            direction_prediction & direction_target
        ).sum()
        counts[f"movement_{name}_predicted_positive_rate"] = direction_prediction.sum()
        counts[f"movement_{name}_target_positive_rate"] = direction_target.sum()
    return counts


__all__ = [
    "EXCLUSIVE_WASD_HOLD_WEIGHT",
    "EXCLUSIVE_WASD_SOFTMAX_IDLE_MARGIN",
    "EXCLUSIVE_WASD_SOFTMAX_TURN_WEIGHTED_V1",
    "EXCLUSIVE_WASD_SOFTMAX_V1",
    "LossOutput",
    "ThoughtFieldObjective",
    "deterministic_eval_thought_noise",
]
