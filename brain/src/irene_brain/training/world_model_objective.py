"""B2 rollout world objective for the latent world-model actor.

This module is intentionally not imported by :mod:`irene_brain.training` so
Phase-0 tooling does not import PyTorch merely by discovering the package.

Where the shared slot-suite objective scores the trunk's direct
future-embedding head against future sensor encodings, the
``irene.world_model_actor.gru_latent.v1`` control must instead roll its GRU
latent through its own learned transition model — teacher-forced on the
recorded action sequence — and decode the rolled latent to the sensor
encoding of each horizon's target frame. That rollout surface is the
defining feature of a world-model actor; it lives here, in the variant's
own recipe family, rather than on the slot-suite manifest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .batches import control_to_vector
from .objective import ThoughtFieldObjective, _rgb_tensor

if TYPE_CHECKING:
    from .batches import TrajectoryBatch


class LatentRolloutObjective(ThoughtFieldObjective):
    """World loss via teacher-forced latent rollouts (B2 own recipe family).

    For each supported horizon ``k`` of the training window, the GRU latent
    produced at the current step is rolled ``k`` times through the actor's
    transition model on the recorded action targets, decoded to sensor
    space, and scored against the frozen pixel-encoder encoding of the
    frame ``k`` steps ahead. Every other objective term — action, value,
    diversity, and all diagnostics — is inherited unchanged from the shared
    objective so the control stays metric-comparable with the slot suite.
    """

    def _world_horizon_losses(
        self,
        *,
        output: object,
        batch: "TrajectoryBatch",
        time_index: int,
        next_pixels: Tensor,
        horizon_offsets: tuple[int, ...],
        slot_groups: tuple[tuple[int, ...], ...] | None,
        device: torch.device,
    ) -> tuple[dict[int, Tensor], Tensor]:
        if slot_groups is not None:
            raise ValueError(
                "the world-model actor carries a single latent; horizon slot"
                " partitions do not apply to its rollout objective"
            )
        roll = getattr(self.model, "roll_sensor_embedding", None)
        if not callable(roll):
            raise ValueError(
                "LatentRolloutObjective requires a model with"
                " roll_sensor_embedding (the B2 latent world-model actor)"
            )
        # The single-latent trunk carries thoughts [batch, 1, width]; the
        # rollout starts from the latent after the current observation,
        # exactly where the direct future-embedding head also predicts from.
        latent = output.next_state.thoughts.flatten(1).float()
        losses: dict[int, Tensor] = {}
        prediction_error: Tensor | None = None
        for horizon in horizon_offsets:
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
            # Teacher forcing: the recorded action targets at steps
            # time_index .. time_index+horizon-1 drive the rollout, so the
            # horizon-k target frame stays exactly the frame those actions
            # produced. h1 needs only the current step's action, which always
            # exists inside the window.
            actions = tuple(
                torch.tensor(
                    [
                        control_to_vector(
                            sequence.transitions[time_index + step].action_target
                        )
                        for sequence in batch.sequences
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                for step in range(horizon)
            )
            predicted = roll(latent, actions)
            with torch.no_grad():
                sensor_target = self.model.pixel_encoder(horizon_pixels).mean(dim=1)
            # [batch, 1] keeps the diagnostics contract: prediction_error is
            # [batch, thoughtlets] and this actor has exactly one latent.
            horizon_error = (
                (predicted - sensor_target.float())
                .square()
                .mean(dim=-1, keepdim=True)
            )
            if horizon == 1:
                # Diagnostics keep the short-horizon error surface.
                prediction_error = horizon_error
            losses[horizon] = horizon_error.mean()
        if prediction_error is None:
            raise RuntimeError("the window must always support the h1 world loss")
        return losses, prediction_error


__all__ = ["LatentRolloutObjective"]
