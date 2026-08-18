"""B2 control: a recurrent latent world-model actor in its own recipe family.

``LatentWorldModelActor`` is the preregistered ``irene.world_model_actor.gru_latent.v1``
control: the parameter-matched monolithic GRU trunk plus a small learned
latent transition model. Where the slot suite predicts future sensor
encodings with a direct head, this control must roll its latent state
forward through its own transition model — teacher-forced on the recorded
action sequence — before decoding to the sensor target. That rollout
objective is the defining feature of a world-model actor and cannot be
expressed by the shared slot-suite objective, so this variant has its own
objective terms and does not ride on the slot-suite manifest.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .baselines import (
    ArchitectureVariantIdentity,
    ParameterMatchedMonolithicBaseline,
)
from .spec import ThoughtFieldConfig


WORLD_MODEL_ACTOR_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.world_model_actor.gru_latent.v1",
    latent_topology="single_widened_gru_latent_with_learned_transition_rollout",
    peer_routing=False,
    thought_workspace_writes=True,
    pooled_recurrent_input=True,
    matching_role="own_recipe_family_world_model_actor_control",
    limitations=(
        "The latent transition model is trained by teacher-forced rollouts on recorded actions, not by imagination-based policy learning.",
        "This control has its own objective terms and recipe family; it does not ride on the slot-suite fairness manifest.",
    ),
)

# Transition and decoder sizing, selected by exact allocated-parameter
# enumeration: with the parameter-matched monolithic trunk (29,643,900
# trainable), these heads add 235,800 trainable parameters for a total of
# 29,879,700 — 0.69% above the reference's 29,674,318, inside the
# preregistered 1% band.
ACTION_EMBED_WIDTH = 64
TRANSITION_HIDDEN_WIDTH = 192
DECODER_BOTTLENECK_WIDTH = 64
CONTROL_VECTOR_WIDTH = 307


class LatentWorldModelActor(ParameterMatchedMonolithicBaseline):
    """Monolithic GRU actor with a learned latent transition model (B2)."""

    architecture_identity = WORLD_MODEL_ACTOR_IDENTITY
    architecture_variant_id = WORLD_MODEL_ACTOR_IDENTITY.variant_id
    # train.py resolves this fail-closed hook instead of the shared
    # slot-suite objective; the rollout world loss lives in its own module.
    training_objective_class_path = (
        "irene_brain.training.world_model_objective:LatentRolloutObjective"
    )

    def __init__(
        self,
        config: ThoughtFieldConfig | None = None,
        *,
        input_resolution: tuple[int, int] = (32, 32),
        plan_steps: int = 3,
    ) -> None:
        super().__init__(
            config,
            input_resolution=input_resolution,
            plan_steps=plan_steps,
        )
        width = self.config.core_width
        self.action_embed = nn.Linear(CONTROL_VECTOR_WIDTH, ACTION_EMBED_WIDTH)
        self.latent_transition = nn.Sequential(
            nn.Linear(width + ACTION_EMBED_WIDTH, TRANSITION_HIDDEN_WIDTH),
            nn.SiLU(),
            nn.Linear(TRANSITION_HIDDEN_WIDTH, width),
        )
        self.latent_decoder = nn.Sequential(
            nn.Linear(width, DECODER_BOTTLENECK_WIDTH),
            nn.SiLU(),
            nn.Linear(DECODER_BOTTLENECK_WIDTH, width),
        )

    def roll_sensor_embedding(self, latent: Tensor, actions: Sequence[Tensor]) -> Tensor:
        """Roll the latent through the transition model and decode to sensor space.

        ``latent`` is the current GRU latent ``[batch, width]``; ``actions``
        are the teacher-forced control vectors ``[batch, 307]`` for each
        rollout step. The transition is residual: each step adds a learned
        correction to the carried latent.
        """

        if not isinstance(actions, Sequence) or not actions:
            raise ValueError("actions must be a non-empty sequence")
        current = latent
        for action in actions:
            if tuple(action.shape) != (latent.shape[0], CONTROL_VECTOR_WIDTH):
                raise ValueError("each action must have shape [batch, 307]")
            embedded = self.action_embed(action.float())
            current = current + self.latent_transition(
                torch.cat((current, embedded), dim=-1)
            )
        return self.latent_decoder(current)


__all__ = [
    "ACTION_EMBED_WIDTH",
    "DECODER_BOTTLENECK_WIDTH",
    "LatentWorldModelActor",
    "TRANSITION_HIDDEN_WIDTH",
    "WORLD_MODEL_ACTOR_IDENTITY",
]
