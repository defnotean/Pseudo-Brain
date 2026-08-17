"""Direct generic-HID actuator queries with no cognitive pooling token."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spec import ActuatorQuerySpec


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    control: Tensor
    button_logits: Tensor
    mouse_zero_logit: Tensor
    mouse_mean: Tensor
    mouse_log_scale: Tensor
    scroll_mean: Tensor
    gamepad_axis_mean: Tensor
    gamepad_axis_log_scale: Tensor
    confidence_logits: Tensor
    hold_duration_seconds: Tensor
    control_plan: Tensor
    query_features: Tensor
    state_attention: Tensor
    thought_attention: Tensor


class DirectActuatorReadout(nn.Module):
    """Let each physical actuator query attend the complete unpooled state."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        actuator: ActuatorQuerySpec,
        plan_steps: int,
    ) -> None:
        super().__init__()
        if plan_steps < 1:
            raise ValueError("plan_steps must be positive")
        self.actuator = actuator
        self.total_queries = actuator.total_queries
        self.plan_steps = plan_steps
        self.queries = nn.Parameter(torch.empty(1, self.total_queries, width))
        self.state_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.coordination = nn.MultiheadAttention(width, heads, batch_first=True)
        self.coordination_norm = nn.LayerNorm(width)
        self.signal = nn.Linear(width, 1)
        self.scale = nn.Linear(width, 1)
        self.confidence = nn.Linear(width, 1)
        self.hold = nn.Linear(width, 1)
        self.plan = nn.Linear(width, plan_steps)
        self.mouse_zero = nn.Linear(width * 2, 1)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(
        self,
        *,
        sensors: Tensor,
        belief: Tensor,
        thoughts: Tensor,
        working_memory: Tensor,
        retrieved_memory: Tensor,
        goal_context: Tensor,
    ) -> ActionPrediction:
        batch = sensors.shape[0]
        thoughtlets = thoughts.shape[1]
        registers = thoughts.shape[2]
        thought_start = sensors.shape[1] + belief.shape[1]
        thought_count = thoughtlets * registers
        complete_state = torch.cat(
            (
                sensors,
                belief,
                thoughts.flatten(1, 2),
                working_memory,
                retrieved_memory.flatten(1, 2),
                goal_context,
            ),
            dim=1,
        )
        queries = self.queries.expand(batch, -1, -1).to(dtype=sensors.dtype)
        features, state_attention = self.cross_attention(
            self.query_norm(queries),
            self.state_norm(complete_state),
            self.state_norm(complete_state),
            need_weights=True,
            average_attn_weights=True,
        )
        coordinated, _ = self.coordination(features, features, features, need_weights=False)
        features = self.coordination_norm(features + coordinated)

        raw_signal = self.signal(features).squeeze(-1)
        raw_scale = self.scale(features).squeeze(-1).clamp(-7.0, 5.0)
        confidence_logits = self.confidence(features).squeeze(-1)
        hold_duration = F.softplus(self.hold(features).squeeze(-1)).clamp_max(2.0)
        control_plan = self.plan(features).transpose(1, 2)

        keyboard_end = self.actuator.keyboard_keys
        mouse_button_end = keyboard_end + self.actuator.mouse_buttons
        mouse_axis_end = mouse_button_end + self.actuator.mouse_axes
        scroll_end = mouse_axis_end + self.actuator.scroll_axes
        gamepad_button_end = scroll_end + self.actuator.gamepad_buttons
        gamepad_axis_end = gamepad_button_end + self.actuator.gamepad_axes

        button_logits = torch.cat(
            (raw_signal[:, :mouse_button_end], raw_signal[:, scroll_end:gamepad_button_end]),
            dim=1,
        )
        mouse_features = features[:, mouse_button_end:mouse_axis_end].flatten(1)
        mouse_zero_logit = self.mouse_zero(mouse_features)
        mouse_mean = raw_signal[:, mouse_button_end:mouse_axis_end]
        mouse_log_scale = raw_scale[:, mouse_button_end:mouse_axis_end]
        scroll_mean = raw_signal[:, mouse_axis_end:scroll_end]
        gamepad_axis_mean = torch.tanh(raw_signal[:, gamepad_button_end:gamepad_axis_end])
        gamepad_axis_log_scale = raw_scale[:, gamepad_button_end:gamepad_axis_end]
        control = raw_signal.clone()
        control[:, gamepad_button_end:gamepad_axis_end] = gamepad_axis_mean

        thought_attention = state_attention[
            :,
            :,
            thought_start : thought_start + thought_count,
        ].reshape(batch, self.total_queries, thoughtlets, registers)
        thought_attention = thought_attention.sum(dim=-1)
        return ActionPrediction(
            control=control,
            button_logits=button_logits,
            mouse_zero_logit=mouse_zero_logit,
            mouse_mean=mouse_mean,
            mouse_log_scale=mouse_log_scale,
            scroll_mean=scroll_mean,
            gamepad_axis_mean=gamepad_axis_mean,
            gamepad_axis_log_scale=gamepad_axis_log_scale,
            confidence_logits=confidence_logits,
            hold_duration_seconds=hold_duration,
            control_plan=control_plan,
            query_features=features,
            state_attention=state_attention,
            thought_attention=thought_attention,
        )
