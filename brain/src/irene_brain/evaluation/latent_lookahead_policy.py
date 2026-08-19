"""Closed-loop lookahead planning policy for Pseudo-Brain evaluation.

Wraps an IreneBrainModel with LatentLookaheadPlanner to evaluate candidate action
sequences forward in latent thoughtlet space at each decision step, selecting
actions that maximize cumulative branch utility while avoiding predicted collision
hazards.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..model.intent import DirectionalAction
from ..model.lookahead_planner import (
    LatentLookaheadPlanner,
    LookaheadPlanResult,
    directional_to_control_vector,
)
from ..model.torch_model import BrainState, IreneBrainModel
from ..training.batches import control_to_vector
from ..training.objective import _rgb_tensor, deterministic_eval_thought_noise
from ..types import GenericControl, HidKey, Observation
from .closed_loop_play import control_audit_stats


# HID usage identifiers for W, A, S, D
_KEY_W = int(HidKey.W)  # 26
_KEY_A = int(HidKey.A)  # 4
_KEY_S = int(HidKey.S)  # 22
_KEY_D = int(HidKey.D)  # 7

_KEY_FOR_DIRECTIONAL_ACTION = {
    DirectionalAction.NONE: (),
    DirectionalAction.W: (_KEY_W,),
    DirectionalAction.A: (_KEY_A,),
    DirectionalAction.S: (_KEY_S,),
    DirectionalAction.D: (_KEY_D,),
}


class LatentLookaheadPolicy:
    """Zero-cheating closed-loop evaluation policy powered by latent lookahead planning.

    Adheres to the standard Irene evaluation policy contract:
    - ``identity`` — stable dotted string identifier.
    - ``uses_privileged_state`` — False (all planning is internal to latent thoughts).
    - ``reset(episode_seed)`` — reseeds model state and planner state deterministically.
    - ``act(observation)`` — returns the planned :class:`GenericControl`.
    - ``decide(observation, elapsed_seconds)`` — returns (control, audit_stats, value).
    """

    identity = "model.latent_lookahead.v1"
    uses_privileged_state = False

    def __init__(
        self,
        model: IreneBrainModel,
        *,
        planner: LatentLookaheadPlanner | None = None,
        horizon: int = 3,
        gamma: float = 0.95,
        hazard_weight: float = 5.0,
        hazard_prune_threshold: float = 0.5,
        dead_end_threshold: float = -10.0,
        policy_prior_weight: float = 1.0,
        selective_gating: bool = True,
        direct_action_safety_threshold: float = 0.35,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.policy_prior_weight = policy_prior_weight
        self.selective_gating = selective_gating
        self.direct_action_safety_threshold = direct_action_safety_threshold
        self.model.eval()

        if planner is not None:
            self.planner = planner
        else:
            self.planner = LatentLookaheadPlanner(
                model=model,
                horizon=horizon,
                gamma=gamma,
                hazard_weight=hazard_weight,
                hazard_prune_threshold=hazard_prune_threshold,
                dead_end_threshold=dead_end_threshold,
            )

        self._state: BrainState | None = None
        self._thought_noise: Tensor | None = None
        self._last_plan: LookaheadPlanResult | None = None
        self._plan_history: list[LookaheadPlanResult] = []
        self._decision_count = 0

    @property
    def last_plan(self) -> LookaheadPlanResult | None:
        """The most recent LookaheadPlanResult produced by the planner."""
        return self._last_plan

    @property
    def plan_history(self) -> tuple[LookaheadPlanResult, ...]:
        """Chronological tuple of all LookaheadPlanResult records in this episode."""
        return tuple(self._plan_history)

    def reset(self, episode_seed: int) -> None:
        """Reset internal recurrent state deterministically for a new episode."""
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")

        self._thought_noise = deterministic_eval_thought_noise(
            thoughtlets=self.model.config.thoughtlets,
            width=self.model.config.core_width,
            batch_size=1,
            device=self.device,
        ).to(dtype=self.dtype)

        self._state = self.model.initial_state(
            1,
            device=self.device,
            dtype=self.dtype,
            thought_noise=self._thought_noise,
        )
        self._last_plan = None
        self._plan_history.clear()
        self._decision_count = 0

    def decide(
        self,
        observation: Observation,
        elapsed_seconds: float = 1.0 / 60.0,
    ) -> tuple[GenericControl, Mapping[str, int | float], float | None]:
        """Compute the next control via latent lookahead planning with full audit stats."""
        if self._state is None:
            self.reset(0)

        resolution = getattr(self.model, "input_resolution", (32, 32))
        pixels = _rgb_tensor(
            (observation.rgb,),
            device=self.device,
            resolution=resolution,
        ).to(dtype=self.dtype)

        prev_vector = torch.tensor(
            [control_to_vector(observation.previous_control)],
            dtype=self.dtype,
            device=self.device,
        )
        elapsed_tensor = torch.tensor(
            [elapsed_seconds],
            dtype=self.dtype,
            device=self.device,
        )

        with torch.no_grad():
            # Ingest observation to produce latest sensory features
            sensors = self.model.pixel_encoder(pixels)

            # Evaluate model forward step to get policy prior logits and next recurrent state
            model_out = self.model(
                pixels,
                prev_vector,
                elapsed_tensor,
                self._state,
                thought_noise=self._thought_noise,
            )
            button_logits = model_out.action.button_logits[0]
            # DirectionalAction enum: NONE=0, W=1, A=2, S=3, D=4
            policy_logits = torch.tensor(
                [
                    0.0,
                    float(button_logits[_KEY_W].item()),
                    float(button_logits[_KEY_A].item()),
                    float(button_logits[_KEY_S].item()),
                    float(button_logits[_KEY_D].item()),
                ],
                device=self.device,
                dtype=self.dtype,
            )

            # Evaluate lookahead planner across candidate latent trajectories
            plan_result = self.planner.plan(
                state=self._state,
                sensors=sensors,
                policy_logits=policy_logits,
                policy_prior_weight=self.policy_prior_weight,
                elapsed_seconds=elapsed_seconds,
            )

            # Determine direct policy action
            d_act_idx = int(policy_logits.argmax().item())

            # Evaluate counterfactual danger if available
            cf_head = getattr(self.model, "counterfactual_foresight_head", None)
            direct_danger = 0.0
            if cf_head is not None:
                cf_preds = cf_head.forward_all_actions(model_out.next_state.thoughts)
                if d_act_idx in cf_preds:
                    direct_danger = float(cf_preds[d_act_idx].hazard_probability[0, 0, 0].item())

            # If selective gating is enabled and direct action is safe, preserve direct action
            if self.selective_gating and direct_danger < self.direct_action_safety_threshold:
                final_action = DirectionalAction(d_act_idx)
            else:
                final_action = plan_result.best_action

            self._state = model_out.next_state.detach()
            self._last_diagnostics = model_out.diagnostics
            self._last_model_output = model_out

        self._last_plan = plan_result
        self._plan_history.append(plan_result)
        self._decision_count += 1

        keys = _KEY_FOR_DIRECTIONAL_ACTION.get(final_action, ())
        control = GenericControl(keys_down=keys)

        stats = dict(control_audit_stats(control))
        stats["lookahead_utility"] = plan_result.best_branch.cumulative_utility
        stats["lookahead_pruned_count"] = plan_result.pruned_count
        stats["lookahead_best_reward"] = plan_result.best_branch.discounted_reward_sum
        stats["lookahead_best_danger"] = plan_result.best_branch.discounted_danger_sum
        stats["lookahead_intervened"] = 1.0 if int(final_action) != d_act_idx else 0.0
        stats["direct_danger"] = direct_danger

        terminal_val = plan_result.best_branch.terminal_value
        return control, stats, terminal_val

    def act(self, observation: Observation) -> GenericControl:
        """Standard policy act interface."""
        control, _, _ = self.decide(observation, elapsed_seconds=1.0 / 60.0)
        return control


__all__ = [
    "LatentLookaheadPolicy",
]
