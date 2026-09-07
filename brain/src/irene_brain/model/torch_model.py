"""First trainable PyTorch implementation of the Irene thought-field model.

Importing this module requires PyTorch.  The parent :mod:`irene_brain.model`
package keeps this import lazy so deterministic Phase 0 tooling remains usable
without a machine-learning runtime installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actuator import ActionPrediction, DirectActuatorReadout
from .adaptive_thought_gate import (
    AdaptiveHaltingController,
    AdaptiveThoughtUpdateGate,
    PredictiveFutureTrajectoryHead,
)
from .counterfactual_foresight import (
    ActionConditionedCounterfactualForesightHead,
    CounterfactualBranchOutput,
)
from .topological_goal import (
    TopologicalGoalFieldHead,
    TopologicalGoalPrediction,
)
from .brain_cell import (
    BrainCell,
    BrainCellCore,
    ContinuousTimeBlend,
    FactorizedLowRankProjection,
    ResidualCrossAttention,
)
from .sensory import PixelEncoder
from .spec import ThoughtFieldConfig


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters of a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


_THOUGHT_IDENTITY_SEED = 0x1A2B3C4D


def deterministic_thought_identity_codes(*, thoughtlets: int, width: int) -> Tensor:
    """Build one fixed Gaussian identity code per thought slot.

    A private CPU generator keeps model construction and inference from
    advancing PyTorch's global RNG stream.  The returned codes are ordinary
    buffers, not per-slot trainable parameters.
    """

    if any(type(value) is not int or value < 1 for value in (thoughtlets, width)):
        raise ValueError("thoughtlets and width must be positive integers")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_THOUGHT_IDENTITY_SEED)
    return torch.randn(
        thoughtlets,
        width,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


@dataclass(frozen=True, slots=True)
class BrainState:
    """Persistent state carried between observations without gradient updates."""

    belief: Tensor
    working_memory: Tensor
    thoughts: Tensor
    goal_context: Tensor
    thought_age_seconds: Tensor
    plastic_weights: Tensor | None = None
    prev_latent_pred: Tensor | None = None
    prev_reward_pred: Tensor | None = None
    prev_outcome_pred: Tensor | None = None

    @property
    def P_t(self) -> Tensor | None:
        return self.plastic_weights

    def detach(self) -> BrainState:
        return replace(
            self,
            belief=self.belief.detach(),
            working_memory=self.working_memory.detach(),
            thoughts=self.thoughts.detach(),
            goal_context=self.goal_context.detach(),
            thought_age_seconds=self.thought_age_seconds.detach(),
            plastic_weights=self.plastic_weights.detach() if self.plastic_weights is not None else None,
            prev_latent_pred=self.prev_latent_pred.detach() if self.prev_latent_pred is not None else None,
            prev_reward_pred=self.prev_reward_pred.detach() if self.prev_reward_pred is not None else None,
            prev_outcome_pred=self.prev_outcome_pred.detach() if self.prev_outcome_pred is not None else None,
        )

    def to(self, *args: object, **kwargs: object) -> BrainState:
        return replace(
            self,
            belief=self.belief.to(*args, **kwargs),
            working_memory=self.working_memory.to(*args, **kwargs),
            thoughts=self.thoughts.to(*args, **kwargs),
            goal_context=self.goal_context.to(*args, **kwargs),
            thought_age_seconds=self.thought_age_seconds.to(*args, **kwargs),
            plastic_weights=self.plastic_weights.to(*args, **kwargs) if self.plastic_weights is not None else None,
            prev_latent_pred=self.prev_latent_pred.to(*args, **kwargs) if self.prev_latent_pred is not None else None,
            prev_reward_pred=self.prev_reward_pred.to(*args, **kwargs) if self.prev_reward_pred is not None else None,
            prev_outcome_pred=self.prev_outcome_pred.to(*args, **kwargs) if self.prev_outcome_pred is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ThoughtPredictions:
    focus_logits: Tensor
    horizon_logits: Tensor
    candidate_action_logits: Tensor
    future_embedding: Tensor
    occurrence_logits: Tensor
    log_variance: Tensor
    urgency: Tensor
    utility: Tensor
    lifecycle_logits: Tensor
    memory_write_logits: Tensor


@dataclass(frozen=True, slots=True)
class ModelDiagnostics:
    routing_indices: tuple[Tensor, ...]
    routing_weights: tuple[Tensor, ...]
    thought_summaries: Tensor
    thought_cosine_similarity: Tensor
    actuator_thought_attention: Tensor
    applied_expire_probability: Tensor
    cycles_completed: int
    uses_pooled_integration_token: bool = False
    thought_update_gates: Tensor | None = None
    future_trajectory_predictions: dict[str, Tensor] | None = None
    halting_probabilities: tuple[Tensor, ...] = ()
    counterfactual_predictions: dict[int, CounterfactualBranchOutput] | None = None
    topological_goal_predictions: TopologicalGoalPrediction | None = None
    plastic_weights: Tensor | None = None
    surprise: Tensor | None = None
    sensory_surprise: Tensor | None = None
    consequence_surprise: Tensor | None = None
    reward_error: Tensor | None = None
    outcome_error: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ModelOutput:
    action: ActionPrediction
    anytime_actions: tuple[ActionPrediction, ...]
    value: Tensor
    world: ThoughtPredictions
    next_state: BrainState
    diagnostics: ModelDiagnostics


class ThoughtPredictionHead(nn.Module):
    def __init__(self, *, width: int, actuator_queries: int) -> None:
        super().__init__()
        self.focus_query = nn.Linear(width, width, bias=False)
        self.horizon = nn.Linear(width, 6)
        self.candidate_action = nn.Linear(width, actuator_queries)
        self.future = nn.Linear(width, width)
        self.occurrence = nn.Linear(width, 1)
        self.log_variance = nn.Linear(width, 1)
        self.urgency = nn.Linear(width, 1)
        self.utility = nn.Linear(width, 1)
        self.lifecycle = nn.Linear(width, 3)
        self.memory_write = nn.Linear(width, 1)

    def forward(self, thoughts: Tensor, sensors: Tensor, belief: Tensor) -> ThoughtPredictions:
        summaries = thoughts.mean(dim=2)
        focus_targets = torch.cat((sensors, belief), dim=1)
        focus_logits = torch.matmul(
            self.focus_query(summaries),
            focus_targets.transpose(-1, -2),
        ) / math.sqrt(summaries.shape[-1])
        return ThoughtPredictions(
            focus_logits=focus_logits,
            horizon_logits=self.horizon(summaries),
            candidate_action_logits=self.candidate_action(summaries),
            future_embedding=self.future(summaries),
            occurrence_logits=self.occurrence(summaries).squeeze(-1),
            log_variance=self.log_variance(summaries).squeeze(-1).clamp(-8.0, 8.0),
            urgency=torch.sigmoid(self.urgency(summaries).squeeze(-1)),
            utility=self.utility(summaries).squeeze(-1),
            lifecycle_logits=self.lifecycle(summaries),
            memory_write_logits=self.memory_write(summaries).squeeze(-1),
        )


class LatentRewardHead(nn.Module):
    """Predicts intermediate reward / consequence value from thought representations."""

    def __init__(self, *, width: int, hidden_width: int | None = None) -> None:
        super().__init__()
        hidden = hidden_width if hidden_width is not None else width
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, thought_summary: Tensor, action_feature: Tensor | None = None) -> Tensor:
        """Compute predicted scalar reward."""
        features = thought_summary if action_feature is None else thought_summary + action_feature
        return self.net(features).squeeze(-1)


class IreneBrainModel(nn.Module):
    """Streaming one-checkpoint model with persistent parallel thoughtlets.

    ``BrainCell`` is instantiated exactly once and called repeatedly, tying all
    recurrent weights across cognitive cycles. Thoughtlets are a batch axis of
    that shared cell, not separately parameterized experts.
    """

    architecture_variant_id = "irene.thought_field.routed.v1"
    architecture_variant_schema = 1

    def _build_brain_cell(self, *, width: int) -> nn.Module:
        return BrainCell(
            width=width,
            heads=self.config.attention_heads,
            routed_neighbors=self.config.routed_neighbors,
            blocks=self.config.brain_cell_blocks,
            dense_routing=getattr(self.config, "dense_routing", False),
            use_cgp=self.use_cgp,
            plastic_decay=getattr(self.config, "plastic_decay", 0.999),
            plastic_lr=getattr(self.config, "plastic_lr", 0.25),
        )

    def _communication_policy(self, cycle: int) -> tuple[bool, bool]:
        """Return peer-routing and thought-to-workspace permissions for a cycle."""

        return cycle > 0, True

    def __init__(
        self,
        config: ThoughtFieldConfig | None = None,
        *,
        input_resolution: tuple[int, int] = (32, 32),
        plan_steps: int = 3,
        enable_adaptive_cognition: bool = False,
        use_cgp: bool = False,
        tier: str | None = None,
    ) -> None:
        super().__init__()
        self.tier = tier
        if tier == "tier2" and config is None:
            config = ThoughtFieldConfig.tier2()
        self.config = config if config is not None else ThoughtFieldConfig.smoke()
        self.enable_adaptive_cognition = enable_adaptive_cognition
        self.use_cgp = bool(use_cgp or getattr(self.config, "use_cgp", False))
        if (
            not isinstance(input_resolution, tuple)
            or len(input_resolution) != 2
            or any(
                isinstance(side, bool) or not isinstance(side, int) or side < 8
                for side in input_resolution
            )
        ):
            raise ValueError("input_resolution must contain two integer sides of at least 8")
        self.input_resolution = input_resolution
        width = self.config.core_width

        self.pixel_encoder = PixelEncoder(
            width=width,
            sensor_tokens=self.config.sensor_tokens,
        )
        self.control_encoder = nn.Sequential(
            nn.Linear(self.config.actuator.total_queries, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.initial_belief = nn.Parameter(
            torch.empty(1, self.config.belief_tokens, width)
        )
        self.initial_working_memory = nn.Parameter(
            torch.empty(1, self.config.working_memory_tokens, width)
        )
        self.initial_thought_registers = nn.Parameter(
            torch.empty(1, 1, self.config.registers_per_thoughtlet, width)
        )
        self.initial_goal_context = nn.Parameter(
            torch.empty(1, self.config.goal_context_tokens, width)
        )
        self.register_buffer(
            "_thought_identity_codes",
            deterministic_thought_identity_codes(
                thoughtlets=self.config.thoughtlets,
                width=width,
            ),
            persistent=False,
        )
        self.noise_projection = nn.Linear(width, width, bias=False)
        self.seed_attention = ResidualCrossAttention(
            width=width,
            heads=self.config.attention_heads,
        )
        self.ingest_attention = ResidualCrossAttention(
            width=width,
            heads=self.config.attention_heads,
        )
        self.ingest_blend = ContinuousTimeBlend(width)

        # This is intentionally one object. The forward loop calls it C times.
        self.brain_cell = self._build_brain_cell(width=width)
        self.actuator = DirectActuatorReadout(
            width=width,
            heads=self.config.attention_heads,
            actuator=self.config.actuator,
            plan_steps=plan_steps,
        )
        self.thought_predictions = ThoughtPredictionHead(
            width=width,
            actuator_queries=self.config.actuator.total_queries,
        )
        self.value_per_thought = nn.Linear(width, 1)
        if enable_adaptive_cognition:
            self.adaptive_thought_gate = AdaptiveThoughtUpdateGate(width=width)
            self.future_trajectory_head = PredictiveFutureTrajectoryHead(width=width, horizons=(1, 3, 5))
            self.counterfactual_foresight_head = ActionConditionedCounterfactualForesightHead(width=width, horizons=(1, 3, 5))
            self.topological_goal_head = TopologicalGoalFieldHead(width=width, thoughtlets=self.config.thoughtlets)
            self.halting_controller = AdaptiveHaltingController(width=width, max_cycles=self.config.cognitive_cycles)
        else:
            self.adaptive_thought_gate = None
            self.future_trajectory_head = None
            self.counterfactual_foresight_head = None
            self.topological_goal_head = None
            self.halting_controller = None

        if self.use_cgp:
            self.reward_head = LatentRewardHead(width=width)
            self.latent_predictor = nn.Sequential(
                nn.Linear(width * 2, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            with torch.no_grad():
                nn.init.zeros_(self.latent_predictor[2].weight)
                nn.init.zeros_(self.latent_predictor[2].bias)
            num_buttons = (
                self.config.actuator.keyboard_keys
                + self.config.actuator.mouse_buttons
                + self.config.actuator.gamepad_buttons
            )
            self.plastic_action_projection = nn.Linear(width, num_buttons)
            with torch.no_grad():
                nn.init.zeros_(self.plastic_action_projection.weight)
                nn.init.zeros_(self.plastic_action_projection.bias)
        else:
            self.reward_head = None
            self.latent_predictor = None
            self.plastic_action_projection = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in (
            self.initial_belief,
            self.initial_working_memory,
            self.initial_thought_registers,
            self.initial_goal_context,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)

    @property
    def actuator_queries(self) -> int:
        return self.config.actuator.total_queries

    def _resolve_thought_noise(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        thought_noise: Tensor | None,
    ) -> Tensor:
        expected = (batch_size, self.config.thoughtlets, self.config.core_width)
        if thought_noise is None:
            return self._thought_identity_codes.to(device=device, dtype=dtype).unsqueeze(
                0
            ).expand(batch_size, -1, -1)
        if tuple(thought_noise.shape) != expected:
            raise ValueError(f"thought_noise must have shape {expected}")
        return thought_noise.to(device=device, dtype=dtype)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        thought_noise: Tensor | None = None,
    ) -> BrainState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        parameter = self.initial_belief
        actual_device = parameter.device if device is None else torch.device(device)
        actual_dtype = parameter.dtype if dtype is None else dtype
        thought_noise = self._resolve_thought_noise(
            batch_size=batch_size,
            device=actual_device,
            dtype=actual_dtype,
            thought_noise=thought_noise,
        )

        thoughts = self.initial_thought_registers.to(
            device=actual_device,
            dtype=actual_dtype,
        ).expand(
            batch_size,
            self.config.thoughtlets,
            self.config.registers_per_thoughtlet,
            self.config.core_width,
        )
        noise = self.noise_projection(thought_noise).unsqueeze(2)
        p_weights = None
        prev_latent = None
        prev_rew = None
        if self.use_cgp:
            width = self.config.core_width
            p_dim = getattr(self.brain_cell, "plastic_dim", width) or width
            p_weights = torch.zeros(batch_size, p_dim, device=actual_device, dtype=actual_dtype)
            prev_latent = torch.zeros(batch_size, width, device=actual_device, dtype=actual_dtype)
            prev_rew = torch.zeros(batch_size, device=actual_device, dtype=actual_dtype)
        return BrainState(
            belief=self.initial_belief.to(device=actual_device, dtype=actual_dtype).expand(
                batch_size, -1, -1
            ),
            working_memory=self.initial_working_memory.to(
                device=actual_device,
                dtype=actual_dtype,
            ).expand(batch_size, -1, -1),
            thoughts=thoughts + noise,
            goal_context=self.initial_goal_context.to(
                device=actual_device,
                dtype=actual_dtype,
            ).expand(batch_size, -1, -1),
            thought_age_seconds=torch.zeros(
                batch_size,
                self.config.thoughtlets,
                device=actual_device,
                dtype=actual_dtype,
            ),
            plastic_weights=p_weights,
            prev_latent_pred=prev_latent,
            prev_reward_pred=prev_rew,
            prev_outcome_pred=None,
        )

    def _validate_state(
        self,
        state: BrainState,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        layout = self.config.state_layout(batch_size=batch)
        expected = {
            "belief": layout.belief.dimensions,
            "working_memory": layout.working_memory.dimensions,
            "thoughts": layout.thought_field.dimensions,
            "goal_context": layout.goal_context.dimensions,
            "thought_age_seconds": (batch, self.config.thoughtlets),
        }
        for name, shape in expected.items():
            tensor = getattr(state, name)
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"state.{name} must have shape {shape}")
            if tensor.device != device:
                raise ValueError(f"state.{name} must be on {device}")
            if tensor.dtype != dtype:
                raise ValueError(f"state.{name} must use dtype {dtype}")
        if state.plastic_weights is not None:
            if not isinstance(state.plastic_weights, Tensor):
                raise ValueError("state.plastic_weights must be a Tensor")
            if state.plastic_weights.shape[0] != batch:
                raise ValueError(
                    f"state.plastic_weights batch size mismatch: expected {batch}, got {state.plastic_weights.shape[0]}"
                )
            if state.plastic_weights.device != device:
                raise ValueError(f"state.plastic_weights must be on {device}")
            if state.plastic_weights.dtype != dtype:
                raise ValueError(f"state.plastic_weights must use dtype {dtype}")

    def _normalize_elapsed(self, elapsed_seconds: Tensor, *, batch: int, pixels: Tensor) -> Tensor:
        if elapsed_seconds.ndim == 1:
            elapsed_seconds = elapsed_seconds.unsqueeze(-1)
        if tuple(elapsed_seconds.shape) != (batch, 1):
            raise ValueError("elapsed_seconds must have shape [batch] or [batch, 1]")
        elapsed_seconds = elapsed_seconds.to(device=pixels.device, dtype=pixels.dtype)
        # Avoid a host/device synchronization in every accelerator training
        # step. The tensorization pipeline owns this invariant on accelerators;
        # eager CPU callers still receive a precise boundary error.
        if elapsed_seconds.device.type == "cpu":
            if not bool(torch.isfinite(elapsed_seconds).all()) or bool(
                (elapsed_seconds < 0).any()
            ):
                raise ValueError("elapsed_seconds must be finite and nonnegative")
        return elapsed_seconds

    def _seed_thoughts(
        self,
        *,
        batch_size: int,
        sensors: Tensor,
        belief: Tensor,
        thought_noise: Tensor | None,
    ) -> Tensor:
        """Compute freshly seeded thought registers from current sensors/belief."""

        thoughtlets = self.config.thoughtlets
        registers = self.config.registers_per_thoughtlet
        width = self.config.core_width
        thought_noise = self._resolve_thought_noise(
            batch_size=batch_size,
            device=sensors.device,
            dtype=sensors.dtype,
            thought_noise=thought_noise,
        )
        seed_query = self.initial_thought_registers.to(dtype=sensors.dtype).expand(
            batch_size,
            thoughtlets,
            registers,
            width,
        ) + self.noise_projection(thought_noise).unsqueeze(2)
        seed_context = torch.cat((sensors, belief), dim=1)
        seed_context = (
            seed_context.unsqueeze(1)
            .expand(batch_size, thoughtlets, seed_context.shape[1], width)
            .reshape(batch_size * thoughtlets, seed_context.shape[1], width)
        )
        return self.seed_attention(
            seed_query.reshape(batch_size * thoughtlets, registers, width),
            seed_context,
        ).reshape(batch_size, thoughtlets, registers, width)

    def _refresh_thoughts(
        self,
        *,
        thoughts: Tensor,
        sensors: Tensor,
        belief: Tensor,
        elapsed_seconds: Tensor,
        thought_age_seconds: Tensor,
        thought_noise: Tensor | None,
        surprise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, thoughtlets, registers, width = thoughts.shape
        seeds = self._seed_thoughts(
            batch_size=batch,
            sensors=sensors,
            belief=belief,
            thought_noise=thought_noise,
        )
        if self.adaptive_thought_gate is not None:
            refreshed, alphas = self.adaptive_thought_gate(
                thoughts=thoughts,
                seeds=seeds,
                sensors=sensors,
                belief=belief,
            )
            lifecycle_logits = self.thought_predictions.lifecycle(thoughts.mean(dim=2))
            legacy_expire_prob = torch.softmax(lifecycle_logits, dim=-1)[..., 2]
            elapsed = elapsed_seconds.expand(batch, thoughtlets)
            ages = torch.where(alphas < 0.5, thought_age_seconds + elapsed, elapsed)
            return refreshed, ages, legacy_expire_prob
        else:
            lifecycle_logits = self.thought_predictions.lifecycle(thoughts.mean(dim=2))
            expire_probability = torch.softmax(lifecycle_logits, dim=-1)[..., 2:3]
            if self.use_cgp and surprise is not None:
                # Consequence-gated persistence: thoughts only expire under consequence surprise.
                # During corridor traversal or blank occlusions, surprise ~ 0, preserving thoughts.
                gate = torch.tanh(surprise)
                if gate.ndim == 2:
                    gate = gate.unsqueeze(1)
                expire_probability = expire_probability * gate
            keep = 1.0 - expire_probability
            refreshed = keep.unsqueeze(-1) * thoughts + (1.0 - keep.unsqueeze(-1)) * seeds
            elapsed = elapsed_seconds.expand(batch, thoughtlets)
            ages = torch.where(keep.squeeze(-1) >= 0.5, thought_age_seconds + elapsed, elapsed)
            return refreshed, ages, expire_probability.squeeze(-1)

    def forward(
        self,
        pixels: Tensor,
        previous_control: Tensor,
        elapsed_seconds: Tensor,
        state: BrainState | None = None,
        *,
        actual_reward: Tensor | None = None,
        actual_outcomes: Tensor | None = None,
        max_cycles: int | None = None,
        thought_noise: Tensor | None = None,
        retrieved_memory: Tensor | None = None,
    ) -> ModelOutput:
        if pixels.ndim != 4:
            raise ValueError("pixels must have shape [batch, 3, height, width]")
        batch, channels, height, width = pixels.shape
        if (channels, height, width) != (3, *self.input_resolution):
            raise ValueError(f"pixels must have shape [batch, 3, {self.input_resolution[0]}, {self.input_resolution[1]}]")
        if previous_control.ndim != 2 or tuple(previous_control.shape) != (
            batch,
            self.config.actuator.total_queries,
        ):
            raise ValueError(f"previous_control must have shape [batch, {self.config.actuator.total_queries}]")

        cycles = self.config.cognitive_cycles if max_cycles is None else max_cycles
        if type(cycles) is not int or cycles < 1:
            raise ValueError("cycles must be a positive integer")

        elapsed = self._normalize_elapsed(elapsed_seconds, batch=batch, pixels=pixels)
        if state is None:
            state = self.initial_state(
                batch,
                device=pixels.device,
                dtype=pixels.dtype,
                thought_noise=thought_noise,
            )
        self._validate_state(
            state,
            batch=batch,
            device=pixels.device,
            dtype=pixels.dtype,
        )

        cur_plastic_weights = state.plastic_weights if state is not None else None
        sensory_surprise_tensor = None
        consequence_surprise_tensor = None
        surprise_tensor = None
        reward_error_tensor = None
        outcome_error_tensor = None

        sensors = self.pixel_encoder(pixels)
        if self.use_cgp and self.reward_head is not None:
            sensory_summary = sensors.mean(dim=1)  # [batch, width]

            # 1. Sensory surprise: ||z_t - \hat{z}_{t|t-1}||
            if state is not None and state.prev_latent_pred is not None:
                sensory_err = torch.norm(sensory_summary - state.prev_latent_pred, dim=-1, keepdim=True)
            else:
                sensory_err = torch.zeros(batch, 1, device=pixels.device, dtype=pixels.dtype)
            sensory_surprise_tensor = sensory_err

            # 2. Reward prediction error: |r_t - \hat{r}_{t|t-1}|
            if actual_reward is not None and state is not None and state.prev_reward_pred is not None:
                rew_target = actual_reward.to(device=pixels.device, dtype=pixels.dtype)
                if rew_target.ndim == 1:
                    rew_target = rew_target.unsqueeze(-1)
                pred_r = state.prev_reward_pred
                if pred_r.ndim == 1:
                    pred_r = pred_r.unsqueeze(-1)
                reward_err = torch.abs(rew_target - pred_r)
            else:
                reward_err = torch.zeros(batch, 1, device=pixels.device, dtype=pixels.dtype)
            reward_error_tensor = reward_err

            # 3. Outcome prediction error: ||m_t - \hat{m}_{t|t-1}||
            if actual_outcomes is not None and state is not None and state.prev_outcome_pred is not None:
                outcomes_target = actual_outcomes.to(device=pixels.device, dtype=pixels.dtype)
                outcome_err = torch.norm(outcomes_target - state.prev_outcome_pred, dim=-1, keepdim=True)
            else:
                outcome_err = torch.zeros(batch, 1, device=pixels.device, dtype=pixels.dtype)
            outcome_error_tensor = outcome_err

            # 4. Consequence surprise: strictly outcome + reward error.
            # INVARIANT: If reward_err == 0 and outcome_err == 0, consequence surprise == 0!
            # Pure sensory noise does NOT enter consequence surprise!
            consequence_surprise_tensor = reward_err + 1.0 * outcome_err
            surprise_tensor = consequence_surprise_tensor

        control_token = self.control_encoder(previous_control).unsqueeze(1)
        time_input = torch.log1p(elapsed * 1_000.0)
        time_token = self.time_encoder(time_input).unsqueeze(1)
        action_time_tokens = torch.cat((control_token, time_token), dim=1)

        ingest_context = torch.cat(
            (sensors, action_time_tokens, state.belief, state.working_memory),
            dim=1,
        )
        belief_proposal = self.ingest_attention(state.belief, ingest_context)
        belief = self.ingest_blend(state.belief, belief_proposal, elapsed)
        try:
            thoughts, thought_ages, applied_expire_probability = self._refresh_thoughts(
                thoughts=state.thoughts,
                sensors=sensors,
                belief=belief,
                elapsed_seconds=elapsed,
                thought_age_seconds=state.thought_age_seconds,
                thought_noise=thought_noise,
                surprise=consequence_surprise_tensor,
            )
        except TypeError:
            thoughts, thought_ages, applied_expire_probability = self._refresh_thoughts(
                thoughts=state.thoughts,
                sensors=sensors,
                belief=belief,
                elapsed_seconds=elapsed,
                thought_age_seconds=state.thought_age_seconds,
                thought_noise=thought_noise,
            )
        working_memory = state.working_memory
        goal_context = state.goal_context

        retrieval_shape = (
            batch,
            self.config.thoughtlets,
            self.config.retrieved_entries_per_thoughtlet,
            self.config.core_width,
        )
        if retrieved_memory is None:
            retrieved_memory = pixels.new_zeros(retrieval_shape)
        elif tuple(retrieved_memory.shape) != retrieval_shape:
            raise ValueError(f"retrieved_memory must have shape {retrieval_shape}")
        else:
            retrieved_memory = retrieved_memory.to(device=pixels.device, dtype=pixels.dtype)

        exits: list[ActionPrediction] = [
            self.actuator(
                sensors=sensors,
                belief=belief,
                thoughts=thoughts,
                working_memory=working_memory,
                retrieved_memory=retrieved_memory,
                goal_context=goal_context,
            )
        ]

        routing_indices: list[Tensor] = []
        routing_weights: list[Tensor] = []
        halting_probabilities: list[Tensor] = []

        for cycle in range(cycles):
            allow_routing, allow_workspace_writes = self._communication_policy(cycle)
            cell_kwargs = {
                "belief": belief,
                "working_memory": working_memory,
                "thoughts": thoughts,
                "sensors": sensors,
                "action_time_tokens": action_time_tokens,
                "goal_context": goal_context,
                "retrieved_memory": retrieved_memory,
                "elapsed_seconds": elapsed,
                "allow_routing": allow_routing,
                "allow_workspace_writes": allow_workspace_writes,
            }
            if self.use_cgp:
                cell_kwargs["plastic_weights"] = cur_plastic_weights
                cell_kwargs["surprise"] = consequence_surprise_tensor
            cell_out = self.brain_cell(**cell_kwargs)
            belief, working_memory, thoughts, cycle_routing = cell_out[:4]
            cur_plastic_weights = getattr(cell_out, "plastic_weights", cur_plastic_weights)
            routing_indices.extend(routing.indices for routing in cycle_routing)
            routing_weights.extend(routing.weights for routing in cycle_routing)
            if self.halting_controller is not None:
                halt_prob, _ = self.halting_controller(thoughts, cycle)
                halting_probabilities.append(halt_prob)
            exits.append(
                self.actuator(
                    sensors=sensors,
                    belief=belief,
                    thoughts=thoughts,
                    working_memory=working_memory,
                    retrieved_memory=retrieved_memory,
                    goal_context=goal_context,
                )
            )

        next_latent_pred = None
        next_reward_pred = None
        if self.use_cgp and self.reward_head is not None and self.latent_predictor is not None:
            post_summary = thoughts.mean(dim=(1, 2))
            next_reward_pred = self.reward_head(post_summary).detach()
            pred_in = torch.cat([post_summary, control_token.squeeze(1)], dim=-1)
            next_latent_pred = (post_summary + self.latent_predictor(pred_in)).detach()

        next_state = BrainState(
            belief=belief,
            working_memory=working_memory,
            thoughts=thoughts,
            goal_context=goal_context,
            thought_age_seconds=thought_ages,
            plastic_weights=cur_plastic_weights,
            prev_latent_pred=next_latent_pred,
            prev_reward_pred=next_reward_pred,
        )
        world = self.thought_predictions(thoughts, sensors, belief)
        future_trajectories = (
            self.future_trajectory_head(thoughts)
            if self.future_trajectory_head is not None
            else None
        )
        counterfactual_preds = (
            self.counterfactual_foresight_head.forward_all_actions(thoughts)
            if getattr(self, "counterfactual_foresight_head", None) is not None
            else None
        )
        topo_preds = (
            self.topological_goal_head(thoughts)
            if getattr(self, "topological_goal_head", None) is not None
            else None
        )
        summaries = thoughts.mean(dim=2)
        normalized = F.normalize(summaries, dim=-1, eps=1e-6)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        value = self.value_per_thought(summaries).mean(dim=1).squeeze(-1)

        final_action = exits[-1]
        if self.use_cgp and self.plastic_action_projection is not None and cur_plastic_weights is not None:
            plastic_bias = self.plastic_action_projection(cur_plastic_weights)
            final_action = replace(
                final_action,
                button_logits=final_action.button_logits + plastic_bias,
            )

        diagnostics = ModelDiagnostics(
            routing_indices=tuple(routing_indices),
            routing_weights=tuple(routing_weights),
            thought_summaries=summaries,
            thought_cosine_similarity=similarity,
            actuator_thought_attention=exits[-1].thought_attention,
            applied_expire_probability=applied_expire_probability,
            cycles_completed=cycles,
            thought_update_gates=applied_expire_probability,
            future_trajectory_predictions=future_trajectories,
            halting_probabilities=tuple(halting_probabilities),
            counterfactual_predictions=counterfactual_preds,
            topological_goal_predictions=topo_preds,
            plastic_weights=cur_plastic_weights,
            surprise=surprise_tensor,
            sensory_surprise=sensory_surprise_tensor,
            consequence_surprise=consequence_surprise_tensor,
            reward_error=reward_error_tensor,
            outcome_error=outcome_error_tensor,
        )
        return ModelOutput(
            action=final_action,
            anytime_actions=tuple(exits),
            value=value,
            world=world,
            next_state=next_state,
            diagnostics=diagnostics,
        )


class Tier2BrainModel(nn.Module):
    """Tier 2 (~50M Parameter) Pseudo-Brain Model.

    Specifies and implements the Three Architectural Laws of Cognitive Scaling:
    1. Law 1 (Slot Width Clamping): Slot width bounded at W=64, K=128 slots.
       Total recurrent state is K*W = 8,192 floats (32 KB state memory, avoiding the
       53k state explosion seen when W was 832).
    2. Law 2 (Factorized Deep Projections): Input and hidden projections factorized:
       W=64 -> rank r=32 -> proj_dim=4096 (delivers the parametric capacity of Tier 2
       ~50M without parameter or compute explosion).
    3. Law 3 (Event-Driven Dynamic Sparsity): Slot evaluation bypassed when
       Cognitive Input Gate salience s_k < epsilon_dormant (0.05).
    """

    def __init__(
        self,
        input_dim: int = 64,
        K: int = 128,
        thought_size: int = 64,
        rank: int = 32,
        proj_dim: int = 4096,
        num_values: int = 8,
        epsilon_dormant: float = 0.05,
        conditional_recurrence: bool = True,
        tier: str = "tier2",
        num_deep_layers: int = 3,
    ) -> None:
        super().__init__()
        self.tier = tier
        self.K = K
        self.thought_size = thought_size
        self.rank = rank
        self.proj_dim = proj_dim
        self.input_dim = input_dim
        self.num_values = num_values
        self.epsilon_dormant = epsilon_dormant
        self.conditional_recurrence = conditional_recurrence

        # Sensory/Input Projection to proj_dim
        self.input_proj = nn.Linear(input_dim, proj_dim)

        # Law 2: Factorized Deep Projections Core (~50M parameters)
        self.brain_cell = BrainCellCore(
            input_size=proj_dim,
            thought_size=thought_size,
            rank=rank,
            proj_dim=proj_dim,
            tier=tier,
            num_deep_layers=num_deep_layers,
        )

        # Cognitive Input Gate (CIG)
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + thought_size, 1),
            nn.Sigmoid(),
        )

        # Thread-Targeted Factorized Readout Head (W=64 -> r=32 -> proj_dim // 4 -> num_values)
        self.slot_head = nn.Sequential(
            FactorizedLowRankProjection(thought_size, proj_dim // 4, rank=rank),
            nn.GELU(),
            nn.Linear(proj_dim // 4, num_values),
        )

        # Deterministic Thought Slot Identity Codes
        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K, width=thought_size),
            persistent=False,
        )

    def state_bytes(self, bytes_per_element: int = 4) -> int:
        """Law 1 Contract: Returns exact recurrent state memory footprint in bytes (32 KB)."""
        return self.K * self.thought_size * bytes_per_element

    def count_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        return count_parameters(self)

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        """Generate initial recurrent state tensor [B, K, W]."""
        dev = torch.device(device)
        return self.slot_identities.to(device=dev).unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward(
        self,
        x_seq: Tensor,
        active_thread_id: Tensor | None = None,
    ) -> Tensor:
        """Run recurrent forward unroll across time steps.

        Args:
            x_seq: [B, T, input_dim] or [B, input_dim] input sensory tensor.
            active_thread_id: optional [B, T] or [B] indicating addressed thread.
        """
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1)

        B, T, D = x_seq.shape
        dev = x_seq.device

        x_proj = self.input_proj(x_seq)  # [B, T, proj_dim]
        thoughts = self.slot_identities.to(device=dev).unsqueeze(0).expand(B, -1, -1).clone()
        logits_list = []

        for t in range(T):
            x_in = x_proj[:, t]  # [B, proj_dim]
            x_exp = x_in.unsqueeze(1).expand(-1, self.K, -1)  # [B, K, proj_dim]

            # CIG Gate
            gate_in = torch.cat([x_exp, thoughts], dim=-1)
            salience = self.cig_gate(gate_in)

            # Law 3: Event-Driven Dynamic Sparsity
            if self.conditional_recurrence and self.epsilon_dormant > 0.0:
                thoughts, _ = self.brain_cell.forward_conditional(
                    thoughts=thoughts,
                    x=x_exp,
                    salience=salience,
                    epsilon_dormant=self.epsilon_dormant,
                )
            else:
                t_flat = thoughts.reshape(B * self.K, self.thought_size)
                x_flat = x_exp.reshape(B * self.K, -1)
                new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)
                thoughts = (1.0 - salience) * thoughts + salience * new_t

            # Readout on queried slot
            if active_thread_id is not None:
                tid = active_thread_id[:, t] if active_thread_id.dim() > 1 else active_thread_id
                batch_idx = torch.arange(B, device=dev)
                queried_slot = thoughts[batch_idx, tid]
            else:
                queried_slot = thoughts[:, 0]

            logits = self.slot_head(queried_slot)
            logits_list.append(logits)

        out = torch.stack(logits_list, dim=1)
        if out.shape[1] == 1:
            return out.squeeze(1)
        return out


def make_tier_model(tier: str = "tier2", **kwargs: object) -> nn.Module:
    """Factory creating parameter-matched models for Tier 0, Tier 1, or Tier 2."""
    if tier == "tier2":
        return Tier2BrainModel(tier=tier, **kwargs)
    elif tier == "tier1":
        return IreneBrainModel(tier=tier, **kwargs)
    elif tier == "tier0":
        return IreneBrainModel(tier=tier, **kwargs)
    else:
        return Tier2BrainModel(tier=tier, **kwargs)


__all__ = [
    "ActionPrediction",
    "BrainState",
    "count_parameters",
    "deterministic_thought_identity_codes",
    "IreneBrainModel",
    "LatentRewardHead",
    "make_tier_model",
    "ModelDiagnostics",
    "ModelOutput",
    "ThoughtPredictions",
    "Tier2BrainModel",
]
