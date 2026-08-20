"""Latent World-Model Rollouts and Dynamic Lookahead Planning engine.

Unrolls cognitive thoughtlet and belief states forward H steps (H in [1, 5])
through internal BrainCell transition dynamics without rendering full pixel images.
Evaluates candidate action sequences using cumulative branch utility:
    U(a_vec) = sum_{k=0}^{H-1} gamma^k * (r_hat_{t+k} - lambda * d_hat_{t+k}) + gamma^H * V_hat(z_{t+H})
and prunes branches that lead to ghost collisions or dead ends.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..types import HidKey
from .brain_cell import BrainCell
from .intent import DirectionalAction
from .spec import ThoughtFieldConfig
from .torch_model import BrainState, IreneBrainModel


# Action indices in the 307-dimensional control vector
_KEY_W_INDEX = int(HidKey.W)
_KEY_A_INDEX = int(HidKey.A)
_KEY_S_INDEX = int(HidKey.S)
_KEY_D_INDEX = int(HidKey.D)

_OPPOSITE_DIRECTIONS = frozenset(
    {
        (DirectionalAction.W, DirectionalAction.S),
        (DirectionalAction.S, DirectionalAction.W),
        (DirectionalAction.A, DirectionalAction.D),
        (DirectionalAction.D, DirectionalAction.A),
    }
)


def directional_to_control_vector(
    action: DirectionalAction | int,
    total_queries: int = 307,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert a DirectionalAction into a 307-d generic HID control vector."""
    vector = torch.zeros(total_queries, dtype=dtype, device=device)
    if isinstance(action, int) and not isinstance(action, DirectionalAction):
        action = DirectionalAction(action)

    if action == DirectionalAction.W:
        vector[_KEY_W_INDEX] = 1.0
    elif action == DirectionalAction.A:
        vector[_KEY_A_INDEX] = 1.0
    elif action == DirectionalAction.S:
        vector[_KEY_S_INDEX] = 1.0
    elif action == DirectionalAction.D:
        vector[_KEY_D_INDEX] = 1.0
    return vector


def generate_directional_candidate_sequences(
    horizon: int,
    *,
    include_none: bool = False,
    prune_reversals: bool = True,
) -> list[tuple[DirectionalAction, ...]]:
    """Generate candidate action sequences of length `horizon`.

    Args:
        horizon: Unroll depth H in [1, 5].
        include_none: Whether to include DirectionalAction.NONE in candidate moves.
        prune_reversals: Whether to eliminate immediate back-and-forth oscillations
                         (e.g., W followed immediately by S).
    """
    if horizon < 1 or horizon > 5:
        raise ValueError(f"horizon must be between 1 and 5, got {horizon}")

    actions = [
        DirectionalAction.W,
        DirectionalAction.A,
        DirectionalAction.S,
        DirectionalAction.D,
    ]
    if include_none:
        actions.insert(0, DirectionalAction.NONE)

    all_sequences = list(itertools.product(actions, repeat=horizon))
    if not prune_reversals or horizon <= 1:
        return all_sequences

    valid_sequences: list[tuple[DirectionalAction, ...]] = []
    for seq in all_sequences:
        reversal = False
        for step in range(len(seq) - 1):
            pair = (seq[step], seq[step + 1])
            if pair in _OPPOSITE_DIRECTIONS:
                reversal = True
                break
        if not reversal:
            valid_sequences.append(seq)

    return valid_sequences


@dataclass(frozen=True, slots=True)
class LookaheadRolloutStep:
    """Single transition step within an unrolled candidate branch."""

    step_index: int
    action: DirectionalAction | int
    predicted_reward: float
    predicted_danger: float
    predicted_value: float
    is_hazard: bool


@dataclass(frozen=True, slots=True)
class LookaheadBranchResult:
    """Result of unrolling a full candidate action sequence forward in latent space."""

    action_sequence: tuple[DirectionalAction | int, ...]
    rollout_steps: tuple[LookaheadRolloutStep, ...]
    cumulative_utility: float
    discounted_reward_sum: float
    discounted_danger_sum: float
    terminal_value: float
    is_pruned: bool
    prune_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LookaheadPlanResult:
    """Final decision result produced by the LatentLookaheadPlanner."""

    best_action: DirectionalAction
    best_action_control: Tensor
    best_branch: LookaheadBranchResult
    all_branches: tuple[LookaheadBranchResult, ...]
    branch_utilities: tuple[float, ...]
    chosen_index: int
    pruned_count: int


class LatentHazardHead(nn.Module):
    """Predicts collision hazard / ghost danger from thoughtlet state and action."""

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
        """Compute danger hazard probability in [0, 1].

        Args:
            thought_summary: [batch, width] pooled thought summary.
            action_feature: Optional [batch, width] control representation.
        """
        features = thought_summary if action_feature is None else thought_summary + action_feature
        logits = self.net(features).squeeze(-1)
        return torch.sigmoid(logits)


class LatentRewardHead(nn.Module):
    """Predicts intermediate reward (pellet collection / target approach)."""

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


class LatentSensoryTransition(nn.Module):
    """Updates latent sensory tokens during unrolls without rendering pixel images."""

    def __init__(self, *, width: int, sensor_tokens: int) -> None:
        super().__init__()
        self.width = width
        self.sensor_tokens = sensor_tokens
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sensors: Tensor, thought_summary: Tensor) -> Tensor:
        """Update sensor tokens residually using cognitive thought context."""
        delta = self.projection(thought_summary).unsqueeze(1)
        return sensors + 0.1 * delta


class LatentLookaheadPlanner(nn.Module):
    """Multi-step Latent World-Model Lookahead Planner for Pseudo-Brain.

    Features:
    - Zero-cheating: Operates purely on internal thoughtlet, belief, and sensory representations.
    - Zero image rendering: Unrolls H steps (H in [1, 5]) in latent space.
    - Hazard-aware branch evaluation: Penalizes and prunes paths leading to collisions or dead ends.
    - Immutable evaluation: Does not mutate the model's persistent BrainState during planning.
    """

    def __init__(
        self,
        *,
        model: IreneBrainModel | None = None,
        config: ThoughtFieldConfig | None = None,
        horizon: int = 3,
        gamma: float = 0.95,
        hazard_weight: float = 8.0,
        hazard_prune_threshold: float = 0.90,
        dead_end_threshold: float = -15.0,
        prune_reversals: bool = True,
        cognitive_cycles_per_step: int = 1,
    ) -> None:
        super().__init__()
        if horizon < 1 or horizon > 5:
            raise ValueError(f"horizon must be an integer between 1 and 5, got {horizon}")
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if hazard_weight < 0.0:
            raise ValueError(f"hazard_weight must be nonnegative, got {hazard_weight}")
        if hazard_prune_threshold <= 0.0 or hazard_prune_threshold > 1.0:
            raise ValueError(
                f"hazard_prune_threshold must be in (0, 1], got {hazard_prune_threshold}"
            )

        self.config = config if config is not None else (
            model.config if model is not None else ThoughtFieldConfig.smoke()
        )
        self.horizon = horizon
        self.gamma = gamma
        self.hazard_weight = hazard_weight
        self.hazard_prune_threshold = hazard_prune_threshold
        self.dead_end_threshold = dead_end_threshold
        self.prune_reversals = prune_reversals
        self.cognitive_cycles_per_step = max(1, cognitive_cycles_per_step)

        width = self.config.core_width
        self.width = width
        self.total_queries = self.config.actuator.total_queries

        # Share or build submodules
        if model is not None:
            self.brain_cell = model.brain_cell
            self.control_encoder = model.control_encoder
            self.time_encoder = model.time_encoder
            self.value_head = model.value_per_thought
            self.counterfactual_foresight_head = getattr(model, "counterfactual_foresight_head", None)
            self.topological_goal_head = getattr(model, "topological_goal_head", None)
        else:
            self.brain_cell = BrainCell(
                width=width,
                heads=self.config.attention_heads,
                routed_neighbors=self.config.routed_neighbors,
                blocks=self.config.brain_cell_blocks,
            )
            self.control_encoder = nn.Sequential(
                nn.Linear(self.total_queries, width),
                nn.LayerNorm(width),
                nn.SiLU(),
            )
            self.time_encoder = nn.Sequential(
                nn.Linear(1, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.LayerNorm(width),
            )
            self.value_head = nn.Linear(width, 1)
            self.counterfactual_foresight_head = None
            self.topological_goal_head = None
        self.hazard_head = LatentHazardHead(width=width)
        self.reward_head = LatentRewardHead(width=width)
        self.sensory_transition = LatentSensoryTransition(
            width=width,
            sensor_tokens=self.config.sensor_tokens,
        )
        if model is not None:
            try:
                device = next(model.parameters()).device
                self.to(device)
            except (StopIteration, RuntimeError):
                pass

    def to(self, device: torch.device | str) -> LatentLookaheadPlanner:
        self.hazard_head.to(device)
        self.reward_head.to(device)
        self.sensory_transition.to(device)
        return self

    def unroll_latent_step(
        self,
        *,
        belief: Tensor,
        working_memory: Tensor,
        thoughts: Tensor,
        sensors: Tensor,
        goal_context: Tensor,
        retrieved_memory: Tensor,
        action_vector: Tensor,
        elapsed_seconds: Tensor,
        action_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Execute one forward latent transition step.

        Returns:
            (next_belief, next_working_memory, next_thoughts, next_sensors,
             predicted_reward, predicted_danger, predicted_value)
        """
        control_token = self.control_encoder(action_vector).unsqueeze(1)
        time_input = torch.log1p(elapsed_seconds * 1_000.0)
        time_token = self.time_encoder(time_input).unsqueeze(1)
        action_time_tokens = torch.cat((control_token, time_token), dim=1)

        cur_belief = belief
        cur_memory = working_memory
        cur_thoughts = thoughts

        for cycle in range(self.cognitive_cycles_per_step):
            allow_routing = cycle > 0 or self.cognitive_cycles_per_step == 1
            cur_belief, cur_memory, cur_thoughts, _ = self.brain_cell(
                belief=cur_belief,
                working_memory=cur_memory,
                thoughts=cur_thoughts,
                sensors=sensors,
                action_time_tokens=action_time_tokens,
                goal_context=goal_context,
                retrieved_memory=retrieved_memory,
                elapsed_seconds=elapsed_seconds,
                allow_routing=allow_routing,
                allow_workspace_writes=True,
            )

        thought_summary = cur_thoughts.mean(dim=2).mean(dim=1)  # [batch, width]
        action_feature = control_token.squeeze(1)

        reward_pred = self.reward_head(thought_summary, action_feature)
        if self.counterfactual_foresight_head is not None and action_indices is not None:
            cf_out = self.counterfactual_foresight_head.forward_single_action(thought_summary, action_indices)
            danger_pred = cf_out.hazard_probability[:, 0, 0]
        else:
            danger_pred = self.hazard_head(thought_summary, action_feature)

        summaries = cur_thoughts.mean(dim=2)
        value_pred = self.value_head(summaries).mean(dim=1).squeeze(-1)

        next_sensors = self.sensory_transition(sensors, thought_summary)

        return (
            cur_belief,
            cur_memory,
            cur_thoughts,
            next_sensors,
            reward_pred,
            danger_pred,
            value_pred,
        )

    def evaluate_candidate_branches(
        self,
        *,
        state: BrainState,
        sensors: Tensor,
        candidate_sequences: Sequence[Sequence[DirectionalAction | int]],
        elapsed_seconds: float = 1.0 / 60.0,
    ) -> tuple[LookaheadBranchResult, ...]:
        """Evaluate a set of candidate action sequences in parallel.

        Guarantees:
        - `state` is treated as strictly read-only and never mutated.
        - Evaluation runs in `torch.no_grad()` without computational graph leaks.
        """
        num_branches = len(candidate_sequences)
        if num_branches == 0:
            raise ValueError("candidate_sequences cannot be empty")

        device = state.belief.device
        dtype = state.belief.dtype
        H = len(candidate_sequences[0])

        # Validate consistent horizon length
        for seq in candidate_sequences:
            if len(seq) != H:
                raise ValueError(f"all candidate sequences must have length {H}, got {len(seq)}")

        # Replicate initial states across all candidate branches: [M, ...]
        belief = state.belief.expand(num_branches, -1, -1).clone()
        working_memory = state.working_memory.expand(num_branches, -1, -1).clone()
        thoughts = state.thoughts.expand(num_branches, -1, -1, -1).clone()
        goal_context = state.goal_context.expand(num_branches, -1, -1).clone()
        cur_sensors = sensors.expand(num_branches, -1, -1).clone()

        retrieval_shape = (
            num_branches,
            self.config.thoughtlets,
            self.config.retrieved_entries_per_thoughtlet,
            self.config.core_width,
        )
        retrieved_memory = torch.zeros(retrieval_shape, dtype=dtype, device=device)
        elapsed_tensor = torch.full((num_branches, 1), elapsed_seconds, dtype=dtype, device=device)

        step_rewards: list[Tensor] = []
        step_dangers: list[Tensor] = []
        step_values: list[Tensor] = []
        branch_steps: list[list[LookaheadRolloutStep]] = [[] for _ in range(num_branches)]
        branch_pruned = [False] * num_branches
        branch_prune_reason: list[str | None] = [None] * num_branches

        with torch.no_grad():
            for k in range(H):
                # Build batch action vectors for step k: [M, total_queries]
                step_actions = [seq[k] for seq in candidate_sequences]
                action_vectors = torch.stack(
                    [
                        directional_to_control_vector(
                            act,
                            total_queries=self.total_queries,
                            device=device,
                            dtype=dtype,
                        )
                        for act in step_actions
                    ],
                    dim=0,
                )
                action_indices = torch.tensor(
                    [int(act) for act in step_actions],
                    dtype=torch.long,
                    device=device,
                )

                (
                    belief,
                    working_memory,
                    thoughts,
                    cur_sensors,
                    r_hat,
                    d_hat,
                    v_hat,
                ) = self.unroll_latent_step(
                    belief=belief,
                    working_memory=working_memory,
                    thoughts=thoughts,
                    sensors=cur_sensors,
                    goal_context=goal_context,
                    retrieved_memory=retrieved_memory,
                    action_vector=action_vectors,
                    elapsed_seconds=elapsed_tensor,
                    action_indices=action_indices,
                )

                step_rewards.append(r_hat)
                step_dangers.append(d_hat)
                step_values.append(v_hat)

                # Record rollout step for each branch
                for b_idx in range(num_branches):
                    r_val = float(r_hat[b_idx].item())
                    d_val = float(d_hat[b_idx].item())
                    v_val = float(v_hat[b_idx].item())
                    is_haz = d_val >= self.hazard_prune_threshold

                    branch_steps[b_idx].append(
                        LookaheadRolloutStep(
                            step_index=k,
                            action=step_actions[b_idx],
                            predicted_reward=r_val,
                            predicted_danger=d_val,
                            predicted_value=v_val,
                            is_hazard=is_haz,
                        )
                    )

                    if is_haz and not branch_pruned[b_idx]:
                        branch_pruned[b_idx] = True
                        branch_prune_reason[b_idx] = f"hazard_collision_step_{k}"

            # Terminal value estimate at z_{t+H}
            terminal_values = step_values[-1]

        # Check counterfactual foresight if head is present
        cf_preds = None
        if getattr(self, "counterfactual_foresight_head", None) is not None:
            cf_preds = self.counterfactual_foresight_head.forward_all_actions(state.thoughts)

        # Check topological goal head if present
        topo_pred = None
        if getattr(self, "topological_goal_head", None) is not None:
            topo_pred = self.topological_goal_head(state.thoughts)

        # Compute branch cumulative utilities
        results: list[LookaheadBranchResult] = []
        for b_idx in range(num_branches):
            disc_r_sum = 0.0
            disc_d_sum = 0.0
            for k in range(H):
                discount = self.gamma**k
                disc_r_sum += discount * float(step_rewards[k][b_idx].item())
                disc_d_sum += discount * float(step_dangers[k][b_idx].item())

            v_term = float(terminal_values[b_idx].item())
            u_cumulative = (self.gamma**H) * v_term

            first_act = candidate_sequences[b_idx][0]
            a_idx = int(first_act) if isinstance(first_act, (int, DirectionalAction)) else 0

            # If Counterfactual Foresight is available, use its supervised predictions for safety
            if cf_preds is not None:
                is_pruned = False
                prune_reason = None
                if a_idx in cf_preds:
                    cf_branch = cf_preds[a_idx]
                    cf_haz = float(cf_branch.hazard_probability[0, 0, 0].item())
                    cf_esc = float(cf_branch.predicted_escape_margin[0, min(1, cf_branch.predicted_escape_margin.shape[1] - 1), 0].item())
                    if cf_haz >= self.hazard_prune_threshold:
                        is_pruned = True
                        prune_reason = "counterfactual_hazard_predicted"
                    else:
                        # Calibrated progressive hazard penalty on trained counterfactual hazard
                        danger_penalty = self.hazard_weight * cf_haz
                        if cf_haz > 0.25:
                            danger_penalty += 4.0 * (cf_haz - 0.25) ** 2
                        u_cumulative -= danger_penalty
                    if cf_esc < 0.0:
                        u_cumulative += cf_esc * 2.0
            else:
                # Standalone fallback when counterfactual foresight is absent
                is_pruned = branch_pruned[b_idx]
                prune_reason = branch_prune_reason[b_idx]
                u_cumulative += disc_r_sum - self.hazard_weight * disc_d_sum

            # Dead end check
            if not is_pruned and u_cumulative < self.dead_end_threshold:
                is_pruned = True
                prune_reason = "dead_end_utility_threshold"

            # Apply Topological Goal and Junction routing bonus if not pruned
            if topo_pred is not None and not is_pruned:
                # Cardinal action direction vectors: None=(0,0), W=(0,-1), A=(-1,0), S=(0,1), D=(1,0)
                _ACT_VEC = {0: (0.0, 0.0), 1: (0.0, -1.0), 2: (-1.0, 0.0), 3: (0.0, 1.0), 4: (1.0, 0.0)}
                vx, vy = _ACT_VEC.get(a_idx, (0.0, 0.0))
                gx = float(topo_pred.pellet_cluster_vector[0, 0].item())
                gy = float(topo_pred.pellet_cluster_vector[0, 1].item())
                goal_alignment = vx * gx + vy * gy
                j_score = float(topo_pred.junction_exit_logits[0, a_idx].item())
                u_cumulative += 1.5 * goal_alignment + 0.5 * j_score

            # Apply severe penalty if pruned
            if is_pruned:
                u_cumulative = -1e9

            results.append(
                LookaheadBranchResult(
                    action_sequence=tuple(candidate_sequences[b_idx]),
                    rollout_steps=tuple(branch_steps[b_idx]),
                    cumulative_utility=u_cumulative,
                    discounted_reward_sum=disc_r_sum,
                    discounted_danger_sum=disc_d_sum,
                    terminal_value=v_term,
                    is_pruned=is_pruned,
                    prune_reason=prune_reason,
                )
            )

        return tuple(results)

    def plan(
        self,
        *,
        state: BrainState,
        sensors: Tensor | None = None,
        candidate_sequences: Sequence[Sequence[DirectionalAction | int]] | None = None,
        policy_logits: Tensor | None = None,
        policy_prior_weight: float = 0.0,
        horizon: int | None = None,
        elapsed_seconds: float = 1.0 / 60.0,
    ) -> LookaheadPlanResult:
        """Perform lookahead planning from current latent state.

        Args:
            state: Current recurrent BrainState.
            sensors: Current sensory tensor [batch, sensor_tokens, width].
                     If None, synthesized zero sensors are used.
            candidate_sequences: Optional explicit candidate sequences to evaluate.
            policy_logits: Optional 5-way policy prior logits [None, W, A, S, D].
            policy_prior_weight: Multiplier weight on policy prior logits.
            horizon: Lookahead depth (defaults to self.horizon).
            elapsed_seconds: Simulated decision interval.

        Returns:
            LookaheadPlanResult containing best action, branch details, and utilities.
        """
        h = self.horizon if horizon is None else horizon
        if h < 1 or h > 5:
            raise ValueError(f"horizon must be in [1, 5], got {h}")

        if candidate_sequences is None:
            candidate_sequences = generate_directional_candidate_sequences(
                h,
                include_none=False,
                prune_reversals=self.prune_reversals,
            )

        if sensors is None:
            sensors = torch.zeros(
                (1, self.config.sensor_tokens, self.width),
                dtype=state.belief.dtype,
                device=state.belief.device,
            )
        elif sensors.ndim == 2:
            sensors = sensors.unsqueeze(0)

        branch_results = self.evaluate_candidate_branches(
            state=state,
            sensors=sensors,
            candidate_sequences=candidate_sequences,
            elapsed_seconds=elapsed_seconds,
        )

        utilities = tuple(b.cumulative_utility for b in branch_results)
        effective_scores = []
        for b in branch_results:
            score = -1e6 if b.is_pruned else b.cumulative_utility
            if policy_logits is not None and policy_prior_weight > 0.0:
                first_act = b.action_sequence[0]
                act_idx = int(first_act) if isinstance(first_act, (int, DirectionalAction)) else 0
                if 0 <= act_idx < policy_logits.shape[-1]:
                    score += policy_prior_weight * float(policy_logits[act_idx].item())
            effective_scores.append(score)

        chosen_idx = int(torch.tensor(effective_scores).argmax().item())
        best_branch = branch_results[chosen_idx]

        first_action = best_branch.action_sequence[0]
        if isinstance(first_action, int) and not isinstance(first_action, DirectionalAction):
            best_action = DirectionalAction(first_action)
        else:
            best_action = first_action

        best_action_control = directional_to_control_vector(
            best_action,
            total_queries=self.total_queries,
            device=state.belief.device,
            dtype=state.belief.dtype,
        )

        pruned_count = sum(1 for b in branch_results if b.is_pruned)

        return LookaheadPlanResult(
            best_action=best_action,
            best_action_control=best_action_control,
            best_branch=best_branch,
            all_branches=branch_results,
            branch_utilities=utilities,
            chosen_index=chosen_idx,
            pruned_count=pruned_count,
        )


__all__ = [
    "DirectionalAction",
    "LatentHazardHead",
    "LatentLookaheadPlanner",
    "LatentRewardHead",
    "LatentSensoryTransition",
    "LookaheadBranchResult",
    "LookaheadPlanResult",
    "LookaheadRolloutStep",
    "directional_to_control_vector",
    "generate_directional_candidate_sequences",
]
