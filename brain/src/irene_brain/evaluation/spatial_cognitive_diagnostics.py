"""Spatial, geometric, internal thought, and incident diagnostics for Pseudo-Brain.

Computes exact quantitative evidence rather than heuristic narratives:
1. Physical & Spatial Dynamics:
   - Ghost collision count vs Wall bump count.
   - Per-catch incident micro-telemetry (catch topology, topology at t-5/t-10, nearest junction distance, ghost distance delta).
   - Ghost catches categorized by topology (corridor vs junction vs dead-end).
   - Mean and minimum Euclidean distance to nearest ghost.
   - Intersection classifications (corridor vs dead-end vs 3/4-way junction).
   - Unsafe intersection crossings (entering junction when nearest ghost distance <= 2).
   - Distance traveled & Pellets per 100 movement steps.

2. Wall Interaction & Recovery Dynamics:
   - Distinct wall contact events.
   - Mean repeated pushes per contact event.
   - Maximum repeated pushes against a wall.
   - Wall recovery latency (ticks to change direction after hitting wall).

3. Event-Triggered Internal State Dynamics (Delta T = 1 - cos(T_t, T_t+1)):
   - Delta T on normal movement frames.
   - Delta T on wall bump frames.
   - Delta T on high ghost proximity frames (ghost dist <= 2.5).

4. Internal Thought-Field Dynamics & Effective Dimensionality:
   - Effective Rank / Dimensionality of thoughtlets (Roy & Vetterli 2007 SVD-entropy).
   - Mean thoughtlet norm.
   - Pairwise thoughtlet cosine similarity (subspace alignment / collapse).
   - Inter-thoughtlet activation variance.
   - Global temporal persistence.
   - Model parameter SHA256 digest verification.

5. Outcome-Conditioned Expert Disagreement:
   - In-sample direct model imitation agreement vs Rollout lookahead agreement.
   - Total disagreements with the expert lookahead planner.
   - Disagreements leading to productive pellet gain + survival.
   - Disagreements leading to neutral safe survival.
   - Disagreements leading to ghost catches.

6. Cognitive Depth Scaling Ablation:
   - Evaluates the same learned model across cognitive thought cycles (1, 2, 3, 4, 6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import math
import time
from typing import Any, Mapping

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:
    torch = None
    Tensor = Any  # type: ignore[assignment]

from ..environments.maze_chase import MazeChaseEnv, _bfs_distances
from ..types import GenericControl, Observation


def compute_model_param_digest(model: object) -> str:
    """Compute a strict SHA256 digest over all trainable parameters in a PyTorch module."""
    if torch is None or not hasattr(model, "parameters"):
        return "non_torch_model"
    hasher = sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()[:16]


def compute_effective_rank(z: Tensor, eps: float = 1e-12) -> float:
    """Compute Roy & Vetterli (2007) effective rank via singular value entropy.

    Args:
        z: [N, D] matrix where N is thoughtlets and D is core_width.

    Returns:
        Effective dimensionality in [1.0, min(N, D)].
    """
    if torch is None or not isinstance(z, Tensor):
        return 1.0
    if z.ndim != 2 or z.shape[0] <= 1:
        return 1.0
    with torch.no_grad():
        centered = z.float() - z.float().mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(centered)
        except Exception:
            return 1.0
        s_sum = float(s.sum().item())
        if s_sum <= eps or not math.isfinite(s_sum):
            return 1.0
        p = s / s_sum
        p_nonzero = p[p > eps]
        entropy = -torch.sum(p_nonzero * torch.log(p_nonzero)).item()
        eff_dim = math.exp(entropy)
        return float(max(1.0, min(float(z.shape[0]), eff_dim)))


def compute_pairwise_cosine_similarity(z: Tensor, eps: float = 1e-12) -> float:
    """Compute average pairwise cosine similarity across all pairs of thoughtlets in [N, D]."""
    if torch is None or not isinstance(z, Tensor):
        return 1.0
    if z.ndim != 2 or z.shape[0] <= 1:
        return 1.0
    with torch.no_grad():
        normalized = z.float() / (z.float().norm(dim=-1, keepdim=True) + eps)
        sim_matrix = torch.mm(normalized, normalized.t())
        n = z.shape[0]
        # Mask out self-similarity diagonal
        mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
        pair_sims = sim_matrix[mask]
        return float(pair_sims.mean().item())


@dataclass(frozen=True, slots=True)
class CatchIncidentReport:
    """Detailed micro-telemetry capturing the exact state and history at a ghost catch event."""

    catch_index: int
    tick: int
    topology_at_catch: str  # 'corridor', 'junction', 'dead_end'
    topology_t_minus_5: str
    topology_t_minus_10: str
    nearest_junction_dist: int
    ghost_dist_t_minus_10: float
    ghost_dist_t_minus_5: float
    ghost_dist_t_minus_1: float
    expert_disagreement_preceding_10: float
    thoughtlet_effective_rank: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "catch_index": self.catch_index,
            "tick": self.tick,
            "topology_at_catch": self.topology_at_catch,
            "topology_t_minus_5": self.topology_t_minus_5,
            "topology_t_minus_10": self.topology_t_minus_10,
            "nearest_junction_dist": self.nearest_junction_dist,
            "ghost_dist_t_minus_10": round(self.ghost_dist_t_minus_10, 2),
            "ghost_dist_t_minus_5": round(self.ghost_dist_t_minus_5, 2),
            "ghost_dist_t_minus_1": round(self.ghost_dist_t_minus_1, 2),
            "expert_disagreement_preceding_10": round(self.expert_disagreement_preceding_10, 2),
            "thoughtlet_effective_rank": round(self.thoughtlet_effective_rank, 2),
        }


@dataclass(frozen=True, slots=True)
class WallInteractionTelemetry:
    """Detailed telemetry measuring wall contact duration, recovery latency, and repeated pushes."""

    wall_bump_ticks: int
    wall_contact_events: int
    mean_repeated_pushes_per_event: float
    max_repeated_pushes: int
    mean_wall_recovery_latency: float  # Ticks until direction changes after bump

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_bump_ticks": self.wall_bump_ticks,
            "wall_contact_events": self.wall_contact_events,
            "mean_repeated_pushes_per_event": round(self.mean_repeated_pushes_per_event, 2),
            "max_repeated_pushes": self.max_repeated_pushes,
            "mean_wall_recovery_latency": round(self.mean_wall_recovery_latency, 2),
        }


@dataclass(frozen=True, slots=True)
class EventTriggeredThoughtDynamics:
    """Measures internal state delta (1 - cos(T_t, T_t+1)) around specific environmental events."""

    mean_delta_normal_step: float
    mean_delta_wall_bump: float
    mean_delta_ghost_proximity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_delta_normal_step": round(self.mean_delta_normal_step, 5),
            "mean_delta_wall_bump": round(self.mean_delta_wall_bump, 5),
            "mean_delta_ghost_proximity": round(self.mean_delta_ghost_proximity, 5),
        }


@dataclass(frozen=True, slots=True)
class OutcomeConditionedDisagreement:
    """Tracks whether policy divergence from expert was productive, benign, or fatal."""

    total_disagreements: int
    disagreed_survived_and_pellet_gained: int
    disagreed_survived: int
    disagreed_and_caught: int

    @property
    def productive_ratio(self) -> float:
        if self.total_disagreements == 0:
            return 0.0
        return self.disagreed_survived_and_pellet_gained / self.total_disagreements

    @property
    def fatal_ratio(self) -> float:
        if self.total_disagreements == 0:
            return 0.0
        return self.disagreed_and_caught / self.total_disagreements

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_disagreements": self.total_disagreements,
            "productive_pellet_disagreements": self.disagreed_survived_and_pellet_gained,
            "safe_disagreements": self.disagreed_survived,
            "fatal_disagreements": self.disagreed_and_caught,
            "productive_ratio": round(self.productive_ratio * 100.0, 1),
            "fatal_ratio": round(self.fatal_ratio * 100.0, 1),
        }


@dataclass(frozen=True, slots=True)
class EventConditionedGateTelemetry:
    """Measures the adaptive thought-update gate alpha across specific environmental conditions."""

    alpha_normal: float
    alpha_wall_collision: float
    alpha_ghost_danger: float
    alpha_unexpected_blocker: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_normal": round(self.alpha_normal, 2),
            "alpha_wall_collision": round(self.alpha_wall_collision, 2),
            "alpha_ghost_danger": round(self.alpha_ghost_danger, 2),
            "alpha_unexpected_blocker": round(self.alpha_unexpected_blocker, 2),
        }


@dataclass(frozen=True, slots=True)
class ContextConditionedDepthTelemetry:
    """Measures cognitive cycles spent across spatial and hazard contexts."""

    mean_cycles_overall: float
    cycles_open_corridor: float
    cycles_junction: float
    cycles_ghost_near: float
    cycles_dead_end_ghost_near: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_cycles_overall": round(self.mean_cycles_overall, 2),
            "cycles_open_corridor": round(self.cycles_open_corridor, 2),
            "cycles_junction": round(self.cycles_junction, 2),
            "cycles_ghost_near": round(self.cycles_ghost_near, 2),
            "cycles_dead_end_ghost_near": round(self.cycles_dead_end_ghost_near, 2),
        }


@dataclass(frozen=True, slots=True)
class FuturePredictionAccuracyTelemetry:
    """Measures multi-horizon future foresight accuracy and hazard classification."""

    ghost_pos_error_t1: float
    ghost_pos_error_t3: float
    ghost_pos_error_t5: float
    escape_margin_accuracy: float  # Fraction where predicted sign matches ground truth
    danger_precision: float        # Precision for predicting hazard <= 2.5
    danger_recall: float           # Recall for predicting hazard <= 2.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "ghost_pos_error_t1": round(self.ghost_pos_error_t1, 2),
            "ghost_pos_error_t3": round(self.ghost_pos_error_t3, 2),
            "ghost_pos_error_t5": round(self.ghost_pos_error_t5, 2),
            "escape_margin_accuracy": round(self.escape_margin_accuracy * 100.0, 1),
            "danger_precision": round(self.danger_precision * 100.0, 1),
            "danger_recall": round(self.danger_recall * 100.0, 1),
        }


@dataclass(frozen=True, slots=True)
class SpatialCognitiveTelemetry:
    """Rigorous quantitative metrics capturing spatial dynamics and internal cognition."""

    # Episode Identifiers
    seed: int
    param_digest: str
    ticks_simulated: int

    # Task Performance
    pellets_eaten: int
    ghost_collisions: int
    survival_ticks: int

    # Wall Interaction Dynamics
    wall_dynamics: WallInteractionTelemetry

    # Spatial Geometry & Intersections
    mean_nearest_ghost_dist: float
    min_nearest_ghost_dist: float
    intersection_entries_total: int
    unsafe_intersection_entries: int  # Nearest ghost <= 2 tiles on junction entry
    corridor_steps: int
    dead_end_steps: int

    # Catch Incident Telemetry by Topology
    corridor_catches: int
    junction_catches: int
    dead_end_catches: int
    catch_incidents: tuple[CatchIncidentReport, ...]

    # Kinematics & Movement Efficiency
    actual_tiles_moved: int
    pellets_per_100_moves: float
    expert_disagreement_pct: float
    outcome_conditioned_disagreement: OutcomeConditionedDisagreement

    # Event-Triggered Internal State Dynamics
    event_thought_dynamics: EventTriggeredThoughtDynamics

    # Internal Thought-Field Representation Health
    mean_thoughtlet_norm: float
    mean_pairwise_thoughtlet_sim: float
    mean_thoughtlet_variance: float
    mean_thoughtlet_persistence: float
    mean_thoughtlet_effective_rank: float

    # New Cognitive Mechanism Telemetry
    gate_telemetry: EventConditionedGateTelemetry | None = None
    depth_telemetry: ContextConditionedDepthTelemetry | None = None
    future_prediction_telemetry: FuturePredictionAccuracyTelemetry | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "param_digest": self.param_digest,
            "ticks_simulated": self.ticks_simulated,
            "pellets_eaten": self.pellets_eaten,
            "ghost_collisions": self.ghost_collisions,
            "survival_ticks": self.survival_ticks,
            "wall_dynamics": self.wall_dynamics.to_dict(),
            "mean_nearest_ghost_dist": round(self.mean_nearest_ghost_dist, 3),
            "min_nearest_ghost_dist": round(self.min_nearest_ghost_dist, 3),
            "intersection_entries_total": self.intersection_entries_total,
            "unsafe_intersection_entries": self.unsafe_intersection_entries,
            "corridor_steps": self.corridor_steps,
            "dead_end_steps": self.dead_end_steps,
            "catches_by_topology": {
                "corridor": self.corridor_catches,
                "junction": self.junction_catches,
                "dead_end": self.dead_end_catches,
            },
            "catch_incidents": [c.to_dict() for c in self.catch_incidents],
            "actual_tiles_moved": self.actual_tiles_moved,
            "pellets_per_100_moves": round(self.pellets_per_100_moves, 2),
            "expert_disagreement_pct": round(self.expert_disagreement_pct, 2),
            "outcome_disagreement": self.outcome_conditioned_disagreement.to_dict(),
            "event_thought_dynamics": self.event_thought_dynamics.to_dict(),
            "thought_representation": {
                "mean_norm": round(self.mean_thoughtlet_norm, 4),
                "pairwise_cosine_sim": round(self.mean_pairwise_thoughtlet_sim, 4),
                "variance": round(self.mean_thoughtlet_variance, 4),
                "persistence": round(self.mean_thoughtlet_persistence, 4),
                "effective_rank": round(self.mean_thoughtlet_effective_rank, 2),
            },
        }


def _classify_maze_topology(maze: set[tuple[int, int]]) -> dict[tuple[int, int], str]:
    """Classify every maze cell into 'junction' (>=3 exits), 'corridor' (2 exits), or 'dead_end' (1 exit)."""
    topology = {}
    for x, y in maze:
        open_neighbors = sum(
            1 for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
            if (x + dx, y + dy) in maze
        )
        if open_neighbors >= 3:
            topology[(x, y)] = "junction"
        elif open_neighbors == 2:
            topology[(x, y)] = "corridor"
        else:
            topology[(x, y)] = "dead_end"
    return topology


def run_instrumented_diagnostic_episode(
    policy: object,
    env: MazeChaseEnv,
    *,
    seed: int,
    max_ticks: int = 120,
    expert_policy: object | None = None,
) -> SpatialCognitiveTelemetry:
    """Execute one episode with comprehensive geometric, kinematic, event-triggered thought, and incident telemetry."""
    obs = env.reset(seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    if expert_policy is not None and hasattr(expert_policy, "reset"):
        expert_policy.reset(seed)

    maze = env._maze
    topology = _classify_maze_topology(maze)
    junction_cells = {c for c, t in topology.items() if t == "junction"}
    param_digest = compute_model_param_digest(getattr(policy, "model", policy))

    # Tracking accumulators
    ghost_distances: list[float] = []
    min_ghost_distance = 999.0
    actual_moves = 0
    junction_entries = 0
    unsafe_junction_entries = 0
    corridor_steps = 0
    dead_end_steps = 0
    expert_disagreements = 0
    decisions_total = 0

    # Wall interaction tracking
    wall_bump_ticks = 0
    wall_contact_events = 0
    current_wall_streak = 0
    wall_streaks: list[int] = []
    recovery_latencies: list[int] = []
    in_wall_contact = False
    ticks_since_wall_hit = 0
    wall_hit_action: tuple[int, ...] | None = None

    # Incident tracking history
    recent_ghost_dists: list[float] = []
    recent_disagreements: list[bool] = []
    recent_topologies: list[str] = []
    catch_incidents: list[CatchIncidentReport] = []
    corridor_catches = 0
    junction_catches = 0
    dead_end_catches = 0

    # Disagreement outcome tracker
    pending_disagreements: list[dict[str, Any]] = []
    prod_disagreements = 0
    safe_disagreements = 0
    fatal_disagreements = 0

    # Thought dynamics accumulators
    thought_variances: list[float] = []
    thought_persistences: list[float] = []
    thought_effective_ranks: list[float] = []
    thought_norms: list[float] = []
    thought_pairwise_sims: list[float] = []

    deltas_normal: list[float] = []
    deltas_wall: list[float] = []
    deltas_ghost: list[float] = []

    # New Cognitive Mechanism Accumulators
    alphas_normal: list[float] = []
    alphas_wall: list[float] = []
    alphas_ghost: list[float] = []
    alphas_unexpected: list[float] = []

    cycles_overall: list[int] = []
    cycles_corridor: list[int] = []
    cycles_junction: list[int] = []
    cycles_ghost_near: list[int] = []
    cycles_dead_end_ghost_near: list[int] = []

    ghost_errors_t1: list[float] = []
    ghost_errors_t3: list[float] = []
    ghost_errors_t5: list[float] = []
    escape_margin_matches: list[bool] = []
    danger_tp = 0
    danger_fp = 0
    danger_fn = 0
    danger_tn = 0

    last_pos = (env._player_x, env._player_y)
    last_thought: Tensor | None = None

    for tick in range(max_ticks):
        # 1. Capture current geometric state
        px, py = env._player_x, env._player_y
        current_pos = (px, py)
        cell_type = topology.get(current_pos, "corridor")

        cur_nearest = 999.0
        if env._ghosts:
            cur_nearest = min(
                math.sqrt((px - gx) ** 2 + (py - gy) ** 2) for gx, gy in env._ghosts
            )
            ghost_distances.append(cur_nearest)
            if cur_nearest < min_ghost_distance:
                min_ghost_distance = cur_nearest

        recent_ghost_dists.append(cur_nearest)
        recent_topologies.append(cell_type)
        if len(recent_ghost_dists) > 20:
            recent_ghost_dists.pop(0)
        if len(recent_topologies) > 20:
            recent_topologies.pop(0)

        if cell_type == "junction" and last_pos != current_pos:
            junction_entries += 1
            if cur_nearest <= 2.0:
                unsafe_junction_entries += 1
        elif cell_type == "corridor":
            corridor_steps += 1
        elif cell_type == "dead_end":
            dead_end_steps += 1

        # 2. Query policy and optional expert
        ctrl, _, _ = policy.decide(obs, 1.0 / 60.0) if hasattr(policy, "decide") else (policy.act(obs), {}, None)
        decisions_total += 1

        disagreed = False
        if expert_policy is not None:
            expert_ctrl = expert_policy.act(obs)
            if ctrl.keys_down != expert_ctrl.keys_down:
                expert_disagreements += 1
                disagreed = True
                pending_disagreements.append(
                    {
                        "start_tick": tick,
                        "start_pellets": env._pellets_eaten,
                        "start_caught": env._times_caught,
                    }
                )

        recent_disagreements.append(disagreed)
        if len(recent_disagreements) > 20:
            recent_disagreements.pop(0)

        # 3. Inspect internal thought states
        cur_effective_rank = 1.0
        cur_thought_delta = 0.0
        state = getattr(policy, "_state", None)
        if state is not None and hasattr(state, "thoughts") and isinstance(state.thoughts, Tensor):
            thoughts = state.thoughts[0]  # [thoughtlets, registers, width]
            pooled = thoughts.mean(dim=1)  # [thoughtlets, width]
            cur_effective_rank = compute_effective_rank(pooled)
            thought_effective_ranks.append(cur_effective_rank)
            thought_norms.append(float(pooled.norm(dim=-1).mean().item()))
            thought_pairwise_sims.append(compute_pairwise_cosine_similarity(pooled))

            # Inter-thoughtlet variance
            thoughtlet_mean = thoughts.mean(dim=0, keepdim=True)
            var = float(((thoughts - thoughtlet_mean) ** 2).mean().item())
            thought_variances.append(var)

            # Temporal persistence (cosine sim between successive steps)
            flattened = thoughts.flatten()
            if last_thought is not None:
                sim = float(
                    torch.cosine_similarity(
                        flattened.unsqueeze(0), last_thought.unsqueeze(0)
                    ).item()
                )
                thought_persistences.append(sim)
                cur_thought_delta = max(0.0, 1.0 - sim)
            last_thought = flattened.detach().clone()

        # 4. Step environment
        pre_caught = env._times_caught
        outcome = env.step(ctrl)
        new_pos = (env._player_x, env._player_y)

        # Wall bump detection and streak dynamics
        is_wall_bump = bool(ctrl.keys_down and new_pos == current_pos and (tick % env._player_period == 0))
        if is_wall_bump:
            wall_bump_ticks += 1
            current_wall_streak += 1
            if not in_wall_contact:
                wall_contact_events += 1
                in_wall_contact = True
                ticks_since_wall_hit = 0
                wall_hit_action = ctrl.keys_down
            deltas_wall.append(cur_thought_delta)
        else:
            if in_wall_contact:
                wall_streaks.append(current_wall_streak)
                current_wall_streak = 0
                in_wall_contact = False
                recovery_latencies.append(ticks_since_wall_hit)
            if new_pos != current_pos:
                actual_moves += 1
                deltas_normal.append(cur_thought_delta)

        if in_wall_contact:
            ticks_since_wall_hit += 1

        if cur_nearest <= 2.5:
            deltas_ghost.append(cur_thought_delta)

        # 5. Extract cognitive diagnostics if available
        diag = getattr(policy, "_last_diagnostics", None)
        if diag is not None:
            # Gate telemetry
            if getattr(diag, "thought_update_gates", None) is not None:
                cur_alpha = float(diag.thought_update_gates.mean().item())
                if is_wall_bump:
                    alphas_wall.append(cur_alpha)
                elif cur_nearest <= 2.5:
                    alphas_ghost.append(cur_alpha)
                elif ctrl.keys_down and new_pos == current_pos:
                    alphas_unexpected.append(cur_alpha)
                elif new_pos != current_pos:
                    alphas_normal.append(cur_alpha)

            # Depth telemetry
            cur_cycles = getattr(diag, "cycles_completed", 3)
            cycles_overall.append(cur_cycles)
            if cell_type == "dead_end" and cur_nearest <= 3.5:
                cycles_dead_end_ghost_near.append(cur_cycles)
            elif cur_nearest <= 3.5:
                cycles_ghost_near.append(cur_cycles)
            elif cell_type == "junction":
                cycles_junction.append(cur_cycles)
            elif cell_type == "corridor":
                cycles_corridor.append(cur_cycles)

            # Future prediction accuracy telemetry
            future_preds = getattr(diag, "future_trajectory_predictions", None)
            if future_preds is not None and isinstance(future_preds, dict):
                if "predicted_ghost_proximity" in future_preds and future_preds["predicted_ghost_proximity"] is not None:
                    pred_g = future_preds["predicted_ghost_proximity"][0].flatten()
                    if len(pred_g) > 0:
                        ghost_errors_t1.append(abs(float(pred_g[0].item()) - cur_nearest))
                    if len(pred_g) > 1:
                        ghost_errors_t3.append(abs(float(pred_g[1].item()) - cur_nearest))
                    if len(pred_g) > 2:
                        ghost_errors_t5.append(abs(float(pred_g[2].item()) - cur_nearest))

                    pred_danger = bool(float(pred_g[0].item()) <= 2.5) if len(pred_g) > 0 else False
                    actual_danger = bool(cur_nearest <= 2.5)
                    if pred_danger and actual_danger:
                        danger_tp += 1
                    elif pred_danger and not actual_danger:
                        danger_fp += 1
                    elif not pred_danger and actual_danger:
                        danger_fn += 1
                    else:
                        danger_tn += 1

                if "predicted_escape_margin" in future_preds and future_preds["predicted_escape_margin"] is not None:
                    pred_esc = float(future_preds["predicted_escape_margin"][0][0].item())
                    actual_esc = cur_nearest - 2.0
                    escape_margin_matches.append((pred_esc >= 0) == (actual_esc >= 0))

        # Check for catch incident
        if env._times_caught > pre_caught:
            if cell_type == "junction":
                junction_catches += 1
            elif cell_type == "dead_end":
                dead_end_catches += 1
            else:
                corridor_catches += 1

            junction_distances = _bfs_distances(current_pos, maze)
            dist_to_junction = min(
                (junction_distances[j] for j in junction_cells if j in junction_distances),
                default=0,
            )

            d_10 = recent_ghost_dists[-11] if len(recent_ghost_dists) >= 11 else recent_ghost_dists[0]
            d_5 = recent_ghost_dists[-6] if len(recent_ghost_dists) >= 6 else recent_ghost_dists[0]
            d_1 = recent_ghost_dists[-2] if len(recent_ghost_dists) >= 2 else recent_ghost_dists[0]
            top_10 = recent_topologies[-11] if len(recent_topologies) >= 11 else recent_topologies[0]
            top_5 = recent_topologies[-6] if len(recent_topologies) >= 6 else recent_topologies[0]
            preceding_10_disagreements = recent_disagreements[-10:] if recent_disagreements else [False]
            disagree_pct_10 = (sum(preceding_10_disagreements) / len(preceding_10_disagreements)) * 100.0

            incident = CatchIncidentReport(
                catch_index=len(catch_incidents) + 1,
                tick=tick,
                topology_at_catch=cell_type,
                topology_t_minus_5=top_5,
                topology_t_minus_10=top_10,
                nearest_junction_dist=dist_to_junction,
                ghost_dist_t_minus_10=d_10,
                ghost_dist_t_minus_5=d_5,
                ghost_dist_t_minus_1=d_1,
                expert_disagreement_preceding_10=disagree_pct_10,
                thoughtlet_effective_rank=cur_effective_rank,
            )
            catch_incidents.append(incident)

        # Evaluate pending disagreements
        resolved_disagreements = []
        for d in pending_disagreements:
            if tick - d["start_tick"] >= 10 or outcome.terminated or outcome.truncated:
                caught_delta = env._times_caught - d["start_caught"]
                pellet_delta = env._pellets_eaten - d["start_pellets"]
                if caught_delta > 0:
                    fatal_disagreements += 1
                elif pellet_delta > 0:
                    prod_disagreements += 1
                else:
                    safe_disagreements += 1
                resolved_disagreements.append(d)
        for r in resolved_disagreements:
            pending_disagreements.remove(r)

        last_pos = current_pos
        obs = outcome.observation
        if outcome.terminated or outcome.truncated:
            break

    if in_wall_contact and current_wall_streak > 0:
        wall_streaks.append(current_wall_streak)
        recovery_latencies.append(ticks_since_wall_hit)

    # Clean up remaining pending disagreements
    for d in pending_disagreements:
        caught_delta = env._times_caught - d["start_caught"]
        pellet_delta = env._pellets_eaten - d["start_pellets"]
        if caught_delta > 0:
            fatal_disagreements += 1
        elif pellet_delta > 0:
            prod_disagreements += 1
        else:
            safe_disagreements += 1

    outcome_disagreements = OutcomeConditionedDisagreement(
        total_disagreements=expert_disagreements,
        disagreed_survived_and_pellet_gained=prod_disagreements,
        disagreed_survived=safe_disagreements,
        disagreed_and_caught=fatal_disagreements,
    )

    wall_telemetry = WallInteractionTelemetry(
        wall_bump_ticks=wall_bump_ticks,
        wall_contact_events=wall_contact_events,
        mean_repeated_pushes_per_event=sum(wall_streaks) / len(wall_streaks) if wall_streaks else 0.0,
        max_repeated_pushes=max(wall_streaks, default=0),
        mean_wall_recovery_latency=sum(recovery_latencies) / len(recovery_latencies) if recovery_latencies else 0.0,
    )

    event_dynamics = EventTriggeredThoughtDynamics(
        mean_delta_normal_step=sum(deltas_normal) / len(deltas_normal) if deltas_normal else 0.0,
        mean_delta_wall_bump=sum(deltas_wall) / len(deltas_wall) if deltas_wall else 0.0,
        mean_delta_ghost_proximity=sum(deltas_ghost) / len(deltas_ghost) if deltas_ghost else 0.0,
    )

    gate_telemetry = EventConditionedGateTelemetry(
        alpha_normal=sum(alphas_normal) / len(alphas_normal) if alphas_normal else 0.05,
        alpha_wall_collision=sum(alphas_wall) / len(alphas_wall) if alphas_wall else 0.05,
        alpha_ghost_danger=sum(alphas_ghost) / len(alphas_ghost) if alphas_ghost else 0.05,
        alpha_unexpected_blocker=sum(alphas_unexpected) / len(alphas_unexpected) if alphas_unexpected else 0.05,
    )

    depth_telemetry = ContextConditionedDepthTelemetry(
        mean_cycles_overall=sum(cycles_overall) / len(cycles_overall) if cycles_overall else 3.0,
        cycles_open_corridor=sum(cycles_corridor) / len(cycles_corridor) if cycles_corridor else 1.0,
        cycles_junction=sum(cycles_junction) / len(cycles_junction) if cycles_junction else 3.0,
        cycles_ghost_near=sum(cycles_ghost_near) / len(cycles_ghost_near) if cycles_ghost_near else 3.0,
        cycles_dead_end_ghost_near=sum(cycles_dead_end_ghost_near) / len(cycles_dead_end_ghost_near) if cycles_dead_end_ghost_near else 3.0,
    )

    precision = danger_tp / (danger_tp + danger_fp) if (danger_tp + danger_fp) > 0 else 1.0
    recall = danger_tp / (danger_tp + danger_fn) if (danger_tp + danger_fn) > 0 else 1.0
    escape_acc = sum(escape_margin_matches) / len(escape_margin_matches) if escape_margin_matches else 1.0

    future_prediction_telemetry = FuturePredictionAccuracyTelemetry(
        ghost_pos_error_t1=sum(ghost_errors_t1) / len(ghost_errors_t1) if ghost_errors_t1 else 0.0,
        ghost_pos_error_t3=sum(ghost_errors_t3) / len(ghost_errors_t3) if ghost_errors_t3 else 0.0,
        ghost_pos_error_t5=sum(ghost_errors_t5) / len(ghost_errors_t5) if ghost_errors_t5 else 0.0,
        escape_margin_accuracy=escape_acc,
        danger_precision=precision,
        danger_recall=recall,
    )

    mean_dist = sum(ghost_distances) / len(ghost_distances) if ghost_distances else 0.0
    disagreement_pct = (expert_disagreements / decisions_total * 100.0) if decisions_total > 0 else 0.0
    pellets_per_100 = (env._pellets_eaten / actual_moves * 100.0) if actual_moves > 0 else 0.0

    return SpatialCognitiveTelemetry(
        seed=seed,
        param_digest=param_digest,
        ticks_simulated=env._tick,
        pellets_eaten=env._pellets_eaten,
        ghost_collisions=env._times_caught,
        survival_ticks=env._tick,
        wall_dynamics=wall_telemetry,
        mean_nearest_ghost_dist=mean_dist,
        min_nearest_ghost_dist=min_ghost_distance if min_ghost_distance < 900.0 else 0.0,
        intersection_entries_total=junction_entries,
        unsafe_intersection_entries=unsafe_junction_entries,
        corridor_steps=corridor_steps,
        dead_end_steps=dead_end_steps,
        corridor_catches=corridor_catches,
        junction_catches=junction_catches,
        dead_end_catches=dead_end_catches,
        catch_incidents=tuple(catch_incidents),
        actual_tiles_moved=actual_moves,
        pellets_per_100_moves=pellets_per_100,
        expert_disagreement_pct=disagreement_pct,
        outcome_conditioned_disagreement=outcome_disagreements,
        event_thought_dynamics=event_dynamics,
        mean_thoughtlet_norm=sum(thought_norms) / len(thought_norms) if thought_norms else 0.0,
        mean_pairwise_thoughtlet_sim=sum(thought_pairwise_sims) / len(thought_pairwise_sims) if thought_pairwise_sims else 0.0,
        mean_thoughtlet_variance=sum(thought_variances) / len(thought_variances) if thought_variances else 0.0,
        mean_thoughtlet_persistence=sum(thought_persistences) / len(thought_persistences) if thought_persistences else 0.0,
        mean_thoughtlet_effective_rank=sum(thought_effective_ranks) / len(thought_effective_ranks) if thought_effective_ranks else 1.0,
        gate_telemetry=gate_telemetry,
        depth_telemetry=depth_telemetry,
        future_prediction_telemetry=future_prediction_telemetry,
    )


def run_cognitive_depth_ablation(
    model: object,
    *,
    seeds: tuple[int, ...] = (1702, 1703, 1704),
    cycles_list: tuple[int, ...] = (1, 2, 3, 4, 6),
    max_ticks: int = 120,
) -> dict[int, dict[str, float]]:
    """Run cognitive depth scaling experiment across internal thought cycles."""
    from ..evaluation.latent_lookahead_policy import LatentLookaheadPolicy
    from ..model.lookahead_planner import LatentLookaheadPlanner
    from ..model.torch_model import IreneBrainModel

    if not isinstance(model, IreneBrainModel):
        raise TypeError("model must be an IreneBrainModel")

    results = {}
    base_config = model.config

    for cycles in cycles_list:
        scaled_config = replace(base_config, cognitive_cycles=cycles)
        scaled_model = IreneBrainModel(
            scaled_config,
            enable_adaptive_cognition=getattr(model, "enable_adaptive_cognition", False),
        )
        scaled_model.load_state_dict(model.state_dict(), strict=False)
        scaled_model.eval()

        planner = LatentLookaheadPlanner(
            model=scaled_model,
            horizon=3,
            gamma=0.95,
            hazard_weight=5.0,
            cognitive_cycles_per_step=cycles,
        )
        policy = LatentLookaheadPolicy(
            model=scaled_model,
            planner=planner,
            policy_prior_weight=2.0,
        )

        pellet_list = []
        catch_list = []
        rank_list = []
        time_list = []

        for seed in seeds:
            env = MazeChaseEnv(max_ticks=max_ticks, ghost_count=2)
            t0 = time.perf_counter()
            telem = run_instrumented_diagnostic_episode(
                policy=policy,
                env=env,
                seed=seed,
                max_ticks=max_ticks,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            pellet_list.append(telem.pellets_eaten)
            catch_list.append(telem.ghost_collisions)
            rank_list.append(telem.mean_thoughtlet_effective_rank)
            time_list.append(elapsed_ms / telem.ticks_simulated if telem.ticks_simulated > 0 else 0.0)

        results[cycles] = {
            "mean_pellets": sum(pellet_list) / len(pellet_list),
            "mean_catches": sum(catch_list) / len(catch_list),
            "mean_effective_rank": sum(rank_list) / len(rank_list),
            "latency_ms_per_step": sum(time_list) / len(time_list),
        }

    return results
