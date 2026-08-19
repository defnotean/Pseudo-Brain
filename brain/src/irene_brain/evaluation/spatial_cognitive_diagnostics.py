"""Spatial, geometric, internal thought, and incident diagnostics for Pseudo-Brain.

Computes exact quantitative evidence rather than heuristic narratives:
1. Physical & Spatial Dynamics:
   - Ghost collision count vs Wall bump count.
   - Per-catch incident micro-telemetry (catch topology, nearest junction distance, ghost distance delta).
   - Ghost catches categorized by topology (corridor vs junction vs dead-end).
   - Mean and minimum Euclidean distance to nearest ghost.
   - Intersection classifications (corridor vs dead-end vs 3/4-way junction).
   - Unsafe intersection crossings (entering junction when nearest ghost distance <= 2).
   - Distance traveled & Pellets per 100 movement steps.

2. Internal Thought-Field Dynamics & Effective Dimensionality:
   - Effective Rank / Dimensionality of thoughtlets (Roy & Vetterli 2007 SVD-entropy).
   - Inter-thoughtlet activation variance and slot differentiation.
   - Thought-slot temporal persistence (cosine similarity between t and t-1).
   - Model parameter SHA256 digest verification.

3. Outcome-Conditioned Expert Disagreement:
   - Total disagreements with the expert lookahead planner.
   - Disagreements leading to productive pellet gain + survival.
   - Disagreements leading to neutral safe survival.
   - Disagreements leading to ghost catches.

4. Cognitive Depth Scaling Ablation:
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
        # Center the matrix
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


@dataclass(frozen=True, slots=True)
class CatchIncidentReport:
    """Detailed micro-telemetry capturing the exact state at a ghost catch event."""

    catch_index: int
    tick: int
    topology: str  # 'corridor', 'junction', 'dead_end'
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
            "topology": self.topology,
            "nearest_junction_dist": self.nearest_junction_dist,
            "ghost_dist_t_minus_10": round(self.ghost_dist_t_minus_10, 2),
            "ghost_dist_t_minus_5": round(self.ghost_dist_t_minus_5, 2),
            "ghost_dist_t_minus_1": round(self.ghost_dist_t_minus_1, 2),
            "expert_disagreement_preceding_10": round(self.expert_disagreement_preceding_10, 2),
            "thoughtlet_effective_rank": round(self.thoughtlet_effective_rank, 2),
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
class SpatialCognitiveTelemetry:
    """Rigorous quantitative metrics capturing spatial dynamics and internal cognition."""

    # Episode Identifiers
    seed: int
    param_digest: str
    ticks_simulated: int

    # Task Performance
    pellets_eaten: int
    ghost_collisions: int
    wall_bumps: int
    survival_ticks: int

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

    # Internal Thought-Field Telemetry
    mean_thoughtlet_variance: float
    mean_thoughtlet_persistence: float
    mean_thoughtlet_effective_rank: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "param_digest": self.param_digest,
            "ticks_simulated": self.ticks_simulated,
            "pellets_eaten": self.pellets_eaten,
            "ghost_collisions": self.ghost_collisions,
            "wall_bumps": self.wall_bumps,
            "survival_ticks": self.survival_ticks,
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
            "mean_thoughtlet_variance": round(self.mean_thoughtlet_variance, 4),
            "mean_thoughtlet_persistence": round(self.mean_thoughtlet_persistence, 4),
            "mean_thoughtlet_effective_rank": round(self.mean_thoughtlet_effective_rank, 2),
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
    """Execute one episode with exact geometric, kinematic, cognitive, and incident telemetry."""
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
    wall_bumps = 0
    actual_moves = 0
    junction_entries = 0
    unsafe_junction_entries = 0
    corridor_steps = 0
    dead_end_steps = 0
    expert_disagreements = 0
    decisions_total = 0
    thought_variances: list[float] = []
    thought_persistences: list[float] = []
    thought_effective_ranks: list[float] = []

    # Incident tracking history
    recent_ghost_dists: list[float] = []
    recent_disagreements: list[bool] = []
    catch_incidents: list[CatchIncidentReport] = []
    corridor_catches = 0
    junction_catches = 0
    dead_end_catches = 0

    # Disagreement outcome tracker
    pending_disagreements: list[dict[str, Any]] = []
    prod_disagreements = 0
    safe_disagreements = 0
    fatal_disagreements = 0

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
        if len(recent_ghost_dists) > 20:
            recent_ghost_dists.pop(0)

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

        # 3. Inspect internal thought states and effective rank
        cur_effective_rank = 1.0
        state = getattr(policy, "_state", None)
        if state is not None and hasattr(state, "thoughts") and isinstance(state.thoughts, Tensor):
            thoughts = state.thoughts[0]  # [thoughtlets, registers, width]
            # Pool registers to get [thoughtlets, width]
            pooled_thoughts = thoughts.mean(dim=1)
            cur_effective_rank = compute_effective_rank(pooled_thoughts)
            thought_effective_ranks.append(cur_effective_rank)

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
            last_thought = flattened.detach().clone()

        # 4. Step environment
        pre_caught = env._times_caught
        outcome = env.step(ctrl)
        new_pos = (env._player_x, env._player_y)

        # Check for catch incident
        if env._times_caught > pre_caught:
            # Ghost catch occurred!
            if cell_type == "junction":
                junction_catches += 1
            elif cell_type == "dead_end":
                dead_end_catches += 1
            else:
                corridor_catches += 1

            # Compute BFS distance to nearest junction
            junction_distances = _bfs_distances(current_pos, maze)
            dist_to_junction = min(
                (junction_distances[j] for j in junction_cells if j in junction_distances),
                default=0,
            )

            d_10 = recent_ghost_dists[-11] if len(recent_ghost_dists) >= 11 else recent_ghost_dists[0]
            d_5 = recent_ghost_dists[-6] if len(recent_ghost_dists) >= 6 else recent_ghost_dists[0]
            d_1 = recent_ghost_dists[-2] if len(recent_ghost_dists) >= 2 else recent_ghost_dists[0]
            preceding_10_disagreements = recent_disagreements[-10:] if recent_disagreements else [False]
            disagree_pct_10 = (sum(preceding_10_disagreements) / len(preceding_10_disagreements)) * 100.0

            incident = CatchIncidentReport(
                catch_index=len(catch_incidents) + 1,
                tick=tick,
                topology=cell_type,
                nearest_junction_dist=dist_to_junction,
                ghost_dist_t_minus_10=d_10,
                ghost_dist_t_minus_5=d_5,
                ghost_dist_t_minus_1=d_1,
                expert_disagreement_preceding_10=disagree_pct_10,
                thoughtlet_effective_rank=cur_effective_rank,
            )
            catch_incidents.append(incident)

        # Evaluate pending disagreements that reached N=10 ticks
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

        # Detect wall bump (action requested but position unchanged)
        if ctrl.keys_down and new_pos == current_pos and (tick % env._player_period == 0):
            wall_bumps += 1
        elif new_pos != current_pos:
            actual_moves += 1

        last_pos = current_pos
        obs = outcome.observation
        if outcome.terminated or outcome.truncated:
            break

    # Clean up remaining pending disagreements at episode end
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

    mean_dist = sum(ghost_distances) / len(ghost_distances) if ghost_distances else 0.0
    disagreement_pct = (expert_disagreements / decisions_total * 100.0) if decisions_total > 0 else 0.0
    pellets_per_100 = (env._pellets_eaten / actual_moves * 100.0) if actual_moves > 0 else 0.0
    mean_var = sum(thought_variances) / len(thought_variances) if thought_variances else 0.0
    mean_persist = sum(thought_persistences) / len(thought_persistences) if thought_persistences else 0.0
    mean_rank = sum(thought_effective_ranks) / len(thought_effective_ranks) if thought_effective_ranks else 1.0

    return SpatialCognitiveTelemetry(
        seed=seed,
        param_digest=param_digest,
        ticks_simulated=env._tick,
        pellets_eaten=env._pellets_eaten,
        ghost_collisions=env._times_caught,
        wall_bumps=wall_bumps,
        survival_ticks=env._tick,
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
        mean_thoughtlet_variance=mean_var,
        mean_thoughtlet_persistence=mean_persist,
        mean_thoughtlet_effective_rank=mean_rank,
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
        scaled_model = IreneBrainModel(scaled_config)
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
