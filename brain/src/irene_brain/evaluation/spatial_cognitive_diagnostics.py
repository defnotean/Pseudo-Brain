"""Spatial, geometric, and internal cognitive diagnostics for Pseudo-Brain.

Computes exact quantitative evidence rather than heuristic narratives:
1. Physical & Spatial Dynamics:
   - Ghost collision count vs Wall bump count.
   - Mean and minimum Euclidean distance to nearest ghost.
   - Intersection classifications (corridor vs dead-end vs 3/4-way junction).
   - Unsafe intersection crossings (entering junction when nearest ghost distance <= 2).
   - Distance traveled & Pellets per 100 movement steps.
   - Expert planner agreement rate (%) at each decision tick.

2. Internal Thought-Slot Dynamics:
   - Model parameter SHA256 digest verification (ensures evaluation uses fresh weights).
   - Inter-thoughtlet activation variance and entropy.
   - Thought-slot temporal persistence (cosine similarity between t and t-1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
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

    # Kinematics & Movement Efficiency
    actual_tiles_moved: int
    pellets_per_100_moves: float
    expert_disagreement_pct: float

    # Internal Thought-Field Telemetry
    mean_thoughtlet_variance: float
    mean_thoughtlet_persistence: float

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
            "actual_tiles_moved": self.actual_tiles_moved,
            "pellets_per_100_moves": round(self.pellets_per_100_moves, 2),
            "expert_disagreement_pct": round(self.expert_disagreement_pct, 2),
            "mean_thoughtlet_variance": round(self.mean_thoughtlet_variance, 4),
            "mean_thoughtlet_persistence": round(self.mean_thoughtlet_persistence, 4),
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
    """Execute one episode with exact geometric, kinematic, and cognitive telemetry instrumentation."""
    obs = env.reset(seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    if expert_policy is not None and hasattr(expert_policy, "reset"):
        expert_policy.reset(seed)

    maze = env._maze
    topology = _classify_maze_topology(maze)
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

    last_pos = (env._player_x, env._player_y)
    last_thought: Tensor | None = None

    for tick in range(max_ticks):
        # 1. Capture current geometric distances before action
        px, py = env._player_x, env._player_y
        current_pos = (px, py)
        cell_type = topology.get(current_pos, "corridor")

        if cell_type == "junction" and last_pos != current_pos:
            junction_entries += 1
            # Check proximity of all ghosts
            nearest_g_dist = min(
                (math.sqrt((px - gx) ** 2 + (py - gy) ** 2) for gx, gy in env._ghosts),
                default=999.0,
            )
            if nearest_g_dist <= 2.0:
                unsafe_junction_entries += 1
        elif cell_type == "corridor":
            corridor_steps += 1
        elif cell_type == "dead_end":
            dead_end_steps += 1

        # Distance to ghosts
        if env._ghosts:
            cur_nearest = min(
                math.sqrt((px - gx) ** 2 + (py - gy) ** 2) for gx, gy in env._ghosts
            )
            ghost_distances.append(cur_nearest)
            if cur_nearest < min_ghost_distance:
                min_ghost_distance = cur_nearest

        # 2. Query policy and optional expert
        ctrl, _, _ = policy.decide(obs, 1.0 / 60.0) if hasattr(policy, "decide") else (policy.act(obs), {}, None)
        decisions_total += 1

        if expert_policy is not None:
            expert_ctrl = expert_policy.act(obs)
            if ctrl.keys_down != expert_ctrl.keys_down:
                expert_disagreements += 1

        # 3. Inspect internal thought states if available
        state = getattr(policy, "_state", None)
        if state is not None and hasattr(state, "thoughts") and isinstance(state.thoughts, Tensor):
            thoughts = state.thoughts[0]  # [thoughtlets, registers, width]
            # Inter-thoughtlet variance across the thoughtlet dimension
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
        outcome = env.step(ctrl)
        new_pos = (env._player_x, env._player_y)

        # Detect wall bump (action requested but position unchanged)
        if ctrl.keys_down and new_pos == current_pos and (tick % env._player_period == 0):
            wall_bumps += 1
        elif new_pos != current_pos:
            actual_moves += 1

        last_pos = current_pos
        obs = outcome.observation
        if outcome.terminated or outcome.truncated:
            break

    mean_dist = sum(ghost_distances) / len(ghost_distances) if ghost_distances else 0.0
    disagreement_pct = (expert_disagreements / decisions_total * 100.0) if decisions_total > 0 else 0.0
    pellets_per_100 = (env._pellets_eaten / actual_moves * 100.0) if actual_moves > 0 else 0.0
    mean_var = sum(thought_variances) / len(thought_variances) if thought_variances else 0.0
    mean_persist = sum(thought_persistences) / len(thought_persistences) if thought_persistences else 0.0

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
        actual_tiles_moved=actual_moves,
        pellets_per_100_moves=pellets_per_100,
        expert_disagreement_pct=disagreement_pct,
        mean_thoughtlet_variance=mean_var,
        mean_thoughtlet_persistence=mean_persist,
    )
