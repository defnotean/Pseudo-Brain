"""Perturbation engines and diagnostic tracking for self-correction experiments.

Injects controlled mistakes during closed-loop policy execution:
1. SingleStepPerturbation: Forces an incorrect move at a specified or random tick.
2. BurstPerturbation: Forces a 3-step random deviation.
3. SlipPerturbation: Drops the attempted action into Idle (movement failure).

Tracks diagnostic metrics:
- recovery_rate: Proportion of perturbed episodes where the policy successfully recovers and reaches the target.
- mean_recovery_steps: Steps taken after perturbation to return to the optimal shortest path.
- failure_rate: Proportion of episodes ending in wall-deadlock or timeout after perturbation.
"""
from __future__ import annotations

from typing import Tuple, List, Dict, Optional
import numpy as np

from irene_brain.environments.keys_doors import KeysDoorsEnv, _bfs_distances
from memory_benchmark.expert import (
    ACTION_IDLE,
    ACTION_UP,
    ACTION_LEFT,
    ACTION_DOWN,
    ACTION_RIGHT,
    control_for_action,
)

ALL_ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT]


class PerturbationEngine:
    """Manages perturbation scheduling and diagnostic tracking per episode."""

    def __init__(
        self,
        mode: str = "single",  # "none", "single", "burst", "slip"
        inject_tick: Optional[int] = None,
        burst_len: int = 3,
        seed: int = 42,
    ):
        self.mode = mode
        self.inject_tick = inject_tick
        self.burst_len = burst_len
        self.rng = np.random.RandomState(seed)

        self.injected = False
        self.injection_active = False
        self.injection_start_tick: Optional[int] = None
        self.injection_end_tick: Optional[int] = None
        self.recovered = False
        self.recovery_tick: Optional[int] = None

    def should_inject(self, tick: int) -> bool:
        if self.mode == "none":
            return False
        if self.injected and not self.injection_active:
            return False

        if not self.injected:
            # Trigger injection if specified tick reached, or randomly between tick 15 and 30
            target_tick = self.inject_tick if self.inject_tick is not None else 20
            if tick >= target_tick:
                self.injected = True
                self.injection_active = True
                self.injection_start_tick = tick
                return True
            return False

        # Active burst
        if self.injection_active:
            if self.mode == "single" or self.mode == "slip":
                self.injection_active = False
                self.injection_end_tick = tick
                return False
            elif self.mode == "burst":
                if tick - self.injection_start_tick < self.burst_len:
                    return True
                else:
                    self.injection_active = False
                    self.injection_end_tick = tick
                    return False
        return False

    def get_perturbed_action(self, proposed_action: int) -> int:
        if self.mode == "slip":
            return ACTION_IDLE
        # Choose a valid non-idle action different from the agent's proposed action
        candidates = [a for a in ALL_ACTIONS if a != proposed_action]
        return int(self.rng.choice(candidates))

    def update_recovery_state(self, env: KeysDoorsEnv, tick: int):
        """Check if agent has returned to the optimal path to current goal."""
        if not self.injected or self.recovered:
            return

        # Once perturbation ends, track whether agent is on the optimal path
        end_tick = self.injection_end_tick if self.injection_end_tick is not None else self.injection_start_tick
        if end_tick is None or tick <= end_tick:
            return

        player = (env._player_x, env._player_y)
        if not env._has_key:
            goal = (env._key_x, env._key_y)
            valid_maze = env._maze - {(env._door_x, env._door_y)}
        elif not env._door_open:
            goal = (env._door_x, env._door_y)
            valid_maze = env._maze
        else:
            goal = (env._target_x, env._target_y)
            valid_maze = env._maze

        dist = _bfs_distances(goal, valid_maze)
        current_d = dist.get(player)

        # Agent is making progress if it has a valid finite path and hasn't stalled
        if current_d is not None:
            # If agent reached the goal or is within normal corridor distance
            if player == goal or env._has_key or env._door_open:
                self.recovered = True
                self.recovery_tick = tick


def get_goal_distance(env: KeysDoorsEnv) -> Optional[int]:
    player = (env._player_x, env._player_y)
    if not env._has_key:
        goal = (env._key_x, env._key_y)
        valid_maze = env._maze - {(env._door_x, env._door_y)}
    elif not env._door_open:
        goal = (env._door_x, env._door_y)
        valid_maze = env._maze
    else:
        goal = (env._target_x, env._target_y)
        valid_maze = env._maze

    dist = _bfs_distances(goal, valid_maze)
    return dist.get(player)
