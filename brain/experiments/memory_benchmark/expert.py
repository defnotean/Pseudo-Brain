"""Optimal BFS teacher for KeysDoorsEnv.

Generates ground-truth optimal trajectories for sequential key-to-door-to-target navigation.
Stages:
1. Player -> Key (door is blocked)
2. Player -> Door (door opens upon contact while holding key)
3. Player -> Target (maze fully open)
"""
from __future__ import annotations

from typing import Tuple, List, Optional
from irene_brain.environments.keys_doors import KeysDoorsEnv, _bfs_distances, _DIRECTIONS
from irene_brain.types import GenericControl, HidKey, StepOutcome

# Action constants
ACTION_IDLE = 0
ACTION_UP = 1     # W
ACTION_LEFT = 2   # A
ACTION_DOWN = 3   # S
ACTION_RIGHT = 4  # D

ACTION_TO_KEYS = {
    ACTION_IDLE: (),
    ACTION_UP: (HidKey.W,),
    ACTION_LEFT: (HidKey.A,),
    ACTION_DOWN: (HidKey.S,),
    ACTION_RIGHT: (HidKey.D,),
}

def control_for_action(action: int) -> GenericControl:
    return GenericControl(keys_down=ACTION_TO_KEYS[action])


class KeysDoorsExpert:
    """Expert navigator that computes optimal moves at each tick."""

    def __init__(self, env: KeysDoorsEnv):
        self.env = env

    def get_action(self) -> int:
        player = (self.env._player_x, self.env._player_y)

        # Stage 1: Key not yet collected -> target the key, door is a wall
        if not self.env._has_key:
            goal = (self.env._key_x, self.env._key_y)
            valid_maze = self.env._maze - {(self.env._door_x, self.env._door_y)}
        # Stage 2: Key collected, door closed -> target the door
        elif not self.env._door_open:
            goal = (self.env._door_x, self.env._door_y)
            valid_maze = self.env._maze
        # Stage 3: Door open -> target the yellow target
        else:
            goal = (self.env._target_x, self.env._target_y)
            valid_maze = self.env._maze

        if player == goal:
            return ACTION_IDLE

        dist = _bfs_distances(goal, valid_maze)
        current_d = dist.get(player)
        if current_d is None:
            # Fallback if somehow unreachable
            return ACTION_IDLE

        # Find neighboring step with distance == current_d - 1
        # Order: W, A, S, D
        for dx, dy, act in [(0, -1, ACTION_UP), (-1, 0, ACTION_LEFT), (0, 1, ACTION_DOWN), (1, 0, ACTION_RIGHT)]:
            nbr = (player[0] + dx, player[1] + dy)
            if dist.get(nbr) == current_d - 1:
                return act

        return ACTION_IDLE

    def run_episode(self, seed: int, max_ticks: int = 300) -> Tuple[List[dict], bool]:
        """Run an episode to completion (first target collected).

        Returns:
            transitions: list of dicts with:
                - rgb: bytes
                - prev_action: int
                - action: int
                - reward: float
                - has_key: int (diagnostic only, not passed to model observation)
                - door_open: int (diagnostic only)
            success: bool (whether target was collected)
        """
        obs = self.env.reset(seed)
        transitions = []
        prev_act = ACTION_IDLE
        success = False

        for tick in range(max_ticks):
            act = self.get_action()
            ctrl = control_for_action(act)

            # Record transition before stepping
            transitions.append({
                "rgb": obs.rgb.pixels,
                "prev_action": prev_act,
                "action": act,
                "has_key": self.env._has_key,
                "door_open": self.env._door_open,
                "tick": tick,
            })

            outcome = self.env.step(ctrl)
            obs = outcome.observation
            prev_act = act

            if outcome.reward > 0.0 or "target_collected" in outcome.events:
                success = True
                break

        return transitions, success
