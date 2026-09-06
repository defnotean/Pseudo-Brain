"""Hidden Rule POMDP Environment for Rapid Online Adaptation.

A controlled 16x16 POMDP environment designed to test whether an agent can:
1. Encounter an ambiguous situation where current visual observation alone cannot
   determine the optimal action.
2. Experience the outcome (reward vs penalty) of its chosen action.
3. Form a fast associative memory/plastic state without backpropagation.
4. Retain and exploit that association in subsequent trials.
5. Adapt rapidly when the hidden rule reverses (Rule A -> Rule B -> Rule A).

Anti-cheating guarantee:
- Visual cues, corridors, and doors are 100% identical between Rule A and Rule B.
- No simulator coordinates, rule IDs, or privileged state are exposed in Observation.
- Only RGB pixels (16x16) and previous action are available to the agent.
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from irene_brain.types import GenericControl, HidKey, Observation, RgbFrame, StepOutcome


class Rule(str, Enum):
    RULE_A = "A"  # Cue -> LEFT is rewarded (+1.0), RIGHT is penalized (-0.5)
    RULE_B = "B"  # Cue -> RIGHT is rewarded (+1.0), LEFT is penalized (-0.5)


class HiddenRuleEnv:
    """A deterministic 16x16 multi-trial POMDP environment with hidden rules."""

    GRID_SIZE = 16
    DEFAULT_TICK_PERIOD_NS = 16_666_667

    # Palette
    WALL_RGB = (42, 48, 66)
    BACKGROUND_EVEN = (8, 11, 18)
    BACKGROUND_ODD = (10, 14, 22)
    PLAYER_RGB = (65, 174, 255)       # Cyan-Blue
    CUE_RGB = (180, 80, 220)          # Purple cue landmark
    DOOR_RGB = (170, 110, 40)         # Orange door (identical appearance Left & Right)
    TARGET_SUCCESS_RGB = (250, 206, 55)# Yellow flash on success
    FAILURE_RGB = (220, 50, 50)       # Red flash on collision/penalty

    # Key mappings
    _KEY_BITS = {
        int(HidKey.W): 1 << 0,
        int(HidKey.A): 1 << 1,
        int(HidKey.S): 1 << 2,
        int(HidKey.D): 1 << 3,
    }

    def __init__(
        self,
        *,
        rule_schedule: Optional[List[Tuple[Rule, int]]] = None,
        max_ticks_per_trial: int = 40,
        feedback_dwell_ticks: int = 1,
    ):
        if rule_schedule is None:
            # Default: 10 trials Rule A, 10 trials Rule B, 10 trials Rule A (30 trials total)
            self.rule_schedule = [
                (Rule.RULE_A, 10),
                (Rule.RULE_B, 10),
                (Rule.RULE_A, 10),
            ]
        else:
            self.rule_schedule = rule_schedule

        self.max_ticks_per_trial = max_ticks_per_trial
        self.feedback_dwell_ticks = feedback_dwell_ticks

        # Build schedule list: trial_idx -> Rule
        self._flattened_schedule: List[Rule] = []
        for r, count in self.rule_schedule:
            self._flattened_schedule.extend([r] * count)
        self.total_trials = len(self._flattened_schedule)

        # Environment geometry
        # Vertical corridor: x=7, y in [7..13]
        # Horizontal hallway: y=7, x in [3..11]
        # Cue at (7, 6)
        # Left door at (3, 7)
        # Right door at (11, 7)
        self._corridor = set()
        for y in range(7, 14):
            self._corridor.add((7, y))
        for x in range(3, 12):
            self._corridor.add((x, 7))

        self.cue_pos = (7, 6)
        self.left_door_pos = (3, 7)
        self.right_door_pos = (11, 7)
        self.start_pos = (7, 13)

        # Dynamic state
        self._tick = 0
        self._trial_idx = 0
        self._trial_tick = 0
        self._player_x = self.start_pos[0]
        self._player_y = self.start_pos[1]
        self._prev_key_mask = 0

        # Consequence feedback display
        self._feedback_mode: Optional[str] = None  # None, success_left, success_right, fail_left, fail_right
        self._feedback_timer = 0

        # Stats
        self.trial_outcomes: List[dict] = []

    @property
    def current_rule(self) -> Rule:
        if self._trial_idx < len(self._flattened_schedule):
            return self._flattened_schedule[self._trial_idx]
        return self._flattened_schedule[-1]

    @property
    def current_trial(self) -> int:
        return self._trial_idx

    def reset(self, seed: int = 0) -> Observation:
        """Reset the environment to trial 0 and initial positions."""
        self._tick = 0
        self._trial_idx = 0
        self._trial_tick = 0
        self._player_x, self._player_y = self.start_pos
        self._prev_key_mask = 0
        self._feedback_mode = None
        self._feedback_timer = 0
        self.trial_outcomes = []
        return self.current_observation

    def _reset_player_for_next_trial(self) -> None:
        self._trial_idx += 1
        self._trial_tick = 0
        self._player_x, self._player_y = self.start_pos
        self._feedback_mode = None
        self._feedback_timer = 0

    @property
    def current_observation(self) -> Observation:
        rgb = self._render()
        control = self._control_from_mask(self._prev_key_mask)
        return Observation(
            frame_id=self._tick,
            capture_tick=self._tick,
            elapsed_ns=self._tick * self.DEFAULT_TICK_PERIOD_NS,
            rgb=rgb,
            previous_control=control,
        )

    def step(self, control: GenericControl) -> StepOutcome:
        self._tick += 1
        self._trial_tick += 1

        applied_mask = self._mask_from_control(control)
        self._prev_key_mask = applied_mask
        applied_control = self._control_from_mask(applied_mask)

        reward = 0.0
        events: List[str] = []
        terminated = False
        truncated = False

        # If currently dwelling in feedback frame:
        if self._feedback_mode is not None:
            self._feedback_timer -= 1
            if self._feedback_timer <= 0:
                self._reset_player_for_next_trial()
                if self._trial_idx >= self.total_trials:
                    terminated = True

            return StepOutcome(
                observation=self.current_observation,
                requested_control=control,
                applied_control=applied_control,
                reward=reward,
                events=tuple(events),
                terminated=terminated,
                truncated=truncated,
            )

        # Compute proposed player movement
        dx = int(bool(applied_mask & self._KEY_BITS[int(HidKey.D)])) - int(
            bool(applied_mask & self._KEY_BITS[int(HidKey.A)])
        )
        dy = int(bool(applied_mask & self._KEY_BITS[int(HidKey.S)])) - int(
            bool(applied_mask & self._KEY_BITS[int(HidKey.W)])
        )

        nx = self._player_x + dx
        ny = self._player_y + dy

        # Check movement target
        if (nx, ny) in self._corridor:
            self._player_x = nx
            self._player_y = ny

        # Check if player reached left door or right door
        player = (self._player_x, self._player_y)
        rule = self.current_rule

        if player == self.left_door_pos:
            # Player chose LEFT
            chosen = "LEFT"
            if rule == Rule.RULE_A:
                reward = 1.0
                events.append("trial_success")
                self._feedback_mode = "success_left"
                is_correct = True
            else:
                reward = -0.5
                events.append("trial_failure")
                self._feedback_mode = "fail_left"
                is_correct = False

            self._feedback_timer = self.feedback_dwell_ticks
            self.trial_outcomes.append({
                "trial": self._trial_idx,
                "rule": rule.value,
                "choice": chosen,
                "correct": is_correct,
                "ticks": self._trial_tick,
                "reward": reward,
            })

        elif player == self.right_door_pos:
            # Player chose RIGHT
            chosen = "RIGHT"
            if rule == Rule.RULE_B:
                reward = 1.0
                events.append("trial_success")
                self._feedback_mode = "success_right"
                is_correct = True
            else:
                reward = -0.5
                events.append("trial_failure")
                self._feedback_mode = "fail_right"
                is_correct = False

            self._feedback_timer = self.feedback_dwell_ticks
            self.trial_outcomes.append({
                "trial": self._trial_idx,
                "rule": rule.value,
                "choice": chosen,
                "correct": is_correct,
                "ticks": self._trial_tick,
                "reward": reward,
            })

        elif self._trial_tick >= self.max_ticks_per_trial:
            # Timeout on this trial
            events.append("trial_timeout")
            reward = -0.2
            self.trial_outcomes.append({
                "trial": self._trial_idx,
                "rule": rule.value,
                "choice": "TIMEOUT",
                "correct": False,
                "ticks": self._trial_tick,
                "reward": reward,
            })
            self._reset_player_for_next_trial()
            if self._trial_idx >= self.total_trials:
                terminated = True

        return StepOutcome(
            observation=self.current_observation,
            requested_control=control,
            applied_control=applied_control,
            reward=reward,
            events=tuple(events),
            terminated=terminated,
            truncated=truncated,
        )

    def _render(self) -> RgbFrame:
        """Render 16x16 RGB image strictly from visual primitives."""
        pixels = bytearray(self.GRID_SIZE * self.GRID_SIZE * 3)

        # 1. Background / walls
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) in self._corridor:
                    color = self.BACKGROUND_EVEN if (x + y) % 2 == 0 else self.BACKGROUND_ODD
                else:
                    color = self.WALL_RGB
                self._paint(pixels, x, y, color)

        # 2. Visual Cue (at top of junction)
        self._paint(pixels, self.cue_pos[0], self.cue_pos[1], self.CUE_RGB)

        # 3. Left and Right Doors
        # In neutral state, both are identical Orange
        left_color = self.DOOR_RGB
        right_color = self.DOOR_RGB

        if self._feedback_mode == "success_left":
            left_color = self.TARGET_SUCCESS_RGB
        elif self._feedback_mode == "fail_left":
            left_color = self.FAILURE_RGB
        elif self._feedback_mode == "success_right":
            right_color = self.TARGET_SUCCESS_RGB
        elif self._feedback_mode == "fail_right":
            right_color = self.FAILURE_RGB

        self._paint(pixels, self.left_door_pos[0], self.left_door_pos[1], left_color)
        self._paint(pixels, self.right_door_pos[0], self.right_door_pos[1], right_color)

        # 4. Player
        self._paint(pixels, self._player_x, self._player_y, self.PLAYER_RGB)

        return RgbFrame(width=self.GRID_SIZE, height=self.GRID_SIZE, pixels=bytes(pixels))

    def _paint(self, pixels: bytearray, x: int, y: int, color: Tuple[int, int, int]) -> None:
        offset = (y * self.GRID_SIZE + x) * 3
        pixels[offset : offset + 3] = bytes(color)

    @classmethod
    def _mask_from_control(cls, control: GenericControl) -> int:
        mask = 0
        for key in control.keys_down:
            mask |= cls._KEY_BITS.get(key, 0)
        return mask

    @classmethod
    def _control_from_mask(cls, mask: int) -> GenericControl:
        keys = tuple(key for key, bit in cls._KEY_BITS.items() if mask & bit)
        return GenericControl(keys_down=keys)


__all__ = ["HiddenRuleEnv", "Rule"]
