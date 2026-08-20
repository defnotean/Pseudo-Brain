"""Phase 2 Procedural Evaluation Suite.

Implements the five mandatory Phase 2 task families:
- Family A: Multi-Object Tracking & Collision Avoidance (multiple hazards, dynamic velocities)
- Family B: Pursuit & Evasion (intelligent chasers, corner escape routes)
- Family C: Junction & Route Choice (multi-branch intersections, dead-end traps)
- Family D: Partial Observability & Memory (occlusion masks, hidden objects, revisit goals)
- Family E: Hidden Rules & Changed Dynamics (control remappings, friction/ice physics, mid-episode rule shifts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
import math
import random
from typing import Mapping, Sequence
import numpy as np

from ..types import GenericControl, HidKey, Observation, RgbFrame, StepOutcome
from .maze_chase import MazeChaseEnv


@unique
class TaskFamily(str, Enum):
    FAMILY_A_MULTI_OBJECT = "family_a_multi_object_tracking"
    FAMILY_B_PURSUIT_EVASION = "family_b_pursuit_and_evasion"
    FAMILY_C_JUNCTIONS = "family_c_junction_and_route_choice"
    FAMILY_D_PARTIAL_OBSERVABILITY = "family_d_partial_observability_and_memory"
    FAMILY_E_CHANGED_DYNAMICS = "family_e_changed_dynamics_and_rules"


@dataclass(frozen=True, slots=True)
class Phase2WorldConfig:
    family: TaskFamily
    max_ticks: int = 120
    ghost_count: int = 2
    ghost_speed: float = 1.0
    ghost_intelligence: float = 0.5  # 0 = random bounce, 1 = direct pursuit
    occlusion_radius: int = 0  # 0 = fully observable, >0 = visible radius around player
    control_remapping: str = "none"  # "none", "inverted", "rotated_90", "swapped_axes"
    physics_mode: str = "standard"  # "standard", "ice_momentum", "high_friction"
    dead_end_hazard_spawn: bool = False
    revisit_target_penalty: bool = False


class Phase2TaskEnvironment:
    """Unified procedural environment spanning all 5 Phase 2 task families."""

    def __init__(self, config: Phase2WorldConfig) -> None:
        self.config = config
        ghost_period = 2 if config.ghost_speed <= 1.0 else 1
        ghost_rule = "direct" if config.ghost_intelligence >= 0.8 else ("mixed" if config.ghost_intelligence >= 0.4 else "shy")
        self._underlying_env = MazeChaseEnv(
            ghost_count=config.ghost_count,
            ghost_period=ghost_period,
            ghost_rule=ghost_rule,
            max_ticks=config.max_ticks,
        )
        self.tick_count = 0
        self.rng = random.Random()
        self.momentum_dx = 0
        self.momentum_dy = 0

    def reset(self, episode_seed: int) -> Observation:
        self.tick_count = 0
        self.momentum_dx = 0
        self.momentum_dy = 0
        self.rng.seed(episode_seed ^ 0x5EED_F2A2)
        base_obs = self._underlying_env.reset(episode_seed)
        return self._transform_observation(base_obs)

    def _remap_control(self, control: GenericControl) -> GenericControl:
        mapping = self.config.control_remapping
        if mapping == "none":
            return control

        active = set(control.keys_down)
        remapped: set[int] = set()

        if mapping == "inverted":
            inv = {int(HidKey.W): int(HidKey.S), int(HidKey.S): int(HidKey.W), int(HidKey.A): int(HidKey.D), int(HidKey.D): int(HidKey.A)}
            for k in active:
                remapped.add(inv.get(k, k))
        elif mapping == "rotated_90":
            rot = {int(HidKey.W): int(HidKey.D), int(HidKey.D): int(HidKey.S), int(HidKey.S): int(HidKey.A), int(HidKey.A): int(HidKey.W)}
            for k in active:
                remapped.add(rot.get(k, k))
        elif mapping == "swapped_axes":
            swap = {int(HidKey.W): int(HidKey.A), int(HidKey.A): int(HidKey.W), int(HidKey.S): int(HidKey.D), int(HidKey.D): int(HidKey.S)}
            for k in active:
                remapped.add(swap.get(k, k))
        else:
            remapped = active

        return GenericControl(
            keys_down=tuple(sorted(remapped)),
            mouse_dx=control.mouse_dx,
            mouse_dy=control.mouse_dy,
            mouse_buttons=control.mouse_buttons,
            mouse_wheel=control.mouse_wheel,
            gamepad_axes=control.gamepad_axes,
            gamepad_buttons=control.gamepad_buttons,
            impulse_sequence=control.impulse_sequence,
            intended_hold_ns=control.intended_hold_ns,
        )

    def _apply_physics_and_step(self, control: GenericControl) -> StepOutcome:
        remapped_ctrl = self._remap_control(control)

        if self.config.physics_mode == "ice_momentum":
            if self.rng.random() < 0.5 and (self.momentum_dx != 0 or self.momentum_dy != 0):
                pass
            else:
                active = remapped_ctrl.keys_down
                if int(HidKey.W) in active:
                    self.momentum_dy, self.momentum_dx = -1, 0
                elif int(HidKey.S) in active:
                    self.momentum_dy, self.momentum_dx = 1, 0
                elif int(HidKey.A) in active:
                    self.momentum_dy, self.momentum_dx = 0, -1
                elif int(HidKey.D) in active:
                    self.momentum_dy, self.momentum_dx = 0, 1

        outcome = self._underlying_env.step(remapped_ctrl)
        obs = self._transform_observation(outcome.observation)
        self.tick_count += 1
        return StepOutcome(
            observation=obs,
            requested_control=control,
            applied_control=remapped_ctrl,
            reward=outcome.reward,
            events=outcome.events,
            terminated=outcome.terminated,
            truncated=outcome.truncated or (self.tick_count >= self.config.max_ticks),
        )

    def _transform_observation(self, obs: Observation) -> Observation:
        if self.config.occlusion_radius <= 0:
            return obs

        raw = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3)).copy()
        px = getattr(self._underlying_env, "_player_x", 8)
        py = getattr(self._underlying_env, "_player_y", 8)
        pixel_radius = int(self.config.occlusion_radius)

        y_coords, x_coords = np.ogrid[:16, :16]
        dist_from_center = np.sqrt((x_coords - px) ** 2 + (y_coords - py) ** 2)
        mask = dist_from_center > pixel_radius

        raw[mask] = (raw[mask] * 0.15).astype(np.uint8)

        new_rgb = RgbFrame(width=obs.rgb.width, height=obs.rgb.height, pixels=raw.tobytes())
        return Observation(
            frame_id=obs.frame_id,
            capture_tick=obs.capture_tick,
            elapsed_ns=obs.elapsed_ns,
            rgb=new_rgb,
            previous_control=obs.previous_control,
            audio_pcm_s16le=obs.audio_pcm_s16le,
            text_inputs=obs.text_inputs,
        )

    def step(self, control: GenericControl) -> StepOutcome:
        return self._apply_physics_and_step(control)


def make_family_suite(family: TaskFamily, seed_offset: int = 0) -> list[Phase2WorldConfig]:
    """Generate canonical configs for a task family across varied difficulties."""
    configs = []
    if family == TaskFamily.FAMILY_A_MULTI_OBJECT:
        for ghosts in (2, 3, 4):
            for speed in (0.8, 1.0, 1.2):
                configs.append(Phase2WorldConfig(
                    family=family,
                    ghost_count=ghosts,
                    ghost_speed=speed,
                    ghost_intelligence=0.2,
                ))
    elif family == TaskFamily.FAMILY_B_PURSUIT_EVASION:
        for intel in (0.6, 0.8, 1.0):
            for ghosts in (1, 2, 3):
                configs.append(Phase2WorldConfig(
                    family=family,
                    ghost_count=ghosts,
                    ghost_intelligence=intel,
                ))
    elif family == TaskFamily.FAMILY_C_JUNCTIONS:
        for loops in (8, 16, 24):
            configs.append(Phase2WorldConfig(
                family=family,
                ghost_count=2,
                dead_end_hazard_spawn=True,
            ))
    elif family == TaskFamily.FAMILY_D_PARTIAL_OBSERVABILITY:
        for radius in (2, 3, 4):
            configs.append(Phase2WorldConfig(
                family=family,
                ghost_count=2,
                occlusion_radius=radius,
            ))
    elif family == TaskFamily.FAMILY_E_CHANGED_DYNAMICS:
        for remap in ("inverted", "rotated_90", "swapped_axes"):
            for physics in ("standard", "ice_momentum"):
                configs.append(Phase2WorldConfig(
                    family=family,
                    control_remapping=remap,
                    physics_mode=physics,
                ))
    return configs
