"""PLAN.md §28 diagnostic policies for the closed-loop moving-shapes world.

The baseline suite requires every phase to report the same evaluator rows for
non-model reference policies: random and no-op policies (§28.1), a simple
scripted diagnostic policy (§28.2), and a privileged-state oracle for
diagnosis only (§28.15). This module implements those four decision sources
behind one tiny contract so they run through
:func:`irene_brain.evaluation.closed_loop_play.run_policy_closed_loop_episode`
with exactly the same clock, deadline, and rejection mechanics as a model.

Policy contract:

- ``identity`` — a stable dotted identifier recorded in the evidence report.
- ``uses_privileged_state`` — True only for the oracle. Privileged policies
  receive the live environment through ``bind(environment)`` after reset and
  may read simulator internals. Everything they produce is diagnosis-only
  evidence and must never be routed into a model input or training target.
- ``reset(episode_seed)`` — reseed per episode so multi-seed campaigns stay
  deterministic and independent.
- ``act(observation)`` — return the next :class:`GenericControl`.

The scripted chaser is deliberately pixel-only: it reads the public
render-contract colors (``PLAYER_RGB`` / ``TARGET_RGB``) from the canonical
observation frame and never touches simulator state, demonstrating that a
procedural teacher needs nothing beyond what a model would see.
"""

from __future__ import annotations

from typing import Sequence

from ..environments.moving_shapes import (
    MovingShapesEnv,
    _SplitMix64,
    _paths_collide_at_same_time,
)
from ..types import GenericControl, Observation
from .closed_loop_play import (
    ClosedLoopPlayConfig,
    ClosedLoopPlayReport,
    run_policy_closed_loop_episode,
)

# HID usage identifiers, bit-packed exactly like MovingShapesEnv._KEY_BITS.
_KEY_W = 26
_KEY_A = 4
_KEY_S = 22
_KEY_D = 7
_KEY_FOR_DELTA = {(0, -1): _KEY_W, (-1, 0): _KEY_A, (0, 1): _KEY_S, (1, 0): _KEY_D}


def _movement_control(mask: int) -> GenericControl:
    keys = []
    if mask & 1:
        keys.append(_KEY_W)
    if mask & 2:
        keys.append(_KEY_A)
    if mask & 4:
        keys.append(_KEY_S)
    if mask & 8:
        keys.append(_KEY_D)
    return GenericControl(keys_down=tuple(keys))


class NoOpPolicy:
    """§28.1 no-op floor: never presses anything."""

    __slots__ = ()
    identity = "diagnostic.noop.v1"
    uses_privileged_state = False

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")

    def act(self, observation: Observation) -> GenericControl:
        return GenericControl()


class RandomMovementPolicy:
    """§28.1 random floor: a uniform W/A/S/D subset per decision."""

    __slots__ = ("_rng",)
    identity = "diagnostic.random_movement.v1"
    uses_privileged_state = False

    def __init__(self) -> None:
        self._rng = _SplitMix64(0)

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        self._rng = _SplitMix64(episode_seed ^ 0x5D1A60571C001)

    def act(self, observation: Observation) -> GenericControl:
        return _movement_control(self._rng.randbelow(16))


class ScriptedTargetChasePolicy:
    """§28.2 scripted diagnostic: pixel-only greedy chase of the target.

    Locates the player and target cells by the public render-contract colors
    and presses the one or two movement keys that reduce the Manhattan
    distance. Hazards are ignored on purpose: this is the simplest teacher
    that solves the task's approach component, and its collision count is
    part of the diagnostic signal.
    """

    __slots__ = ()
    identity = "diagnostic.scripted_chase.v1"
    uses_privileged_state = False

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")

    def act(self, observation: Observation) -> GenericControl:
        frame = observation.rgb
        grid = MovingShapesEnv.GRID_SIZE
        if frame.width != grid or frame.height != grid:
            raise ValueError("scripted chaser requires the canonical grid frame")
        player: tuple[int, int] | None = None
        target: tuple[int, int] | None = None
        pixels = frame.pixels
        for y in range(grid):
            for x in range(grid):
                offset = (y * grid + x) * 3
                color = (
                    pixels[offset],
                    pixels[offset + 1],
                    pixels[offset + 2],
                )
                if color == MovingShapesEnv.TARGET_RGB:
                    target = (x, y)
                elif color in (
                    MovingShapesEnv.PLAYER_RGB,
                    MovingShapesEnv.PLAYER_HAZARD_OVERLAP_RGB,
                ):
                    player = (x, y)
        if player is None:
            raise RuntimeError("scripted chaser could not locate the player")
        if target is None:
            # The player is standing on the target cell (the player is painted
            # last, so the target is occluded for exactly this observation).
            # Hold still; the relocated target is visible again next tick.
            return GenericControl()
        keys = []
        if target[1] < player[1]:
            keys.append(_KEY_W)
        elif target[1] > player[1]:
            keys.append(_KEY_S)
        if target[0] < player[0]:
            keys.append(_KEY_A)
        elif target[0] > player[0]:
            keys.append(_KEY_D)
        return GenericControl(keys_down=tuple(keys))


class OraclePolicy:
    """§28.15 privileged-state oracle, for diagnosis only.

    Reads simulator internals (player, target, hazard positions and
    velocities), predicts the hazards' next cells with the exact bounce rule,
    and takes the safe move that minimizes Manhattan distance to the target.
    "Safe" reuses the environment's own swept-path collision test, so the
    oracle's collision count measures only unavoidable timing traps, never
    prediction error. Its outputs must never be routed into a model input or
    training target.
    """

    __slots__ = ("_environment",)
    identity = "diagnostic.oracle_privileged.v1"
    uses_privileged_state = True

    def __init__(self) -> None:
        self._environment: MovingShapesEnv | None = None

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("episode_seed must be an integer")
        self._environment = None

    def bind(self, environment: MovingShapesEnv) -> None:
        if not isinstance(environment, MovingShapesEnv):
            raise TypeError("oracle can only bind a MovingShapesEnv")
        self._environment = environment

    def act(self, observation: Observation) -> GenericControl:
        env = self._environment
        if env is None:
            raise RuntimeError("oracle act() before bind()")
        grid = MovingShapesEnv.GRID_SIZE
        player = (env._player_x, env._player_y)
        target = (env._target_x, env._target_y)
        hazard_paths = []
        for hazard in env._hazards:
            before = (hazard.x, hazard.y)
            next_x, vx = hazard.x + hazard.vx, hazard.vx
            if next_x < 0 or next_x >= grid:
                vx, next_x = -vx, hazard.x - hazard.vx
            next_y, vy = hazard.y + hazard.vy, hazard.vy
            if next_y < 0 or next_y >= grid:
                vy, next_y = -vy, hazard.y - hazard.vy
            hazard_paths.append((before, (next_x, next_y)))

        # Deterministic preference order: straight moves, stay, diagonals.
        candidates = (
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (0, 0),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        )

        def distance(delta: tuple[int, int]) -> int:
            new_x = min(grid - 1, max(0, player[0] + delta[0]))
            new_y = min(grid - 1, max(0, player[1] + delta[1]))
            return abs(new_x - target[0]) + abs(new_y - target[1])

        def safe(delta: tuple[int, int]) -> bool:
            new_x = min(grid - 1, max(0, player[0] + delta[0]))
            new_y = min(grid - 1, max(0, player[1] + delta[1]))
            after = (new_x, new_y)
            return not any(
                _paths_collide_at_same_time(player, after, before, hazard_after)
                for before, hazard_after in hazard_paths
            )

        safe_candidates = [delta for delta in candidates if safe(delta)]
        pool = safe_candidates if safe_candidates else list(candidates)
        chosen = min(pool, key=lambda delta: (distance(delta), candidates.index(delta)))
        keys = []
        if chosen[1] != 0:
            keys.append(_KEY_FOR_DELTA[(0, chosen[1])])
        if chosen[0] != 0:
            keys.append(_KEY_FOR_DELTA[(chosen[0], 0)])
        return GenericControl(keys_down=tuple(keys))


def default_diagnostic_policies() -> tuple[object, ...]:
    """Return the four §28 diagnostic policies in canonical suite order."""

    return (
        NoOpPolicy(),
        RandomMovementPolicy(),
        ScriptedTargetChasePolicy(),
        OraclePolicy(),
    )


def evaluate_diagnostic_policy_suite(
    policies: Sequence[object],
    *,
    config: ClosedLoopPlayConfig,
) -> tuple[ClosedLoopPlayReport, ...]:
    """Run each policy over every registered seed and return one report each.

    Reports share the exact ``ClosedLoopPlayReport`` schema the model
    evaluator emits, so diagnostic rows and model rows compare column for
    column. The oracle row is diagnosis-only evidence per PLAN.md §28.15.
    """

    if not isinstance(config, ClosedLoopPlayConfig):
        raise ValueError("config must be a ClosedLoopPlayConfig")
    if not policies:
        raise ValueError("policies cannot be empty")
    identities = [getattr(policy, "identity", None) for policy in policies]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("every policy must define a non-empty identity")
    if len(set(identities)) != len(identities):
        raise ValueError("policy identities must be unique")
    reports = []
    for policy, identity in zip(policies, identities):
        episodes = tuple(
            run_policy_closed_loop_episode(policy, seed=seed, config=config)
            for seed in config.episode_seeds
        )
        reports.append(
            ClosedLoopPlayReport(
                schema_version=1,
                config=config,
                model_description=f"diagnostic policy {identity}",
                episodes=episodes,
            )
        )
    return tuple(reports)


__all__ = [
    "NoOpPolicy",
    "OraclePolicy",
    "RandomMovementPolicy",
    "ScriptedTargetChasePolicy",
    "default_diagnostic_policies",
    "evaluate_diagnostic_policy_suite",
]
