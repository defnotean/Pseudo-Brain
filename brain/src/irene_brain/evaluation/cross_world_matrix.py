"""Cross-world diagnostic matrix: the same policies on every ladder world.

With the in-repo environment ladder complete (moving shapes, pursuit,
junction, occlusion, keys/doors, maze_chase), every future model row needs
the same floor/reference context on every world. This module runs the
non-privileged PLAN.md §28 diagnostic policies across every registered
world slot through the closed-loop evaluator and assembles one canonical
evidence record.

The privileged oracle is deliberately excluded: it binds moving-shapes
internals only, and privileged rows on the wrong world would be meaningless
rather than merely unflattering. The scripted chaser reads the shared public
render contract, so it runs everywhere unmodified — its rows on the maze
worlds (where walls defeat greedy pursuit of the target pixel) and on the
occlusion world (where the target is usually invisible) are diagnostic
signal, not bugs. The world-specific scripted teachers hold still on
foreign worlds, so their rows there are honest zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Sequence

from ..environments.junction import JunctionEnv
from ..environments.keys_doors import KeysDoorsEnv
from ..environments.moving_shapes import MovingShapesEnv
from ..environments.occlusion import OcclusionEnv
from ..environments.pursuit import PursuitEnv
from ..environments.maze_chase import MazeChaseEnv
from .closed_loop_play import (
    ClosedLoopPlayConfig,
    ClosedLoopPlayReport,
    _canonical_json,
    run_policy_closed_loop_episode,
)
from .diagnostic_policies import (
    NoOpPolicy,
    RandomMovementPolicy,
    ScriptedJunctionSolver,
    ScriptedKeysDoorsSolver,
    ScriptedMazeChasePlannerPolicy,
    ScriptedOcclusionMemoryPolicy,
    ScriptedPelletTeacherPolicy,
    ScriptedTargetChasePolicy,
)


@dataclass(frozen=True, slots=True)
class WorldSlot:
    """One ladder world with its canonical matrix configuration."""

    identity: str
    factory: Callable[[ClosedLoopPlayConfig], object]


def default_world_slots() -> tuple[WorldSlot, ...]:
    """The in-repo worlds in canonical matrix order.

    The first six slots are the ladder worlds in their canonical
    configurations. The remaining slots promote each registered maze_chase
    variant axis (ghost AI rules, speed curves, sticky/delayed input) to its
    own world slot so the matrix measures how every policy degrades when the
    mechanics move away from the canonical slot. Variant slots are appended
    only — never reordered — so historical cells stay byte-identical.
    """

    def maze_chase_variant(**overrides: object) -> Callable[[ClosedLoopPlayConfig], object]:
        def factory(config: ClosedLoopPlayConfig) -> object:
            knobs: dict[str, object] = {
                "ghost_count": 3,
                "ghost_period": 2,
                "extra_loops": 16,
                "tick_period_ns": config.tick_period_ns,
                "max_ticks": config.max_ticks,
            }
            knobs.update(overrides)
            return MazeChaseEnv(**knobs)  # type: ignore[arg-type]

        return factory

    return (
        WorldSlot(
            "world.moving_shapes.v1",
            lambda config: MovingShapesEnv(
                hazard_count=config.hazard_count,
                tick_period_ns=config.tick_period_ns,
                max_ticks=config.max_ticks,
            ),
        ),
        WorldSlot(
            "world.pursuit.v1",
            lambda config: PursuitEnv(
                pursuer_count=2,
                pursuer_period=2,
                tick_period_ns=config.tick_period_ns,
                max_ticks=config.max_ticks,
            ),
        ),
        WorldSlot(
            "world.junction.v1",
            lambda config: JunctionEnv(
                chaser_count=1,
                chaser_period=2,
                tick_period_ns=config.tick_period_ns,
                max_ticks=config.max_ticks,
            ),
        ),
        WorldSlot(
            "world.occlusion.v1",
            lambda config: OcclusionEnv(
                hazard_count=config.hazard_count,
                view_radius=4,
                tick_period_ns=config.tick_period_ns,
                max_ticks=config.max_ticks,
            ),
        ),
        WorldSlot(
            "world.keys_doors.v1",
            lambda config: KeysDoorsEnv(
                tick_period_ns=config.tick_period_ns,
                max_ticks=config.max_ticks,
            ),
        ),
        WorldSlot(
            "world.maze_chase.v1",
            maze_chase_variant(),
        ),
        WorldSlot(
            "world.maze_chase.ambush.v1",
            maze_chase_variant(ghost_rule="ambush"),
        ),
        WorldSlot(
            "world.maze_chase.shy.v1",
            maze_chase_variant(ghost_rule="shy"),
        ),
        WorldSlot(
            "world.maze_chase.mixed.v1",
            maze_chase_variant(ghost_rule="mixed"),
        ),
        WorldSlot(
            "world.maze_chase.elroy.v1",
            maze_chase_variant(ghost_elroy=True),
        ),
        WorldSlot(
            "world.maze_chase.slow_player.v1",
            maze_chase_variant(player_period=2),
        ),
        WorldSlot(
            "world.maze_chase.delayed_input.v1",
            maze_chase_variant(input_delay_ticks=2),
        ),
        WorldSlot(
            "world.maze_chase.sticky.v1",
            maze_chase_variant(sticky_direction=True),
        ),
    )


def matrix_policies() -> tuple[object, ...]:
    """The non-privileged §28 diagnostics in canonical matrix order."""

    return (
        NoOpPolicy(),
        RandomMovementPolicy(),
        ScriptedTargetChasePolicy(),
        ScriptedPelletTeacherPolicy(),
        ScriptedMazeChasePlannerPolicy(),
        ScriptedKeysDoorsSolver(),
        ScriptedJunctionSolver(),
        ScriptedOcclusionMemoryPolicy(),
    )


@dataclass(frozen=True, slots=True)
class CrossWorldMatrixReport:
    """Canonical evidence record for one policy × world matrix."""

    schema_version: int
    config: ClosedLoopPlayConfig
    cells: tuple[tuple[str, str, ClosedLoopPlayReport], ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported cross-world matrix schema")
        seen = set()
        for world_identity, policy_identity, report in self.cells:
            key = (world_identity, policy_identity)
            if key in seen:
                raise ValueError("duplicate matrix cell")
            seen.add(key)
            if report.config != self.config:
                raise ValueError("cell reports must share the matrix config")

    def to_dict(self) -> dict[str, object]:
        return {
            "cells": [
                {
                    "policy": policy_identity,
                    "report": report.to_dict(),
                    "report_sha256": report.sha256,
                    "world": world_identity,
                }
                for world_identity, policy_identity, report in self.cells
            ],
            "config": self.config.to_dict(),
            "schema_version": self.schema_version,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def cell(self, world_identity: str, policy_identity: str) -> ClosedLoopPlayReport:
        for world, policy, report in self.cells:
            if world == world_identity and policy == policy_identity:
                return report
        raise KeyError((world_identity, policy_identity))


def evaluate_cross_world_matrix(
    *,
    config: ClosedLoopPlayConfig,
    worlds: Sequence[WorldSlot] | None = None,
    policies: Sequence[object] | None = None,
) -> CrossWorldMatrixReport:
    """Run every policy on every world over the registered seeds."""

    if not isinstance(config, ClosedLoopPlayConfig):
        raise ValueError("config must be a ClosedLoopPlayConfig")
    world_slots = tuple(worlds) if worlds is not None else default_world_slots()
    policy_list = list(policies) if policies is not None else list(matrix_policies())
    if not world_slots:
        raise ValueError("worlds cannot be empty")
    if not policy_list:
        raise ValueError("policies cannot be empty")
    world_identities = [slot.identity for slot in world_slots]
    if len(set(world_identities)) != len(world_identities):
        raise ValueError("world identities must be unique")
    for policy in policy_list:
        if getattr(policy, "uses_privileged_state", False):
            raise ValueError("privileged policies are excluded from the matrix")

    cells = []
    for slot in world_slots:
        for policy in policy_list:
            episodes = tuple(
                run_policy_closed_loop_episode(
                    policy,
                    seed=seed,
                    config=config,
                    environment_factory=lambda slot=slot: slot.factory(config),
                )
                for seed in config.episode_seeds
            )
            cells.append(
                (
                    slot.identity,
                    policy.identity,
                    ClosedLoopPlayReport(
                        schema_version=1,
                        config=config,
                        model_description=(
                            f"diagnostic policy {policy.identity}"
                            f" on {slot.identity}"
                        ),
                        episodes=episodes,
                    ),
                )
            )
    return CrossWorldMatrixReport(
        schema_version=1, config=config, cells=tuple(cells)
    )


__all__ = [
    "CrossWorldMatrixReport",
    "WorldSlot",
    "default_world_slots",
    "evaluate_cross_world_matrix",
    "matrix_policies",
]
