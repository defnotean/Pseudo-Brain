"""Lazy deterministic maze_chase supervision from the frontier planner.

This mirrors :mod:`irene_brain.data.moving_shapes_dataset` for the
maze_chase ladder world: a virtual, lazily materialized dataset whose
action targets come from the pixel-only lookahead planner
(``diagnostic.scripted_maze_chase_planner.v1``) — the strongest scripted
policy on the canonical slot (3/3 clears, +426 reward over seeds 5/9/13).
The teacher is mechanics-matched: the world's ``ghost_period``,
``player_period``, ``input_delay_ticks``, and ``ghost_elroy`` knobs are
forwarded to the planner's actuation-awareness knobs, so the demonstrations
stay frontier-quality on every registered variant configuration the planner
can model. Non-direct ghost rules remain a documented planner blind spot;
sequences from those configurations are still valid demonstrations, just
optimistic ones.

Sealed-range status: no sealed TEST ranges are registered for maze_chase
(the RCQ-v2/v3 seals cover moving_shapes only). The split namespaces are the
same disjoint 62-bit tags, so train/validation/test episode seeds never
overlap. If a future maze_chase campaign seals a TEST range, a capability
guard like the moving_shapes one must be added here *before* any TEST
construction.

Only :class:`~irene_brain.types.ModelObservation` values are model inputs.
World seeds, split identities, rewards, events, and future observations
remain sequence metadata or explicit supervision targets.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import gcd
from struct import pack
from typing import Any, overload

from ..environments.maze_chase import _GHOST_RULES, MazeChaseEnv
from .moving_shapes_dataset import (
    DatasetSplit,
    MovingShapesSequence,
    MovingShapesTransition,
    _TransitionWithoutValue,
    _integer,
    _number,
    _split,
    split_episode_seed,
)

_SEED_NAMESPACE_SIZE = 1 << 62
_DATASET_SCHEMA_VERSION = 1
_GENERATOR_ID = "irene.maze_chase.planner_teacher.v1"
_LICENSE_RECORD_ID = "original-project-content"

# The transition and sequence containers are family-generic: their
# invariants (boundary-frame continuity, terminal placement, discounted
# value targets) are exactly what maze_chase needs. Aliased here so maze
# code reads in its own vocabulary.
MazeChaseTransition = MovingShapesTransition
MazeChaseSequence = MovingShapesSequence


@dataclass(frozen=True, slots=True)
class MazeChaseDatasetConfig:
    """Identity and generation bounds for a lazy maze_chase dataset.

    Defaults describe 1,048,576 transitions of the canonical matrix slot,
    but construction performs no rollout work. Callers can choose smaller
    counts for smoke runs. Note that materializing one sequence runs the
    lookahead planner for up to ``sequence_length`` decisions, which costs
    milliseconds per decision — generation at scale is accelerator-window
    work, exactly like the other ladder datasets.
    """

    split: DatasetSplit = DatasetSplit.TRAIN
    sequence_count: int = 8_192
    sequence_length: int = 128
    seed_offset: int = 0
    ghost_count: int = 3
    ghost_period: int = 2
    player_period: int = 1
    extra_loops: int = 16
    ghost_rule: str = "direct"
    ghost_elroy: bool = False
    input_delay_ticks: int = 0
    sticky_direction: bool = False
    tick_period_ns: int = MazeChaseEnv.DEFAULT_TICK_PERIOD_NS
    discount: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", _split(self.split))
        count = _integer(
            self.sequence_count,
            name="sequence_count",
            minimum=1,
            maximum=_SEED_NAMESPACE_SIZE,
        )
        length = _integer(
            self.sequence_length,
            name="sequence_length",
            minimum=1,
            maximum=0xFFFFFFFF,
        )
        offset = _integer(
            self.seed_offset,
            name="seed_offset",
            maximum=_SEED_NAMESPACE_SIZE - 1,
        )
        if offset + count > _SEED_NAMESPACE_SIZE:
            raise ValueError("seed_offset + sequence_count exceeds the split namespace")
        _integer(self.ghost_count, name="ghost_count", minimum=1, maximum=8)
        _integer(
            self.ghost_period,
            name="ghost_period",
            minimum=1,
            maximum=MazeChaseEnv.MAX_GHOST_PERIOD,
        )
        _integer(
            self.player_period,
            name="player_period",
            minimum=1,
            maximum=MazeChaseEnv.MAX_PLAYER_PERIOD,
        )
        _integer(
            self.extra_loops,
            name="extra_loops",
            minimum=0,
            maximum=MazeChaseEnv.MAX_EXTRA_LOOPS,
        )
        if not isinstance(self.ghost_rule, str) or self.ghost_rule not in _GHOST_RULES:
            raise ValueError(f"ghost_rule must be one of {sorted(_GHOST_RULES)}")
        if not isinstance(self.ghost_elroy, bool):
            raise ValueError("ghost_elroy must be a boolean")
        _integer(
            self.input_delay_ticks,
            name="input_delay_ticks",
            minimum=0,
            maximum=MazeChaseEnv.MAX_INPUT_DELAY,
        )
        if not isinstance(self.sticky_direction, bool):
            raise ValueError("sticky_direction must be a boolean")
        _integer(self.tick_period_ns, name="tick_period_ns", minimum=1)
        discount = _number(self.discount, name="discount")
        if discount < 0.0 or discount > 1.0:
            raise ValueError("discount must be in [0, 1]")
        object.__setattr__(self, "sequence_count", count)
        object.__setattr__(self, "sequence_length", length)
        object.__setattr__(self, "seed_offset", offset)
        object.__setattr__(self, "discount", discount)

    @property
    def total_transitions(self) -> int:
        return self.sequence_count * self.sequence_length

    def manifest_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe identity of this virtual dataset."""

        return {
            "schema_version": _DATASET_SCHEMA_VERSION,
            "generator_id": _GENERATOR_ID,
            "origin": "in-repository deterministic procedural environment",
            "license_record_id": _LICENSE_RECORD_ID,
            "environment_family": "maze_chase",
            "split": self.split.value,
            "seed_namespace_tag": {
                DatasetSplit.TRAIN: 0,
                DatasetSplit.VALIDATION: 1,
                DatasetSplit.TEST: 2,
            }[self.split],
            "seed_offset": self.seed_offset,
            "sequence_count": self.sequence_count,
            "sequence_length": self.sequence_length,
            "total_transitions": self.total_transitions,
            "ghost_count": self.ghost_count,
            "ghost_period": self.ghost_period,
            "player_period": self.player_period,
            "extra_loops": self.extra_loops,
            "ghost_rule": self.ghost_rule,
            "ghost_elroy": self.ghost_elroy,
            "input_delay_ticks": self.input_delay_ticks,
            "sticky_direction": self.sticky_direction,
            "tick_period_ns": self.tick_period_ns,
            "discount_hex": self.discount.hex(),
            "teacher_identity": "diagnostic.scripted_maze_chase_planner.v1",
            "teacher_inputs": "visible_rgb_only",
            "world_target": "next_model_observation",
            "value_target": "zero_bootstrapped_discounted_sequence_return",
        }


def maze_chase_dataset_manifest_sha256(config: MazeChaseDatasetConfig) -> str:
    """Hash a configuration using a process-independent canonical manifest."""

    if not isinstance(config, MazeChaseDatasetConfig):
        raise ValueError("config must be a MazeChaseDatasetConfig")
    encoded = dumps(
        config.manifest_dict(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRMCDATASET\x01" + encoded).hexdigest()


class MazeChaseSequenceDataset(Sequence[MazeChaseSequence]):
    """Random-access, on-the-fly maze_chase dataset with deterministic epochs."""

    __slots__ = ("config", "manifest_sha256")

    def __init__(self, config: MazeChaseDatasetConfig) -> None:
        if not isinstance(config, MazeChaseDatasetConfig):
            raise ValueError("config must be a MazeChaseDatasetConfig")
        self.config = config
        self.manifest_sha256 = maze_chase_dataset_manifest_sha256(config)

    def __len__(self) -> int:
        return self.config.sequence_count

    @overload
    def __getitem__(self, index: int) -> MazeChaseSequence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[MazeChaseSequence, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> MazeChaseSequence | tuple[MazeChaseSequence, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer or slice")
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("dataset index out of range")
        return self._generate(normalized)

    def _generate(self, sequence_index: int) -> MazeChaseSequence:
        # Lazy import: evaluation modules already depend on ..data, so the
        # planner policy is imported at materialization time to keep the
        # package import graph acyclic.
        from ..evaluation.diagnostic_policies import (
            ScriptedMazeChasePlannerPolicy,
        )

        config = self.config
        episode_seed = split_episode_seed(
            config.split,
            config.seed_offset + sequence_index,
        )
        environment = MazeChaseEnv(
            ghost_count=config.ghost_count,
            ghost_period=config.ghost_period,
            player_period=config.player_period,
            extra_loops=config.extra_loops,
            ghost_rule=config.ghost_rule,
            ghost_elroy=config.ghost_elroy,
            input_delay_ticks=config.input_delay_ticks,
            sticky_direction=config.sticky_direction,
            tick_period_ns=config.tick_period_ns,
            max_ticks=config.sequence_length,
        )
        teacher = ScriptedMazeChasePlannerPolicy(
            ghost_period=config.ghost_period,
            input_delay_ticks=config.input_delay_ticks,
            player_period=config.player_period,
            ghost_elroy=config.ghost_elroy,
        )
        teacher.reset(episode_seed)
        raw_observation = environment.reset(episode_seed)
        raw_transitions: list[_TransitionWithoutValue] = []

        for _ in range(config.sequence_length):
            observation = raw_observation.to_model_observation(observation_age_ns=0)
            action_target = teacher.act(raw_observation)
            outcome = environment.step(action_target)
            next_observation = outcome.observation.to_model_observation(
                observation_age_ns=0
            )
            raw_transitions.append(
                _TransitionWithoutValue(
                    observation=observation,
                    applied_control=outcome.applied_control,
                    action_target=outcome.requested_control,
                    next_observation_target=next_observation,
                    reward_target=outcome.reward,
                    event_targets=outcome.events,
                    terminated_target=outcome.terminated,
                    truncated_target=outcome.truncated,
                )
            )
            raw_observation = outcome.observation
            if outcome.terminated:
                break

        running_return = 0.0
        reversed_transitions: list[MazeChaseTransition] = []
        for raw in reversed(raw_transitions):
            done = raw.terminated_target or raw.truncated_target
            running_return = raw.reward_target + (
                0.0 if done else config.discount * running_return
            )
            reversed_transitions.append(
                MazeChaseTransition(
                    observation=raw.observation,
                    applied_control=raw.applied_control,
                    action_target=raw.action_target,
                    next_observation_target=raw.next_observation_target,
                    reward_target=raw.reward_target,
                    value_target=running_return,
                    event_targets=raw.event_targets,
                    terminated_target=raw.terminated_target,
                    truncated_target=raw.truncated_target,
                )
            )

        return MazeChaseSequence(
            split=config.split,
            sequence_index=sequence_index,
            episode_seed=episode_seed,
            discount=config.discount,
            manifest_sha256=self.manifest_sha256,
            transitions=tuple(reversed(reversed_transitions)),
        )

    def epoch_indices(self, *, epoch: int, shuffle: bool = True) -> tuple[int, ...]:
        """Return a deterministic bijection over sequence indices for an epoch."""

        epoch_number = _integer(epoch, name="epoch")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        count = len(self)
        if not shuffle or count == 1:
            return tuple(range(count))

        seed = sha256(
            b"IRMSEPOCH\x01"
            + bytes.fromhex(self.manifest_sha256)
            + pack(">Q", epoch_number)
        ).digest()
        multiplier = int.from_bytes(seed[:8], "big") % count
        if multiplier == 0:
            multiplier = 1
        while gcd(multiplier, count) != 1:
            multiplier = (multiplier + 1) % count
            if multiplier == 0:
                multiplier = 1
        offset = int.from_bytes(seed[8:16], "big") % count
        return tuple((multiplier * index + offset) % count for index in range(count))

    def iter_epoch(
        self,
        *,
        epoch: int,
        shuffle: bool = True,
    ) -> Iterator[MazeChaseSequence]:
        """Generate one deterministic epoch without retaining trajectories."""

        return (self[index] for index in self.epoch_indices(epoch=epoch, shuffle=shuffle))


__all__ = [
    "DatasetSplit",
    "MazeChaseDatasetConfig",
    "MazeChaseSequence",
    "MazeChaseSequenceDataset",
    "MazeChaseTransition",
    "maze_chase_dataset_manifest_sha256",
]
