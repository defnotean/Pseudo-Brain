"""Lazy deterministic solver demonstrations for the ladder's solver worlds.

This mirrors :mod:`irene_brain.data.maze_chase_dataset` for the four
ladder worlds whose scripted frontiers are mapped by pixel-only solvers:
keys_doors (ordered planning plus unrendered key-possession memory),
junction (unique-path planning under pursuit), occlusion (episodic target
memory and hazard-belief tracking under fog), and pursuit (careful
collection with mover-rule identification and the lure). Each world's
teacher is the corresponding ``diagnostic.scripted_*`` policy from
:mod:`irene_brain.evaluation.diagnostic_policies`, run through the
environment's own step function, so demonstrations are frontier-quality
pixel-derived behavior — the exact skills the thought-state model must
distill beyond reactive play.

The two open-field worlds share a pixel signature, but only pursuit is
registered here: moving_shapes is covered by the sealed RCQ dataset
family, and adding a second moving_shapes generator would need a
capability guard against the sealed TEST ranges *before* any TEST
construction. None of the four registered worlds has sealed ranges (the
RCQ-v2/v3 seals cover moving_shapes only); if a future campaign seals a
TEST range for any of them, a capability guard like the moving_shapes one
must be added here first. The split namespaces are the same disjoint
62-bit tags, so train/validation/test episode seeds never overlap.

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
from typing import Any, Callable, overload

from ..environments.junction import JunctionEnv
from ..environments.keys_doors import KeysDoorsEnv
from ..environments.occlusion import OcclusionEnv
from ..environments.pursuit import PursuitEnv
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
_LICENSE_RECORD_ID = "original-project-content"

# The transition and sequence containers are family-generic: their
# invariants (boundary-frame continuity, terminal placement, discounted
# value targets) are exactly what the solver worlds need. Aliased here so
# solver code reads in its own vocabulary.
SolverTransition = MovingShapesTransition
SolverSequence = MovingShapesSequence


@dataclass(frozen=True, slots=True)
class _WorldSpec:
    """One solver world's canonical environment and teacher wiring."""

    environment_factory: Callable[[int, int], object]  # (max_ticks, tick_period_ns)
    policy_name: str
    teacher_identity: str


def _keys_doors_env(max_ticks: int, tick_period_ns: int) -> object:
    return KeysDoorsEnv(tick_period_ns=tick_period_ns, max_ticks=max_ticks)


def _junction_env(max_ticks: int, tick_period_ns: int) -> object:
    return JunctionEnv(
        chaser_count=1,
        chaser_period=2,
        tick_period_ns=tick_period_ns,
        max_ticks=max_ticks,
    )


def _occlusion_env(max_ticks: int, tick_period_ns: int) -> object:
    return OcclusionEnv(
        hazard_count=3,
        view_radius=4,
        tick_period_ns=tick_period_ns,
        max_ticks=max_ticks,
    )


def _pursuit_env(max_ticks: int, tick_period_ns: int) -> object:
    return PursuitEnv(
        pursuer_count=2,
        pursuer_period=2,
        tick_period_ns=tick_period_ns,
        max_ticks=max_ticks,
    )


_SOLVER_WORLDS = {
    "keys_doors": _WorldSpec(
        _keys_doors_env,
        "ScriptedKeysDoorsSolver",
        "diagnostic.scripted_keys_doors_solver.v1",
    ),
    "junction": _WorldSpec(
        _junction_env,
        "ScriptedJunctionSolver",
        "diagnostic.scripted_junction_solver.v1",
    ),
    "occlusion": _WorldSpec(
        _occlusion_env,
        "ScriptedOcclusionMemoryPolicy",
        "diagnostic.scripted_occlusion_memory.v1",
    ),
    "pursuit": _WorldSpec(
        _pursuit_env,
        "ScriptedOpenFieldCollectorPolicy",
        "diagnostic.scripted_open_field_collector.v1",
    ),
}

SOLVER_WORLD_NAMES = tuple(_SOLVER_WORLDS)


def solver_environment_factory(world: str) -> Callable[[int, int], object]:
    """Return the canonical ``(max_ticks, tick_period_ns)`` env factory.

    Play and evaluation harnesses use this so they cannot drift from the
    canonical world configurations the solver dataset teaches on.
    """

    if not isinstance(world, str) or world not in _SOLVER_WORLDS:
        raise ValueError(f"world must be one of {sorted(_SOLVER_WORLDS)}")
    return _SOLVER_WORLDS[world].environment_factory


@dataclass(frozen=True, slots=True)
class SolverDatasetConfig:
    """Identity and generation bounds for a lazy solver dataset.

    Defaults describe 1,048,576 transitions of the canonical world
    configuration, but construction performs no rollout work. Callers can
    choose smaller counts for smoke runs. Materializing one sequence runs
    the world's solver for up to ``sequence_length`` decisions; generation
    at scale is accelerator-window work, exactly like the other ladder
    datasets.
    """

    world: str = "keys_doors"
    split: DatasetSplit = DatasetSplit.TRAIN
    sequence_count: int = 8_192
    sequence_length: int = 128
    seed_offset: int = 0
    tick_period_ns: int = KeysDoorsEnv.DEFAULT_TICK_PERIOD_NS
    discount: float = 0.99

    def __post_init__(self) -> None:
        if not isinstance(self.world, str) or self.world not in _SOLVER_WORLDS:
            raise ValueError(f"world must be one of {sorted(_SOLVER_WORLDS)}")
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

    @property
    def generator_id(self) -> str:
        return f"irene.{self.world}.solver_teacher.v1"

    def manifest_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe identity of this virtual dataset."""

        spec = _SOLVER_WORLDS[self.world]
        return {
            "schema_version": _DATASET_SCHEMA_VERSION,
            "generator_id": self.generator_id,
            "origin": "in-repository deterministic procedural environment",
            "license_record_id": _LICENSE_RECORD_ID,
            "environment_family": self.world,
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
            "tick_period_ns": self.tick_period_ns,
            "discount_hex": self.discount.hex(),
            "teacher_identity": spec.teacher_identity,
            "teacher_inputs": "visible_rgb_only",
            "world_target": "next_model_observation",
            "value_target": "zero_bootstrapped_discounted_sequence_return",
        }


def solver_dataset_manifest_sha256(config: SolverDatasetConfig) -> str:
    """Hash a configuration using a process-independent canonical manifest."""

    if not isinstance(config, SolverDatasetConfig):
        raise ValueError("config must be a SolverDatasetConfig")
    encoded = dumps(
        config.manifest_dict(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRMSOLVERDATASET\x01" + encoded).hexdigest()


class SolverSequenceDataset(Sequence[SolverSequence]):
    """Random-access, on-the-fly solver dataset with deterministic epochs."""

    __slots__ = ("config", "manifest_sha256")

    def __init__(self, config: SolverDatasetConfig) -> None:
        if not isinstance(config, SolverDatasetConfig):
            raise ValueError("config must be a SolverDatasetConfig")
        self.config = config
        self.manifest_sha256 = solver_dataset_manifest_sha256(config)

    def __len__(self) -> int:
        return self.config.sequence_count

    @overload
    def __getitem__(self, index: int) -> SolverSequence: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SolverSequence, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> SolverSequence | tuple[SolverSequence, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer or slice")
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("dataset index out of range")
        return self._generate(normalized)

    def _generate(self, sequence_index: int) -> SolverSequence:
        # Lazy import: evaluation modules already depend on ..data, so the
        # solver policies are imported at materialization time to keep the
        # package import graph acyclic.
        from ..evaluation import diagnostic_policies

        config = self.config
        spec = _SOLVER_WORLDS[config.world]
        episode_seed = split_episode_seed(
            config.split,
            config.seed_offset + sequence_index,
        )
        environment = spec.environment_factory(
            config.sequence_length, config.tick_period_ns
        )
        teacher = getattr(diagnostic_policies, spec.policy_name)()
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
        reversed_transitions: list[SolverTransition] = []
        for raw in reversed(raw_transitions):
            done = raw.terminated_target or raw.truncated_target
            running_return = raw.reward_target + (
                0.0 if done else config.discount * running_return
            )
            reversed_transitions.append(
                SolverTransition(
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

        return SolverSequence(
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
    ) -> Iterator[SolverSequence]:
        """Generate one deterministic epoch without retaining trajectories."""

        return (self[index] for index in self.epoch_indices(epoch=epoch, shuffle=shuffle))


__all__ = [
    "DatasetSplit",
    "SOLVER_WORLD_NAMES",
    "SolverDatasetConfig",
    "SolverSequence",
    "SolverSequenceDataset",
    "SolverTransition",
    "solver_dataset_manifest_sha256",
    "solver_environment_factory",
]
