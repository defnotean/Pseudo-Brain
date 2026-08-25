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
from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
from math import gcd
from struct import pack
from typing import Any, overload

from ..environments.maze_chase import _GHOST_RULES, MazeChaseEnv
from ..types import GenericControl, HidKey, ModelObservation
from .moving_shapes_dataset import (
    DatasetSplit,
    MovingShapesSequence,
    MovingShapesTransition,
    _TransitionWithoutValue,
    _integer,
    _number,
    _sha256_string,
    _split,
    split_episode_seed,
)

_SEED_NAMESPACE_SIZE = 1 << 62
_DATASET_SCHEMA_VERSION = 1
_GENERATOR_ID = "irene.maze_chase.planner_teacher.v1"
_EPISODE_WINDOW_GENERATOR_ID = (
    "irene.maze_chase.planner_teacher.episode_windows.v1"
)
_TILED_WINDOW_GENERATOR_ID = (
    "irene.maze_chase.planner_teacher.tiled_windows.v1"
)
_INTERVENTION_GENERATOR_ID = (
    "irene.maze_chase.planner_labels.balanced_behavior_intervention.v1"
)
_INTERVENTION_EPISODE_WINDOW_GENERATOR_ID = (
    "irene.maze_chase.planner_labels.balanced_behavior_intervention.episode_windows.v1"
)
_INTERVENTION_TILED_WINDOW_GENERATOR_ID = (
    "irene.maze_chase.planner_labels.balanced_behavior_intervention.tiled_windows.v1"
)
_UNIFORM_WINDOW_SAMPLING = "uniform_start_across_episode"
_TILED_WINDOW_SAMPLING = "tiled_stride_across_episode"
_LICENSE_RECORD_ID = "original-project-content"
_BEHAVIOR_POLICIES = frozenset({"teacher", "balanced_intervention_v1"})
_COUNTERFACTUAL_TARGETS = frozenset({"none", "all_actions_v1"})
_ACTION_KEYS: tuple[int | None, ...] = (
    None,
    int(HidKey.W),
    int(HidKey.A),
    int(HidKey.S),
    int(HidKey.D),
)

# The transition and sequence containers are family-generic: their
# invariants (boundary-frame continuity, terminal placement, discounted
# value targets) are exactly what maze_chase needs. Aliased here so maze
# code reads in its own vocabulary.
MazeChaseTransition = MovingShapesTransition
MazeChaseSequence = MovingShapesSequence


@dataclass(frozen=True, slots=True)
class MazeChaseCounterfactualTarget:
    """One labelled action branch from the transition's exact root state."""

    action_class: int
    requested_control: GenericControl
    applied_control: GenericControl
    next_observation_target: ModelObservation
    reward_target: float
    event_targets: tuple[str, ...]
    terminated_target: bool
    truncated_target: bool
    result_state_sha256: str

    def __post_init__(self) -> None:
        action_class = _integer(
            self.action_class,
            name="counterfactual action_class",
            maximum=len(_ACTION_KEYS) - 1,
        )
        if not isinstance(self.requested_control, GenericControl):
            raise ValueError("counterfactual requested_control must be a GenericControl")
        if not isinstance(self.applied_control, GenericControl):
            raise ValueError("counterfactual applied_control must be a GenericControl")
        expected_key = _ACTION_KEYS[action_class]
        expected_control = (
            GenericControl()
            if expected_key is None
            else GenericControl(keys_down=(expected_key,))
        )
        if self.requested_control != expected_control:
            raise ValueError("counterfactual action_class and requested_control disagree")
        if self.applied_control != self.requested_control:
            raise ValueError("counterfactual v1 requires requested control to be applied")
        if not isinstance(self.next_observation_target, ModelObservation):
            raise ValueError(
                "counterfactual next_observation_target must be a ModelObservation"
            )
        object.__setattr__(
            self,
            "reward_target",
            _number(self.reward_target, name="counterfactual reward_target"),
        )
        if not isinstance(self.event_targets, tuple) or any(
            not isinstance(event, str) or not event for event in self.event_targets
        ):
            raise ValueError(
                "counterfactual event_targets must be a tuple of non-empty strings"
            )
        if not isinstance(self.terminated_target, bool):
            raise ValueError("counterfactual terminated_target must be a boolean")
        if not isinstance(self.truncated_target, bool):
            raise ValueError("counterfactual truncated_target must be a boolean")
        _sha256_string(
            self.result_state_sha256,
            name="counterfactual result_state_sha256",
        )


@dataclass(frozen=True, slots=True)
class MazeChaseCounterfactualTransition(MovingShapesTransition):
    """A factual recurrent transition plus exhaustive one-step action branches."""

    root_state_sha256: str
    counterfactual_targets: tuple[MazeChaseCounterfactualTarget, ...]

    def __post_init__(self) -> None:
        MovingShapesTransition.__post_init__(self)
        _sha256_string(self.root_state_sha256, name="root_state_sha256")
        targets = self.counterfactual_targets
        if (
            not isinstance(targets, tuple)
            or len(targets) != len(_ACTION_KEYS)
            or any(
                not isinstance(target, MazeChaseCounterfactualTarget)
                for target in targets
            )
        ):
            raise ValueError("counterfactual_targets must contain all five actions")
        if tuple(target.action_class for target in targets) != tuple(
            range(len(_ACTION_KEYS))
        ):
            raise ValueError("counterfactual_targets must be ordered by action class")
        for target in targets:
            branch_observation = target.next_observation_target
            if branch_observation.frame_id != self.observation.frame_id + 1:
                raise ValueError("every counterfactual must be the immediately following frame")
            if branch_observation.elapsed_ns <= self.observation.elapsed_ns:
                raise ValueError("every counterfactual must occur after the branch root")
            if branch_observation.previous_control != target.applied_control:
                raise ValueError("counterfactual next observation must carry its applied action")

        factual_class = next(
            target.action_class
            for target in targets
            if target.applied_control == self.applied_control
        )
        factual = targets[factual_class]
        if (
            factual.next_observation_target != self.next_observation_target
            or factual.reward_target.hex() != self.reward_target.hex()
            or factual.event_targets != self.event_targets
            or factual.terminated_target != self.terminated_target
            or factual.truncated_target != self.truncated_target
        ):
            raise ValueError("the applied-action branch must equal the factual transition")


@dataclass(frozen=True, slots=True)
class MazeChaseCounterfactualSequence(MovingShapesSequence):
    """A recurrent factual sequence whose siblings never enter its history."""

    def __post_init__(self) -> None:
        MovingShapesSequence.__post_init__(self)
        if any(
            not isinstance(transition, MazeChaseCounterfactualTransition)
            for transition in self.transitions
        ):
            raise ValueError(
                "counterfactual sequences require counterfactual transitions"
            )

    @property
    def content_sha256(self) -> str:
        digest = sha256(b"IRMCCFSEQUENCE\x01")
        digest.update(bytes.fromhex(MovingShapesSequence.content_sha256.fget(self)))
        for transition in self.transitions:
            digest.update(bytes.fromhex(transition.root_state_sha256))
            digest.update(pack(">I", len(transition.counterfactual_targets)))
            for target in transition.counterfactual_targets:
                digest.update(pack(">I", target.action_class))
                for branch_control in (
                    target.requested_control,
                    target.applied_control,
                ):
                    control = branch_control.canonical_bytes()
                    digest.update(pack(">I", len(control)))
                    digest.update(control)
                digest.update(bytes.fromhex(target.next_observation_target.content_hash))
                digest.update(pack(">d", target.reward_target))
                digest.update(pack(">I", len(target.event_targets)))
                for event in target.event_targets:
                    encoded = event.encode("utf-8")
                    digest.update(pack(">I", len(encoded)))
                    digest.update(encoded)
                digest.update(
                    bytes((target.terminated_target, target.truncated_target))
                )
                digest.update(bytes.fromhex(target.result_state_sha256))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CounterfactualTransitionWithoutValue(_TransitionWithoutValue):
    root_state_sha256: str
    counterfactual_targets: tuple[MazeChaseCounterfactualTarget, ...]


@dataclass(frozen=True, slots=True)
class MazeChaseDatasetConfig:
    """Identity and generation bounds for a lazy maze_chase dataset.

    Defaults describe 1,048,576 transitions of the canonical matrix slot,
    but construction performs no rollout work. Callers can choose smaller
    counts for smoke runs. Note that materializing one sequence runs the
    lookahead planner for up to ``sequence_length`` decisions, or up to
    ``episode_horizon`` when later-tick windows are selected, which costs
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
    # 0 keeps spawn-only generation (max_ticks = sequence_length), so
    # historical planner_teacher.v1 manifests stay byte-identical. A positive
    # value rolls the planner through a full episode and slices a
    # sequence_length window from a uniform start tick — including later
    # pellets and corridor choices the 8-tick spawn snippets never showed.
    episode_horizon: int = 0
    # "uniform" keeps the hashed start. "tiled" walks consecutive
    # non-overlapping windows of one episode so every teacher tick is a
    # label (1:1 with a 240-tick play-eval horizon when sequence_count
    # covers episode_horizon / sequence_length). Spawn-only (horizon 0)
    # stays uniform-only so planner_teacher.v1 hashes stay byte-identical.
    window_sampling: str = "uniform"
    # The teacher remains the action-label policy. In intervention mode a
    # separate deterministic behavior policy occasionally applies a balanced
    # idle/W/A/S/D action, giving the world-model heads causal action coverage
    # instead of only the teacher's on-policy consequences.
    behavior_policy: str = "teacher"
    behavior_intervention_rate: float = 0.0
    # Exhaustive one-step branches are supervision siblings only. Exactly one
    # factual behavior action still advances the recurrent trajectory.
    counterfactual_targets: str = "none"

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
        horizon = _integer(
            self.episode_horizon,
            name="episode_horizon",
            maximum=0xFFFFFFFF,
        )
        if horizon != 0 and horizon <= length:
            raise ValueError(
                "episode_horizon must be 0 or greater than sequence_length"
            )
        sampling = self.window_sampling
        if not isinstance(sampling, str) or sampling not in {"uniform", "tiled"}:
            raise ValueError("window_sampling must be uniform or tiled")
        if sampling == "tiled":
            if horizon <= 0:
                raise ValueError("tiled window_sampling requires episode_horizon > 0")
            if horizon % length != 0:
                raise ValueError(
                    "tiled window_sampling requires episode_horizon divisible "
                    "by sequence_length"
                )
        behavior_policy = self.behavior_policy
        if (
            not isinstance(behavior_policy, str)
            or behavior_policy not in _BEHAVIOR_POLICIES
        ):
            raise ValueError(
                f"behavior_policy must be one of {sorted(_BEHAVIOR_POLICIES)}"
            )
        intervention_rate = _number(
            self.behavior_intervention_rate,
            name="behavior_intervention_rate",
        )
        if intervention_rate < 0.0 or intervention_rate > 1.0:
            raise ValueError("behavior_intervention_rate must be in [0, 1]")
        if behavior_policy == "teacher" and intervention_rate != 0.0:
            raise ValueError(
                "teacher behavior_policy requires behavior_intervention_rate == 0"
            )
        if behavior_policy == "balanced_intervention_v1":
            if intervention_rate <= 0.0:
                raise ValueError(
                    "balanced_intervention_v1 requires behavior_intervention_rate > 0"
                )
            # The current planner models its own submitted-action FIFO. Once a
            # separate behavior stream is introduced that FIFO would no longer
            # describe the environment, so fail closed until the delayed-input
            # teacher exposes an explicit behavior-synchronization contract.
            if self.input_delay_ticks != 0 or self.sticky_direction:
                raise ValueError(
                    "balanced_intervention_v1 currently requires zero input delay "
                    "and sticky_direction=False"
                )
        counterfactual_targets = self.counterfactual_targets
        if (
            not isinstance(counterfactual_targets, str)
            or counterfactual_targets not in _COUNTERFACTUAL_TARGETS
        ):
            raise ValueError(
                f"counterfactual_targets must be one of {sorted(_COUNTERFACTUAL_TARGETS)}"
            )
        if counterfactual_targets == "all_actions_v1" and (
            self.input_delay_ticks != 0 or self.sticky_direction
        ):
            raise ValueError(
                "all_actions_v1 currently requires zero input delay and "
                "sticky_direction=False"
            )
        if counterfactual_targets == "all_actions_v1" and horizon != 0:
            raise ValueError(
                "all_actions_v1 currently requires spawn-only sequences; "
                "episode windows need a separate sequence-boundary mask"
            )
        object.__setattr__(self, "sequence_count", count)
        object.__setattr__(self, "sequence_length", length)
        object.__setattr__(self, "seed_offset", offset)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "episode_horizon", horizon)
        object.__setattr__(self, "window_sampling", sampling)
        object.__setattr__(self, "behavior_policy", behavior_policy)
        object.__setattr__(
            self,
            "behavior_intervention_rate",
            intervention_rate,
        )
        object.__setattr__(self, "counterfactual_targets", counterfactual_targets)

    @property
    def generator_id(self) -> str:
        if self.counterfactual_targets == "all_actions_v1":
            return (
                "irene.maze_chase.planner_labels.factual_rollout."
                "all_action_branches.v1"
            )
        if self.behavior_policy == "balanced_intervention_v1":
            if self.window_sampling == "tiled":
                return _INTERVENTION_TILED_WINDOW_GENERATOR_ID
            if self.episode_horizon > 0:
                return _INTERVENTION_EPISODE_WINDOW_GENERATOR_ID
            return _INTERVENTION_GENERATOR_ID
        if self.window_sampling == "tiled":
            return _TILED_WINDOW_GENERATOR_ID
        if self.episode_horizon > 0:
            return _EPISODE_WINDOW_GENERATOR_ID
        return _GENERATOR_ID

    @property
    def windows_per_episode(self) -> int:
        if self.window_sampling != "tiled":
            return 1
        return self.episode_horizon // self.sequence_length

    @property
    def total_transitions(self) -> int:
        return self.sequence_count * self.sequence_length

    def manifest_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe identity of this virtual dataset."""

        payload: dict[str, Any] = {
            "schema_version": _DATASET_SCHEMA_VERSION,
            "generator_id": self.generator_id,
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
        # Omit the horizon while it is 0 so spawn-only dataset hashes stay
        # byte-identical to planner_teacher.v1.
        if self.episode_horizon > 0:
            payload["episode_horizon"] = self.episode_horizon
            payload["window_sampling"] = (
                _TILED_WINDOW_SAMPLING
                if self.window_sampling == "tiled"
                else _UNIFORM_WINDOW_SAMPLING
            )
        if self.behavior_policy != "teacher":
            payload["behavior_policy"] = self.behavior_policy
            payload["behavior_intervention_rate_hex"] = (
                self.behavior_intervention_rate.hex()
            )
            payload["action_supervision_policy"] = (
                "diagnostic.scripted_maze_chase_planner.v1"
            )
        if self.counterfactual_targets != "none":
            payload["counterfactual_targets"] = self.counterfactual_targets
            payload.update(
                {
                    "transition_schema": "irene.maze_chase.all_action_transition.v1",
                    "outcome_table_schema": "irene.discrete_action_outcome_table_target.v1",
                    "branch_root": "exact_pre_action_snapshot",
                    "branch_horizon_ticks": 1,
                    "branch_action_ids": list(range(len(_ACTION_KEYS))),
                    "branch_action_semantics": ["idle", "W", "A", "S", "D"],
                    "branch_row_serialization": "ascending_action_id",
                    "branch_cardinality": len(_ACTION_KEYS),
                    "branch_coverage": "exhaustive",
                    "branch_restore_contract": "snapshot_restore_state_hash_v1",
                    "factual_branch_equivalence_required": True,
                    "recurrent_path": "factual_only",
                    "expert_ce_exposure": "once_per_factual_state",
                    "branch_value_target": "none_one_step_only",
                    "actuation_contract": (
                        "zero_delay_nonsticky_requested_equals_applied_v1"
                    ),
                    "model_input_excludes_branch_targets": True,
                }
            )
        return payload


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
        episode_index, window_index = self._episode_and_window(sequence_index)
        episode_seed = split_episode_seed(
            config.split,
            config.seed_offset + episode_index,
        )
        rollout_ticks = (
            config.episode_horizon
            if config.episode_horizon > 0
            else config.sequence_length
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
            max_ticks=rollout_ticks,
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

        for _ in range(rollout_ticks):
            observation = raw_observation.to_model_observation(observation_age_ns=0)
            action_target = teacher.act(raw_observation)
            counterfactual = self._counterfactual_targets(environment)
            counterfactual_targets = (
                None if counterfactual is None else counterfactual[1]
            )
            behavior_action = self._behavior_action(
                action_target,
                episode_seed=episode_seed,
                tick=raw_observation.frame_id,
            )
            outcome = environment.step(behavior_action)
            next_observation = outcome.observation.to_model_observation(
                observation_age_ns=0
            )
            raw_transition_type = (
                _CounterfactualTransitionWithoutValue
                if counterfactual_targets is not None
                else _TransitionWithoutValue
            )
            raw_kwargs: dict[str, Any] = {
                "observation": observation,
                "applied_control": outcome.applied_control,
                "action_target": action_target,
                "next_observation_target": next_observation,
                "reward_target": outcome.reward,
                "event_targets": outcome.events,
                "terminated_target": outcome.terminated,
                "truncated_target": outcome.truncated,
            }
            if counterfactual_targets is not None:
                raw_kwargs["root_state_sha256"] = counterfactual[0]
                raw_kwargs["counterfactual_targets"] = counterfactual_targets
            raw_transitions.append(
                raw_transition_type(
                    **raw_kwargs,
                )
            )
            raw_observation = outcome.observation
            if outcome.terminated:
                break

        raw_transitions = self._episode_window(
            sequence_index,
            raw_transitions,
            window_index=window_index,
        )

        running_return = 0.0
        reversed_transitions: list[MazeChaseTransition] = []
        for raw in reversed(raw_transitions):
            done = raw.terminated_target or raw.truncated_target
            running_return = raw.reward_target + (
                0.0 if done else config.discount * running_return
            )
            transition_type = (
                MazeChaseCounterfactualTransition
                if isinstance(raw, _CounterfactualTransitionWithoutValue)
                else MazeChaseTransition
            )
            transition_kwargs: dict[str, Any] = {
                "observation": raw.observation,
                "applied_control": raw.applied_control,
                "action_target": raw.action_target,
                "next_observation_target": raw.next_observation_target,
                "reward_target": raw.reward_target,
                "value_target": running_return,
                "event_targets": raw.event_targets,
                "terminated_target": raw.terminated_target,
                "truncated_target": raw.truncated_target,
            }
            if isinstance(raw, _CounterfactualTransitionWithoutValue):
                transition_kwargs["root_state_sha256"] = raw.root_state_sha256
                transition_kwargs["counterfactual_targets"] = (
                    raw.counterfactual_targets
                )
            reversed_transitions.append(
                transition_type(
                    **transition_kwargs,
                )
            )

        sequence_type = (
            MazeChaseCounterfactualSequence
            if config.counterfactual_targets == "all_actions_v1"
            else MazeChaseSequence
        )
        return sequence_type(
            split=config.split,
            sequence_index=sequence_index,
            episode_seed=episode_seed,
            discount=config.discount,
            manifest_sha256=self.manifest_sha256,
            transitions=tuple(reversed(reversed_transitions)),
        )

    def _counterfactual_targets(
        self,
        environment: MazeChaseEnv,
    ) -> tuple[str, tuple[MazeChaseCounterfactualTarget, ...]] | None:
        """Evaluate every action from one root and restore that root exactly."""

        if self.config.counterfactual_targets == "none":
            return None
        root = environment.snapshot()
        root_hash = environment.state_hash()
        targets: list[MazeChaseCounterfactualTarget] = []
        try:
            for action_class, key in enumerate(_ACTION_KEYS):
                environment.restore(root)
                if environment.state_hash() != root_hash:
                    raise RuntimeError("counterfactual branch root changed after restore")
                control = (
                    GenericControl()
                    if key is None
                    else GenericControl(keys_down=(key,))
                )
                outcome = environment.step(control)
                if outcome.applied_control != control:
                    raise RuntimeError("counterfactual branch action was not applied exactly")
                targets.append(
                    MazeChaseCounterfactualTarget(
                        action_class=action_class,
                        requested_control=control,
                        applied_control=outcome.applied_control,
                        next_observation_target=(
                            outcome.observation.to_model_observation(
                                observation_age_ns=0
                            )
                        ),
                        reward_target=outcome.reward,
                        event_targets=outcome.events,
                        terminated_target=outcome.terminated,
                        truncated_target=outcome.truncated,
                        result_state_sha256=environment.state_hash(),
                    )
                )
        finally:
            environment.restore(root)
        if environment.state_hash() != root_hash:
            raise RuntimeError("counterfactual evaluation mutated the factual root")
        return root_hash, tuple(targets)

    def _behavior_action(
        self,
        teacher_action: GenericControl,
        *,
        episode_seed: int,
        tick: int,
    ) -> GenericControl:
        """Return the causal rollout action while preserving teacher labels."""

        config = self.config
        if config.behavior_policy == "teacher":
            return teacher_action
        digest = sha256(
            b"IRMCBEHAVIOR\x01"
            + bytes.fromhex(self.manifest_sha256)
            + pack(">Q", episode_seed)
            + pack(">Q", tick)
        ).digest()
        draw = int.from_bytes(digest[:8], "big")
        threshold = int(config.behavior_intervention_rate * (1 << 64))
        if draw >= threshold:
            return teacher_action
        action_class = int.from_bytes(digest[8:16], "big") % len(_ACTION_KEYS)
        key = _ACTION_KEYS[action_class]
        if key is None:
            return GenericControl()
        return GenericControl(keys_down=(key,))

    def _episode_and_window(self, sequence_index: int) -> tuple[int, int | None]:
        """Return (episode_index, tiled window index or None for uniform)."""

        if self.config.window_sampling != "tiled":
            return sequence_index, None
        windows = self.config.windows_per_episode
        return sequence_index // windows, sequence_index % windows

    def _episode_window(
        self,
        sequence_index: int,
        raw_transitions: list[_TransitionWithoutValue],
        *,
        window_index: int | None,
    ) -> list[_TransitionWithoutValue]:
        """Slice a fixed-length window from a longer planner trajectory.

        Spawn-only datasets (``episode_horizon == 0``) already generated
        exactly ``sequence_length`` ticks from reset, so they pass through.
        Episode-window datasets pick a deterministic uniform start so later
        pellets and corridor choices can appear. Tiled sampling walks
        consecutive non-overlapping windows of one episode instead, so the
        teacher actions match a full play-eval horizon 1:1. Value targets
        are then recomputed on the sliced window (zero-bootstrap at the
        window end) so the 8-tick value scale stays comparable to the
        working probes — lengthening the *window* was the falsified
        window-32 idea.
        """

        length = self.config.sequence_length
        if self.config.episode_horizon <= 0:
            return raw_transitions
        available = len(raw_transitions)
        if available < length:
            raise RuntimeError(
                "episode window needs "
                f"{length} ticks but the planner trajectory has {available}"
            )
        max_start = available - length
        if window_index is not None:
            start = window_index * length
            if start > max_start:
                raise RuntimeError(
                    "tiled window "
                    f"{window_index} starts at tick {start} past {max_start}"
                )
        elif max_start == 0:
            start = 0
        else:
            digest = sha256(
                b"IRMCWINDOW\x01"
                + bytes.fromhex(self.manifest_sha256)
                + pack(">Q", sequence_index)
            ).digest()
            start = int.from_bytes(digest[:8], "big") % (max_start + 1)
        window = raw_transitions[start : start + length]
        last = window[-1]
        if not last.terminated_target and not last.truncated_target:
            window[-1] = replace(
                last,
                terminated_target=False,
                truncated_target=True,
            )
        return window

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
    "MazeChaseCounterfactualSequence",
    "MazeChaseCounterfactualTarget",
    "MazeChaseCounterfactualTransition",
    "MazeChaseSequence",
    "MazeChaseSequenceDataset",
    "MazeChaseTransition",
    "maze_chase_dataset_manifest_sha256",
]
