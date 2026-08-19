"""Interactive On-Policy DAgger (Dataset Aggregation) distillation engine.

Offline behavior cloning on static planner windows leads to policy collapse
(e.g., sticky keys or wall-hugging) in closed-loop play due to covariate shift.
DAgger addresses this by letting the student model run closed-loop in the
environment, querying the expert lookahead planner at visited states for exact
corrective recovery actions, aggregating those recovery transitions into a replay
buffer, and performing iterative bounded updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import random
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from ..data.moving_shapes_dataset import (
    DatasetSplit,
    MovingShapesSequence,
    MovingShapesTransition,
    _TransitionWithoutValue,
    split_episode_seed,
)
from ..environments.maze_chase import MazeChaseEnv
from ..evaluation.closed_loop_play import (
    EXCLUSIVE_ARGMAX_WASD_V1,
    PLAY_DECODE_KINDS,
    decode_closed_loop_control,
)
from ..evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
from ..types import GenericControl, Observation
from .batches import CONTINUOUS_TARGET_INDICES, TrajectoryBatch, control_to_vector
from .objective import _rgb_tensor, deterministic_eval_thought_noise
from .protocol import TrainingStepResult
from .torch_system import TorchTrainingSystem

_DAGGER_MANIFEST_SEED = b"IRENE_DAGGER_BUFFER_V1"


def _validate_int(value: object, *, name: str, minimum: int = 0, maximum: int = 1 << 62) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _validate_float(value: object, *, name: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    val = float(value)
    if not math.isfinite(val) or val < minimum or val > maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    return val


@dataclass(frozen=True, slots=True)
class DAggerConfig:
    """Configuration for on-policy interactive DAgger distillation."""

    iterations: int = 10
    episodes_per_iteration: int = 4
    max_ticks_per_episode: int = 240
    initial_beta: float = 1.0
    beta_decay: float = 0.5
    min_beta: float = 0.0
    sequence_length: int = 8
    burn_in_steps: int = 1
    batch_size: int = 4
    updates_per_iteration: int = 32
    buffer_capacity: int = 100_000
    discount: float = 0.99
    stride: int | None = None
    decode_kind: str = EXCLUSIVE_ARGMAX_WASD_V1
    seed: int = 20260818
    ghost_count: int = 3
    ghost_period: int = 2
    extra_loops: int = 16

    def __post_init__(self) -> None:
        _validate_int(self.iterations, name="iterations", minimum=1)
        _validate_int(self.episodes_per_iteration, name="episodes_per_iteration", minimum=1)
        _validate_int(self.max_ticks_per_episode, name="max_ticks_per_episode", minimum=1)
        _validate_float(self.initial_beta, name="initial_beta", minimum=0.0, maximum=1.0)
        _validate_float(self.beta_decay, name="beta_decay", minimum=0.0, maximum=1.0)
        _validate_float(self.min_beta, name="min_beta", minimum=0.0, maximum=1.0)
        _validate_int(self.sequence_length, name="sequence_length", minimum=2)
        _validate_int(self.burn_in_steps, name="burn_in_steps", minimum=0)
        if self.burn_in_steps >= self.sequence_length:
            raise ValueError("burn_in_steps must be strictly less than sequence_length")
        _validate_int(self.batch_size, name="batch_size", minimum=1)
        _validate_int(self.updates_per_iteration, name="updates_per_iteration", minimum=1)
        _validate_int(self.buffer_capacity, name="buffer_capacity", minimum=1)
        _validate_float(self.discount, name="discount", minimum=0.0, maximum=1.0)
        if self.stride is not None:
            _validate_int(self.stride, name="stride", minimum=1)
        if not isinstance(self.decode_kind, str) or self.decode_kind not in PLAY_DECODE_KINDS:
            raise ValueError(f"decode_kind must be one of {sorted(PLAY_DECODE_KINDS)}")
        _validate_int(self.seed, name="seed", minimum=0)
        _validate_int(self.ghost_count, name="ghost_count", minimum=1, maximum=8)
        _validate_int(self.ghost_period, name="ghost_period", minimum=1, maximum=64)
        _validate_int(self.extra_loops, name="extra_loops", minimum=0, maximum=64)

    def beta_for_iteration(self, iteration: int) -> float:
        """Compute the expert sampling probability beta for a given iteration."""
        _validate_int(iteration, name="iteration", minimum=0)
        computed = self.initial_beta * (self.beta_decay ** iteration)
        return max(self.min_beta, min(1.0, float(computed)))


@dataclass(frozen=True, slots=True)
class DAggerIterationResult:
    """Telemetry report produced after completing one DAgger iteration."""

    iteration: int
    beta: float
    episodes_collected: int
    transitions_collected: int
    sequences_in_buffer: int
    mean_train_loss: float
    training_metrics: Mapping[str, float]
    rollout_metrics: Mapping[str, float]


def _build_sequence_from_window(
    window: Sequence[MovingShapesTransition],
    *,
    split: DatasetSplit,
    sequence_index: int,
    discount: float,
    manifest_sha256: str,
) -> MovingShapesSequence:
    """Package a contiguous window of transitions into an immutable MovingShapesSequence."""
    if not window:
        raise ValueError("window cannot be empty")
    length = len(window)
    raw_window = list(window)
    last = raw_window[-1]
    is_terminal = last.terminated_target or last.truncated_target
    if not is_terminal:
        raw_window[-1] = MovingShapesTransition(
            observation=last.observation,
            applied_control=last.applied_control,
            action_target=last.action_target,
            next_observation_target=last.next_observation_target,
            reward_target=last.reward_target,
            value_target=0.0,
            event_targets=last.event_targets,
            terminated_target=False,
            truncated_target=True,
        )

    running_return = 0.0
    reversed_transitions: list[MovingShapesTransition] = []
    for raw in reversed(raw_window):
        done = raw.terminated_target or raw.truncated_target
        running_return = raw.reward_target + (0.0 if done else discount * running_return)
        reversed_transitions.append(
            MovingShapesTransition(
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

    seq_transitions = tuple(reversed(reversed_transitions))
    episode_seed = split_episode_seed(split, sequence_index)
    return MovingShapesSequence(
        split=split,
        sequence_index=sequence_index,
        episode_seed=episode_seed,
        discount=discount,
        manifest_sha256=manifest_sha256,
        transitions=seq_transitions,
    )


class DAggerAggregationBuffer:
    """Aggregation buffer holding causal sequences for DAgger training."""

    __slots__ = (
        "_sequence_length",
        "_capacity",
        "_discount",
        "_stride",
        "_manifest_sha256",
        "_sequences",
        "_next_seq_idx",
        "_total_transitions_added",
        "_total_episodes_added",
    )

    def __init__(
        self,
        *,
        sequence_length: int = 8,
        capacity: int = 100_000,
        discount: float = 0.99,
        stride: int | None = None,
    ) -> None:
        _validate_int(sequence_length, name="sequence_length", minimum=2)
        _validate_int(capacity, name="capacity", minimum=1)
        _validate_float(discount, name="discount", minimum=0.0, maximum=1.0)
        if stride is not None:
            _validate_int(stride, name="stride", minimum=1)
        self._sequence_length = sequence_length
        self._capacity = capacity
        self._discount = discount
        self._stride = sequence_length if stride is None else stride
        self._manifest_sha256 = sha256(
            _DAGGER_MANIFEST_SEED + f":len={sequence_length}:disc={discount}".encode("ascii")
        ).hexdigest()
        self._sequences: list[MovingShapesSequence] = []
        self._next_seq_idx = 0
        self._total_transitions_added = 0
        self._total_episodes_added = 0

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_transitions(self) -> int:
        return self._total_transitions_added

    @property
    def total_episodes(self) -> int:
        return self._total_episodes_added

    @property
    def sequences(self) -> tuple[MovingShapesSequence, ...]:
        return tuple(self._sequences)

    def __len__(self) -> int:
        return len(self._sequences)

    def clear(self) -> None:
        """Reset the aggregation buffer."""
        self._sequences.clear()
        self._next_seq_idx = 0
        self._total_transitions_added = 0
        self._total_episodes_added = 0

    def add_sequence(self, sequence: MovingShapesSequence) -> None:
        """Add an existing pre-built MovingShapesSequence directly into the buffer."""
        if not isinstance(sequence, MovingShapesSequence):
            raise TypeError("sequence must be a MovingShapesSequence")
        if len(sequence.transitions) != self._sequence_length:
            raise ValueError(
                f"sequence length {len(sequence.transitions)} does not match buffer length {self._sequence_length}"
            )
        self._sequences.append(sequence)
        self._total_transitions_added += len(sequence.transitions)
        if len(self._sequences) > self._capacity:
            excess = len(self._sequences) - self._capacity
            self._sequences = self._sequences[excess:]

    def add_rollout(
        self,
        transitions: Sequence[MovingShapesTransition],
        *,
        split: DatasetSplit = DatasetSplit.TRAIN,
    ) -> int:
        """Slice an episode rollout of transitions into sequences and append them to the buffer."""
        if not isinstance(transitions, Sequence) or not transitions:
            return 0
        for t in transitions:
            if not isinstance(t, MovingShapesTransition):
                raise TypeError("transitions elements must be MovingShapesTransition")

        self._total_episodes_added += 1
        self._total_transitions_added += len(transitions)
        length = self._sequence_length
        stride = self._stride
        total = len(transitions)

        if total < length:
            # Episode shorter than sequence length cannot form a full unpadded window.
            return 0

        added_count = 0
        starts: list[int] = list(range(0, total - length + 1, stride))
        last_possible = total - length
        if last_possible not in starts:
            starts.append(last_possible)

        for start in starts:
            window = transitions[start : start + length]
            seq = _build_sequence_from_window(
                window,
                split=split,
                sequence_index=self._next_seq_idx,
                discount=self._discount,
                manifest_sha256=self._manifest_sha256,
            )
            self._sequences.append(seq)
            self._next_seq_idx += 1
            added_count += 1

        if len(self._sequences) > self._capacity:
            excess = len(self._sequences) - self._capacity
            self._sequences = self._sequences[excess:]

        return added_count

    def sample_batch(
        self,
        batch_size: int,
        burn_in_steps: int = 1,
        *,
        rng: random.Random | None = None,
    ) -> TrajectoryBatch:
        """Sample a TrajectoryBatch uniformly from the aggregation buffer."""
        if not self._sequences:
            raise ValueError("cannot sample from an empty aggregation buffer")
        _validate_int(batch_size, name="batch_size", minimum=1)
        _validate_int(burn_in_steps, name="burn_in_steps", minimum=0)
        if burn_in_steps >= self._sequence_length:
            raise ValueError("burn_in_steps must be strictly less than sequence_length")

        if rng is None:
            rng = random.Random()

        if len(self._sequences) >= batch_size:
            sampled = rng.sample(self._sequences, k=batch_size)
        else:
            sampled = rng.choices(self._sequences, k=batch_size)

        return TrajectoryBatch(
            split="train",
            burn_in_steps=burn_in_steps,
            sequences=tuple(sampled),
        )


def collect_interactive_rollout(
    student_model: object | None,
    env: object,
    expert_planner: object,
    beta: float,
    *,
    seed: int | None = None,
    discount: float = 0.99,
    decode_kind: str = EXCLUSIVE_ARGMAX_WASD_V1,
    max_ticks: int | None = None,
    rng: random.Random | None = None,
) -> tuple[MovingShapesTransition, ...]:
    """Execute an interactive rollout with student driving and expert labeling.

    At each step of the episode:
    1. Query the expert planner for the corrective action given the current observation.
    2. Query the student model (if provided) for its predicted action.
    3. Mix actions with probability beta: with probability beta use the expert action;
       with probability (1 - beta) use the student action.
    4. Step the environment using the chosen action (driving the trajectory into the
       student's visited state distribution).
    5. Record the transition where the applied control is the executed action, but the
       supervision action target is ALWAYS the expert planner's corrective label.
    6. Compute exact discounted value targets backwards across the episode.

    Returns:
        tuple of MovingShapesTransition forming a complete closed-loop episode.
    """
    _validate_float(beta, name="beta", minimum=0.0, maximum=1.0)
    _validate_float(discount, name="discount", minimum=0.0, maximum=1.0)
    if decode_kind not in PLAY_DECODE_KINDS:
        raise ValueError(f"decode_kind must be one of {sorted(PLAY_DECODE_KINDS)}")

    if rng is None:
        rng = random.Random(seed) if seed is not None else random.Random()

    if seed is not None:
        if hasattr(expert_planner, "reset") and callable(expert_planner.reset):
            expert_planner.reset(seed)
        if getattr(expert_planner, "uses_privileged_state", False) and hasattr(expert_planner, "bind"):
            expert_planner.bind(env)
        raw_observation = env.reset(seed)
    else:
        raw_observation = getattr(env, "current_observation", None)
        if raw_observation is None:
            raw_observation = env.reset(0)

    is_torch_model = isinstance(student_model, nn.Module)
    was_training = False
    thought_noise = None
    device = torch.device("cpu")
    resolution = None
    state: list[object | None] = [None]

    if is_torch_model:
        parameter = next(student_model.parameters(), None)
        if parameter is not None:
            device = parameter.device
        was_training = student_model.training
        student_model.eval()
        if hasattr(student_model, "config") and hasattr(student_model.config, "thoughtlets"):
            thought_noise = deterministic_eval_thought_noise(
                thoughtlets=student_model.config.thoughtlets,
                width=student_model.config.core_width,
                batch_size=1,
                device=device,
            )
        resolution = getattr(student_model, "input_resolution", None)

    raw_transitions: list[_TransitionWithoutValue] = []
    tick_count = 0

    try:
        while True:
            if max_ticks is not None and tick_count >= max_ticks:
                break

            # 1. Evaluate expert planner's recommended action on current state observation.
            expert_action = expert_planner.act(raw_observation)
            if not isinstance(expert_action, GenericControl):
                raise TypeError(f"expert_planner.act() must return GenericControl, got {type(expert_action)}")

            # 2. Evaluate student action if available.
            student_action: GenericControl | None = None
            if student_model is not None:
                if is_torch_model:
                    pixels = _rgb_tensor(
                        (raw_observation.rgb,),
                        device=device,
                        resolution=resolution,
                    )
                    previous = torch.tensor(
                        [control_to_vector(raw_observation.previous_control)],
                        dtype=torch.float32,
                        device=device,
                    )
                    elapsed_seconds = (
                        0.0
                        if tick_count == 0
                        else (
                            raw_observation.elapsed_ns
                            - raw_transitions[-1].observation.elapsed_ns
                        )
                        / 1_000_000_000.0
                    )
                    elapsed = torch.tensor([elapsed_seconds], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        output = student_model(
                            pixels,
                            previous,
                            elapsed,
                            state[0],
                            thought_noise=thought_noise,
                        )
                    state[0] = output.next_state.detach()
                    logits = output.action.button_logits[0].float().cpu()
                    continuous_values = output.action.control[0].float().cpu()
                    student_action, _ = decode_closed_loop_control(
                        logits.tolist(),
                        [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                        decode_kind=decode_kind,
                    )
                elif hasattr(student_model, "act") and callable(student_model.act):
                    student_action = student_model.act(raw_observation)

            # 3. Action selection: mixture with probability beta of expert.
            if student_action is None or beta >= 1.0:
                applied_action = expert_action
            elif beta <= 0.0:
                applied_action = student_action
            else:
                applied_action = expert_action if rng.random() < beta else student_action

            # 4. Step environment.
            outcome = env.step(applied_action)
            tick_count += 1

            # 5. Record transition with expert recovery action target.
            obs_model = raw_observation.to_model_observation(observation_age_ns=0)
            next_obs_model = outcome.observation.to_model_observation(observation_age_ns=0)
            raw_transitions.append(
                _TransitionWithoutValue(
                    observation=obs_model,
                    applied_control=outcome.applied_control,
                    action_target=expert_action,
                    next_observation_target=next_obs_model,
                    reward_target=outcome.reward,
                    event_targets=tuple(outcome.events),
                    terminated_target=outcome.terminated,
                    truncated_target=outcome.truncated,
                )
            )
            raw_observation = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
    finally:
        if is_torch_model and was_training:
            student_model.train()

    # 6. Compute discounted sequence returns backwards.
    running_return = 0.0
    reversed_transitions: list[MovingShapesTransition] = []
    for raw in reversed(raw_transitions):
        done = raw.terminated_target or raw.truncated_target
        running_return = raw.reward_target + (0.0 if done else discount * running_return)
        reversed_transitions.append(
            MovingShapesTransition(
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

    return tuple(reversed(reversed_transitions))


class DAggerDistiller:
    """Interactive On-Policy DAgger distillation coordinator."""

    def __init__(
        self,
        config: DAggerConfig,
        student_model: object | None,
        training_system: TorchTrainingSystem | object | None = None,
        *,
        buffer: DAggerAggregationBuffer | None = None,
        env_factory: Callable[[], object] | None = None,
        expert_planner_factory: Callable[[], object] | None = None,
    ) -> None:
        if not isinstance(config, DAggerConfig):
            raise TypeError("config must be a DAggerConfig")
        self.config = config
        self.student_model = student_model
        self.training_system = training_system
        self.buffer = (
            buffer
            if buffer is not None
            else DAggerAggregationBuffer(
                sequence_length=config.sequence_length,
                capacity=config.buffer_capacity,
                discount=config.discount,
                stride=config.stride,
            )
        )
        self._env_factory = (
            env_factory
            if env_factory is not None
            else lambda: MazeChaseEnv(
                ghost_count=config.ghost_count,
                ghost_period=config.ghost_period,
                extra_loops=config.extra_loops,
                max_ticks=config.max_ticks_per_episode,
            )
        )
        self._expert_planner_factory = (
            expert_planner_factory
            if expert_planner_factory is not None
            else lambda: ScriptedMazeChasePlannerPolicy(
                ghost_period=config.ghost_period,
            )
        )
        self._rng = random.Random(config.seed)

    def collect_interactive_rollout(
        self,
        student_model: object | None = None,
        env: object | None = None,
        expert_planner: object | None = None,
        beta: float | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[MovingShapesTransition, ...]:
        """Collect one interactive rollout."""
        model = student_model if student_model is not None else self.student_model
        target_env = env if env is not None else self._env_factory()
        planner = expert_planner if expert_planner is not None else self._expert_planner_factory()
        sampling_beta = self.config.initial_beta if beta is None else beta

        return collect_interactive_rollout(
            student_model=model,
            env=target_env,
            expert_planner=planner,
            beta=sampling_beta,
            seed=seed,
            discount=self.config.discount,
            decode_kind=self.config.decode_kind,
            max_ticks=self.config.max_ticks_per_episode,
            rng=self._rng,
        )

    def run_dagger_iteration(
        self,
        iteration: int,
        *,
        beta: float | None = None,
        num_episodes: int | None = None,
        num_updates: int | None = None,
    ) -> DAggerIterationResult:
        """Run one full DAgger iteration: collect rollouts -> aggregate -> optimize."""
        _validate_int(iteration, name="iteration", minimum=0)
        current_beta = self.config.beta_for_iteration(iteration) if beta is None else beta
        episodes_to_collect = (
            self.config.episodes_per_iteration if num_episodes is None else num_episodes
        )
        updates_to_run = (
            self.config.updates_per_iteration if num_updates is None else num_updates
        )

        transitions_collected = 0
        total_reward = 0.0
        total_ticks = 0
        event_counts: dict[str, int] = {}

        # 1. Collect interactive rollouts on the student policy with expert labeling.
        for ep in range(episodes_to_collect):
            ep_seed = self.config.seed + iteration * 10_000 + ep
            env = self._env_factory()
            expert = self._expert_planner_factory()
            rollout = self.collect_interactive_rollout(
                student_model=self.student_model,
                env=env,
                expert_planner=expert,
                beta=current_beta,
                seed=ep_seed,
            )
            self.buffer.add_rollout(rollout)
            transitions_collected += len(rollout)
            total_reward += sum(t.reward_target for t in rollout)
            total_ticks += len(rollout)
            for t in rollout:
                for ev in t.event_targets:
                    event_counts[ev] = event_counts.get(ev, 0) + 1

        # 2. Run bounded optimizer updates on the aggregated buffer.
        loss_sum = 0.0
        metric_sums: dict[str, float] = {}
        actual_updates = 0

        if self.training_system is not None and len(self.buffer) > 0:
            for _ in range(updates_to_run):
                batch = self.buffer.sample_batch(
                    batch_size=self.config.batch_size,
                    burn_in_steps=self.config.burn_in_steps,
                    rng=self._rng,
                )
                step_result: TrainingStepResult = self.training_system.train_optimizer_step((batch,))
                loss_sum += step_result.loss
                for k, v in step_result.metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                actual_updates += 1

        mean_loss = loss_sum / actual_updates if actual_updates > 0 else 0.0
        mean_metrics = (
            {k: v / actual_updates for k, v in metric_sums.items()}
            if actual_updates > 0
            else {}
        )
        rollout_metrics = {
            "mean_reward": total_reward / episodes_to_collect if episodes_to_collect > 0 else 0.0,
            "mean_ticks": total_ticks / episodes_to_collect if episodes_to_collect > 0 else 0.0,
            "pellets_eaten": float(event_counts.get("pellet_eaten", 0)),
            "collisions": float(event_counts.get("collision", 0) + event_counts.get("caught", 0)),
        }

        return DAggerIterationResult(
            iteration=iteration,
            beta=current_beta,
            episodes_collected=episodes_to_collect,
            transitions_collected=transitions_collected,
            sequences_in_buffer=len(self.buffer),
            mean_train_loss=mean_loss,
            training_metrics=mean_metrics,
            rollout_metrics=rollout_metrics,
        )

    def train_dagger(
        self,
        num_iterations: int | None = None,
    ) -> tuple[DAggerIterationResult, ...]:
        """Execute the full sequence of DAgger distillation iterations."""
        total_iterations = self.config.iterations if num_iterations is None else num_iterations
        results: list[DAggerIterationResult] = []
        for it in range(total_iterations):
            res = self.run_dagger_iteration(it)
            results.append(res)
        return tuple(results)


__all__ = [
    "DAggerAggregationBuffer",
    "DAggerConfig",
    "DAggerDistiller",
    "DAggerIterationResult",
    "collect_interactive_rollout",
]
