"""Closed-loop moving-shapes play evaluator.

Open-loop gates (RCQ development slices) measure teacher-forced action
agreement. This module measures the other thing the roadmap needs: a model
driving the in-repo world in real time through
:class:`irene_brain.runtime.continuous.ContinuousDriver`, with a deterministic
:class:`irene_brain.runtime.clock.ManualClock` and a declared simulated
inference latency. It reports closed-loop task outcomes (targets, collisions,
reward), timing behavior (deadline misses, stale-frame rejections, dropped
observations), and the two action-path failure modes from the RCQ-v2
post-mortem (opposite-direction conflicts, continuous outputs outside the
deadzone) so closed-loop evidence can be compared with open-loop gate rows.

Everything here is simulated and deterministic: no wall-clock latency is ever
reported as a physical measurement, no real HID output occurs, and the model
only ever sees canonical observations. The module is not a qualification
component; it produces exploratory evidence records with a canonical SHA-256
so results can be cited and compared exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Mapping, Sequence

from ..runtime.clock import ManualClock
from ..runtime.continuous import (
    ContinuousDriver,
    PostAdvanceRuntimeError,
    PostAdvanceValueError,
)
from ..training.batches import BUTTON_TARGET_INDICES
from ..types import ActionEnvelope, GenericControl, Observation, StepOutcome


_DEADZONE = 0.05
# HID usage identifiers in W, A, S, D bit order (matches MovingShapesEnv._KEY_BITS).
_MOVEMENT_KEYS = (26, 4, 22, 7)
_NANOSECONDS_PER_SECOND = 1_000_000_000


def _plain_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class ClosedLoopPlayConfig:
    """Fixed knobs for one closed-loop play evaluation campaign."""

    episode_seeds: tuple[int, ...]
    max_ticks: int = 600
    hazard_count: int = 3
    tick_period_ns: int = 16_666_667
    decision_interval_ns: int = 16_666_667
    inference_latency_ns: int = 0
    submit_deadline_slack_ns: int = 0
    expiry_slack_ns: int = 33_333_334

    def __post_init__(self) -> None:
        if not isinstance(self.episode_seeds, tuple) or not self.episode_seeds:
            raise ValueError("episode_seeds must be a non-empty tuple")
        if len(set(self.episode_seeds)) != len(self.episode_seeds):
            raise ValueError("episode_seeds must be unique")
        for seed in self.episode_seeds:
            _plain_int(seed, name="episode seed", minimum=0, maximum=2**64 - 1)
        _plain_int(self.max_ticks, name="max_ticks", minimum=1, maximum=2**32 - 1)
        _plain_int(self.hazard_count, name="hazard_count", minimum=1, maximum=253)
        _plain_int(
            self.tick_period_ns,
            name="tick_period_ns",
            minimum=1,
            maximum=2**63 - 1,
        )
        _plain_int(
            self.decision_interval_ns,
            name="decision_interval_ns",
            minimum=1,
            maximum=2**63 - 1,
        )
        _plain_int(
            self.inference_latency_ns,
            name="inference_latency_ns",
            minimum=0,
            maximum=2**63 - 1,
        )
        _plain_int(
            self.submit_deadline_slack_ns,
            name="submit_deadline_slack_ns",
            minimum=0,
            maximum=2**63 - 1,
        )
        _plain_int(
            self.expiry_slack_ns,
            name="expiry_slack_ns",
            minimum=1,
            maximum=2**63 - 1,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_interval_ns": self.decision_interval_ns,
            "episode_seeds": list(self.episode_seeds),
            "expiry_slack_ns": self.expiry_slack_ns,
            "hazard_count": self.hazard_count,
            "inference_latency_ns": self.inference_latency_ns,
            "max_ticks": self.max_ticks,
            "submit_deadline_slack_ns": self.submit_deadline_slack_ns,
            "tick_period_ns": self.tick_period_ns,
        }


class _EventCountingEnvironment:
    """EnvironmentProtocol wrapper that records per-step events and rewards."""

    def __init__(self, environment: object) -> None:
        self._environment = environment
        self.events: list[str] = []
        self.reward_sum = 0.0

    @property
    def tick_period_ns(self) -> int:
        return self._environment.tick_period_ns

    @property
    def current_observation(self) -> Observation:
        return self._environment.current_observation

    def step(self, control: GenericControl) -> StepOutcome:
        outcome = self._environment.step(control)
        self.events.extend(outcome.events)
        self.reward_sum += outcome.reward
        return outcome


def _control_stats(
    keys: Sequence[int],
    mouse_buttons: Sequence[int],
    gamepad_buttons: Sequence[int],
    continuous: Sequence[float],
) -> dict[str, int | float]:
    movement = [key for key in keys if key in _MOVEMENT_KEYS]
    movement_set = set(movement)
    opposite_conflicts = int(
        (26 in movement_set and 22 in movement_set)  # W and S together
        or (4 in movement_set and 7 in movement_set)  # A and D together
    )
    outside_deadzone = sum(
        1 for value in continuous if abs(float(value)) > _DEADZONE
    )
    return {
        "active_button_count": len(keys) + len(mouse_buttons) + len(gamepad_buttons),
        "movement_mask": sum(1 << bit for bit in range(4) if _MOVEMENT_KEYS[bit] in movement_set),
        "non_movement_key_count": len([key for key in keys if key not in _MOVEMENT_KEYS]),
        "mouse_button_count": len(mouse_buttons),
        "gamepad_button_count": len(gamepad_buttons),
        "opposite_conflict": opposite_conflicts,
        "continuous_outside_deadzone": outside_deadzone,
        "continuous_max_abs": max(
            (abs(float(value)) for value in continuous), default=0.0
        ),
    }


def control_audit_stats(control: GenericControl) -> dict[str, int | float]:
    """Return the decode-style audit stats for an already-formed control.

    Non-model decision sources (the diagnostic policies) produce a
    ``GenericControl`` directly; this helper fills the same stat fields the
    model decode produces so diagnostic and model episode reports stay
    column-comparable. Continuous channels are structurally zero here.
    """

    if not isinstance(control, GenericControl):
        raise TypeError("control must be a GenericControl")
    return _control_stats(
        control.keys_down,
        control.mouse_buttons,
        control.gamepad_buttons,
        (),
    )


def decode_closed_loop_control(
    button_logits: Sequence[float],
    continuous: Sequence[float],
) -> tuple[GenericControl, dict[str, int | float]]:
    """Decode the final-exit action into a GenericControl plus audit stats.

    ``button_logits`` is the packed 296-entry vector emitted by the model; it
    is unpacked through ``BUTTON_TARGET_INDICES`` exactly like the open-loop
    RCQ decode, so a channel activates strictly above a zero logit. Continuous
    channels never actuate in the closed loop (the moving-shapes world reads
    only W/A/S/D), but their deadzone violations are counted so the RCQ-v2
    quiescence failure stays visible here.
    """

    if len(button_logits) != len(BUTTON_TARGET_INDICES):
        raise ValueError(
            f"button_logits must contain exactly {len(BUTTON_TARGET_INDICES)} entries"
        )
    if len(continuous) != 11:
        raise ValueError("continuous must contain exactly 11 entries")
    active = [
        BUTTON_TARGET_INDICES[index]
        for index, logit in enumerate(button_logits)
        if logit > 0.0
    ]
    keys = tuple(index for index in active if index < 256)
    mouse_buttons = tuple(index - 256 for index in active if 256 <= index < 264)
    gamepad_buttons = tuple(index - 267 for index in active if index >= 267)
    stats = _control_stats(keys, mouse_buttons, gamepad_buttons, continuous)
    control = GenericControl(
        keys_down=keys,
        mouse_buttons=mouse_buttons,
        gamepad_buttons=gamepad_buttons,
    )
    return control, stats


@dataclass(frozen=True, slots=True)
class ClosedLoopEpisodeReport:
    episode_seed: int
    ticks_advanced: int
    reward_sum: float
    targets_collected: int
    collisions: int
    decisions_submitted: int
    decisions_rejected: int
    rejections_by_reason: tuple[tuple[str, int], ...]
    observations_dropped: int
    movement_mask_histogram: tuple[tuple[int, int], ...]
    opposite_conflicts: int
    non_movement_key_activations: int
    continuous_outside_deadzone: int
    continuous_max_abs: float
    mean_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "collisions": self.collisions,
            "continuous_max_abs": self.continuous_max_abs,
            "continuous_outside_deadzone": self.continuous_outside_deadzone,
            "decisions_rejected": self.decisions_rejected,
            "decisions_submitted": self.decisions_submitted,
            "episode_seed": self.episode_seed,
            "mean_value": self.mean_value,
            "movement_mask_histogram": [
                [mask, count] for mask, count in self.movement_mask_histogram
            ],
            "non_movement_key_activations": self.non_movement_key_activations,
            "observations_dropped": self.observations_dropped,
            "opposite_conflicts": self.opposite_conflicts,
            "rejections_by_reason": [
                [reason, count] for reason, count in self.rejections_by_reason
            ],
            "reward_sum": self.reward_sum,
            "targets_collected": self.targets_collected,
            "ticks_advanced": self.ticks_advanced,
        }


@dataclass(frozen=True, slots=True)
class ClosedLoopPlayReport:
    """Canonical evidence record for one closed-loop play evaluation."""

    schema_version: int
    config: ClosedLoopPlayConfig
    model_description: str
    episodes: tuple[ClosedLoopEpisodeReport, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported closed-loop report schema")
        if not isinstance(self.model_description, str) or not self.model_description:
            raise ValueError("model_description must be a non-empty string")
        if len(self.episodes) != len(self.config.episode_seeds):
            raise ValueError("one episode report is required per registered seed")
        if tuple(episode.episode_seed for episode in self.episodes) != tuple(
            self.config.episode_seeds
        ):
            raise ValueError("episode reports must follow the registered seed order")

    def to_dict(self) -> dict[str, object]:
        targets = sum(episode.targets_collected for episode in self.episodes)
        collisions = sum(episode.collisions for episode in self.episodes)
        rejected = sum(episode.decisions_rejected for episode in self.episodes)
        submitted = sum(episode.decisions_submitted for episode in self.episodes)
        return {
            "config": self.config.to_dict(),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "model_description": self.model_description,
            "schema_version": self.schema_version,
            "totals": {
                "collisions": collisions,
                "decisions_rejected": rejected,
                "decisions_submitted": submitted,
                "episodes": len(self.episodes),
                "reward_sum": sum(episode.reward_sum for episode in self.episodes),
                "targets_collected": targets,
                "ticks_advanced": sum(
                    episode.ticks_advanced for episode in self.episodes
                ),
            },
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


# A decision source maps (observation, seconds since the previous decision)
# to a control, decode-style audit stats, and an optional value estimate.
_DecisionFn = Callable[
    ["Observation", float],
    "tuple[GenericControl, Mapping[str, int | float], float | None]",
]


def _run_episode_core(
    *,
    seed: int,
    config: ClosedLoopPlayConfig,
    decide: _DecisionFn,
    on_environment: Callable[[object], None] | None = None,
    environment_factory: Callable[[], object] | None = None,
) -> ClosedLoopEpisodeReport:
    """Run one deterministic closed-loop episode for any decision source.

    ``environment_factory`` defaults to the moving-shapes world driven by the
    config knobs; pass any branchable in-repo world (for example
    ``PursuitEnv``) to evaluate decision sources on the environment ladder.
    Contact events are counted as ``collisions`` whether the world calls them
    "collision" (moving shapes) or "caught" (pursuit).
    """

    if environment_factory is None:
        from ..environments.moving_shapes import MovingShapesEnv

        def environment_factory() -> object:
            return MovingShapesEnv(
                hazard_count=config.hazard_count,
                tick_period_ns=config.tick_period_ns,
                max_ticks=config.max_ticks,
            )

    environment = _EventCountingEnvironment(environment_factory())
    environment._environment.reset(seed)
    if on_environment is not None:
        on_environment(environment._environment)
    clock = ManualClock(frequency_hz=_NANOSECONDS_PER_SECOND)
    driver = ContinuousDriver(environment, clock, start_tick=clock.now_ticks())

    action_sequence = 0
    decisions_submitted = 0
    rejections: dict[str, int] = {}
    mask_histogram: dict[int, int] = {}
    opposite_conflicts = 0
    non_movement_activations = 0
    outside_deadzone = 0
    continuous_max_abs = 0.0
    value_sum = 0.0
    value_count = 0
    previous_observation_elapsed_ns: int | None = None

    while not driver.halted:
        observation = driver.poll_latest_observation()
        if observation.frame_id >= config.max_ticks:
            break
        elapsed_ns = observation.elapsed_ns
        if previous_observation_elapsed_ns is None:
            elapsed_seconds = 0.0
        else:
            elapsed_seconds = (
                elapsed_ns - previous_observation_elapsed_ns
            ) / _NANOSECONDS_PER_SECOND
        previous_observation_elapsed_ns = elapsed_ns

        control, stats, value = decide(observation, elapsed_seconds)
        if value is not None:
            value_sum += value
            value_count += 1
        mask_histogram[int(stats["movement_mask"])] = (
            mask_histogram.get(int(stats["movement_mask"]), 0) + 1
        )
        opposite_conflicts += int(stats["opposite_conflict"])
        non_movement_activations += int(stats["non_movement_key_count"])
        outside_deadzone += int(stats["continuous_outside_deadzone"])
        continuous_max_abs = max(
            continuous_max_abs, float(stats["continuous_max_abs"])
        )

        # Simulated inference latency: the world keeps running while the
        # decision source "computes", so a slow source submits against newer
        # frames.
        created_tick = clock.now_ticks()
        clock.advance_ns(config.inference_latency_ns)
        ready_tick = clock.now_ticks()
        action_sequence += 1
        envelope = ActionEnvelope(
            action_sequence=action_sequence,
            source_frame_id=observation.frame_id,
            model_state_version=action_sequence,
            created_qpc=created_tick,
            ready_qpc=ready_tick,
            submit_deadline_qpc=(
                ready_tick + config.submit_deadline_slack_ns
            ),
            expires_qpc=(
                ready_tick
                + config.submit_deadline_slack_ns
                + config.expiry_slack_ns
            ),
            control=control,
        )
        try:
            driver.submit_action(envelope)
        except PostAdvanceValueError as error:
            reason = str(error).split(":", 1)[0]
            rejections[reason] = rejections.get(reason, 0) + 1
        except PostAdvanceRuntimeError:
            break
        else:
            decisions_submitted += 1

        next_decision_tick = created_tick + config.decision_interval_ns
        if clock.now_ticks() < next_decision_tick:
            clock.set_ticks(next_decision_tick)

    return ClosedLoopEpisodeReport(
        episode_seed=seed,
        ticks_advanced=driver.step_count,
        reward_sum=environment.reward_sum,
        targets_collected=environment.events.count("target_collected"),
        # Moving shapes reports "collision"; pursuit reports "caught". Both
        # are the same column: the player made contact with a threat.
        collisions=(
            environment.events.count("collision")
            + environment.events.count("caught")
        ),
        decisions_submitted=decisions_submitted,
        decisions_rejected=sum(rejections.values()),
        rejections_by_reason=tuple(sorted(rejections.items())),
        observations_dropped=driver.total_dropped_observations,
        movement_mask_histogram=tuple(sorted(mask_histogram.items())),
        opposite_conflicts=opposite_conflicts,
        non_movement_key_activations=non_movement_activations,
        continuous_outside_deadzone=outside_deadzone,
        continuous_max_abs=continuous_max_abs,
        mean_value=value_sum / value_count if value_count else 0.0,
    )


def run_closed_loop_episode(
    model: object,
    *,
    seed: int,
    config: ClosedLoopPlayConfig,
    device: object,
    environment_factory: Callable[[], object] | None = None,
) -> ClosedLoopEpisodeReport:
    """Play one deterministic closed-loop moving-shapes episode."""

    import torch

    from ..training.batches import CONTINUOUS_TARGET_INDICES, control_to_vector
    from ..training.objective import _rgb_tensor, deterministic_eval_thought_noise

    thought_noise = deterministic_eval_thought_noise(
        thoughtlets=model.config.thoughtlets,
        width=model.config.core_width,
        batch_size=1,
        device=device,
    )
    resolution = getattr(model, "input_resolution", None)
    state: list[object | None] = [None]

    def decide(
        observation: Observation, elapsed_seconds: float
    ) -> tuple[GenericControl, Mapping[str, int | float], float | None]:
        pixels = _rgb_tensor(
            (observation.rgb,), device=device, resolution=resolution
        )
        previous = torch.tensor(
            [control_to_vector(observation.previous_control)],
            dtype=torch.float32,
            device=device,
        )
        elapsed = torch.tensor(
            [elapsed_seconds], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            output = model(
                pixels,
                previous,
                elapsed,
                state[0],
                thought_noise=thought_noise,
            )
        state[0] = output.next_state.detach()
        logits = output.action.button_logits[0].float().cpu()
        continuous_values = output.action.control[0].float().cpu()
        if not (
            bool(torch.isfinite(logits).all())
            and bool(torch.isfinite(continuous_values).all())
        ):
            raise RuntimeError("model produced non-finite closed-loop outputs")
        control, stats = decode_closed_loop_control(
            logits.tolist(),
            [
                float(continuous_values[index])
                for index in CONTINUOUS_TARGET_INDICES
            ],
        )
        return control, stats, float(output.value[0].float().cpu())

    return _run_episode_core(
        seed=seed,
        config=config,
        decide=decide,
        environment_factory=environment_factory,
    )


def run_policy_closed_loop_episode(
    policy: object,
    *,
    seed: int,
    config: ClosedLoopPlayConfig,
    environment_factory: Callable[[], object] | None = None,
) -> ClosedLoopEpisodeReport:
    """Play one closed-loop episode for a non-model diagnostic policy.

    The policy contract lives in ``evaluation/diagnostic_policies.py``:
    ``reset(episode_seed)`` reseeds per episode, ``act(observation)`` returns
    a ``GenericControl``, and policies declaring ``uses_privileged_state``
    receive the environment through ``bind(environment)`` after reset. The
    timing, deadline, and rejection mechanics are identical to the model
    path, so diagnostic and model rows are directly comparable.
    """

    for attribute in ("reset", "act"):
        if not callable(getattr(policy, attribute, None)):
            raise TypeError(f"policy must define {attribute}()")
    policy.reset(seed)

    def bind(environment: object) -> None:
        if getattr(policy, "uses_privileged_state", False):
            policy.bind(environment)

    def decide(
        observation: Observation, elapsed_seconds: float
    ) -> tuple[GenericControl, Mapping[str, int | float], float | None]:
        control = policy.act(observation)
        return control, control_audit_stats(control), None

    return _run_episode_core(
        seed=seed,
        config=config,
        decide=decide,
        on_environment=bind,
        environment_factory=environment_factory,
    )


def evaluate_closed_loop_play(
    model: object,
    *,
    config: ClosedLoopPlayConfig,
    model_description: str,
    environment_factory: Callable[[], object] | None = None,
) -> ClosedLoopPlayReport:
    """Run every registered seed and assemble the canonical report."""

    import torch

    if not isinstance(config, ClosedLoopPlayConfig):
        raise ValueError("config must be a ClosedLoopPlayConfig")
    if not isinstance(model_description, str) or not model_description:
        raise ValueError("model_description must be a non-empty string")
    device = torch.device("cpu")
    was_training = model.training
    model.eval()
    try:
        episodes = tuple(
            run_closed_loop_episode(
                model,
                seed=seed,
                config=config,
                device=device,
                environment_factory=environment_factory,
            )
            for seed in config.episode_seeds
        )
    finally:
        if was_training:
            model.train()
    return ClosedLoopPlayReport(
        schema_version=1,
        config=config,
        model_description=model_description,
        episodes=episodes,
    )


__all__ = [
    "ClosedLoopEpisodeReport",
    "ClosedLoopPlayConfig",
    "ClosedLoopPlayReport",
    "control_audit_stats",
    "decode_closed_loop_control",
    "evaluate_closed_loop_play",
    "run_closed_loop_episode",
    "run_policy_closed_loop_episode",
]
