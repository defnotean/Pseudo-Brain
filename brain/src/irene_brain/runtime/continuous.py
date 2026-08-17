"""Synchronous driver that preserves continuous environment time.

This module deliberately creates no thread, task, timer, queue, or background
loop. A caller advances the driver to an injected clock reading. If several
environment ticks elapsed, every tick is simulated with the held control but
only the newest observation is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Protocol, runtime_checkable

from ..types import ActionEnvelope, GenericControl, Observation, StepOutcome
from .clock import ClockProtocol, NANOSECONDS_PER_SECOND


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Narrow structural contract needed for continuous stepping."""

    @property
    def tick_period_ns(self) -> int:
        """Duration of one deterministic logical environment tick."""
        ...

    @property
    def current_observation(self) -> Observation:
        """Newest observation produced by the environment."""
        ...

    def step(self, control: GenericControl) -> StepOutcome:
        """Advance exactly one logical tick using a generic device state."""
        ...


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    """Summary of one synchronous catch-up operation.

    Intermediate observations are intentionally absent. ``latest_observation``
    is the only observation made available for the entire operation.
    """

    target_tick: int
    steps_advanced: int
    observations_dropped: int
    latest_observation: Observation
    reward_sum: float
    terminated: bool
    truncated: bool


class PostAdvanceValueError(ValueError):
    """A rejected submission that retains work completed before validation."""

    def __init__(self, message: str, advance_result: AdvanceResult) -> None:
        self.advance_result = advance_result
        super().__init__(message)


class PostAdvanceRuntimeError(RuntimeError):
    """A runtime rejection that retains work completed before validation."""

    def __init__(self, message: str, advance_result: AdvanceResult) -> None:
        self.advance_result = advance_result
        super().__init__(message)


class CatchUpLimitExceeded(RuntimeError):
    """Raised after a clock jump is rejected before any unbounded catch-up."""

    def __init__(
        self,
        *,
        due_steps: int,
        max_catch_up_steps: int,
        advance_result: AdvanceResult,
    ) -> None:
        self.due_steps = due_steps
        self.max_catch_up_steps = max_catch_up_steps
        self.advance_result = advance_result
        super().__init__(
            f"{due_steps} environment steps are due, exceeding the "
            f"catch-up limit of {max_catch_up_steps}"
        )


class ContinuousDriver:
    """Advance an environment according to real or manually injected time.

    A submission first advances the world through its timestamp with the
    previously held control, then installs the new control for future ticks.
    Relative impulses accepted between ticks are accumulated rather than
    overwritten; persistent key/button/axis state always comes from the newest
    accepted control.
    """

    DEFAULT_MAX_CATCH_UP_STEPS = 8

    __slots__ = (
        "_clock",
        "_environment",
        "_halted",
        "_held_control",
        "_held_control_expiry_tick",
        "_last_action_sequence",
        "_last_impulse_sequence",
        "_last_model_state_version",
        "_last_source_frame_id",
        "_last_target_tick",
        "_latest_observation",
        "_max_catch_up_steps",
        "_origin_tick",
        "_step_count",
        "_terminated",
        "_tick_period_ns",
        "_total_dropped_observations",
        "_truncated",
    )

    def __init__(
        self,
        environment: EnvironmentProtocol,
        clock: ClockProtocol,
        *,
        start_tick: int | None = None,
        initial_control: GenericControl | None = None,
        max_catch_up_steps: int | None = DEFAULT_MAX_CATCH_UP_STEPS,
    ) -> None:
        frequency_hz = clock.frequency_hz
        if isinstance(frequency_hz, bool) or not isinstance(frequency_hz, int):
            raise TypeError("clock.frequency_hz must be an integer")
        if frequency_hz <= 0:
            raise ValueError("clock.frequency_hz must be positive")

        tick_period_ns = environment.tick_period_ns
        if isinstance(tick_period_ns, bool) or not isinstance(tick_period_ns, int):
            raise TypeError("environment.tick_period_ns must be an integer")
        if tick_period_ns <= 0:
            raise ValueError("environment.tick_period_ns must be positive")

        if start_tick is None:
            start_tick = clock.now_ticks()
        if isinstance(start_tick, bool) or not isinstance(start_tick, int):
            raise TypeError("start_tick must be an integer")
        if start_tick < 0:
            raise ValueError("start_tick cannot be negative")

        if initial_control is None:
            initial_control = GenericControl.neutral()
        if not isinstance(initial_control, GenericControl):
            raise TypeError("initial_control must be a GenericControl")

        if max_catch_up_steps is not None:
            if isinstance(max_catch_up_steps, bool) or not isinstance(
                max_catch_up_steps, int
            ):
                raise TypeError("max_catch_up_steps must be an integer or None")
            if max_catch_up_steps <= 0:
                raise ValueError("max_catch_up_steps must be positive")

        latest_observation = environment.current_observation
        if not isinstance(latest_observation, Observation):
            raise TypeError("environment.current_observation must be an Observation")

        self._environment = environment
        self._clock = clock
        self._tick_period_ns = tick_period_ns
        self._origin_tick = start_tick
        self._last_target_tick = start_tick
        self._step_count = 0
        self._held_control = initial_control
        self._held_control_expiry_tick: int | None = None
        self._last_action_sequence = -1
        self._last_impulse_sequence = (
            initial_control.impulse_sequence
            if initial_control.has_relative_impulse
            else 0
        )
        self._last_model_state_version = -1
        self._last_source_frame_id = -1
        self._latest_observation = latest_observation
        self._max_catch_up_steps = max_catch_up_steps
        self._total_dropped_observations = 0
        self._halted = False
        self._terminated = False
        self._truncated = False

        initial_expiry = self._expiry_tick_for_control(
            initial_control,
            accepted_tick=start_tick,
        )
        if not self._can_reach_next_tick(initial_expiry):
            raise ValueError("initial_control expires before the next environment tick")
        self._held_control_expiry_tick = initial_expiry

    @property
    def clock(self) -> ClockProtocol:
        return self._clock

    @property
    def environment(self) -> EnvironmentProtocol:
        return self._environment

    @property
    def held_control(self) -> GenericControl:
        return self._held_control

    @property
    def latest_observation(self) -> Observation:
        """Return the last produced observation without reading or waiting."""

        return self._latest_observation

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def total_dropped_observations(self) -> int:
        return self._total_dropped_observations

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def max_catch_up_steps(self) -> int | None:
        return self._max_catch_up_steps

    @property
    def uses_background_threads(self) -> bool:
        """Make the Phase 0 execution guarantee directly inspectable."""

        return False

    def _target_step_count(self, target_tick: int) -> int:
        elapsed_ticks = target_tick - self._origin_tick
        frequency_hz = self._clock.frequency_hz
        # Compare the two clock domains as an exact rational. This avoids
        # cumulative drift for rates such as 60 Hz.
        return (
            elapsed_ticks * NANOSECONDS_PER_SECOND
            // (self._tick_period_ns * frequency_hz)
        )

    def _scheduled_tick_for_step(self, one_based_step: int) -> int:
        numerator = one_based_step * self._tick_period_ns * self._clock.frequency_hz
        delta = (numerator + NANOSECONDS_PER_SECOND - 1) // NANOSECONDS_PER_SECOND
        return self._origin_tick + delta

    def _duration_ticks(self, duration_ns: int) -> int:
        numerator = duration_ns * self._clock.frequency_hz
        return (numerator + NANOSECONDS_PER_SECOND - 1) // NANOSECONDS_PER_SECOND

    def _expiry_tick_for_control(
        self,
        control: GenericControl,
        *,
        accepted_tick: int,
        hard_expiry_tick: int | None = None,
    ) -> int | None:
        """Combine an external deadline with the control's intended hold."""

        expiry_tick = hard_expiry_tick
        if control.intended_hold_ns > 0:
            hold_expiry = accepted_tick + self._duration_ticks(control.intended_hold_ns)
            expiry_tick = (
                hold_expiry
                if expiry_tick is None
                else min(expiry_tick, hold_expiry)
            )
        return expiry_tick

    def _can_reach_next_tick(self, expiry_tick: int | None) -> bool:
        return (
            expiry_tick is None
            or self._scheduled_tick_for_step(self._step_count + 1) <= expiry_tick
        )

    def _coalesce_control(self, newest: GenericControl) -> GenericControl:
        """Keep pending relative motion while replacing persistent state."""

        pending = self._held_control
        has_pending = pending.has_relative_impulse
        return GenericControl(
            keys_down=newest.keys_down,
            mouse_dx=fsum((pending.mouse_dx if has_pending else 0.0, newest.mouse_dx)),
            mouse_dy=fsum((pending.mouse_dy if has_pending else 0.0, newest.mouse_dy)),
            mouse_buttons=newest.mouse_buttons,
            mouse_wheel=fsum(
                (pending.mouse_wheel if has_pending else 0.0, newest.mouse_wheel)
            ),
            gamepad_axes=newest.gamepad_axes,
            gamepad_buttons=newest.gamepad_buttons,
            impulse_sequence=(
                newest.impulse_sequence
                if newest.has_relative_impulse
                else pending.impulse_sequence if has_pending else newest.impulse_sequence
            ),
            intended_hold_ns=newest.intended_hold_ns,
        )

    def _neutralize(self) -> None:
        self._held_control = GenericControl.neutral()
        self._held_control_expiry_tick = None

    def _halt_on_fault(self) -> None:
        self._halted = True
        self._truncated = True
        self._neutralize()

    @staticmethod
    def _validate_outcome(
        outcome: StepOutcome,
        *,
        requested_control: GenericControl,
        previous_observation: Observation,
    ) -> None:
        if outcome.requested_control != requested_control:
            raise RuntimeError(
                "environment outcome requested_control does not match submitted control"
            )
        if outcome.observation.previous_control != outcome.applied_control:
            raise RuntimeError(
                "environment observation previous_control does not match applied_control"
            )
        previous_counters = (
            previous_observation.frame_id,
            previous_observation.capture_tick,
            previous_observation.elapsed_ns,
        )
        current_counters = (
            outcome.observation.frame_id,
            outcome.observation.capture_tick,
            outcome.observation.elapsed_ns,
        )
        if any(
            current <= previous
            for current, previous in zip(current_counters, previous_counters)
        ):
            raise RuntimeError(
                "environment observation frame_id, capture_tick, and elapsed_ns "
                "must all strictly increase"
            )

    def _empty_result(self, target_tick: int) -> AdvanceResult:
        return AdvanceResult(
            target_tick=target_tick,
            steps_advanced=0,
            observations_dropped=0,
            latest_observation=self._latest_observation,
            reward_sum=0.0,
            terminated=self._terminated,
            truncated=self._truncated,
        )

    def advance_to(self, target_tick: int) -> AdvanceResult:
        """Synchronously advance all environment ticks due at ``target_tick``."""

        if isinstance(target_tick, bool) or not isinstance(target_tick, int):
            raise TypeError("target_tick must be an integer")
        if target_tick < self._last_target_tick:
            raise ValueError("continuous time cannot move backwards")

        self._last_target_tick = target_tick
        target_step_count = self._target_step_count(target_tick)
        due_steps = max(0, target_step_count - self._step_count)
        advanced = 0
        reward_sum = 0.0

        if (
            not self._halted
            and self._max_catch_up_steps is not None
            and due_steps > self._max_catch_up_steps
        ):
            self._halt_on_fault()
            result = self._empty_result(target_tick)
            raise CatchUpLimitExceeded(
                due_steps=due_steps,
                max_catch_up_steps=self._max_catch_up_steps,
                advance_result=result,
            )

        if not self._halted:
            for _ in range(due_steps):
                scheduled_tick = self._scheduled_tick_for_step(self._step_count + 1)
                if (
                    self._held_control_expiry_tick is not None
                    and scheduled_tick > self._held_control_expiry_tick
                ):
                    self._neutralize()
                requested_control = self._held_control
                previous_observation = self._latest_observation
                try:
                    outcome = self._environment.step(requested_control)
                    if not isinstance(outcome, StepOutcome):
                        raise TypeError("environment.step must return a StepOutcome")
                    self._validate_outcome(
                        outcome,
                        requested_control=requested_control,
                        previous_observation=previous_observation,
                    )
                    self._held_control = requested_control.without_relative_impulse()
                except BaseException:
                    self._halt_on_fault()
                    raise
                self._latest_observation = outcome.observation
                self._step_count += 1
                advanced += 1
                reward_sum += outcome.reward
                if outcome.terminated or outcome.truncated:
                    self._halted = True
                    self._terminated = outcome.terminated
                    self._truncated = outcome.truncated
                    self._neutralize()
                    break

        dropped = max(0, advanced - 1)
        self._total_dropped_observations += dropped
        return AdvanceResult(
            target_tick=target_tick,
            steps_advanced=advanced,
            observations_dropped=dropped,
            latest_observation=self._latest_observation,
            reward_sum=reward_sum,
            terminated=self._terminated,
            truncated=self._truncated,
        )

    def advance_to_now(self) -> AdvanceResult:
        """Advance to one reading from the injected clock."""

        return self.advance_to(self._clock.now_ticks())

    def poll_latest_observation(self, target_tick: int | None = None) -> Observation:
        """Advance if needed and expose only the freshest observation."""

        if target_tick is None:
            result = self.advance_to_now()
        else:
            result = self.advance_to(target_tick)
        return result.latest_observation

    def submit_control(
        self,
        control: GenericControl,
        *,
        submission_tick: int | None = None,
    ) -> AdvanceResult:
        """Install the latest control after catching up to its submission time."""

        if not isinstance(control, GenericControl):
            raise TypeError("control must be a GenericControl")
        if submission_tick is None:
            submission_tick = self._clock.now_ticks()
        result = self.advance_to(submission_tick)
        if self._halted:
            self._neutralize()
            raise PostAdvanceRuntimeError(
                "cannot submit control after the environment halted",
                result,
            )
        if (
            control.has_relative_impulse
            and control.impulse_sequence <= self._last_impulse_sequence
        ):
            raise PostAdvanceValueError(
                "relative control impulse is duplicated or stale",
                result,
            )

        expiry_tick = self._expiry_tick_for_control(
            control,
            accepted_tick=submission_tick,
        )
        if not self._can_reach_next_tick(expiry_tick):
            raise PostAdvanceValueError(
                "control expires before the next environment tick",
                result,
            )
        try:
            coalesced = self._coalesce_control(control)
        except (OverflowError, ValueError) as error:
            raise PostAdvanceValueError(str(error), result) from error

        self._held_control = coalesced
        self._held_control_expiry_tick = expiry_tick
        if control.has_relative_impulse:
            self._last_impulse_sequence = control.impulse_sequence
        return result

    def submit_action(
        self,
        envelope: ActionEnvelope,
        *,
        submission_tick: int | None = None,
    ) -> AdvanceResult:
        """Submit a deadline-bound action and reject stale model results.

        Time is advanced before validation so a rejected late action can never
        pause or rewind the environment. Every rejection after that advancement
        carries its :class:`AdvanceResult` on the exception.
        """

        if not isinstance(envelope, ActionEnvelope):
            raise TypeError("envelope must be an ActionEnvelope")
        actual_tick = (
            self._clock.now_ticks() if submission_tick is None else submission_tick
        )
        result = self.advance_to(actual_tick)
        if self._halted:
            self._neutralize()
            raise PostAdvanceRuntimeError(
                "cannot submit action after the environment halted",
                result,
            )
        if actual_tick < envelope.ready_qpc:
            raise PostAdvanceValueError(
                "action cannot be submitted before it is ready",
                result,
            )
        if actual_tick > envelope.submit_deadline_qpc:
            raise PostAdvanceValueError("action missed its submit deadline", result)
        if actual_tick > envelope.expires_qpc:
            raise PostAdvanceValueError("action has expired", result)
        if envelope.action_sequence <= self._last_action_sequence:
            raise PostAdvanceValueError(
                "action sequence is duplicated or stale",
                result,
            )
        if envelope.model_state_version < self._last_model_state_version:
            raise PostAdvanceValueError("model state version is stale", result)
        if envelope.source_frame_id != self._latest_observation.frame_id:
            raise PostAdvanceValueError(
                "action does not target the newest observation",
                result,
            )
        if (
            envelope.control.has_relative_impulse
            and envelope.control.impulse_sequence <= self._last_impulse_sequence
        ):
            raise PostAdvanceValueError(
                "relative control impulse is duplicated or stale",
                result,
            )

        expiry_tick = self._expiry_tick_for_control(
            envelope.control,
            accepted_tick=actual_tick,
            hard_expiry_tick=envelope.expires_qpc,
        )
        if not self._can_reach_next_tick(expiry_tick):
            raise PostAdvanceValueError(
                "action expires before the next environment tick",
                result,
            )
        try:
            coalesced = self._coalesce_control(envelope.control)
        except (OverflowError, ValueError) as error:
            raise PostAdvanceValueError(str(error), result) from error

        self._held_control = coalesced
        self._held_control_expiry_tick = expiry_tick
        self._last_action_sequence = envelope.action_sequence
        self._last_model_state_version = envelope.model_state_version
        self._last_source_frame_id = envelope.source_frame_id
        if envelope.control.has_relative_impulse:
            self._last_impulse_sequence = envelope.control.impulse_sequence
        return result

    def watchdog_neutralize(self) -> None:
        """Synchronously return the simulated actuator state to neutral."""

        self._neutralize()


# This more explicit name reads well in runtime configuration and tests.
ContinuousEnvironmentDriver = ContinuousDriver


__all__ = [
    "AdvanceResult",
    "CatchUpLimitExceeded",
    "ContinuousDriver",
    "ContinuousEnvironmentDriver",
    "EnvironmentProtocol",
    "PostAdvanceRuntimeError",
    "PostAdvanceValueError",
]
