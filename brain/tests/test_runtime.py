from __future__ import annotations

import threading
import unittest

from irene_brain.environments import MovingShapesEnv
from irene_brain.runtime.clock import (
    ManualClock,
    nanoseconds_to_ticks,
    ticks_to_nanoseconds,
)
from irene_brain.runtime.continuous import (
    CatchUpLimitExceeded,
    ContinuousDriver,
    PostAdvanceRuntimeError,
    PostAdvanceValueError,
)
from irene_brain.types import (
    ActionEnvelope,
    GenericControl,
    HidKey,
    Observation,
    RgbFrame,
    StepOutcome,
)


class RecordingEnvironment:
    tick_period_ns = 10

    def __init__(self) -> None:
        self.controls: list[GenericControl] = []
        self._frame_id = 0

    @property
    def current_observation(self) -> Observation:
        previous = self.controls[-1] if self.controls else GenericControl.neutral()
        return Observation(
            frame_id=self._frame_id,
            capture_tick=self._frame_id,
            elapsed_ns=self._frame_id * self.tick_period_ns,
            rgb=RgbFrame(width=1, height=1, pixels=b"\x00\x00\x00"),
            previous_control=previous,
        )

    def step(self, control: GenericControl) -> StepOutcome:
        self.controls.append(control)
        self._frame_id += 1
        return StepOutcome(
            observation=self.current_observation,
            requested_control=control,
            applied_control=control,
            reward=0.0,
        )


class InvalidOutcomeEnvironment(RecordingEnvironment):
    def step(self, control: GenericControl) -> object:
        self.controls.append(control)
        self._frame_id += 1
        return object()


class RewardingEnvironment(RecordingEnvironment):
    def step(self, control: GenericControl) -> StepOutcome:
        outcome = super().step(control)
        return StepOutcome(
            observation=outcome.observation,
            requested_control=outcome.requested_control,
            applied_control=outcome.applied_control,
            reward=1.0,
        )


class InvalidContractEnvironment(RecordingEnvironment):
    def __init__(self, violation: str) -> None:
        super().__init__()
        self.violation = violation

    def step(self, control: GenericControl) -> StepOutcome:
        self.controls.append(control)
        previous_frame = self._frame_id
        self._frame_id += 1
        applied = control
        requested = control
        previous_control = applied
        frame_id = self._frame_id
        capture_tick = self._frame_id
        elapsed_ns = self._frame_id * self.tick_period_ns
        if self.violation == "requested":
            requested = GenericControl.neutral()
        elif self.violation == "previous_control":
            previous_control = GenericControl.neutral()
        elif self.violation == "counters":
            frame_id = previous_frame
            capture_tick = previous_frame
            elapsed_ns = previous_frame * self.tick_period_ns
        observation = Observation(
            frame_id=frame_id,
            capture_tick=capture_tick,
            elapsed_ns=elapsed_ns,
            rgb=RgbFrame(width=1, height=1, pixels=b"\x00\x00\x00"),
            previous_control=previous_control,
        )
        return StepOutcome(
            observation=observation,
            requested_control=requested,
            applied_control=applied,
            reward=0.0,
        )


class ClockTests(unittest.TestCase):
    def test_integer_tick_conversions_do_not_use_float(self) -> None:
        self.assertEqual(ticks_to_nanoseconds(30, 3), 10_000_000_000)
        self.assertEqual(nanoseconds_to_ticks(10_000_000_000, 3), 30)

    def test_manual_clock_retains_fractional_ticks(self) -> None:
        clock = ManualClock(frequency_hz=3)
        for _ in range(10):
            clock.advance_ns(100_000_000)
        self.assertEqual(clock.now_ticks(), 3)


class ContinuousDriverTests(unittest.TestCase):
    def test_catch_up_exposes_only_newest_observation(self) -> None:
        period = MovingShapesEnv.DEFAULT_TICK_PERIOD_NS
        environment = MovingShapesEnv(tick_period_ns=period)
        clock = ManualClock()
        driver = ContinuousDriver(environment, clock, start_tick=0)

        clock.advance_ns(period * 5)
        result = driver.advance_to_now()

        self.assertEqual(result.steps_advanced, 5)
        self.assertEqual(result.observations_dropped, 4)
        self.assertEqual(result.latest_observation.frame_id, 5)
        self.assertEqual(driver.total_dropped_observations, 4)

    def test_new_control_is_not_applied_retroactively(self) -> None:
        period = MovingShapesEnv.DEFAULT_TICK_PERIOD_NS
        environment = MovingShapesEnv(tick_period_ns=period)
        environment.reset(3)
        clock = ManualClock()
        driver = ContinuousDriver(environment, clock, start_tick=0)
        forward = GenericControl(keys_down=(int(HidKey.W),))

        clock.advance_ns(period)
        submitted = driver.submit_control(forward)
        self.assertEqual(submitted.latest_observation.previous_control, GenericControl.neutral())

        clock.advance_ns(period)
        applied = driver.advance_to_now()
        self.assertEqual(applied.latest_observation.previous_control, forward)

    def test_relative_impulse_is_applied_exactly_once(self) -> None:
        environment = RecordingEnvironment()
        clock = ManualClock()
        driver = ContinuousDriver(environment, clock, start_tick=0)
        impulse = GenericControl(mouse_dx=3.0, impulse_sequence=1)
        driver.submit_control(impulse, submission_tick=0)

        clock.advance_ns(30)
        driver.advance_to_now()

        self.assertEqual(environment.controls[0].mouse_dx, 3.0)
        self.assertEqual(environment.controls[1].mouse_dx, 0.0)
        self.assertEqual(environment.controls[2].mouse_dx, 0.0)
        with self.assertRaises(ValueError):
            driver.submit_control(impulse, submission_tick=clock.now_ticks())

    def test_pending_impulses_coalesce_with_newest_persistent_state(self) -> None:
        environment = RecordingEnvironment()
        driver = ContinuousDriver(environment, ManualClock(), start_tick=0)
        first = GenericControl(
            keys_down=(int(HidKey.W),),
            mouse_dx=2.0,
            mouse_dy=1.0,
            impulse_sequence=1,
        )
        second = GenericControl(
            keys_down=(int(HidKey.D),),
            mouse_dx=3.0,
            mouse_dy=-2.0,
            impulse_sequence=2,
        )
        newest_state = GenericControl(keys_down=(int(HidKey.S),))

        driver.submit_control(first, submission_tick=0)
        driver.submit_control(second, submission_tick=0)
        driver.submit_control(newest_state, submission_tick=0)
        driver.advance_to(20)

        self.assertEqual(environment.controls[0].keys_down, newest_state.keys_down)
        self.assertEqual(environment.controls[0].mouse_dx, 5.0)
        self.assertEqual(environment.controls[0].mouse_dy, -1.0)
        self.assertEqual(environment.controls[0].impulse_sequence, 2)
        self.assertEqual(environment.controls[1].keys_down, newest_state.keys_down)
        self.assertFalse(environment.controls[1].has_relative_impulse)
        with self.assertRaises(ValueError):
            driver.submit_control(second, submission_tick=20)

    def test_rejected_impulse_submission_still_catches_up_time(self) -> None:
        environment = RewardingEnvironment()
        driver = ContinuousDriver(environment, ManualClock(), start_tick=0)
        impulse = GenericControl(mouse_dx=3.0, impulse_sequence=1)
        driver.submit_control(impulse, submission_tick=0)

        with self.assertRaises(PostAdvanceValueError) as raised:
            driver.submit_control(impulse, submission_tick=20)

        self.assertEqual(raised.exception.advance_result.steps_advanced, 2)
        self.assertEqual(raised.exception.advance_result.observations_dropped, 1)
        self.assertEqual(raised.exception.advance_result.reward_sum, 2.0)
        self.assertEqual(driver.step_count, 2)
        self.assertEqual(environment.controls[0], impulse)
        self.assertEqual(environment.controls[1].mouse_dx, 0.0)

    def test_invalid_environment_outcome_neutralizes_watchdog_state(self) -> None:
        environment = InvalidOutcomeEnvironment()
        driver = ContinuousDriver(environment, ManualClock(), start_tick=0)
        forward = GenericControl(keys_down=(int(HidKey.W),))
        driver.submit_control(forward, submission_tick=0)

        with self.assertRaises(TypeError):
            driver.advance_to(10)

        self.assertEqual(driver.held_control, GenericControl.neutral())
        self.assertTrue(driver.halted)

    def test_environment_contract_violations_halt_and_neutralize(self) -> None:
        forward = GenericControl(keys_down=(int(HidKey.W),))
        for violation in ("requested", "previous_control", "counters"):
            with self.subTest(violation=violation):
                environment = InvalidContractEnvironment(violation)
                driver = ContinuousDriver(environment, ManualClock(), start_tick=0)
                driver.submit_control(forward, submission_tick=0)
                with self.assertRaises(RuntimeError):
                    driver.advance_to(10)
                self.assertTrue(driver.halted)
                self.assertEqual(driver.held_control, GenericControl.neutral())

    def test_clock_jump_halts_before_default_limit_does_work(self) -> None:
        environment = RecordingEnvironment()
        driver = ContinuousDriver(environment, ManualClock(), start_tick=0)

        with self.assertRaises(CatchUpLimitExceeded) as raised:
            driver.advance_to(90)

        self.assertEqual(raised.exception.due_steps, 9)
        self.assertEqual(raised.exception.advance_result.steps_advanced, 0)
        self.assertEqual(environment.controls, [])
        self.assertTrue(driver.halted)
        self.assertEqual(driver.held_control, GenericControl.neutral())

    def test_offline_catch_up_can_be_explicitly_unlimited(self) -> None:
        environment = RecordingEnvironment()
        driver = ContinuousDriver(
            environment,
            ManualClock(),
            start_tick=0,
            max_catch_up_steps=None,
        )

        result = driver.advance_to(100)

        self.assertEqual(result.steps_advanced, 10)
        self.assertFalse(driver.halted)

    def test_driver_creates_no_background_thread(self) -> None:
        before = tuple(thread.ident for thread in threading.enumerate())
        driver = ContinuousDriver(RecordingEnvironment(), ManualClock(), start_tick=0)
        after = tuple(thread.ident for thread in threading.enumerate())

        self.assertFalse(driver.uses_background_threads)
        self.assertEqual(after, before)

    def test_action_is_accepted_at_exact_deadline_and_expires(self) -> None:
        environment = RecordingEnvironment()
        clock = ManualClock()
        driver = ContinuousDriver(environment, clock, start_tick=0)
        forward = GenericControl(
            keys_down=(int(HidKey.W),),
            intended_hold_ns=30,
        )
        envelope = ActionEnvelope(
            action_sequence=1,
            source_frame_id=0,
            model_state_version=1,
            created_qpc=0,
            ready_qpc=0,
            submit_deadline_qpc=5,
            expires_qpc=10,
            control=forward,
        )

        driver.submit_action(envelope, submission_tick=5)
        driver.advance_to(20)

        self.assertEqual(environment.controls[0], forward)
        self.assertEqual(environment.controls[1], GenericControl.neutral())

    def test_initial_and_submitted_control_holds_are_enforced(self) -> None:
        held = GenericControl(
            keys_down=(int(HidKey.W),),
            intended_hold_ns=10,
        )
        initial_environment = RecordingEnvironment()
        initial_driver = ContinuousDriver(
            initial_environment,
            ManualClock(),
            start_tick=0,
            initial_control=held,
        )
        initial_driver.advance_to(20)
        self.assertEqual(initial_environment.controls, [held, GenericControl.neutral()])

        submitted_environment = RecordingEnvironment()
        submitted_driver = ContinuousDriver(
            submitted_environment,
            ManualClock(),
            start_tick=0,
        )
        submitted_driver.submit_control(held, submission_tick=0)
        submitted_driver.advance_to(20)
        self.assertEqual(
            submitted_environment.controls,
            [held, GenericControl.neutral()],
        )

    def test_controls_that_cannot_reach_a_tick_are_rejected_with_advance_result(self) -> None:
        short = GenericControl(
            keys_down=(int(HidKey.W),),
            intended_hold_ns=5,
        )
        with self.assertRaisesRegex(ValueError, "next environment tick"):
            ContinuousDriver(
                RecordingEnvironment(),
                ManualClock(),
                start_tick=0,
                initial_control=short,
            )

        environment = RecordingEnvironment()
        driver = ContinuousDriver(environment, ManualClock(), start_tick=0)
        with self.assertRaises(PostAdvanceValueError) as raised:
            driver.submit_control(short, submission_tick=0)
        self.assertEqual(raised.exception.advance_result.target_tick, 0)
        self.assertEqual(driver.held_control, GenericControl.neutral())

        envelope = ActionEnvelope(
            action_sequence=1,
            source_frame_id=0,
            model_state_version=1,
            created_qpc=0,
            ready_qpc=0,
            submit_deadline_qpc=0,
            expires_qpc=20,
            control=short,
        )
        with self.assertRaises(PostAdvanceValueError) as action_raised:
            driver.submit_action(envelope, submission_tick=0)
        self.assertEqual(action_raised.exception.advance_result.steps_advanced, 0)

    def test_halted_submission_retains_terminal_advance_result(self) -> None:
        environment = MovingShapesEnv(tick_period_ns=10, max_ticks=1)
        driver = ContinuousDriver(environment, ManualClock(), start_tick=0)

        terminal = driver.advance_to(10)
        self.assertTrue(terminal.truncated)
        with self.assertRaises(PostAdvanceRuntimeError) as raised:
            driver.submit_control(GenericControl.neutral(), submission_tick=10)

        self.assertTrue(raised.exception.advance_result.truncated)
        self.assertEqual(raised.exception.advance_result.latest_observation.frame_id, 1)

    def test_late_duplicate_and_stale_frame_actions_are_rejected(self) -> None:
        environment = RecordingEnvironment()
        clock = ManualClock()
        driver = ContinuousDriver(environment, clock, start_tick=0)
        first = ActionEnvelope(
            action_sequence=1,
            source_frame_id=0,
            model_state_version=1,
            created_qpc=0,
            ready_qpc=0,
            submit_deadline_qpc=5,
            expires_qpc=20,
            control=GenericControl.neutral(),
        )
        driver.submit_action(first, submission_tick=5)
        with self.assertRaisesRegex(ValueError, "duplicated or stale") as duplicate:
            driver.submit_action(first, submission_tick=5)
        self.assertEqual(duplicate.exception.advance_result.target_tick, 5)

        environment_two = RecordingEnvironment()
        driver_two = ContinuousDriver(environment_two, ManualClock(), start_tick=0)
        late = ActionEnvelope(
            action_sequence=1,
            source_frame_id=0,
            model_state_version=1,
            created_qpc=0,
            ready_qpc=0,
            submit_deadline_qpc=5,
            expires_qpc=20,
            control=GenericControl.neutral(),
        )
        with self.assertRaisesRegex(ValueError, "missed"):
            driver_two.submit_action(late, submission_tick=6)

        stale = ActionEnvelope(
            action_sequence=2,
            source_frame_id=0,
            model_state_version=2,
            created_qpc=6,
            ready_qpc=6,
            submit_deadline_qpc=10,
            expires_qpc=20,
            control=GenericControl.neutral(),
        )
        with self.assertRaisesRegex(ValueError, "newest observation"):
            driver_two.submit_action(stale, submission_tick=10)

    def test_watchdog_forces_neutral_state(self) -> None:
        environment = RecordingEnvironment()
        driver = ContinuousDriver(environment, ManualClock(), start_tick=0)
        driver.submit_control(
            GenericControl(
                keys_down=(int(HidKey.W),),
                mouse_dx=3.0,
                impulse_sequence=1,
            ),
            submission_tick=0,
        )
        driver.watchdog_neutralize()
        driver.advance_to(10)

        self.assertEqual(driver.held_control, GenericControl.neutral())
        self.assertEqual(environment.controls, [GenericControl.neutral()])


if __name__ == "__main__":
    unittest.main()
