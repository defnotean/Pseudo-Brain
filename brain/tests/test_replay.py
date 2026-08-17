from __future__ import annotations

from dataclasses import replace
import unittest

from irene_brain.data.branching import evaluate_branches
from irene_brain.data.replay import (
    ReplayMismatch,
    ReplayTrace,
    record_trace,
    verify_trace,
)
from irene_brain.environments import MovingShapesEnv
from irene_brain.types import GenericControl, HidKey, StepOutcome


class WrongRequestedControlEnvironment(MovingShapesEnv):
    def step(self, control: GenericControl) -> StepOutcome:
        outcome = super().step(control)
        return replace(outcome, requested_control=GenericControl.neutral())


class ReplayTests(unittest.TestCase):
    def test_recorded_trace_replays_and_preserves_caller_state(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(45)
        trace = record_trace(
            environment,
            (
                GenericControl(keys_down=(int(HidKey.W),)),
                GenericControl(keys_down=(int(HidKey.D),)),
                GenericControl.neutral(),
            ),
        )
        state_before_verify = environment.state_hash()

        verify_trace(environment, trace)

        self.assertEqual(environment.state_hash(), state_before_verify)
        self.assertEqual(len(trace.trace_sha256), 64)

    def test_replay_reports_exact_divergent_field(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(9)
        trace = record_trace(environment, (GenericControl.neutral(),))
        damaged_step = replace(trace.steps[0], observation_sha256="0" * 64)
        damaged = replace(trace, steps=(damaged_step,))

        with self.assertRaises(ReplayMismatch) as raised:
            verify_trace(environment, damaged)
        self.assertEqual(raised.exception.step, 0)
        self.assertEqual(raised.exception.field, "observation_sha256")

    def test_replay_asserts_environment_requested_control_provenance(self) -> None:
        environment = WrongRequestedControlEnvironment()
        environment.reset(4)

        with self.assertRaises(ReplayMismatch) as raised:
            record_trace(
                environment,
                (GenericControl(keys_down=(int(HidKey.W),)),),
            )

        self.assertEqual(raised.exception.step, 0)
        self.assertEqual(raised.exception.field, "requested_control")

    def test_replay_captures_outcome_requested_control_hash(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(12)
        trace = record_trace(environment, (GenericControl.neutral(),))
        damaged_step = replace(
            trace.steps[0],
            outcome_requested_control_sha256="0" * 64,
        )
        damaged = replace(trace, steps=(damaged_step,))

        with self.assertRaises(ReplayMismatch) as raised:
            verify_trace(environment, damaged)

        self.assertEqual(raised.exception.field, "outcome_requested_control_sha256")

    def test_replay_records_reject_mutable_or_noncanonical_fields(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(15)
        trace = record_trace(environment, (GenericControl.neutral(),))
        step = trace.steps[0]

        malformed_steps = (
            {"applied_control_sha256": "A" * 64},
            {"reward_hex": "nan"},
            {"events": ["mutable"]},
            {"terminated": 1},
        )
        for changes in malformed_steps:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(step, **changes)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            ReplayTrace(
                root_snapshot=bytearray(trace.root_snapshot),  # type: ignore[arg-type]
                root_state_sha256=trace.root_state_sha256,
                steps=trace.steps,
            )
        with self.assertRaises(ValueError):
            ReplayTrace(
                root_snapshot=trace.root_snapshot,
                root_state_sha256=trace.root_state_sha256,
                steps=list(trace.steps),  # type: ignore[arg-type]
            )

    def test_trace_rejects_steps_after_terminal(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(22)
        trace = record_trace(
            environment,
            (GenericControl.neutral(), GenericControl.neutral()),
        )
        terminal = replace(trace.steps[0], truncated=True)

        with self.assertRaisesRegex(ValueError, "after a terminal transition"):
            ReplayTrace(
                root_snapshot=trace.root_snapshot,
                root_state_sha256=trace.root_state_sha256,
                steps=(terminal, trace.steps[1]),
            )

    def test_branch_execution_order_cannot_change_trace_identity(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(77)
        for _ in range(10):
            environment.step(GenericControl.neutral())
        root = environment.snapshot()
        caller_state = environment.state_hash()
        branches = {
            "north": (GenericControl(keys_down=(int(HidKey.W),)),) * 6,
            "east": (GenericControl(keys_down=(int(HidKey.D),)),) * 6,
            "neutral": (GenericControl.neutral(),) * 6,
        }

        first = evaluate_branches(
            environment,
            root,
            branches,
            order=("north", "east", "neutral"),
        )
        second = evaluate_branches(
            environment,
            root,
            branches,
            order=("neutral", "north", "east"),
        )

        self.assertEqual(environment.state_hash(), caller_state)
        self.assertEqual(
            {name: trace.trace_sha256 for name, trace in first.items()},
            {name: trace.trace_sha256 for name, trace in second.items()},
        )


if __name__ == "__main__":
    unittest.main()
