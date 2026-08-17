"""Bit-exact in-memory replay traces for deterministic environments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Iterable

from ..environments.protocol import BranchableEnvironment
from ..types import GenericControl, StepOutcome


class ReplayMismatch(AssertionError):
    def __init__(self, step: int, field: str, expected: object, actual: object) -> None:
        self.step = step
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"replay diverged at step {step} field {field}: "
            f"expected {expected!r}, received {actual!r}"
        )


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _require_reward_hex(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("reward_hex must be a canonical finite float encoding")
    try:
        reward = float.fromhex(value)
    except ValueError as error:
        raise ValueError(
            "reward_hex must be a canonical finite float encoding"
        ) from error
    canonical = reward.hex()
    if not isfinite(reward) or canonical != value or (reward == 0.0 and value.startswith("-")):
        raise ValueError("reward_hex must be a canonical finite float encoding")
    return value


@dataclass(frozen=True, slots=True)
class ReplayStep:
    requested_control: GenericControl
    outcome_requested_control_sha256: str
    applied_control_sha256: str
    observation_sha256: str
    reward_hex: str
    events: tuple[str, ...]
    terminated: bool
    truncated: bool
    state_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.requested_control, GenericControl):
            raise ValueError("requested_control must be a GenericControl")
        for name in (
            "outcome_requested_control_sha256",
            "applied_control_sha256",
            "observation_sha256",
            "state_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_reward_hex(self.reward_hex)
        if not isinstance(self.events, tuple) or any(
            not isinstance(event, str) or not event for event in self.events
        ):
            raise ValueError("events must be an immutable tuple of non-empty strings")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminal flags must be booleans")

    @classmethod
    def capture(
        cls,
        environment: BranchableEnvironment,
        requested_control: GenericControl,
        outcome: StepOutcome,
    ) -> ReplayStep:
        if not isinstance(requested_control, GenericControl):
            raise ValueError("requested_control must be a GenericControl")
        if not isinstance(outcome, StepOutcome):
            raise ValueError("outcome must be a StepOutcome")
        return cls(
            requested_control=requested_control,
            outcome_requested_control_sha256=sha256(
                outcome.requested_control.canonical_bytes()
            ).hexdigest(),
            applied_control_sha256=sha256(
                outcome.applied_control.canonical_bytes()
            ).hexdigest(),
            observation_sha256=outcome.observation.content_hash,
            reward_hex=float(outcome.reward).hex(),
            events=outcome.events,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            state_sha256=environment.state_hash(),
        )


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    root_snapshot: bytes
    root_state_sha256: str
    steps: tuple[ReplayStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.root_snapshot, bytes):
            raise ValueError("root_snapshot must be immutable bytes")
        _require_sha256(self.root_state_sha256, name="root_state_sha256")
        if not isinstance(self.steps, tuple) or any(
            not isinstance(step, ReplayStep) for step in self.steps
        ):
            raise ValueError("steps must be an immutable tuple of ReplayStep values")
        for index, step in enumerate(self.steps[:-1]):
            if step.terminated or step.truncated:
                raise ValueError(
                    f"trace contains step {index + 1} after a terminal transition"
                )

    @property
    def trace_sha256(self) -> str:
        # Version 2 includes the environment-reported requested-control digest.
        digest = sha256(b"IRREPLAY\x02")
        digest.update(bytes.fromhex(self.root_state_sha256))
        for step in self.steps:
            digest.update(step.requested_control.canonical_bytes())
            digest.update(bytes.fromhex(step.outcome_requested_control_sha256))
            digest.update(bytes.fromhex(step.applied_control_sha256))
            digest.update(bytes.fromhex(step.observation_sha256))
            reward = step.reward_hex.encode("ascii")
            digest.update(len(reward).to_bytes(2, "big"))
            digest.update(reward)
            digest.update(len(step.events).to_bytes(4, "big"))
            for event in step.events:
                encoded = event.encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "big"))
                digest.update(encoded)
            digest.update(bytes((step.terminated, step.truncated)))
            digest.update(bytes.fromhex(step.state_sha256))
        return digest.hexdigest()


def record_trace(
    environment: BranchableEnvironment,
    controls: Iterable[GenericControl],
) -> ReplayTrace:
    root_snapshot = environment.snapshot()
    root_hash = environment.state_hash()
    captured: list[ReplayStep] = []
    for index, control in enumerate(controls):
        if not isinstance(control, GenericControl):
            raise ValueError("controls must contain only GenericControl values")
        if captured and (captured[-1].terminated or captured[-1].truncated):
            raise ValueError("control sequence continues after a terminal transition")
        outcome = environment.step(control)
        if not isinstance(outcome, StepOutcome):
            raise ValueError("environment.step must return a StepOutcome")
        _equal(index, "requested_control", control, outcome.requested_control)
        captured.append(ReplayStep.capture(environment, control, outcome))
    return ReplayTrace(
        root_snapshot=root_snapshot,
        root_state_sha256=root_hash,
        steps=tuple(captured),
    )


def verify_trace(
    environment: BranchableEnvironment,
    trace: ReplayTrace,
    *,
    preserve_environment: bool = True,
) -> None:
    original = environment.snapshot() if preserve_environment else None
    try:
        environment.restore(trace.root_snapshot)
        _equal(-1, "root_state_sha256", trace.root_state_sha256, environment.state_hash())
        for index, expected in enumerate(trace.steps):
            outcome = environment.step(expected.requested_control)
            if not isinstance(outcome, StepOutcome):
                raise ReplayMismatch(index, "outcome_type", StepOutcome, type(outcome))
            _equal(
                index,
                "requested_control",
                expected.requested_control,
                outcome.requested_control,
            )
            actual = ReplayStep.capture(
                environment,
                expected.requested_control,
                outcome,
            )
            for field in (
                "outcome_requested_control_sha256",
                "applied_control_sha256",
                "observation_sha256",
                "reward_hex",
                "events",
                "terminated",
                "truncated",
                "state_sha256",
            ):
                _equal(index, field, getattr(expected, field), getattr(actual, field))
    finally:
        if original is not None:
            environment.restore(original)


def _equal(step: int, field: str, expected: object, actual: object) -> None:
    if actual != expected:
        raise ReplayMismatch(step, field, expected, actual)
