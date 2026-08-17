"""Contracts for deterministic, branchable sensorimotor environments."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import GenericControl, Observation, StepOutcome


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Minimal interface shared by accelerated and continuous runtimes.

    A call to :meth:`step` advances exactly one logical environment tick.  The
    model-facing values are limited to :class:`Observation`; snapshots and
    hashes are evaluator/data-generation facilities and must never be routed to
    a policy as observations.
    """

    @property
    def tick_period_ns(self) -> int:
        """Nominal duration of one logical environment tick in nanoseconds."""
        ...

    @property
    def current_observation(self) -> Observation:
        """Return the current model-visible observation without advancing time."""
        ...

    def reset(self, seed: int) -> Observation:
        """Start a deterministic lifetime from ``seed``."""
        ...

    def step(self, control: GenericControl) -> StepOutcome:
        """Apply a generic control state and advance one logical tick."""
        ...

    def snapshot(self) -> bytes:
        """Serialize the complete branchable simulator state."""
        ...

    def restore(self, snapshot: bytes) -> Observation:
        """Restore a byte-exact snapshot and return its visible observation."""
        ...

    def state_hash(self) -> str:
        """Return the lowercase SHA-256 digest of the canonical snapshot."""
        ...


# The descriptive alias is convenient in dataset and evaluator annotations.
BranchableEnvironment = EnvironmentProtocol


__all__ = ["BranchableEnvironment", "EnvironmentProtocol"]
