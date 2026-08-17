"""Order-independent counterfactual rollouts from exact snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..environments.protocol import BranchableEnvironment
from ..types import GenericControl
from .replay import ReplayTrace, record_trace


def evaluate_branches(
    environment: BranchableEnvironment,
    root_snapshot: bytes,
    branches: Mapping[str, Sequence[GenericControl]],
    *,
    order: Sequence[str] | None = None,
) -> dict[str, ReplayTrace]:
    """Evaluate named futures and restore the caller's original state.

    Every branch begins at the identical validated root. Branch results are
    returned in the requested execution order, but each trace identity must be
    independent of that order.
    """

    if not branches:
        raise ValueError("at least one branch is required")
    if any(not isinstance(name, str) or not name for name in branches):
        raise ValueError("branch names must be non-empty strings")
    names = tuple(branches) if order is None else tuple(order)
    if len(names) != len(set(names)):
        raise ValueError("branch execution order cannot contain duplicates")
    if set(names) != set(branches):
        raise ValueError("branch execution order must contain every branch exactly once")

    original_snapshot = environment.snapshot()
    traces: dict[str, ReplayTrace] = {}
    try:
        environment.restore(root_snapshot)
        root_hash = environment.state_hash()
        for name in names:
            environment.restore(root_snapshot)
            if environment.state_hash() != root_hash:
                raise RuntimeError("restored branch root hash changed")
            trace = record_trace(environment, branches[name])
            if trace.root_state_sha256 != root_hash:
                raise RuntimeError("branch trace did not begin at the requested root")
            traces[name] = trace
    finally:
        environment.restore(original_snapshot)
    return traces
