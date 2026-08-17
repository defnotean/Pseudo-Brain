"""Pure learning-rate schedules shared by configuration tests and PyTorch."""

from __future__ import annotations

import math


COSINE_AFTER_WARMUP = "cosine_after_warmup"
CONSTANT_AFTER_WARMUP = "constant_after_warmup"
SCHEDULER_KINDS = frozenset({COSINE_AFTER_WARMUP, CONSTANT_AFTER_WARMUP})


def learning_rate_multiplier(
    *,
    scheduler_kind: str,
    warmup_steps: int,
    max_optimizer_steps: int,
    step: int,
) -> float:
    """Return the multiplier for a zero-based optimizer-update index."""

    if scheduler_kind not in SCHEDULER_KINDS:
        raise ValueError(f"unsupported scheduler kind: {scheduler_kind}")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if scheduler_kind == CONSTANT_AFTER_WARMUP:
        return 1.0
    if max_optimizer_steps <= warmup_steps:
        return 1.0
    progress = min(
        1.0,
        max(0.0, (step - warmup_steps) / (max_optimizer_steps - warmup_steps)),
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


__all__ = [
    "CONSTANT_AFTER_WARMUP",
    "COSINE_AFTER_WARMUP",
    "SCHEDULER_KINDS",
    "learning_rate_multiplier",
]
