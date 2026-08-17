"""Contracts and lazily loaded trainable Pseudo-Brain thought-field model."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .spec import (
    ActuatorQuerySpec,
    AttentionContract,
    StateLayout,
    TensorShape,
    ThoughtFieldConfig,
)

__all__ = [
    "ActionPrediction",
    "ActuatorQuerySpec",
    "AttentionContract",
    "BrainState",
    "MonolithicRecurrentBaseline",
    "NoCommunicationSlotBaseline",
    "ParameterMatchedMonolithicBaseline",
    "IreneBrainModel",
    "ModelDiagnostics",
    "ModelOutput",
    "StateLayout",
    "TensorShape",
    "ThoughtPredictions",
    "ThoughtFieldConfig",
]

_TORCH_EXPORTS = {
    "ActionPrediction",
    "BrainState",
    "IreneBrainModel",
    "ModelDiagnostics",
    "ModelOutput",
    "ThoughtPredictions",
}

_BASELINE_EXPORTS = {
    "MonolithicRecurrentBaseline",
    "NoCommunicationSlotBaseline",
    "ParameterMatchedMonolithicBaseline",
}


def __getattr__(name: str) -> Any:
    if name in _TORCH_EXPORTS:
        torch_model = import_module(f"{__name__}.torch_model")
        return getattr(torch_model, name)
    if name in _BASELINE_EXPORTS:
        baselines = import_module(f"{__name__}.baselines")
        return getattr(baselines, name)
    raise AttributeError(name)
