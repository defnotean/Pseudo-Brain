"""Pseudo-Brain research package under the historical ``irene_brain`` namespace.

The Phase 0 package is intentionally dependency-free and CPU-only. Neural
model and native capture dependencies will be introduced only after the
deterministic data and timing contracts are tested.
"""

from .types import (
    ActionEnvelope,
    GenericControl,
    HidKey,
    ModelObservation,
    Observation,
    RgbFrame,
    StepOutcome,
    TextInput,
    TextProvenance,
)

__all__ = [
    "ActionEnvelope",
    "GenericControl",
    "HidKey",
    "ModelObservation",
    "Observation",
    "RgbFrame",
    "StepOutcome",
    "TextInput",
    "TextProvenance",
]

__version__ = "0.0.1"
