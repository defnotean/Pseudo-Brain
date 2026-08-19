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
    "DEFAULT_MODIFIER_KEYS",
    "DEFAULT_MOVEMENT_KEYS",
    "DEFAULT_TRIGGER_KEYS",
    "DEFAULT_TRIGGER_MOUSE_BUTTONS",
    "DirectionalAction",
    "DirectionalActionHead",
    "DirectionalActionPrediction",
    "IntentEncoder",
    "IntentKind",
    "LatentHazardHead",
    "LatentIntentHead",
    "LatentIntentPrediction",
    "LatentIntentPredictor",
    "LatentLookaheadPlanner",
    "LatentRewardHead",
    "LatentSensoryTransition",
    "LookaheadBranchResult",
    "LookaheadPlanResult",
    "LookaheadRolloutStep",
    "MonolithicRecurrentBaseline",
    "MovementDirection5",
    "MovementDirection9",
    "MovementMode",
    "NoCommunicationSlotBaseline",
    "ParameterMatchedMonolithicBaseline",
    "IreneBrainModel",
    "ModelDiagnostics",
    "ModelOutput",
    "StateLayout",
    "StructuredActionLoss",
    "StructuredActionPrediction",
    "StructuredActionSpec",
    "StructuredActuatorHead",
    "StructuredLossOutput",
    "TensorShape",
    "ThoughtPredictions",
    "ThoughtFieldConfig",
    "directional_to_control_vector",
    "generate_directional_candidate_sequences",
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

_INTENT_EXPORTS = {
    "IntentEncoder",
    "IntentKind",
    "LatentIntentHead",
    "LatentIntentPrediction",
    "LatentIntentPredictor",
}

_ACTUATOR_EXPORTS = {
    "DirectionalAction",
    "DirectionalActionHead",
    "DirectionalActionPrediction",
}

_LOOKAHEAD_EXPORTS = {
    "LatentHazardHead",
    "LatentLookaheadPlanner",
    "LatentRewardHead",
    "LatentSensoryTransition",
    "LookaheadBranchResult",
    "LookaheadPlanResult",
    "LookaheadRolloutStep",
    "directional_to_control_vector",
    "generate_directional_candidate_sequences",
}

_ACTION_GROUP_EXPORTS = {
    "DEFAULT_MODIFIER_KEYS",
    "DEFAULT_MOVEMENT_KEYS",
    "DEFAULT_TRIGGER_KEYS",
    "DEFAULT_TRIGGER_MOUSE_BUTTONS",
    "MovementDirection5",
    "MovementDirection9",
    "MovementMode",
    "StructuredActionLoss",
    "StructuredActionPrediction",
    "StructuredActionSpec",
    "StructuredActuatorHead",
    "StructuredLossOutput",
}


def __getattr__(name: str) -> Any:
    if name in _TORCH_EXPORTS:
        torch_model = import_module(f"{__name__}.torch_model")
        return getattr(torch_model, name)
    if name in _BASELINE_EXPORTS:
        baselines = import_module(f"{__name__}.baselines")
        return getattr(baselines, name)
    if name in _INTENT_EXPORTS or name in _ACTUATOR_EXPORTS:
        intent = import_module(f"{__name__}.intent")
        return getattr(intent, name)
    if name in _LOOKAHEAD_EXPORTS:
        lookahead = import_module(f"{__name__}.lookahead_planner")
        return getattr(lookahead, name)
    if name in _ACTION_GROUP_EXPORTS:
        action_groups = import_module(f"{__name__}.action_groups")
        return getattr(action_groups, name)
    raise AttributeError(name)

