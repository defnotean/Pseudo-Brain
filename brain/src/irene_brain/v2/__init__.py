"""irene_brain.v2 — Core V2 package."""
from .config import (
    CoreV2Config, FeatureFlags,
    CONFIG_A_DECISION_ONLY, CONFIG_B_PREDICTIVE, CONFIG_C_FULL,
    DEFAULT_CONFIG, DEFAULT_FLAGS,
)
from .state import (
    CoreV2State, FastState, SessionState,
    PredictionErrorState, PendingPrediction,
    EpisodicMemoryV2, EpisodicMemoryEntry,
)
from .core import CoreV2Model, StepOutput

__all__ = [
    "CoreV2Config", "FeatureFlags",
    "CONFIG_A_DECISION_ONLY", "CONFIG_B_PREDICTIVE", "CONFIG_C_FULL",
    "DEFAULT_CONFIG", "DEFAULT_FLAGS",
    "CoreV2State", "FastState", "SessionState",
    "PredictionErrorState", "PendingPrediction",
    "EpisodicMemoryV2", "EpisodicMemoryEntry",
    "CoreV2Model", "StepOutput",
]
