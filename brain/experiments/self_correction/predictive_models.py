"""Phase B, C, F: Predictive recurrent models with surprise feedback & fast plasticity.

Re-exports core architectures from self_correction.models for backwards compatibility.
"""
from __future__ import annotations

from self_correction.models import (
    LATENT_DIM,
    SURPRISE_DIM,
    SurpriseEncoder,
    FutureLatentPredictor,
    FastPlasticityModule,
    PredictiveGRUModel,
    PredictiveThoughtletModel,
)

__all__ = [
    "SurpriseEncoder",
    "FutureLatentPredictor",
    "FastPlasticityModule",
    "PredictiveGRUModel",
    "PredictiveThoughtletModel",
    "LATENT_DIM",
    "SURPRISE_DIM",
]
