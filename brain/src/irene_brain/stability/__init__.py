"""Pseudo-Brain Recurrent Dynamics and Stability Package.

Guarantees deep recurrent stability, eliminates attractor drift, and enables
24-32 layer scaling with bounded gradients and Lyapunov stability.
"""

from __future__ import annotations

from .deep_recurrent_stack import (
    DeepRecurrentBlock,
    DeepRecurrentStack,
    DeepStableBrainCellCore,
    GradientHighwayGate,
)
from .recurrent_norm import (
    RecurrentL2Norm,
    RecurrentLayerNorm,
    RecurrentRMSNorm,
    RecurrentSlotNorm,
)
from .spectral_norm import (
    CayleyLinear,
    SpectralNormalizedLinear,
    compute_spectral_norm,
    compute_spectral_radius,
    verify_lyapunov_stability,
    verify_spectral_radius,
)

__all__ = [
    "CayleyLinear",
    "DeepRecurrentBlock",
    "DeepRecurrentStack",
    "DeepStableBrainCellCore",
    "GradientHighwayGate",
    "RecurrentL2Norm",
    "RecurrentLayerNorm",
    "RecurrentRMSNorm",
    "RecurrentSlotNorm",
    "SpectralNormalizedLinear",
    "compute_spectral_norm",
    "compute_spectral_radius",
    "verify_lyapunov_stability",
    "verify_spectral_radius",
]
