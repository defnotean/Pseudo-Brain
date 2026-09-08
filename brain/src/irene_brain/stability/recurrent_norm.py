"""Recurrent slot normalization modules for Pseudo-Brain stability.

Prevents unbounded internal state norm growth, collapse, and attractor drift
across long unrolled recurrence horizons (1,000+ steps) and deep stacks (24-32 layers).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


class RecurrentRMSNorm(nn.Module):
    """Root-Mean-Square Normalization directly for recurrent slot transitions.

    Ensures that slot state vectors maintain a calibrated root-mean-square magnitude:
        RMS(x) = sqrt( mean(x^2) + eps )
        y = (x / RMS(x)) * gamma

    When gamma=1.0 and target_norm=1.0, RMS(y) == 1.0 identically.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-12,
        target_norm: float = 1.0,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.target_norm = float(target_norm)
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS along the last dimension (slot width W)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        normalized = (x / rms) * self.target_norm
        if self.gamma is not None:
            normalized = normalized * self.gamma
        return normalized

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, target_norm={self.target_norm}, affine={self.affine}"


class RecurrentLayerNorm(nn.Module):
    """Layer Normalization tailored for recurrent slot transitions.

    Centers mean and scales variance across the slot feature dimension:
        y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        target_norm: float = 1.0,
        affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.target_norm = float(target_norm)
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim)) if bias else None
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        normalized = ((x - mean) / std) * self.target_norm
        if self.gamma is not None:
            normalized = normalized * self.gamma
        if self.beta is not None:
            normalized = normalized + self.beta
        return normalized

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, target_norm={self.target_norm}, affine={self.affine}"


class RecurrentL2Norm(nn.Module):
    """Unit L2 Sphere Projection for recurrent slot states.

    Projects slot state vectors onto an L2 sphere of specified radius (default 1.0):
        ||x||_2 = sqrt( sum(x^2) + eps )
        y = target_norm * (x / ||x||_2) * gamma
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-8,
        target_norm: float = 1.0,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.target_norm = float(target_norm)
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: Tensor) -> Tensor:
        l2_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        normalized = (x / torch.clamp(l2_norm, min=self.eps)) * self.target_norm
        if self.gamma is not None:
            normalized = normalized * self.gamma
        return normalized

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, target_norm={self.target_norm}, affine={self.affine}"


class RecurrentSlotNorm(nn.Module):
    """Unified Recurrent Slot Normalizer with hard mathematical bounds.

    Guarantees that slot norms remain strictly bounded in [min_bound, max_bound]
    (default [0.9, 1.1]) across thousands of unrolled recurrent steps.

    Modes:
        - "rms": RecurrentRMSNorm (standard for deep transformer/recurrent slots)
        - "layer_norm": RecurrentLayerNorm (zero-mean unit-variance)
        - "l2": RecurrentL2Norm (exact Euclidean unit sphere projection)
    """

    def __init__(
        self,
        dim: int,
        norm_type: Literal["rms", "layer_norm", "l2"] = "rms",
        eps: float = 1e-12,
        target_norm: float = 1.0,
        affine: bool = True,
        min_bound: float = 0.90,
        max_bound: float = 1.10,
        enforce_hard_bounds: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm_type = norm_type
        self.eps = eps
        self.target_norm = target_norm
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.enforce_hard_bounds = enforce_hard_bounds

        if norm_type == "rms":
            self.normalizer: nn.Module = RecurrentRMSNorm(
                dim=dim,
                eps=eps,
                target_norm=target_norm,
                affine=affine,
            )
        elif norm_type == "layer_norm":
            self.normalizer = RecurrentLayerNorm(
                dim=dim,
                eps=eps,
                target_norm=target_norm,
                affine=affine,
            )
        elif norm_type == "l2":
            self.normalizer = RecurrentL2Norm(
                dim=dim,
                eps=eps,
                target_norm=target_norm,
                affine=affine,
            )
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}. Must be 'rms', 'layer_norm', or 'l2'.")

    def forward(self, x: Tensor) -> Tensor:
        """Normalize slot states and enforce [min_bound, max_bound] safety window.

        Args:
            x: Tensor of shape [..., dim]

        Returns:
            Normalized tensor with internal norms strictly in [min_bound, max_bound].
        """
        y = self.normalizer(x)

        if self.enforce_hard_bounds:
            # Measure the norm metric matching the normalizer type
            if self.norm_type == "l2":
                curr_norm = torch.linalg.norm(y, dim=-1, keepdim=True)
            else:
                # RMS norm: sqrt(mean(y^2))
                curr_norm = torch.sqrt(torch.mean(y.pow(2), dim=-1, keepdim=True) + self.eps)

            # Clamp norm to [min_bound, max_bound]
            clamped_norm = torch.clamp(curr_norm, min=self.min_bound, max=self.max_bound)
            scale = clamped_norm / torch.clamp(curr_norm, min=self.eps)
            y = y * scale

        return y

    def compute_norm(self, x: Tensor) -> Tensor:
        """Compute the norm of tensor x along the feature dimension."""
        if self.norm_type == "l2":
            return torch.linalg.norm(x, dim=-1)
        return torch.sqrt(torch.mean(x.pow(2), dim=-1) + self.eps)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, norm_type={self.norm_type}, target_norm={self.target_norm}, "
            f"bounds=[{self.min_bound}, {self.max_bound}], enforce_hard_bounds={self.enforce_hard_bounds}"
        )


__all__ = [
    "RecurrentL2Norm",
    "RecurrentLayerNorm",
    "RecurrentRMSNorm",
    "RecurrentSlotNorm",
]
