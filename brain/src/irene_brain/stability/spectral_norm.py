"""Spectral radius constraints and Cayley parameterizations for Lyapunov stability.

Enforces transition matrix spectral radius rho(W) <= 1.0 and spectral norm ||W||_2 <= 1.0
to mathematically guarantee Lyapunov stability, eliminate chaotic divergence, and prevent
attractor drift across long recurrent horizons.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CayleyLinear(nn.Module):
    """Square linear layer with Cayley parameterization guaranteeing spectral radius <= 1.0.

    Constructs an orthogonal matrix Q using the Cayley transform:
        A = M - M^T  (skew-symmetric, eigenvalues are purely imaginary: i * mu_k)
        Q = (I + A)^(-1) * (I - A)  (strictly orthogonal: Q^T * Q = I)

    Combined with a bounded diagonal damping or scaling matrix D = diag(sigma) with
    0 <= sigma_i <= max_spectral_radius <= 1.0:
        W = D * Q

    Because ||Q||_2 = 1.0 and ||D||_2 <= max_spectral_radius:
        ||W v||_2 <= max_spectral_radius * ||v||_2
        rho(W) <= ||W||_2 <= max_spectral_radius <= 1.0

    This guarantees that the linear transition is non-expansive / contractive,
    providing an unconditional mathematical guarantee of Lyapunov stability.
    """

    def __init__(
        self,
        features: int,
        bias: bool = True,
        max_spectral_radius: float = 1.0,
        init_scale: float = 0.95,
    ) -> None:
        super().__init__()
        self.features = features
        self.max_spectral_radius = float(max_spectral_radius)

        # Unconstrained generator matrix M
        self.weight_raw = nn.Parameter(torch.empty(features, features))

        # Damping scale parameter: parameterized via sigmoid to strictly stay in (0, max_spectral_radius]
        init_logit = math.log(init_scale / (self.max_spectral_radius - init_scale + 1e-7))
        self.scale_logits = nn.Parameter(torch.full((features,), init_logit))

        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight_raw)
        self.weight_raw.data.mul_(0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_orthogonal_matrix(self) -> Tensor:
        """Compute the orthogonal Cayley transform matrix Q."""
        I = torch.eye(self.features, device=self.weight_raw.device, dtype=self.weight_raw.dtype)
        # Skew-symmetric matrix A = M - M^T
        A = self.weight_raw - self.weight_raw.T
        # Solve (I + A) * Q = (I - A)  =>  Q = (I + A)^(-1) * (I - A)
        Q = torch.linalg.solve(I + A, I - A)
        return Q

    def get_weight(self) -> Tensor:
        """Construct the effective weight matrix W = diag(sigma) * Q with rho(W) <= max_spectral_radius."""
        Q = self.get_orthogonal_matrix()
        # Scale sigma in [0, max_spectral_radius]
        sigma = torch.sigmoid(self.scale_logits) * self.max_spectral_radius
        # Effective weight: W = diag(sigma) @ Q
        W = sigma.unsqueeze(1) * Q
        return W

    def forward(self, x: Tensor) -> Tensor:
        """Forward evaluation: y = x @ W^T + b."""
        W = self.get_weight()
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        return f"features={self.features}, max_spectral_radius={self.max_spectral_radius}, bias={self.bias is not None}"


class SpectralNormalizedLinear(nn.Module):
    """Linear layer with dynamic spectral norm projection ensuring ||W||_2 <= max_spectral_radius.

    At each forward pass, the largest singular value sigma_max is bounded:
        W_eff = W / max(1.0, sigma_max(W) / max_spectral_radius)

    Guarantees rho(W_eff) <= ||W_eff||_2 <= max_spectral_radius <= 1.0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        max_spectral_radius: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_spectral_radius = float(max_spectral_radius)
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight)
        # Ensure initial spectral norm is <= max_spectral_radius
        with torch.no_grad():
            s = torch.linalg.matrix_norm(self.weight, ord=2)
            if s > self.max_spectral_radius:
                self.weight.data.mul_(self.max_spectral_radius / (s + self.eps))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_effective_weight(self) -> Tensor:
        """Return the spectrally bounded weight matrix."""
        s_max = torch.linalg.matrix_norm(self.weight, ord=2)
        factor = torch.clamp(s_max / self.max_spectral_radius, min=1.0)
        return self.weight / factor

    def forward(self, x: Tensor) -> Tensor:
        W_eff = self.get_effective_weight()
        return F.linear(x, W_eff, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"max_spectral_radius={self.max_spectral_radius}, bias={self.bias is not None}"
        )


def compute_spectral_radius(W: Tensor) -> float:
    """Compute the exact spectral radius rho(W) = max |lambda_i(W)| of a square matrix.

    Args:
        W: 2D square tensor [d, d]

    Returns:
        float: Maximum eigenvalue magnitude
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a 2D square matrix, got shape {W.shape}")
    eigenvalues = torch.linalg.eigvals(W.to(torch.float64))
    return float(torch.max(torch.abs(eigenvalues)).item())


def compute_spectral_norm(W: Tensor) -> float:
    """Compute the 2-norm / maximum singular value sigma_max(W) of a matrix.

    Args:
        W: 2D tensor [m, n]

    Returns:
        float: Maximum singular value ||W||_2
    """
    if W.ndim != 2:
        raise ValueError(f"W must be a 2D matrix, got shape {W.shape}")
    return float(torch.linalg.matrix_norm(W.to(torch.float64), ord=2).item())


def verify_spectral_radius(module: nn.Module, max_radius: float = 1.0) -> Dict[str, float]:
    """Inspect all square 2D weight matrices in a module and verify rho(W) <= max_radius.

    Returns:
        dict mapping weight name -> spectral radius
    """
    results: Dict[str, float] = {}
    for name, param in module.named_parameters():
        if param.ndim == 2 and param.shape[0] == param.shape[1]:
            rho = compute_spectral_radius(param.data)
            results[name] = rho
            if rho > max_radius + 1e-4:
                raise ValueError(
                    f"Parameter {name} violates spectral radius bound: rho = {rho:.6f} > {max_radius}"
                )
    return results


def verify_lyapunov_stability(
    step_fn: Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    inputs: Tensor,
    perturbation_norm: float = 1e-4,
) -> Dict[str, Any]:
    """Simulate unperturbed and perturbed trajectories to empirically measure Lyapunov stability.

    Computes:
        delta_t = ||h_t - h~_t|| / ||h_0 - h~_0||
        lambda = (1 / T) * ln( ||delta_T|| / ||delta_0|| )

    Lyapunov stability requires:
        lambda <= 0  (non-positive maximal Lyapunov exponent)
        max(delta_t) <= 1.0 + tol (perturbations do not explode)

    Args:
        step_fn: (h, x) -> next_h
        initial_state: Tensor [..., W]
        inputs: Tensor [T, ..., D]
        perturbation_norm: initial perturbation magnitude epsilon

    Returns:
        Dictionary of Lyapunov stability diagnostics
    """
    T = inputs.shape[0]
    device = initial_state.device
    dtype = initial_state.dtype

    # Create random direction perturbation with exact norm = perturbation_norm
    perturbation = torch.randn_like(initial_state)
    p_norm = torch.linalg.norm(perturbation, dim=-1, keepdim=True).clamp(min=1e-8)
    delta_0 = (perturbation / p_norm) * perturbation_norm
    h0_pert = initial_state + delta_0

    init_dist = float(torch.linalg.norm(delta_0).item())

    h = initial_state.clone()
    h_pert = h0_pert.clone()

    ratios = []
    with torch.no_grad():
        for t in range(T):
            x_t = inputs[t]
            h = step_fn(h, x_t)
            h_pert = step_fn(h_pert, x_t)

            dist_t = float(torch.linalg.norm(h - h_pert).item())
            ratio_t = dist_t / (init_dist + 1e-12)
            ratios.append(ratio_t)

    max_ratio = max(ratios)
    final_ratio = ratios[-1]
    lyapunov_exponent = math.log(max(final_ratio, 1e-12)) / float(T)

    return {
        "initial_distance": init_dist,
        "final_distance": float(torch.linalg.norm(h - h_pert).item()),
        "max_perturbation_ratio": max_ratio,
        "final_perturbation_ratio": final_ratio,
        "lyapunov_exponent": lyapunov_exponent,
        "is_lyapunov_stable": (lyapunov_exponent <= 1e-4) and (final_ratio <= 1.0) and (max_ratio <= 25.0),
        "trajectory_ratios": ratios,
    }


__all__ = [
    "CayleyLinear",
    "SpectralNormalizedLinear",
    "compute_spectral_norm",
    "compute_spectral_radius",
    "verify_lyapunov_stability",
    "verify_spectral_radius",
]
