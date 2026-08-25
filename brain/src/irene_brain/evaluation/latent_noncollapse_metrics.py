"""Deterministic latent-health metrics for qualification gates.

The functions in this module operate only on an already-produced ``[sample,
feature]`` matrix.  They do not load data, run a model, or choose gate
thresholds implicitly.  Callers must preregister explicit thresholds so a
qualification decision cannot drift between runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class LatentNonCollapseThresholds:
    """Explicit thresholds for one latent non-collapse gate."""

    minimum_samples: int
    minimum_nonzero_variance_fraction: float
    minimum_covariance_effective_rank: float
    minimum_mean_pairwise_cosine_distance: float
    variance_epsilon: float = 1.0e-12
    norm_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be at least 2")
        if not 0.0 <= self.minimum_nonzero_variance_fraction <= 1.0:
            raise ValueError("minimum_nonzero_variance_fraction must be in [0, 1]")
        if (
            not math.isfinite(self.minimum_covariance_effective_rank)
            or self.minimum_covariance_effective_rank < 0.0
        ):
            raise ValueError(
                "minimum_covariance_effective_rank must be finite and nonnegative"
            )
        if not 0.0 <= self.minimum_mean_pairwise_cosine_distance <= 2.0:
            raise ValueError("minimum_mean_pairwise_cosine_distance must be in [0, 2]")
        if not math.isfinite(self.variance_epsilon) or self.variance_epsilon <= 0.0:
            raise ValueError("variance_epsilon must be finite and positive")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be finite and positive")


@dataclass(frozen=True, slots=True)
class LatentNonCollapseMetrics:
    """Machine-readable result of a latent non-collapse evaluation.

    ``computable`` distinguishes malformed or insufficient evidence from valid
    evidence that demonstrates collapse.  Undefined metrics are represented by
    ``None`` rather than ``nan`` so JSON reports remain standards-compliant.
    """

    sample_count: int
    latent_dimension: int
    all_finite: bool
    variance_is_finite: bool
    has_nonzero_variance: bool
    mean_feature_variance: float | None
    maximum_feature_variance: float | None
    nonzero_variance_dimensions: int | None
    nonzero_variance_fraction: float | None
    covariance_effective_rank: float | None
    covariance_effective_rank_fraction: float | None
    mean_pairwise_cosine_distance: float | None
    computable: bool
    passed: bool
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, bool | float | int | list[str] | None]:
        """Return a deterministic, JSON-compatible representation."""

        return {
            "sample_count": self.sample_count,
            "latent_dimension": self.latent_dimension,
            "all_finite": self.all_finite,
            "variance_is_finite": self.variance_is_finite,
            "has_nonzero_variance": self.has_nonzero_variance,
            "mean_feature_variance": self.mean_feature_variance,
            "maximum_feature_variance": self.maximum_feature_variance,
            "nonzero_variance_dimensions": self.nonzero_variance_dimensions,
            "nonzero_variance_fraction": self.nonzero_variance_fraction,
            "covariance_effective_rank": self.covariance_effective_rank,
            "covariance_effective_rank_fraction": self.covariance_effective_rank_fraction,
            "mean_pairwise_cosine_distance": self.mean_pairwise_cosine_distance,
            "computable": self.computable,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


def _mean_pairwise_cosine_distance(
    matrix: np.ndarray,
    *,
    norm_epsilon: float,
) -> float | None:
    """Compute the exact all-pairs mean without constructing an ``N x N`` matrix."""

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= norm_epsilon):
        return None
    unit = matrix / norms[:, np.newaxis]
    sample_count = len(unit)
    pair_count = sample_count * (sample_count - 1) / 2.0
    unit_sum = np.sum(unit, axis=0, dtype=np.float64)
    pairwise_cosine_sum = (float(np.dot(unit_sum, unit_sum)) - sample_count) / 2.0
    distance = 1.0 - pairwise_cosine_sum / pair_count
    # Floating-point accumulation can miss the mathematical [0, 2] range by a
    # few ulps for large, nearly identical populations.
    return float(np.clip(distance, 0.0, 2.0))


def evaluate_latent_noncollapse(
    latents: Sequence[Sequence[float]] | np.ndarray,
    *,
    thresholds: LatentNonCollapseThresholds,
) -> LatentNonCollapseMetrics:
    """Compute deterministic metrics and apply explicit non-collapse thresholds.

    The input must be a two-dimensional matrix with at least one feature.
    Too few samples, non-finite values, and zero-norm rows yield a failed report
    rather than an exception or a report containing ``nan``.  Shape/type errors
    still raise ``ValueError`` because they indicate a caller contract bug.
    """

    try:
        matrix = np.asarray(latents, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("latents must be a numeric two-dimensional matrix") from error
    if matrix.ndim != 2:
        raise ValueError(f"latents must have shape [sample, feature], got {matrix.shape}")
    sample_count, latent_dimension = matrix.shape
    if latent_dimension == 0:
        raise ValueError("latents must contain at least one feature")

    all_finite = bool(np.isfinite(matrix).all())
    reasons: list[str] = []
    if sample_count < thresholds.minimum_samples:
        reasons.append(f"insufficient_samples:{sample_count}<{thresholds.minimum_samples}")
    if not all_finite:
        reasons.append("non_finite_latents")

    if sample_count < 2 or not all_finite:
        return LatentNonCollapseMetrics(
            sample_count=sample_count,
            latent_dimension=latent_dimension,
            all_finite=all_finite,
            variance_is_finite=False,
            has_nonzero_variance=False,
            mean_feature_variance=None,
            maximum_feature_variance=None,
            nonzero_variance_dimensions=None,
            nonzero_variance_fraction=None,
            covariance_effective_rank=None,
            covariance_effective_rank_fraction=None,
            mean_pairwise_cosine_distance=None,
            computable=False,
            passed=False,
            failure_reasons=tuple(reasons),
        )

    centered = matrix - np.mean(matrix, axis=0, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        feature_variances = np.sum(centered * centered, axis=0) / (sample_count - 1)
    variance_is_finite = bool(np.isfinite(feature_variances).all())
    if not variance_is_finite:
        reasons.append("non_finite_variance")
        return LatentNonCollapseMetrics(
            sample_count=sample_count,
            latent_dimension=latent_dimension,
            all_finite=all_finite,
            variance_is_finite=False,
            has_nonzero_variance=False,
            mean_feature_variance=None,
            maximum_feature_variance=None,
            nonzero_variance_dimensions=None,
            nonzero_variance_fraction=None,
            covariance_effective_rank=None,
            covariance_effective_rank_fraction=None,
            mean_pairwise_cosine_distance=None,
            computable=False,
            passed=False,
            failure_reasons=tuple(reasons),
        )
    mean_variance = float(np.mean(feature_variances))
    maximum_variance = float(np.max(feature_variances))
    nonzero_dimensions = int(np.count_nonzero(feature_variances > thresholds.variance_epsilon))
    nonzero_fraction = nonzero_dimensions / latent_dimension
    has_nonzero_variance = nonzero_dimensions > 0

    covariance = centered.T @ centered / (sample_count - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_covariance_variance = float(np.sum(eigenvalues))
    if total_covariance_variance <= thresholds.variance_epsilon:
        effective_rank = 0.0
        reasons.append("zero_total_variance")
    else:
        normalized_spectrum = eigenvalues[eigenvalues > 0.0] / total_covariance_variance
        spectral_entropy = -float(
            np.sum(normalized_spectrum * np.log(normalized_spectrum), dtype=np.float64)
        )
        effective_rank = float(math.exp(spectral_entropy))
    maximum_possible_rank = min(latent_dimension, sample_count - 1)
    effective_rank_fraction = effective_rank / maximum_possible_rank

    cosine_distance = _mean_pairwise_cosine_distance(
        matrix,
        norm_epsilon=thresholds.norm_epsilon,
    )
    if cosine_distance is None:
        reasons.append("zero_norm_latent")

    if nonzero_fraction < thresholds.minimum_nonzero_variance_fraction:
        reasons.append(
            "nonzero_variance_fraction_below_minimum:"
            f"{nonzero_fraction:.12g}<{thresholds.minimum_nonzero_variance_fraction:.12g}"
        )
    if effective_rank < thresholds.minimum_covariance_effective_rank:
        reasons.append(
            "covariance_effective_rank_below_minimum:"
            f"{effective_rank:.12g}<{thresholds.minimum_covariance_effective_rank:.12g}"
        )
    if (
        cosine_distance is not None
        and cosine_distance < thresholds.minimum_mean_pairwise_cosine_distance
    ):
        reasons.append(
            "mean_pairwise_cosine_distance_below_minimum:"
            f"{cosine_distance:.12g}"
            f"<{thresholds.minimum_mean_pairwise_cosine_distance:.12g}"
        )

    computable = sample_count >= thresholds.minimum_samples and cosine_distance is not None
    passed = computable and not reasons
    return LatentNonCollapseMetrics(
        sample_count=sample_count,
        latent_dimension=latent_dimension,
        all_finite=all_finite,
        variance_is_finite=variance_is_finite,
        has_nonzero_variance=has_nonzero_variance,
        mean_feature_variance=mean_variance,
        maximum_feature_variance=maximum_variance,
        nonzero_variance_dimensions=nonzero_dimensions,
        nonzero_variance_fraction=nonzero_fraction,
        covariance_effective_rank=effective_rank,
        covariance_effective_rank_fraction=effective_rank_fraction,
        mean_pairwise_cosine_distance=cosine_distance,
        computable=computable,
        passed=passed,
        failure_reasons=tuple(reasons),
    )
