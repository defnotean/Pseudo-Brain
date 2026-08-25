"""Leakage-resistant TRAIN-only calibration for action-conditioned hazard logits.

The recurrent model may learn a useful hazard ordering while its probabilities
remain systematically too high or too low.  This module fits a constrained
Platt transform independently for each action::

    calibrated_logit = positive_scale[action] * raw_logit + bias[action]

The positive scale preserves ranking within an action.  Fitting is deliberately
separate from the outcome model so qualification can prove that calibration
uses a disjoint ``TRAIN-CAL`` partition before parameters are baked into live
inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


_SCHEMA = "irene.hazard.per_action_affine.v1"
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_id_digest(values: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(set(values)),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrainCalibrationProvenance:
    """Identity boundary proving calibration is disjoint and TRAIN-only.

    ``calibration_group_ids`` normally identify factual roots or episodes and
    may repeat across action branches.  ``upstream_model_fit_group_ids`` are
    the groups used to fit the model that emitted the raw logits.  Requiring
    these sets to be disjoint prevents calibration from silently becoming an
    extra optimization pass over the model-fit records.
    """

    source_namespace: str
    source_split: str
    source_partition: str
    calibration_group_ids: tuple[str, ...]
    upstream_model_fit_group_ids: tuple[str, ...]
    upstream_checkpoint_sha256: str
    dataset_manifest_sha256: str
    source_bundle_sha256: str
    partition_algorithm: str

    def validate(self, observation_count: int) -> None:
        if not self.source_namespace.strip():
            raise ValueError("source_namespace must be non-empty")
        if self.source_split.strip().upper() != "TRAIN":
            raise ValueError("hazard calibration may only consume the TRAIN split")
        if self.source_partition.strip().upper() != "TRAIN-CAL":
            raise ValueError("source_partition must be the disjoint TRAIN-CAL partition")
        if len(self.calibration_group_ids) != observation_count:
            raise ValueError("calibration_group_ids must identify every observation")
        if not self.calibration_group_ids:
            raise ValueError("calibration_group_ids must be non-empty")
        if any(not value for value in self.calibration_group_ids):
            raise ValueError("calibration_group_ids may not contain empty IDs")
        if not self.upstream_model_fit_group_ids:
            raise ValueError("upstream_model_fit_group_ids must be non-empty")
        if any(not value for value in self.upstream_model_fit_group_ids):
            raise ValueError("upstream_model_fit_group_ids may not contain empty IDs")

        overlap = set(self.calibration_group_ids).intersection(
            self.upstream_model_fit_group_ids
        )
        if overlap:
            example = min(overlap)
            raise ValueError(
                "TRAIN-CAL groups overlap upstream model-fit groups; "
                f"first overlap: {example!r}"
            )
        _require_sha256(self.upstream_checkpoint_sha256, "upstream_checkpoint_sha256")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_sha256(self.source_bundle_sha256, "source_bundle_sha256")
        if not self.partition_algorithm.strip():
            raise ValueError("partition_algorithm must be non-empty")

    @property
    def calibration_group_digest(self) -> str:
        return _canonical_id_digest(self.calibration_group_ids)

    @property
    def upstream_model_fit_group_digest(self) -> str:
        return _canonical_id_digest(self.upstream_model_fit_group_ids)


@dataclass(frozen=True)
class PerActionCalibrationReport:
    action_id: int
    observations: int
    positives: int
    negatives: int
    before_bce: float
    after_bce: float
    accepted: bool = True

    def __post_init__(self) -> None:
        numeric = (self.before_bce, self.after_bce)
        if self.action_id < 0 or self.observations < 1:
            raise ValueError("per-action calibration identifiers/counts are invalid")
        if self.positives < 0 or self.negatives < 0:
            raise ValueError("per-action class counts must be nonnegative")
        if self.positives + self.negatives != self.observations:
            raise ValueError("per-action class counts must sum to observations")
        if any(not torch.isfinite(torch.tensor(value)) or value < 0.0 for value in numeric):
            raise ValueError("per-action BCE values must be finite and nonnegative")
        if self.after_bce > self.before_bce:
            raise ValueError("per-action calibration may not worsen BCE")
        if self.accepted != (self.after_bce < self.before_bce):
            raise ValueError("per-action acceptance must match strict BCE improvement")


@dataclass(frozen=True)
class PerActionAffineHazardCalibrator:
    """Immutable affine parameters and the evidence describing their fit."""

    scales: tuple[float, ...]
    biases: tuple[float, ...]
    source_namespace: str
    source_split: str
    source_partition: str
    calibration_group_digest: str
    upstream_model_fit_group_digest: str
    upstream_checkpoint_sha256: str
    dataset_manifest_sha256: str
    source_bundle_sha256: str
    partition_algorithm: str
    calibration_groups: int
    observations: int
    before_bce: float
    after_bce: float
    per_action: tuple[PerActionCalibrationReport, ...]
    fit_mode: str = "per_action_affine"
    accepted: bool = True

    def __post_init__(self) -> None:
        if not self.scales or len(self.scales) != len(self.biases):
            raise ValueError("scales and biases must have the same non-zero length")
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in self.scales):
            raise ValueError("every calibration scale must be finite and positive")
        if any(not math.isfinite(bias) for bias in self.biases):
            raise ValueError("every calibration bias must be finite")
        if len(self.per_action) != len(self.scales):
            raise ValueError("per_action reports must cover every action")
        if self.fit_mode not in {"bias_only", "per_action_affine"}:
            raise ValueError("fit_mode must be bias_only or per_action_affine")
        if self.fit_mode == "bias_only" and any(scale != 1.0 for scale in self.scales):
            raise ValueError("bias_only calibration requires every scale to equal 1")
        if tuple(report.action_id for report in self.per_action) != tuple(
            range(len(self.scales))
        ):
            raise ValueError("per_action reports must be in canonical action order")
        if self.source_split.upper() != "TRAIN":
            raise ValueError("exported calibration provenance must identify TRAIN")
        if self.source_partition.upper() != "TRAIN-CAL":
            raise ValueError("exported calibration provenance must identify TRAIN-CAL")
        if self.calibration_groups < 1 or self.observations < 1:
            raise ValueError("exported calibration counts must be positive")
        if self.calibration_groups > self.observations:
            raise ValueError("calibration groups cannot exceed observations")
        if not math.isfinite(self.before_bce) or not math.isfinite(self.after_bce):
            raise ValueError("exported calibration BCE values must be finite")
        if self.before_bce < 0.0 or self.after_bce < 0.0:
            raise ValueError("exported calibration BCE values must be nonnegative")
        if self.after_bce > self.before_bce:
            raise ValueError("exported calibration may not worsen BCE")
        if self.accepted != (self.after_bce < self.before_bce):
            raise ValueError("aggregate acceptance must match strict BCE improvement")
        if not self.accepted and (
            any(scale != 1.0 for scale in self.scales)
            or any(bias != 0.0 for bias in self.biases)
        ):
            raise ValueError("a rejected calibrator must retain the identity transform")
        report_observations = sum(report.observations for report in self.per_action)
        if report_observations != self.observations:
            raise ValueError("per-action observations must sum to aggregate observations")
        weighted_before = sum(
            report.observations * report.before_bce for report in self.per_action
        ) / self.observations
        weighted_after = sum(
            report.observations * report.after_bce for report in self.per_action
        ) / self.observations
        if not math.isclose(self.before_bce, weighted_before, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("aggregate before_bce differs from per-action reports")
        if not math.isclose(self.after_bce, weighted_after, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("aggregate after_bce differs from per-action reports")
        _require_sha256(self.calibration_group_digest, "calibration_group_digest")
        _require_sha256(
            self.upstream_model_fit_group_digest,
            "upstream_model_fit_group_digest",
        )
        _require_sha256(self.upstream_checkpoint_sha256, "upstream_checkpoint_sha256")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_sha256(self.source_bundle_sha256, "source_bundle_sha256")
        if not self.partition_algorithm.strip():
            raise ValueError("partition_algorithm must be non-empty")

    @property
    def action_count(self) -> int:
        return len(self.scales)

    def transform(self, logits: Tensor, action_ids: Tensor) -> Tensor:
        """Calibrate arbitrary logits labelled by an equally shaped action tensor."""
        if logits.shape != action_ids.shape:
            raise ValueError("logits and action_ids must have identical shape")
        if not torch.is_floating_point(logits):
            raise ValueError("logits must be floating point")
        raw_ids = torch.as_tensor(action_ids, device=logits.device)
        if raw_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("action_ids must use an integer dtype")
        ids = raw_ids.to(dtype=torch.long)
        if ids.numel() and (
            int(ids.min().item()) < 0 or int(ids.max().item()) >= self.action_count
        ):
            raise ValueError("action_ids contain an out-of-range action")
        scales = torch.as_tensor(self.scales, device=logits.device, dtype=logits.dtype)
        biases = torch.as_tensor(self.biases, device=logits.device, dtype=logits.dtype)
        return logits * scales[ids] + biases[ids]

    def transform_all_actions(self, logits: Tensor, action_ids: Tensor) -> Tensor:
        """Calibrate an action table using explicit semantic column labels."""
        if not torch.is_floating_point(logits):
            raise ValueError("logits must be floating point")
        raw_ids = torch.as_tensor(action_ids, device=logits.device)
        if raw_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("action_ids must use an integer dtype")
        if (
            logits.dim() >= 2
            and logits.shape[-2] == self.action_count
            and logits.shape[-1] == 1
        ):
            if raw_ids.shape != logits.shape[:-1]:
                raise ValueError("action_ids must label every action-table logit")
            semantic_ids = raw_ids
            table_logits = logits.squeeze(-1)
        else:
            if logits.shape != raw_ids.shape or logits.shape[-1] != self.action_count:
                raise ValueError("logits/action_ids must end in the complete action count")
            semantic_ids = raw_ids
            table_logits = logits
        rows = semantic_ids.to(dtype=torch.long).reshape(-1, self.action_count)
        canonical = torch.arange(
            self.action_count,
            device=logits.device,
        ).expand(rows.shape[0], -1)
        if not bool(rows.sort(dim=1).values.eq(canonical).all()):
            raise ValueError("each action table must label every semantic action once")
        transformed = self.transform(table_logits, semantic_ids)
        if logits.dim() >= 2 and logits.shape[-1] == 1:
            return transformed.unsqueeze(-1)
        return transformed

    def export_parameters(self) -> dict[str, Any]:
        """Return a JSON-serializable payload suitable for live inference."""
        return {
            "schema": _SCHEMA,
            "formula": "calibrated_logit[action] = scale[action] * raw_logit + bias[action]",
            "action_count": self.action_count,
            "fit_mode": self.fit_mode,
            "accepted": self.accepted,
            "scale": list(self.scales),
            "bias": list(self.biases),
            "fitted_on": {
                "source_namespace": self.source_namespace,
                "source_split": self.source_split,
                "source_partition": self.source_partition,
                "calibration_group_sha256": self.calibration_group_digest,
                "upstream_model_fit_group_sha256": self.upstream_model_fit_group_digest,
                "upstream_checkpoint_sha256": self.upstream_checkpoint_sha256,
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "source_bundle_sha256": self.source_bundle_sha256,
                "partition_algorithm": self.partition_algorithm,
                "calibration_groups": self.calibration_groups,
                "observations": self.observations,
            },
            "fit_bce": {
                "before": self.before_bce,
                "after": self.after_bce,
            },
            "per_action": [
                {
                    "action_id": report.action_id,
                    "observations": report.observations,
                    "positives": report.positives,
                    "negatives": report.negatives,
                    "before_bce": report.before_bce,
                    "after_bce": report.after_bce,
                    "accepted": report.accepted,
                }
                for report in self.per_action
            ],
        }

    @classmethod
    def from_export_parameters(
        cls,
        payload: Mapping[str, Any],
    ) -> PerActionAffineHazardCalibrator:
        """Reconstruct a calibrator from :meth:`export_parameters` output."""
        if payload.get("schema") != _SCHEMA:
            raise ValueError("unsupported hazard calibration schema")
        fitted_on = payload.get("fitted_on")
        fit_bce = payload.get("fit_bce")
        per_action_payload = payload.get("per_action")
        if not isinstance(fitted_on, Mapping) or not isinstance(fit_bce, Mapping):
            raise ValueError("calibration payload is missing fit provenance")
        if not isinstance(per_action_payload, list):
            raise ValueError("calibration payload is missing per-action reports")

        scales = tuple(float(value) for value in payload["scale"])
        biases = tuple(float(value) for value in payload["bias"])
        if int(payload["action_count"]) != len(scales):
            raise ValueError("action_count does not match exported parameters")
        reports = tuple(
            PerActionCalibrationReport(
                action_id=int(value["action_id"]),
                observations=int(value["observations"]),
                positives=int(value["positives"]),
                negatives=int(value["negatives"]),
                before_bce=float(value["before_bce"]),
                after_bce=float(value["after_bce"]),
                accepted=bool(value.get("accepted", True)),
            )
            for value in per_action_payload
        )
        return cls(
            scales=scales,
            biases=biases,
            source_namespace=str(fitted_on["source_namespace"]),
            source_split=str(fitted_on["source_split"]),
            source_partition=str(fitted_on["source_partition"]),
            calibration_group_digest=str(fitted_on["calibration_group_sha256"]),
            upstream_model_fit_group_digest=str(
                fitted_on["upstream_model_fit_group_sha256"]
            ),
            upstream_checkpoint_sha256=str(fitted_on["upstream_checkpoint_sha256"]),
            dataset_manifest_sha256=str(fitted_on["dataset_manifest_sha256"]),
            source_bundle_sha256=str(fitted_on["source_bundle_sha256"]),
            partition_algorithm=str(fitted_on["partition_algorithm"]),
            calibration_groups=int(fitted_on["calibration_groups"]),
            observations=int(fitted_on["observations"]),
            before_bce=float(fit_bce["before"]),
            after_bce=float(fit_bce["after"]),
            per_action=reports,
            fit_mode=str(payload.get("fit_mode", "per_action_affine")),
            accepted=bool(payload.get("accepted", True)),
        )


def _binary_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def _fit_one_action(
    logits: Tensor,
    targets: Tensor,
    *,
    l2_regularization: float,
    minimum_scale: float,
    max_iterations: int,
    tolerance: float,
    fit_mode: str,
) -> tuple[float, float]:
    """Fit a convex, positive-slope logistic calibration with damped Newton."""

    if fit_mode == "bias_only":
        # The unique intercept MLE makes mean calibrated probability equal the
        # observed prevalence.  Data-derived finite bounds safely bracket the
        # root even for large finite input logits.
        target_mean = float(targets.mean())
        lower = -float(logits.max()) - 40.0
        upper = -float(logits.min()) + 40.0
        for _ in range(max_iterations):
            midpoint = 0.5 * (lower + upper)
            predicted_mean = float(torch.sigmoid(logits + midpoint).mean())
            if predicted_mean < target_mean:
                lower = midpoint
            else:
                upper = midpoint
            if upper - lower <= tolerance:
                break
        return 1.0, 0.5 * (lower + upper)
    if fit_mode != "per_action_affine":
        raise ValueError("fit_mode must be bias_only or per_action_affine")

    def objective(scale: Tensor, bias: Tensor) -> Tensor:
        calibrated = scale * logits + bias
        bce = (F.softplus(calibrated) - targets * calibrated).mean()
        penalty = 0.5 * l2_regularization * (
            (scale - 1.0).square() + bias.square()
        )
        return bce + penalty

    scale = logits.new_tensor(1.0)
    bias = logits.new_tensor(0.0)
    current = objective(scale, bias)

    for _ in range(max_iterations):
        calibrated = scale * logits + bias
        probability = torch.sigmoid(calibrated)
        residual = probability - targets
        curvature = probability * (1.0 - probability)

        grad_scale = (residual * logits).mean() + l2_regularization * (scale - 1.0)
        grad_bias = residual.mean() + l2_regularization * bias
        if max(abs(float(grad_scale)), abs(float(grad_bias))) <= tolerance:
            break

        h_scale_scale = (
            (curvature * logits.square()).mean() + l2_regularization + 1.0e-12
        )
        h_scale_bias = (curvature * logits).mean()
        h_bias_bias = curvature.mean() + l2_regularization + 1.0e-12
        determinant = h_scale_scale * h_bias_bias - h_scale_bias.square()
        if float(determinant) <= 0.0:
            raise RuntimeError("hazard calibration Hessian is not positive definite")
        delta_scale = (
            h_bias_bias * grad_scale - h_scale_bias * grad_bias
        ) / determinant
        delta_bias = (
            h_scale_scale * grad_bias - h_scale_bias * grad_scale
        ) / determinant

        accepted = False
        step = 1.0
        for _ in range(50):
            candidate_scale = torch.clamp(
                scale - step * delta_scale,
                min=minimum_scale,
            )
            candidate_bias = bias - step * delta_bias
            candidate = objective(candidate_scale, candidate_bias)
            if float(candidate) <= float(current):
                accepted = True
                scale = candidate_scale
                bias = candidate_bias
                if abs(float(current - candidate)) <= tolerance:
                    current = candidate
                    break
                current = candidate
                break
            step *= 0.5
        if not accepted:
            break

    return float(scale), float(bias)


def fit_train_only_per_action_affine(
    logits: Tensor,
    targets: Tensor,
    action_ids: Tensor,
    *,
    provenance: TrainCalibrationProvenance,
    action_count: int,
    l2_regularization: float = 1.0e-6,
    minimum_scale: float = 1.0e-4,
    minimum_examples_per_action: int = 20,
    minimum_class_examples: int = 2,
    max_iterations: int = 100,
    tolerance: float = 1.0e-10,
    fit_mode: Literal["bias_only", "per_action_affine"] = "per_action_affine",
) -> PerActionAffineHazardCalibrator:
    """Fit deterministic per-action calibration on a disjoint TRAIN-CAL set.

    The three tensors are flattened branch observations.  The optimizer runs
    in CPU float64, starts from the identity transform, uses no randomness, and
    only accepts steps that improve its regularized objective.
    """
    if isinstance(action_count, bool) or not isinstance(action_count, int):
        raise ValueError("action_count must be an integer")
    if action_count < 1:
        raise ValueError("action_count must be positive")
    if (
        isinstance(l2_regularization, bool)
        or not isinstance(l2_regularization, (int, float))
        or not math.isfinite(l2_regularization)
        or l2_regularization <= 0.0
    ):
        raise ValueError("l2_regularization must be positive")
    if (
        isinstance(minimum_scale, bool)
        or not isinstance(minimum_scale, (int, float))
        or not math.isfinite(minimum_scale)
        or minimum_scale <= 0.0
    ):
        raise ValueError("minimum_scale must be positive")
    if (
        isinstance(minimum_examples_per_action, bool)
        or not isinstance(minimum_examples_per_action, int)
        or isinstance(minimum_class_examples, bool)
        or not isinstance(minimum_class_examples, int)
        or minimum_examples_per_action < 1
        or minimum_class_examples < 1
    ):
        raise ValueError("minimum example counts must be positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("optimizer limits must be positive")
    if fit_mode not in {"bias_only", "per_action_affine"}:
        raise ValueError("fit_mode must be bias_only or per_action_affine")

    fit_logits = torch.as_tensor(logits).detach().to(device="cpu", dtype=torch.float64)
    fit_targets = torch.as_tensor(targets).detach().to(device="cpu", dtype=torch.float64)
    raw_action_ids = torch.as_tensor(action_ids).detach().to(device="cpu")
    if raw_action_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("action_ids must use an integer dtype")
    fit_action_ids = raw_action_ids.to(dtype=torch.long)
    if fit_logits.dim() != 1:
        raise ValueError("logits must be a one-dimensional tensor")
    if fit_logits.shape != fit_targets.shape or fit_logits.shape != fit_action_ids.shape:
        raise ValueError("logits, targets, and action_ids must have identical shape")
    if not bool(torch.isfinite(fit_logits).all()):
        raise ValueError("logits must be finite")
    if not bool(torch.isfinite(fit_targets).all()):
        raise ValueError("targets must be finite")
    if not bool(((fit_targets == 0.0) | (fit_targets == 1.0)).all()):
        raise ValueError("targets must be binary")
    if fit_action_ids.numel() == 0:
        raise ValueError("at least one calibration observation is required")
    if int(fit_action_ids.min()) < 0 or int(fit_action_ids.max()) >= action_count:
        raise ValueError("action_ids contain an out-of-range action")
    provenance.validate(fit_logits.numel())

    scales: list[float] = []
    biases: list[float] = []
    reports: list[PerActionCalibrationReport] = []
    calibrated = torch.empty_like(fit_logits)
    for action_id in range(action_count):
        mask = fit_action_ids.eq(action_id)
        action_logits = fit_logits[mask]
        action_targets = fit_targets[mask]
        observations = int(mask.sum())
        positives = int(action_targets.sum())
        negatives = observations - positives
        if observations < minimum_examples_per_action:
            raise ValueError(
                f"action {action_id} has {observations} observations; "
                f"requires at least {minimum_examples_per_action}"
            )
        if positives < minimum_class_examples or negatives < minimum_class_examples:
            raise ValueError(
                f"action {action_id} requires at least {minimum_class_examples} "
                "positive and negative observations"
            )

        scale, bias = _fit_one_action(
            action_logits,
            action_targets,
            l2_regularization=l2_regularization,
            minimum_scale=minimum_scale,
            max_iterations=max_iterations,
            tolerance=tolerance,
            fit_mode=fit_mode,
        )
        action_calibrated = action_logits * scale + bias
        before_bce = float(_binary_cross_entropy(action_logits, action_targets))
        after_bce = float(_binary_cross_entropy(action_calibrated, action_targets))
        accepted = after_bce < before_bce - tolerance
        if not accepted:
            scale = 1.0
            bias = 0.0
            action_calibrated = action_logits
            after_bce = before_bce
        calibrated[mask] = action_calibrated
        scales.append(scale)
        biases.append(bias)
        reports.append(
            PerActionCalibrationReport(
                action_id=action_id,
                observations=observations,
                positives=positives,
                negatives=negatives,
                before_bce=before_bce,
                after_bce=after_bce,
                accepted=accepted,
            )
        )

    aggregate_before_bce = float(_binary_cross_entropy(fit_logits, fit_targets))
    aggregate_after_bce = float(_binary_cross_entropy(calibrated, fit_targets))
    aggregate_accepted = aggregate_after_bce < aggregate_before_bce
    if not aggregate_accepted:
        scales = [1.0] * action_count
        biases = [0.0] * action_count
        reports = [
            PerActionCalibrationReport(
                action_id=report.action_id,
                observations=report.observations,
                positives=report.positives,
                negatives=report.negatives,
                before_bce=report.before_bce,
                after_bce=report.before_bce,
                accepted=False,
            )
            for report in reports
        ]
        aggregate_after_bce = aggregate_before_bce
    return PerActionAffineHazardCalibrator(
        scales=tuple(scales),
        biases=tuple(biases),
        source_namespace=provenance.source_namespace,
        source_split=provenance.source_split.upper(),
        source_partition=provenance.source_partition.upper(),
        calibration_group_digest=provenance.calibration_group_digest,
        upstream_model_fit_group_digest=provenance.upstream_model_fit_group_digest,
        upstream_checkpoint_sha256=provenance.upstream_checkpoint_sha256,
        dataset_manifest_sha256=provenance.dataset_manifest_sha256,
        source_bundle_sha256=provenance.source_bundle_sha256,
        partition_algorithm=provenance.partition_algorithm,
        calibration_groups=len(set(provenance.calibration_group_ids)),
        observations=fit_logits.numel(),
        before_bce=aggregate_before_bce,
        after_bce=aggregate_after_bce,
        per_action=tuple(reports),
        fit_mode=fit_mode,
        accepted=aggregate_accepted,
    )


__all__ = [
    "PerActionAffineHazardCalibrator",
    "PerActionCalibrationReport",
    "TrainCalibrationProvenance",
    "fit_train_only_per_action_affine",
]
