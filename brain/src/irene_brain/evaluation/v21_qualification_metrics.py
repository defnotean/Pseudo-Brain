"""Deterministic qualification metrics for V2.1 all-action hazard predictions.

This module is deliberately independent from model and runner code.  It accepts
already-produced probabilities and binary outcomes, never opens a dataset split,
and never trains or mutates a model.  The strict validation is intentional:
undefined AUCs, missing action coverage, or impossible causal shuffles are
qualification failures rather than values that should silently become ``nan``.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Hashable, Sequence

import numpy as np


_BCE_EPSILON = 1.0e-12
DEFAULT_DERANGEMENT_REPETITIONS = 20
DEFAULT_DERANGEMENT_SEED = 21_000_021
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 21_000_043


class QualificationMetricsError(ValueError):
    """Raised when evidence is incomplete or a requested metric is undefined."""


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Strict binary probability metrics for one population."""

    count: int
    positives: int
    prevalence: float
    mean_probability: float
    bce: float
    brier: float
    roc_auc: float
    pr_auc: float
    ece_equal_mass: float
    calibration_bias: float
    ece_bins: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "positives": self.positives,
            "prevalence": self.prevalence,
            "mean_probability": self.mean_probability,
            "bce": self.bce,
            "brier": self.brier,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "ece_equal_mass": self.ece_equal_mass,
            "calibration_bias": self.calibration_bias,
            "ece_bins": self.ece_bins,
        }


@dataclass(frozen=True)
class ActionMetric:
    action_id: int
    metrics: BinaryClassificationMetrics

    def as_dict(self) -> dict[str, object]:
        return {"action_id": self.action_id, "metrics": self.metrics.as_dict()}


@dataclass(frozen=True)
class AllActionMetricReport:
    """Aggregate and semantic per-action metrics for a ``[root, action]`` table."""

    action_ids: tuple[int, ...]
    aggregate: BinaryClassificationMetrics
    per_action: tuple[ActionMetric, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "action_ids": list(self.action_ids),
            "aggregate": self.aggregate.as_dict(),
            "per_action": [entry.as_dict() for entry in self.per_action],
        }


@dataclass(frozen=True)
class ActionPriorBaseline:
    """Action-conditional Bernoulli priors fitted only from TRAIN labels."""

    source: str
    action_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    positive_counts: tuple[int, ...]
    sample_counts: tuple[int, ...]

    def probability_table(self, root_count: int) -> np.ndarray:
        if root_count <= 0:
            raise QualificationMetricsError("root_count must be positive")
        row = np.asarray(self.probabilities, dtype=np.float64)
        return np.broadcast_to(row, (root_count, len(row))).copy()

    def probabilities_for_actions(self, actions: Sequence[int] | np.ndarray) -> np.ndarray:
        action_array = _action_vector(actions, "actions")
        positions = {action_id: index for index, action_id in enumerate(self.action_ids)}
        try:
            columns = np.fromiter(
                (positions[int(action)] for action in action_array),
                dtype=np.int64,
                count=len(action_array),
            )
        except KeyError as error:
            raise QualificationMetricsError(
                f"action {int(error.args[0])} was absent from the fitted baseline"
            ) from error
        return np.asarray(self.probabilities, dtype=np.float64)[columns]

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "action_ids": list(self.action_ids),
            "probabilities": list(self.probabilities),
            "positive_counts": list(self.positive_counts),
            "sample_counts": list(self.sample_counts),
        }


@dataclass(frozen=True)
class TrainFittedBaselines:
    """Separate all-branch and factual action priors fitted from TRAIN."""

    all_action: ActionPriorBaseline
    factual: ActionPriorBaseline

    def as_dict(self) -> dict[str, object]:
        return {
            "all_action": self.all_action.as_dict(),
            "factual": self.factual.as_dict(),
        }


@dataclass(frozen=True)
class BaselineMetricReport:
    all_action: AllActionMetricReport
    factual: BinaryClassificationMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "all_action": self.all_action.as_dict(),
            "factual": self.factual.as_dict(),
        }


@dataclass(frozen=True)
class ActionDerangementSummary:
    action_id: int
    real_bce: float
    median_shuffled_bce: float
    real_brier: float
    median_shuffled_brier: float
    real_roc_auc: float
    median_shuffled_roc_auc: float
    roc_auc_drop: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "action_id": self.action_id,
            "real_bce": self.real_bce,
            "median_shuffled_bce": self.median_shuffled_bce,
            "real_brier": self.real_brier,
            "median_shuffled_brier": self.median_shuffled_brier,
            "real_roc_auc": self.real_roc_auc,
            "median_shuffled_roc_auc": self.median_shuffled_roc_auc,
            "roc_auc_drop": self.roc_auc_drop,
        }


@dataclass(frozen=True)
class CrossEpisodeDerangementReport:
    """Real metrics and the fixed cross-episode null distribution."""

    repetitions: int
    seed: int
    real: AllActionMetricReport
    shuffled: tuple[AllActionMetricReport, ...]
    median_shuffled_bce: float
    median_shuffled_brier: float
    median_shuffled_roc_auc: float
    aggregate_roc_auc_drop: float
    per_action: tuple[ActionDerangementSummary, ...]

    def as_dict(self, *, include_repetitions: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "repetitions": self.repetitions,
            "seed": self.seed,
            "real": self.real.as_dict(),
            "median_shuffled_bce": self.median_shuffled_bce,
            "median_shuffled_brier": self.median_shuffled_brier,
            "median_shuffled_roc_auc": self.median_shuffled_roc_auc,
            "aggregate_roc_auc_drop": self.aggregate_roc_auc_drop,
            "per_action": [entry.as_dict() for entry in self.per_action],
        }
        if include_repetitions:
            result["shuffled"] = [entry.as_dict() for entry in self.shuffled]
        return result


@dataclass(frozen=True)
class ClusteredImprovementBootstrap:
    """One-sided clustered-bootstrap bounds for baseline-minus-model losses."""

    resamples: int
    seed: int
    confidence_level: float
    cluster_count: int
    observation_count: int
    observed_bce_improvement: float
    bce_improvement_lower_bound: float
    observed_brier_improvement: float
    brier_improvement_lower_bound: float
    replicate_sha256: str

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "resamples": self.resamples,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "cluster_count": self.cluster_count,
            "observation_count": self.observation_count,
            "observed_bce_improvement": self.observed_bce_improvement,
            "bce_improvement_lower_bound": self.bce_improvement_lower_bound,
            "observed_brier_improvement": self.observed_brier_improvement,
            "brier_improvement_lower_bound": self.brier_improvement_lower_bound,
            "replicate_sha256": self.replicate_sha256,
        }


def binary_probability_metrics(
    targets: Sequence[int | float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    ece_bins: int = 10,
) -> BinaryClassificationMetrics:
    """Compute strict BCE, Brier, ROC-AUC, AP, equal-mass ECE, and bias.

    ``pr_auc`` is threshold-grouped average precision (step integration), so a
    constant score has PR-AUC equal to prevalence.  ``calibration_bias`` is
    signed ``mean(probability - target)``; qualification gates can apply an
    absolute-value threshold without losing the direction of miscalibration.
    """

    target_array, probability_array = _binary_inputs(targets, probabilities)
    if ece_bins <= 0:
        raise QualificationMetricsError("ece_bins must be positive")
    positives = int(np.sum(target_array))
    count = len(target_array)
    if positives == 0 or positives == count:
        raise QualificationMetricsError(
            "ROC-AUC and PR-AUC require at least one positive and one negative outcome"
        )

    bce_losses = _binary_cross_entropy_losses(target_array, probability_array)
    brier_losses = np.square(probability_array - target_array)
    used_bins = min(ece_bins, count)
    return BinaryClassificationMetrics(
        count=count,
        positives=positives,
        prevalence=float(np.mean(target_array)),
        mean_probability=float(np.mean(probability_array)),
        bce=float(np.mean(bce_losses)),
        brier=float(np.mean(brier_losses)),
        roc_auc=_roc_auc(target_array, probability_array),
        pr_auc=_average_precision(target_array, probability_array),
        ece_equal_mass=_equal_mass_ece(target_array, probability_array, used_bins),
        calibration_bias=float(np.mean(probability_array - target_array)),
        ece_bins=used_bins,
    )


def all_action_probability_metrics(
    targets: Sequence[Sequence[int | float]] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    action_ids: Sequence[int] | np.ndarray | None = None,
    ece_bins: int = 10,
) -> AllActionMetricReport:
    """Compute aggregate and per-action metrics without losing action identity."""

    target_table, probability_table = _all_action_inputs(targets, probabilities)
    semantic_actions = _semantic_action_ids(action_ids, target_table.shape[1])
    aggregate = binary_probability_metrics(
        target_table.reshape(-1), probability_table.reshape(-1), ece_bins=ece_bins
    )
    per_action = tuple(
        ActionMetric(
            action_id=action_id,
            metrics=binary_probability_metrics(
                target_table[:, column], probability_table[:, column], ece_bins=ece_bins
            ),
        )
        for column, action_id in enumerate(semantic_actions)
    )
    return AllActionMetricReport(
        action_ids=semantic_actions,
        aggregate=aggregate,
        per_action=per_action,
    )


def gather_factual_rows(
    all_action_targets: Sequence[Sequence[int | float]] | np.ndarray,
    all_action_probabilities: Sequence[Sequence[float]] | np.ndarray,
    factual_actions: Sequence[int] | np.ndarray,
    *,
    action_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather factual labels/probabilities by semantic action ID."""

    targets, probabilities = _all_action_inputs(all_action_targets, all_action_probabilities)
    semantic_actions = _semantic_action_ids(action_ids, targets.shape[1])
    actions = _action_vector(factual_actions, "factual_actions")
    if len(actions) != targets.shape[0]:
        raise QualificationMetricsError(
            "factual_actions length must equal the all-action root count"
        )
    positions = {action_id: index for index, action_id in enumerate(semantic_actions)}
    try:
        columns = np.fromiter(
            (positions[int(action)] for action in actions),
            dtype=np.int64,
            count=len(actions),
        )
    except KeyError as error:
        raise QualificationMetricsError(
            f"factual action {int(error.args[0])} is absent from action_ids"
        ) from error
    rows = np.arange(targets.shape[0], dtype=np.int64)
    return targets[rows, columns].copy(), probabilities[rows, columns].copy()


def fit_train_all_action_priors(
    train_targets: Sequence[Sequence[int | float]] | np.ndarray,
    *,
    action_ids: Sequence[int] | np.ndarray | None = None,
) -> ActionPriorBaseline:
    """Fit one empirical hazard probability per action from all TRAIN branches."""

    target_table = _binary_target_table(train_targets, "train_targets")
    semantic_actions = _semantic_action_ids(action_ids, target_table.shape[1])
    positives = np.sum(target_table, axis=0).astype(np.int64)
    counts = np.full(target_table.shape[1], target_table.shape[0], dtype=np.int64)
    return ActionPriorBaseline(
        source="train_all_action_targets",
        action_ids=semantic_actions,
        probabilities=tuple(float(value) for value in positives / counts),
        positive_counts=tuple(int(value) for value in positives),
        sample_counts=tuple(int(value) for value in counts),
    )


def fit_train_factual_action_priors(
    train_actions: Sequence[int] | np.ndarray,
    train_targets: Sequence[int | float] | np.ndarray,
    *,
    action_ids: Sequence[int] | np.ndarray,
) -> ActionPriorBaseline:
    """Fit factual action priors from TRAIN factual actions and outcomes only."""

    actions = _action_vector(train_actions, "train_actions")
    targets = _binary_target_vector(train_targets, "train_targets")
    if len(actions) != len(targets):
        raise QualificationMetricsError("train_actions and train_targets lengths differ")
    semantic_actions = _semantic_action_ids(action_ids, len(action_ids))
    positives: list[int] = []
    counts: list[int] = []
    probabilities: list[float] = []
    for action_id in semantic_actions:
        mask = actions == action_id
        count = int(np.sum(mask))
        if count == 0:
            raise QualificationMetricsError(
                f"TRAIN factual baseline has no samples for action {action_id}"
            )
        positive_count = int(np.sum(targets[mask]))
        positives.append(positive_count)
        counts.append(count)
        probabilities.append(positive_count / count)
    return ActionPriorBaseline(
        source="train_factual_targets",
        action_ids=semantic_actions,
        probabilities=tuple(probabilities),
        positive_counts=tuple(positives),
        sample_counts=tuple(counts),
    )


def fit_train_baselines(
    train_all_action_targets: Sequence[Sequence[int | float]] | np.ndarray,
    train_factual_actions: Sequence[int] | np.ndarray,
    train_factual_targets: Sequence[int | float] | np.ndarray,
    *,
    action_ids: Sequence[int] | np.ndarray | None = None,
) -> TrainFittedBaselines:
    """Fit the branch and factual priors; no evaluation labels are consulted."""

    target_table = _binary_target_table(train_all_action_targets, "train_all_action_targets")
    semantic_actions = _semantic_action_ids(action_ids, target_table.shape[1])
    return TrainFittedBaselines(
        all_action=fit_train_all_action_priors(
            target_table,
            action_ids=semantic_actions,
        ),
        factual=fit_train_factual_action_priors(
            train_factual_actions,
            train_factual_targets,
            action_ids=semantic_actions,
        ),
    )


def evaluate_train_fitted_baselines(
    baselines: TrainFittedBaselines,
    evaluation_all_action_targets: Sequence[Sequence[int | float]] | np.ndarray,
    evaluation_factual_actions: Sequence[int] | np.ndarray,
    evaluation_factual_targets: Sequence[int | float] | np.ndarray,
    *,
    ece_bins: int = 10,
) -> BaselineMetricReport:
    """Evaluate frozen TRAIN priors on a separate qualification population."""

    target_table = _binary_target_table(
        evaluation_all_action_targets, "evaluation_all_action_targets"
    )
    if target_table.shape[1] != len(baselines.all_action.action_ids):
        raise QualificationMetricsError(
            "evaluation action width differs from the fitted all-action baseline"
        )
    all_action_report = all_action_probability_metrics(
        target_table,
        baselines.all_action.probability_table(target_table.shape[0]),
        action_ids=baselines.all_action.action_ids,
        ece_bins=ece_bins,
    )
    factual_targets = _binary_target_vector(
        evaluation_factual_targets, "evaluation_factual_targets"
    )
    factual_actions = _action_vector(evaluation_factual_actions, "evaluation_factual_actions")
    if len(factual_actions) != len(factual_targets):
        raise QualificationMetricsError(
            "evaluation_factual_actions and evaluation_factual_targets lengths differ"
        )
    factual_probabilities = baselines.factual.probabilities_for_actions(factual_actions)
    return BaselineMetricReport(
        all_action=all_action_report,
        factual=binary_probability_metrics(
            factual_targets, factual_probabilities, ece_bins=ece_bins
        ),
    )


def fixed_cross_episode_derangement_indices(
    cluster_ids: Sequence[Hashable] | np.ndarray,
    action_count: int,
    *,
    repetitions: int = DEFAULT_DERANGEMENT_REPETITIONS,
    seed: int = DEFAULT_DERANGEMENT_SEED,
) -> np.ndarray:
    """Return fixed ``[repeat, action, recipient]`` donor indices.

    Every action is shuffled only within its own score column.  A donor is
    always from a different cluster/episode than the recipient.  The algorithm
    groups cluster members, independently randomizes group/member order for each
    action and repetition, then rotates by the largest group size.  Such a
    derangement exists exactly when no cluster owns more than half the roots.
    """

    if action_count <= 0:
        raise QualificationMetricsError("action_count must be positive")
    if repetitions <= 0:
        raise QualificationMetricsError("repetitions must be positive")
    if seed < 0:
        raise QualificationMetricsError("seed must be nonnegative")
    encoded, groups = _encoded_clusters(cluster_ids)
    root_count = len(encoded)
    largest_group = max(len(group) for group in groups)
    if largest_group * 2 > root_count:
        raise QualificationMetricsError(
            "cross-episode derangement is impossible because one cluster owns "
            "more than half of the roots"
        )

    result = np.empty((repetitions, action_count, root_count), dtype=np.int64)
    for repetition in range(repetitions):
        for action_position in range(action_count):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, repetition, action_position])
            )
            group_order = rng.permutation(len(groups))
            recipient_chunks = [rng.permutation(groups[index]) for index in group_order]
            recipient_order = np.concatenate(recipient_chunks)
            donor_order = np.roll(recipient_order, -largest_group)
            donors = np.empty(root_count, dtype=np.int64)
            donors[recipient_order] = donor_order
            if np.any(encoded[donors] == encoded):
                raise AssertionError("constructed derangement crossed no episode boundary")
            result[repetition, action_position] = donors
    return result


def fixed_cross_episode_derangements(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    cluster_ids: Sequence[Hashable] | np.ndarray,
    *,
    repetitions: int = DEFAULT_DERANGEMENT_REPETITIONS,
    seed: int = DEFAULT_DERANGEMENT_SEED,
) -> np.ndarray:
    """Apply the fixed donor indices and return ``[repeat, root, action]`` scores."""

    probability_table = _probability_table(probabilities, "probabilities")
    indices = fixed_cross_episode_derangement_indices(
        cluster_ids,
        probability_table.shape[1],
        repetitions=repetitions,
        seed=seed,
    )
    shuffled = np.empty(
        (repetitions, probability_table.shape[0], probability_table.shape[1]),
        dtype=np.float64,
    )
    for repetition in range(repetitions):
        for action_position in range(probability_table.shape[1]):
            shuffled[repetition, :, action_position] = probability_table[
                indices[repetition, action_position], action_position
            ]
    return shuffled


def evaluate_cross_episode_derangements(
    targets: Sequence[Sequence[int | float]] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    cluster_ids: Sequence[Hashable] | np.ndarray,
    *,
    action_ids: Sequence[int] | np.ndarray | None = None,
    repetitions: int = DEFAULT_DERANGEMENT_REPETITIONS,
    seed: int = DEFAULT_DERANGEMENT_SEED,
    ece_bins: int = 10,
) -> CrossEpisodeDerangementReport:
    """Measure how much state signal is destroyed by cross-episode shuffling."""

    target_table, probability_table = _all_action_inputs(targets, probabilities)
    semantic_actions = _semantic_action_ids(action_ids, target_table.shape[1])
    if len(_cluster_vector(cluster_ids)) != target_table.shape[0]:
        raise QualificationMetricsError("cluster_ids length must equal the root count")
    real = all_action_probability_metrics(
        target_table,
        probability_table,
        action_ids=semantic_actions,
        ece_bins=ece_bins,
    )
    shuffled_tables = fixed_cross_episode_derangements(
        probability_table,
        cluster_ids,
        repetitions=repetitions,
        seed=seed,
    )
    shuffled = tuple(
        all_action_probability_metrics(
            target_table,
            table,
            action_ids=semantic_actions,
            ece_bins=ece_bins,
        )
        for table in shuffled_tables
    )
    median_bce = float(np.median([entry.aggregate.bce for entry in shuffled]))
    median_brier = float(np.median([entry.aggregate.brier for entry in shuffled]))
    median_auc = float(np.median([entry.aggregate.roc_auc for entry in shuffled]))
    per_action = tuple(
        ActionDerangementSummary(
            action_id=action_id,
            real_bce=real.per_action[column].metrics.bce,
            median_shuffled_bce=float(
                np.median([entry.per_action[column].metrics.bce for entry in shuffled])
            ),
            real_brier=real.per_action[column].metrics.brier,
            median_shuffled_brier=float(
                np.median([entry.per_action[column].metrics.brier for entry in shuffled])
            ),
            real_roc_auc=real.per_action[column].metrics.roc_auc,
            median_shuffled_roc_auc=float(
                np.median([entry.per_action[column].metrics.roc_auc for entry in shuffled])
            ),
            roc_auc_drop=(
                real.per_action[column].metrics.roc_auc
                - float(
                    np.median(
                        [entry.per_action[column].metrics.roc_auc for entry in shuffled]
                    )
                )
            ),
        )
        for column, action_id in enumerate(semantic_actions)
    )
    return CrossEpisodeDerangementReport(
        repetitions=repetitions,
        seed=seed,
        real=real,
        shuffled=shuffled,
        median_shuffled_bce=median_bce,
        median_shuffled_brier=median_brier,
        median_shuffled_roc_auc=median_auc,
        aggregate_roc_auc_drop=real.aggregate.roc_auc - median_auc,
        per_action=per_action,
    )


def clustered_improvement_bootstrap(
    targets: Sequence[int | float] | Sequence[Sequence[int | float]] | np.ndarray,
    model_probabilities: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    baseline_probabilities: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    cluster_ids: Sequence[Hashable] | np.ndarray,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> ClusteredImprovementBootstrap:
    """Bootstrap baseline-minus-model BCE/Brier, resampling whole clusters.

    Inputs may be one-dimensional factual rows or two-dimensional all-action
    tables.  For a table, ``cluster_ids`` still has one entry per root; every
    action branch from that root remains in the same resampled cluster.  Repeated
    root IDs can bind an entire episode into one cluster.
    """

    if resamples <= 0:
        raise QualificationMetricsError("resamples must be positive")
    if seed < 0:
        raise QualificationMetricsError("seed must be nonnegative")
    if not 0.5 < confidence_level < 1.0:
        raise QualificationMetricsError("confidence_level must be between 0.5 and 1.0")

    target_array = np.asarray(targets, dtype=np.float64)
    model_array = np.asarray(model_probabilities, dtype=np.float64)
    baseline_array = np.asarray(baseline_probabilities, dtype=np.float64)
    if target_array.ndim not in (1, 2):
        raise QualificationMetricsError("bootstrap targets must be a vector or table")
    if target_array.shape != model_array.shape or target_array.shape != baseline_array.shape:
        raise QualificationMetricsError("bootstrap target/probability shapes differ")
    _validate_binary_values(target_array, "targets")
    _validate_probability_values(model_array, "model_probabilities")
    _validate_probability_values(baseline_array, "baseline_probabilities")
    if target_array.shape[0] == 0:
        raise QualificationMetricsError("bootstrap inputs cannot be empty")

    encoded_roots, groups = _encoded_clusters(cluster_ids)
    if len(encoded_roots) != target_array.shape[0]:
        raise QualificationMetricsError("cluster_ids length must equal the root count")
    cluster_count = len(groups)
    if cluster_count < 2:
        raise QualificationMetricsError("clustered bootstrap requires at least two clusters")

    if target_array.ndim == 2:
        width = target_array.shape[1]
        observation_clusters = np.repeat(encoded_roots, width)
    else:
        observation_clusters = encoded_roots
    flat_targets = target_array.reshape(-1)
    flat_model = model_array.reshape(-1)
    flat_baseline = baseline_array.reshape(-1)
    bce_delta = _binary_cross_entropy_losses(
        flat_targets, flat_baseline
    ) - _binary_cross_entropy_losses(flat_targets, flat_model)
    brier_delta = np.square(flat_baseline - flat_targets) - np.square(
        flat_model - flat_targets
    )
    cluster_observations = np.bincount(
        observation_clusters, minlength=cluster_count
    ).astype(np.float64)
    cluster_bce_sums = np.bincount(
        observation_clusters, weights=bce_delta, minlength=cluster_count
    )
    cluster_brier_sums = np.bincount(
        observation_clusters, weights=brier_delta, minlength=cluster_count
    )

    rng = np.random.default_rng(seed)
    bce_replicates = np.empty(resamples, dtype=np.float64)
    brier_replicates = np.empty(resamples, dtype=np.float64)
    chunk_size = 256
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        draws = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        sampled_counts = np.sum(cluster_observations[draws], axis=1)
        bce_replicates[start:stop] = np.sum(cluster_bce_sums[draws], axis=1) / sampled_counts
        brier_replicates[start:stop] = (
            np.sum(cluster_brier_sums[draws], axis=1) / sampled_counts
        )

    lower_quantile = 1.0 - confidence_level
    digest_values = np.column_stack((bce_replicates, brier_replicates)).astype(
        "<f8", copy=False
    )
    return ClusteredImprovementBootstrap(
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
        cluster_count=cluster_count,
        observation_count=len(flat_targets),
        observed_bce_improvement=float(np.mean(bce_delta)),
        bce_improvement_lower_bound=float(
            np.quantile(bce_replicates, lower_quantile, method="linear")
        ),
        observed_brier_improvement=float(np.mean(brier_delta)),
        brier_improvement_lower_bound=float(
            np.quantile(brier_replicates, lower_quantile, method="linear")
        ),
        replicate_sha256=sha256(digest_values.tobytes(order="C")).hexdigest(),
    )


def _all_action_inputs(
    targets: Sequence[Sequence[int | float]] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_table = _binary_target_table(targets, "targets")
    probability_table = _probability_table(probabilities, "probabilities")
    if target_table.shape != probability_table.shape:
        raise QualificationMetricsError("target and probability table shapes differ")
    return target_table, probability_table


def _binary_inputs(
    targets: Sequence[int | float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_array = _binary_target_vector(targets, "targets")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim != 1:
        raise QualificationMetricsError("probabilities must be one-dimensional")
    if target_array.shape != probability_array.shape:
        raise QualificationMetricsError("target and probability vector shapes differ")
    _validate_probability_values(probability_array, "probabilities")
    return target_array, probability_array


def _binary_target_vector(
    values: Sequence[int | float] | np.ndarray,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0:
        raise QualificationMetricsError(f"{name} must be a nonempty vector")
    _validate_binary_values(result, name)
    return result


def _binary_target_table(
    values: Sequence[Sequence[int | float]] | np.ndarray,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise QualificationMetricsError(f"{name} must be a nonempty two-dimensional table")
    _validate_binary_values(result, name)
    return result


def _probability_table(
    values: Sequence[Sequence[float]] | np.ndarray,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise QualificationMetricsError(f"{name} must be a nonempty two-dimensional table")
    _validate_probability_values(result, name)
    return result


def _validate_binary_values(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise QualificationMetricsError(f"{name} contains a non-finite value")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise QualificationMetricsError(f"{name} must contain only binary 0/1 outcomes")


def _validate_probability_values(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise QualificationMetricsError(f"{name} contains a non-finite value")
    if not np.all((0.0 <= values) & (values <= 1.0)):
        raise QualificationMetricsError(f"{name} must contain probabilities in [0, 1]")


def _semantic_action_ids(
    action_ids: Sequence[int] | np.ndarray | None,
    action_count: int,
) -> tuple[int, ...]:
    if action_ids is None:
        return tuple(range(action_count))
    result = _action_vector(action_ids, "action_ids")
    if len(result) != action_count:
        raise QualificationMetricsError("action_ids length does not match the action width")
    semantic = tuple(int(value) for value in result)
    if len(set(semantic)) != len(semantic):
        raise QualificationMetricsError("action_ids must be unique")
    return semantic


def _action_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) == 0:
        raise QualificationMetricsError(f"{name} must be a nonempty vector")
    if not np.issubdtype(raw.dtype, np.integer):
        raise QualificationMetricsError(f"{name} must contain integer semantic IDs")
    return raw.astype(np.int64, copy=False)


def _cluster_vector(values: Sequence[Hashable] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=object)
    if result.ndim != 1 or len(result) == 0:
        raise QualificationMetricsError("cluster_ids must be a nonempty vector")
    return result


def _encoded_clusters(
    values: Sequence[Hashable] | np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    clusters = _cluster_vector(values)
    positions: dict[Hashable, int] = {}
    encoded = np.empty(len(clusters), dtype=np.int64)
    members: list[list[int]] = []
    for row, raw_value in enumerate(clusters):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        try:
            position = positions.get(value)
        except TypeError as error:
            raise QualificationMetricsError("cluster_ids must be hashable") from error
        if position is None:
            position = len(members)
            positions[value] = position
            members.append([])
        encoded[row] = position
        members[position].append(row)
    groups = tuple(np.asarray(group, dtype=np.int64) for group in members)
    return encoded, groups


def _binary_cross_entropy_losses(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, _BCE_EPSILON, 1.0 - _BCE_EPSILON)
    return -(targets * np.log(clipped) + (1.0 - targets) * np.log1p(-clipped))


def _roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""

    order = np.argsort(probabilities, kind="stable")
    sorted_scores = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(probabilities):
        stop = start + 1
        while stop < len(probabilities) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive = targets == 1.0
    positive_count = int(np.sum(positive))
    negative_count = len(targets) - positive_count
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _average_precision(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Threshold-grouped average precision with deterministic stable sorting."""

    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    sorted_scores = probabilities[order]
    true_positives = np.cumsum(sorted_targets)
    false_positives = np.cumsum(1.0 - sorted_targets)
    threshold_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    precision = true_positives[threshold_ends] / (
        true_positives[threshold_ends] + false_positives[threshold_ends]
    )
    recall = true_positives[threshold_ends] / float(np.sum(targets))
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision))


def _equal_mass_ece(targets: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    order = np.argsort(probabilities, kind="stable")
    total = len(targets)
    ece = 0.0
    for indices in np.array_split(order, bins):
        weight = len(indices) / total
        ece += weight * abs(float(np.mean(probabilities[indices]) - np.mean(targets[indices])))
    return ece
