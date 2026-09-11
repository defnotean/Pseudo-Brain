from __future__ import annotations

import os
import inspect
import json
import re
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Mapping
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


class _Approx:
    def __init__(self, expected: float, *, rel: float | None, abs: float | None) -> None:
        self.expected = float(expected)
        self.rel = 1.0e-12 if rel is None else rel
        self.abs = 1.0e-12 if abs is None else abs

    def __eq__(self, observed: object) -> bool:
        return bool(
            np.isclose(
                float(observed),
                self.expected,
                rtol=self.rel,
                atol=self.abs,
            )
        )


class _Raises:
    def __init__(self, error: type[BaseException], match: str | None) -> None:
        self.error = error
        self.match = match

    def __enter__(self) -> None:
        return None

    def __exit__(self, error_type, error, _traceback) -> bool:
        if error_type is None:
            raise AssertionError(f"expected {self.error.__name__} to be raised")
        if not issubclass(error_type, self.error):
            return False
        if self.match is not None and re.search(self.match, str(error)) is None:
            raise AssertionError(
                f"exception {error!r} did not match regular expression {self.match!r}"
            )
        return True


def _parameterize(
    names: str | tuple[str, ...],
    values: tuple[tuple[object, ...], ...],
):
    def decorate(function):
        frozen_names = (
            tuple(name.strip() for name in names.split(","))
            if isinstance(names, str)
            else tuple(names)
        )
        function._unittest_parametrize = (frozen_names, values)
        return function

    return decorate


def _approx(
    expected: float,
    *,
    rel: float | None = None,
    abs: float | None = None,
) -> _Approx:
    return _Approx(expected, rel=rel, abs=abs)


def _raises(error: type[BaseException], *, match: str | None = None) -> _Raises:
    return _Raises(error, match)


class _MonkeyPatch:
    def __init__(self) -> None:
        self._patchers: list[object] = []

    def setattr(self, target: object, name: str, value: object) -> None:
        patcher = patch.object(target, name, value)
        patcher.start()
        self._patchers.append(patcher)

    def undo(self) -> None:
        for patcher in reversed(self._patchers):
            patcher.stop()


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(
    os.environ.get(
        "PSEUDO_BRAIN_CANONICAL_ROOT",
        str(ROOT),
    )
)
sys.path[:0] = [
    str(CANONICAL_ROOT / "scripts"),
    str(ROOT / "scripts"),
    str(CANONICAL_ROOT / "src"),
    str(ROOT / "src"),
]

import v21k_nz_dualcal_diagnostic_v1 as diagnostic


def _provenance(observations: int) -> diagnostic.TrainCalibrationProvenance:
    return diagnostic.TrainCalibrationProvenance(
        source_namespace="maze_chase.pb21k.test.v1",
        source_split="TRAIN",
        source_partition="TRAIN-CAL",
        calibration_group_ids=tuple(f"cal:{index}" for index in range(observations)),
        upstream_model_fit_group_ids=("fit:sealed",),
        upstream_checkpoint_sha256="0" * 64,
        dataset_manifest_sha256="1" * 64,
        source_bundle_sha256="2" * 64,
        partition_algorithm=diagnostic.PARTITION_ALGORITHM,
    )


def _synthetic_cal_tables(
    roots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roots = diagnostic.ROOTS_PER_CAL if roots is None else roots
    actions = np.arange(roots, dtype=np.int64) % diagnostic.ACTION_COUNT
    phase = (np.arange(roots, dtype=np.int64) // diagnostic.ACTION_COUNT) % 2
    targets = np.empty((roots, diagnostic.ACTION_COUNT), dtype=np.float64)
    logits = np.empty_like(targets)
    for action in range(diagnostic.ACTION_COUNT):
        target = ((phase + action) % 2).astype(np.float64)
        targets[:, action] = target
        logits[:, action] = np.where(target == 1.0, 0.45, -0.15) + 0.35
    return logits, targets, actions


def _minimal_evidence_arrays() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, dtype in diagnostic.EVIDENCE_DTYPES.items():
        if name == "schema_version":
            value = np.asarray(1, dtype=dtype)
        elif np.dtype(dtype).kind == "S":
            value = np.asarray([b"x"], dtype=dtype)
        else:
            value = np.asarray([0], dtype=dtype)
        result[name] = value
    return result


def _valid_small_evidence(
    monkeypatch,
) -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, object]], dict[str, str]
]:
    monkeypatch.setattr(diagnostic, "ROOTS_PER_FIT", 100)
    monkeypatch.setattr(diagnostic, "ROOTS_PER_CAL", 100)
    monkeypatch.setattr(diagnostic, "DEV_EPISODES", 4)
    monkeypatch.setattr(diagnostic, "ROOTS_PER_EPISODE", 2)
    monkeypatch.setattr(diagnostic, "DEV_ROOTS", 8)
    monkeypatch.setattr(diagnostic, "BOOTSTRAP_RESAMPLES", 7)
    monkeypatch.setattr(diagnostic, "DERANGEMENT_REPETITIONS", 2)
    logits, targets, actions = _synthetic_cal_tables(100)
    tape = SimpleNamespace(hazard_targets=targets, factual_actions=actions)
    provenance = _provenance(logits.size)
    aa, _ = diagnostic.fit_aa_calibrator(logits, tape, provenance)
    mix, _ = diagnostic.fit_mix_calibrator(logits, tape, provenance)
    calibrators = (aa, mix)
    cells = tuple(
        f"{fit}/init-{seed}"
        for fit in diagnostic.FIT_COHORTS
        for seed in diagnostic.INIT_SEEDS
    )
    dev_targets = np.tile(np.asarray([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.float64), (4, 1))
    dev_actions = np.arange(8, dtype=np.int64) % 5
    base_dev_logits = np.linspace(-1.0, 1.0, 40, dtype=np.float64).reshape(8, 5)
    raw_logits = np.stack(
        [
            np.stack([base_dev_logits + 0.01 * arm for arm in range(5)])
            for _ in cells
        ]
    )
    raw_probabilities = diagnostic._sigmoid(raw_logits)
    nz_index = diagnostic.ARMS.index("NZ")
    calibrated_logits = np.stack([
        (
            torch.from_numpy(raw_logits[:, nz_index])
            * torch.as_tensor(calibrator.scales, dtype=torch.float64)[None, None, :]
            + torch.as_tensor(calibrator.biases, dtype=torch.float64)[None, None, :]
        ).numpy()
        for calibrator in calibrators
    ])
    calibrated_probabilities = diagnostic._sigmoid(calibrated_logits)
    cluster_ids = ("DEV:episode:0", "DEV:episode:1", "DEV:episode:2", "DEV:episode:3")
    cluster_ordinal = np.repeat(np.arange(4, dtype=np.int32), 2)
    arrays = {
        "schema_version": np.asarray(1, dtype=np.uint16),
        "action_ids": np.arange(5, dtype=np.int16),
        "fit_ids": np.asarray(diagnostic.FIT_COHORTS, dtype="S2"),
        "cal_ids": np.asarray(diagnostic.CAL_COHORTS, dtype="S2"),
        "cell_ids": np.asarray(cells, dtype="S16"),
        "cell_fit_index": np.repeat(np.arange(3, dtype=np.int16), 3),
        "cell_cal_index": np.repeat(np.arange(3, dtype=np.int16), 3),
        "cell_init_seed": np.tile(np.asarray(diagnostic.INIT_SEEDS, dtype=np.int32), 3),
        "arm_ids": np.asarray(diagnostic.ARMS, dtype="S3"),
        "calibrator_ids": np.asarray(diagnostic.CALIBRATORS, dtype="S3"),
        "fit_targets": np.stack([targets] * 3),
        "fit_actions": np.stack([actions] * 3),
        "fit_root_ids": np.asarray(
            [[f"{label}:root:{root}" for root in range(100)] for label in diagnostic.FIT_COHORTS],
            dtype="S64",
        ),
        "cal_targets": np.stack([targets] * 3),
        "cal_actions": np.stack([actions] * 3),
        "cal_root_ids": np.asarray(
            [[f"{label}:root:{root}" for root in range(100)] for label in diagnostic.CAL_COHORTS],
            dtype="S64",
        ),
        "cal_nz_raw_logits": np.stack([logits] * 9),
        "calibrator_scales": np.asarray(
            [[calibrator.scales for _ in cells] for calibrator in calibrators],
            dtype=np.float64,
        ),
        "calibrator_biases": np.asarray(
            [[calibrator.biases for _ in cells] for calibrator in calibrators],
            dtype=np.float64,
        ),
        "calibrator_accepted": np.asarray(
            [[calibrator.accepted for _ in cells] for calibrator in calibrators],
            dtype=np.bool_,
        ),
        "mix_proposed_scales": np.asarray([mix.proposed_scales] * 9),
        "mix_proposed_biases": np.asarray([mix.proposed_biases] * 9),
        "mix_per_action_accepted": np.asarray(
            [[value.accepted for value in mix.per_action]] * 9, dtype=np.bool_
        ),
        "dev_targets": dev_targets,
        "dev_actions": dev_actions,
        "dev_root_ids": np.asarray(
            [f"DEV:root:{root}" for root in range(8)], dtype="S64"
        ),
        "dev_cluster_ordinal": cluster_ordinal,
        "dev_cluster_ids": np.asarray(cluster_ids, dtype="S64"),
        "dev_raw_logits": raw_logits,
        "dev_raw_probabilities": raw_probabilities,
        "nz_calibrated_logits": calibrated_logits,
        "nz_calibrated_probabilities": calibrated_probabilities,
        "bootstrap_indices": diagnostic.bootstrap_index_matrix(
            resamples=7, seed=82_042, groups=4
        ),
        "episode_derangements": diagnostic.episode_derangement_indices(4),
    }
    arrays = {
        name: diagnostic._canonical_evidence_array(value) for name, value in arrays.items()
    }
    expanded = tuple(cluster_ids[int(value)] for value in cluster_ordinal)
    episode_digest = diagnostic.strict._ordered_sequence_digest(expanded)
    source_evidence: dict[str, dict[str, object]] = {}
    for index, label in enumerate(diagnostic.FIT_COHORTS):
        source_evidence[label] = {
            "label": label,
            "roots": len(arrays["fit_targets"][index]),
            "actions_per_root": diagnostic.ACTION_COUNT,
            "target_sha256": diagnostic._array_sha256(arrays["fit_targets"][index]),
            "factual_action_sha256": diagnostic._array_sha256(arrays["fit_actions"][index]),
            "ordered_root_sha256": diagnostic.strict._ordered_sequence_digest(
                diagnostic._decode_byte_strings(arrays["fit_root_ids"][index])
            ),
        }
    for index, label in enumerate(diagnostic.CAL_COHORTS):
        source_evidence[label] = {
            "label": label,
            "roots": len(arrays["cal_targets"][index]),
            "actions_per_root": diagnostic.ACTION_COUNT,
            "target_sha256": diagnostic._array_sha256(arrays["cal_targets"][index]),
            "factual_action_sha256": diagnostic._array_sha256(arrays["cal_actions"][index]),
            "ordered_root_sha256": diagnostic.strict._ordered_sequence_digest(
                diagnostic._decode_byte_strings(arrays["cal_root_ids"][index])
            ),
        }
    source_evidence["DEV"] = {
        "label": "DEV",
        "roots": len(arrays["dev_targets"]),
        "actions_per_root": diagnostic.ACTION_COUNT,
        "target_sha256": diagnostic._array_sha256(arrays["dev_targets"]),
        "factual_action_sha256": diagnostic._array_sha256(arrays["dev_actions"]),
        "ordered_root_sha256": diagnostic.strict._ordered_sequence_digest(
            diagnostic._decode_byte_strings(arrays["dev_root_ids"])
        ),
        "ordered_episode_sha256": episode_digest,
    }
    cal_raw_hashes = {
        cell: diagnostic._array_sha256(arrays["cal_nz_raw_logits"][index])
        for index, cell in enumerate(cells)
    }
    return arrays, source_evidence, cal_raw_hashes


def test_frozen_geometry_schedule_and_no_nbz() -> None:
    assert diagnostic.ARMS == ("N", "Z", "BZ", "NZ", "U0")
    assert "NBZ" not in diagnostic.ARMS
    assert diagnostic.TOTAL_HEADS == 45
    assert diagnostic.TOTAL_OPTIMIZER_STEPS == 184_320
    assert diagnostic.INIT_SEEDS == (51_042, 52_042, 53_042)
    assert diagnostic.PERMUTATION_SEED == 60_042
    assert diagnostic.BOOTSTRAP_SEED == 82_042
    assert diagnostic.DERANGEMENT_SEED == 83_042
    assert diagnostic.DEV_EPISODES == 768
    assert diagnostic.DEV_ROOTS == 9_216


def test_fresh_ranges_and_central_collision_audit() -> None:
    intervals = {
        value.label: (value.split.value, value.seed_offset, value.seed_offset + value.episodes)
        for value in diagnostic.PARTITION_SPECS
    }
    assert intervals == {
        "F0": ("train", 134_217_728, 134_217_984),
        "C0": ("train", 134_217_984, 134_218_240),
        "F1": ("train", 134_218_240, 134_218_496),
        "C1": ("train", 134_218_496, 134_218_752),
        "F2": ("train", 134_218_752, 134_219_008),
        "C2": ("train", 134_219_008, 134_219_264),
        "DEV": ("validation", 150_994_944, 150_995_712),
    }
    audit = diagnostic.assert_no_range_collisions()
    assert audit["passed"] is True
    owners = {value["owner"] for value in audit["records"]}
    assert "V2.1 CPU-QUAL reserved" in owners
    assert "wallclock repetition 0" in owners
    assert "PLAY-QUAL sealed" in owners
    assert "RCQ-v3 TEST sealed" in owners
    assert "PB21J DEV" in owners
    assert "PB21K DEV" in owners


def test_partition_factory_is_config_only_and_rejects_unregistered_labels() -> None:
    class FakeSource:
        def __init__(self, contract: object) -> None:
            self.contract = contract
            self.manifest_sha256 = diagnostic.maze_chase_dataset_manifest_sha256(
                contract.dataset_config
            )

    values = diagnostic.partition_sources(("F0", "C0"), source_factory=FakeSource)
    assert tuple(values) == ("F0", "C0")
    with _raises(ValueError, match="registered PB21K"):
        diagnostic.partition_sources(("CPU-QUAL",), source_factory=FakeSource)
    with _raises(ValueError, match="registered PB21K"):
        diagnostic.partition_sources(("TEST",), source_factory=FakeSource)


def test_exact_masked_head_lineage_and_nz_pruning() -> None:
    first = diagnostic.initialized_head(diagnostic.INIT_SEEDS[0])
    second = diagnostic.initialized_head(diagnostic.INIT_SEEDS[0])
    assert sum(value.numel() for value in first.parameters()) == 87_961
    assert diagnostic._state_sha256(first) == diagnostic._state_sha256(second)
    with _raises(ValueError, match="not registered"):
        diagnostic.initialized_head(42)
    pruned, mapping = diagnostic.pb21j.prune_head(first, diagnostic.ARM_NZ)
    assert mapping["active_columns"] == list(range(0, 120)) + list(range(240, 360))
    assert mapping["input_width"] == 240
    assert mapping["parameter_count"] == 58_561
    features = np.zeros((7, diagnostic.SUPERSET_WIDTH), dtype=np.float32)
    transfer = diagnostic.pb21j.pruning_equivalence_audit(
        first, pruned, features, diagnostic.ARM_NZ
    )
    assert transfer["batch_size"] == 1
    assert transfer["passed"] is True


def test_superset_rejects_nbz_and_checks_pending_flag() -> None:
    roots = 2
    updater = np.zeros((roots, diagnostic.strict.CAPACITY_STATE_WIDTH), dtype=np.float64)
    updater[:, 365] = 1.0
    tape = diagnostic.pb21j.FreshLiveTape(
        beliefs=np.zeros((roots, 120), dtype=np.float64),
        updater_inputs=updater,
        hazard_targets=np.zeros((roots, 5), dtype=np.float64),
        parent_raw_logits=np.zeros((roots, 5), dtype=np.float64),
        factual_actions=np.asarray([0, 1], dtype=np.int64),
        episode_group_ids=("a", "b"),
        root_state_ids=("r0", "r1"),
        prior_assignment_count=np.zeros(roots, dtype=np.int64),
        prior_disagreement_count=np.zeros(roots, dtype=np.int64),
    )
    assert diagnostic.superset_features(tape, "NZ").shape == (roots, 485)
    with _raises(ValueError, match="forbidden"):
        diagnostic.superset_features(tape, "NBZ")
    tape.updater_inputs[:, 365] = 0.0
    with _raises(RuntimeError, match="pending flag"):
        diagnostic.superset_features(tape, "NZ")


def test_mix_weights_are_equal_domain_within_action() -> None:
    actions = np.asarray([0] * 25 + [1, 2, 3, 4] * 7, dtype=np.int64)
    weights = diagnostic._mix_weights(actions, 0)
    expected = np.full(len(actions), 0.5 / len(actions))
    expected[actions == 0] += 0.5 / 25
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=0.0)
    assert weights.sum() == _approx(1.0, abs=1.0e-14)


def test_mixed_objective_has_exact_once_penalty() -> None:
    logits = torch.tensor([-1.0, 0.25, 2.0, -0.5], dtype=torch.float64)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    scale = torch.tensor(1.7, dtype=torch.float64)
    bias = torch.tensor(-0.3, dtype=torch.float64)
    observed = diagnostic._mixed_objective(logits, targets, weights, scale, bias)
    calibrated = scale * logits + bias
    expected = torch.sum(
        weights * (F.softplus(calibrated) - targets * calibrated)
    ) + 0.5e-6 * ((scale - 1.0).square() + bias.square())
    assert float(observed) == _approx(float(expected), rel=0.0, abs=1.0e-15)


def test_mixed_analytic_gradient_and_hessian_match_finite_differences() -> None:
    logits = torch.linspace(-2.0, 2.0, 40, dtype=torch.float64)
    targets = (torch.arange(40) % 3 == 0).to(torch.float64)
    weights = torch.linspace(1.0, 2.0, 40, dtype=torch.float64)
    weights /= weights.sum()
    scale = 1.3
    bias = -0.2

    def objective(s: float, b: float) -> float:
        return float(
            diagnostic._mixed_objective(
                logits,
                targets,
                weights,
                torch.tensor(s, dtype=torch.float64),
                torch.tensor(b, dtype=torch.float64),
            )
        )

    probability = torch.sigmoid(scale * logits + bias)
    residual = probability - targets
    curvature = probability * (1.0 - probability)
    analytic_gradient = np.asarray(
        [
            float(torch.sum(weights * residual * logits) + 1.0e-6 * (scale - 1.0)),
            float(torch.sum(weights * residual) + 1.0e-6 * bias),
        ]
    )
    analytic_hessian = np.asarray(
        [
            [
                float(torch.sum(weights * curvature * logits.square()) + 1.0e-6),
                float(torch.sum(weights * curvature * logits)),
            ],
            [
                float(torch.sum(weights * curvature * logits)),
                float(torch.sum(weights * curvature) + 1.0e-6),
            ],
        ]
    )
    epsilon = 1.0e-4
    finite_gradient = np.asarray(
        [
            (objective(scale + epsilon, bias) - objective(scale - epsilon, bias))
            / (2.0 * epsilon),
            (objective(scale, bias + epsilon) - objective(scale, bias - epsilon))
            / (2.0 * epsilon),
        ]
    )
    finite_hessian = np.asarray(
        [
            [
                (objective(scale + epsilon, bias) - 2.0 * objective(scale, bias) + objective(scale - epsilon, bias))
                / epsilon**2,
                (
                    objective(scale + epsilon, bias + epsilon)
                    - objective(scale + epsilon, bias - epsilon)
                    - objective(scale - epsilon, bias + epsilon)
                    + objective(scale - epsilon, bias - epsilon)
                )
                / (4.0 * epsilon**2),
            ],
            [0.0, (objective(scale, bias + epsilon) - 2.0 * objective(scale, bias) + objective(scale, bias - epsilon)) / epsilon**2],
        ]
    )
    finite_hessian[1, 0] = finite_hessian[0, 1]
    np.testing.assert_allclose(finite_gradient, analytic_gradient, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(finite_hessian, analytic_hessian, rtol=0.0, atol=2.0e-7)


def test_mixed_optimizer_is_deterministic_positive_and_improves() -> None:
    logits, targets, actions = _synthetic_cal_tables()
    weights = diagnostic._mix_weights(actions, 3)
    first = diagnostic._fit_one_mixed_action(logits[:, 3], targets[:, 3], weights)
    second = diagnostic._fit_one_mixed_action(logits[:, 3], targets[:, 3], weights)
    assert first == second
    assert first[0] >= diagnostic.CALIBRATION_MINIMUM_SCALE
    identity = float(
        diagnostic._mixed_objective(
            torch.from_numpy(logits[:, 3]),
            torch.from_numpy(targets[:, 3]),
            torch.from_numpy(weights),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )
    )
    assert first[2]["objective"] < identity


def test_mixed_optimizer_honors_scale_boundary_kkt() -> None:
    logits = np.tile(np.asarray([-4.0, 4.0], dtype=np.float64), 40)
    targets = np.tile(np.asarray([1.0, 0.0], dtype=np.float64), 40)
    weights = np.full(len(logits), 1.0 / len(logits), dtype=np.float64)
    scale, bias, report = diagnostic._fit_one_mixed_action(logits, targets, weights)
    assert scale == _approx(diagnostic.CALIBRATION_MINIMUM_SCALE, abs=1.0e-12)
    assert bias == _approx(0.0, abs=1.0e-9)
    assert report["termination"] == "projected_gradient_KKT"


def test_mixed_optimizer_iteration_and_line_search_exhaustion_fail_closed(monkeypatch) -> None:
    logits, targets, actions = _synthetic_cal_tables(100)
    weights = diagnostic._mix_weights(actions, 0)
    monkeypatch.setattr(diagnostic, "CALIBRATION_MAX_ITERATIONS", 1)
    with _raises(RuntimeError, match="maximum iterations"):
        diagnostic._fit_one_mixed_action(logits[:, 0], targets[:, 0], weights)
    monkeypatch.setattr(diagnostic, "CALIBRATION_MAX_ITERATIONS", 100)
    original = diagnostic._mixed_objective
    calls = 0

    def nonfinite_candidates(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(*args, **kwargs)
        return torch.tensor(float("inf"), dtype=torch.float64)

    monkeypatch.setattr(diagnostic, "_mixed_objective", nonfinite_candidates)
    with _raises(RuntimeError, match="line search exhausted"):
        diagnostic._fit_one_mixed_action(logits[:, 0], targets[:, 0], weights)


def test_mix_support_and_provenance_fail_closed() -> None:
    logits, targets, actions = _synthetic_cal_tables()
    tape = SimpleNamespace(hazard_targets=targets, factual_actions=actions)
    too_few = actions.copy()
    too_few[too_few == 4] = 0
    too_few[:19] = 4
    with _raises(ValueError, match="insufficient factual"):
        diagnostic.fit_mix_calibrator(
            logits,
            SimpleNamespace(hazard_targets=targets, factual_actions=too_few),
            _provenance(logits.size),
        )
    one_class = targets.copy()
    one_class[actions == 3, 3] = 0.0
    with _raises(ValueError, match="class support"):
        diagnostic.fit_mix_calibrator(
            logits,
            SimpleNamespace(hazard_targets=one_class, factual_actions=actions),
            _provenance(logits.size),
        )
    with _raises(ValueError, match="identify every observation"):
        diagnostic.fit_mix_calibrator(logits, tape, _provenance(logits.size - 1))
    non_train = diagnostic.TrainCalibrationProvenance(
        **{
            **_provenance(logits.size).__dict__,
            "source_split": "VALIDATION",
        }
    )
    with _raises(ValueError, match="TRAIN split"):
        diagnostic.fit_mix_calibrator(logits, tape, non_train)
    overlap = diagnostic.TrainCalibrationProvenance(
        **{
            **_provenance(logits.size).__dict__,
            "upstream_model_fit_group_ids": ("cal:0",),
        }
    )
    with _raises(ValueError, match="overlap"):
        diagnostic.fit_mix_calibrator(logits, tape, overlap)


def test_aa_is_exact_shared_fitter_and_both_reports_bind_same_raw_logits() -> None:
    logits, targets, actions = _synthetic_cal_tables()
    tape = SimpleNamespace(hazard_targets=targets, factual_actions=actions)
    provenance = _provenance(logits.size)
    aa, aa_report = diagnostic.fit_aa_calibrator(logits, tape, provenance)
    action_ids = np.broadcast_to(np.arange(5, dtype=np.int64), logits.shape)
    direct = diagnostic.fit_train_only_per_action_affine(
        torch.from_numpy(logits.reshape(-1)),
        torch.from_numpy(targets.reshape(-1)),
        torch.from_numpy(action_ids.reshape(-1)),
        provenance=provenance,
        action_count=5,
        l2_regularization=diagnostic.CALIBRATION_L2,
        minimum_scale=diagnostic.CALIBRATION_MINIMUM_SCALE,
        minimum_examples_per_action=diagnostic.CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
        minimum_class_examples=diagnostic.CALIBRATION_MINIMUM_CLASS_EXAMPLES,
        max_iterations=diagnostic.CALIBRATION_MAX_ITERATIONS,
        tolerance=diagnostic.CALIBRATION_TOLERANCE,
        fit_mode="per_action_affine",
    )
    assert aa.export_parameters() == direct.export_parameters()
    _, mix_report = diagnostic.fit_mix_calibrator(logits, tape, provenance)
    assert aa_report["raw_CAL_logit_sha256"] == mix_report["raw_CAL_logit_sha256"]


def test_mix_calibrator_accepts_only_strict_all_and_factual_improvement() -> None:
    logits, targets, actions = _synthetic_cal_tables()
    tape = SimpleNamespace(hazard_targets=targets, factual_actions=actions)
    calibrator, report = diagnostic.fit_mix_calibrator(
        logits, tape, _provenance(logits.size)
    )
    assert report["accepted"] is True
    assert calibrator.accepted is True
    assert all(value.accepted for value in calibrator.per_action)
    assert calibrator.scales == calibrator.proposed_scales
    assert all(value > 0.0 for value in calibrator.scales)


def test_mix_rejection_applies_global_identity_but_retains_proposal(monkeypatch) -> None:
    logits, targets, actions = _synthetic_cal_tables()
    tape = SimpleNamespace(hazard_targets=targets, factual_actions=actions)

    def bad_fit(*_args: object, **_kwargs: object) -> tuple[float, float, dict[str, object]]:
        return 1.0, 10.0, {"iterations": 1, "termination": "synthetic", "objective": 99.0}

    monkeypatch.setattr(diagnostic, "_fit_one_mixed_action", bad_fit)
    calibrator, report = diagnostic.fit_mix_calibrator(
        logits, tape, _provenance(logits.size)
    )
    assert report["accepted"] is False
    assert calibrator.accepted is False
    assert calibrator.scales == (1.0,) * 5
    assert calibrator.biases == (0.0,) * 5
    assert calibrator.proposed_biases == (10.0,) * 5
    raw = torch.from_numpy(logits[:3])
    ids = torch.arange(5, dtype=torch.long).expand(3, -1)
    assert torch.equal(calibrator.transform_all_actions(raw, ids), raw)


def test_dual_domain_transform_requires_integer_semantic_action_ids() -> None:
    logits, targets, actions = _synthetic_cal_tables()
    tape = SimpleNamespace(hazard_targets=targets, factual_actions=actions)
    calibrator, _ = diagnostic.fit_mix_calibrator(
        logits, tape, _provenance(logits.size)
    )
    values = torch.from_numpy(logits[:2])
    with _raises(ValueError, match="integer dtype"):
        calibrator.transform_all_actions(values, torch.zeros_like(values))


def test_candidate_replica_gate_requires_every_frozen_check(monkeypatch) -> None:
    monkeypatch.setattr(diagnostic, "DEV_EPISODES", 2)
    monkeypatch.setattr(diagnostic, "ROOTS_PER_EPISODE", 5)
    fit_targets = np.zeros((10, diagnostic.ACTION_COUNT), dtype=np.float64)
    fit_actions = np.arange(10, dtype=np.int64) % diagnostic.ACTION_COUNT
    dev_targets = np.zeros_like(fit_targets)
    dev_actions = fit_actions.copy()
    raw_scores = np.zeros_like(dev_targets)
    raw_probability = np.full_like(dev_targets, 0.5)
    calibrated_scores = np.zeros_like(dev_targets)
    calibrated_probability = np.full_like(dev_targets, 0.4)
    cluster_ordinal = np.repeat(np.arange(2, dtype=np.int32), 5)
    sampled = np.tile(np.arange(2, dtype=np.int32), (7, 1))
    derangements = np.asarray([[1, 0]], dtype=np.int32)
    check_names = (
        "MIX_CAL_accepted",
        "calibrated_DEV_BCE_not_worse_than_raw",
        "all_action_BCE_beats_FIT_prior_ratio",
        "all_action_Brier_beats_FIT_prior_ratio",
        "aggregate_ECE",
        "every_action_probability_gate",
        "calibrated_ranking",
        "FIT_factual_both_classes_every_action",
        "DEV_factual_both_classes_every_action",
        "factual_BCE_strictly_beats_FIT_prior",
        "factual_Brier_strictly_beats_FIT_prior",
        "all_action_BCE_LCB_positive",
        "all_action_Brier_LCB_positive",
        "factual_BCE_LCB_positive",
        "factual_Brier_LCB_positive",
        "joint_derangement",
        "factual_aggregate_ECE",
        "factual_aggregate_absolute_bias",
        "every_factual_action_ECE_and_bias",
        "all_metrics_finite",
    )
    enabled = {name: True for name in check_names}

    baselines = SimpleNamespace(
        all_action=SimpleNamespace(
            probability_table=lambda roots: np.zeros(
                (roots, diagnostic.ACTION_COUNT), dtype=np.float64
            )
        ),
        factual=SimpleNamespace(
            probabilities_for_actions=lambda actions: np.zeros(
                len(actions), dtype=np.float64
            )
        ),
    )

    def all_action_metrics(_targets, probabilities, **_kwargs):
        if probabilities is raw_probability:
            raw_bce = (
                0.2
                if enabled["calibrated_DEV_BCE_not_worse_than_raw"]
                else 0.05
            )
            return SimpleNamespace(as_dict=lambda: {"aggregate": {"bce": raw_bce}})
        action_ece = (
            0.01
            if enabled["every_action_probability_gate"]
            else diagnostic.pb21j.MAXIMUM_PER_ACTION_ECE + 0.01
        )
        report = {
            "aggregate": {
                "bce": 0.1,
                "brier": 0.1,
                "ece_equal_mass": (
                    0.01
                    if enabled["aggregate_ECE"]
                    else diagnostic.pb21j.MAXIMUM_AGGREGATE_ECE + 0.01
                ),
            },
            "per_action": [
                {
                    "metrics": {
                        "ece_equal_mass": action_ece,
                        "calibration_bias": 0.01,
                        "pr_auc": 1.0,
                        "prevalence": 0.0,
                        "brier": 0.1,
                    }
                }
                for _ in range(diagnostic.ACTION_COUNT)
            ],
        }
        return SimpleNamespace(as_dict=lambda: report)

    def binary_metrics(targets, _probabilities, **_kwargs):
        if len(targets) == len(dev_targets):
            report = {
                "bce": (
                    0.1
                    if enabled["factual_BCE_strictly_beats_FIT_prior"]
                    else 0.2
                ),
                "brier": (
                    0.1
                    if enabled["factual_Brier_strictly_beats_FIT_prior"]
                    else 0.2
                ),
                "ece_equal_mass": (
                    0.01 if enabled["factual_aggregate_ECE"] else 0.051
                ),
                "calibration_bias": (
                    0.01
                    if enabled["factual_aggregate_absolute_bias"]
                    else 0.051
                ),
            }
        else:
            report = {
                "ece_equal_mass": (
                    0.01
                    if enabled["every_factual_action_ECE_and_bias"]
                    else 0.076
                ),
                "calibration_bias": 0.01,
            }
        return SimpleNamespace(as_dict=lambda: report)

    def baseline_metrics(*_args, **_kwargs):
        report = {
            "all_action": {
                "aggregate": {
                    "roc_auc": 0.5,
                    "bce": (
                        1.0
                        if enabled["all_action_BCE_beats_FIT_prior_ratio"]
                        else 0.05
                    ),
                    "brier": (
                        1.0
                        if enabled["all_action_Brier_beats_FIT_prior_ratio"]
                        else 0.05
                    ),
                },
                "per_action": [
                    {"metrics": {"brier": 0.2}}
                    for _ in range(diagnostic.ACTION_COUNT)
                ],
            },
            "factual": {"bce": 0.2, "brier": 0.2},
        }
        return SimpleNamespace(as_dict=lambda: report)

    def bootstrap_report(targets, *_args, **_kwargs):
        factual = np.asarray(targets).ndim == 1
        prefix = "factual" if factual else "all_action"
        return {
            "BCE_one_sided_lower_bound": (
                0.1 if enabled[f"{prefix}_BCE_LCB_positive"] else 0.0
            ),
            "Brier_one_sided_lower_bound": (
                0.1 if enabled[f"{prefix}_Brier_LCB_positive"] else 0.0
            ),
        }

    monkeypatch.setattr(diagnostic, "all_action_probability_metrics", all_action_metrics)
    monkeypatch.setattr(diagnostic, "binary_probability_metrics", binary_metrics)
    monkeypatch.setattr(diagnostic, "evaluate_train_fitted_baselines", baseline_metrics)
    monkeypatch.setattr(diagnostic, "proper_score_improvement_bootstrap", bootstrap_report)
    monkeypatch.setattr(
        diagnostic.pb21j,
        "score_ranking_gate",
        lambda *_args, **_kwargs: {
            "passed": enabled["calibrated_ranking"]
            if _args[1] is calibrated_scores
            else True
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_both_classes_per_factual_action",
        lambda targets, _actions: {
            "passed": enabled[
                "FIT_factual_both_classes_every_action"
                if targets is fit_targets
                else "DEV_factual_both_classes_every_action"
            ]
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "joint_derangement_report",
        lambda *_args, **_kwargs: {"passed": enabled["joint_derangement"]},
    )
    monkeypatch.setattr(
        diagnostic.pb21j,
        "_all_numeric_finite",
        lambda _value: enabled["all_metrics_finite"],
    )

    def evaluate():
        return diagnostic.candidate_replica_gate(
            fit_targets=fit_targets,
            fit_actions=fit_actions,
            dev_targets=dev_targets,
            dev_actions=dev_actions,
            cluster_ordinal=cluster_ordinal,
            raw_scores=raw_scores,
            raw_probability=raw_probability,
            calibrated_scores=calibrated_scores,
            calibrated_probability=calibrated_probability,
            baselines=baselines,
            sampled=sampled,
            episode_derangements=derangements,
            calibrator_accepted=enabled["MIX_CAL_accepted"],
        )

    baseline = evaluate()
    assert tuple(baseline["checks"]) == check_names
    assert all(baseline["checks"].values())
    assert baseline["passed"] is True
    for check in check_names:
        enabled[check] = False
        failed = evaluate()
        assert failed["checks"][check] is False, check
        assert failed["passed"] is False, check
        enabled[check] = True


def test_bootstrap_and_derangements_are_deterministic_on_small_geometry() -> None:
    first = diagnostic.bootstrap_index_matrix(resamples=17, seed=82_042, groups=11)
    second = diagnostic.bootstrap_index_matrix(resamples=17, seed=82_042, groups=11)
    assert first.dtype == np.int32
    assert np.array_equal(first, second)
    deranged = diagnostic.episode_derangement_indices(group_count=31)
    identity = np.arange(31, dtype=np.int32)
    assert deranged.shape == (20, 31)
    assert all(not np.any(value == identity) for value in deranged)
    assert all(np.array_equal(np.sort(value), identity) for value in deranged)


def test_primary_iut_uses_strict_thresholds_and_every_cell_guards(monkeypatch) -> None:
    monkeypatch.setattr(diagnostic, "DEV_EPISODES", 2)
    monkeypatch.setattr(diagnostic, "ROOTS_PER_EPISODE", 1)
    monkeypatch.setattr(
        diagnostic.pb21j,
        "_score_auc",
        lambda _targets, scores: float(np.asarray(scores)[0, 0]),
    )
    monkeypatch.setattr(
        diagnostic,
        "_leave_one_auc",
        lambda _targets, scores, _rows: np.full(
            2, float(np.asarray(scores)[0, 0]), dtype=np.float64
        ),
    )
    cells = tuple(f"cell-{index}" for index in range(9))
    targets = np.zeros((2, diagnostic.ACTION_COUNT), dtype=np.float64)
    cluster_ordinal = np.arange(2, dtype=np.int32)
    sampled = np.tile(np.arange(2, dtype=np.int32), (17, 1))
    logits = np.zeros((9, len(diagnostic.ARMS), 2, diagnostic.ACTION_COUNT))
    for arm, value in {"N": 0.4, "NZ": 0.5, "U0": 0.52}.items():
        logits[:, diagnostic.ARMS.index(arm)] = value

    passed = diagnostic.primary_iut_report(
        targets,
        logits,
        cluster_ordinal,
        sampled,
        cell_ids=cells,
        arm_ids=diagnostic.ARMS,
    )
    assert passed["NZ_path_passed"] is True
    assert all(
        comparison["every_cell_point_guard"] is True
        for comparison in passed["comparisons"].values()
    )

    equal_n = logits.copy()
    equal_n[0, diagnostic.ARMS.index("N")] = 0.5
    failed_n = diagnostic.primary_iut_report(
        targets,
        equal_n,
        cluster_ordinal,
        sampled,
        cell_ids=cells,
        arm_ids=diagnostic.ARMS,
    )
    n_comparison = failed_n["comparisons"]["NZ_minus_N"]
    assert n_comparison["grand_mean_full_delta"] > n_comparison["threshold"]
    assert n_comparison["cell_point_guards"][cells[0]] is False
    assert n_comparison["every_cell_point_guard"] is False
    assert failed_n["NZ_path_passed"] is False

    margin_edge = logits.copy()
    margin_edge[0, diagnostic.ARMS.index("U0")] = 0.5 + diagnostic.NZ_U0_MARGIN
    failed_margin = diagnostic.primary_iut_report(
        targets,
        margin_edge,
        cluster_ordinal,
        sampled,
        cell_ids=cells,
        arm_ids=diagnostic.ARMS,
    )
    u0_comparison = failed_margin["comparisons"]["NZ_minus_U0"]
    assert u0_comparison["cell_point_guards"][cells[0]] is False
    assert failed_margin["NZ_path_passed"] is False


def test_calibrator_comparison_uses_aa_minus_mix_and_frozen_margins(monkeypatch) -> None:
    monkeypatch.setattr(diagnostic, "DEV_EPISODES", 2)
    monkeypatch.setattr(diagnostic, "ROOTS_PER_EPISODE", 5)
    monkeypatch.setattr(diagnostic, "DEV_ROOTS", 10)
    targets = np.zeros((10, diagnostic.ACTION_COUNT), dtype=np.float64)
    factual_actions = np.zeros(10, dtype=np.int64)
    cluster_ordinal = np.repeat(np.arange(2, dtype=np.int32), 5)
    sampled = np.tile(np.arange(2, dtype=np.int32), (31, 1))
    cells = tuple(f"cell-{index}" for index in range(9))
    aa = np.full((9, 10, diagnostic.ACTION_COUNT), 0.2, dtype=np.float64)
    mix = np.full_like(aa, 0.203)
    mix[:, :, 0] = 0.19

    passed = diagnostic.calibrator_comparison_report(
        targets,
        factual_actions,
        cluster_ordinal,
        np.stack((aa, mix)),
        sampled,
        cell_ids=cells,
    )
    assert passed["passed"] is True
    assert passed["AA_can_nominate"] is False
    assert passed["comparisons"]["factual_BCE"]["grand_mean_improvement"] > 0.0
    assert passed["comparisons"]["factual_Brier"]["grand_mean_improvement"] > 0.0
    for name, margin in (
        ("all_action_BCE", diagnostic.AA_BCE_MARGIN),
        ("all_action_Brier", diagnostic.AA_BRIER_MARGIN),
    ):
        comparison = passed["comparisons"][name]
        assert comparison["definition"] == "AA_loss_minus_MIX_loss"
        assert comparison["threshold"] == -margin
        assert -margin < comparison["grand_mean_improvement"] < 0.0
        assert comparison["passed"] is True

    reversed_direction = diagnostic.calibrator_comparison_report(
        targets,
        factual_actions,
        cluster_ordinal,
        np.stack((mix, aa)),
        sampled,
        cell_ids=cells,
    )
    assert reversed_direction["comparisons"]["factual_BCE"]["passed"] is False
    assert reversed_direction["comparisons"]["factual_Brier"]["passed"] is False

    outside_margin = mix.copy()
    outside_margin[:, :, 1:] = 0.22
    failed_noninferiority = diagnostic.calibrator_comparison_report(
        targets,
        factual_actions,
        cluster_ordinal,
        np.stack((aa, outside_margin)),
        sampled,
        cell_ids=cells,
    )
    assert failed_noninferiority["comparisons"]["factual_BCE"]["passed"] is True
    assert failed_noninferiority["comparisons"]["factual_Brier"]["passed"] is True
    assert failed_noninferiority["comparisons"]["all_action_BCE"]["passed"] is False
    assert failed_noninferiority["comparisons"]["all_action_Brier"]["passed"] is False

    one_bad_cell = mix.copy()
    one_bad_cell[0, :, 0] = 0.21
    failed_cell = diagnostic.calibrator_comparison_report(
        targets,
        factual_actions,
        cluster_ordinal,
        np.stack((aa, one_bad_cell)),
        sampled,
        cell_ids=cells,
    )
    factual = failed_cell["comparisons"]["factual_BCE"]
    assert factual["grand_mean_improvement"] > 0.0
    assert factual["cell_point_guards"][cells[0]] is False
    assert factual["every_cell_point_guard"] is False
    assert failed_cell["passed"] is False


def test_authoritative_selection_is_full_conjunction_and_aa_cannot_nominate(monkeypatch) -> None:
    cells = tuple(f"cell-{index}" for index in range(9))
    arrays = {
        "cell_ids": np.asarray(cells, dtype="S16"),
        "arm_ids": np.asarray(diagnostic.ARMS, dtype="S3"),
        "cell_fit_index": np.zeros(9, dtype=np.int16),
        "fit_targets": np.zeros((3, 1, diagnostic.ACTION_COUNT), dtype=np.float64),
        "fit_actions": np.zeros((3, 1), dtype=np.int64),
        "dev_targets": np.zeros((1, diagnostic.ACTION_COUNT), dtype=np.float64),
        "dev_actions": np.zeros(1, dtype=np.int64),
        "dev_cluster_ordinal": np.zeros(1, dtype=np.int32),
        "bootstrap_indices": np.zeros((1, 1), dtype=np.int32),
        "episode_derangements": np.zeros((1, 1), dtype=np.int32),
        "dev_raw_logits": np.zeros((9, 5, 1, 5), dtype=np.float64),
        "dev_raw_probabilities": np.full((9, 5, 1, 5), 0.5, dtype=np.float64),
        "nz_calibrated_logits": np.zeros((2, 9, 1, 5), dtype=np.float64),
        "nz_calibrated_probabilities": np.full((2, 9, 1, 5), 0.5, dtype=np.float64),
        "calibrator_accepted": np.ones((2, 9), dtype=np.bool_),
    }
    switches = {"raw": True, "deploy": True, "representation": True, "dual": True}
    baseline_report = {"all_action": {"aggregate": {"roc_auc": 0.5}}}
    empty_metric = lambda: SimpleNamespace(as_dict=lambda: {})
    monkeypatch.setattr(diagnostic, "fit_baselines_from_arrays", lambda *_args: object())
    monkeypatch.setattr(
        diagnostic,
        "evaluate_train_fitted_baselines",
        lambda *_args, **_kwargs: SimpleNamespace(as_dict=lambda: baseline_report),
    )
    monkeypatch.setattr(
        diagnostic, "all_action_probability_metrics", lambda *_args, **_kwargs: empty_metric()
    )
    monkeypatch.setattr(
        diagnostic, "binary_probability_metrics", lambda *_args, **_kwargs: empty_metric()
    )
    monkeypatch.setattr(
        diagnostic.pb21j,
        "score_ranking_gate",
        lambda *_args, **_kwargs: {"passed": switches["raw"]},
    )
    monkeypatch.setattr(
        diagnostic,
        "candidate_replica_gate",
        lambda **_kwargs: {"passed": switches["deploy"]},
    )
    monkeypatch.setattr(
        diagnostic,
        "primary_iut_report",
        lambda *_args, **_kwargs: {"NZ_path_passed": switches["representation"]},
    )
    monkeypatch.setattr(
        diagnostic,
        "calibrator_comparison_report",
        lambda *_args, **_kwargs: {
            "passed": switches["dual"],
            "AA_can_nominate": False,
        },
    )
    monkeypatch.setattr(diagnostic.pb21j, "_all_numeric_finite", lambda _value: True)

    selected = diagnostic.evaluate_authoritative_evidence(arrays)
    assert selected["selection"]["eligible"] is True
    assert selected["selection"]["selected_calibrator"] == "MIX"
    assert selected["selection"]["AA_can_nominate"] is False

    for failed_gate in tuple(switches):
        switches[failed_gate] = False
        rejected = diagnostic.evaluate_authoritative_evidence(arrays)
        assert rejected["selection"]["eligible"] is False, failed_gate
        assert rejected["selection"]["selected_calibrator"] is None, failed_gate
        switches[failed_gate] = True

    arrays["calibrator_accepted"][1, 0] = False
    aa_only = diagnostic.evaluate_authoritative_evidence(arrays)
    assert arrays["calibrator_accepted"][0].all()
    assert aa_only["selection"]["MIX_acceptance_passed"] is False
    assert aa_only["selection"]["AA_can_nominate"] is False
    assert aa_only["selection"]["eligible"] is False
    assert aa_only["selection"]["selected_calibrator"] is None


def test_evidence_schema_freezes_keys_shapes_and_dtypes() -> None:
    assert set(diagnostic.EVIDENCE_DTYPES) == diagnostic.EVIDENCE_KEYS
    assert diagnostic._expected_evidence_shapes()["bootstrap_indices"] == (50_000, 768)
    assert diagnostic._expected_evidence_shapes()["dev_targets"] == (9_216, 5)
    assert diagnostic.EVIDENCE_DTYPES["bootstrap_indices"] == "<i4"
    assert diagnostic.EVIDENCE_DTYPES["dev_cluster_ids"] == "|S64"
    assert len(diagnostic.evidence_key_set_sha256()) == 64


def test_deterministic_npz_is_atomic_create_only_and_non_object(tmp_path: Path) -> None:
    arrays = _minimal_evidence_arrays()
    first = diagnostic.publish_evidence_create_only(
        tmp_path / "first.npz", arrays, attempt_sha256="a" * 64
    )
    second = diagnostic.publish_evidence_create_only(
        tmp_path / "second.npz", arrays, attempt_sha256="b" * 64
    )
    assert first["sha256"] == second["sha256"]
    assert first["byte_length"] == second["byte_length"]
    assert first["array_manifest_sha256"] == second["array_manifest_sha256"]
    with np.load(tmp_path / "first.npz", allow_pickle=False) as loaded:
        assert set(loaded.files) == diagnostic.EVIDENCE_KEYS
        assert all(not loaded[name].dtype.hasobject for name in loaded.files)
    with _raises(FileExistsError, match="already exists"):
        diagnostic.publish_evidence_create_only(
            tmp_path / "first.npz", arrays, attempt_sha256="c" * 64
        )


def test_authoritative_loader_accepts_valid_small_fixture_and_refits(tmp_path: Path, monkeypatch) -> None:
    arrays, source_evidence, cal_raw_hashes = _valid_small_evidence(monkeypatch)
    publication = diagnostic.publish_evidence_create_only(
        tmp_path / "valid.npz", arrays, attempt_sha256="d" * 64
    )
    reloaded, report = diagnostic.reload_and_validate_evidence(
        tmp_path / "valid.npz",
        publication,
        source_evidence=source_evidence,
        cal_nz_raw_logit_sha256_by_cell=cal_raw_hashes,
    )
    assert set(reloaded) == diagnostic.EVIDENCE_KEYS
    assert report["AA_and_MIX_deterministic_refits_verified"] is True
    assert report[
        "all_seven_source_partition_target_action_root_cross_links_verified"
    ] is True
    assert report["CAL_NZ_raw_logit_prepublication_cross_links_verified"] is True
    assert report["DEV_ordered_episode_source_cross_link_verified"] is True
    assert report["passed"] is True


@_parameterize(
    ("case", "message"),
    (
        ("dtype", "dtype drifted"),
        ("probability", "raw probabilities"),
        ("cluster", "cluster ordinal"),
        ("aa_refit", "AA deterministic evidence refit"),
        ("mix_refit", "MIX applied/proposed|MIX deterministic evidence refit"),
        ("bootstrap", "bootstrap matrix"),
        ("derangement", "derangement matrix"),
    ),
)
def test_authoritative_loader_rejects_tampering(
    case: str,
    message: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays, source_evidence, cal_raw_hashes = _valid_small_evidence(monkeypatch)
    if case == "dtype":
        arrays["dev_actions"] = arrays["dev_actions"].astype(np.int32)
    elif case == "probability":
        arrays["dev_raw_probabilities"] = arrays["dev_raw_probabilities"].copy()
        arrays["dev_raw_probabilities"][0, 0, 0, 0] += 1.0e-6
    elif case == "cluster":
        arrays["dev_cluster_ordinal"] = arrays["dev_cluster_ordinal"].copy()
        arrays["dev_cluster_ordinal"][0] = 1
    elif case == "aa_refit":
        arrays["calibrator_scales"] = arrays["calibrator_scales"].copy()
        arrays["calibrator_scales"][0, 0, 0] += 0.01
        raw = arrays["dev_raw_logits"][0, diagnostic.ARMS.index("NZ")]
        scales = arrays["calibrator_scales"][0, 0]
        biases = arrays["calibrator_biases"][0, 0]
        arrays["nz_calibrated_logits"] = arrays["nz_calibrated_logits"].copy()
        arrays["nz_calibrated_probabilities"] = arrays["nz_calibrated_probabilities"].copy()
        arrays["nz_calibrated_logits"][0, 0] = raw * scales[None, :] + biases[None, :]
        arrays["nz_calibrated_probabilities"][0, 0] = diagnostic._sigmoid(
            arrays["nz_calibrated_logits"][0, 0]
        )
    elif case == "bootstrap":
        arrays["bootstrap_indices"] = np.zeros_like(arrays["bootstrap_indices"])
    elif case == "derangement":
        arrays["episode_derangements"] = np.roll(
            arrays["episode_derangements"],
            shift=1,
            axis=0,
        )
    else:
        arrays["mix_proposed_biases"] = arrays["mix_proposed_biases"].copy()
        arrays["mix_proposed_biases"][0, 0] += 0.01
    publication = diagnostic.publish_evidence_create_only(
        tmp_path / f"{case}.npz", arrays, attempt_sha256="e" * 64
    )
    with _raises(ValueError, match=message):
        diagnostic.reload_and_validate_evidence(
            tmp_path / f"{case}.npz",
            publication,
            source_evidence=source_evidence,
            cal_nz_raw_logit_sha256_by_cell=cal_raw_hashes,
        )


@_parameterize(
    ("array_name", "index", "label", "kind"),
    (
        ("fit_targets", (0, 0, 0), "F0", "target"),
        ("fit_actions", (0, 0), "F0", "factual-action"),
        ("cal_targets", (0, 0, 0), "C0", "target"),
        ("cal_actions", (0, 0), "C0", "factual-action"),
        ("dev_targets", (0, 0), "DEV", "target"),
        ("dev_actions", (0,), "DEV", "factual-action"),
    ),
)
def test_authoritative_loader_rejects_source_array_bit_tampering(
    array_name: str,
    index: tuple[int, ...],
    label: str,
    kind: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays, source_evidence, cal_raw_hashes = _valid_small_evidence(monkeypatch)
    arrays[array_name] = arrays[array_name].copy()
    current = arrays[array_name][index]
    if "targets" in array_name:
        arrays[array_name][index] = 1.0 - current
    else:
        arrays[array_name][index] = (int(current) + 1) % diagnostic.ACTION_COUNT
    publication = diagnostic.publish_evidence_create_only(
        tmp_path / f"{array_name}.npz", arrays, attempt_sha256="7" * 64
    )
    with _raises(ValueError, match=rf"{label} {kind} .*cross-link drifted"):
        diagnostic.reload_and_validate_evidence(
            tmp_path / f"{array_name}.npz",
            publication,
            source_evidence=source_evidence,
            cal_nz_raw_logit_sha256_by_cell=cal_raw_hashes,
        )


def test_authoritative_loader_rejects_ordered_root_id_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays, source_evidence, cal_raw_hashes = _valid_small_evidence(monkeypatch)
    arrays["fit_root_ids"] = arrays["fit_root_ids"].copy()
    arrays["fit_root_ids"][0, 0] = b"F0:root:tampered"
    publication = diagnostic.publish_evidence_create_only(
        tmp_path / "root-id.npz", arrays, attempt_sha256="6" * 64
    )
    with _raises(ValueError, match="F0 ordered-root source-evidence cross-link drifted"):
        diagnostic.reload_and_validate_evidence(
            tmp_path / "root-id.npz",
            publication,
            source_evidence=source_evidence,
            cal_nz_raw_logit_sha256_by_cell=cal_raw_hashes,
        )


def test_authoritative_loader_rejects_cal_raw_logit_digest_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays, source_evidence, cal_raw_hashes = _valid_small_evidence(monkeypatch)
    arrays["cal_nz_raw_logits"] = arrays["cal_nz_raw_logits"].copy()
    arrays["cal_nz_raw_logits"][0, 0, 0] += 1.0e-6
    publication = diagnostic.publish_evidence_create_only(
        tmp_path / "cal-raw.npz", arrays, attempt_sha256="8" * 64
    )
    with _raises(ValueError, match="CAL NZ raw-logit source cross-link drifted"):
        diagnostic.reload_and_validate_evidence(
            tmp_path / "cal-raw.npz",
            publication,
            source_evidence=source_evidence,
            cal_nz_raw_logit_sha256_by_cell=cal_raw_hashes,
        )


def test_authoritative_loader_rejects_self_consistent_sub_tolerance_logit_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays, source_evidence, cal_raw_hashes = _valid_small_evidence(monkeypatch)
    arrays["nz_calibrated_logits"] = arrays["nz_calibrated_logits"].copy()
    arrays["nz_calibrated_probabilities"] = arrays[
        "nz_calibrated_probabilities"
    ].copy()
    arrays["nz_calibrated_logits"][0, 0, 0, 0] += 5.0e-13
    arrays["nz_calibrated_probabilities"][0, 0] = diagnostic._sigmoid(
        arrays["nz_calibrated_logits"][0, 0]
    )
    publication = diagnostic.publish_evidence_create_only(
        tmp_path / "sub-tolerance.npz", arrays, attempt_sha256="9" * 64
    )
    with _raises(
        ValueError,
        match="calibrated logits do not exactly match applied parameters",
    ):
        diagnostic.reload_and_validate_evidence(
            tmp_path / "sub-tolerance.npz",
            publication,
            source_evidence=source_evidence,
            cal_nz_raw_logit_sha256_by_cell=cal_raw_hashes,
        )


def test_evidence_publication_rejects_missing_or_extra_keys(tmp_path: Path) -> None:
    arrays = _minimal_evidence_arrays()
    arrays.pop("action_ids")
    with _raises(ValueError, match="key set"):
        diagnostic.publish_evidence_create_only(
            tmp_path / "missing.npz", arrays, attempt_sha256="f" * 64
        )


    arrays = _minimal_evidence_arrays()
    arrays["unexpected"] = np.asarray([0], dtype=np.int8)
    with _raises(ValueError, match="key set"):
        diagnostic.publish_evidence_create_only(
            tmp_path / "extra.npz", arrays, attempt_sha256="f" * 64
        )


def test_canonical_path_and_attempt_receipt_are_fail_closed(tmp_path: Path, monkeypatch) -> None:
    expected = (tmp_path / "canonical.json").resolve()
    assert diagnostic._require_canonical_path(expected, expected, role="test") == expected
    with _raises(ValueError, match="single canonical path"):
        diagnostic._require_canonical_path(tmp_path / "other.json", expected, role="test")

    attempt = (tmp_path / "attempt.json").resolve()
    monkeypatch.setattr(diagnostic, "canonical_paths", lambda _root: {"attempt": attempt})
    registration = {
        "sha256": "a" * 64,
        "payload": {"source_bundle": {"sha256": "b" * 64}},
    }
    first = diagnostic._publish_attempt(tmp_path, registration)
    assert first["payload"]["retry_allowed"] is False
    assert first["payload"]["published_before_any_source_or_data_construction"] is True
    with _raises(FileExistsError, match="retry is forbidden"):
        diagnostic._publish_attempt(tmp_path, registration)


def test_lifecycle_order_is_structurally_frozen() -> None:
    run_source = inspect.getsource(diagnostic.run)
    impl_source = inspect.getsource(diagnostic._run_impl)
    assert run_source.index("_publish_attempt") < run_source.index("_run_impl")
    assert impl_source.index("fit_sources = partition_sources") < impl_source.index(
        "cal_sources = partition_sources"
    )
    assert impl_source.index("fit_mix_calibrator") < impl_source.index(
        'dev_source = partition_sources(("DEV",))'
    )
    assert impl_source.index("source_evidence =") < impl_source.index(
        "reload_and_validate_evidence"
    )
    assert impl_source.index("publish_evidence_create_only") < impl_source.index(
        "reload_and_validate_evidence"
    ) < impl_source.index("evaluate_authoritative_evidence")
    assert impl_source.index("evaluate_authoritative_evidence") < impl_source.index(
        "_publish_json_create_only(output, result)"
    )


def test_failure_receipt_binds_already_published_evidence(tmp_path: Path, monkeypatch) -> None:
    paths = {
        "upstream_result": (tmp_path / "upstream.json").resolve(),
        "registration": (tmp_path / "registration.json").resolve(),
        "attempt": (tmp_path / "attempt.json").resolve(),
        "evidence": (tmp_path / "evidence.npz").resolve(),
        "result": (tmp_path / "result.json").resolve(),
    }
    paths["upstream_result"].write_text("{}", encoding="utf-8")
    paths["registration"].write_text("{}", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "canonical_paths", lambda _root: paths)
    monkeypatch.setattr(
        diagnostic,
        "validate_registration",
        lambda _path: {
            "path": str(paths["registration"]),
            "sha256": "3" * 64,
            "payload": {"source_bundle": {"sha256": "4" * 64}},
        },
    )
    monkeypatch.setattr(diagnostic.strict, "load_exact_parent", lambda _path: (object(), {}, {}))

    def fail_after_evidence(**_kwargs: object) -> None:
        paths["evidence"].write_bytes(b"sealed evidence")
        raise RuntimeError("synthetic post-evidence failure")

    monkeypatch.setattr(diagnostic, "_run_impl", fail_after_evidence)
    with _raises(RuntimeError, match="post-evidence"):
        diagnostic.run(
            upstream_result=paths["upstream_result"],
            output=paths["result"],
            registration=paths["registration"],
        )
    receipt = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert receipt["classification"] == "diagnostic_run_failed_not_candidate"
    assert receipt["retry_allowed"] is False
    assert receipt["authoritative_evidence"]["published_before_failure"] is True
    assert receipt["authoritative_evidence"]["byte_length"] == len(b"sealed evidence")


def test_decision_contract_only_mix_can_nominate() -> None:
    contract = diagnostic.decision_contract()
    assert contract["only_candidate_path"] == "NZ_MIX"
    assert contract["AA_can_nominate"] is False
    assert contract["selection_by_observed_performance"] is False
    assert diagnostic.AA_BCE_MARGIN == 0.010
    assert diagnostic.AA_BRIER_MARGIN == 0.005
    assert diagnostic.ONE_SIDED_ALPHA == 0.025


class PB21KNZDualCalTests(unittest.TestCase):
    """Generated wrappers keep the focused suite dependency-free."""


def _make_unittest_wrapper(function, parameters: Mapping[str, object] | None = None):
    def run_case(self: unittest.TestCase) -> None:
        monkeypatch = _MonkeyPatch()
        temporary: TemporaryDirectory[str] | None = None
        kwargs = dict(parameters or {})
        signature = inspect.signature(function)
        if "monkeypatch" in signature.parameters:
            kwargs["monkeypatch"] = monkeypatch
        if "tmp_path" in signature.parameters:
            temporary = TemporaryDirectory()
            kwargs["tmp_path"] = Path(temporary.name)
        try:
            function(**kwargs)
        finally:
            monkeypatch.undo()
            if temporary is not None:
                temporary.cleanup()

    return run_case


for _name, _function in tuple(globals().items()):
    if not _name.startswith("test_") or not callable(_function):
        continue
    # The generated unittest methods below already cover every parameter set.
    # Do not also collect the templates as unparametrized pytest functions.
    _function.__test__ = False
    _parameterized = getattr(_function, "_unittest_parametrize", None)
    if _parameterized is None:
        setattr(PB21KNZDualCalTests, _name, _make_unittest_wrapper(_function))
        continue
    _parameter_names, _parameter_values = _parameterized
    for _index, _values in enumerate(_parameter_values):
        _mapping = dict(zip(_parameter_names, _values, strict=True))
        setattr(
            PB21KNZDualCalTests,
            f"{_name}_{_index}",
            _make_unittest_wrapper(_function, _mapping),
        )


if __name__ == "__main__":
    unittest.main()
