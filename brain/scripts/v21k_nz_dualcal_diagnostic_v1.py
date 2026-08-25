"""Preregistered PB21K NZ dual-domain calibration diagnostic.

This CPU-only diagnostic repeats the exact PB21J masked-485 scorer recipe on
fresh FIT cohorts, prunes the fixed NZ representation, and changes only its
TRAIN-CAL calibration.  The existing all-action affine calibrator is a sealed
control; only the preregistered equal-domain MIX calibrator may nominate the
whole masked-superset-train/pruned-deployment recipe.  The irreversible
evidence NPZ is published and reloaded before any numerical DEV decision is
computed.  No checkpoint or candidate can be emitted here.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from irene_brain.data import (
    DatasetSplit,
    MazeChaseDatasetConfig,
    maze_chase_dataset_manifest_sha256,
)
from irene_brain.evaluation.v21_qualification_metrics import (
    all_action_probability_metrics,
    binary_probability_metrics,
    evaluate_train_fitted_baselines,
    fit_train_baselines,
    gather_factual_rows,
)
from irene_brain.v2.hazard_calibration import (
    PerActionAffineHazardCalibrator,
    TrainCalibrationProvenance,
    fit_train_only_per_action_affine,
)
from run_provenance import apply_deterministic_mode
import v21i_development_runner as development
import v21i_strict_live_representation_probe_v1 as strict
import v21j_fresh_bz_production_form_diagnostic_v1 as pb21j


SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = 1
MODE = "pb21k_nz_dualcal_diagnostic_v1"
REGISTRATION_MODE = "pb21k_nz_dualcal_registration_v1"
CLASSIFICATION = "diagnostic_not_candidate"
RUN_ID = "2026-08-25-pb21k-nz-dualcal-v1"

CANONICAL_UPSTREAM_RESULT = pb21j.CANONICAL_UPSTREAM_RESULT
CANONICAL_REGISTRATION = (
    "brain/runs/v21k-diagnostics/2026-08-25-pb21k-nz-dualcal-v1.registration.json"
)
CANONICAL_ATTEMPT = (
    "brain/runs/v21k-diagnostics/2026-08-25-pb21k-nz-dualcal-v1.attempt.json"
)
CANONICAL_EVIDENCE = (
    "brain/runs/v21k-diagnostics/2026-08-25-pb21k-nz-dualcal-v1.evidence.npz"
)
CANONICAL_RESULT = "brain/runs/v21k-diagnostics/2026-08-25-pb21k-nz-dualcal-v1.json"
PB21J_REGISTRATION = pb21j.CANONICAL_REGISTRATION
PB21J_ATTEMPT = pb21j.CANONICAL_ATTEMPT
PB21J_RESULT = pb21j.CANONICAL_RESULT
EXACT_PB21J_REGISTRATION_SHA256 = (
    "8ecebd1b6317f94dbab3d0e2bf82af4cdd44952ebe9e01e478a74aeb8648c25b"
)
EXACT_PB21J_ATTEMPT_SHA256 = (
    "cc17c4d507ef3d3fd84e6e6cc01e7757bdf7da3a157304d6e691de47859c43fe"
)
EXACT_PB21J_RESULT_SHA256 = (
    "21fafd0b7107ce2f3edf7cf4c8352caee8d0b519395a15e737249edba7e41182"
)
EXACT_PB21J_SOURCE_BUNDLE_SHA256 = (
    "6207c1a93cd732694e2a0508b117636d1fad2632bf51581e91e1306015af9482"
)
EXACT_HAZARD_CALIBRATION_SHA256 = (
    "227ba0fddb6ca9fe96b673334865f8a66f0c398baa8f163d87777dfb2256f3b0"
)

_PREREGISTRATION = (
    "brain/docs/preregistrations/2026-08-25-pb21k-nz-dualcal-v1.md"
)
_TEST_FILE = "brain/tests/test_v21k_nz_dualcal_diagnostic_v1.py"
_DIAGNOSTIC_FILES = (
    "brain/scripts/v21k_nz_dualcal_diagnostic_v1.py",
    _PREREGISTRATION,
    _TEST_FILE,
)
_HAZARD_CALIBRATION_FILE = "brain/src/irene_brain/v2/hazard_calibration.py"

ACTION_COUNT = pb21j.ACTION_COUNT
SUPERSET_WIDTH = pb21j.SUPERSET_WIDTH
TRAIN_HEAD_PARAMETERS = pb21j.TRAIN_HEAD_PARAMETERS
ARMS = (pb21j.ARM_N, pb21j.ARM_Z, pb21j.ARM_BZ, pb21j.ARM_NZ, pb21j.ARM_U0)
ARM_N, ARM_Z, ARM_BZ, ARM_NZ, ARM_U0 = ARMS
ACTIVE_COLUMNS = {arm: pb21j.ACTIVE_COLUMNS[arm] for arm in ARMS}
PRUNED_PARAMETER_COUNTS = {arm: pb21j.PRUNED_PARAMETER_COUNTS[arm] for arm in ARMS}

FIT_COHORTS = ("F0", "F1", "F2")
CAL_COHORTS = ("C0", "C1", "C2")
COHORT_PAIRS = tuple(zip(FIT_COHORTS, CAL_COHORTS, strict=True))
INIT_SEEDS = (51_042, 52_042, 53_042)
PERMUTATION_SEED = 60_042
BOOTSTRAP_SEED = 82_042
DERANGEMENT_SEED = 83_042
PASSES = pb21j.PASSES
ROOT_BATCH_SIZE = pb21j.ROOT_BATCH_SIZE
ROOTS_PER_FIT = pb21j.ROOTS_PER_FIT
ROOTS_PER_CAL = 3_072
STEPS_PER_HEAD = pb21j.STEPS_PER_HEAD
TOTAL_HEADS = 45
TOTAL_OPTIMIZER_STEPS = 184_320

ROOTS_PER_EPISODE = pb21j.ROOTS_PER_EPISODE
DEV_EPISODES = 768
DEV_ROOTS = DEV_EPISODES * ROOTS_PER_EPISODE
BOOTSTRAP_RESAMPLES = 50_000
DERANGEMENT_REPETITIONS = 20
ONE_SIDED_ALPHA = 0.025
QUANTILE_METHOD = "linear"
NZ_U0_MARGIN = 0.025
AA_BCE_MARGIN = 0.010
AA_BRIER_MARGIN = 0.005
ECE_BINS = pb21j.ECE_BINS

CALIBRATION_L2 = pb21j.CALIBRATION_L2
CALIBRATION_MINIMUM_SCALE = pb21j.CALIBRATION_MINIMUM_SCALE
CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION = pb21j.CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION
CALIBRATION_MINIMUM_CLASS_EXAMPLES = pb21j.CALIBRATION_MINIMUM_CLASS_EXAMPLES
CALIBRATION_MAX_ITERATIONS = pb21j.CALIBRATION_MAX_ITERATIONS
CALIBRATION_TOLERANCE = pb21j.CALIBRATION_TOLERANCE

CALIBRATORS = ("AA", "MIX")
_INTEGER_TORCH_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
PRIMARY_COMPARISONS = (
    ("NZ_minus_N", ARM_NZ, ARM_N, 0.0),
    ("NZ_minus_U0", ARM_NZ, ARM_U0, -NZ_U0_MARGIN),
)
CALIBRATOR_COMPARISONS = (
    ("factual_BCE", 0.0),
    ("factual_Brier", 0.0),
    ("all_action_BCE", -AA_BCE_MARGIN),
    ("all_action_Brier", -AA_BRIER_MARGIN),
)


@dataclass(frozen=True)
class FreshPartitionSpec:
    label: str
    role: str
    split: DatasetSplit
    seed_offset: int
    episodes: int


PARTITION_SPECS = (
    FreshPartitionSpec("F0", "TRAIN-FIT", DatasetSplit.TRAIN, 134_217_728, 256),
    FreshPartitionSpec("C0", "TRAIN-CAL", DatasetSplit.TRAIN, 134_217_984, 256),
    FreshPartitionSpec("F1", "TRAIN-FIT", DatasetSplit.TRAIN, 134_218_240, 256),
    FreshPartitionSpec("C1", "TRAIN-CAL", DatasetSplit.TRAIN, 134_218_496, 256),
    FreshPartitionSpec("F2", "TRAIN-FIT", DatasetSplit.TRAIN, 134_218_752, 256),
    FreshPartitionSpec("C2", "TRAIN-CAL", DatasetSplit.TRAIN, 134_219_008, 256),
    FreshPartitionSpec("DEV", "DEV", DatasetSplit.VALIDATION, 150_994_944, 768),
)

PARTITION_ALGORITHM = (
    "pb21k-v1:F0=train[134217728,134217984);C0=train[134217984,134218240);"
    "F1=train[134218240,134218496);C1=train[134218496,134218752);"
    "F2=train[134218752,134219008);C2=train[134219008,134219264);"
    "DEV=validation[150994944,150995712)"
)

EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "action_ids",
        "fit_ids",
        "cal_ids",
        "cell_ids",
        "cell_fit_index",
        "cell_cal_index",
        "cell_init_seed",
        "arm_ids",
        "calibrator_ids",
        "fit_targets",
        "fit_actions",
        "fit_root_ids",
        "cal_targets",
        "cal_actions",
        "cal_root_ids",
        "cal_nz_raw_logits",
        "calibrator_scales",
        "calibrator_biases",
        "calibrator_accepted",
        "mix_proposed_scales",
        "mix_proposed_biases",
        "mix_per_action_accepted",
        "dev_targets",
        "dev_actions",
        "dev_root_ids",
        "dev_cluster_ordinal",
        "dev_cluster_ids",
        "dev_raw_logits",
        "dev_raw_probabilities",
        "nz_calibrated_logits",
        "nz_calibrated_probabilities",
        "bootstrap_indices",
        "episode_derangements",
    }
)
EVIDENCE_DTYPES = {
    "schema_version": "<u2",
    "action_ids": "<i2",
    "fit_ids": "|S2",
    "cal_ids": "|S2",
    "cell_ids": "|S16",
    "cell_fit_index": "<i2",
    "cell_cal_index": "<i2",
    "cell_init_seed": "<i4",
    "arm_ids": "|S3",
    "calibrator_ids": "|S3",
    "fit_targets": "<f8",
    "fit_actions": "<i8",
    "fit_root_ids": "|S64",
    "cal_targets": "<f8",
    "cal_actions": "<i8",
    "cal_root_ids": "|S64",
    "cal_nz_raw_logits": "<f8",
    "calibrator_scales": "<f8",
    "calibrator_biases": "<f8",
    "calibrator_accepted": "|b1",
    "mix_proposed_scales": "<f8",
    "mix_proposed_biases": "<f8",
    "mix_per_action_accepted": "|b1",
    "dev_targets": "<f8",
    "dev_actions": "<i8",
    "dev_root_ids": "|S64",
    "dev_cluster_ordinal": "<i4",
    "dev_cluster_ids": "|S64",
    "dev_raw_logits": "<f8",
    "dev_raw_probabilities": "<f8",
    "nz_calibrated_logits": "<f8",
    "nz_calibrated_probabilities": "<f8",
    "bootstrap_indices": "<i4",
    "episode_derangements": "<i4",
}


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return strict._array_sha256(values)


def evidence_key_set_sha256() -> str:
    encoded = json.dumps(sorted(EVIDENCE_KEYS), separators=(",", ":")).encode("ascii")
    return sha256(b"IRPB21KEVIDENCEKEYS\x01" + encoded).hexdigest()


def _state_sha256(module: torch.nn.Module) -> str:
    return development._state_dict_sha256(module.state_dict())


def _common_dataset_config() -> dict[str, object]:
    return pb21j._common_dataset_config()


def partition_contracts() -> dict[str, development.PartitionContract]:
    return {
        spec.label: development.PartitionContract(
            spec.role,
            MazeChaseDatasetConfig(
                split=spec.split,
                seed_offset=spec.seed_offset,
                sequence_count=spec.episodes,
                **_common_dataset_config(),
            ),
        )
        for spec in PARTITION_SPECS
    }


def partition_manifests() -> dict[str, str]:
    return {
        label: maze_chase_dataset_manifest_sha256(contract.dataset_config)
        for label, contract in partition_contracts().items()
    }


def partition_sources(
    labels: Sequence[str],
    *,
    source_factory: Callable[[development.PartitionContract], object] = development.PartitionSource,
) -> dict[str, object]:
    allowed = {spec.label for spec in PARTITION_SPECS}
    if not labels or any(label not in allowed for label in labels):
        raise ValueError("only registered PB21K FIT/CAL/DEV labels may be constructed")
    if len(set(labels)) != len(labels):
        raise ValueError("partition labels must be unique")
    contracts = partition_contracts()
    expected = partition_manifests()
    sources = {label: source_factory(contracts[label]) for label in labels}
    for label, source in sources.items():
        if getattr(source, "manifest_sha256", None) != expected[label]:
            raise RuntimeError(f"constructed {label} source manifest differs from registration")
    return sources


def occupied_range_registry() -> list[dict[str, object]]:
    records = list(pb21j.occupied_range_registry())
    for spec in PARTITION_SPECS:
        effective_start, effective_stop = pb21j._effective_interval(
            spec.split.value, spec.seed_offset, spec.seed_offset + spec.episodes
        )
        records.append(
            {
                "dataset_family": "maze_chase",
                "split": spec.split.value,
                "local_start": spec.seed_offset,
                "local_stop_exclusive": spec.seed_offset + spec.episodes,
                "effective_start": effective_start,
                "effective_stop_exclusive": effective_stop,
                "owner": f"PB21K {spec.label}",
            }
        )
    return records


def assert_no_range_collisions(
    records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    frozen = list(occupied_range_registry() if records is None else records)
    collisions: list[tuple[str, str]] = []
    for index, left in enumerate(frozen):
        for right in frozen[index + 1 :]:
            if left["dataset_family"] != right["dataset_family"]:
                continue
            if max(int(left["effective_start"]), int(right["effective_start"])) < min(
                int(left["effective_stop_exclusive"]),
                int(right["effective_stop_exclusive"]),
            ):
                collisions.append((str(left["owner"]), str(right["owner"])))
    if collisions:
        raise RuntimeError(f"occupied seed ranges collide: {collisions}")
    return {
        "algorithm": "split-prefix-effective-u64-interval-audit-v1",
        "records": frozen,
        "collisions": [],
        "passed": True,
    }


def validate_pb21j_parent(project_root: Path) -> dict[str, object]:
    registration = project_root / PB21J_REGISTRATION
    attempt = project_root / PB21J_ATTEMPT
    result = project_root / PB21J_RESULT
    expected = (
        (registration, EXACT_PB21J_REGISTRATION_SHA256, "registration"),
        (attempt, EXACT_PB21J_ATTEMPT_SHA256, "attempt"),
        (result, EXACT_PB21J_RESULT_SHA256, "result"),
    )
    for path, digest, role in expected:
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"exact PB21J {role} absent or drifted")
    payload = json.loads(result.read_text(encoding="utf-8"))
    selection = payload.get("selection", {})
    nz_status = selection.get("path_status", {}).get("NZ", {})
    if (
        payload.get("mode") != pb21j.MODE
        or payload.get("classification") != pb21j.CLASSIFICATION
        or payload.get("candidate_publication_allowed") is not False
        or payload.get("source_bundle", {}).get("sha256") != EXACT_PB21J_SOURCE_BUNDLE_SHA256
        or selection.get("diagnosis") != "representation_supported_deployability_unresolved"
        or selection.get("selected_arm") is not None
        or nz_status.get("representation_inference_passed") is not True
        or nz_status.get("calibration_acceptance_passed") is not True
        or nz_status.get("deployability_excluding_calibration_passed") is not False
    ):
        raise ValueError("PB21J scientific parent envelope drifted")
    current = pb21j._diagnostic_source_bundle(project_root)
    if current.get("sha256") != EXACT_PB21J_SOURCE_BUNDLE_SHA256:
        raise ValueError("PB21J source bundle drifted")
    if _sha256_file(project_root / _HAZARD_CALIBRATION_FILE) != EXACT_HAZARD_CALIBRATION_SHA256:
        raise ValueError("shared hazard calibrator drifted")
    return {
        "registration": PB21J_REGISTRATION,
        "registration_sha256": EXACT_PB21J_REGISTRATION_SHA256,
        "attempt": PB21J_ATTEMPT,
        "attempt_sha256": EXACT_PB21J_ATTEMPT_SHA256,
        "result": PB21J_RESULT,
        "result_sha256": EXACT_PB21J_RESULT_SHA256,
        "source_bundle_sha256": EXACT_PB21J_SOURCE_BUNDLE_SHA256,
        "frozen_diagnosis": "representation_supported_deployability_unresolved",
        "NZ_representation_inference_passed": True,
        "NZ_deployability_unresolved": True,
        "selected_arm": None,
    }


def _diagnostic_source_bundle(project_root: Path) -> dict[str, object]:
    parent = pb21j._diagnostic_source_bundle(project_root)
    if parent.get("sha256") != EXACT_PB21J_SOURCE_BUNDLE_SHA256:
        raise ValueError("exact PB21J source bundle drifted")
    digest = sha256(b"IRPB21KNZDUALCAL\x01")
    digest.update(bytes.fromhex(EXACT_PB21J_SOURCE_BUNDLE_SHA256))
    files: dict[str, str] = {}
    for relative in _DIAGNOSTIC_FILES:
        observed = _sha256_file(project_root / relative)
        files[relative] = observed
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(observed))
    for relative in (PB21J_REGISTRATION, PB21J_ATTEMPT, PB21J_RESULT, _HAZARD_CALIBRATION_FILE):
        observed = _sha256_file(project_root / relative)
        files[relative] = observed
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(observed))
    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "PB21J_source_bundle": parent,
        "bound_files": files,
    }


def feature_contract() -> dict[str, object]:
    parent = pb21j.feature_contract()
    return {
        **parent,
        "arm_active_columns": {arm: list(ACTIVE_COLUMNS[arm]) for arm in ARMS},
        "NBZ_omitted_and_rejected": True,
    }


def schedule_record() -> dict[str, object]:
    return {
        "fit_cohorts": list(FIT_COHORTS),
        "cal_cohorts": list(CAL_COHORTS),
        "matched_pairs": [list(value) for value in COHORT_PAIRS],
        "initialization_seeds": list(INIT_SEEDS),
        "arms": list(ARMS),
        "head_count": TOTAL_HEADS,
        "equal_training_head_parameters": TRAIN_HEAD_PARAMETERS,
        "passes": PASSES,
        "roots_per_pass": ROOTS_PER_FIT,
        "root_batch_size": ROOT_BATCH_SIZE,
        "steps_per_head": STEPS_PER_HEAD,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "permutation_seed": PERMUTATION_SEED,
        "optimizer_and_loss": "exact_PB21J_AdamW_unweighted_all_action_BCE",
        "scorer_change_from_PB21J": False,
        "ordering": [
            "collect_all_FIT",
            "fit_all_45_masked_485_heads",
            "prune_all_heads_exactly",
            "collect_all_CAL",
            "fit_AA_and_MIX_on_each_of_9_frozen_NZ_heads",
            "construct_shared_DEV_once",
            "publish_create_only_evidence_NPZ",
            "reload_validate_evidence",
            "compute_all_DEV_gates_from_reloaded_evidence",
        ],
        "all_scorers_and_calibrators_complete_before_DEV_construction": True,
        "device": "cpu",
        "threads": 1,
        "cuda_visible_devices": "-1",
    }


def calibration_contract() -> dict[str, object]:
    return {
        "candidate_arm": ARM_NZ,
        "shared_frozen_scorer": True,
        "AA": "exact_existing_fit_train_only_per_action_affine_control",
        "MIX": {
            "fit_mode": "per_action_affine_equal_domain_mix_v1",
            "per_action_objective": (
                "0.5*mean_all_BCE+0.5*mean_current_factual_action_BCE+"
                "0.5e-6*((scale-1)^2+bias^2)"
            ),
            "dtype": "float64_cpu",
            "minimum_scale": CALIBRATION_MINIMUM_SCALE,
            "maximum_iterations": CALIBRATION_MAX_ITERATIONS,
            "tolerance": CALIBRATION_TOLERANCE,
            "projected_gradient_KKT_at_scale_bound": True,
            "bias_only_Newton_direction_at_satisfied_scale_bound": True,
            "every_action_requires_mixed_objective_improvement_gt_tolerance": True,
            "every_action_requires_strict_AA_and_factual_BCE_and_Brier_improvement": True,
            "aggregate_requires_strict_AA_and_factual_BCE_and_Brier_improvement": True,
            "whole_calibrator_identity_fallback_if_any_check_fails": True,
            "proposed_parameters_retained_diagnostically": True,
        },
        "only_MIX_may_nominate": True,
        "AA_never_nominates": True,
        "same_CAL_raw_NZ_logit_digest_required": True,
    }


def inference_contract() -> dict[str, object]:
    return {
        "raw_representation_comparisons": [
            {"name": name, "left": left, "right": right, "threshold": threshold}
            for name, left, right, threshold in PRIMARY_COMPARISONS
        ],
        "AA_minus_MIX_proper_score_comparisons": [
            {"name": name, "threshold": threshold}
            for name, threshold in CALIBRATOR_COMPARISONS
        ],
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "derangement_seed": DERANGEMENT_SEED,
        "episode_clusters": DEV_EPISODES,
        "roots_per_episode": ROOTS_PER_EPISODE,
        "fixed_replica_grid": "3_FIT_cohorts_x_3_initializations",
        "replicas_resampled": False,
        "quantile_method": QUANTILE_METHOD,
        "shared_int32_episode_index_matrix": True,
    }


def decision_contract() -> dict[str, object]:
    return {
        "only_candidate_path": "NZ_MIX",
        "recipe": "masked_superset_train_pruned_deployment_v1_plus_MIX_calibration",
        "requires": [
            "all_Z_BZ_NZ_U0_raw_ranking_gates_in_all_9_cells",
            "all_9_MIX_acceptance_and_unchanged_PB21J_NZ_deployability_gates",
            "NZ_minus_N_and_NZ_minus_U0_every_cell_and_grand_LCB",
            "AA_minus_MIX_every_cell_and_grand_LCB_proper_score_comparisons",
        ],
        "selection_by_observed_performance": False,
        "AA_can_nominate": False,
        "checkpoint_emitted": False,
        "candidate_publication_allowed": False,
        "same_DEV_retry_allowed": False,
    }


def canonical_paths(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve()
    return {
        "upstream_result": (root / CANONICAL_UPSTREAM_RESULT).resolve(),
        "registration": (root / CANONICAL_REGISTRATION).resolve(),
        "attempt": (root / CANONICAL_ATTEMPT).resolve(),
        "evidence": (root / CANONICAL_EVIDENCE).resolve(),
        "result": (root / CANONICAL_RESULT).resolve(),
    }


def _require_canonical_path(path: Path, expected: Path, *, role: str) -> Path:
    if path.resolve() != expected.resolve():
        raise ValueError(f"{role} must use the single canonical path {expected}")
    return path.resolve()


def registration_payload(project_root: Path) -> dict[str, object]:
    manifests = partition_manifests()
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": REGISTRATION_MODE,
        "diagnostic_mode": MODE,
        "classification": "development_diagnostic_registration_not_candidate",
        "create_only": True,
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "checkpoint_emitted": False,
        "run_identity": {
            "run_id": RUN_ID,
            "upstream_result": CANONICAL_UPSTREAM_RESULT,
            "registration": CANONICAL_REGISTRATION,
            "attempt": CANONICAL_ATTEMPT,
            "evidence": CANONICAL_EVIDENCE,
            "result": CANONICAL_RESULT,
            "alternate_paths_allowed": False,
            "retry_allowed": False,
        },
        "PB21J_parent": validate_pb21j_parent(project_root),
        "source_bundle": _diagnostic_source_bundle(project_root),
        "partitions": [
            {
                "label": spec.label,
                "role": spec.role,
                "split": spec.split.value,
                "seed_offset": spec.seed_offset,
                "episodes": spec.episodes,
                "stop_exclusive": spec.seed_offset + spec.episodes,
                "dataset_manifest_sha256": manifests[spec.label],
            }
            for spec in PARTITION_SPECS
        ],
        "occupied_range_audit": assert_no_range_collisions(),
        "feature_contract": feature_contract(),
        "schedule": schedule_record(),
        "pruning": {
            "recipe": "masked_superset_train_pruned_deployment_v1",
            "timing": "after_FIT_before_CAL_or_DEV",
            "active_columns": {arm: list(value) for arm, value in ACTIVE_COLUMNS.items()},
            "minimal_parameter_counts": PRUNED_PARAMETER_COUNTS,
            "every_FIT_row_batch_size_one_equivalence_max_abs": (
                pb21j.PRUNING_EQUIVALENCE_MAX_ABS
            ),
            "only_pruned_logits_used_for_CAL_and_DEV": True,
            "compact_from_initialization_equivalence_claimed": False,
        },
        "calibration": calibration_contract(),
        "inference": inference_contract(),
        "decision": decision_contract(),
        "evidence_schema": {
            "path": CANONICAL_EVIDENCE,
            "format": "deterministic_ZIP_DEFLATED_NPZ_v1",
            "allowed_keys": sorted(EVIDENCE_KEYS),
            "allowed_key_set_sha256": evidence_key_set_sha256(),
            "dtypes": dict(sorted(EVIDENCE_DTYPES.items())),
            "shapes": {
                name: list(shape) for name, shape in sorted(_expected_evidence_shapes().items())
            },
            "authoritative_for_numerical_DEV_decision": True,
            "published_and_reloaded_before_numerical_DEV_decision": True,
        },
        "scope": {
            "current_PB21J_DEV_reused": False,
            "CPU_QUAL_opened": False,
            "TEST_opened": False,
            "candidate_or_checkpoint_possible": False,
        },
    }


def register(registration: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(root)
    destination = _require_canonical_path(registration, paths["registration"], role="registration")
    for role in ("result", "attempt", "evidence"):
        if paths[role].exists():
            raise FileExistsError(f"canonical {role} already exists; retry is forbidden")
    development._publish_json_create_only(destination, registration_payload(root))


def validate_registration(registration: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    destination = _require_canonical_path(
        registration, canonical_paths(root)["registration"], role="registration"
    )
    if not destination.is_file():
        raise FileNotFoundError("PB21K registration is required")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if payload != registration_payload(root):
        raise ValueError("PB21K registration or bound source drifted")
    return {"path": str(destination), "sha256": _sha256_file(destination), "payload": payload}


@torch.no_grad()
def collect_fresh_live_tape(
    model: object,
    source: object,
    *,
    partition_label: str,
) -> pb21j.FreshLiveTape:
    """Exact PB21J batch-one replay over a registered PB21K partition."""

    registered = {spec.label: spec for spec in PARTITION_SPECS}
    if partition_label not in registered:
        raise ValueError("collector accepts only registered PB21K partitions")
    spec = registered[partition_label]
    contract = source.contract
    if (
        contract.name != spec.role
        or contract.dataset_config.split is not spec.split
        or contract.dataset_config.seed_offset != spec.seed_offset
        or contract.dataset_config.sequence_count != spec.episodes
    ):
        raise ValueError("source does not match its registered PB21K partition")
    was_training = model.training
    model.eval()
    beliefs: list[np.ndarray] = []
    updater_inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    parent_logits: list[np.ndarray] = []
    factual_actions: list[np.ndarray] = []
    episode_groups: list[str] = []
    root_ids: list[str] = []
    assignments: list[int] = []
    disagreements: list[int] = []
    device = torch.device("cpu")
    for batch in source.iter_all_action_batches(epoch=0, batch_size=1):
        state = model.init_state(batch.batch_size, device)
        prior_action = torch.zeros(batch.batch_size, dtype=torch.long, device=device)
        assignment_count = np.zeros(batch.batch_size, dtype=np.int64)
        disagreement_count = np.zeros(batch.batch_size, dtype=np.int64)
        for tick in range(batch.sequence_length):
            transitions = tuple(sequence.transitions[tick] for sequence in batch.sequences)
            applied = torch.tensor(
                [strict.control_action_class(value.applied_control) for value in transitions],
                dtype=torch.long,
                device=device,
            )
            step = strict.strict_live_reference_step(
                model,
                tuple(value.observation.rgb for value in transitions),
                state,
                prior_action,
                first_tick=tick == 0,
            )
            state = step.state
            table = step.output.outcome_table
            if tick >= batch.burn_in_steps:
                beliefs.append(step.output.belief.detach().to(torch.float64).cpu().numpy())
                updater_inputs.append(step.updater_input.detach().to(torch.float64).cpu().numpy())
                parent_logits.append(step.canonical_raw_hazard_logits)
                factual_actions.append(applied.detach().cpu().numpy().astype(np.int64, copy=False))
                targets.append(
                    np.asarray(
                        [
                            [
                                strict.transition_hazard(branch.event_targets)
                                for branch in transition.counterfactual_targets
                            ]
                            for transition in transitions
                        ],
                        dtype=np.float64,
                    )
                )
                assignments.extend(int(value) for value in assignment_count)
                disagreements.extend(int(value) for value in disagreement_count)
                for sequence, transition in zip(batch.sequences, transitions, strict=True):
                    episode_groups.append(
                        f"{partition_label}:episode:{sequence.episode_seed}"
                    )
                    root_ids.append(transition.root_state_sha256)
            strict._install_applied_pending(state, table, applied)
            for index, (sequence, transition) in enumerate(
                zip(batch.sequences, transitions, strict=True)
            ):
                assignment_count[index] += int(
                    strict.behavior_assignment(
                        source.manifest_sha256,
                        episode_seed=sequence.episode_seed,
                        tick=tick,
                    )
                )
                disagreement_count[index] += int(
                    strict.control_action_class(transition.applied_control)
                    != strict.control_action_class(transition.action_target)
                )
            prior_action = applied
    if was_training:
        model.train()
    tape = pb21j.FreshLiveTape(
        beliefs=np.concatenate(beliefs, axis=0),
        updater_inputs=np.concatenate(updater_inputs, axis=0),
        hazard_targets=np.concatenate(targets, axis=0),
        parent_raw_logits=np.concatenate(parent_logits, axis=0),
        factual_actions=np.concatenate(factual_actions, axis=0),
        episode_group_ids=tuple(episode_groups),
        root_state_ids=tuple(root_ids),
        prior_assignment_count=np.asarray(assignments, dtype=np.int64),
        prior_disagreement_count=np.asarray(disagreements, dtype=np.int64),
    )
    expected_roots = spec.episodes * ROOTS_PER_EPISODE
    if len(tape.hazard_targets) != expected_roots:
        raise RuntimeError("fresh strict-live root count drifted")
    if len(set(tape.root_state_ids)) != len(tape.root_state_ids):
        raise RuntimeError("fresh strict-live replay contains duplicate roots")
    return tape


def superset_features(tape: pb21j.FreshLiveTape, arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(f"unknown or forbidden PB21K arm: {arm}")
    return pb21j.superset_features(tape, arm)


def initialized_head(seed: int) -> pb21j.MaskedSupersetHazardHead:
    if seed not in INIT_SEEDS:
        raise ValueError("head initialization seed is not registered")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return pb21j.MaskedSupersetHazardHead()


def deterministic_permutations(root_count: int = ROOTS_PER_FIT) -> tuple[Tensor, ...]:
    if root_count != ROOTS_PER_FIT:
        raise ValueError("PB21K requires exactly 3072 FIT roots")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PERMUTATION_SEED)
    values = tuple(torch.randperm(root_count, generator=generator) for _ in range(PASSES))
    if any(int(value.unique().numel()) != root_count for value in values):
        raise RuntimeError("FIT permutation is not a complete bijection")
    return values


def _binary_bce(targets: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.asarray(targets, dtype=np.float64)
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
    if target.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("BCE inputs must be finite and aligned")
    return float(np.mean(-(target * np.log(probability) + (1.0 - target) * np.log1p(-probability))))


def _binary_brier(targets: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.asarray(targets, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if target.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("Brier inputs must be finite and aligned")
    return float(np.mean(np.square(target - probability)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return torch.sigmoid(torch.from_numpy(np.asarray(values, dtype=np.float64))).numpy()


def _factual_vectors(
    targets: np.ndarray,
    values: np.ndarray,
    factual_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(targets, dtype=np.float64)
    table = np.asarray(values, dtype=np.float64)
    actions = np.asarray(factual_actions, dtype=np.int64)
    if target.shape != table.shape or target.shape != (len(actions), ACTION_COUNT):
        raise ValueError("factual extraction geometry drifted")
    rows = np.arange(len(actions), dtype=np.int64)
    return target[rows, actions], table[rows, actions]


def _mix_weights(factual_actions: np.ndarray, action: int) -> np.ndarray:
    actions = np.asarray(factual_actions, dtype=np.int64)
    mask = actions == action
    factual_count = int(mask.sum())
    if factual_count < CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION:
        raise ValueError(f"action {action} has insufficient factual CAL support")
    weights = np.full(len(actions), 0.5 / len(actions), dtype=np.float64)
    weights[mask] += 0.5 / factual_count
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-14):
        raise RuntimeError("MIX weights must sum to one within each action")
    return weights


def _mixed_objective(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    scale: Tensor,
    bias: Tensor,
) -> Tensor:
    calibrated = scale * logits + bias
    data = torch.sum(weights * (F.softplus(calibrated) - targets * calibrated))
    penalty = 0.5 * CALIBRATION_L2 * ((scale - 1.0).square() + bias.square())
    return data + penalty


def _fit_one_mixed_action(
    logits: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, dict[str, object]]:
    """Deterministic constrained Newton fit with the scale-bound KKT rule."""

    x = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    y = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
    w = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    if x.ndim != 1 or x.shape != y.shape or x.shape != w.shape or len(x) < 1:
        raise ValueError("MIX action arrays must be non-empty aligned vectors")
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(w).all()):
        raise ValueError("MIX action arrays must be finite")
    if not bool(((y == 0.0) | (y == 1.0)).all()) or bool((w <= 0.0).any()):
        raise ValueError("MIX targets must be binary and weights positive")
    if not math.isclose(float(w.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("MIX weights must sum to one")
    scale = x.new_tensor(1.0)
    bias = x.new_tensor(0.0)
    current = _mixed_objective(x, y, w, scale, bias)
    if not bool(torch.isfinite(current)):
        raise FloatingPointError("non-finite initial MIX objective")
    iterations = 0
    terminated = "maximum_iterations"
    for iteration in range(CALIBRATION_MAX_ITERATIONS):
        iterations = iteration + 1
        calibrated = scale * x + bias
        probability = torch.sigmoid(calibrated)
        residual = probability - y
        curvature = probability * (1.0 - probability)
        grad_scale = torch.sum(w * residual * x) + CALIBRATION_L2 * (scale - 1.0)
        grad_bias = torch.sum(w * residual) + CALIBRATION_L2 * bias
        if not bool(torch.isfinite(grad_scale) and torch.isfinite(grad_bias)):
            raise FloatingPointError("non-finite MIX projected gradient")
        at_bound = float(scale) <= CALIBRATION_MINIMUM_SCALE + CALIBRATION_TOLERANCE
        scale_kkt = at_bound and float(grad_scale) >= 0.0
        projected_scale_gradient = 0.0 if scale_kkt else float(grad_scale)
        if max(abs(projected_scale_gradient), abs(float(grad_bias))) <= CALIBRATION_TOLERANCE:
            terminated = "projected_gradient_KKT"
            break
        h_ss = torch.sum(w * curvature * x.square()) + CALIBRATION_L2
        h_sb = torch.sum(w * curvature * x)
        h_bb = torch.sum(w * curvature) + CALIBRATION_L2
        if scale_kkt:
            delta_scale = x.new_tensor(0.0)
            delta_bias = grad_bias / h_bb
        else:
            determinant = h_ss * h_bb - h_sb.square()
            if float(determinant) <= 0.0:
                raise RuntimeError("MIX calibration Hessian is not positive definite")
            delta_scale = (h_bb * grad_scale - h_sb * grad_bias) / determinant
            delta_bias = (h_ss * grad_bias - h_sb * grad_scale) / determinant
        accepted_step = False
        step = 1.0
        for _ in range(50):
            candidate_scale = torch.clamp(
                scale - step * delta_scale, min=CALIBRATION_MINIMUM_SCALE
            )
            candidate_bias = bias - step * delta_bias
            candidate = _mixed_objective(x, y, w, candidate_scale, candidate_bias)
            if not bool(torch.isfinite(candidate)):
                step *= 0.5
                continue
            if float(candidate) <= float(current):
                improvement = float(current - candidate)
                parameter_move = max(
                    abs(float(candidate_scale - scale)),
                    abs(float(candidate_bias - bias)),
                )
                scale = candidate_scale
                bias = candidate_bias
                current = candidate
                accepted_step = True
                if (
                    improvement <= CALIBRATION_TOLERANCE
                    and parameter_move <= CALIBRATION_TOLERANCE
                ):
                    terminated = "objective_tolerance"
                break
            step *= 0.5
        if not accepted_step:
            raise RuntimeError("MIX calibration line search exhausted")
        if terminated == "objective_tolerance":
            break
    if terminated == "maximum_iterations":
        raise RuntimeError("MIX calibration exhausted maximum iterations")
    return float(scale), float(bias), {
        "iterations": iterations,
        "termination": terminated,
        "projected_scale_constraint": True,
        "objective": float(current),
    }


@dataclass(frozen=True)
class DualDomainActionReport:
    action_id: int
    all_observations: int
    all_positives: int
    all_negatives: int
    factual_observations: int
    factual_positives: int
    factual_negatives: int
    identity_mixed_objective: float
    proposed_mixed_objective: float
    identity_all_BCE: float
    proposed_all_BCE: float
    identity_all_Brier: float
    proposed_all_Brier: float
    identity_factual_BCE: float
    proposed_factual_BCE: float
    identity_factual_Brier: float
    proposed_factual_Brier: float
    accepted: bool
    optimizer: Mapping[str, object]

    def __post_init__(self) -> None:
        numeric = (
            self.identity_mixed_objective,
            self.proposed_mixed_objective,
            self.identity_all_BCE,
            self.proposed_all_BCE,
            self.identity_all_Brier,
            self.proposed_all_Brier,
            self.identity_factual_BCE,
            self.proposed_factual_BCE,
            self.identity_factual_Brier,
            self.proposed_factual_Brier,
        )
        if self.action_id not in range(ACTION_COUNT) or self.all_observations < 1:
            raise ValueError("invalid dual-domain action identity/count")
        if self.all_positives + self.all_negatives != self.all_observations:
            raise ValueError("all-action class counts must sum to observations")
        if self.factual_positives + self.factual_negatives != self.factual_observations:
            raise ValueError("factual class counts must sum to observations")
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("dual-domain action report must be finite and nonnegative")


@dataclass(frozen=True)
class DualDomainAffineCalibrator:
    """Truthful local type for MIX; rejected fits apply exact identity globally."""

    scales: tuple[float, ...]
    biases: tuple[float, ...]
    proposed_scales: tuple[float, ...]
    proposed_biases: tuple[float, ...]
    provenance: TrainCalibrationProvenance
    per_action: tuple[DualDomainActionReport, ...]
    aggregate: Mapping[str, float]
    accepted: bool
    fit_mode: str = "per_action_affine_equal_domain_mix_v1"

    def __post_init__(self) -> None:
        if not all(
            len(value) == ACTION_COUNT
            for value in (self.scales, self.biases, self.proposed_scales, self.proposed_biases)
        ):
            raise ValueError("MIX parameter vectors must cover every action")
        if tuple(value.action_id for value in self.per_action) != tuple(range(ACTION_COUNT)):
            raise ValueError("MIX reports must use canonical action order")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scales + self.proposed_scales):
            raise ValueError("MIX scales must be finite and positive")
        if any(not math.isfinite(value) for value in self.biases + self.proposed_biases):
            raise ValueError("MIX biases must be finite")
        if any(not math.isfinite(float(value)) for value in self.aggregate.values()):
            raise ValueError("MIX aggregate report must be finite")
        if self.accepted:
            if self.scales != self.proposed_scales or self.biases != self.proposed_biases:
                raise ValueError("accepted MIX parameters must equal the proposal")
        elif self.scales != (1.0,) * ACTION_COUNT or self.biases != (0.0,) * ACTION_COUNT:
            raise ValueError("rejected MIX calibrator must apply whole-calibrator identity")

    def transform_all_actions(self, logits: Tensor, action_ids: Tensor) -> Tensor:
        if not torch.is_floating_point(logits):
            raise ValueError("logits must be floating point")
        raw_ids = torch.as_tensor(action_ids, device=logits.device)
        if raw_ids.dtype not in _INTEGER_TORCH_DTYPES:
            raise ValueError("action IDs must use an integer dtype")
        table = logits.squeeze(-1) if logits.dim() >= 2 and logits.shape[-1] == 1 else logits
        semantic = raw_ids
        if table.shape != semantic.shape or table.shape[-1] != ACTION_COUNT:
            raise ValueError("logits/action IDs must be aligned complete action tables")
        ids = semantic.to(dtype=torch.long).reshape(-1, ACTION_COUNT)
        canonical = torch.arange(ACTION_COUNT, device=logits.device).expand(len(ids), -1)
        if not bool(ids.sort(dim=1).values.eq(canonical).all()):
            raise ValueError("each table must contain every semantic action exactly once")
        scales = torch.as_tensor(self.scales, dtype=table.dtype, device=table.device)
        biases = torch.as_tensor(self.biases, dtype=table.dtype, device=table.device)
        transformed = table * scales[semantic.to(dtype=torch.long)] + biases[semantic.to(dtype=torch.long)]
        return transformed.unsqueeze(-1) if logits.dim() >= 2 and logits.shape[-1] == 1 else transformed

    def export_parameters(self) -> dict[str, object]:
        return {
            "schema": "irene.hazard.per_action_affine.equal_domain_mix.v1",
            "fit_mode": self.fit_mode,
            "accepted": self.accepted,
            "applied_scale": list(self.scales),
            "applied_bias": list(self.biases),
            "proposed_scale": list(self.proposed_scales),
            "proposed_bias": list(self.proposed_biases),
            "aggregate": dict(self.aggregate),
            "per_action": [
                {
                    "action_id": value.action_id,
                    "all_observations": value.all_observations,
                    "all_positives": value.all_positives,
                    "all_negatives": value.all_negatives,
                    "factual_observations": value.factual_observations,
                    "factual_positives": value.factual_positives,
                    "factual_negatives": value.factual_negatives,
                    "identity_mixed_objective": value.identity_mixed_objective,
                    "proposed_mixed_objective": value.proposed_mixed_objective,
                    "identity_all_BCE": value.identity_all_BCE,
                    "proposed_all_BCE": value.proposed_all_BCE,
                    "identity_all_Brier": value.identity_all_Brier,
                    "proposed_all_Brier": value.proposed_all_Brier,
                    "identity_factual_BCE": value.identity_factual_BCE,
                    "proposed_factual_BCE": value.proposed_factual_BCE,
                    "identity_factual_Brier": value.identity_factual_Brier,
                    "proposed_factual_Brier": value.proposed_factual_Brier,
                    "accepted": value.accepted,
                    "optimizer": dict(value.optimizer),
                }
                for value in self.per_action
            ],
            "provenance": {
                "source_namespace": self.provenance.source_namespace,
                "source_split": self.provenance.source_split,
                "source_partition": self.provenance.source_partition,
                "calibration_group_sha256": self.provenance.calibration_group_digest,
                "upstream_model_fit_group_sha256": self.provenance.upstream_model_fit_group_digest,
                "upstream_checkpoint_sha256": self.provenance.upstream_checkpoint_sha256,
                "dataset_manifest_sha256": self.provenance.dataset_manifest_sha256,
                "source_bundle_sha256": self.provenance.source_bundle_sha256,
                "partition_algorithm": self.provenance.partition_algorithm,
            },
        }


def calibration_provenance(
    head: pb21j.PrunedHazardHead,
    cal_tape: pb21j.FreshLiveTape,
    fit_tape: pb21j.FreshLiveTape,
    *,
    cal_manifest_sha256: str,
    source_bundle_sha256: str,
) -> TrainCalibrationProvenance:
    provenance = TrainCalibrationProvenance(
        source_namespace="maze_chase.pb21k.train-only.v1",
        source_split="TRAIN",
        source_partition="TRAIN-CAL",
        calibration_group_ids=tuple(
            group for group in cal_tape.episode_group_ids for _ in range(ACTION_COUNT)
        ),
        upstream_model_fit_group_ids=tuple(sorted(set(fit_tape.episode_group_ids))),
        upstream_checkpoint_sha256=_state_sha256(head),
        dataset_manifest_sha256=cal_manifest_sha256,
        source_bundle_sha256=source_bundle_sha256,
        partition_algorithm=PARTITION_ALGORITHM,
    )
    provenance.validate(len(cal_tape.hazard_targets) * ACTION_COUNT)
    return provenance


def fit_aa_calibrator(
    raw_logits: np.ndarray,
    cal_tape: pb21j.FreshLiveTape,
    provenance: TrainCalibrationProvenance,
) -> tuple[PerActionAffineHazardCalibrator, dict[str, object]]:
    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.shape != cal_tape.hazard_targets.shape:
        raise ValueError("AA CAL logits must align with all-action targets")
    action_ids = np.broadcast_to(np.arange(ACTION_COUNT, dtype=np.int64), logits.shape)
    calibrator = fit_train_only_per_action_affine(
        torch.from_numpy(logits.reshape(-1)),
        torch.from_numpy(cal_tape.hazard_targets.reshape(-1)),
        torch.from_numpy(action_ids.reshape(-1)),
        provenance=provenance,
        action_count=ACTION_COUNT,
        l2_regularization=CALIBRATION_L2,
        minimum_scale=CALIBRATION_MINIMUM_SCALE,
        minimum_examples_per_action=CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
        minimum_class_examples=CALIBRATION_MINIMUM_CLASS_EXAMPLES,
        max_iterations=CALIBRATION_MAX_ITERATIONS,
        tolerance=CALIBRATION_TOLERANCE,
        fit_mode="per_action_affine",
    )
    finite = all(math.isfinite(value) for value in calibrator.scales + calibrator.biases)
    return calibrator, {
        "role": "sealed_all_action_control_never_nominates",
        "raw_CAL_logit_sha256": _array_sha256(logits),
        "parameters": calibrator.export_parameters(),
        "finite": finite,
        "accepted": calibrator.accepted and finite and all(value > 0.0 for value in calibrator.scales),
    }


def fit_mix_calibrator(
    raw_logits: np.ndarray,
    cal_tape: pb21j.FreshLiveTape,
    provenance: TrainCalibrationProvenance,
) -> tuple[DualDomainAffineCalibrator, dict[str, object]]:
    logits = np.asarray(raw_logits, dtype=np.float64)
    targets = np.asarray(cal_tape.hazard_targets, dtype=np.float64)
    actions = np.asarray(cal_tape.factual_actions, dtype=np.int64)
    if logits.shape != (ROOTS_PER_CAL, ACTION_COUNT) or targets.shape != logits.shape:
        raise ValueError("MIX CAL table geometry drifted")
    if actions.shape != (ROOTS_PER_CAL,):
        raise ValueError("MIX factual action geometry drifted")
    provenance.validate(logits.size)
    proposed_scales: list[float] = []
    proposed_biases: list[float] = []
    reports: list[DualDomainActionReport] = []
    raw_probability = _sigmoid(logits)
    proposed_logits = np.empty_like(logits)
    for action in range(ACTION_COUNT):
        weights = _mix_weights(actions, action)
        mask = actions == action
        factual_targets = targets[mask, action]
        factual_positives = int(factual_targets.sum())
        factual_negatives = len(factual_targets) - factual_positives
        all_positives = int(targets[:, action].sum())
        all_negatives = len(targets) - all_positives
        if (
            factual_positives < CALIBRATION_MINIMUM_CLASS_EXAMPLES
            or factual_negatives < CALIBRATION_MINIMUM_CLASS_EXAMPLES
            or all_positives < CALIBRATION_MINIMUM_CLASS_EXAMPLES
            or all_negatives < CALIBRATION_MINIMUM_CLASS_EXAMPLES
        ):
            raise ValueError(f"action {action} lacks required CAL class support")
        scale, bias, optimizer = _fit_one_mixed_action(
            logits[:, action], targets[:, action], weights
        )
        proposed_scales.append(scale)
        proposed_biases.append(bias)
        proposal = scale * logits[:, action] + bias
        proposed_logits[:, action] = proposal
        proposed_probability = _sigmoid(proposal)
        identity_mixed = float(
            _mixed_objective(
                torch.from_numpy(logits[:, action]),
                torch.from_numpy(targets[:, action]),
                torch.from_numpy(weights),
                torch.tensor(1.0, dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
            )
        )
        proposed_mixed = float(
            _mixed_objective(
                torch.from_numpy(logits[:, action]),
                torch.from_numpy(targets[:, action]),
                torch.from_numpy(weights),
                torch.tensor(scale, dtype=torch.float64),
                torch.tensor(bias, dtype=torch.float64),
            )
        )
        identity_all_bce = _binary_bce(targets[:, action], raw_probability[:, action])
        proposed_all_bce = _binary_bce(targets[:, action], proposed_probability)
        identity_all_brier = _binary_brier(targets[:, action], raw_probability[:, action])
        proposed_all_brier = _binary_brier(targets[:, action], proposed_probability)
        identity_factual_bce = _binary_bce(factual_targets, raw_probability[mask, action])
        proposed_factual_bce = _binary_bce(factual_targets, proposed_probability[mask])
        identity_factual_brier = _binary_brier(factual_targets, raw_probability[mask, action])
        proposed_factual_brier = _binary_brier(factual_targets, proposed_probability[mask])
        accepted = (
            identity_mixed - proposed_mixed > CALIBRATION_TOLERANCE
            and proposed_all_bce < identity_all_bce
            and proposed_all_brier < identity_all_brier
            and proposed_factual_bce < identity_factual_bce
            and proposed_factual_brier < identity_factual_brier
        )
        reports.append(
            DualDomainActionReport(
                action_id=action,
                all_observations=len(targets),
                all_positives=all_positives,
                all_negatives=all_negatives,
                factual_observations=int(mask.sum()),
                factual_positives=factual_positives,
                factual_negatives=factual_negatives,
                identity_mixed_objective=identity_mixed,
                proposed_mixed_objective=proposed_mixed,
                identity_all_BCE=identity_all_bce,
                proposed_all_BCE=proposed_all_bce,
                identity_all_Brier=identity_all_brier,
                proposed_all_Brier=proposed_all_brier,
                identity_factual_BCE=identity_factual_bce,
                proposed_factual_BCE=proposed_factual_bce,
                identity_factual_Brier=identity_factual_brier,
                proposed_factual_Brier=proposed_factual_brier,
                accepted=accepted,
                optimizer=optimizer,
            )
        )
    proposed_probability = _sigmoid(proposed_logits)
    factual_targets, raw_factual = _factual_vectors(targets, raw_probability, actions)
    _, proposed_factual = _factual_vectors(targets, proposed_probability, actions)
    aggregate = {
        "identity_mixed_objective_mean": float(
            np.mean([value.identity_mixed_objective for value in reports])
        ),
        "proposed_mixed_objective_mean": float(
            np.mean([value.proposed_mixed_objective for value in reports])
        ),
        "identity_all_BCE": _binary_bce(targets, raw_probability),
        "proposed_all_BCE": _binary_bce(targets, proposed_probability),
        "identity_all_Brier": _binary_brier(targets, raw_probability),
        "proposed_all_Brier": _binary_brier(targets, proposed_probability),
        "identity_factual_BCE": _binary_bce(factual_targets, raw_factual),
        "proposed_factual_BCE": _binary_bce(factual_targets, proposed_factual),
        "identity_factual_Brier": _binary_brier(factual_targets, raw_factual),
        "proposed_factual_Brier": _binary_brier(factual_targets, proposed_factual),
    }
    aggregate_checks = {
        "mixed_objective": (
            aggregate["identity_mixed_objective_mean"]
            - aggregate["proposed_mixed_objective_mean"]
            > CALIBRATION_TOLERANCE
        ),
        "all_BCE": aggregate["proposed_all_BCE"] < aggregate["identity_all_BCE"],
        "all_Brier": aggregate["proposed_all_Brier"] < aggregate["identity_all_Brier"],
        "factual_BCE": (
            aggregate["proposed_factual_BCE"] < aggregate["identity_factual_BCE"]
        ),
        "factual_Brier": (
            aggregate["proposed_factual_Brier"] < aggregate["identity_factual_Brier"]
        ),
    }
    accepted = all(value.accepted for value in reports) and all(aggregate_checks.values())
    applied_scales = tuple(proposed_scales) if accepted else (1.0,) * ACTION_COUNT
    applied_biases = tuple(proposed_biases) if accepted else (0.0,) * ACTION_COUNT
    calibrator = DualDomainAffineCalibrator(
        scales=applied_scales,
        biases=applied_biases,
        proposed_scales=tuple(proposed_scales),
        proposed_biases=tuple(proposed_biases),
        provenance=provenance,
        per_action=tuple(reports),
        aggregate=aggregate,
        accepted=accepted,
    )
    finite = pb21j._all_numeric_finite(calibrator.export_parameters())
    return calibrator, {
        "role": "sole_preregistered_calibration_candidate",
        "raw_CAL_logit_sha256": _array_sha256(logits),
        "parameters": calibrator.export_parameters(),
        "per_action_acceptance": [value.accepted for value in reports],
        "aggregate_checks": aggregate_checks,
        "whole_calibrator_identity_fallback_applied": not accepted,
        "finite": finite,
        "accepted": accepted and finite,
    }


def calibrated_output_tables(
    calibrator: PerActionAffineHazardCalibrator | DualDomainAffineCalibrator,
    raw_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_ids = torch.arange(ACTION_COUNT, dtype=torch.long).expand(len(raw_logits), -1)
    transformed = calibrator.transform_all_actions(torch.from_numpy(raw_logits), action_ids)
    logits = transformed.to(torch.float64).numpy()
    probabilities = _sigmoid(logits)
    if (
        logits.shape != np.asarray(raw_logits).shape
        or probabilities.shape != logits.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("calibrator emitted invalid logits or probabilities")
    return logits, probabilities


def bootstrap_index_matrix(
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    groups: int = DEV_EPISODES,
) -> np.ndarray:
    if resamples < 1 or groups < 2:
        raise ValueError("bootstrap geometry must be positive")
    result = np.random.default_rng(seed).integers(
        0, groups, size=(resamples, groups), dtype=np.int32
    )
    if result.dtype != np.int32:
        raise RuntimeError("bootstrap matrix must remain int32")
    return result


def episode_derangement_indices(group_count: int = DEV_EPISODES) -> np.ndarray:
    if group_count < 2:
        raise ValueError("derangement requires at least two groups")
    rng = np.random.default_rng(DERANGEMENT_SEED)
    result = np.empty((DERANGEMENT_REPETITIONS, group_count), dtype=np.int32)
    identity = np.arange(group_count, dtype=np.int32)
    for repetition in range(DERANGEMENT_REPETITIONS):
        permutation = rng.permutation(group_count).astype(np.int32, copy=False)
        while np.any(permutation == identity):
            permutation = rng.permutation(group_count).astype(np.int32, copy=False)
        result[repetition] = permutation
    return result


def _canonical_evidence_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in {"b", "i", "u", "f", "S"}:
        raise TypeError("evidence arrays must be non-object numeric or byte-string arrays")
    if array.dtype.kind in {"i", "u", "f"}:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = array.copy() if array.ndim == 0 else np.ascontiguousarray(array)
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("evidence arrays must be finite")
    return array


def _byte_string_array(values: Sequence[str], *, width: int) -> np.ndarray:
    encoded = [value.encode("utf-8") for value in values]
    if any(len(value) > width for value in encoded):
        raise ValueError(f"identifier exceeds frozen S{width} evidence width")
    return np.asarray(encoded, dtype=f"S{width}")


def build_evidence_arrays(
    *,
    fit_tapes: Mapping[str, pb21j.FreshLiveTape],
    cal_tapes: Mapping[str, pb21j.FreshLiveTape],
    dev_tape: pb21j.FreshLiveTape,
    cells: Sequence[str],
    cal_raw_logits: Mapping[str, np.ndarray],
    calibrators: Mapping[
        str,
        Mapping[str, PerActionAffineHazardCalibrator | DualDomainAffineCalibrator],
    ],
    dev_raw_logits: Mapping[str, Mapping[str, np.ndarray]],
    dev_raw_probabilities: Mapping[str, Mapping[str, np.ndarray]],
    nz_calibrated_logits: Mapping[str, Mapping[str, np.ndarray]],
    nz_calibrated_probabilities: Mapping[str, Mapping[str, np.ndarray]],
    sampled: np.ndarray,
    derangements: np.ndarray,
) -> dict[str, np.ndarray]:
    cell_order = tuple(cells)
    expected_cells = tuple(
        f"{fit_label}/init-{seed}" for fit_label in FIT_COHORTS for seed in INIT_SEEDS
    )
    if cell_order != expected_cells:
        raise ValueError("evidence requires canonical 3x3 cell order")
    fit_index = np.asarray(
        [FIT_COHORTS.index(cell.split("/", 1)[0]) for cell in cell_order], dtype=np.int16
    )
    cal_index = fit_index.copy()
    init_seed = np.asarray([int(cell.rsplit("-", 1)[1]) for cell in cell_order], dtype=np.int32)
    ordered_groups = tuple(dict.fromkeys(dev_tape.episode_group_ids))
    group_to_index = {value: index for index, value in enumerate(ordered_groups)}
    cluster_ordinal = np.asarray(
        [group_to_index[value] for value in dev_tape.episode_group_ids], dtype=np.int32
    )
    aa_mix = tuple(CALIBRATORS)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.uint16),
        "action_ids": np.arange(ACTION_COUNT, dtype=np.int16),
        "fit_ids": _byte_string_array(FIT_COHORTS, width=2),
        "cal_ids": _byte_string_array(CAL_COHORTS, width=2),
        "cell_ids": _byte_string_array(cell_order, width=16),
        "cell_fit_index": fit_index,
        "cell_cal_index": cal_index,
        "cell_init_seed": init_seed,
        "arm_ids": _byte_string_array(ARMS, width=3),
        "calibrator_ids": _byte_string_array(aa_mix, width=3),
        "fit_targets": np.stack(
            [fit_tapes[label].hazard_targets for label in FIT_COHORTS]
        ),
        "fit_actions": np.stack(
            [fit_tapes[label].factual_actions for label in FIT_COHORTS]
        ),
        "fit_root_ids": np.stack(
            [_byte_string_array(fit_tapes[label].root_state_ids, width=64) for label in FIT_COHORTS]
        ),
        "cal_targets": np.stack(
            [cal_tapes[label].hazard_targets for label in CAL_COHORTS]
        ),
        "cal_actions": np.stack(
            [cal_tapes[label].factual_actions for label in CAL_COHORTS]
        ),
        "cal_root_ids": np.stack(
            [_byte_string_array(cal_tapes[label].root_state_ids, width=64) for label in CAL_COHORTS]
        ),
        "cal_nz_raw_logits": np.stack([cal_raw_logits[cell] for cell in cell_order]),
        "calibrator_scales": np.asarray(
            [
                [calibrators[cell][name].scales for cell in cell_order]
                for name in aa_mix
            ],
            dtype=np.float64,
        ),
        "calibrator_biases": np.asarray(
            [
                [calibrators[cell][name].biases for cell in cell_order]
                for name in aa_mix
            ],
            dtype=np.float64,
        ),
        "calibrator_accepted": np.asarray(
            [
                [calibrators[cell][name].accepted for cell in cell_order]
                for name in aa_mix
            ],
            dtype=np.bool_,
        ),
        "mix_proposed_scales": np.asarray(
            [
                tuple(
                    getattr(calibrators[cell]["MIX"], "proposed_scales")
                )
                for cell in cell_order
            ],
            dtype=np.float64,
        ),
        "mix_proposed_biases": np.asarray(
            [
                tuple(
                    getattr(calibrators[cell]["MIX"], "proposed_biases")
                )
                for cell in cell_order
            ],
            dtype=np.float64,
        ),
        "mix_per_action_accepted": np.asarray(
            [
                [value.accepted for value in getattr(calibrators[cell]["MIX"], "per_action")]
                for cell in cell_order
            ],
            dtype=np.bool_,
        ),
        "dev_targets": dev_tape.hazard_targets,
        "dev_actions": dev_tape.factual_actions,
        "dev_root_ids": _byte_string_array(dev_tape.root_state_ids, width=64),
        "dev_cluster_ordinal": cluster_ordinal,
        "dev_cluster_ids": _byte_string_array(ordered_groups, width=64),
        "dev_raw_logits": np.stack(
            [
                np.stack([dev_raw_logits[cell][arm] for arm in ARMS])
                for cell in cell_order
            ]
        ),
        "dev_raw_probabilities": np.stack(
            [
                np.stack([dev_raw_probabilities[cell][arm] for arm in ARMS])
                for cell in cell_order
            ]
        ),
        "nz_calibrated_logits": np.stack(
            [
                np.stack([nz_calibrated_logits[cell][name] for cell in cell_order])
                for name in aa_mix
            ]
        ),
        "nz_calibrated_probabilities": np.stack(
            [
                np.stack([nz_calibrated_probabilities[cell][name] for cell in cell_order])
                for name in aa_mix
            ]
        ),
        "bootstrap_indices": sampled,
        "episode_derangements": derangements,
    }
    return {name: _canonical_evidence_array(value) for name, value in arrays.items()}


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }


def _array_manifest_sha256(manifest: Mapping[str, Mapping[str, object]]) -> str:
    encoded = json.dumps(
        {name: dict(manifest[name]) for name in sorted(manifest)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRPB21KARRAYMANIFEST\x01" + encoded).hexdigest()


def publish_evidence_create_only(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    attempt_sha256: str,
) -> dict[str, object]:
    if set(arrays) != EVIDENCE_KEYS:
        raise ValueError("evidence key set drifted")
    canonical = {name: _canonical_evidence_array(value) for name, value in arrays.items()}
    manifest = _array_manifest(canonical)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("canonical evidence already exists; retry is forbidden")
    temporary = path.with_name(f".{path.name}.{attempt_sha256[:16]}.partial")
    if temporary.exists():
        raise FileExistsError("partial evidence exists; retry is forbidden")
    with zipfile.ZipFile(
        temporary,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for name in sorted(canonical):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.lib.format.write_array(member, canonical[name], allow_pickle=False)
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "byte_length": path.stat().st_size,
        "format": "deterministic_ZIP_DEFLATED_NPZ_v1",
        "allowed_key_set_sha256": evidence_key_set_sha256(),
        "arrays": manifest,
        "array_manifest_sha256": _array_manifest_sha256(manifest),
        "published_create_only_atomically": True,
    }


def _decode_byte_strings(values: np.ndarray) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError("identifier arrays must be one-dimensional byte strings")
    return tuple(value.rstrip(b"\x00").decode("utf-8") for value in array.tolist())


def _expected_evidence_shapes() -> dict[str, tuple[int, ...]]:
    cells = len(FIT_COHORTS) * len(INIT_SEEDS)
    return {
        "schema_version": (),
        "action_ids": (ACTION_COUNT,),
        "fit_ids": (len(FIT_COHORTS),),
        "cal_ids": (len(CAL_COHORTS),),
        "cell_ids": (cells,),
        "cell_fit_index": (cells,),
        "cell_cal_index": (cells,),
        "cell_init_seed": (cells,),
        "arm_ids": (len(ARMS),),
        "calibrator_ids": (len(CALIBRATORS),),
        "fit_targets": (len(FIT_COHORTS), ROOTS_PER_FIT, ACTION_COUNT),
        "fit_actions": (len(FIT_COHORTS), ROOTS_PER_FIT),
        "fit_root_ids": (len(FIT_COHORTS), ROOTS_PER_FIT),
        "cal_targets": (len(CAL_COHORTS), ROOTS_PER_CAL, ACTION_COUNT),
        "cal_actions": (len(CAL_COHORTS), ROOTS_PER_CAL),
        "cal_root_ids": (len(CAL_COHORTS), ROOTS_PER_CAL),
        "cal_nz_raw_logits": (cells, ROOTS_PER_CAL, ACTION_COUNT),
        "calibrator_scales": (len(CALIBRATORS), cells, ACTION_COUNT),
        "calibrator_biases": (len(CALIBRATORS), cells, ACTION_COUNT),
        "calibrator_accepted": (len(CALIBRATORS), cells),
        "mix_proposed_scales": (cells, ACTION_COUNT),
        "mix_proposed_biases": (cells, ACTION_COUNT),
        "mix_per_action_accepted": (cells, ACTION_COUNT),
        "dev_targets": (DEV_ROOTS, ACTION_COUNT),
        "dev_actions": (DEV_ROOTS,),
        "dev_root_ids": (DEV_ROOTS,),
        "dev_cluster_ordinal": (DEV_ROOTS,),
        "dev_cluster_ids": (DEV_EPISODES,),
        "dev_raw_logits": (cells, len(ARMS), DEV_ROOTS, ACTION_COUNT),
        "dev_raw_probabilities": (cells, len(ARMS), DEV_ROOTS, ACTION_COUNT),
        "nz_calibrated_logits": (
            len(CALIBRATORS), cells, DEV_ROOTS, ACTION_COUNT
        ),
        "nz_calibrated_probabilities": (
            len(CALIBRATORS), cells, DEV_ROOTS, ACTION_COUNT
        ),
        "bootstrap_indices": (BOOTSTRAP_RESAMPLES, DEV_EPISODES),
        "episode_derangements": (DERANGEMENT_REPETITIONS, DEV_EPISODES),
    }


def _recompute_mix_acceptance(
    raw_logits: np.ndarray,
    targets: np.ndarray,
    factual_actions: np.ndarray,
    proposed_scales: np.ndarray,
    proposed_biases: np.ndarray,
) -> tuple[np.ndarray, bool, dict[str, float]]:
    raw = np.asarray(raw_logits, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    actions = np.asarray(factual_actions, dtype=np.int64)
    scales = np.asarray(proposed_scales, dtype=np.float64)
    biases = np.asarray(proposed_biases, dtype=np.float64)
    proposed_logits = raw * scales[None, :] + biases[None, :]
    raw_probability = _sigmoid(raw)
    proposed_probability = _sigmoid(proposed_logits)
    per_action = np.empty(ACTION_COUNT, dtype=np.bool_)
    identity_objectives: list[float] = []
    proposed_objectives: list[float] = []
    for action in range(ACTION_COUNT):
        weights = _mix_weights(actions, action)
        mask = actions == action
        factual = target[mask, action]
        positives = int(factual.sum())
        negatives = len(factual) - positives
        all_positives = int(target[:, action].sum())
        all_negatives = len(target) - all_positives
        support = (
            len(factual) >= CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION
            and positives >= CALIBRATION_MINIMUM_CLASS_EXAMPLES
            and negatives >= CALIBRATION_MINIMUM_CLASS_EXAMPLES
            and all_positives >= CALIBRATION_MINIMUM_CLASS_EXAMPLES
            and all_negatives >= CALIBRATION_MINIMUM_CLASS_EXAMPLES
        )
        identity_objective = float(
            _mixed_objective(
                torch.from_numpy(raw[:, action]),
                torch.from_numpy(target[:, action]),
                torch.from_numpy(weights),
                torch.tensor(1.0, dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
            )
        )
        proposed_objective = float(
            _mixed_objective(
                torch.from_numpy(raw[:, action]),
                torch.from_numpy(target[:, action]),
                torch.from_numpy(weights),
                torch.tensor(float(scales[action]), dtype=torch.float64),
                torch.tensor(float(biases[action]), dtype=torch.float64),
            )
        )
        identity_objectives.append(identity_objective)
        proposed_objectives.append(proposed_objective)
        per_action[action] = (
            support
            and identity_objective - proposed_objective > CALIBRATION_TOLERANCE
            and _binary_bce(target[:, action], proposed_probability[:, action])
            < _binary_bce(target[:, action], raw_probability[:, action])
            and _binary_brier(target[:, action], proposed_probability[:, action])
            < _binary_brier(target[:, action], raw_probability[:, action])
            and _binary_bce(factual, proposed_probability[mask, action])
            < _binary_bce(factual, raw_probability[mask, action])
            and _binary_brier(factual, proposed_probability[mask, action])
            < _binary_brier(factual, raw_probability[mask, action])
        )
    factual_targets, raw_factual = _factual_vectors(target, raw_probability, actions)
    _, proposed_factual = _factual_vectors(target, proposed_probability, actions)
    aggregate = {
        "identity_mixed_objective_mean": float(np.mean(identity_objectives)),
        "proposed_mixed_objective_mean": float(np.mean(proposed_objectives)),
        "identity_all_BCE": _binary_bce(target, raw_probability),
        "proposed_all_BCE": _binary_bce(target, proposed_probability),
        "identity_all_Brier": _binary_brier(target, raw_probability),
        "proposed_all_Brier": _binary_brier(target, proposed_probability),
        "identity_factual_BCE": _binary_bce(factual_targets, raw_factual),
        "proposed_factual_BCE": _binary_bce(factual_targets, proposed_factual),
        "identity_factual_Brier": _binary_brier(factual_targets, raw_factual),
        "proposed_factual_Brier": _binary_brier(factual_targets, proposed_factual),
    }
    aggregate_accepted = (
        aggregate["identity_mixed_objective_mean"]
        - aggregate["proposed_mixed_objective_mean"]
        > CALIBRATION_TOLERANCE
        and aggregate["proposed_all_BCE"] < aggregate["identity_all_BCE"]
        and aggregate["proposed_all_Brier"] < aggregate["identity_all_Brier"]
        and aggregate["proposed_factual_BCE"] < aggregate["identity_factual_BCE"]
        and aggregate["proposed_factual_Brier"] < aggregate["identity_factual_Brier"]
    )
    return per_action, bool(per_action.all() and aggregate_accepted), aggregate


def reload_and_validate_evidence(
    path: Path,
    publication: Mapping[str, object],
    *,
    source_evidence: Mapping[str, Mapping[str, object]],
    cal_nz_raw_logit_sha256_by_cell: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not path.is_file() or _sha256_file(path) != publication["sha256"]:
        raise ValueError("published evidence artifact absent or drifted")
    if path.stat().st_size != publication["byte_length"]:
        raise ValueError("published evidence byte length drifted")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != EVIDENCE_KEYS:
            raise ValueError("published evidence key set drifted")
        arrays = {
            name: _canonical_evidence_array(loaded[name])
            for name in sorted(loaded.files)
        }
    if _array_manifest(arrays) != publication["arrays"]:
        raise ValueError("published evidence per-array manifest drifted")
    if _array_manifest_sha256(publication["arrays"]) != publication["array_manifest_sha256"]:
        raise ValueError("published evidence array-manifest digest drifted")
    expected_shapes = _expected_evidence_shapes()
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"published evidence shape drifted for {name}")
        if arrays[name].dtype.str != EVIDENCE_DTYPES[name]:
            raise ValueError(f"published evidence dtype drifted for {name}")
    expected_cells = tuple(
        f"{fit_label}/init-{seed}" for fit_label in FIT_COHORTS for seed in INIT_SEEDS
    )
    if (
        int(arrays["schema_version"]) != 1
        or not np.array_equal(arrays["action_ids"], np.arange(ACTION_COUNT, dtype=np.int16))
        or
        _decode_byte_strings(arrays["fit_ids"]) != FIT_COHORTS
        or _decode_byte_strings(arrays["cal_ids"]) != CAL_COHORTS
        or _decode_byte_strings(arrays["cell_ids"]) != expected_cells
        or _decode_byte_strings(arrays["arm_ids"]) != ARMS
        or _decode_byte_strings(arrays["calibrator_ids"]) != CALIBRATORS
    ):
        raise ValueError("published evidence identifier axes drifted")
    expected_fit_index = np.repeat(
        np.arange(len(FIT_COHORTS), dtype=np.int16), len(INIT_SEEDS)
    )
    expected_seeds = np.tile(np.asarray(INIT_SEEDS, dtype=np.int32), len(FIT_COHORTS))
    if (
        not np.array_equal(arrays["cell_fit_index"], expected_fit_index)
        or not np.array_equal(arrays["cell_cal_index"], expected_fit_index)
        or not np.array_equal(arrays["cell_init_seed"], expected_seeds)
    ):
        raise ValueError("published evidence cell mapping drifted")
    if set(cal_nz_raw_logit_sha256_by_cell) != set(expected_cells):
        raise ValueError("CAL NZ raw-logit source digest cell set drifted")
    for cell_index, cell in enumerate(expected_cells):
        if cal_nz_raw_logit_sha256_by_cell[cell] != _array_sha256(
            arrays["cal_nz_raw_logits"][cell_index]
        ):
            raise ValueError(f"{cell} CAL NZ raw-logit source cross-link drifted")
    for name in ("fit_targets", "cal_targets", "dev_targets"):
        values = arrays[name]
        if not np.logical_or(values == 0.0, values == 1.0).all():
            raise ValueError(f"published {name} must remain binary")
    for name in ("fit_actions", "cal_actions", "dev_actions"):
        values = arrays[name]
        if values.dtype.kind not in {"i", "u"} or values.min() < 0 or values.max() >= ACTION_COUNT:
            raise ValueError(f"published {name} contains invalid semantic action IDs")
    expected_ordinal = np.repeat(
        np.arange(DEV_EPISODES, dtype=np.int32), ROOTS_PER_EPISODE
    )
    if not np.array_equal(arrays["dev_cluster_ordinal"], expected_ordinal):
        raise ValueError("DEV cluster ordinal/order drifted")
    if len(set(_decode_byte_strings(arrays["dev_cluster_ids"]))) != DEV_EPISODES:
        raise ValueError("DEV cluster IDs must be unique")
    cluster_ids = _decode_byte_strings(arrays["dev_cluster_ids"])
    expanded_cluster_ids = tuple(
        cluster_ids[int(value)] for value in arrays["dev_cluster_ordinal"]
    )
    expanded_digest = strict._ordered_sequence_digest(expanded_cluster_ids)
    expected_source_labels = (*FIT_COHORTS, *CAL_COHORTS, "DEV")
    if set(source_evidence) != set(expected_source_labels):
        raise ValueError("source evidence must cover exactly all seven PB21K partitions")
    source_rows = [
        (
            label,
            arrays["fit_targets"][index],
            arrays["fit_actions"][index],
            arrays["fit_root_ids"][index],
        )
        for index, label in enumerate(FIT_COHORTS)
    ]
    source_rows.extend(
        (
            label,
            arrays["cal_targets"][index],
            arrays["cal_actions"][index],
            arrays["cal_root_ids"][index],
        )
        for index, label in enumerate(CAL_COHORTS)
    )
    source_rows.append(
        ("DEV", arrays["dev_targets"], arrays["dev_actions"], arrays["dev_root_ids"])
    )
    for label, targets, actions, root_ids in source_rows:
        manifest = source_evidence[label]
        if not isinstance(manifest, Mapping):
            raise ValueError(f"{label} source evidence manifest must be a mapping")
        if (
            manifest.get("label") != label
            or manifest.get("roots") != len(targets)
            or manifest.get("actions_per_root") != ACTION_COUNT
        ):
            raise ValueError(f"{label} source evidence geometry/identity drifted")
        if manifest.get("target_sha256") != _array_sha256(targets):
            raise ValueError(f"{label} target source-evidence cross-link drifted")
        if manifest.get("factual_action_sha256") != _array_sha256(actions):
            raise ValueError(f"{label} factual-action source-evidence cross-link drifted")
        ordered_root_sha256 = strict._ordered_sequence_digest(
            _decode_byte_strings(root_ids)
        )
        if manifest.get("ordered_root_sha256") != ordered_root_sha256:
            raise ValueError(f"{label} ordered-root source-evidence cross-link drifted")
    if source_evidence["DEV"].get("ordered_episode_sha256") != expanded_digest:
        raise ValueError("DEV evidence cluster IDs do not cross-link to source evidence")
    sampled = arrays["bootstrap_indices"]
    if sampled.dtype != np.dtype("<i4") or sampled.min() < 0 or sampled.max() >= DEV_EPISODES:
        raise ValueError("bootstrap matrix must be in-range little-endian int32")
    expected_sampled = bootstrap_index_matrix(
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        groups=DEV_EPISODES,
    )
    if not np.array_equal(sampled, expected_sampled):
        raise ValueError("bootstrap matrix does not match the frozen seed")
    derangements = arrays["episode_derangements"]
    identity = np.arange(DEV_EPISODES, dtype=np.int32)
    if derangements.dtype != np.dtype("<i4") or any(
        np.any(row == identity) or not np.array_equal(np.sort(row), identity)
        for row in derangements
    ):
        raise ValueError("episode derangements must be fixed-point-free bijections")
    expected_derangements = episode_derangement_indices(group_count=DEV_EPISODES)
    if not np.array_equal(derangements, expected_derangements):
        raise ValueError("derangement matrix does not match the frozen seed")
    root_derangements = (
        derangements[:, :, None] * ROOTS_PER_EPISODE
        + np.arange(ROOTS_PER_EPISODE, dtype=np.int32)[None, None, :]
    ).reshape(DERANGEMENT_REPETITIONS, DEV_ROOTS)
    raw_logits = arrays["dev_raw_logits"]
    if not np.array_equal(arrays["dev_raw_probabilities"], _sigmoid(raw_logits)):
        raise ValueError("published raw probabilities do not exactly match raw logits")
    nz_index = ARMS.index(ARM_NZ)
    for calibrator_index in range(len(CALIBRATORS)):
        expected_logits = (
            torch.from_numpy(raw_logits[:, nz_index])
            * torch.from_numpy(arrays["calibrator_scales"][calibrator_index])[:, None, :]
            + torch.from_numpy(arrays["calibrator_biases"][calibrator_index])[:, None, :]
        ).numpy()
        if not np.array_equal(
            arrays["nz_calibrated_logits"][calibrator_index], expected_logits
        ):
            raise ValueError(
                "published calibrated logits do not exactly match applied parameters"
            )
        if not np.array_equal(
            arrays["nz_calibrated_probabilities"][calibrator_index],
            _sigmoid(arrays["nz_calibrated_logits"][calibrator_index]),
        ):
            raise ValueError("published calibrated probabilities do not match logits")
    for cell in range(len(expected_cells)):
        cal_index = int(arrays["cell_cal_index"][cell])
        raw = arrays["cal_nz_raw_logits"][cell]
        target = arrays["cal_targets"][cal_index]
        actions = arrays["cal_actions"][cal_index]
        per_action, mix_accepted, _ = _recompute_mix_acceptance(
            raw,
            target,
            actions,
            arrays["mix_proposed_scales"][cell],
            arrays["mix_proposed_biases"][cell],
        )
        if not np.array_equal(per_action, arrays["mix_per_action_accepted"][cell]):
            raise ValueError("MIX per-action acceptance cannot be reproduced")
        if mix_accepted != bool(arrays["calibrator_accepted"][1, cell]):
            raise ValueError("MIX whole-calibrator acceptance cannot be reproduced")
        if mix_accepted:
            expected_scales = arrays["mix_proposed_scales"][cell]
            expected_biases = arrays["mix_proposed_biases"][cell]
        else:
            expected_scales = np.ones(ACTION_COUNT, dtype=np.float64)
            expected_biases = np.zeros(ACTION_COUNT, dtype=np.float64)
        if not np.array_equal(arrays["calibrator_scales"][1, cell], expected_scales) or not np.array_equal(
            arrays["calibrator_biases"][1, cell], expected_biases
        ):
            raise ValueError("MIX applied/proposed identity-fallback contract drifted")
        aa_probability = _sigmoid(
            raw * arrays["calibrator_scales"][0, cell][None, :]
            + arrays["calibrator_biases"][0, cell][None, :]
        )
        aa_accepted = _binary_bce(target, aa_probability) < _binary_bce(target, _sigmoid(raw))
        if aa_accepted != bool(arrays["calibrator_accepted"][0, cell]):
            raise ValueError("AA acceptance cannot be reproduced")
        dummy_provenance = TrainCalibrationProvenance(
            source_namespace="maze_chase.pb21k.evidence-refit.v1",
            source_split="TRAIN",
            source_partition="TRAIN-CAL",
            calibration_group_ids=tuple(
                f"cal:{index // ACTION_COUNT}" for index in range(ROOTS_PER_CAL * ACTION_COUNT)
            ),
            upstream_model_fit_group_ids=("fit:sealed",),
            upstream_checkpoint_sha256="0" * 64,
            dataset_manifest_sha256="1" * 64,
            source_bundle_sha256="2" * 64,
            partition_algorithm=PARTITION_ALGORITHM,
        )
        action_ids = np.broadcast_to(
            np.arange(ACTION_COUNT, dtype=np.int64), raw.shape
        )
        refit_aa = fit_train_only_per_action_affine(
            torch.from_numpy(raw.reshape(-1)),
            torch.from_numpy(target.reshape(-1)),
            torch.from_numpy(action_ids.reshape(-1)),
            provenance=dummy_provenance,
            action_count=ACTION_COUNT,
            l2_regularization=CALIBRATION_L2,
            minimum_scale=CALIBRATION_MINIMUM_SCALE,
            minimum_examples_per_action=CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
            minimum_class_examples=CALIBRATION_MINIMUM_CLASS_EXAMPLES,
            max_iterations=CALIBRATION_MAX_ITERATIONS,
            tolerance=CALIBRATION_TOLERANCE,
            fit_mode="per_action_affine",
        )
        if (
            not np.array_equal(
                arrays["calibrator_scales"][0, cell], np.asarray(refit_aa.scales)
            )
            or not np.array_equal(
                arrays["calibrator_biases"][0, cell], np.asarray(refit_aa.biases)
            )
            or bool(arrays["calibrator_accepted"][0, cell]) != refit_aa.accepted
        ):
            raise ValueError("AA deterministic evidence refit drifted")
        refit_mix_scales = []
        refit_mix_biases = []
        for action in range(ACTION_COUNT):
            scale, bias, _ = _fit_one_mixed_action(
                raw[:, action], target[:, action], _mix_weights(actions, action)
            )
            refit_mix_scales.append(scale)
            refit_mix_biases.append(bias)
        if not np.array_equal(
            arrays["mix_proposed_scales"][cell], np.asarray(refit_mix_scales)
        ) or not np.array_equal(
            arrays["mix_proposed_biases"][cell], np.asarray(refit_mix_biases)
        ):
            raise ValueError("MIX deterministic evidence refit drifted")
    return arrays, {
        "allow_pickle": False,
        "exact_allowed_keys": True,
        "allowed_key_set_sha256": evidence_key_set_sha256(),
        "per_array_manifest_verified": True,
        "all_seven_source_partition_target_action_root_cross_links_verified": True,
        "CAL_NZ_raw_logit_prepublication_cross_links_verified": True,
        "cluster_geometry_verified": True,
        "expanded_DEV_episode_sha256": expanded_digest,
        "DEV_ordered_episode_source_cross_link_verified": True,
        "calibrator_outputs_verified": True,
        "AA_and_MIX_deterministic_refits_verified": True,
        "MIX_acceptance_recomputed": True,
        "root_derangement_sha256": _array_sha256(root_derangements),
        "root_derangement_reconstructed_from_episode_indices": True,
        "passed": True,
    }


def _ordered_rows(cluster_ordinal: np.ndarray) -> tuple[np.ndarray, ...]:
    ordinal = np.asarray(cluster_ordinal, dtype=np.int64)
    rows = tuple(np.flatnonzero(ordinal == index) for index in range(DEV_EPISODES))
    if any(len(value) != ROOTS_PER_EPISODE for value in rows):
        raise ValueError("DEV must contain 768 clusters of 12 roots")
    return rows


def _bootstrap_means(
    episode_values: np.ndarray,
    sampled: np.ndarray,
    *,
    chunk_size: int = 1_024,
) -> np.ndarray:
    values = np.asarray(episode_values, dtype=np.float64)
    indices = np.asarray(sampled, dtype=np.int32)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != DEV_EPISODES or indices.shape[1] != DEV_EPISODES:
        raise ValueError("bootstrap episode geometry drifted")
    result = np.empty((len(indices), values.shape[1]), dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        result[start:stop] = values[indices[start:stop]].mean(axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("bootstrap draws must be finite")
    return result


def _leave_one_auc(
    targets: np.ndarray,
    scores: np.ndarray,
    rows: Sequence[np.ndarray],
) -> np.ndarray:
    values = np.empty(len(rows), dtype=np.float64)
    retained = np.ones(len(targets), dtype=np.bool_)
    for index, removed in enumerate(rows):
        retained[:] = True
        retained[removed] = False
        try:
            values[index] = pb21j._score_auc(targets[retained], scores[retained])
        except ValueError as error:
            raise RuntimeError(f"degenerate delete-one AUC at episode {index}") from error
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite delete-one AUC")
    return values


def primary_iut_report(
    targets: np.ndarray,
    raw_logits: np.ndarray,
    cluster_ordinal: np.ndarray,
    sampled: np.ndarray,
    *,
    cell_ids: Sequence[str],
    arm_ids: Sequence[str],
) -> dict[str, object]:
    rows = _ordered_rows(cluster_ordinal)
    cells = tuple(cell_ids)
    arms = tuple(arm_ids)
    if len(cells) != 9 or tuple(arms) != ARMS:
        raise ValueError("primary inference requires canonical cells and arms")
    compared = tuple(dict.fromkeys(arm for _, left, right, _ in PRIMARY_COMPARISONS for arm in (left, right)))
    cache: dict[tuple[int, str], tuple[float, np.ndarray]] = {}
    for cell in range(len(cells)):
        for arm in compared:
            values = raw_logits[cell, arms.index(arm)]
            cache[cell, arm] = (
                pb21j._score_auc(targets, values),
                _leave_one_auc(targets, values, rows),
            )
    comparisons: dict[str, object] = {}
    path_passed = True
    for name, left, right, threshold in PRIMARY_COMPARISONS:
        full_values: list[float] = []
        pseudovalues: list[np.ndarray] = []
        point_guards: dict[str, bool] = {}
        for cell, cell_id in enumerate(cells):
            left_full, left_delete = cache[cell, left]
            right_full, right_delete = cache[cell, right]
            delta = left_full - right_full
            delete_delta = left_delete - right_delete
            pseudo = DEV_EPISODES * delta - (DEV_EPISODES - 1) * delete_delta
            if not np.isfinite(pseudo).all():
                raise RuntimeError("non-finite AUC jackknife pseudovalue")
            full_values.append(delta)
            pseudovalues.append(pseudo)
            point_guards[cell_id] = delta > threshold
        grand = float(np.mean(full_values))
        mean_pseudo = np.mean(np.stack(pseudovalues), axis=0)
        draws = grand + _bootstrap_means(mean_pseudo, sampled)[:, 0] - mean_pseudo.mean()
        lower = float(np.quantile(draws, ONE_SIDED_ALPHA, method=QUANTILE_METHOD))
        passed = all(point_guards.values()) and lower > threshold
        path_passed = path_passed and passed
        comparisons[name] = {
            "left": left,
            "right": right,
            "threshold": threshold,
            "cell_point_deltas": dict(zip(cells, full_values, strict=True)),
            "cell_point_guards": point_guards,
            "every_cell_point_guard": all(point_guards.values()),
            "grand_mean_full_delta": grand,
            "one_sided_lower_bound": lower,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "passed": passed,
        }
    return {
        "method": "cluster_jackknife_pseudovalue_bootstrap_v1",
        "conditional_on_fixed_3x3_grid": True,
        "population_level_initialization_or_training_robustness_claimed": False,
        "shared_sample_matrix_sha256": _array_sha256(sampled),
        "comparisons": comparisons,
        "NZ_path_passed": path_passed,
    }


def _proper_episode_improvements(
    targets: np.ndarray,
    model_probability: np.ndarray,
    baseline_probability: np.ndarray,
    rows: Sequence[np.ndarray],
) -> np.ndarray:
    target = np.asarray(targets, dtype=np.float64)
    model = np.clip(np.asarray(model_probability, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
    baseline = np.clip(np.asarray(baseline_probability, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
    if target.shape != model.shape or target.shape != baseline.shape:
        raise ValueError("proper-score inputs must align")
    model_bce = -(target * np.log(model) + (1.0 - target) * np.log1p(-model))
    baseline_bce = -(target * np.log(baseline) + (1.0 - target) * np.log1p(-baseline))
    model_brier = np.square(target - model)
    baseline_brier = np.square(target - baseline)
    return np.asarray(
        [
            [
                np.mean(baseline_bce[value] - model_bce[value]),
                np.mean(baseline_brier[value] - model_brier[value]),
            ]
            for value in rows
        ],
        dtype=np.float64,
    )


def proper_score_improvement_bootstrap(
    targets: np.ndarray,
    model_probability: np.ndarray,
    baseline_probability: np.ndarray,
    rows: Sequence[np.ndarray],
    sampled: np.ndarray,
) -> dict[str, object]:
    episode = _proper_episode_improvements(
        targets, model_probability, baseline_probability, rows
    )
    draws = _bootstrap_means(episode, sampled)
    return {
        "method": "shared_exact_episode_cluster_bootstrap_for_additive_losses_v1",
        "observed_BCE_improvement": float(episode[:, 0].mean()),
        "BCE_one_sided_lower_bound": float(
            np.quantile(draws[:, 0], ONE_SIDED_ALPHA, method=QUANTILE_METHOD)
        ),
        "observed_Brier_improvement": float(episode[:, 1].mean()),
        "Brier_one_sided_lower_bound": float(
            np.quantile(draws[:, 1], ONE_SIDED_ALPHA, method=QUANTILE_METHOD)
        ),
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "shared_sample_matrix_sha256": _array_sha256(sampled),
    }


def _root_derangements(
    episode_derangements: np.ndarray,
    cluster_ordinal: np.ndarray,
) -> np.ndarray:
    rows = _ordered_rows(cluster_ordinal)
    result = np.empty((len(episode_derangements), DEV_ROOTS), dtype=np.int32)
    for repetition, mapping in enumerate(episode_derangements):
        for recipient, donor in enumerate(mapping):
            result[repetition, rows[recipient]] = rows[int(donor)]
        if not np.array_equal(np.sort(result[repetition]), np.arange(DEV_ROOTS)):
            raise RuntimeError("root derangement is not a complete bijection")
    return result


def joint_derangement_report(
    targets: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_probability: np.ndarray,
    cluster_ordinal: np.ndarray,
    episode_derangements: np.ndarray,
) -> dict[str, object]:
    root_donors = _root_derangements(episode_derangements, cluster_ordinal)
    raw_real_auc = pb21j._score_auc(targets, raw_scores)
    raw_real_per_action = [
        pb21j._score_auc(targets[:, action], raw_scores[:, action])
        for action in range(ACTION_COUNT)
    ]
    calibrated_real = all_action_probability_metrics(
        targets, calibrated_probability, ece_bins=ECE_BINS
    ).as_dict()
    shuffled_bce: list[float] = []
    shuffled_auc: list[float] = []
    shuffled_per_action: list[list[float]] = []
    for donor in root_donors:
        shuffled_raw = raw_scores[donor]
        shuffled_probability = calibrated_probability[donor]
        shuffled_bce.append(
            float(
                all_action_probability_metrics(
                    targets, shuffled_probability, ece_bins=ECE_BINS
                ).as_dict()["aggregate"]["bce"]
            )
        )
        shuffled_auc.append(pb21j._score_auc(targets, shuffled_raw))
        shuffled_per_action.append(
            [
                pb21j._score_auc(targets[:, action], shuffled_raw[:, action])
                for action in range(ACTION_COUNT)
            ]
        )
    median_bce = float(np.median(shuffled_bce))
    median_auc = float(np.median(shuffled_auc))
    per_action_drop = [
        raw_real_per_action[action]
        - float(np.median([value[action] for value in shuffled_per_action]))
        for action in range(ACTION_COUNT)
    ]
    ratio = median_bce / float(calibrated_real["aggregate"]["bce"])
    checks = {
        "calibrated_shuffled_BCE_ratio": ratio >= pb21j.MINIMUM_SHUFFLED_BCE_RATIO,
        "raw_aggregate_AUC_drop": (
            raw_real_auc - median_auc >= pb21j.MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP
        ),
        "every_action_raw_AUC_drop": all(
            value >= pb21j.MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP for value in per_action_drop
        ),
    }
    return {
        "algorithm": "joint_entire_five_action_row_cross_episode_derangement_v1",
        "repetitions": DERANGEMENT_REPETITIONS,
        "seed": DERANGEMENT_SEED,
        "episode_donor_index_sha256": _array_sha256(episode_derangements),
        "root_donor_index_sha256": _array_sha256(root_donors),
        "median_calibrated_shuffled_BCE": median_bce,
        "calibrated_shuffled_BCE_ratio": ratio,
        "median_raw_shuffled_AUC": median_auc,
        "raw_aggregate_AUC_drop": raw_real_auc - median_auc,
        "per_action_raw_AUC_drop": per_action_drop,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _both_classes_per_factual_action(
    targets: np.ndarray,
    factual_actions: np.ndarray,
) -> dict[str, object]:
    target = np.asarray(targets, dtype=np.float64)
    actions = np.asarray(factual_actions, dtype=np.int64)
    rows = np.arange(len(actions), dtype=np.int64)
    factual = target[rows, actions]
    records = []
    for action in range(ACTION_COUNT):
        values = factual[actions == action]
        positives = int(values.sum())
        negatives = len(values) - positives
        records.append(
            {
                "action_id": action,
                "observations": len(values),
                "positives": positives,
                "negatives": negatives,
                "both_classes": positives > 0 and negatives > 0,
            }
        )
    return {"per_action": records, "passed": all(value["both_classes"] for value in records)}


def fit_baselines_from_arrays(targets: np.ndarray, factual_actions: np.ndarray):
    factual_targets, _ = _factual_vectors(
        targets,
        np.full_like(targets, 0.5, dtype=np.float64),
        factual_actions,
    )
    return fit_train_baselines(
        targets,
        factual_actions,
        factual_targets,
        action_ids=tuple(range(ACTION_COUNT)),
    )


def candidate_replica_gate(
    *,
    fit_targets: np.ndarray,
    fit_actions: np.ndarray,
    dev_targets: np.ndarray,
    dev_actions: np.ndarray,
    cluster_ordinal: np.ndarray,
    raw_scores: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_scores: np.ndarray,
    calibrated_probability: np.ndarray,
    baselines: object,
    sampled: np.ndarray,
    episode_derangements: np.ndarray,
    calibrator_accepted: bool,
) -> dict[str, object]:
    raw = all_action_probability_metrics(
        dev_targets, raw_probability, ece_bins=ECE_BINS
    ).as_dict()
    calibrated = all_action_probability_metrics(
        dev_targets, calibrated_probability, ece_bins=ECE_BINS
    ).as_dict()
    factual_targets, factual_probability = _factual_vectors(
        dev_targets, calibrated_probability, dev_actions
    )
    factual = binary_probability_metrics(
        factual_targets, factual_probability, ece_bins=ECE_BINS
    ).as_dict()
    baseline_metrics = evaluate_train_fitted_baselines(
        baselines,
        dev_targets,
        dev_actions,
        factual_targets,
        ece_bins=ECE_BINS,
    ).as_dict()
    baseline_all = baselines.all_action.probability_table(len(dev_targets))
    baseline_factual = baselines.factual.probabilities_for_actions(dev_actions)
    rows = _ordered_rows(cluster_ordinal)
    all_bootstrap = proper_score_improvement_bootstrap(
        dev_targets, calibrated_probability, baseline_all, rows, sampled
    )
    factual_bootstrap = proper_score_improvement_bootstrap(
        factual_targets, factual_probability, baseline_factual, rows, sampled
    )
    prior_auc = float(baseline_metrics["all_action"]["aggregate"]["roc_auc"])
    raw_ranking = pb21j.score_ranking_gate(dev_targets, raw_scores, prior_auc=prior_auc)
    calibrated_ranking = pb21j.score_ranking_gate(
        dev_targets, calibrated_scores, prior_auc=prior_auc
    )
    all_action_checks = []
    factual_action_checks = []
    for action, value in enumerate(calibrated["per_action"]):
        metrics = value["metrics"]
        baseline_action = baseline_metrics["all_action"]["per_action"][action]["metrics"]
        all_action_checks.append(
            {
                "action_id": action,
                "ECE": float(metrics["ece_equal_mass"]) <= pb21j.MAXIMUM_PER_ACTION_ECE,
                "absolute_bias": (
                    abs(float(metrics["calibration_bias"]))
                    <= pb21j.MAXIMUM_ABSOLUTE_PER_ACTION_BIAS
                ),
                "PR_prevalence_gain": (
                    float(metrics["pr_auc"])
                    >= float(metrics["prevalence"])
                    + pb21j.MINIMUM_PER_ACTION_PR_PREVALENCE_GAIN
                ),
                "Brier_skill_nonnegative": (
                    float(metrics["brier"]) <= float(baseline_action["brier"])
                ),
            }
        )
        mask = dev_actions == action
        per_factual = binary_probability_metrics(
            factual_targets[mask], factual_probability[mask], ece_bins=ECE_BINS
        ).as_dict()
        factual_action_checks.append(
            {
                "action_id": action,
                "metrics": per_factual,
                "ECE": float(per_factual["ece_equal_mass"]) <= 0.075,
                "absolute_bias": abs(float(per_factual["calibration_bias"])) <= 0.075,
            }
        )
    fit_classes = _both_classes_per_factual_action(fit_targets, fit_actions)
    dev_classes = _both_classes_per_factual_action(dev_targets, dev_actions)
    derangement = joint_derangement_report(
        dev_targets,
        raw_scores,
        calibrated_probability,
        cluster_ordinal,
        episode_derangements,
    )
    checks = {
        "MIX_CAL_accepted": calibrator_accepted,
        "calibrated_DEV_BCE_not_worse_than_raw": (
            float(calibrated["aggregate"]["bce"]) <= float(raw["aggregate"]["bce"])
        ),
        "all_action_BCE_beats_FIT_prior_ratio": (
            float(calibrated["aggregate"]["bce"])
            <= pb21j.MAX_BASELINE_BCE_RATIO
            * float(baseline_metrics["all_action"]["aggregate"]["bce"])
        ),
        "all_action_Brier_beats_FIT_prior_ratio": (
            float(calibrated["aggregate"]["brier"])
            <= pb21j.MAX_BASELINE_BRIER_RATIO
            * float(baseline_metrics["all_action"]["aggregate"]["brier"])
        ),
        "aggregate_ECE": (
            float(calibrated["aggregate"]["ece_equal_mass"])
            <= pb21j.MAXIMUM_AGGREGATE_ECE
        ),
        "every_action_probability_gate": all(
            all(value for key, value in item.items() if key != "action_id")
            for item in all_action_checks
        ),
        "calibrated_ranking": calibrated_ranking["passed"] is True,
        "FIT_factual_both_classes_every_action": fit_classes["passed"] is True,
        "DEV_factual_both_classes_every_action": dev_classes["passed"] is True,
        "factual_BCE_strictly_beats_FIT_prior": (
            float(factual["bce"]) < float(baseline_metrics["factual"]["bce"])
        ),
        "factual_Brier_strictly_beats_FIT_prior": (
            float(factual["brier"]) < float(baseline_metrics["factual"]["brier"])
        ),
        "all_action_BCE_LCB_positive": all_bootstrap["BCE_one_sided_lower_bound"] > 0.0,
        "all_action_Brier_LCB_positive": (
            all_bootstrap["Brier_one_sided_lower_bound"] > 0.0
        ),
        "factual_BCE_LCB_positive": factual_bootstrap["BCE_one_sided_lower_bound"] > 0.0,
        "factual_Brier_LCB_positive": (
            factual_bootstrap["Brier_one_sided_lower_bound"] > 0.0
        ),
        "joint_derangement": derangement["passed"] is True,
        "factual_aggregate_ECE": float(factual["ece_equal_mass"]) <= 0.05,
        "factual_aggregate_absolute_bias": abs(float(factual["calibration_bias"])) <= 0.05,
        "every_factual_action_ECE_and_bias": all(
            item["ECE"] and item["absolute_bias"] for item in factual_action_checks
        ),
    }
    finite_payload = {
        "raw": raw,
        "calibrated": calibrated,
        "factual": factual,
        "baseline": baseline_metrics,
        "raw_ranking": raw_ranking,
        "calibrated_ranking": calibrated_ranking,
        "all_bootstrap": all_bootstrap,
        "factual_bootstrap": factual_bootstrap,
        "derangement": derangement,
        "factual_action_checks": factual_action_checks,
    }
    checks["all_metrics_finite"] = pb21j._all_numeric_finite(finite_payload)
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "raw": raw,
        "calibrated": calibrated,
        "calibrated_factual": factual,
        "baseline_metrics": baseline_metrics,
        "all_action_per_action_checks": all_action_checks,
        "factual_per_action_checks": factual_action_checks,
        "raw_score_ranking": raw_ranking,
        "calibrated_score_ranking": calibrated_ranking,
        "FIT_factual_class_support": fit_classes,
        "DEV_factual_class_support": dev_classes,
        "all_action_proper_score_bootstrap": all_bootstrap,
        "factual_proper_score_bootstrap": factual_bootstrap,
        "joint_derangement": derangement,
    }


def calibrator_comparison_report(
    targets: np.ndarray,
    factual_actions: np.ndarray,
    cluster_ordinal: np.ndarray,
    calibrated_probabilities: np.ndarray,
    sampled: np.ndarray,
    *,
    cell_ids: Sequence[str],
) -> dict[str, object]:
    if calibrated_probabilities.shape != (2, 9, DEV_ROOTS, ACTION_COUNT):
        raise ValueError("dual-calibrator DEV probability geometry drifted")
    rows = _ordered_rows(cluster_ordinal)
    cells = tuple(cell_ids)
    endpoint_episode: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in CALIBRATOR_COMPARISONS
    }
    point_values: dict[str, dict[str, float]] = {
        name: {} for name, _ in CALIBRATOR_COMPARISONS
    }
    for cell, cell_id in enumerate(cells):
        aa = calibrated_probabilities[0, cell]
        mix = calibrated_probabilities[1, cell]
        all_values = _proper_episode_improvements(targets, mix, aa, rows)
        factual_targets, aa_factual = _factual_vectors(targets, aa, factual_actions)
        _, mix_factual = _factual_vectors(targets, mix, factual_actions)
        factual_values = _proper_episode_improvements(
            factual_targets, mix_factual, aa_factual, rows
        )
        values = {
            "factual_BCE": factual_values[:, 0],
            "factual_Brier": factual_values[:, 1],
            "all_action_BCE": all_values[:, 0],
            "all_action_Brier": all_values[:, 1],
        }
        for name, episode in values.items():
            endpoint_episode[name].append(episode)
            point_values[name][cell_id] = float(episode.mean())
    comparisons: dict[str, object] = {}
    overall = True
    for name, threshold in CALIBRATOR_COMPARISONS:
        mean_episode = np.mean(np.stack(endpoint_episode[name]), axis=0)
        draws = _bootstrap_means(mean_episode, sampled)[:, 0]
        lower = float(np.quantile(draws, ONE_SIDED_ALPHA, method=QUANTILE_METHOD))
        guards = {cell: value > threshold for cell, value in point_values[name].items()}
        passed = all(guards.values()) and lower > threshold
        overall = overall and passed
        comparisons[name] = {
            "definition": "AA_loss_minus_MIX_loss",
            "threshold": threshold,
            "cell_point_improvements": point_values[name],
            "cell_point_guards": guards,
            "every_cell_point_guard": all(guards.values()),
            "grand_mean_improvement": float(mean_episode.mean()),
            "one_sided_lower_bound": lower,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "passed": passed,
        }
    return {
        "method": "paired_exact_episode_cluster_bootstrap_for_additive_losses_v1",
        "AA_can_nominate": False,
        "shared_sample_matrix_sha256": _array_sha256(sampled),
        "comparisons": comparisons,
        "passed": overall,
    }


def evaluate_authoritative_evidence(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Compute every numerical DEV result only from the reloaded NPZ arrays."""

    cells = _decode_byte_strings(arrays["cell_ids"])
    arms = _decode_byte_strings(arrays["arm_ids"])
    dev_targets = arrays["dev_targets"]
    dev_actions = arrays["dev_actions"]
    cluster_ordinal = arrays["dev_cluster_ordinal"]
    sampled = arrays["bootstrap_indices"]
    derangements = arrays["episode_derangements"]
    baselines = [
        fit_baselines_from_arrays(arrays["fit_targets"][index], arrays["fit_actions"][index])
        for index in range(len(FIT_COHORTS))
    ]
    raw_ranking_gates: dict[str, dict[str, object]] = {}
    candidate_gates: dict[str, object] = {}
    metrics: dict[str, object] = {}
    for cell_index, cell in enumerate(cells):
        fit_index = int(arrays["cell_fit_index"][cell_index])
        baseline = baselines[fit_index]
        factual_targets, _ = _factual_vectors(
            dev_targets,
            np.full_like(dev_targets, 0.5, dtype=np.float64),
            dev_actions,
        )
        baseline_metrics = evaluate_train_fitted_baselines(
            baseline,
            dev_targets,
            dev_actions,
            factual_targets,
            ece_bins=ECE_BINS,
        ).as_dict()
        prior_auc = float(baseline_metrics["all_action"]["aggregate"]["roc_auc"])
        raw_ranking_gates[cell] = {}
        raw_metrics: dict[str, object] = {}
        for arm_index, arm in enumerate(arms):
            scores = arrays["dev_raw_logits"][cell_index, arm_index]
            probabilities = arrays["dev_raw_probabilities"][cell_index, arm_index]
            raw_metrics[arm] = {
                "probability_metrics": all_action_probability_metrics(
                    dev_targets, probabilities, ece_bins=ECE_BINS
                ).as_dict(),
                "score_ranking": pb21j.score_ranking_gate(
                    dev_targets, scores, prior_auc=prior_auc
                ),
                "logit_sha256": _array_sha256(scores),
                "probability_sha256": _array_sha256(probabilities),
            }
            if arm in {ARM_Z, ARM_BZ, ARM_NZ, ARM_U0}:
                raw_ranking_gates[cell][arm] = raw_metrics[arm]["score_ranking"]
        nz_index = arms.index(ARM_NZ)
        aa_probability = arrays["nz_calibrated_probabilities"][0, cell_index]
        mix_scores = arrays["nz_calibrated_logits"][1, cell_index]
        mix_probability = arrays["nz_calibrated_probabilities"][1, cell_index]
        candidate = candidate_replica_gate(
            fit_targets=arrays["fit_targets"][fit_index],
            fit_actions=arrays["fit_actions"][fit_index],
            dev_targets=dev_targets,
            dev_actions=dev_actions,
            cluster_ordinal=cluster_ordinal,
            raw_scores=arrays["dev_raw_logits"][cell_index, nz_index],
            raw_probability=arrays["dev_raw_probabilities"][cell_index, nz_index],
            calibrated_scores=mix_scores,
            calibrated_probability=mix_probability,
            baselines=baseline,
            sampled=sampled,
            episode_derangements=derangements,
            calibrator_accepted=bool(arrays["calibrator_accepted"][1, cell_index]),
        )
        candidate_gates[cell] = candidate
        aa_factual_targets, aa_factual = _factual_vectors(
            dev_targets, aa_probability, dev_actions
        )
        _, mix_factual = _factual_vectors(dev_targets, mix_probability, dev_actions)
        metrics[cell] = {
            "raw": raw_metrics,
            "AA": {
                "all_action": all_action_probability_metrics(
                    dev_targets, aa_probability, ece_bins=ECE_BINS
                ).as_dict(),
                "factual": binary_probability_metrics(
                    aa_factual_targets, aa_factual, ece_bins=ECE_BINS
                ).as_dict(),
                "calibrated_logit_sha256": _array_sha256(
                    arrays["nz_calibrated_logits"][0, cell_index]
                ),
                "calibrated_probability_sha256": _array_sha256(aa_probability),
            },
            "MIX": {
                "all_action": all_action_probability_metrics(
                    dev_targets, mix_probability, ece_bins=ECE_BINS
                ).as_dict(),
                "factual": binary_probability_metrics(
                    aa_factual_targets, mix_factual, ece_bins=ECE_BINS
                ).as_dict(),
                "calibrated_logit_sha256": _array_sha256(mix_scores),
                "calibrated_probability_sha256": _array_sha256(mix_probability),
            },
        }
    representation = primary_iut_report(
        dev_targets,
        arrays["dev_raw_logits"],
        cluster_ordinal,
        sampled,
        cell_ids=cells,
        arm_ids=arms,
    )
    dual_calibration = calibrator_comparison_report(
        dev_targets,
        dev_actions,
        cluster_ordinal,
        arrays["nz_calibrated_probabilities"],
        sampled,
        cell_ids=cells,
    )
    raw_passed = all(
        raw_ranking_gates[cell][arm]["passed"] is True
        for cell in cells
        for arm in (ARM_Z, ARM_BZ, ARM_NZ, ARM_U0)
    )
    deployability_passed = all(candidate_gates[cell]["passed"] is True for cell in cells)
    mix_acceptance_passed = bool(arrays["calibrator_accepted"][1].all())
    representation_passed = representation["NZ_path_passed"] is True
    dual_calibration_passed = dual_calibration["passed"] is True
    eligible = (
        raw_passed
        and deployability_passed
        and mix_acceptance_passed
        and representation_passed
        and dual_calibration_passed
    )
    if eligible:
        diagnosis = "NZ_MIX_recipe_nominated_for_fresh_production_runner"
    elif not representation_passed:
        diagnosis = "fresh_NZ_representation_not_supported"
    elif not raw_passed:
        diagnosis = "NZ_representation_supported_raw_ranking_unresolved"
    elif not mix_acceptance_passed:
        diagnosis = "NZ_representation_supported_MIX_CAL_acceptance_unresolved"
    elif not dual_calibration_passed:
        diagnosis = "NZ_representation_supported_dual_calibration_comparison_unresolved"
    else:
        diagnosis = "NZ_representation_supported_MIX_deployability_unresolved"
    selection = {
        "diagnosis": diagnosis,
        "eligible": eligible,
        "selected_arm": ARM_NZ if eligible else None,
        "selected_calibrator": "MIX" if eligible else None,
        "selected_recipe": (
            "masked_superset_train_pruned_deployment_v1_plus_MIX_calibration"
            if eligible
            else None
        ),
        "AA_can_nominate": False,
        "selection_by_observed_performance": False,
        "raw_ranking_passed": raw_passed,
        "MIX_acceptance_passed": mix_acceptance_passed,
        "PB21J_NZ_deployability_gates_passed": deployability_passed,
        "representation_inference_passed": representation_passed,
        "dual_calibration_comparison_passed": dual_calibration_passed,
        "architecture_and_calibration_recipe_nomination_only": eligible,
        "checkpoint_emitted": False,
        "candidate_publication_allowed": False,
        "fresh_from_scratch_model_seed_runner_required": True,
    }
    result = {
        "metrics": metrics,
        "raw_ranking_gates": raw_ranking_gates,
        "MIX_candidate_deployability_gates": candidate_gates,
        "raw_representation_inference": representation,
        "AA_vs_MIX_inference": dual_calibration,
        "selection": selection,
    }
    if not pb21j._all_numeric_finite(result):
        raise RuntimeError("authoritative evidence produced non-finite numerical result")
    return result


def _publish_attempt(
    project_root: Path,
    registration_record: Mapping[str, object],
) -> dict[str, object]:
    path = canonical_paths(project_root)["attempt"]
    if path.exists():
        raise FileExistsError("canonical PB21K attempt already exists; retry is forbidden")
    payload = {
        "schema_version": 1,
        "mode": "pb21k_nz_dualcal_attempt_v1",
        "run_id": RUN_ID,
        "classification": "attempt_consumed_no_retry",
        "registration_sha256": registration_record["sha256"],
        "source_bundle_sha256": registration_record["payload"]["source_bundle"]["sha256"],
        "registration_path": CANONICAL_REGISTRATION,
        "attempt_path": CANONICAL_ATTEMPT,
        "evidence_path": CANONICAL_EVIDENCE,
        "result_path": CANONICAL_RESULT,
        "published_before_any_source_or_data_construction": True,
        "retry_allowed": False,
    }
    digest = development._publish_json_create_only(path, payload)
    return {"path": str(path), "sha256": digest, "payload": payload}


def _run_impl(
    *,
    upstream_result: Path,
    output: Path,
    evidence_path: Path,
    registration_record: Mapping[str, object],
    attempt_record: Mapping[str, object],
    preloaded_parent: tuple[object, dict[str, object], dict[str, object]],
) -> None:
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("PB21K must hide CUDA")
    root = Path(__file__).resolve().parents[2]
    source_bundle = registration_record["payload"]["source_bundle"]
    if source_bundle != _diagnostic_source_bundle(root):
        raise RuntimeError("registered PB21K source bundle drifted")
    manifests = partition_manifests()
    registered_manifests = {
        value["label"]: value["dataset_manifest_sha256"]
        for value in registration_record["payload"]["partitions"]
    }
    if manifests != registered_manifests:
        raise RuntimeError("registered PB21K manifests drifted")
    model, _, provenance = preloaded_parent
    model.to(torch.device("cpu"))
    parent_before = development._state_dict_sha256(model.state_dict())

    # The attempt above is durable.  These are the first fresh source objects.
    fit_sources = partition_sources(FIT_COHORTS)
    fit_tapes = {
        label: collect_fresh_live_tape(model, fit_sources[label], partition_label=label)
        for label in FIT_COHORTS
    }
    permutations = deterministic_permutations()
    padded_heads: dict[str, dict[str, pb21j.MaskedSupersetHazardHead]] = {}
    fit_features: dict[str, dict[str, np.ndarray]] = {}
    training: dict[str, dict[str, object]] = {}
    for fit_label in FIT_COHORTS:
        fit_features[fit_label] = {
            arm: superset_features(fit_tapes[fit_label], arm) for arm in ARMS
        }
        for seed in INIT_SEEDS:
            cell = f"{fit_label}/init-{seed}"
            padded_heads[cell] = {}
            training[cell] = {}
            initial_digests = set()
            for arm in ARMS:
                head = initialized_head(seed)
                initial_digests.add(_state_sha256(head))
                training[cell][arm] = pb21j.train_head(
                    head,
                    fit_features[fit_label][arm],
                    fit_tapes[fit_label].hazard_targets,
                    permutations,
                )
                padded_heads[cell][arm] = head
            if len(initial_digests) != 1:
                raise RuntimeError("arms within a cell did not share byte-identical initialization")

    pruned_heads: dict[str, dict[str, pb21j.PrunedHazardHead]] = {}
    pruning: dict[str, dict[str, object]] = {}
    for cell in sorted(padded_heads):
        fit_label = cell.split("/", 1)[0]
        pruned_heads[cell] = {}
        pruning[cell] = {}
        for arm in ARMS:
            pruned, mapping = pb21j.prune_head(padded_heads[cell][arm], arm)
            transfer = pb21j.pruning_equivalence_audit(
                padded_heads[cell][arm],
                pruned,
                fit_features[fit_label][arm],
                arm,
            )
            pruned_heads[cell][arm] = pruned
            pruning[cell][arm] = {
                "mapping": mapping,
                "FIT_inference_transfer": transfer,
            }
    del padded_heads

    cal_sources = partition_sources(CAL_COHORTS)
    cal_tapes = {
        label: collect_fresh_live_tape(model, cal_sources[label], partition_label=label)
        for label in CAL_COHORTS
    }
    pair_map = dict(COHORT_PAIRS)
    calibrators: dict[
        str,
        dict[str, PerActionAffineHazardCalibrator | DualDomainAffineCalibrator],
    ] = {}
    calibration: dict[str, dict[str, object]] = {}
    cal_raw_logits: dict[str, np.ndarray] = {}
    for cell in sorted(pruned_heads):
        fit_label = cell.split("/", 1)[0]
        cal_label = pair_map[fit_label]
        features = superset_features(cal_tapes[cal_label], ARM_NZ)
        raw = pb21j.raw_logit_table(
            pruned_heads[cell][ARM_NZ],
            pb21j.compact_features(features, ARM_NZ),
            batch_size=1,
        )
        fit_provenance = calibration_provenance(
            pruned_heads[cell][ARM_NZ],
            cal_tapes[cal_label],
            fit_tapes[fit_label],
            cal_manifest_sha256=manifests[cal_label],
            source_bundle_sha256=source_bundle["sha256"],
        )
        aa, aa_report = fit_aa_calibrator(raw, cal_tapes[cal_label], fit_provenance)
        mix, mix_report = fit_mix_calibrator(raw, cal_tapes[cal_label], fit_provenance)
        if aa_report["raw_CAL_logit_sha256"] != mix_report["raw_CAL_logit_sha256"]:
            raise RuntimeError("AA and MIX did not consume the same frozen NZ CAL logits")
        calibrators[cell] = {"AA": aa, "MIX": mix}
        calibration[cell] = {
            "shared_raw_CAL_logit_sha256": _array_sha256(raw),
            "AA": aa_report,
            "MIX": mix_report,
            "shared_frozen_scorer_state_sha256": _state_sha256(
                pruned_heads[cell][ARM_NZ]
            ),
        }
        cal_raw_logits[cell] = raw

    # The one DEV source is constructed only after all 45 scorers are fitted,
    # pruned, and all 18 NZ calibrators are completely frozen.
    if development._state_dict_sha256(model.state_dict()) != parent_before:
        raise RuntimeError("PB21K mutated the frozen parent before DEV construction")
    if validate_pb21j_parent(root) != registration_record["payload"]["PB21J_parent"]:
        raise RuntimeError("exact PB21J parent drifted before DEV construction")
    dev_source = partition_sources(("DEV",))["DEV"]
    dev_tape = collect_fresh_live_tape(model, dev_source, partition_label="DEV")
    observed_manifests = {
        **{label: source.manifest_sha256 for label, source in fit_sources.items()},
        **{label: source.manifest_sha256 for label, source in cal_sources.items()},
        "DEV": dev_source.manifest_sha256,
    }
    source_tapes = {**fit_tapes, **cal_tapes, "DEV": dev_tape}
    source_evidence = {
        label: pb21j.evidence_manifest(label, source_tapes[label], observed_manifests[label])
        for label in (*FIT_COHORTS, *CAL_COHORTS, "DEV")
    }
    sampled = bootstrap_index_matrix()
    derangements = episode_derangement_indices()
    cells = tuple(
        f"{fit_label}/init-{seed}" for fit_label in FIT_COHORTS for seed in INIT_SEEDS
    )
    dev_raw_logits: dict[str, dict[str, np.ndarray]] = {}
    dev_raw_probabilities: dict[str, dict[str, np.ndarray]] = {}
    nz_calibrated_logits: dict[str, dict[str, np.ndarray]] = {}
    nz_calibrated_probabilities: dict[str, dict[str, np.ndarray]] = {}
    for cell in cells:
        dev_raw_logits[cell] = {}
        dev_raw_probabilities[cell] = {}
        for arm in ARMS:
            features = superset_features(dev_tape, arm)
            logits = pb21j.raw_logit_table(
                pruned_heads[cell][arm],
                pb21j.compact_features(features, arm),
                batch_size=1,
            )
            dev_raw_logits[cell][arm] = logits
            dev_raw_probabilities[cell][arm] = _sigmoid(logits)
        nz_calibrated_logits[cell] = {}
        nz_calibrated_probabilities[cell] = {}
        for name in CALIBRATORS:
            logits, probabilities = calibrated_output_tables(
                calibrators[cell][name], dev_raw_logits[cell][ARM_NZ]
            )
            nz_calibrated_logits[cell][name] = logits
            nz_calibrated_probabilities[cell][name] = probabilities

    arrays = build_evidence_arrays(
        fit_tapes=fit_tapes,
        cal_tapes=cal_tapes,
        dev_tape=dev_tape,
        cells=cells,
        cal_raw_logits=cal_raw_logits,
        calibrators=calibrators,
        dev_raw_logits=dev_raw_logits,
        dev_raw_probabilities=dev_raw_probabilities,
        nz_calibrated_logits=nz_calibrated_logits,
        nz_calibrated_probabilities=nz_calibrated_probabilities,
        sampled=sampled,
        derangements=derangements,
    )
    cal_nz_raw_logit_sha256_by_cell = {
        cell: str(calibration[cell]["shared_raw_CAL_logit_sha256"])
        for cell in cells
    }
    evidence_publication = publish_evidence_create_only(
        evidence_path,
        arrays,
        attempt_sha256=str(attempt_record["sha256"]),
    )
    del arrays
    authoritative_arrays, evidence_validation = reload_and_validate_evidence(
        evidence_path,
        evidence_publication,
        source_evidence=source_evidence,
        cal_nz_raw_logit_sha256_by_cell=cal_nz_raw_logit_sha256_by_cell,
    )
    numerical = evaluate_authoritative_evidence(authoritative_arrays)
    del authoritative_arrays

    parent_after = development._state_dict_sha256(model.state_dict())
    if parent_after != parent_before:
        raise RuntimeError("PB21K mutated the exact frozen parent")
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "candidate_checkpoint": {"published": False, "checkpoint_emitted": False},
        "device": "cpu",
        "threads": 1,
        "cuda_visible_devices": "-1",
        "registration": {
            "path": registration_record["path"],
            "sha256": registration_record["sha256"],
            "verified_before_attempt": True,
        },
        "attempt": attempt_record,
        "PB21J_parent": registration_record["payload"]["PB21J_parent"],
        "upstream": provenance,
        "source_bundle": source_bundle,
        "occupied_range_audit": registration_record["payload"]["occupied_range_audit"],
        "feature_contract": feature_contract(),
        "schedule": schedule_record(),
        "calibration_contract": calibration_contract(),
        "inference_contract": inference_contract(),
        "source_evidence": source_evidence,
        "authoritative_evidence": {
            **evidence_publication,
            "validation": evidence_validation,
            "numerical_DEV_decision_computed_only_after_reload": True,
        },
        "parent_state": {
            "before_sha256": parent_before,
            "after_sha256": parent_after,
            "byte_identical": True,
        },
        "live_training_pruning_calibration_reports_are_diagnostic_only": True,
        "training": training,
        "pruning": pruning,
        "calibration": calibration,
        **numerical,
        "interpretation_constraints": {
            "architecture_and_calibration_recipe_nomination_only": True,
            "masked_superset_fit_lineage_required": True,
            "pruning_proves_inference_transfer_only": True,
            "compact_head_trained_from_initialization_equivalence_claimed": False,
            "AA_control_can_nominate": False,
            "fresh_from_scratch_model_seed_runner_required": True,
            "same_DEV_retry_allowed": False,
            "checkpoint_or_candidate_claim": False,
        },
    }
    if validate_pb21j_parent(root) != registration_record["payload"]["PB21J_parent"]:
        raise RuntimeError("exact PB21J parent drifted before result publication")
    if _diagnostic_source_bundle(root) != source_bundle:
        raise RuntimeError("PB21K source bundle drifted before result publication")
    if _sha256_file(evidence_path) != evidence_publication["sha256"]:
        raise RuntimeError("authoritative evidence drifted before result publication")
    development._publish_json_create_only(output, result)


def run(*, upstream_result: Path, output: Path, registration: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(root)
    upstream_result = _require_canonical_path(
        upstream_result, paths["upstream_result"], role="upstream result"
    )
    output = _require_canonical_path(output, paths["result"], role="result")
    registration = _require_canonical_path(
        registration, paths["registration"], role="registration"
    )
    for role in ("result", "attempt", "evidence"):
        if paths[role].exists():
            raise FileExistsError(f"PB21K canonical {role} exists; retry is forbidden")
    registration_record: Mapping[str, object] | None = None
    attempt_record: Mapping[str, object] | None = None
    try:
        registration_record = validate_registration(registration)
        preloaded_parent = strict.load_exact_parent(upstream_result)
        attempt_record = _publish_attempt(root, registration_record)
        _run_impl(
            upstream_result=upstream_result,
            output=output,
            evidence_path=paths["evidence"],
            registration_record=registration_record,
            attempt_record=attempt_record,
            preloaded_parent=preloaded_parent,
        )
    except BaseException as error:
        if attempt_record is not None and not output.exists():
            evidence = None
            if paths["evidence"].is_file():
                evidence = {
                    "path": str(paths["evidence"]),
                    "sha256": _sha256_file(paths["evidence"]),
                    "byte_length": paths["evidence"].stat().st_size,
                    "published_before_failure": True,
                }
            development._publish_json_create_only(
                output,
                {
                    "schema_version": SCHEMA_VERSION,
                    "implementation_revision": IMPLEMENTATION_REVISION,
                    "mode": MODE,
                    "classification": "diagnostic_run_failed_not_candidate",
                    "qualification_claimed": False,
                    "candidate_publication_allowed": False,
                    "candidate_checkpoint": {"published": False, "checkpoint_emitted": False},
                    "registration": {
                        "path": registration_record["path"],
                        "sha256": registration_record["sha256"],
                        "verified": True,
                    },
                    "attempt": attempt_record,
                    "authoritative_evidence": evidence,
                    "retry_allowed": False,
                    "failure": {"type": type(error).__name__, "message": str(error)},
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", required=True)
    parser.add_argument("--upstream-result")
    parser.add_argument("--output")
    args = parser.parse_args()
    registration = Path(args.registration).expanduser().resolve()
    if args.register_only:
        if args.upstream_result is not None or args.output is not None:
            parser.error("--register-only accepts only --registration")
        register(registration)
        return
    if args.upstream_result is None or args.output is None:
        parser.error("run requires --upstream-result, --output, and --registration")
    run(
        upstream_result=Path(args.upstream_result).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        registration=registration,
    )


if __name__ == "__main__":
    main()
