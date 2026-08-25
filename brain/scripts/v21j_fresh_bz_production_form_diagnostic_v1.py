"""Fresh, preregistered PB21J production-form NBZ diagnostic.

This CPU-only diagnostic tests fixed detached NZ and NBZ production-form
bypasses without weakening the recurrent model.  It
cannot publish a checkpoint or a candidate.  The one canonical DEV endpoint
is opened only after every scorer and TRAIN-CAL calibrator is frozen.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from irene_brain.data import (
    DatasetSplit,
    MazeChaseDatasetConfig,
    maze_chase_dataset_manifest_sha256,
)
from irene_brain.evaluation.v21_qualification_metrics import (
    all_action_probability_metrics,
    binary_probability_metrics,
    evaluate_cross_episode_derangements,
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
import v21i_strict_live_component_localization_v1 as component
import v21i_strict_live_representation_probe_v1 as strict


SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = 1
MODE = "pb21j_fresh_bz_production_form_diagnostic_v1"
REGISTRATION_MODE = "pb21j_fresh_bz_production_form_registration_v1"
CLASSIFICATION = "diagnostic_not_candidate"
RUN_ID = "2026-08-24-pb21j-prod-bz-v1"

CANONICAL_UPSTREAM_RESULT = "brain/runs/v21i-development/2026-08-24-seed42-v3.json"
CANONICAL_REGISTRATION = (
    "brain/runs/v21j-diagnostics/2026-08-24-pb21j-prod-bz-v1.registration.json"
)
CANONICAL_RESULT = "brain/runs/v21j-diagnostics/2026-08-24-pb21j-prod-bz-v1.json"
CANONICAL_ATTEMPT = (
    "brain/runs/v21j-diagnostics/2026-08-24-pb21j-prod-bz-v1.attempt.json"
)
COMPONENT_REGISTRATION = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-component-localization-v1.registration.json"
)
COMPONENT_RESULT = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-component-localization-v1.json"
)
EXACT_COMPONENT_REGISTRATION_SHA256 = (
    "44e03cc556232238522fcf0aa1d5d08db9b47da77d47af0cff94e320d9ae2b00"
)
EXACT_COMPONENT_RESULT_SHA256 = (
    "9d402342d49969176c2ae53e707bb34632d5292e67e3af2590aa035f72e5ec0f"
)
EXACT_COMPONENT_SOURCE_BUNDLE_SHA256 = (
    "9b463ca377c4ce2600075ffb812bac646bad5b8e895be74ad39aed0e8b2296c5"
)
EXACT_COMPONENT_BZ_MINUS_U0_LOWER = -0.027505939925940276
EXACT_BOOTSTRAP_INDEX_SHA256 = (
    "2f08ef405e90095ed4360ff30d57b22f5b4bdbcfd3f208f7e8bc07199e1915cc"
)
EXACT_DERANGEMENT_EPISODE_INDEX_SHA256 = (
    "f69f0514910c88ac25cfd548466e084e23f32760fe3bcc2ee7eab468c8e9ce1f"
)
EXACT_DERANGEMENT_ROOT_INDEX_SHA256 = (
    "9d6134737d2ac8d20f4463223eb7f2b2e485d2dd9a9854a4612128c791e5e95e"
)

_PREREGISTRATION = (
    "brain/docs/preregistrations/2026-08-24-pb21j-prod-bz-v1.md"
)
_TEST_FILE = "brain/tests/test_v21j_fresh_bz_production_form_diagnostic_v1.py"
_DIAGNOSTIC_FILES = (
    "brain/scripts/v21j_fresh_bz_production_form_diagnostic_v1.py",
    _PREREGISTRATION,
    _TEST_FILE,
)

ACTION_COUNT = 5
WIDTH = 120
SUPERSET_WIDTH = 485
HIDDEN = 120
TRAIN_HEAD_PARAMETERS = 87_961
ARMS = ("N", "Z", "BZ", "NZ", "NBZ", "U0")
ARM_N, ARM_Z, ARM_BZ, ARM_NZ, ARM_NBZ, ARM_U0 = ARMS
ACTIVE_COLUMNS: dict[str, tuple[int, ...]] = {
    ARM_N: tuple(range(0, 120)),
    ARM_Z: tuple(range(240, 360)),
    ARM_BZ: tuple(range(120, 360)),
    ARM_NZ: tuple(range(0, 120)) + tuple(range(240, 360)),
    ARM_NBZ: tuple(range(0, 360)),
    ARM_U0: tuple(range(120, 485)),
}
PRUNED_PARAMETER_COUNTS = {
    ARM_N: 44_161,
    ARM_Z: 44_161,
    ARM_BZ: 58_561,
    ARM_NZ: 58_561,
    ARM_NBZ: 72_961,
    ARM_U0: 73_561,
}

FIT_COHORTS = ("F0", "F1", "F2")
CAL_COHORTS = ("C0", "C1", "C2")
COHORT_PAIRS = tuple(zip(FIT_COHORTS, CAL_COHORTS, strict=True))
INIT_SEEDS = (41_042, 42_042, 43_042)
PERMUTATION_SEED = 50_042
BOOTSTRAP_SEED = 62_042
DERANGEMENT_SEED = 71_042
PASSES = 32
ROOT_BATCH_SIZE = 24
ROOTS_PER_FIT = 3_072
STEPS_PER_PASS = 128
STEPS_PER_HEAD = 4_096
TOTAL_HEADS = 54
TOTAL_OPTIMIZER_STEPS = 221_184
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
CLIP_NORM = 1.0
PRUNING_EQUIVALENCE_MAX_ABS = 1.0e-5

DEV_EPISODES = 384
ROOTS_PER_EPISODE = 12
DEV_ROOTS = DEV_EPISODES * ROOTS_PER_EPISODE
BOOTSTRAP_RESAMPLES = 50_000
ONE_SIDED_ALPHA = 0.05
CANDIDATE_PATH_ALPHA = 0.025
QUANTILE_METHOD = "linear"
NONINFERIORITY_MARGIN = 0.025
ECE_BINS = 10
DERANGEMENT_REPETITIONS = 20

MINIMUM_AGGREGATE_ROC_AUC = 0.65
MINIMUM_BASELINE_ROC_AUC_GAIN = 0.10
MINIMUM_PER_ACTION_ROC_AUC = 0.60
MAX_BASELINE_BCE_RATIO = 0.98
MAX_BASELINE_BRIER_RATIO = 0.95
MAXIMUM_AGGREGATE_ECE = 0.05
MAXIMUM_PER_ACTION_ECE = 0.075
MAXIMUM_ABSOLUTE_PER_ACTION_BIAS = 0.05
MINIMUM_PER_ACTION_PR_PREVALENCE_GAIN = 0.05
MINIMUM_SHUFFLED_BCE_RATIO = 1.05
MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP = 0.05
MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP = 0.03

CALIBRATION_L2 = 1.0e-6
CALIBRATION_MINIMUM_SCALE = 1.0e-4
CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION = 20
CALIBRATION_MINIMUM_CLASS_EXAMPLES = 2
CALIBRATION_MAX_ITERATIONS = 100
CALIBRATION_TOLERANCE = 1.0e-10

PRIMARY_COMPARISONS = (
    ("NZ_minus_N", ARM_NZ, ARM_N, 0.0, CANDIDATE_PATH_ALPHA, "NZ"),
    ("NZ_minus_U0", ARM_NZ, ARM_U0, -NONINFERIORITY_MARGIN, CANDIDATE_PATH_ALPHA, "NZ"),
    ("NBZ_minus_N", ARM_NBZ, ARM_N, 0.0, CANDIDATE_PATH_ALPHA, "NBZ"),
    ("NBZ_minus_U0", ARM_NBZ, ARM_U0, -NONINFERIORITY_MARGIN, CANDIDATE_PATH_ALPHA, "NBZ"),
    ("NBZ_minus_NZ", ARM_NBZ, ARM_NZ, 0.0, CANDIDATE_PATH_ALPHA, "NBZ"),
    ("NBZ_minus_BZ", ARM_NBZ, ARM_BZ, -NONINFERIORITY_MARGIN, CANDIDATE_PATH_ALPHA, "NBZ"),
    ("BZ_minus_Z", ARM_BZ, ARM_Z, 0.0, ONE_SIDED_ALPHA, "mechanistic"),
)


@dataclass(frozen=True)
class FreshPartitionSpec:
    label: str
    role: str
    split: DatasetSplit
    seed_offset: int
    episodes: int


PARTITION_SPECS = (
    FreshPartitionSpec("F0", "TRAIN-FIT", DatasetSplit.TRAIN, 100_663_296, 256),
    FreshPartitionSpec("C0", "TRAIN-CAL", DatasetSplit.TRAIN, 100_663_552, 128),
    FreshPartitionSpec("F1", "TRAIN-FIT", DatasetSplit.TRAIN, 100_663_680, 256),
    FreshPartitionSpec("C1", "TRAIN-CAL", DatasetSplit.TRAIN, 100_663_936, 128),
    FreshPartitionSpec("F2", "TRAIN-FIT", DatasetSplit.TRAIN, 100_664_064, 256),
    FreshPartitionSpec("C2", "TRAIN-CAL", DatasetSplit.TRAIN, 100_664_320, 128),
    FreshPartitionSpec("DEV", "DEV", DatasetSplit.VALIDATION, 117_440_512, 384),
)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return strict._array_sha256(values)


def _state_sha256(module: nn.Module) -> str:
    return development._state_dict_sha256(module.state_dict())


def _common_dataset_config() -> dict[str, object]:
    return {
        "sequence_length": 16,
        "ghost_count": 5,
        "ghost_period": 1,
        "ghost_rule": "direct",
        "input_delay_ticks": 0,
        "sticky_direction": False,
        "episode_horizon": 0,
        "behavior_policy": "balanced_intervention_v1",
        "behavior_intervention_rate": 0.5,
        "counterfactual_targets": "all_actions_v1",
    }


def partition_contracts() -> dict[str, development.PartitionContract]:
    contracts: dict[str, development.PartitionContract] = {}
    for spec in PARTITION_SPECS:
        contracts[spec.label] = development.PartitionContract(
            spec.role,
            MazeChaseDatasetConfig(
                split=spec.split,
                seed_offset=spec.seed_offset,
                sequence_count=spec.episodes,
                **_common_dataset_config(),
            ),
        )
    return contracts


def partition_manifests() -> dict[str, str]:
    return {
        name: maze_chase_dataset_manifest_sha256(contract.dataset_config)
        for name, contract in partition_contracts().items()
    }


def partition_sources(
    labels: Sequence[str],
    *,
    source_factory: Callable[[development.PartitionContract], object] = (
        development.PartitionSource
    ),
) -> dict[str, object]:
    allowed = {spec.label for spec in PARTITION_SPECS}
    if not labels or any(label not in allowed for label in labels):
        raise ValueError("only registered PB21J FIT/CAL/DEV labels may be constructed")
    if len(set(labels)) != len(labels):
        raise ValueError("partition labels must be unique")
    contracts = partition_contracts()
    sources = {label: source_factory(contracts[label]) for label in labels}
    expected = partition_manifests()
    for label, source in sources.items():
        observed = getattr(source, "manifest_sha256", None)
        if observed != expected[label]:
            raise RuntimeError(
                f"constructed {label} source manifest does not match registration"
            )
    return sources


def _effective_interval(split: str, start: int, stop: int) -> tuple[int, int]:
    tags = {"train": 0, "validation": 1, "test": 2, "play": 3}
    if split not in tags or not (0 <= start < stop <= (1 << 62)):
        raise ValueError("invalid seed interval")
    prefix = tags[split] << 62
    return prefix + start, prefix + stop


def occupied_range_registry() -> list[dict[str, object]]:
    """Central collision audit, including sealed and non-dataset seed users."""

    records = [
        ("maze_chase", "train", 16_777_216, 16_777_344, "v21i TRAIN-FIT"),
        ("maze_chase", "train", 16_777_344, 16_777_408, "v21i TRAIN-CAL"),
        ("maze_chase", "validation", 25_165_824, 25_165_888, "v21i DEV"),
        ("maze_chase", "validation", 33_554_432, 33_554_624, "V2.1 CPU-QUAL reserved"),
        ("maze_chase", "train", 67_108_864, 67_108_874, "wallclock repetition 0"),
        ("maze_chase", "train", 68_157_440, 68_157_450, "wallclock repetition 1"),
        ("maze_chase", "train", 69_206_016, 69_206_026, "wallclock repetition 2"),
        ("maze_chase", "play", 0, 64, "PLAY-QUAL sealed"),
        ("moving_shapes", "test", 3_145_728, 3_147_264, "RCQ-v2 TEST sealed"),
        ("moving_shapes", "test", 4_194_304, 4_195_840, "RCQ-v3 TEST sealed"),
    ]
    for spec in PARTITION_SPECS:
        records.append(
            (
                "maze_chase",
                spec.split.value,
                spec.seed_offset,
                spec.seed_offset + spec.episodes,
                f"PB21J {spec.label}",
            )
        )
    result = []
    for family, split, start, stop, owner in records:
        effective_start, effective_stop = _effective_interval(split, start, stop)
        result.append(
            {
                "dataset_family": family,
                "split": split,
                "local_start": start,
                "local_stop_exclusive": stop,
                "effective_start": effective_start,
                "effective_stop_exclusive": effective_stop,
                "owner": owner,
            }
        )
    return result


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


def validate_component_parent(project_root: Path) -> dict[str, object]:
    registration = project_root / COMPONENT_REGISTRATION
    result = project_root / COMPONENT_RESULT
    if _sha256_file(registration) != EXACT_COMPONENT_REGISTRATION_SHA256:
        raise ValueError("exact component registration absent or drifted")
    if _sha256_file(result) != EXACT_COMPONENT_RESULT_SHA256:
        raise ValueError("exact component result absent or drifted")
    payload = json.loads(result.read_text(encoding="utf-8"))
    if (
        payload.get("mode") != component.MODE
        or payload.get("classification") != component.CLASSIFICATION
        or payload.get("selection", {}).get("diagnosis")
        != "multi_component_dependence_unresolved"
        or payload.get("selection", {}).get("selected_arm") is not None
        or payload.get("candidate_publication_allowed") is not False
        or payload.get("source_bundle", {}).get("sha256")
        != EXACT_COMPONENT_SOURCE_BUNDLE_SHA256
        or payload.get("bootstrap_inference", {})
        .get("comparisons", {})
        .get("BZ", {})
        .get("minus_U0", {})
        .get("simultaneous_one_sided_lower")
        != EXACT_COMPONENT_BZ_MINUS_U0_LOWER
    ):
        raise ValueError("component result scientific envelope drifted")
    current = component._diagnostic_source_bundle(project_root)
    if current.get("sha256") != EXACT_COMPONENT_SOURCE_BUNDLE_SHA256:
        raise ValueError("component source bundle drifted")
    return {
        "registration": COMPONENT_REGISTRATION,
        "registration_sha256": EXACT_COMPONENT_REGISTRATION_SHA256,
        "result": COMPONENT_RESULT,
        "result_sha256": EXACT_COMPONENT_RESULT_SHA256,
        "source_bundle_sha256": EXACT_COMPONENT_SOURCE_BUNDLE_SHA256,
        "frozen_diagnosis": "multi_component_dependence_unresolved",
        "selected_arm": None,
        "BZ_minus_U0_registered_lower": EXACT_COMPONENT_BZ_MINUS_U0_LOWER,
        "registered_noninferiority_threshold": -NONINFERIORITY_MARGIN,
        "old_BZ_was_eligible": False,
        "new_BZ_hypothesis_is_explicitly_post_hoc": True,
    }


def _diagnostic_source_bundle(project_root: Path) -> dict[str, object]:
    parent = component._diagnostic_source_bundle(project_root)
    if parent.get("sha256") != EXACT_COMPONENT_SOURCE_BUNDLE_SHA256:
        raise ValueError("complete component source bundle drifted")
    digest = sha256(b"IRPB21JFRESHBZPRODUCTIONFORM\x01")
    digest.update(bytes.fromhex(str(parent["sha256"])))
    files: dict[str, str] = {}
    for relative in _DIAGNOSTIC_FILES:
        observed = _sha256_file(project_root / relative)
        files[relative] = observed
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(observed))
    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "component_source_bundle": parent,
        "diagnostic_files": files,
    }


def feature_contract() -> dict[str, object]:
    return {
        "superset_width": SUPERSET_WIDTH,
        "slots": [
            {"name": "N", "slice": [0, 120], "value": "post_update_belief"},
            {"name": "B", "slice": [120, 240], "value": "old_belief"},
            {"name": "Z", "slice": [240, 360], "value": "current_rgb_encoder_latent"},
            {"name": "E", "slice": [360, 480], "value": "live_cognitive_prediction_error"},
            {"name": "A", "slice": [480, 485], "value": "owned_prior_action_onehot"},
        ],
        "pending_prediction_flag": "structurally_omitted",
        "arm_active_columns": {arm: list(columns) for arm, columns in ACTIVE_COLUMNS.items()},
        "all_features_detached": True,
        "masked_values": "canonical_float32_positive_zero",
        "actual_reward": None,
        "actual_hazard": None,
        "feature_denylist": list(strict._FORBIDDEN_FEATURES),
    }


def schedule_record() -> dict[str, object]:
    return {
        "fit_cohorts": list(FIT_COHORTS),
        "cal_cohorts": list(CAL_COHORTS),
        "matched_pairs": [list(pair) for pair in COHORT_PAIRS],
        "initialization_seeds": list(INIT_SEEDS),
        "arms": list(ARMS),
        "head_count": TOTAL_HEADS,
        "equal_training_head_parameters": TRAIN_HEAD_PARAMETERS,
        "passes": PASSES,
        "roots_per_pass": ROOTS_PER_FIT,
        "root_batch_size": ROOT_BATCH_SIZE,
        "steps_per_pass": STEPS_PER_PASS,
        "steps_per_head": STEPS_PER_HEAD,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "permutation_seed": PERMUTATION_SEED,
        "same_relative_permutations_every_head": True,
        "optimizer": "AdamW_fresh_restart",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "loss": "unweighted_binary_cross_entropy_with_logits_all_five_actions",
        "early_stopping": False,
        "retry_allowed": False,
        "adaptation_after_DEV": False,
        "ordering": [
            "construct_and_collect_all_FIT",
            "fit_all_54_equal_heads",
            "prune_all_heads_to_literal_minimal_forms",
            "construct_and_collect_all_CAL",
            "fit_all_54_CAL_only_calibrators",
            "construct_shared_DEV_once",
            "compute_all_DEV_predictions_and_frozen_gates",
        ],
        "all_fits_and_calibrators_complete_before_DEV_construction": True,
        "device": "cpu",
        "threads": 1,
        "cuda_visible_devices": "-1",
    }


def inference_contract() -> dict[str, object]:
    return {
        "comparisons": [
            {
                "name": name,
                "left": left,
                "right": right,
                "threshold": threshold,
                "one_sided_alpha": alpha,
                "path": path,
            }
            for name, left, right, threshold, alpha, path in PRIMARY_COMPARISONS
        ],
        "test": "cluster_jackknife_pseudovalue_bootstrap_v1",
        "fixed_grid": "3_FIT_cohorts_x_3_initializations",
        "replicas_resampled": False,
        "episode_clusters": DEV_EPISODES,
        "roots_per_episode": ROOTS_PER_EPISODE,
        "delete_one_episode_AUCs_per_arm_replica": DEV_EPISODES,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "bootstrap_index_sha256": EXACT_BOOTSTRAP_INDEX_SHA256,
        "derangement_episode_index_sha256": EXACT_DERANGEMENT_EPISODE_INDEX_SHA256,
        "derangement_root_index_sha256": EXACT_DERANGEMENT_ROOT_INDEX_SHA256,
        "shared_episode_index_matrix_all_endpoints": True,
        "candidate_path_one_sided_alpha": CANDIDATE_PATH_ALPHA,
        "mechanistic_one_sided_alpha": ONE_SIDED_ALPHA,
        "quantile_method": QUANTILE_METHOD,
        "alpha_split": False,
        "multiplicity": (
            "Bonferroni_.05_across_two_candidate_paths; limbs_within_each_path_"
            "are_a_conjunctive_intersection_union_test_without_further_split"
        ),
        "every_replica_point_guard_required": True,
        "degenerate_delete_sample_policy": "fail_closed",
    }


def calibration_contract() -> dict[str, object]:
    return {
        "fit_partition": "matched_TRAIN-CAL_only_after_scorer_frozen",
        "fit_mode": "per_action_affine",
        "scorer_prediction_batch_size": 1,
        "dtype": "float64_cpu",
        "l2_regularization": CALIBRATION_L2,
        "minimum_scale": CALIBRATION_MINIMUM_SCALE,
        "minimum_examples_per_action": CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
        "minimum_class_examples": CALIBRATION_MINIMUM_CLASS_EXAMPLES,
        "max_iterations": CALIBRATION_MAX_ITERATIONS,
        "tolerance": CALIBRATION_TOLERANCE,
        "aggregate_CAL_BCE_strict_improvement_required": True,
        "every_scale_strictly_positive": True,
        "raw_logits_reserved_for_ranking_and_primary_IUT": True,
        "calibrated_probabilities_reserved_for_proper_scores": True,
        "protocol_change_from_v21i_bias_only": True,
        "downstream_production_runner_must_implement_if_nominated": True,
    }


def canonical_paths(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve()
    return {
        "upstream_result": (root / CANONICAL_UPSTREAM_RESULT).resolve(),
        "registration": (root / CANONICAL_REGISTRATION).resolve(),
        "result": (root / CANONICAL_RESULT).resolve(),
        "attempt": (root / CANONICAL_ATTEMPT).resolve(),
    }


def _require_canonical_path(path: Path, expected: Path, *, role: str) -> Path:
    if path.resolve() != expected.resolve():
        raise ValueError(f"{role} must use the single canonical path {expected}")
    return path.resolve()


def registration_payload(project_root: Path) -> dict[str, object]:
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
            "result": CANONICAL_RESULT,
            "attempt": CANONICAL_ATTEMPT,
            "alternate_paths_allowed": False,
            "retry_allowed": False,
        },
        "component_parent": validate_component_parent(project_root),
        "upstream": {
            "v3_result_sha256": strict.EXACT_V3_RESULT_SHA256,
            "v3_parent_sha256": strict.EXACT_V3_PARENT_SHA256,
            "v3_parent_state_sha256": strict.EXACT_V3_PARENT_STATE_SHA256,
            "v3_source_bundle_sha256": strict.EXACT_V3_SOURCE_BUNDLE_SHA256,
        },
        "source_bundle": _diagnostic_source_bundle(project_root),
        "partitions": [
            {
                "label": spec.label,
                "role": spec.role,
                "split": spec.split.value,
                "seed_offset": spec.seed_offset,
                "episodes": spec.episodes,
                "stop_exclusive": spec.seed_offset + spec.episodes,
                "dataset_manifest_sha256": partition_manifests()[spec.label],
            }
            for spec in PARTITION_SPECS
        ],
        "occupied_range_audit": assert_no_range_collisions(),
        "feature_contract": feature_contract(),
        "schedule": schedule_record(),
        "pruning": {
            "recipe": "masked_superset_train_pruned_deployment_v1",
            "timing": "after_all_FIT_before_any_CAL_or_DEV",
            "active_columns": {arm: list(value) for arm, value in ACTIVE_COLUMNS.items()},
            "minimal_parameter_counts": PRUNED_PARAMETER_COUNTS,
            "copies": "selected_first_linear_columns_plus_bias_and_all_downstream_parameters",
            "padded_vs_pruned_audit_batch_size": 1,
            "numeric_equivalence_max_abs": PRUNING_EQUIVALENCE_MAX_ABS,
            "equivalence_population": "every_FIT_row_batch_size_one",
            "bit_equality_required": False,
            "only_pruned_logits_used_for_CAL_and_DEV": True,
            "inference_transfer_only": True,
            "compact_from_initialization_equivalence_claimed": False,
            "downstream_must_reproduce_fan_in_485_training_then_prune": True,
        },
        "calibration": calibration_contract(),
        "inference": inference_contract(),
        "decision": decision_contract(),
        "scope": {
            "constructed_before_run": [],
            "current_v21i_DEV_reused": False,
            "CPU_QUAL_opened": False,
            "TEST_opened": False,
            "candidate_or_checkpoint_possible": False,
        },
    }


def register(registration: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(root)
    destination = _require_canonical_path(registration, paths["registration"], role="registration")
    if paths["result"].exists():
        raise FileExistsError("canonical result already exists; retry is forbidden")
    if paths["attempt"].exists():
        raise FileExistsError("canonical attempt already exists; retry is forbidden")
    development._publish_json_create_only(destination, registration_payload(root))


def validate_registration(registration: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    destination = _require_canonical_path(
        registration, canonical_paths(root)["registration"], role="registration"
    )
    if not destination.is_file():
        raise FileNotFoundError("PB21J registration is required")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    if payload != registration_payload(root):
        raise ValueError("PB21J registration or bound source drifted")
    return {
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "payload": payload,
    }


def decision_contract() -> dict[str, object]:
    return {
        "candidate_paths": ["NZ", "NBZ"],
        "fixed_minimality_preference": ["NZ", "NBZ"],
        "nomination_recipe": "masked_superset_train_pruned_deployment_v1",
        "requires": [
            "NZ_path_requires_Z_BZ_NZ_U0_raw_ranking_in_all_nine_cells",
            "NBZ_path_requires_Z_BZ_NZ_NBZ_U0_raw_ranking_in_all_nine_cells",
            "each_path_requires_its_own_nine_CAL_fits_accepted",
            "all_path_replica_calibrated_deployability_gates",
            "raw_and_calibrated_ranking_use_unsquashed_logits",
            "all_path_replica_point_guards",
            "all_path_grand_mean_one_sided_LCB_limbs",
        ],
        "selection_by_observed_AUC": False,
        "BZ_minus_Z_role": "mechanistic_confirmation_only_cannot_nominate",
        "pass_is_architecture_nomination_only": True,
        "from_scratch_fresh_model_seed_runner_still_required": True,
        "checkpoint_emitted": False,
        "candidate_publication_allowed": False,
        "same_DEV_retry_allowed": False,
    }


@dataclass(frozen=True)
class FreshLiveTape:
    beliefs: np.ndarray
    updater_inputs: np.ndarray
    hazard_targets: np.ndarray
    parent_raw_logits: np.ndarray
    factual_actions: np.ndarray
    episode_group_ids: tuple[str, ...]
    root_state_ids: tuple[str, ...]
    prior_assignment_count: np.ndarray
    prior_disagreement_count: np.ndarray

    def __post_init__(self) -> None:
        roots = self.hazard_targets.shape[0]
        if self.beliefs.shape != (roots, WIDTH):
            raise ValueError("belief tape shape drifted")
        if self.updater_inputs.shape != (roots, strict.CAPACITY_STATE_WIDTH):
            raise ValueError("updater tape shape drifted")
        if self.hazard_targets.shape != (roots, ACTION_COUNT):
            raise ValueError("target table shape drifted")
        if self.parent_raw_logits.shape != (roots, ACTION_COUNT):
            raise ValueError("parent logit table shape drifted")
        if self.factual_actions.shape != (roots,):
            raise ValueError("current factual actions do not cover every root")
        if self.factual_actions.dtype != np.int64 or (
            roots and (self.factual_actions.min() < 0 or self.factual_actions.max() >= ACTION_COUNT)
        ):
            raise ValueError("factual actions must be semantic int64 action IDs")
        if len(self.episode_group_ids) != roots or len(self.root_state_ids) != roots:
            raise ValueError("root provenance does not cover every root")
        if self.prior_assignment_count.shape != (roots,) or self.prior_disagreement_count.shape != (roots,):
            raise ValueError("history does not cover every root")
        for value in (
            self.beliefs,
            self.updater_inputs,
            self.hazard_targets,
            self.parent_raw_logits,
        ):
            if not np.isfinite(value).all():
                raise ValueError("fresh strict-live tape contains non-finite values")


@torch.no_grad()
def collect_fresh_live_tape(
    model: object,
    source: object,
    *,
    partition_label: str,
) -> FreshLiveTape:
    """Generalized exact batch-one replay, including the current applied action."""

    registered = {spec.label: spec for spec in PARTITION_SPECS}
    if partition_label not in registered:
        raise ValueError("collector accepts only registered PB21J partitions")
    spec = registered[partition_label]
    contract = source.contract
    if (
        contract.name != spec.role
        or contract.dataset_config.split is not spec.split
        or contract.dataset_config.seed_offset != spec.seed_offset
        or contract.dataset_config.sequence_count != spec.episodes
    ):
        raise ValueError("source does not match its registered PB21J partition")
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
                    episode_groups.append(f"{partition_label}:episode:{sequence.episode_seed}")
                    root_ids.append(transition.root_state_sha256)
            # This mirrors live policy ownership: updater A saw prior_action;
            # current applied is recorded only as a label and installed after inference.
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
    tape = FreshLiveTape(
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


def superset_features(tape: FreshLiveTape, arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(f"unknown PB21J arm: {arm}")
    updater = tape.updater_inputs
    if updater.shape[1] != 366:
        raise ValueError("strict-live updater width drifted")
    pending = updater[:, 365:366]
    if not np.array_equal(pending, np.ones_like(pending)) or np.signbit(pending).any():
        raise RuntimeError("strict-live pending flag must be exact positive one before omission")
    result = np.zeros((len(updater), SUPERSET_WIDTH), dtype=np.float32)
    result[:, 0:120] = tape.beliefs
    result[:, 120:240] = updater[:, 0:120]
    result[:, 240:360] = updater[:, 120:240]
    result[:, 360:480] = updater[:, 240:360]
    result[:, 480:485] = updater[:, 360:365]
    active = np.zeros(SUPERSET_WIDTH, dtype=np.bool_)
    active[np.asarray(ACTIVE_COLUMNS[arm], dtype=np.int64)] = True
    result[:, ~active] = 0.0
    if result.dtype != np.float32 or not result.flags.c_contiguous:
        result = np.ascontiguousarray(result, dtype=np.float32)
    masked = result[:, ~active]
    if (
        not np.isfinite(result).all()
        or np.any(masked != 0.0)
        or np.any(np.signbit(masked))
    ):
        raise RuntimeError("feature masks must be finite canonical positive zero")
    return result


def compact_features(superset: np.ndarray, arm: str) -> np.ndarray:
    if arm not in ACTIVE_COLUMNS or superset.shape[1] != SUPERSET_WIDTH:
        raise ValueError("invalid compact-feature request")
    return np.ascontiguousarray(superset[:, ACTIVE_COLUMNS[arm]], dtype=np.float32)


class MaskedSupersetHazardHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_linear = nn.Linear(SUPERSET_WIDTH, HIDDEN)
        self.action_embedding = nn.Embedding(ACTION_COUNT, HIDDEN)
        self.outcome_linear = nn.Linear(HIDDEN * 2, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)
        if sum(value.numel() for value in self.parameters()) != TRAIN_HEAD_PARAMETERS:
            raise RuntimeError("PB21J equal training-head parameter count drifted")

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() != 2 or state.shape[1] != SUPERSET_WIDTH:
            raise ValueError("masked-superset state width drifted")
        encoded = F.layer_norm(F.relu(self.state_linear(state)), (HIDDEN,), eps=1.0e-5)
        action_ids = torch.arange(ACTION_COUNT, device=state.device).expand(len(state), -1)
        action = F.layer_norm(self.action_embedding(action_ids), (HIDDEN,), eps=1.0e-5)
        joined = torch.cat((encoded.unsqueeze(1).expand(-1, ACTION_COUNT, -1), action), dim=-1)
        return self.head(F.relu(self.outcome_linear(joined))).squeeze(-1)


class PrunedHazardHead(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.state_linear = nn.Linear(input_width, HIDDEN)
        self.action_embedding = nn.Embedding(ACTION_COUNT, HIDDEN)
        self.outcome_linear = nn.Linear(HIDDEN * 2, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() != 2 or state.shape[1] != self.input_width:
            raise ValueError("pruned state width drifted")
        encoded = F.layer_norm(F.relu(self.state_linear(state)), (HIDDEN,), eps=1.0e-5)
        action_ids = torch.arange(ACTION_COUNT, device=state.device).expand(len(state), -1)
        action = F.layer_norm(self.action_embedding(action_ids), (HIDDEN,), eps=1.0e-5)
        joined = torch.cat((encoded.unsqueeze(1).expand(-1, ACTION_COUNT, -1), action), dim=-1)
        return self.head(F.relu(self.outcome_linear(joined))).squeeze(-1)


def initialized_head(seed: int) -> MaskedSupersetHazardHead:
    if seed not in INIT_SEEDS:
        raise ValueError("head initialization seed is not registered")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MaskedSupersetHazardHead()


def deterministic_permutations(root_count: int = ROOTS_PER_FIT) -> tuple[Tensor, ...]:
    if root_count != ROOTS_PER_FIT:
        raise ValueError("PB21J requires exactly 3072 FIT roots")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PERMUTATION_SEED)
    values = tuple(torch.randperm(root_count, generator=generator) for _ in range(PASSES))
    if any(int(value.unique().numel()) != root_count for value in values):
        raise RuntimeError("FIT permutation is not a complete bijection")
    return values


def train_head(
    head: MaskedSupersetHazardHead,
    features: np.ndarray,
    targets: np.ndarray,
    permutations: Sequence[Tensor],
) -> dict[str, object]:
    if features.shape != (ROOTS_PER_FIT, SUPERSET_WIDTH) or targets.shape != (
        ROOTS_PER_FIT,
        ACTION_COUNT,
    ):
        raise ValueError("PB21J FIT table geometry drifted")
    states = torch.from_numpy(features)
    labels = torch.from_numpy(targets).to(torch.float32)
    parameters = tuple(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    initial = _state_sha256(head)
    records: list[dict[str, object]] = []
    steps = 0
    head.train()
    for pass_index, permutation in enumerate(permutations, start=1):
        if len(permutation) != ROOTS_PER_FIT or int(permutation.unique().numel()) != ROOTS_PER_FIT:
            raise ValueError("training permutation drifted")
        loss_sum = 0.0
        maximum_gradient = 0.0
        for start in range(0, ROOTS_PER_FIT, ROOT_BATCH_SIZE):
            indices = permutation[start : start + ROOT_BATCH_SIZE]
            loss = F.binary_cross_entropy_with_logits(head(states[indices]), labels[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite PB21J training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(parameters, CLIP_NORM))
            if not math.isfinite(gradient):
                raise FloatingPointError("non-finite PB21J gradient")
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(indices)
            maximum_gradient = max(maximum_gradient, gradient)
        records.append(
            {
                "pass": pass_index,
                "mean_all_action_BCE": loss_sum / ROOTS_PER_FIT,
                "maximum_preclip_gradient_norm": maximum_gradient,
                "permutation_sha256": _array_sha256(
                    permutation.numpy().astype(np.int64, copy=False)
                ),
            }
        )
    if steps != STEPS_PER_HEAD:
        raise RuntimeError("PB21J optimizer-step count drifted")
    head.eval()
    return {
        "initial_state_sha256": initial,
        "final_state_sha256": _state_sha256(head),
        "passes": PASSES,
        "optimizer_steps": steps,
        "pass_records": records,
        "early_stopping": False,
        "model_selection": False,
    }


def prune_head(
    padded: MaskedSupersetHazardHead,
    arm: str,
) -> tuple[PrunedHazardHead, dict[str, object]]:
    columns = ACTIVE_COLUMNS[arm]
    pruned = PrunedHazardHead(len(columns))
    with torch.no_grad():
        pruned.state_linear.weight.copy_(padded.state_linear.weight[:, list(columns)])
        pruned.state_linear.bias.copy_(padded.state_linear.bias)
        pruned.action_embedding.weight.copy_(padded.action_embedding.weight)
        pruned.outcome_linear.weight.copy_(padded.outcome_linear.weight)
        pruned.outcome_linear.bias.copy_(padded.outcome_linear.bias)
        pruned.head.weight.copy_(padded.head.weight)
        pruned.head.bias.copy_(padded.head.bias)
    count = sum(value.numel() for value in pruned.parameters())
    if count != PRUNED_PARAMETER_COUNTS[arm]:
        raise RuntimeError("literal pruned parameter count drifted")
    exact = (
        torch.equal(pruned.state_linear.weight, padded.state_linear.weight[:, list(columns)])
        and torch.equal(pruned.state_linear.bias, padded.state_linear.bias)
        and torch.equal(pruned.action_embedding.weight, padded.action_embedding.weight)
        and torch.equal(pruned.outcome_linear.weight, padded.outcome_linear.weight)
        and torch.equal(pruned.outcome_linear.bias, padded.outcome_linear.bias)
        and torch.equal(pruned.head.weight, padded.head.weight)
        and torch.equal(pruned.head.bias, padded.head.bias)
    )
    if not exact:
        raise RuntimeError("pruned tensor mapping is not byte exact")
    pruned.eval()
    return pruned, {
        "recipe": "masked_superset_train_pruned_deployment_v1",
        "active_columns": list(columns),
        "input_width": len(columns),
        "parameter_count": count,
        "mapped_tensors_byte_exact": True,
        "pruned_state_sha256": _state_sha256(pruned),
    }


@torch.no_grad()
def raw_logit_table(head: nn.Module, features: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    head.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        chunks.append(head(torch.from_numpy(features[start : start + batch_size]).to(torch.float32)))
    result = torch.cat(chunks, dim=0).to(torch.float64).numpy()
    if result.shape != (len(features), ACTION_COUNT) or not np.isfinite(result).all():
        raise RuntimeError("invalid PB21J logit table")
    return result


def pruning_equivalence_audit(
    padded: MaskedSupersetHazardHead,
    pruned: PrunedHazardHead,
    superset: np.ndarray,
    arm: str,
) -> dict[str, object]:
    # Batch size one is part of the contract and covers every FIT row.
    padded_logits = raw_logit_table(padded, superset, batch_size=1)
    pruned_logits = raw_logit_table(pruned, compact_features(superset, arm), batch_size=1)
    maximum = float(np.max(np.abs(padded_logits - pruned_logits)))
    passed = math.isfinite(maximum) and maximum <= PRUNING_EQUIVALENCE_MAX_ABS
    if not passed:
        raise RuntimeError("padded-to-pruned inference transfer exceeded 1e-5")
    return {
        "rows": len(superset),
        "batch_size": 1,
        "padded_logit_sha256": _array_sha256(padded_logits),
        "pruned_logit_sha256": _array_sha256(pruned_logits),
        "maximum_absolute_difference": maximum,
        "maximum_allowed": PRUNING_EQUIVALENCE_MAX_ABS,
        "bit_equality_required": False,
        "passed": True,
    }


def evidence_manifest(label: str, tape: FreshLiveTape, manifest_sha256: str) -> dict[str, object]:
    return {
        "label": label,
        "dataset_manifest_sha256": manifest_sha256,
        "roots": len(tape.hazard_targets),
        "actions_per_root": ACTION_COUNT,
        "belief_sha256": _array_sha256(tape.beliefs),
        "updater_input_sha256": _array_sha256(tape.updater_inputs),
        "target_sha256": _array_sha256(tape.hazard_targets),
        "parent_raw_logit_sha256": _array_sha256(tape.parent_raw_logits),
        "factual_action_sha256": _array_sha256(tape.factual_actions),
        "ordered_episode_sha256": strict._ordered_sequence_digest(tape.episode_group_ids),
        "ordered_root_sha256": strict._ordered_sequence_digest(tape.root_state_ids),
        "current_factual_action_is_label_only": True,
        "updater_A_is_previous_owned_action": True,
        "actual_reward_input": None,
        "actual_hazard_input": None,
    }


def fit_calibrator(
    head: PrunedHazardHead,
    arm: str,
    cal_features: np.ndarray,
    cal_tape: FreshLiveTape,
    fit_tape: FreshLiveTape,
    *,
    cal_manifest_sha256: str,
    source_bundle_sha256: str,
) -> tuple[PerActionAffineHazardCalibrator, dict[str, object]]:
    logits = raw_logit_table(
        head, compact_features(cal_features, arm), batch_size=1
    )
    action_ids = np.broadcast_to(np.arange(ACTION_COUNT, dtype=np.int64), logits.shape)
    provenance = TrainCalibrationProvenance(
        source_namespace="maze_chase.pb21j.train-only.v1",
        source_split="TRAIN",
        source_partition="TRAIN-CAL",
        calibration_group_ids=tuple(
            group for group in cal_tape.episode_group_ids for _ in range(ACTION_COUNT)
        ),
        upstream_model_fit_group_ids=tuple(sorted(set(fit_tape.episode_group_ids))),
        upstream_checkpoint_sha256=_state_sha256(head),
        dataset_manifest_sha256=cal_manifest_sha256,
        source_bundle_sha256=source_bundle_sha256,
        partition_algorithm=(
            "pb21j-v1:F0=train[100663296,100663552);C0=train[100663552,100663680);"
            "F1=train[100663680,100663936);C1=train[100663936,100664064);"
            "F2=train[100664064,100664320);C2=train[100664320,100664448);"
            "DEV=validation[117440512,117440896)"
        ),
    )
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
    exported = calibrator.export_parameters()
    finite = all(math.isfinite(value) for value in (*calibrator.scales, *calibrator.biases))
    accepted = (
        calibrator.accepted
        and calibrator.after_bce < calibrator.before_bce
        and finite
        and all(value > 0.0 for value in calibrator.scales)
    )
    return calibrator, {
        "raw_CAL_logit_sha256": _array_sha256(logits),
        "parameters": exported,
        "finite": finite,
        "aggregate_BCE_strictly_improved": calibrator.after_bce < calibrator.before_bce,
        "all_scales_positive": all(value > 0.0 for value in calibrator.scales),
        "accepted": accepted,
    }


def calibrated_output_tables(
    calibrator: PerActionAffineHazardCalibrator,
    raw_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_ids = torch.arange(ACTION_COUNT, dtype=torch.long).expand(len(raw_logits), -1)
    transformed = calibrator.transform_all_actions(torch.from_numpy(raw_logits), action_ids)
    logits = transformed.to(torch.float64).numpy()
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    if (
        logits.shape != raw_logits.shape
        or probabilities.shape != raw_logits.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("calibrator emitted invalid logits or probabilities")
    return logits, probabilities


def fit_baselines(fit_tape: FreshLiveTape):
    factual_targets, _ = gather_factual_rows(
        fit_tape.hazard_targets,
        np.full_like(fit_tape.hazard_targets, 0.5, dtype=np.float64),
        fit_tape.factual_actions,
        action_ids=tuple(range(ACTION_COUNT)),
    )
    return fit_train_baselines(
        fit_tape.hazard_targets,
        fit_tape.factual_actions,
        factual_targets,
        action_ids=tuple(range(ACTION_COUNT)),
    )


def _score_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware float64 ROC-AUC for arbitrary finite scores, including logits."""

    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.shape != score.shape or target.size == 0:
        raise ValueError("score AUC inputs must be non-empty and aligned")
    if not np.isfinite(target).all() or not np.isfinite(score).all():
        raise ValueError("score AUC inputs must be finite")
    if not np.logical_or(target == 0.0, target == 1.0).all():
        raise ValueError("score AUC targets must be binary")
    positives = int(target.sum())
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("score AUC requires both classes")
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return float(
        (ranks[target == 1.0].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def score_ranking_gate(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    prior_auc: float,
) -> dict[str, object]:
    target = np.asarray(targets, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    if target.shape != score.shape or target.ndim != 2 or target.shape[1] != ACTION_COUNT:
        raise ValueError("ranking score table geometry drifted")
    aggregate_auc = _score_auc(target, score)
    per_action_auc = [
        _score_auc(target[:, action], score[:, action])
        for action in range(ACTION_COUNT)
    ]
    checks = {
        "aggregate_AUC_at_least_0.65": aggregate_auc >= MINIMUM_AGGREGATE_ROC_AUC,
        "FIT_prior_AUC_gain_at_least_0.10": (
            aggregate_auc - prior_auc >= MINIMUM_BASELINE_ROC_AUC_GAIN
        ),
        "every_action_AUC_at_least_0.60": all(
            value >= MINIMUM_PER_ACTION_ROC_AUC for value in per_action_auc
        ),
    }
    return {
        "score_domain": "finite_float64_logits_without_sigmoid",
        "aggregate_roc_auc": aggregate_auc,
        "per_action_roc_auc": per_action_auc,
        "checks": checks,
        "passed": all(checks.values()),
        "prior_aggregate_AUC": prior_auc,
    }


def _ordered_group_rows(group_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    ordered = tuple(dict.fromkeys(group_ids))
    rows = tuple(np.flatnonzero(np.asarray(group_ids, dtype=object) == group) for group in ordered)
    if len(ordered) != DEV_EPISODES or any(len(value) != ROOTS_PER_EPISODE for value in rows):
        raise ValueError("shared DEV must contain 384 episode clusters of 12 roots")
    return ordered, rows


def bootstrap_index_matrix(
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    groups: int = DEV_EPISODES,
) -> np.ndarray:
    if resamples < 1 or groups < 2:
        raise ValueError("bootstrap geometry must be positive")
    rng = np.random.default_rng(seed)
    result = rng.integers(0, groups, size=(resamples, groups), dtype=np.int32)
    if result.dtype != np.int32:
        raise RuntimeError("bootstrap matrix must remain int32")
    if (
        resamples == BOOTSTRAP_RESAMPLES
        and seed == BOOTSTRAP_SEED
        and groups == DEV_EPISODES
        and _array_sha256(result) != EXACT_BOOTSTRAP_INDEX_SHA256
    ):
        raise RuntimeError("canonical bootstrap index matrix digest drifted")
    return result


def _leave_one_auc(targets: np.ndarray, probabilities: np.ndarray, rows: Sequence[np.ndarray]) -> np.ndarray:
    values = np.empty(len(rows), dtype=np.float64)
    retained = np.ones(len(targets), dtype=np.bool_)
    for index, removed in enumerate(rows):
        retained[:] = True
        retained[removed] = False
        try:
            values[index] = _score_auc(targets[retained], probabilities[retained])
        except ValueError as error:
            raise RuntimeError(f"degenerate delete-one AUC at episode {index}") from error
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite delete-one AUC")
    return values


def primary_iut_report(
    targets: np.ndarray,
    raw_scores: Mapping[str, Mapping[str, np.ndarray]],
    group_ids: Sequence[str],
    sampled: np.ndarray,
) -> dict[str, object]:
    ordered, rows = _ordered_group_rows(group_ids)
    if sampled.shape[1] != len(rows):
        raise ValueError("bootstrap matrix group width drifted")
    cells = tuple(sorted(raw_scores))
    if len(cells) != len(FIT_COHORTS) * len(INIT_SEEDS):
        raise ValueError("primary inference requires the fixed 3x3 replica grid")
    compared_arms = tuple(
        dict.fromkeys(
            arm
            for _, left, right, _, _, _ in PRIMARY_COMPARISONS
            for arm in (left, right)
        )
    )
    auc_cache = {
        cell: {
            arm: (
                _score_auc(targets, raw_scores[cell][arm]),
                _leave_one_auc(targets, raw_scores[cell][arm], rows),
            )
            for arm in compared_arms
        }
        for cell in cells
    }
    comparisons: dict[str, object] = {}
    path_pass = {"NZ": True, "NBZ": True, "mechanistic": True}
    for name, left, right, threshold, alpha, path in PRIMARY_COMPARISONS:
        cell_full: list[float] = []
        cell_pseudo: list[np.ndarray] = []
        point_guards: dict[str, bool] = {}
        for cell in cells:
            left_full, left_delete = auc_cache[cell][left]
            right_full, right_delete = auc_cache[cell][right]
            delta = left_full - right_full
            delete_delta = left_delete - right_delete
            pseudo = DEV_EPISODES * delta - (DEV_EPISODES - 1) * delete_delta
            if not np.isfinite(pseudo).all():
                raise RuntimeError("non-finite jackknife pseudovalue")
            cell_full.append(delta)
            cell_pseudo.append(pseudo)
            point_guards[cell] = delta > threshold
        grand = float(np.mean(cell_full))
        mean_pseudo = np.mean(np.stack(cell_pseudo), axis=0)
        draws = grand + mean_pseudo[sampled].mean(axis=1) - mean_pseudo.mean()
        lower = float(np.quantile(draws, alpha, method=QUANTILE_METHOD))
        passed = all(point_guards.values()) and lower > threshold
        path_pass[path] = path_pass[path] and passed
        comparisons[name] = {
            "left": left,
            "right": right,
            "threshold": threshold,
            "cell_point_deltas": dict(zip(cells, cell_full, strict=True)),
            "every_cell_point_guard": all(point_guards.values()),
            "cell_point_guards": point_guards,
            "grand_mean_full_delta": grand,
            "one_sided_lower_bound": lower,
            "one_sided_alpha": alpha,
            "one_sided_confidence_level": 1.0 - alpha,
            "path": path,
            "passed": passed,
        }
    return {
        "method": "cluster_jackknife_pseudovalue_bootstrap_v1",
        "conditional_on_fixed_3x3_grid": True,
        "population_level_initialization_or_training_robustness_claimed": False,
        "ordered_episode_sha256": strict._ordered_sequence_digest(ordered),
        "shared_sample_matrix_sha256": _array_sha256(sampled),
        "sample_matrix_dtype": str(sampled.dtype),
        "resamples": len(sampled),
        "candidate_path_alpha": CANDIDATE_PATH_ALPHA,
        "mechanistic_alpha": ONE_SIDED_ALPHA,
        "within_path_alpha_split": False,
        "comparisons": comparisons,
        "path_passed": path_pass,
    }


def decomposable_improvement_bootstrap(
    targets: np.ndarray,
    model_probability: np.ndarray,
    baseline_probability: np.ndarray,
    rows: Sequence[np.ndarray],
    sampled: np.ndarray,
    *,
    alpha: float = CANDIDATE_PATH_ALPHA,
) -> dict[str, object]:
    target = np.asarray(targets, dtype=np.float64)
    model = np.clip(np.asarray(model_probability, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
    baseline = np.clip(np.asarray(baseline_probability, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
    if target.shape != model.shape or target.shape != baseline.shape:
        raise ValueError("proper-score bootstrap inputs must align")
    bce_model = -(target * np.log(model) + (1.0 - target) * np.log1p(-model))
    bce_baseline = -(target * np.log(baseline) + (1.0 - target) * np.log1p(-baseline))
    brier_model = np.square(target - model)
    brier_baseline = np.square(target - baseline)
    episode_bce = np.asarray([np.mean(bce_baseline[value] - bce_model[value]) for value in rows])
    episode_brier = np.asarray([np.mean(brier_baseline[value] - brier_model[value]) for value in rows])
    bce_draws = episode_bce[sampled].mean(axis=1)
    brier_draws = episode_brier[sampled].mean(axis=1)
    return {
        "method": "shared_exact_episode_cluster_bootstrap_for_additive_losses_v1",
        "observed_BCE_improvement": float(episode_bce.mean()),
        "BCE_one_sided_lower_bound": float(
            np.quantile(bce_draws, alpha, method=QUANTILE_METHOD)
        ),
        "observed_Brier_improvement": float(episode_brier.mean()),
        "Brier_one_sided_lower_bound": float(
            np.quantile(brier_draws, alpha, method=QUANTILE_METHOD)
        ),
        "one_sided_alpha": alpha,
        "one_sided_confidence_level": 1.0 - alpha,
        "shared_sample_matrix_sha256": _array_sha256(sampled),
    }


def _joint_derangement_indices(group_count: int = DEV_EPISODES) -> np.ndarray:
    rng = np.random.default_rng(DERANGEMENT_SEED)
    values = np.empty((DERANGEMENT_REPETITIONS, group_count), dtype=np.int32)
    identity = np.arange(group_count, dtype=np.int32)
    for repetition in range(DERANGEMENT_REPETITIONS):
        permutation = rng.permutation(group_count).astype(np.int32, copy=False)
        while np.any(permutation == identity):
            permutation = rng.permutation(group_count).astype(np.int32, copy=False)
        values[repetition] = permutation
    if (
        group_count == DEV_EPISODES
        and _array_sha256(values) != EXACT_DERANGEMENT_EPISODE_INDEX_SHA256
    ):
        raise RuntimeError("canonical derangement episode-index digest drifted")
    return values


def _root_derangement_indices(
    rows: Sequence[np.ndarray],
    episode_donors: np.ndarray,
) -> np.ndarray:
    total_rows = sum(len(value) for value in rows)
    result = np.empty((len(episode_donors), total_rows), dtype=np.int32)
    for repetition, mapping in enumerate(episode_donors):
        for recipient, source in enumerate(mapping):
            recipient_rows = rows[recipient]
            source_rows = rows[int(source)]
            if int(source) == recipient or len(recipient_rows) != len(source_rows):
                raise RuntimeError("derangement must use an equal-size different episode")
            result[repetition, recipient_rows] = source_rows
        if not np.array_equal(np.sort(result[repetition]), np.arange(total_rows)):
            raise RuntimeError("derangement root mapping must be a complete bijection")
    if (
        len(rows) == DEV_EPISODES
        and total_rows == DEV_ROOTS
        and episode_donors.shape
        == (DERANGEMENT_REPETITIONS, DEV_EPISODES)
        and _array_sha256(result) != EXACT_DERANGEMENT_ROOT_INDEX_SHA256
    ):
        raise RuntimeError("canonical derangement root-index digest drifted")
    return result


def joint_cross_episode_derangement_report(
    targets: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_probability: np.ndarray,
    group_ids: Sequence[str],
) -> dict[str, object]:
    _, rows = _ordered_group_rows(group_ids)
    episode_donor = _joint_derangement_indices(len(rows))
    root_donor = _root_derangement_indices(rows, episode_donor)
    if np.asarray(targets).shape != np.asarray(raw_scores).shape:
        raise ValueError("raw causal scores must align with all-action targets")
    raw_real_auc = _score_auc(targets, raw_scores)
    raw_real_per_action_auc = [
        _score_auc(targets[:, action], raw_scores[:, action])
        for action in range(ACTION_COUNT)
    ]
    cal_real = all_action_probability_metrics(targets, calibrated_probability, ece_bins=ECE_BINS).as_dict()
    shuffled_raw = []
    shuffled_cal = []
    for mapping in root_donor:
        raw_table = raw_scores[mapping]
        cal_table = calibrated_probability[mapping]
        shuffled_raw.append(
            {
                "aggregate_roc_auc": _score_auc(targets, raw_table),
                "per_action_roc_auc": [
                    _score_auc(targets[:, action], raw_table[:, action])
                    for action in range(ACTION_COUNT)
                ],
            }
        )
        shuffled_cal.append(all_action_probability_metrics(targets, cal_table, ece_bins=ECE_BINS).as_dict())
    median_cal_bce = float(np.median([value["aggregate"]["bce"] for value in shuffled_cal]))
    median_raw_auc = float(np.median([value["aggregate_roc_auc"] for value in shuffled_raw]))
    per_action_drop = []
    for action in range(ACTION_COUNT):
        median = float(
            np.median([value["per_action_roc_auc"][action] for value in shuffled_raw])
        )
        per_action_drop.append(raw_real_per_action_auc[action] - median)
    checks = {
        "calibrated_shuffled_BCE_ratio": median_cal_bce / float(cal_real["aggregate"]["bce"]) >= MINIMUM_SHUFFLED_BCE_RATIO,
        "raw_aggregate_AUC_drop": raw_real_auc - median_raw_auc >= MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP,
        "every_action_raw_AUC_drop": all(value >= MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP for value in per_action_drop),
    }
    return {
        "algorithm": "joint_entire_five_action_row_cross_episode_derangement_v1",
        "repetitions": DERANGEMENT_REPETITIONS,
        "seed": DERANGEMENT_SEED,
        "episode_donor_index_sha256": _array_sha256(episode_donor),
        "root_donor_index_sha256": _array_sha256(root_donor),
        "root_donor_index_shape": list(root_donor.shape),
        "same_ordinal_root_within_donor_episode": True,
        "same_donor_mapping_for_raw_and_calibrated": True,
        "median_calibrated_shuffled_BCE": median_cal_bce,
        "calibrated_shuffled_BCE_ratio": median_cal_bce / float(cal_real["aggregate"]["bce"]),
        "median_raw_shuffled_AUC": median_raw_auc,
        "raw_aggregate_AUC_drop": raw_real_auc - median_raw_auc,
        "per_action_raw_AUC_drop": per_action_drop,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _both_classes_per_factual_action(tape: FreshLiveTape) -> dict[str, object]:
    factual_targets, _ = gather_factual_rows(
        tape.hazard_targets,
        np.full_like(tape.hazard_targets, 0.5, dtype=np.float64),
        tape.factual_actions,
        action_ids=tuple(range(ACTION_COUNT)),
    )
    records = []
    for action in range(ACTION_COUNT):
        values = factual_targets[tape.factual_actions == action]
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


def _all_numeric_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    return True


def candidate_replica_gate(
    arm: str,
    fit_tape: FreshLiveTape,
    dev_tape: FreshLiveTape,
    raw_scores: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_scores: np.ndarray,
    calibrated_probability: np.ndarray,
    baselines: object,
    sampled: np.ndarray,
    *,
    calibrator_accepted: bool,
) -> dict[str, object]:
    if arm not in {ARM_NZ, ARM_NBZ}:
        raise ValueError("deployability gates apply only to candidate paths")
    raw = all_action_probability_metrics(
        dev_tape.hazard_targets, raw_probability, ece_bins=ECE_BINS
    ).as_dict()
    calibrated = all_action_probability_metrics(
        dev_tape.hazard_targets, calibrated_probability, ece_bins=ECE_BINS
    ).as_dict()
    factual_targets, factual_probability = gather_factual_rows(
        dev_tape.hazard_targets,
        calibrated_probability,
        dev_tape.factual_actions,
        action_ids=tuple(range(ACTION_COUNT)),
    )
    factual = binary_probability_metrics(
        factual_targets, factual_probability, ece_bins=ECE_BINS
    ).as_dict()
    baseline_metrics = evaluate_train_fitted_baselines(
        baselines,
        dev_tape.hazard_targets,
        dev_tape.factual_actions,
        factual_targets,
        ece_bins=ECE_BINS,
    ).as_dict()
    baseline_all = baselines.all_action.probability_table(len(dev_tape.hazard_targets))
    baseline_factual = baselines.factual.probabilities_for_actions(dev_tape.factual_actions)
    _, rows = _ordered_group_rows(dev_tape.episode_group_ids)
    all_action_bootstrap = decomposable_improvement_bootstrap(
        dev_tape.hazard_targets,
        calibrated_probability,
        baseline_all,
        rows,
        sampled,
    )
    factual_bootstrap = decomposable_improvement_bootstrap(
        factual_targets,
        factual_probability,
        baseline_factual,
        rows,
        sampled,
    )
    fit_classes = _both_classes_per_factual_action(fit_tape)
    dev_classes = _both_classes_per_factual_action(dev_tape)
    prior_auc = float(baseline_metrics["all_action"]["aggregate"]["roc_auc"])
    raw_score_ranking = score_ranking_gate(
        dev_tape.hazard_targets, raw_scores, prior_auc=prior_auc
    )
    calibrated_ranking = score_ranking_gate(
        dev_tape.hazard_targets, calibrated_scores, prior_auc=prior_auc
    )
    per_action_checks = []
    for action, value in enumerate(calibrated["per_action"]):
        metrics = value["metrics"]
        baseline_action = baseline_metrics["all_action"]["per_action"][action]["metrics"]
        per_action_checks.append(
            {
                "action_id": action,
                "ECE": float(metrics["ece_equal_mass"]) <= MAXIMUM_PER_ACTION_ECE,
                "absolute_bias": abs(float(metrics["calibration_bias"])) <= MAXIMUM_ABSOLUTE_PER_ACTION_BIAS,
                "PR_prevalence_gain": float(metrics["pr_auc"]) >= float(metrics["prevalence"]) + MINIMUM_PER_ACTION_PR_PREVALENCE_GAIN,
                "Brier_skill_nonnegative": float(metrics["brier"]) <= float(baseline_action["brier"]),
            }
        )
    derangement = joint_cross_episode_derangement_report(
        dev_tape.hazard_targets,
        raw_scores,
        calibrated_probability,
        dev_tape.episode_group_ids,
    )
    checks = {
        "CAL_accepted": calibrator_accepted,
        "calibrated_DEV_BCE_not_worse_than_raw": float(calibrated["aggregate"]["bce"]) <= float(raw["aggregate"]["bce"]),
        "all_action_BCE_beats_FIT_prior_ratio": float(calibrated["aggregate"]["bce"]) <= MAX_BASELINE_BCE_RATIO * float(baseline_metrics["all_action"]["aggregate"]["bce"]),
        "all_action_Brier_beats_FIT_prior_ratio": float(calibrated["aggregate"]["brier"]) <= MAX_BASELINE_BRIER_RATIO * float(baseline_metrics["all_action"]["aggregate"]["brier"]),
        "aggregate_ECE": float(calibrated["aggregate"]["ece_equal_mass"]) <= MAXIMUM_AGGREGATE_ECE,
        "every_action_probability_gate": all(all(entry.values()) for entry in ({key: value for key, value in item.items() if key != "action_id"} for item in per_action_checks)),
        "calibrated_ranking": calibrated_ranking["passed"] is True,
        "FIT_factual_both_classes_every_action": fit_classes["passed"] is True,
        "DEV_factual_both_classes_every_action": dev_classes["passed"] is True,
        "factual_BCE_strictly_beats_FIT_prior": float(factual["bce"]) < float(baseline_metrics["factual"]["bce"]),
        "factual_Brier_strictly_beats_FIT_prior": float(factual["brier"]) < float(baseline_metrics["factual"]["brier"]),
        "all_action_BCE_LCB_positive": (
            all_action_bootstrap["BCE_one_sided_lower_bound"] > 0.0
        ),
        "all_action_Brier_LCB_positive": (
            all_action_bootstrap["Brier_one_sided_lower_bound"] > 0.0
        ),
        "factual_BCE_LCB_positive": (
            factual_bootstrap["BCE_one_sided_lower_bound"] > 0.0
        ),
        "factual_Brier_LCB_positive": (
            factual_bootstrap["Brier_one_sided_lower_bound"] > 0.0
        ),
        "joint_derangement": derangement["passed"] is True,
    }
    finite_payload = {
        "raw": raw,
        "calibrated": calibrated,
        "baseline": baseline_metrics,
        "factual": factual,
        "raw_score_ranking": raw_score_ranking,
        "calibrated_score_ranking": calibrated_ranking,
        "all_action_bootstrap": all_action_bootstrap,
        "factual_bootstrap": factual_bootstrap,
        "derangement": derangement,
    }
    checks["all_metrics_finite"] = _all_numeric_finite(finite_payload)
    return {
        "arm": arm,
        "checks": checks,
        "passed": all(checks.values()),
        "raw": raw,
        "calibrated": calibrated,
        "baseline_metrics": baseline_metrics,
        "calibrated_factual": factual,
        "per_action_checks": per_action_checks,
        "raw_score_ranking": raw_score_ranking,
        "calibrated_ranking": calibrated_ranking,
        "FIT_factual_class_support": fit_classes,
        "DEV_factual_class_support": dev_classes,
        "all_action_proper_score_bootstrap": all_action_bootstrap,
        "factual_proper_score_bootstrap": factual_bootstrap,
        "joint_derangement": derangement,
    }


def select_recipe(
    raw_gates: Mapping[str, Mapping[str, object]],
    candidate_gates: Mapping[str, Mapping[str, object]],
    inference: Mapping[str, object],
) -> dict[str, object]:
    cells = tuple(sorted(raw_gates))
    ranking_required = {
        ARM_NZ: (ARM_Z, ARM_BZ, ARM_NZ, ARM_U0),
        ARM_NBZ: (ARM_Z, ARM_BZ, ARM_NZ, ARM_NBZ, ARM_U0),
    }
    ranking_pass = {
        path: all(
            raw_gates[cell][arm]["passed"] is True
            for cell in cells
            for arm in ranking_required[path]
        )
        for path in (ARM_NZ, ARM_NBZ)
    }
    deployability_pass = {
        path: all(candidate_gates[cell][path]["passed"] is True for cell in cells)
        for path in (ARM_NZ, ARM_NBZ)
    }
    calibration_pass = {
        path: all(
            candidate_gates[cell][path]["checks"]["CAL_accepted"] is True
            for cell in cells
        )
        for path in (ARM_NZ, ARM_NBZ)
    }
    deployability_without_calibration_pass = {
        path: all(
            all(
                passed is True
                for name, passed in candidate_gates[cell][path]["checks"].items()
                if name != "CAL_accepted"
            )
            for cell in cells
        )
        for path in (ARM_NZ, ARM_NBZ)
    }
    inferential = inference["path_passed"]
    eligible = {
        ARM_NZ: ranking_pass[ARM_NZ] and deployability_pass[ARM_NZ] and inferential[ARM_NZ],
        ARM_NBZ: ranking_pass[ARM_NBZ] and deployability_pass[ARM_NBZ] and inferential[ARM_NBZ],
    }
    selected = ARM_NZ if eligible[ARM_NZ] else (ARM_NBZ if eligible[ARM_NBZ] else None)
    supported_paths = [
        path for path in (ARM_NZ, ARM_NBZ) if inferential[path] is True
    ]
    if selected is not None:
        diagnosis = "masked_superset_train_pruned_deployment_recipe_nominated"
    elif not supported_paths:
        diagnosis = "no_representation_path_supported"
    elif any(not ranking_pass[path] for path in supported_paths):
        diagnosis = "representation_supported_raw_ranking_unresolved"
    elif any(not calibration_pass[path] for path in supported_paths):
        diagnosis = "representation_supported_calibration_unresolved"
    else:
        diagnosis = "representation_supported_deployability_unresolved"
    path_status = {
        path: {
            "representation_inference_passed": inferential[path],
            "raw_ranking_passed": ranking_pass[path],
            "calibration_acceptance_passed": calibration_pass[path],
            "deployability_excluding_calibration_passed": (
                deployability_without_calibration_pass[path]
            ),
            "eligible": eligible[path],
        }
        for path in (ARM_NZ, ARM_NBZ)
    }
    return {
        "diagnosis": diagnosis,
        "eligible": eligible,
        "path_status": path_status,
        "fixed_minimality_preference": [ARM_NZ, ARM_NBZ],
        "selected_arm": selected,
        "selected_recipe": (
            "masked_superset_train_pruned_deployment_v1" if selected is not None else None
        ),
        "selection_by_observed_AUC": False,
        "BZ_minus_Z_mechanistic_passed": inferential["mechanistic"],
        "BZ_minus_Z_can_nominate": False,
        "architecture_nomination_only": selected is not None,
        "checkpoint_emitted": False,
        "candidate_publication_allowed": False,
        "fresh_from_scratch_model_seed_runner_required": True,
    }


def _publish_attempt(project_root: Path, registration_record: Mapping[str, object]) -> dict[str, object]:
    attempt = canonical_paths(project_root)["attempt"]
    if attempt.exists():
        raise FileExistsError("canonical PB21J attempt already exists; retry is forbidden")
    payload = {
        "schema_version": 1,
        "mode": "pb21j_fresh_bz_production_form_attempt_v1",
        "run_id": RUN_ID,
        "classification": "attempt_consumed_no_retry",
        "registration_sha256": registration_record["sha256"],
        "source_bundle_sha256": registration_record["payload"]["source_bundle"]["sha256"],
        "published_before_any_source_or_data_construction": True,
        "retry_allowed": False,
    }
    digest = development._publish_json_create_only(attempt, payload)
    return {"path": str(attempt), "sha256": digest, "payload": payload}


def _run_impl(
    *,
    upstream_result: Path,
    output: Path,
    registration_record: Mapping[str, object],
    attempt_record: Mapping[str, object],
    preloaded_parent: tuple[object, dict[str, object], dict[str, object]],
) -> None:
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("PB21J must hide CUDA")
    root = Path(__file__).resolve().parents[2]
    source_bundle = registration_record["payload"]["source_bundle"]
    if source_bundle != _diagnostic_source_bundle(root):
        raise RuntimeError("registered PB21J source bundle drifted")
    manifests = partition_manifests()
    registered_manifests = {
        value["label"]: value["dataset_manifest_sha256"]
        for value in registration_record["payload"]["partitions"]
    }
    if manifests != registered_manifests:
        raise RuntimeError("registered PB21J manifests drifted")
    model, _, provenance = preloaded_parent
    model.to(torch.device("cpu"))
    parent_before = development._state_dict_sha256(model.state_dict())

    # Attempt was durably consumed above.  These are the first source objects.
    fit_sources = partition_sources(FIT_COHORTS)
    fit_tapes = {
        label: collect_fresh_live_tape(model, fit_sources[label], partition_label=label)
        for label in FIT_COHORTS
    }
    permutations = deterministic_permutations()
    padded_heads: dict[str, dict[str, MaskedSupersetHazardHead]] = {}
    training: dict[str, dict[str, object]] = {}
    fit_features: dict[str, dict[str, np.ndarray]] = {}
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
                training[cell][arm] = train_head(
                    head,
                    fit_features[fit_label][arm],
                    fit_tapes[fit_label].hazard_targets,
                    permutations,
                )
                padded_heads[cell][arm] = head
            if len(initial_digests) != 1:
                raise RuntimeError("arms within a replica did not share byte-identical initialization")

    pruned_heads: dict[str, dict[str, PrunedHazardHead]] = {}
    pruning: dict[str, dict[str, object]] = {}
    for cell in sorted(padded_heads):
        fit_label = cell.split("/", 1)[0]
        pruned_heads[cell] = {}
        pruning[cell] = {}
        for arm in ARMS:
            pruned, mapping = prune_head(padded_heads[cell][arm], arm)
            transfer = pruning_equivalence_audit(
                padded_heads[cell][arm], pruned, fit_features[fit_label][arm], arm
            )
            pruned_heads[cell][arm] = pruned
            pruning[cell][arm] = {"mapping": mapping, "FIT_inference_transfer": transfer}
    del padded_heads

    cal_sources = partition_sources(CAL_COHORTS)
    cal_tapes = {
        label: collect_fresh_live_tape(model, cal_sources[label], partition_label=label)
        for label in CAL_COHORTS
    }
    calibrators: dict[str, dict[str, PerActionAffineHazardCalibrator]] = {}
    calibration: dict[str, dict[str, object]] = {}
    pair_map = dict(COHORT_PAIRS)
    for cell in sorted(pruned_heads):
        fit_label = cell.split("/", 1)[0]
        cal_label = pair_map[fit_label]
        calibrators[cell] = {}
        calibration[cell] = {}
        for arm in ARMS:
            cal_features = superset_features(cal_tapes[cal_label], arm)
            fitted, report = fit_calibrator(
                pruned_heads[cell][arm],
                arm,
                cal_features,
                cal_tapes[cal_label],
                fit_tapes[fit_label],
                cal_manifest_sha256=manifests[cal_label],
                source_bundle_sha256=source_bundle["sha256"],
            )
            calibrators[cell][arm] = fitted
            calibration[cell][arm] = report

    # The sole fresh DEV source is not constructed until every fit, prune, and
    # calibrator above has completed and frozen.
    if development._state_dict_sha256(model.state_dict()) != parent_before:
        raise RuntimeError("PB21J mutated the frozen parent before DEV construction")
    if validate_component_parent(root) != registration_record["payload"]["component_parent"]:
        raise RuntimeError("exact component parent drifted before DEV construction")
    dev_source = partition_sources(("DEV",))["DEV"]
    dev_tape = collect_fresh_live_tape(model, dev_source, partition_label="DEV")
    sampled = bootstrap_index_matrix()
    raw_scores: dict[str, dict[str, np.ndarray]] = {}
    raw_probabilities: dict[str, dict[str, np.ndarray]] = {}
    calibrated_scores: dict[str, dict[str, np.ndarray]] = {}
    calibrated_tables: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, object]] = {}
    raw_gates: dict[str, dict[str, object]] = {}
    candidate_gates: dict[str, dict[str, object]] = {}
    baselines_by_fit = {label: fit_baselines(tape) for label, tape in fit_tapes.items()}
    for cell in sorted(pruned_heads):
        fit_label = cell.split("/", 1)[0]
        baselines = baselines_by_fit[fit_label]
        baseline_factual_targets, _ = gather_factual_rows(
            dev_tape.hazard_targets,
            np.full_like(dev_tape.hazard_targets, 0.5),
            dev_tape.factual_actions,
            action_ids=tuple(range(ACTION_COUNT)),
        )
        baseline_report = evaluate_train_fitted_baselines(
            baselines,
            dev_tape.hazard_targets,
            dev_tape.factual_actions,
            baseline_factual_targets,
            ece_bins=ECE_BINS,
        ).as_dict()
        prior_auc = float(baseline_report["all_action"]["aggregate"]["roc_auc"])
        raw_scores[cell] = {}
        raw_probabilities[cell] = {}
        calibrated_scores[cell] = {}
        calibrated_tables[cell] = {}
        metrics[cell] = {}
        raw_gates[cell] = {}
        candidate_gates[cell] = {}
        for arm in ARMS:
            features = superset_features(dev_tape, arm)
            logits = raw_logit_table(
                pruned_heads[cell][arm], compact_features(features, arm), batch_size=1
            )
            raw_probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
            calibrated_score, calibrated_probability = calibrated_output_tables(
                calibrators[cell][arm], logits
            )
            raw_scores[cell][arm] = logits
            raw_probabilities[cell][arm] = raw_probability
            calibrated_scores[cell][arm] = calibrated_score
            calibrated_tables[cell][arm] = calibrated_probability
            raw_report = all_action_probability_metrics(
                dev_tape.hazard_targets, raw_probability, ece_bins=ECE_BINS
            ).as_dict()
            calibrated_report = all_action_probability_metrics(
                dev_tape.hazard_targets, calibrated_probability, ece_bins=ECE_BINS
            ).as_dict()
            metrics[cell][arm] = {
                "raw": raw_report,
                "calibrated": calibrated_report,
                "raw_score_ranking": score_ranking_gate(
                    dev_tape.hazard_targets, logits, prior_auc=prior_auc
                ),
                "calibrated_score_ranking": score_ranking_gate(
                    dev_tape.hazard_targets, calibrated_score, prior_auc=prior_auc
                ),
                "raw_logit_sha256": _array_sha256(logits),
                "raw_probability_sha256": _array_sha256(raw_probability),
                "calibrated_logit_sha256": _array_sha256(calibrated_score),
                "calibrated_probability_sha256": _array_sha256(calibrated_probability),
            }
            if arm in {ARM_Z, ARM_BZ, ARM_NZ, ARM_NBZ, ARM_U0}:
                raw_gates[cell][arm] = score_ranking_gate(
                    dev_tape.hazard_targets, logits, prior_auc=prior_auc
                )
        for arm in (ARM_NZ, ARM_NBZ):
            candidate_gates[cell][arm] = candidate_replica_gate(
                arm,
                fit_tapes[fit_label],
                dev_tape,
                raw_scores[cell][arm],
                raw_probabilities[cell][arm],
                calibrated_scores[cell][arm],
                calibrated_tables[cell][arm],
                baselines,
                sampled,
                calibrator_accepted=calibration[cell][arm]["accepted"] is True,
            )
    inference = primary_iut_report(
        dev_tape.hazard_targets,
        raw_scores,
        dev_tape.episode_group_ids,
        sampled,
    )
    selection = select_recipe(raw_gates, candidate_gates, inference)
    parent_after = development._state_dict_sha256(model.state_dict())
    if parent_after != parent_before:
        raise RuntimeError("PB21J mutated the exact frozen parent")
    observed_manifests = {
        **{label: source.manifest_sha256 for label, source in fit_sources.items()},
        **{label: source.manifest_sha256 for label, source in cal_sources.items()},
        "DEV": dev_source.manifest_sha256,
    }
    evidence = {
        label: evidence_manifest(label, tape, observed_manifests[label])
        for label, tape in {**fit_tapes, **cal_tapes, "DEV": dev_tape}.items()
    }
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
        "registration": {"path": registration_record["path"], "sha256": registration_record["sha256"], "verified_before_attempt": True},
        "attempt": attempt_record,
        "component_parent": registration_record["payload"]["component_parent"],
        "upstream": provenance,
        "source_bundle": source_bundle,
        "occupied_range_audit": registration_record["payload"]["occupied_range_audit"],
        "feature_contract": feature_contract(),
        "schedule": schedule_record(),
        "calibration_contract": calibration_contract(),
        "inference_contract": inference_contract(),
        "evidence": evidence,
        "parent_state": {"before_sha256": parent_before, "after_sha256": parent_after, "byte_identical": True},
        "training": training,
        "pruning": pruning,
        "calibration": calibration,
        "metrics": metrics,
        "raw_ranking_gates": raw_gates,
        "candidate_deployability_gates": candidate_gates,
        "primary_inference": inference,
        "selection": selection,
        "interpretation_constraints": {
            "architecture_nomination_only": True,
            "masked_superset_fit_lineage_required": True,
            "pruning_proves_inference_transfer_only": True,
            "compact_head_trained_from_initialization_equivalence_claimed": False,
            "per_action_affine_is_protocol_change_from_v21i_bias_only": True,
            "fresh_from_scratch_model_seed_runner_required": True,
            "same_DEV_retry_allowed": False,
            "checkpoint_or_candidate_claim": False,
        },
    }
    if validate_component_parent(root) != registration_record["payload"]["component_parent"]:
        raise RuntimeError("exact component parent drifted before result publication")
    if _diagnostic_source_bundle(root) != source_bundle:
        raise RuntimeError("PB21J source bundle drifted before result publication")
    development._publish_json_create_only(output, result)


def run(*, upstream_result: Path, output: Path, registration: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(root)
    upstream_result = _require_canonical_path(upstream_result, paths["upstream_result"], role="upstream result")
    output = _require_canonical_path(output, paths["result"], role="result")
    registration = _require_canonical_path(registration, paths["registration"], role="registration")
    if output.exists() or paths["attempt"].exists():
        raise FileExistsError("PB21J canonical attempt/result already exists; retry is forbidden")
    registration_record: Mapping[str, object] | None = None
    attempt_record: Mapping[str, object] | None = None
    try:
        registration_record = validate_registration(registration)
        # Validate the exact result/checkpoint/model envelope before consuming
        # the irreversible attempt. This opens no fresh PB21J data namespace.
        preloaded_parent = strict.load_exact_parent(upstream_result)
        attempt_record = _publish_attempt(root, registration_record)
        _run_impl(
            upstream_result=upstream_result,
            output=output,
            registration_record=registration_record,
            attempt_record=attempt_record,
            preloaded_parent=preloaded_parent,
        )
    except BaseException as error:
        if attempt_record is not None and not output.exists():
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
                    "registration": {"path": registration_record["path"], "sha256": registration_record["sha256"], "verified": True},
                    "attempt": attempt_record,
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
