"""Strict-live V2.1i component-localization diagnostic (development only).

This preregistered probe localizes which live-available updater components carry
the hazard ranking signal found by the strict-live representation probe.  It is
deliberately unable to publish a model or checkpoint.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import v21i_strict_live_representation_probe_v1 as strict
from irene_brain.evaluation.v21_qualification_metrics import (
    all_action_probability_metrics,
    binary_probability_metrics,
)
from run_provenance import apply_deterministic_mode
from v21i_development_runner import _publish_json_create_only, _state_dict_sha256


SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = 1
MODE = "v21i_strict_live_component_localization_v1"
REGISTRATION_MODE = "v21i_strict_live_component_localization_registration_v1"
CLASSIFICATION = "diagnostic_not_candidate"

STRICT_RESULT = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-probe-v2.json"
)
STRICT_REGISTRATION = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-probe-v2.registration.json"
)
CANONICAL_UPSTREAM_RESULT = (
    "brain/runs/v21i-development/2026-08-24-seed42-v3.json"
)
CANONICAL_REGISTRATION = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-component-localization-v1.registration.json"
)
CANONICAL_RESULT = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-component-localization-v1.json"
)
RUN_ID = "2026-08-24-seed42-v3-strict-live-component-localization-v1"
EXACT_STRICT_RESULT_SHA256 = (
    "2ab123d5d2d42a1d14327bcd76710aa93538564f8897c8c2431a1133127bd2aa"
)
EXACT_STRICT_REGISTRATION_SHA256 = (
    "1577d518ae138250f495f875cb0160193adb170a821dddb2798ec9530941864c"
)
EXACT_STRICT_SOURCE_BUNDLE_SHA256 = (
    "e2d3b5373077fef817d03aed95330fafb9cf01a6950a05bdc55a603b72d0429a"
)
EXACT_STRICT_SCRIPT_SHA256 = (
    "a10271f7e6a7fff891d5df9e6437f9cd0b3c3366f082490bbca9661ce8058412"
)
EXACT_STRICT_DOC_SHA256 = (
    "65ace6fa94d9a29c7eed6e0c7168ee5742dc265c85351a4cdea9f2faa6871fdb"
)
EXACT_STRICT_TEST_SHA256 = (
    "76a9af5194f35f3faae0d47916549bf65030b65fb01e68837c626406febfda96"
)

STRICT_DEPENDENCY_FILES = {
    "brain/scripts/v21i_strict_live_representation_probe_v1.py": (
        EXACT_STRICT_SCRIPT_SHA256
    ),
    "brain/docs/preregistrations/2026-08-24-v21i-strict-live-representation-probe-v1.md": (
        EXACT_STRICT_DOC_SHA256
    ),
    "brain/tests/test_v21i_strict_live_representation_probe.py": (
        EXACT_STRICT_TEST_SHA256
    ),
}
_PREREGISTRATION = (
    "brain/docs/preregistrations/"
    "2026-08-24-v21i-strict-live-component-localization-v1.md"
)
_TEST_FILE = "brain/tests/test_v21i_strict_live_component_localization_v1.py"
_DIAGNOSTIC_FILES = (
    "brain/scripts/v21i_strict_live_component_localization_v1.py",
    _PREREGISTRATION,
    _TEST_FILE,
)

ARM_N = "N"
ARM_U = "U"
ARM_U0 = "U0"
ARM_B = "B"
ARM_Z = "Z"
ARM_E = "E"
ARM_A = "A"
ARM_BZ = "BZ"
ARM_BE = "BE"
ARM_ZE = "ZE"
ARM_BZE = "BZE"
ARMS = (
    ARM_N, ARM_U, ARM_U0, ARM_B, ARM_Z, ARM_E, ARM_A, ARM_BZ, ARM_BE, ARM_ZE,
    ARM_BZE,
)
REDUCED_ARMS = (ARM_B, ARM_Z, ARM_E, ARM_A, ARM_BZ, ARM_BE, ARM_ZE, ARM_BZE)
PRODUCTION_ARMS = (ARM_B, ARM_Z, ARM_E, ARM_BZ, ARM_BE, ARM_ZE, ARM_BZE)
SINGLE_COMPONENT_ARMS = (ARM_B, ARM_Z, ARM_E)
PAIR_ARMS = (ARM_BZ, ARM_BE, ARM_ZE)
FIXED_TIE_ORDER = (ARM_Z, ARM_E, ARM_B, ARM_ZE, ARM_BZ, ARM_BE, ARM_BZE)

WIDTH = strict.model_config().width
CAPACITY_STATE_WIDTH = strict.CAPACITY_STATE_WIDTH
CAPACITY_HEAD_PARAMETERS = strict.CAPACITY_HEAD_PARAMETERS
PROBE_SEED = strict.PROBE_SEED
BOOTSTRAP_SEED = 32_042
BOOTSTRAP_RESAMPLES = 10_000
PASSES = strict.TOTAL_PROBE_PASSES
NONINFERIORITY_MARGIN = 0.025
EXPLORATORY_REFERENCE_ALPHA = 0.05
SIMULTANEOUS_LOWER_QUANTILE = EXPLORATORY_REFERENCE_ALPHA / len(REDUCED_ARMS)
QUANTILE_METHOD = "linear"
EXPECTED_DEV_CLUSTER_COUNT = 64
EXPECTED_ROOTS_PER_DEV_CLUSTER = 12
EXPECTED_BRANCHES_PER_ROOT = strict.ACTION_COUNT
EXPECTED_INITIAL_STATE_SHA256 = (
    "4a508e6450c509e877e12cfb03d0532d4c396df25292cd7951d2e18c6005fec5"
)

EXPECTED_CONTROL = {
    ARM_N: {
        "final_state_sha256": (
            "dd3aa2792773c3ecb8ae266d0ac3dc9927cd76ead35bd2c33e4d653fe3ea13c1"
        ),
        "TRAIN-FIT_probability_sha256": (
            "115956b79b416f7221c4271511b87e51f12d467358a7da4523b79881fc51e020"
        ),
        "DEV_probability_sha256": (
            "3e6ebc9babfbf2edd06b6b797becdaa3d3b1d3cc84934b5e8de976c771e4c508"
        ),
        "DEV_aggregate_roc_auc": 0.6040297854619737,
    },
    ARM_U: {
        "final_state_sha256": (
            "dc9a854b596066827d6b0ec576c035520d5c39f0901eebbe8d3cbb667258b00b"
        ),
        "TRAIN-FIT_probability_sha256": (
            "f1f775e8363cfb7f4d9ec45d9f91d405f0a3499638ec104a7d1316123653d8cd"
        ),
        "DEV_probability_sha256": (
            "6f4a3e8578b4294d5a48adf9b33644cd91b88bf0e8923b9b676ee635eeac5647"
        ),
        "DEV_aggregate_roc_auc": 0.7321055105872505,
    },
}
EXPECTED_PRIOR_DEV_AGGREGATE_ROC_AUC = 0.5380051490468308


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return strict._array_sha256(values)


def _strict_dependency_bundle(project_root: Path) -> dict[str, object]:
    files: dict[str, str] = {}
    digest = sha256(b"IRV21ICOMPONENTDEPENDENCIES\x01")
    for relative, expected in STRICT_DEPENDENCY_FILES.items():
        observed = _sha256_file(project_root / relative)
        if observed != expected:
            raise ValueError(f"strict-live dependency drifted: {relative}")
        files[relative] = observed
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(observed))
    return {"schema_version": 1, "sha256": digest.hexdigest(), "files": files}


def _diagnostic_source_bundle(project_root: Path) -> dict[str, object]:
    dependencies = _strict_dependency_bundle(project_root)
    strict_live_source = strict._diagnostic_source_bundle(project_root)
    if strict_live_source.get("sha256") != EXACT_STRICT_SOURCE_BUNDLE_SHA256:
        raise ValueError("complete strict-live source bundle drifted")
    digest = sha256(b"IRV21ICOMPONENTLOCALIZATION\x01")
    digest.update(bytes.fromhex(dependencies["sha256"]))
    digest.update(bytes.fromhex(strict_live_source["sha256"]))
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
        "strict_live_dependencies": dependencies,
        "strict_live_source_bundle": strict_live_source,
        "diagnostic_files": files,
    }


def validate_strict_result(project_root: Path) -> dict[str, object]:
    result_path = (project_root / STRICT_RESULT).resolve()
    registration_path = (project_root / STRICT_REGISTRATION).resolve()
    if _sha256_file(result_path) != EXACT_STRICT_RESULT_SHA256:
        raise ValueError("exact strict-live v2 result is absent or drifted")
    if _sha256_file(registration_path) != EXACT_STRICT_REGISTRATION_SHA256:
        raise ValueError("exact strict-live v2 registration is absent or drifted")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    source = payload.get("source_bundle")
    metrics = payload.get("metrics")
    if (
        payload.get("mode") != strict.MODE
        or payload.get("classification") != strict.CLASSIFICATION
        or payload.get("qualification_claimed") is not False
        or payload.get("candidate_publication_allowed") is not False
        or not isinstance(source, Mapping)
        or source.get("sha256") != EXACT_STRICT_SOURCE_BUNDLE_SHA256
        or not isinstance(metrics, Mapping)
    ):
        raise ValueError("strict-live v2 result envelope drifted")
    current_source = strict._diagnostic_source_bundle(project_root)
    if current_source != source:
        raise ValueError("current complete strict-live source bundle differs from result")
    return {
        "artifact": STRICT_RESULT,
        "sha256": EXACT_STRICT_RESULT_SHA256,
        "registration": {
            "artifact": STRICT_REGISTRATION,
            "sha256": EXACT_STRICT_REGISTRATION_SHA256,
        },
        "source_bundle_sha256": EXACT_STRICT_SOURCE_BUNDLE_SHA256,
        "source_bundle": current_source,
        "diagnosis": payload.get("interpretation", {}).get("diagnosis"),
    }


def feature_contract() -> dict[str, object]:
    return {
        "width": CAPACITY_STATE_WIDTH,
        "slots": [
            {"slice": [0, 120], "value": "B_or_N"},
            {"slice": [120, 240], "value": "Z"},
            {"slice": [240, 360], "value": "E"},
            {"slice": [360, 365], "value": "A"},
            {"slice": [365, 366], "value": "P"},
        ],
        "symbols": {
            "N": "post_update_belief",
            "B": "old_belief",
            "Z": "current_rgb_encoder_latent",
            "E": "live_only_cognitive_prediction_error",
            "A": "internally_owned_prior_applied_action_onehot",
            "P": "has_pending_prediction",
        },
        "pending_contract": (
            "P must be exactly one after burn-in; it is unidentifiable and is zero "
            "in every reduced arm"
        ),
        "feature_denylist": list(strict._FORBIDDEN_FEATURES),
    }


def schedule_record() -> dict[str, object]:
    roots = strict.TRAIN_FIT_EPISODES * (
        strict.SEQUENCE_LENGTH - strict.BURN_IN_STEPS
    )
    steps_per_pass = math.ceil(roots / strict.ROOT_BATCH_SIZE)
    return {
        "train_partition": "TRAIN-FIT",
        "evaluation_partitions": ["TRAIN-FIT", "DEV"],
        "fit_all_arms_before_any_DEV_prediction_or_metric": True,
        "train_cal_constructed": False,
        "cpu_qual_opened": False,
        "test_split_opened": False,
        "strict_live_replay_batch_size": 1,
        "arms": list(ARMS),
        "passes_per_arm": PASSES,
        "roots_per_pass": roots,
        "root_batch_size": strict.ROOT_BATCH_SIZE,
        "steps_per_pass": steps_per_pass,
        "optimizer_steps_per_arm": PASSES * steps_per_pass,
        "optimizer": "AdamW_fresh_restart",
        "learning_rate": strict.LEARNING_RATE,
        "weight_decay": strict.WEIGHT_DECAY,
        "clip_norm": strict.CLIP_NORM,
        "loss": "unweighted_binary_cross_entropy_with_logits_all_five_actions",
        "initialization_seed": PROBE_SEED,
        "permutation_seed": PROBE_SEED,
        "identical_initialization_all_arms": True,
        "identical_complete_root_permutations_all_arms": True,
        "early_stopping": False,
        "retry_allowed": False,
        "model_selection_from_observed_auc": False,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "DEV_episode_cluster_paired_with_replacement",
            "shared_index_matrix_for_every_comparison": True,
            "inference_status": (
                "deterministic_exploratory_engineering_bound_on_already_consumed_DEV_"
                "not_confirmatory_nominal_FWER"
            ),
            "exploratory_reference_alpha": EXPLORATORY_REFERENCE_ALPHA,
            "reduced_arm_count": len(REDUCED_ARMS),
            "one_sided_lower_quantile": SIMULTANEOUS_LOWER_QUANTILE,
            "quantile_method": QUANTILE_METHOD,
            "delta_orientation": "left_minus_right",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "margin_status": "exploratory_not_scientifically_validated",
            "fresh_DEV_required_for_confirmation": True,
            "expected_DEV_geometry": {
                "episode_clusters": EXPECTED_DEV_CLUSTER_COUNT,
                "roots_per_episode_cluster": EXPECTED_ROOTS_PER_DEV_CLUSTER,
                "counterfactual_action_branches_per_root": EXPECTED_BRANCHES_PER_ROOT,
            },
        },
    }


def diagnostic_scope() -> dict[str, object]:
    return {
        "sensory_signature": "rgb_plus_internally_owned_prior_applied_action_v1",
        "actual_reward": None,
        "actual_hazard": None,
        "constructed_partitions": ["TRAIN-FIT", "DEV"],
        "fit_partition": "TRAIN-FIT",
        "dev_role": "one_joint_final_endpoint_after_all_arms_are_fitted",
        "train_cal_constructed": False,
        "cpu_qual_opened": False,
        "test_split_opened": False,
        "candidate_publication_allowed": False,
        "checkpoint_emitted": False,
        "architecture_nomination_only": True,
        "direct_production_architecture_change_supported": False,
        "consumed_dev_reusable_for_production_qualification": False,
    }


def canonical_paths(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve()
    return {
        "upstream_result": (root / CANONICAL_UPSTREAM_RESULT).resolve(),
        "registration": (root / CANONICAL_REGISTRATION).resolve(),
        "result": (root / CANONICAL_RESULT).resolve(),
    }


def run_identity() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "upstream_result": CANONICAL_UPSTREAM_RESULT,
        "registration": CANONICAL_REGISTRATION,
        "result": CANONICAL_RESULT,
        "retry_allowed": False,
        "alternate_paths_allowed": False,
    }


def _require_canonical_path(path: Path, expected: Path, *, role: str) -> Path:
    observed = path.resolve()
    if observed != expected.resolve():
        raise ValueError(f"{role} must use the single canonical path {expected}")
    return observed


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
        "run_identity": run_identity(),
        "strict_live_v2": validate_strict_result(project_root),
        "upstream": {
            "v3_result_sha256": strict.EXACT_V3_RESULT_SHA256,
            "v3_parent_sha256": strict.EXACT_V3_PARENT_SHA256,
            "v3_parent_state_sha256": strict.EXACT_V3_PARENT_STATE_SHA256,
            "v3_source_bundle_sha256": strict.EXACT_V3_SOURCE_BUNDLE_SHA256,
            "train_fit_manifest_sha256": strict.EXACT_TRAIN_FIT_MANIFEST_SHA256,
            "dev_manifest_sha256": strict.EXACT_DEV_MANIFEST_SHA256,
        },
        "diagnostic_source_bundle": _diagnostic_source_bundle(project_root),
        "feature_contract": feature_contract(),
        "schedule": schedule_record(),
        "scope": diagnostic_scope(),
        "control_expectations": EXPECTED_CONTROL,
        "selection_contract": selection_contract(),
    }


def register(registration: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(root)
    canonical_registration = _require_canonical_path(
        registration, paths["registration"], role="registration"
    )
    if paths["result"].exists():
        raise FileExistsError("canonical diagnostic result already exists; retry is forbidden")
    _publish_json_create_only(canonical_registration, registration_payload(root))


def validate_registration(registration: Path) -> dict[str, object]:
    registration = registration.resolve()
    root = Path(__file__).resolve().parents[2]
    registration = _require_canonical_path(
        registration, canonical_paths(root)["registration"], role="registration"
    )
    if not registration.is_file():
        raise FileNotFoundError("component-localization registration is required")
    payload = json.loads(registration.read_text(encoding="utf-8"))
    if payload != registration_payload(root):
        raise ValueError("component-localization registration or sources drifted")
    return {
        "path": str(registration),
        "sha256": _sha256_file(registration),
        "mode": REGISTRATION_MODE,
        "diagnostic_source_bundle": payload["diagnostic_source_bundle"],
        "strict_live_v2": payload["strict_live_v2"],
    }


def _zero_features(rows: int, *, dtype: np.dtype) -> np.ndarray:
    return np.zeros((rows, CAPACITY_STATE_WIDTH), dtype=dtype)


def component_features(tape: strict.StrictLiveFeatureTape, arm: str) -> np.ndarray:
    """Project one exact 366-D preregistered arm without target/provenance inputs."""

    if WIDTH != 120 or CAPACITY_STATE_WIDTH != 366:
        raise RuntimeError("registered component widths drifted")
    updater = tape.updater_inputs
    if updater.shape[1] != CAPACITY_STATE_WIDTH:
        raise ValueError("updater input width drifted")
    if not np.array_equal(updater[:, 365:366], np.ones_like(updater[:, 365:366])):
        raise RuntimeError("P must be exactly one at every post-burn-in root")
    active = np.zeros(CAPACITY_STATE_WIDTH, dtype=np.bool_)
    if arm == ARM_U:
        output = np.array(updater, dtype=np.float32, order="C", copy=True)
        active[:] = True
    elif arm == ARM_U0:
        output = np.array(updater, dtype=np.float32, order="C", copy=True)
        output[:, 365:366] = 0.0
        active[:365] = True
    else:
        output = _zero_features(len(updater), dtype=np.dtype(np.float32))
        if arm == ARM_N:
            output[:, 0:120] = tape.beliefs
            active[0:120] = True
        else:
            components = {
                ARM_B: ((0, 120),),
                ARM_Z: ((120, 240),),
                ARM_E: ((240, 360),),
                ARM_A: ((360, 365),),
                ARM_BZ: ((0, 120), (120, 240)),
                ARM_BE: ((0, 120), (240, 360)),
                ARM_ZE: ((120, 240), (240, 360)),
                ARM_BZE: ((0, 120), (120, 240), (240, 360)),
            }
            if arm not in components:
                raise ValueError(f"unknown component-localization arm: {arm}")
            for start, stop in components[arm]:
                output[:, start:stop] = updater[:, start:stop]
                active[start:stop] = True
    if output.dtype != np.float32 or not output.flags.c_contiguous:
        raise RuntimeError("projected feature table must be C-contiguous float32")
    if not np.isfinite(output).all():
        raise RuntimeError("projected feature table contains non-finite values")
    masked = output[:, ~active]
    if np.any(masked != 0.0) or np.any(np.signbit(masked)):
        raise RuntimeError("masked feature padding must be canonical positive zero")
    return output


def initialized_heads() -> dict[str, strict.CapacityHazardProbe]:
    heads = {
        arm: strict.initialized_probe(strict.CapacityHazardProbe, seed=PROBE_SEED)
        for arm in ARMS
    }
    digests = {_state_dict_sha256(head.state_dict()) for head in heads.values()}
    if digests != {EXPECTED_INITIAL_STATE_SHA256}:
        raise RuntimeError("the eleven arms do not have identical initialization")
    for head in heads.values():
        if sum(parameter.numel() for parameter in head.parameters()) != CAPACITY_HEAD_PARAMETERS:
            raise RuntimeError("matched capacity-head parameter count drifted")
    return heads


def control_reproduction(
    training: Mapping[str, Mapping[str, object]],
    metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    arms: dict[str, object] = {}
    for arm in (ARM_N, ARM_U):
        expected = EXPECTED_CONTROL[arm]
        observed = {
            "final_state_sha256": training[arm]["final_state_sha256"],
            "TRAIN-FIT_probability_sha256": metrics[arm]["probability_sha256"]["TRAIN-FIT"],
            "DEV_probability_sha256": metrics[arm]["probability_sha256"]["DEV"],
            "DEV_aggregate_roc_auc": metrics[arm]["DEV"]["aggregate"]["roc_auc"],
        }
        checks = {name: observed[name] == value for name, value in expected.items()}
        arms[arm] = {
            "passed": all(checks.values()),
            "checks": checks,
            "expected": expected,
            "observed": observed,
            "comparison": "exact_no_tolerance",
        }
    passed = all(record["passed"] is True for record in arms.values())
    return {"passed": passed, "arms": arms, "failure_diagnosis": None if passed else "control_reproduction_failure"}


def _bootstrap_indices(
    episode_group_ids: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], np.ndarray]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    groups = tuple(dict.fromkeys(episode_group_ids))
    if not groups:
        raise ValueError("episode groups must be nonempty")
    group_array = np.asarray(episode_group_ids, dtype=object)
    group_rows = {group: np.flatnonzero(group_array == group) for group in groups}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(resamples, len(groups)), dtype=np.int64)
    return groups, group_rows, indices


def _auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(binary_probability_metrics(targets.reshape(-1), probabilities.reshape(-1)).roc_auc)


def shared_bootstrap_report(
    targets: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    episode_group_ids: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Use one shared DEV episode index matrix for all reduced-arm IUTs."""

    if any(value.shape != targets.shape for value in probabilities.values()):
        raise ValueError("bootstrap probability tables are not aligned")
    if targets.ndim != 2 or targets.shape[1] != EXPECTED_BRANCHES_PER_ROOT:
        raise ValueError("DEV bootstrap must contain exactly five branches per root")
    groups, group_rows, sampled = _bootstrap_indices(
        episode_group_ids, resamples=resamples, seed=seed
    )
    if len(groups) != EXPECTED_DEV_CLUSTER_COUNT:
        raise ValueError("DEV bootstrap must contain exactly 64 episode clusters")
    if any(len(rows) != EXPECTED_ROOTS_PER_DEV_CLUSTER for rows in group_rows.values()):
        raise ValueError("every DEV episode cluster must contain exactly 12 roots")
    deltas = {
        arm: {
            "minus_N": np.empty(resamples, dtype=np.float64),
            "minus_U0": np.empty(resamples, dtype=np.float64),
            "minus_U": np.empty(resamples, dtype=np.float64),
        }
        for arm in REDUCED_ARMS
    }
    u_minus_u0 = np.empty(resamples, dtype=np.float64)
    for index, sample in enumerate(sampled):
        rows = np.concatenate([group_rows[groups[int(value)]] for value in sample])
        target_rows = targets[rows]
        auc_n = _auc(target_rows, probabilities[ARM_N][rows])
        auc_u = _auc(target_rows, probabilities[ARM_U][rows])
        auc_u0 = _auc(target_rows, probabilities[ARM_U0][rows])
        u_minus_u0[index] = auc_u - auc_u0
        for arm in REDUCED_ARMS:
            auc_arm = _auc(target_rows, probabilities[arm][rows])
            deltas[arm]["minus_N"][index] = auc_arm - auc_n
            deltas[arm]["minus_U0"][index] = auc_arm - auc_u0
            deltas[arm]["minus_U"][index] = auc_arm - auc_u
    comparisons: dict[str, object] = {}
    point_n = _auc(targets, probabilities[ARM_N])
    point_u = _auc(targets, probabilities[ARM_U])
    point_u0 = _auc(targets, probabilities[ARM_U0])
    for arm in REDUCED_ARMS:
        point = _auc(targets, probabilities[arm])
        record: dict[str, object] = {}
        for name, point_delta in (
            ("minus_N", point - point_n),
            ("minus_U0", point - point_u0),
            ("minus_U", point - point_u),
        ):
            values = deltas[arm][name]
            record[name] = {
                "point_delta": point_delta,
                "descriptive_lower_95": float(
                    np.quantile(values, 0.025, method=QUANTILE_METHOD)
                ),
                "descriptive_upper_95": float(
                    np.quantile(values, 0.975, method=QUANTILE_METHOD)
                ),
                "simultaneous_one_sided_lower": float(
                    np.quantile(
                        values,
                        SIMULTANEOUS_LOWER_QUANTILE,
                        method=QUANTILE_METHOD,
                    )
                ),
                "delta_orientation": "left_minus_right",
                "quantile_method": QUANTILE_METHOD,
            }
        comparisons[arm] = record
    return {
        "resamples": resamples,
        "seed": seed,
        "clustering": "DEV_episode_paired_with_replacement",
        "group_count": len(groups),
        "roots_per_group": EXPECTED_ROOTS_PER_DEV_CLUSTER,
        "branches_per_root": EXPECTED_BRANCHES_PER_ROOT,
        "ordered_groups_sha256": strict._ordered_sequence_digest(groups),
        "shared_sampled_group_index_sha256": _array_sha256(sampled),
        "multiplicity_method": (
            "eight_arm_exploratory_Bonferroni_within_arm_intersection_union_test"
        ),
        "inference_status": (
            "deterministic_exploratory_engineering_evidence_on_already_consumed_"
            "DEV_not_confirmatory_nominal_FWER"
        ),
        "exploratory_reference_alpha": EXPLORATORY_REFERENCE_ALPHA,
        "one_sided_lower_quantile": SIMULTANEOUS_LOWER_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "delta_orientation": "left_minus_right",
        "fresh_DEV_required_for_confirmation": True,
        "U_minus_U0_descriptive": {
            "point_delta": point_u - point_u0,
            "lower_95": float(
                np.quantile(u_minus_u0, 0.025, method=QUANTILE_METHOD)
            ),
            "upper_95": float(
                np.quantile(u_minus_u0, 0.975, method=QUANTILE_METHOD)
            ),
            "delta_orientation": "U_minus_U0",
            "quantile_method": QUANTILE_METHOD,
            "used_as_eligibility_gate": False,
            "purpose": "detect_constant_pending_intercept_or_optimization_effect",
        },
        "comparisons": comparisons,
    }


def selection_contract() -> dict[str, object]:
    known_u_minus_n_gain = (
        EXPECTED_CONTROL[ARM_U]["DEV_aggregate_roc_auc"]
        - EXPECTED_CONTROL[ARM_N]["DEV_aggregate_roc_auc"]
    )
    return {
        "ranking_gate": {
            "minimum_aggregate_roc_auc": strict.MINIMUM_AGGREGATE_ROC_AUC,
            "minimum_prior_auc_gain": strict.MINIMUM_BASELINE_ROC_AUC_GAIN,
            "minimum_every_action_roc_auc": strict.MINIMUM_PER_ACTION_ROC_AUC,
            "exact_prior_dev_aggregate_roc_auc": EXPECTED_PRIOR_DEV_AGGREGATE_ROC_AUC,
        },
        "reduced_arm_eligibility": [
            "U0_unchanged_ranking_gate_passes_as_global_precondition",
            "unchanged_ranking_gate_passes",
            "Bonferroni_simultaneous_lower_arm_minus_N_gt_0",
            "exploratory_Bonferroni_lower_arm_minus_U0_gt_-0.025",
            "exploratory_Bonferroni_lower_arm_minus_U_gt_-0.025",
        ],
        "inference": {
            "test": "exploratory_intersection_union_engineering_rule",
            "three_IUT_limbs_share_same_quantile_without_within_arm_alpha_split": True,
            "confirmatory_nominal_FWER_claimed": False,
            "already_consumed_DEV": True,
            "fresh_DEV_required_for_confirmation": True,
            "reduced_arm_count": len(REDUCED_ARMS),
            "exploratory_reference_alpha": EXPLORATORY_REFERENCE_ALPHA,
            "one_sided_lower_quantile": SIMULTANEOUS_LOWER_QUANTILE,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "noninferiority_margin_status": (
                "exploratory_not_scientifically_validated"
            ),
            "known_U_minus_N_gain": known_u_minus_n_gain,
            "approximate_retention_of_known_U_minus_N_gain": (
                1.0 - NONINFERIORITY_MARGIN / known_u_minus_n_gain
            ),
        },
        "exact_U_role": "byte_reproduction_control_and_direct_noninferiority_reference",
        "U0_role": "pending_zero_full_updater_inferential_comparator",
        "native_slot_initialization_caveat": (
            "fixed native input slots use different random first-layer blocks; arm "
            "ordering is initialization-and-slot-confounded"
        ),
        "claim_scope": (
            "nomination_for_fresh_production_form_probe_only_not_unique_causality"
        ),
        "fresh_production_form_requirements": (
            "multiple_fixed_initializations_and_fresh_seeds"
        ),
        "prediction_error_action_caveat": (
            "E contains error against the pending predicted-next latent from the prior "
            "selected action; BZE versus U0 isolates only incremental explicit A onehot"
        ),
        "production_eligible_arms": list(PRODUCTION_ARMS),
        "diagnostic_only_arms": [ARM_A],
        "prefer_single_component_then_pair_then_dense_triple": True,
        "fixed_tie_order": list(FIXED_TIE_ORDER),
        "tie_order_independent_of_observed_auc": True,
    }


def select_component(
    ranking_gates: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, object],
    controls: Mapping[str, object],
) -> dict[str, object]:
    if controls.get("passed") is not True:
        return {
            "diagnosis": "control_reproduction_failure",
            "scientific_interpretation_allowed": False,
            "selected_arm": None,
            "architecture_change_supported": False,
            "eligible_reduced_arms": [],
        }
    comparisons = bootstrap["comparisons"]
    if ranking_gates[ARM_U0]["passed"] is not True:
        return {
            "diagnosis": "pending_intercept_optimization_confounded",
            "scientific_interpretation_allowed": False,
            "selected_arm": None,
            "architecture_change_supported": False,
            "eligible_reduced_arms": [],
            "exact_U_ranking_gate_passed": ranking_gates[ARM_U]["passed"] is True,
            "U0_ranking_gate_passed": False,
            "reason": (
                "exact U passed but pending-zero U0 failed, so omission cannot be "
                "separated from a constant-intercept or optimization effect"
            ),
        }
    eligibility: dict[str, object] = {}
    for arm in REDUCED_ARMS:
        checks = {
            "ranking_gate": ranking_gates[arm]["passed"] is True,
            "superior_to_N": comparisons[arm]["minus_N"]["simultaneous_one_sided_lower"] > 0.0,
            "exploratory_noninferior_to_U0": (
                comparisons[arm]["minus_U0"]["simultaneous_one_sided_lower"]
                > -NONINFERIORITY_MARGIN
            ),
            "exploratory_noninferior_to_U": (
                comparisons[arm]["minus_U"]["simultaneous_one_sided_lower"]
                > -NONINFERIORITY_MARGIN
            ),
        }
        eligibility[arm] = {"eligible": all(checks.values()), "checks": checks}
    eligible = [arm for arm in REDUCED_ARMS if eligibility[arm]["eligible"]]
    production = [arm for arm in FIXED_TIE_ORDER if arm in eligible]
    eligible_singles = [arm for arm in production if arm in SINGLE_COMPONENT_ARMS]
    selected = eligible_singles[0] if eligible_singles else (production[0] if production else None)
    if selected is not None:
        diagnosis = "component_set_nominated_for_production_form_probe"
    elif eligible == [ARM_A]:
        diagnosis = "behavior_history_shortcut_supported"
    elif ranking_gates[ARM_U0]["passed"] is True:
        diagnosis = "multi_component_dependence_unresolved"
    else:
        diagnosis = "no_architecture_change_supported"
    return {
        "diagnosis": diagnosis,
        "scientific_interpretation_allowed": True,
        "selected_arm": selected,
        "architecture_change_supported": False,
        "next_production_form_probe_nominated": selected is not None,
        "eligible_reduced_arms": eligible,
        "eligibility": eligibility,
        "selection_order": list(FIXED_TIE_ORDER),
        "single_component_preferred_before_pair_then_dense_triple": True,
        "observed_auc_used_to_break_ties": False,
        "A_is_diagnostic_only": True,
        "U0_ranking_gate_global_precondition": True,
        "exact_U_used_as_byte_control_and_direct_noninferiority_reference": True,
        "noninferiority_margin_status": "exploratory_not_scientifically_validated",
        "unique_component_causality_claimed": False,
        "native_slot_initialization_confounded": True,
        "fresh_production_form_probe_requires_multiple_fixed_initializations_and_seeds": True,
        "BZE_minus_U0_scope": (
            "incremental_explicit_prior_action_onehot_only_not_all_action_dependence"
        ),
        "production_claim": False,
    }


def _run_impl(
    *,
    upstream_result: Path,
    output: Path,
    registration_record: Mapping[str, object],
) -> None:
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(PROBE_SEED)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("component localization must hide CUDA")
    project_root = Path(__file__).resolve().parents[2]
    if registration_record["diagnostic_source_bundle"] != _diagnostic_source_bundle(project_root):
        raise RuntimeError("registered diagnostic source bundle drifted before execution")
    model, _, provenance = strict.load_exact_parent(upstream_result)
    model.to(torch.device("cpu"))
    parent_before = _state_dict_sha256(model.state_dict())
    sources = strict.strict_live_partition_sources()
    if sources["TRAIN-FIT"].manifest_sha256 != strict.EXACT_TRAIN_FIT_MANIFEST_SHA256:
        raise RuntimeError("TRAIN-FIT manifest differs from the registered source")
    if sources["DEV"].manifest_sha256 != strict.EXACT_DEV_MANIFEST_SHA256:
        raise RuntimeError("DEV manifest differs from the registered source")
    train_tape = strict.collect_strict_live_tape(model, sources["TRAIN-FIT"])
    dev_tape = strict.collect_strict_live_tape(model, sources["DEV"])
    if _state_dict_sha256(model.state_dict()) != parent_before:
        raise RuntimeError("strict-live collection mutated the parent")

    train_features = {arm: component_features(train_tape, arm) for arm in ARMS}
    dev_features = {arm: component_features(dev_tape, arm) for arm in ARMS}
    heads = initialized_heads()
    permutations = strict.deterministic_permutations(
        len(train_tape.beliefs), passes=PASSES, seed=PROBE_SEED
    )
    training: dict[str, object] = {}
    # No DEV metric or prediction is computed until all eleven fits are complete.
    for arm in ARMS:
        training[arm] = strict.train_probe_head(
            heads[arm], train_features[arm], train_tape.hazard_targets, permutations
        )

    probabilities: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, object] = {}

    def evaluate_arm(arm: str) -> None:
        train_probability = strict.probability_table(heads[arm], train_features[arm])
        dev_probability = strict.probability_table(heads[arm], dev_features[arm])
        probabilities[arm] = {"TRAIN-FIT": train_probability, "DEV": dev_probability}
        metrics[arm] = {
            "TRAIN-FIT": all_action_probability_metrics(
                train_tape.hazard_targets, train_probability
            ).as_dict(),
            "DEV": all_action_probability_metrics(
                dev_tape.hazard_targets, dev_probability
            ).as_dict(),
            "probability_sha256": {
                "TRAIN-FIT": _array_sha256(train_probability),
                "DEV": _array_sha256(dev_probability),
            },
        }

    # Exact controls are the only DEV outputs computed until the frozen prior
    # experiment is reproduced byte-for-byte.  A mismatch terminates the sole
    # canonical run before any reduced-arm metric or bootstrap is exposed.
    evaluate_arm(ARM_N)
    evaluate_arm(ARM_U)
    controls = control_reproduction(training, metrics)
    if controls["passed"] is not True:
        raise RuntimeError("exact N/U control reproduction failed")
    for arm in ARMS:
        if arm not in {ARM_N, ARM_U}:
            evaluate_arm(arm)
    prior_reports, _, _ = strict.fixed_baseline_reports(train_tape, dev_tape)
    prior_auc = float(
        prior_reports["train_fitted_action_prior"]["partitions"]["DEV"]["aggregate"]["roc_auc"]
    )
    if prior_auc != EXPECTED_PRIOR_DEV_AGGREGATE_ROC_AUC:
        raise RuntimeError("exact TRAIN-FIT-fitted prior DEV AUC drifted")
    ranking_gates = {
        arm: strict.ranking_gate(metrics[arm]["DEV"], prior_aggregate_auc=prior_auc)
        for arm in ARMS
    }
    bootstrap = shared_bootstrap_report(
        dev_tape.hazard_targets,
        {arm: probabilities[arm]["DEV"] for arm in ARMS},
        dev_tape.episode_group_ids,
    )
    selection = select_component(ranking_gates, bootstrap, controls)
    parent_after = _state_dict_sha256(model.state_dict())
    if parent_after != parent_before:
        raise RuntimeError("component localization mutated the frozen parent")
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "candidate_checkpoint": {
            "published": False,
            "publication_allowed": False,
            "checkpoint_emitted": False,
        },
        "device": "cpu",
        "threads": 1,
        "cuda_visible_devices": "-1",
        "registration": {
            "path": registration_record["path"],
            "sha256": registration_record["sha256"],
            "verified_before_run": True,
        },
        "strict_live_v2": registration_record["strict_live_v2"],
        "upstream": provenance,
        "source_bundle": registration_record["diagnostic_source_bundle"],
        "namespace_policy": diagnostic_scope(),
        "feature_contract": feature_contract(),
        "schedule": schedule_record(),
        "evidence": {
            "TRAIN-FIT": strict.evidence_manifest(
                "TRAIN-FIT", train_tape,
                dataset_manifest_sha256=strict.EXACT_TRAIN_FIT_MANIFEST_SHA256,
            ),
            "DEV": strict.evidence_manifest(
                "DEV", dev_tape,
                dataset_manifest_sha256=strict.EXACT_DEV_MANIFEST_SHA256,
            ),
        },
        "parent_state": {
            "before_sha256": parent_before,
            "after_sha256": parent_after,
            "byte_identical": True,
            "asserted_before_publication": True,
        },
        "training": training,
        "metrics": {"baselines": prior_reports, "arms": metrics},
        "control_reproduction": controls,
        "ranking_gates": ranking_gates,
        "bootstrap_inference": bootstrap,
        "selection": selection,
        "interpretation_constraints": {
            "architecture_nomination_only": True,
            "direct_production_architecture_change_supported": False,
            "checkpoint_or_candidate_claim": False,
            "current_DEV_is_consumed_for_development": True,
            "fresh_preregistered_DEV_required_for_production_qualification": True,
            "fresh_production_form_probe_requires_multiple_fixed_initializations_and_seeds": True,
            "unique_component_causality_claimed": False,
            "native_slot_initialization_confounded": True,
            "E_contains_prior_selected_action_dependent_prediction_error": True,
            "BZE_vs_U0_isolates_only_incremental_explicit_A_onehot": True,
            "longer_history_information_loss_ruled_out": False,
        },
    }
    _publish_json_create_only(output.resolve(), result)


def run(*, upstream_result: Path, output: Path, registration: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(project_root)
    upstream_result = _require_canonical_path(
        upstream_result, paths["upstream_result"], role="upstream result"
    )
    output = _require_canonical_path(output, paths["result"], role="result")
    registration = _require_canonical_path(
        registration, paths["registration"], role="registration"
    )
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    registration_record: Mapping[str, object] | None = None
    try:
        registration_record = validate_registration(registration)
        _run_impl(
            upstream_result=upstream_result,
            output=output,
            registration_record=registration_record,
        )
    except BaseException as error:
        if not output.exists():
            _publish_json_create_only(
                output.resolve(),
                {
                    "schema_version": SCHEMA_VERSION,
                    "implementation_revision": IMPLEMENTATION_REVISION,
                    "mode": MODE,
                    "classification": "diagnostic_run_failed_not_candidate",
                    "qualification_claimed": False,
                    "candidate_publication_allowed": False,
                    "candidate_checkpoint": {
                        "published": False,
                        "publication_allowed": False,
                        "checkpoint_emitted": False,
                    },
                    "registration": {
                        "path": str(registration.resolve()),
                        "sha256": (
                            _sha256_file(registration.resolve())
                            if registration.resolve().is_file() else None
                        ),
                        "verified_before_run": registration_record is not None,
                    },
                    "strict_sensory_and_namespace_scope": diagnostic_scope(),
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
