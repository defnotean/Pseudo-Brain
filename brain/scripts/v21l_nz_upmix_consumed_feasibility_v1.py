"""PB21L NZ scorer-upmix feasibility study on already-consumed PB21K data.

This runner is permanently exploratory and nonqualifying.  It reuses exactly
four frozen PB21K cells and compares the original all-action NZ scorer loss to
two preregistered losses which also weight the current factual branch.  It can
issue only a fresh-data architecture license; it cannot nominate, publish a
checkpoint, qualify a model, or authorize a same-DEV retry.
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
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from irene_brain.evaluation.v21_qualification_metrics import (
    binary_probability_metrics,
)
from run_provenance import apply_deterministic_mode
import v21i_development_runner as development
import v21i_strict_live_representation_probe_v1 as strict
import v21j_fresh_bz_production_form_diagnostic_v1 as pb21j
import v21k_nz_dualcal_diagnostic_v1 as pb21k


SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = 2
MODE = "pb21l_nz_upmix_consumed_feasibility_v1"
REGISTRATION_MODE = "pb21l_nz_upmix_consumed_feasibility_registration_v1"
CLASSIFICATION = "exploratory_consumed_data_nonqualifying"
RUN_ID = "2026-08-25-pb21l-nz-upmix-consumed-v1"

CANONICAL_UPSTREAM_RESULT = pb21k.CANONICAL_UPSTREAM_RESULT
CANONICAL_REGISTRATION = (
    "brain/runs/v21l-diagnostics/"
    "2026-08-25-pb21l-nz-upmix-consumed-v1.registration.json"
)
CANONICAL_ATTEMPT = (
    "brain/runs/v21l-diagnostics/"
    "2026-08-25-pb21l-nz-upmix-consumed-v1.attempt.json"
)
CANONICAL_EVIDENCE = (
    "brain/runs/v21l-diagnostics/"
    "2026-08-25-pb21l-nz-upmix-consumed-v1.evidence.npz"
)
CANONICAL_RESULT = (
    "brain/runs/v21l-diagnostics/2026-08-25-pb21l-nz-upmix-consumed-v1.json"
)

PB21K_REGISTRATION = pb21k.CANONICAL_REGISTRATION
PB21K_ATTEMPT = pb21k.CANONICAL_ATTEMPT
PB21K_EVIDENCE = pb21k.CANONICAL_EVIDENCE
PB21K_RESULT = pb21k.CANONICAL_RESULT
EXACT_PB21K_REGISTRATION_SHA256 = (
    "f6bc3e25412ea6c3058bc38ace7622e30fdec72c6732388cb7a7d7dfa38f7058"
)
EXACT_PB21K_ATTEMPT_SHA256 = (
    "bd2ae53311733900a421b9928f2db73f32e5aa82d14a523f27f7bcf367b2a094"
)
EXACT_PB21K_EVIDENCE_SHA256 = (
    "9d57bf4a6bd60121e8f11e2b9405c1a6d64a9346381503054d6ad9cd4ece385c"
)
EXACT_PB21K_RESULT_SHA256 = (
    "083ac9c44d3327f4f806b1e4842ea3991ec54e5bab65992e7c5446ad09ca02d4"
)
EXACT_PB21K_SOURCE_BUNDLE_SHA256 = (
    "b13b764c2ff55dd15649945fec7c05211a3c59fc04c7ecfb667d1b711590e03a"
)
EXACT_PB21K_EVIDENCE_MANIFEST_SHA256 = (
    "256f25b832aad90457fd8528979a6a60a24992d265dc864771b8a1f569c29be5"
)
EXACT_PB21K_EVIDENCE_KEY_SET_SHA256 = (
    "b11966d1003e1082586adbedbad46c69e6ce2af11964dcad24eb305ffbdc562f"
)
EXACT_PB21K_BOOTSTRAP_SHA256 = (
    "80c2c95e3273d510ec6f36fea873c78fcd3f5d059e9985132b1bbbdd973dc140"
)
EXACT_PB21K_DERANGEMENT_SHA256 = (
    "c142d8b756ff30f58417faf2f29546e59b53f792973f4dddcb9833ca4f790672"
)

_PREREGISTRATION = (
    "brain/docs/preregistrations/2026-08-25-pb21l-nz-upmix-consumed-v1.md"
)
_TEST_FILE = "brain/tests/test_v21l_nz_upmix_consumed_feasibility_v1.py"
_DIAGNOSTIC_FILES = (
    "brain/scripts/v21l_nz_upmix_consumed_feasibility_v1.py",
    _PREREGISTRATION,
    _TEST_FILE,
)

ACTION_COUNT = pb21j.ACTION_COUNT
SUPERSET_WIDTH = pb21j.SUPERSET_WIDTH
ARM_NZ = pb21j.ARM_NZ
SCORER_BASE = "BASE"
SCORER_POOL = "POOL-UPMIX"
SCORER_BAL = "BAL-UPMIX"
SCORERS = (SCORER_BASE, SCORER_POOL, SCORER_BAL)
FIT_COHORTS = ("F0", "F2")
CAL_COHORTS = ("C0", "C2")
COHORT_PAIRS = (("F0", "C0"), ("F2", "C2"))
INIT_SEEDS = (51_042, 53_042)
CELLS = tuple(
    f"{fit_label}/init-{seed}"
    for fit_label in FIT_COHORTS
    for seed in INIT_SEEDS
)
PB21K_CELL_INDICES = np.asarray((0, 2, 6, 8), dtype=np.int16)
PB21K_FIT_INDICES = np.asarray((0, 2), dtype=np.int16)
PB21K_CAL_INDICES = np.asarray((0, 2), dtype=np.int16)
PB21K_AA_INDEX = 0
PB21K_NZ_INDEX = pb21k.ARMS.index(ARM_NZ)

PASSES = pb21k.PASSES
ROOT_BATCH_SIZE = pb21k.ROOT_BATCH_SIZE
ROOTS_PER_FIT = pb21k.ROOTS_PER_FIT
ROOTS_PER_CAL = pb21k.ROOTS_PER_CAL
STEPS_PER_HEAD = pb21k.STEPS_PER_HEAD
PERMUTATION_SEED = pb21k.PERMUTATION_SEED
TRAIN_HEAD_PARAMETERS = pb21k.TRAIN_HEAD_PARAMETERS
DEV_EPISODES = pb21k.DEV_EPISODES
DEV_ROOTS = pb21k.DEV_ROOTS
ROOTS_PER_EPISODE = pb21k.ROOTS_PER_EPISODE
BOOTSTRAP_RESAMPLES = pb21k.BOOTSTRAP_RESAMPLES
DERANGEMENT_REPETITIONS = pb21k.DERANGEMENT_REPETITIONS
ONE_SIDED_ALPHA = pb21k.ONE_SIDED_ALPHA
QUANTILE_METHOD = pb21k.QUANTILE_METHOD
AA_BCE_MARGIN = 0.010
AA_BRIER_MARGIN = 0.005

DOMAINS = ("all_action", "factual", "nonselected_complement")
MAXIMUM_AGGREGATE_ECE = 0.05
MAXIMUM_PER_ACTION_ECE = 0.075
MAXIMUM_PER_ACTION_BIAS = 0.05
MAXIMUM_FACTUAL_PER_ACTION_BIAS = 0.075
MINIMUM_PR_PREVALENCE_GAIN = 0.05
AA_SOURCE_NAMESPACE = "maze_chase.pb21k.train-only.v1"
AA_SOURCE_SPLIT = "TRAIN"
AA_SOURCE_PARTITION = "TRAIN-CAL"

BASE_FINAL_STATE_SHA256 = {
    "F0/init-51042": "4c77ef99561ddb0aabe656aa5414cbd35e917ca423461bdc33de50c1e1b9fdcb",
    "F0/init-53042": "5ffae5687580e7cbbe9222af5fb0f06d8d1a1d20ed4d7adbbdaba5d9f9f850ad",
    "F2/init-51042": "7cdb516996ce2d12dabb4f2f56887560d03ad9878d742608635bb780d1f4e843",
    "F2/init-53042": "824450b0e7f447ba1285a06cb8e38112dab48e99134f1552601f9455c768ab44",
}
BASE_PRUNED_STATE_SHA256 = {
    "F0/init-51042": "94ce782bbf65faaf236957ddaf552548daf0813346ee9f3297fe931c1ff7875f",
    "F0/init-53042": "82f9eba516747d7c8e0da1af6d1152f9f566a8eec88c221df454c58268959e01",
    "F2/init-51042": "cd391735e0f81a5c7203bbb02b886864c0bd800a0b30d20bc0b7bc19a0b4e79e",
    "F2/init-53042": "5a001858ee8e76a525e808c01fd2b06091f6136aabcc724e8caf70f7652e3167",
}


@dataclass(frozen=True)
class ExpectedSourceLineage:
    fit_group_ids: tuple[tuple[str, ...], ...]
    cal_group_ids: tuple[tuple[str, ...], ...]
    fit_group_ordered_sha256: tuple[str, ...]
    cal_group_ordered_sha256: tuple[str, ...]
    fit_nz_feature_sha256: tuple[str, ...]

EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "scorer_ids",
        "cell_ids",
        "cell_fit_index",
        "cell_init_seed",
        "factual_action_counts",
        "factual_balance_weights",
        "factual_balance_weights_consumed_float32",
        "training_initial_state_sha256",
        "training_final_state_sha256",
        "training_pass_number",
        "training_pass_mean_all_BCE",
        "training_pass_mean_factual_component_BCE",
        "training_pass_mean_objective",
        "training_pass_maximum_preclip_gradient_norm",
        "training_pass_permutation_sha256",
        "training_factual_components_recorded",
        "pruned_state_sha256",
        "pruning_mapping_sha256",
        "pruning_active_columns",
        "pruning_parameter_count",
        "pruning_padded_logit_sha256",
        "pruning_pruned_logit_sha256",
        "pruning_maximum_absolute_difference",
        "calibration_raw_logit_sha256",
        "calibration_group_sha256",
        "calibration_fit_group_sha256",
        "calibration_source_namespace",
        "calibration_source_split",
        "calibration_source_partition",
        "calibration_upstream_checkpoint_sha256",
        "calibration_dataset_manifest_sha256",
        "calibration_source_bundle_sha256",
        "calibration_partition_algorithm",
        "fit_targets",
        "fit_actions",
        "fit_root_ids",
        "fit_group_ids",
        "fit_group_ordered_sha256",
        "fit_nz_feature_sha256",
        "cal_targets",
        "cal_actions",
        "cal_root_ids",
        "cal_group_ids",
        "cal_group_ordered_sha256",
        "dev_targets",
        "dev_actions",
        "dev_root_ids",
        "dev_cluster_ordinal",
        "dev_cluster_ids",
        "cal_raw_logits",
        "aa_scales",
        "aa_biases",
        "aa_accepted",
        "dev_raw_logits",
        "dev_raw_probabilities",
        "dev_calibrated_logits",
        "dev_calibrated_probabilities",
        "bootstrap_indices",
        "episode_derangements",
    }
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return pb21k._array_sha256(np.asarray(values))


def _state_sha256(module: nn.Module) -> str:
    return pb21k._state_sha256(module)


def _byte_strings(values: Sequence[str], width: int) -> np.ndarray:
    return pb21k._byte_string_array(tuple(values), width=width)


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return pb21k._decode_byte_strings(np.asarray(values))


def evidence_key_set_sha256() -> str:
    payload = "\n".join(sorted(EVIDENCE_KEYS)).encode("ascii")
    return sha256(b"IRPB21LKEYSET\x01" + payload).hexdigest()


def expected_evidence_contract() -> dict[str, dict[str, object]]:
    shapes = {
        "schema_version": ((), "<u2"),
        "scorer_ids": ((3,), "|S12"),
        "cell_ids": ((4,), "|S16"),
        "cell_fit_index": ((4,), "<i2"),
        "cell_init_seed": ((4,), "<i4"),
        "factual_action_counts": ((4, ACTION_COUNT), "<i8"),
        "factual_balance_weights": ((4, ROOTS_PER_FIT), "<f8"),
        "factual_balance_weights_consumed_float32": ((4, ROOTS_PER_FIT), "<f4"),
        "training_initial_state_sha256": ((3, 4), "|S64"),
        "training_final_state_sha256": ((3, 4), "|S64"),
        "training_pass_number": ((3, 4, PASSES), "<i2"),
        "training_pass_mean_all_BCE": ((3, 4, PASSES), "<f8"),
        "training_pass_mean_factual_component_BCE": ((3, 4, PASSES), "<f8"),
        "training_pass_mean_objective": ((3, 4, PASSES), "<f8"),
        "training_pass_maximum_preclip_gradient_norm": ((3, 4, PASSES), "<f8"),
        "training_pass_permutation_sha256": ((3, 4, PASSES), "|S64"),
        "training_factual_components_recorded": ((3,), "|b1"),
        "pruned_state_sha256": ((3, 4), "|S64"),
        "pruning_mapping_sha256": ((3, 4), "|S64"),
        "pruning_active_columns": ((len(pb21j.ACTIVE_COLUMNS[ARM_NZ]),), "<i2"),
        "pruning_parameter_count": ((3, 4), "<i4"),
        "pruning_padded_logit_sha256": ((3, 4), "|S64"),
        "pruning_pruned_logit_sha256": ((3, 4), "|S64"),
        "pruning_maximum_absolute_difference": ((3, 4), "<f8"),
        "calibration_raw_logit_sha256": ((3, 4), "|S64"),
        "calibration_group_sha256": ((3, 4), "|S64"),
        "calibration_fit_group_sha256": ((3, 4), "|S64"),
        "calibration_source_namespace": ((3, 4), "|S64"),
        "calibration_source_split": ((3, 4), "|S16"),
        "calibration_source_partition": ((3, 4), "|S16"),
        "calibration_upstream_checkpoint_sha256": ((3, 4), "|S64"),
        "calibration_dataset_manifest_sha256": ((3, 4), "|S64"),
        "calibration_source_bundle_sha256": ((3, 4), "|S64"),
        "calibration_partition_algorithm": ((3, 4), "|S512"),
        "fit_targets": ((2, ROOTS_PER_FIT, ACTION_COUNT), "<f8"),
        "fit_actions": ((2, ROOTS_PER_FIT), "<i8"),
        "fit_root_ids": ((2, ROOTS_PER_FIT), "|S64"),
        "fit_group_ids": ((2, ROOTS_PER_FIT), "|S64"),
        "fit_group_ordered_sha256": ((2,), "|S64"),
        "fit_nz_feature_sha256": ((2,), "|S64"),
        "cal_targets": ((2, ROOTS_PER_CAL, ACTION_COUNT), "<f8"),
        "cal_actions": ((2, ROOTS_PER_CAL), "<i8"),
        "cal_root_ids": ((2, ROOTS_PER_CAL), "|S64"),
        "cal_group_ids": ((2, ROOTS_PER_CAL), "|S64"),
        "cal_group_ordered_sha256": ((2,), "|S64"),
        "dev_targets": ((DEV_ROOTS, ACTION_COUNT), "<f8"),
        "dev_actions": ((DEV_ROOTS,), "<i8"),
        "dev_root_ids": ((DEV_ROOTS,), "|S64"),
        "dev_cluster_ordinal": ((DEV_ROOTS,), "<i4"),
        "dev_cluster_ids": ((DEV_EPISODES,), "|S64"),
        "cal_raw_logits": ((3, 4, ROOTS_PER_CAL, ACTION_COUNT), "<f8"),
        "aa_scales": ((3, 4, ACTION_COUNT), "<f8"),
        "aa_biases": ((3, 4, ACTION_COUNT), "<f8"),
        "aa_accepted": ((3, 4), "|b1"),
        "dev_raw_logits": ((3, 4, DEV_ROOTS, ACTION_COUNT), "<f8"),
        "dev_raw_probabilities": ((3, 4, DEV_ROOTS, ACTION_COUNT), "<f8"),
        "dev_calibrated_logits": ((3, 4, DEV_ROOTS, ACTION_COUNT), "<f8"),
        "dev_calibrated_probabilities": ((3, 4, DEV_ROOTS, ACTION_COUNT), "<f8"),
        "bootstrap_indices": ((BOOTSTRAP_RESAMPLES, DEV_EPISODES), "<i4"),
        "episode_derangements": ((DERANGEMENT_REPETITIONS, DEV_EPISODES), "<i4"),
    }
    if set(shapes) != EVIDENCE_KEYS:
        raise RuntimeError("PB21L evidence schema/key contract drifted")
    return {
        name: {"shape": list(shape), "dtype": dtype}
        for name, (shape, dtype) in sorted(shapes.items())
    }


def canonical_paths(project_root: Path) -> dict[str, Path]:
    return {
        "upstream_result": project_root / CANONICAL_UPSTREAM_RESULT,
        "registration": project_root / CANONICAL_REGISTRATION,
        "attempt": project_root / CANONICAL_ATTEMPT,
        "evidence": project_root / CANONICAL_EVIDENCE,
        "result": project_root / CANONICAL_RESULT,
    }


def _require_canonical_path(path: Path, expected: Path, *, role: str) -> Path:
    observed = path.resolve()
    canonical = expected.resolve()
    if observed != canonical:
        raise ValueError(f"PB21L {role} must use canonical path {canonical}")
    return observed


def validate_pb21k_parent(project_root: Path) -> dict[str, object]:
    artifacts = (
        (PB21K_REGISTRATION, EXACT_PB21K_REGISTRATION_SHA256, "registration"),
        (PB21K_ATTEMPT, EXACT_PB21K_ATTEMPT_SHA256, "attempt"),
        (PB21K_EVIDENCE, EXACT_PB21K_EVIDENCE_SHA256, "evidence"),
        (PB21K_RESULT, EXACT_PB21K_RESULT_SHA256, "result"),
    )
    for relative, expected, label in artifacts:
        path = project_root / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"exact PB21K {label} absent or drifted")
    result = json.loads((project_root / PB21K_RESULT).read_text(encoding="utf-8"))
    evidence = result.get("authoritative_evidence", {})
    selection = result.get("selection", {})
    if (
        result.get("mode") != pb21k.MODE
        or result.get("classification") != pb21k.CLASSIFICATION
        or result.get("candidate_publication_allowed") is not False
        or result.get("source_bundle", {}).get("sha256")
        != EXACT_PB21K_SOURCE_BUNDLE_SHA256
        or evidence.get("sha256") != EXACT_PB21K_EVIDENCE_SHA256
        or evidence.get("array_manifest_sha256")
        != EXACT_PB21K_EVIDENCE_MANIFEST_SHA256
        or evidence.get("allowed_key_set_sha256")
        != EXACT_PB21K_EVIDENCE_KEY_SET_SHA256
        or evidence.get("validation", {}).get("passed") is not True
        or selection.get("eligible") is not False
        or selection.get("selected_recipe") is not None
        or selection.get("diagnosis")
        != "NZ_representation_supported_dual_calibration_comparison_unresolved"
    ):
        raise ValueError("PB21K scientific parent envelope drifted")
    current_bundle = pb21k._diagnostic_source_bundle(project_root)
    if current_bundle.get("sha256") != EXACT_PB21K_SOURCE_BUNDLE_SHA256:
        raise ValueError("PB21K source bundle drifted")
    return {
        "registration": PB21K_REGISTRATION,
        "registration_sha256": EXACT_PB21K_REGISTRATION_SHA256,
        "attempt": PB21K_ATTEMPT,
        "attempt_sha256": EXACT_PB21K_ATTEMPT_SHA256,
        "evidence": PB21K_EVIDENCE,
        "evidence_sha256": EXACT_PB21K_EVIDENCE_SHA256,
        "evidence_array_manifest_sha256": EXACT_PB21K_EVIDENCE_MANIFEST_SHA256,
        "result": PB21K_RESULT,
        "result_sha256": EXACT_PB21K_RESULT_SHA256,
        "source_bundle_sha256": EXACT_PB21K_SOURCE_BUNDLE_SHA256,
        "frozen_diagnosis": selection["diagnosis"],
        "consumed_data": True,
    }


def _diagnostic_source_bundle(project_root: Path) -> dict[str, object]:
    parent = pb21k._diagnostic_source_bundle(project_root)
    if parent.get("sha256") != EXACT_PB21K_SOURCE_BUNDLE_SHA256:
        raise ValueError("exact PB21K source bundle drifted")
    digest = sha256(b"IRPB21LNZUPMIXCONSUMED\x01")
    digest.update(bytes.fromhex(EXACT_PB21K_SOURCE_BUNDLE_SHA256))
    files: dict[str, str] = {}
    for relative in (*_DIAGNOSTIC_FILES, PB21K_REGISTRATION, PB21K_ATTEMPT,
                     PB21K_EVIDENCE, PB21K_RESULT):
        observed = _sha256_file(project_root / relative)
        files[relative] = observed
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(observed))
    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "PB21K_source_bundle": parent,
        "bound_files": files,
    }


def aa_provenance_contract(pb21l_source_bundle_sha256: str) -> dict[str, object]:
    try:
        digest_bytes = bytes.fromhex(pb21l_source_bundle_sha256)
    except ValueError as error:
        raise ValueError("PB21L source-bundle digest must be hexadecimal") from error
    if len(digest_bytes) != 32:
        raise ValueError("PB21L source-bundle digest must be SHA-256")
    manifests = pb21k.partition_manifests()
    cell_manifests = tuple(manifests[dict(COHORT_PAIRS)[cell.split("/", 1)[0]]]
                           for cell in CELLS)
    source_bundles = (
        (EXACT_PB21K_SOURCE_BUNDLE_SHA256,) * len(CELLS),
        (pb21l_source_bundle_sha256,) * len(CELLS),
        (pb21l_source_bundle_sha256,) * len(CELLS),
    )
    return {
        "source_namespace": AA_SOURCE_NAMESPACE,
        "source_split": AA_SOURCE_SPLIT,
        "source_partition": AA_SOURCE_PARTITION,
        "partition_algorithm": pb21k.PARTITION_ALGORITHM,
        "dataset_manifest_sha256_by_cell": list(cell_manifests),
        "source_bundle_sha256_by_scorer_and_cell": [
            list(row) for row in source_bundles
        ],
    }


def objective_contract() -> dict[str, object]:
    return {
        "BASE": "mean_all_root_action_BCE",
        "POOL-UPMIX": "0.5*mean_all_root_action_BCE+0.5*mean_root_factual_BCE",
        "BAL-UPMIX": (
            "0.5*mean_action(mean_root_BCE_for_action)+"
            "0.5*mean_factual_action(mean_matching_root_BCE)"
        ),
        "BAL_minibatch_weight": "ROOTS_PER_FIT/(5*full_FIT_factual_count[action])",
        "factual_selector": "loss_only_current_applied_action_never_feature",
    }


def inference_contract() -> dict[str, object]:
    return {
        "domains": list(DOMAINS),
        "all_and_factual_absolute_gates": "PB21K_AA_plus_fixed_per_action_proper_score",
        "complement_definition": "semantic_action_a_on_roots_with_factual_action_not_a",
        "complement_maximum_per_action_absolute_bias": MAXIMUM_PER_ACTION_BIAS,
        "complement_maximum_per_action_ECE": MAXIMUM_PER_ACTION_ECE,
        "direct_comparisons": {
            "factual_BCE": 0.0,
            "factual_Brier": 0.0,
            "all_action_BCE": -AA_BCE_MARGIN,
            "all_action_Brier": -AA_BRIER_MARGIN,
            "nonselected_complement_BCE": -AA_BCE_MARGIN,
            "nonselected_complement_Brier": -AA_BRIER_MARGIN,
        },
        "every_cell_point_and_grand_LCB_required": True,
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "bootstrap": "exact_frozen_PB21K_shared_episode_matrix",
        "bootstrap_sha256": EXACT_PB21K_BOOTSTRAP_SHA256,
        "derangement": "exact_frozen_PB21K_shared_episode_derangements",
        "derangement_sha256": EXACT_PB21K_DERANGEMENT_SHA256,
    }


def schedule_record() -> dict[str, object]:
    return {
        "fit_cohorts": list(FIT_COHORTS),
        "cal_cohorts": list(CAL_COHORTS),
        "matched_pairs": [list(pair) for pair in COHORT_PAIRS],
        "initialization_seeds": list(INIT_SEEDS),
        "cells": list(CELLS),
        "scorers": list(SCORERS),
        "heads": len(CELLS) * len(SCORERS),
        "passes": PASSES,
        "root_batch_size": ROOT_BATCH_SIZE,
        "roots_per_pass": ROOTS_PER_FIT,
        "steps_per_head": STEPS_PER_HEAD,
        "total_optimizer_steps": len(CELLS) * len(SCORERS) * STEPS_PER_HEAD,
        "permutation_seed": PERMUTATION_SEED,
        "same_NZ_features_head_initialization_and_permutations": True,
        "standard_AA_calibration_only": True,
        "DEV_consumed_from_PB21K": True,
        "new_seed_namespace_opened": False,
        "device": "cpu",
        "threads": 1,
        "cuda_visible_devices": "-1",
    }


def decision_contract() -> dict[str, object]:
    return {
        "precedence": [SCORER_BAL, SCORER_POOL, None],
        "BAL_selected_if_passes": True,
        "POOL_selected_only_if_POOL_passes_and_BAL_fails": True,
        "output": "fresh_data_license_only",
        "nomination_allowed": False,
        "checkpoint_emitted": False,
        "candidate_publication_allowed": False,
        "qualification_claimed": False,
        "same_consumed_DEV_retry_allowed": False,
    }


def registration_payload(project_root: Path) -> dict[str, object]:
    parent = validate_pb21k_parent(project_root)
    source_bundle = _diagnostic_source_bundle(project_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": REGISTRATION_MODE,
        "classification": "exploratory_consumed_data_registration_nonqualifying",
        "create_only": True,
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "checkpoint_emitted": False,
        "PB21K_parent": parent,
        "source_bundle": source_bundle,
        "AA_provenance_contract": aa_provenance_contract(str(source_bundle["sha256"])),
        "consumed_partitions": ["F0", "F2", "C0", "C2", "DEV"],
        "consumed_cells": list(CELLS),
        "fresh_partition_constructed": False,
        "objective_contract": objective_contract(),
        "schedule": schedule_record(),
        "inference": inference_contract(),
        "decision": decision_contract(),
        "authoritative_evidence_contract": {
            "allowed_key_set_sha256": evidence_key_set_sha256(),
            "arrays": expected_evidence_contract(),
            "allow_pickle": False,
            "deterministic_create_only_NPZ": True,
        },
    }


def register(registration: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(project_root)
    expected = paths["registration"]
    registration = _require_canonical_path(registration, expected, role="registration")
    for role in ("attempt", "evidence", "result"):
        if paths[role].exists():
            raise FileExistsError(
                f"PB21L canonical {role} exists; registration/retry is forbidden"
            )
    development._publish_json_create_only(registration, registration_payload(project_root))


def validate_registration(registration: Path) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    expected = canonical_paths(project_root)["registration"]
    registration = _require_canonical_path(registration, expected, role="registration")
    if not registration.is_file():
        raise FileNotFoundError("PB21L registration is absent")
    payload = json.loads(registration.read_text(encoding="utf-8"))
    if payload != registration_payload(project_root):
        raise ValueError("PB21L registration payload drifted")
    return {
        "path": str(registration),
        "sha256": _sha256_file(registration),
        "payload": payload,
    }


def load_frozen_pb21k_evidence(
    project_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    parent = validate_pb21k_parent(project_root)
    result = json.loads((project_root / PB21K_RESULT).read_text(encoding="utf-8"))
    publication = result["authoritative_evidence"]
    path = project_root / PB21K_EVIDENCE
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(publication["arrays"]):
            raise ValueError("PB21K evidence key set drifted")
        for name in archive.files:
            value = pb21k._canonical_evidence_array(archive[name])
            expected = publication["arrays"][name]
            if (
                value.dtype.str != expected["dtype"]
                or list(value.shape) != expected["shape"]
                or _array_sha256(value) != expected["sha256"]
            ):
                raise ValueError(f"PB21K evidence array drifted: {name}")
            arrays[name] = value
    if (
        _array_sha256(arrays["bootstrap_indices"]) != EXACT_PB21K_BOOTSTRAP_SHA256
        or _array_sha256(arrays["episode_derangements"])
        != EXACT_PB21K_DERANGEMENT_SHA256
    ):
        raise ValueError("PB21K shared inference arrays drifted")
    return arrays, parent


def build_expected_source_lineage(
    *,
    fit_tapes: Mapping[str, pb21j.FreshLiveTape],
    cal_tapes: Mapping[str, pb21j.FreshLiveTape],
    fit_features: Mapping[str, np.ndarray],
    source_evidence: Mapping[str, Mapping[str, object]],
) -> ExpectedSourceLineage:
    fit_groups = tuple(tuple(fit_tapes[label].episode_group_ids) for label in FIT_COHORTS)
    cal_groups = tuple(tuple(cal_tapes[label].episode_group_ids) for label in CAL_COHORTS)
    if any(len(row) != ROOTS_PER_FIT for row in fit_groups) or any(
        len(row) != ROOTS_PER_CAL for row in cal_groups
    ):
        raise ValueError("PB21L source-lineage root geometry drifted")
    fit_digests = tuple(strict._ordered_sequence_digest(row) for row in fit_groups)
    cal_digests = tuple(strict._ordered_sequence_digest(row) for row in cal_groups)
    feature_digests = tuple(_array_sha256(fit_features[label]) for label in FIT_COHORTS)
    for index, label in enumerate(FIT_COHORTS):
        evidence = source_evidence[label]
        if (
            evidence.get("ordered_episode_sha256") != fit_digests[index]
            or evidence.get("NZ_feature_sha256") != feature_digests[index]
        ):
            raise ValueError(f"prepublication FIT lineage drifted for {label}")
    for index, label in enumerate(CAL_COHORTS):
        if source_evidence[label].get("ordered_episode_sha256") != cal_digests[index]:
            raise ValueError(f"prepublication CAL lineage drifted for {label}")
    return ExpectedSourceLineage(
        fit_group_ids=fit_groups,
        cal_group_ids=cal_groups,
        fit_group_ordered_sha256=fit_digests,
        cal_group_ordered_sha256=cal_digests,
        fit_nz_feature_sha256=feature_digests,
    )


def factual_action_counts(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.int64)
    if values.shape != (ROOTS_PER_FIT,) or np.any((values < 0) | (values >= ACTION_COUNT)):
        raise ValueError("FIT factual action geometry drifted")
    counts = np.bincount(values, minlength=ACTION_COUNT).astype(np.int64, copy=False)
    if np.any(counts == 0) or int(counts.sum()) != ROOTS_PER_FIT:
        raise ValueError("every factual action needs FIT support")
    return counts


def factual_balance_weights(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.int64)
    counts = factual_action_counts(values)
    weights = ROOTS_PER_FIT / (ACTION_COUNT * counts[values].astype(np.float64))
    if not math.isclose(float(weights.mean()), 1.0, rel_tol=0.0, abs_tol=1.0e-14):
        raise RuntimeError("balanced factual weights must have unit root mean")
    return weights


def consumed_factual_balance_weights(actions: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(factual_balance_weights(actions), dtype=np.float32)


def upmix_loss(
    logits: Tensor,
    targets: Tensor,
    factual_actions: Tensor,
    *,
    scorer: str,
    balanced_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape[1] != ACTION_COUNT:
        raise ValueError("upmix loss tables must be Bx5 and aligned")
    if factual_actions.shape != (len(logits),) or factual_actions.dtype != torch.long:
        raise ValueError("factual selector must be aligned int64 action IDs")
    all_loss = F.binary_cross_entropy_with_logits(logits, targets)
    rows = torch.arange(len(logits), device=logits.device)
    factual_terms = F.binary_cross_entropy_with_logits(
        logits[rows, factual_actions],
        targets[rows, factual_actions],
        reduction="none",
    )
    pooled = factual_terms.mean()
    if scorer == SCORER_BASE:
        objective = all_loss
        factual_component = pooled
    elif scorer == SCORER_POOL:
        objective = 0.5 * all_loss + 0.5 * pooled
        factual_component = pooled
    elif scorer == SCORER_BAL:
        if balanced_weights is None or balanced_weights.shape != (len(logits),):
            raise ValueError("BAL-UPMIX requires frozen per-root factual weights")
        balanced = torch.mean(factual_terms * balanced_weights)
        objective = 0.5 * all_loss + 0.5 * balanced
        factual_component = balanced
    else:
        raise ValueError(f"unknown scorer: {scorer}")
    return objective, {
        "all_action_BCE": float(all_loss.detach()),
        "factual_component_BCE": float(factual_component.detach()),
        "objective": float(objective.detach()),
    }


def initialized_head(seed: int) -> pb21j.MaskedSupersetHazardHead:
    if seed not in INIT_SEEDS:
        raise ValueError("PB21L initialization seed is not preregistered")
    return pb21k.initialized_head(seed)


def train_upmix_head(
    head: pb21j.MaskedSupersetHazardHead,
    features: np.ndarray,
    targets: np.ndarray,
    factual_actions: np.ndarray,
    permutations: Sequence[Tensor],
    *,
    scorer: str,
) -> dict[str, object]:
    if scorer == SCORER_BASE:
        record = pb21j.train_head(head, features, targets, permutations)
        return {**record, "scorer": scorer, "objective": objective_contract()[scorer]}
    if features.shape != (ROOTS_PER_FIT, SUPERSET_WIDTH) or targets.shape != (
        ROOTS_PER_FIT,
        ACTION_COUNT,
    ):
        raise ValueError("PB21L FIT geometry drifted")
    actions_np = np.asarray(factual_actions, dtype=np.int64)
    counts = factual_action_counts(actions_np)
    weights_np = factual_balance_weights(actions_np)
    consumed_weights_np = consumed_factual_balance_weights(actions_np)
    states = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    labels = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
    actions = torch.from_numpy(actions_np).to(torch.long)
    weights = torch.from_numpy(consumed_weights_np)
    parameters = tuple(head.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=pb21j.LEARNING_RATE, weight_decay=pb21j.WEIGHT_DECAY
    )
    initial = _state_sha256(head)
    pass_records: list[dict[str, object]] = []
    steps = 0
    head.train()
    for pass_index, permutation in enumerate(permutations, start=1):
        if len(permutation) != ROOTS_PER_FIT or int(permutation.unique().numel()) != ROOTS_PER_FIT:
            raise ValueError("training permutation drifted")
        sums = np.zeros(3, dtype=np.float64)
        maximum_gradient = 0.0
        for start in range(0, ROOTS_PER_FIT, ROOT_BATCH_SIZE):
            indices = permutation[start : start + ROOT_BATCH_SIZE]
            loss, components = upmix_loss(
                head(states[indices]),
                labels[indices],
                actions[indices],
                scorer=scorer,
                balanced_weights=(weights[indices] if scorer == SCORER_BAL else None),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite PB21L scorer loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(parameters, pb21j.CLIP_NORM))
            if not math.isfinite(gradient):
                raise FloatingPointError("non-finite PB21L scorer gradient")
            optimizer.step()
            batch = len(indices)
            sums += batch * np.asarray(
                [components["all_action_BCE"], components["factual_component_BCE"],
                 components["objective"]],
                dtype=np.float64,
            )
            maximum_gradient = max(maximum_gradient, gradient)
            steps += 1
        pass_records.append(
            {
                "pass": pass_index,
                "mean_all_action_BCE": float(sums[0] / ROOTS_PER_FIT),
                "mean_factual_component_BCE": float(sums[1] / ROOTS_PER_FIT),
                "mean_objective": float(sums[2] / ROOTS_PER_FIT),
                "maximum_preclip_gradient_norm": maximum_gradient,
                "permutation_sha256": _array_sha256(
                    permutation.numpy().astype(np.int64, copy=False)
                ),
            }
        )
    if steps != STEPS_PER_HEAD:
        raise RuntimeError("PB21L optimizer-step count drifted")
    head.eval()
    return {
        "scorer": scorer,
        "objective": objective_contract()[scorer],
        "initial_state_sha256": initial,
        "final_state_sha256": _state_sha256(head),
        "passes": PASSES,
        "optimizer_steps": steps,
        "factual_action_counts": counts.tolist(),
        "factual_balance_weights_by_action": (
            ROOTS_PER_FIT / (ACTION_COUNT * counts.astype(np.float64))
        ).tolist(),
        "factual_balance_weights_consumed_float32_by_action": [
            float(consumed_weights_np[np.flatnonzero(actions_np == action)[0]])
            for action in range(ACTION_COUNT)
        ],
        "factual_selector_is_loss_only": True,
        "pass_records": pass_records,
        "early_stopping": False,
        "model_selection": False,
    }


def _canonical_array(value: np.ndarray) -> np.ndarray:
    return pb21k._canonical_evidence_array(np.asarray(value))


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return pb21k._array_manifest(arrays)


def _manifest_sha256(manifest: Mapping[str, Mapping[str, object]]) -> str:
    encoded = json.dumps(
        {name: dict(manifest[name]) for name in sorted(manifest)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRPB21LARRAYMANIFEST\x01" + encoded).hexdigest()


def _pruning_mapping_sha256(mapping: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(mapping), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return sha256(b"IRPB21LPRUNINGMAP\x01" + encoded).hexdigest()


def build_evidence_arrays(
    *,
    frozen: Mapping[str, np.ndarray],
    source_lineage: ExpectedSourceLineage,
    training: Mapping[str, Mapping[str, object]],
    pruning: Mapping[str, Mapping[str, object]],
    calibration: Mapping[str, Mapping[str, object]],
    cal_raw_logits: np.ndarray,
    calibrators: Sequence[Sequence[object]],
    dev_raw_logits: np.ndarray,
    dev_raw_probabilities: np.ndarray,
    dev_calibrated_logits: np.ndarray,
    dev_calibrated_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    fit_targets = frozen["fit_targets"][PB21K_FIT_INDICES]
    fit_actions = frozen["fit_actions"][PB21K_FIT_INDICES]
    def pass_value(scorer: str, cell: str, index: int, key: str) -> float:
        record = training[cell][scorer]["pass_records"][index]
        if key == "mean_factual_component_BCE" and key not in record:
            return 0.0
        if key == "mean_objective" and key not in record:
            return float(record["mean_all_action_BCE"])
        return float(record[key])

    initial_states = [
        [str(training[cell][scorer]["initial_state_sha256"]) for cell in CELLS]
        for scorer in SCORERS
    ]
    final_states = [
        [str(training[cell][scorer]["final_state_sha256"]) for cell in CELLS]
        for scorer in SCORERS
    ]
    pruned_states = [
        [str(pruning[cell][scorer]["mapping"]["pruned_state_sha256"]) for cell in CELLS]
        for scorer in SCORERS
    ]
    provenance = [
        [calibrators[s][c].provenance for c in range(len(CELLS))]
        for s in range(len(SCORERS))
    ]
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.uint16),
        "scorer_ids": _byte_strings(SCORERS, 12),
        "cell_ids": _byte_strings(CELLS, 16),
        "cell_fit_index": np.asarray((0, 0, 1, 1), dtype=np.int16),
        "cell_init_seed": np.asarray((51_042, 53_042, 51_042, 53_042), dtype=np.int32),
        "factual_action_counts": np.stack(
            [factual_action_counts(fit_actions[index]) for index in (0, 0, 1, 1)]
        ),
        "factual_balance_weights": np.stack(
            [factual_balance_weights(fit_actions[index]) for index in (0, 0, 1, 1)]
        ),
        "factual_balance_weights_consumed_float32": np.stack(
            [
                consumed_factual_balance_weights(fit_actions[index])
                for index in (0, 0, 1, 1)
            ]
        ),
        "training_initial_state_sha256": np.asarray(initial_states, dtype="S64"),
        "training_final_state_sha256": np.asarray(final_states, dtype="S64"),
        "training_pass_number": np.asarray(
            [[[int(training[cell][scorer]["pass_records"][index]["pass"])
               for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS],
            dtype=np.int16,
        ),
        "training_pass_mean_all_BCE": np.asarray(
            [[[pass_value(scorer, cell, index, "mean_all_action_BCE")
               for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS],
            dtype=np.float64,
        ),
        "training_pass_mean_factual_component_BCE": np.asarray(
            [[[pass_value(scorer, cell, index, "mean_factual_component_BCE")
               for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS],
            dtype=np.float64,
        ),
        "training_pass_mean_objective": np.asarray(
            [[[pass_value(scorer, cell, index, "mean_objective")
               for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS],
            dtype=np.float64,
        ),
        "training_pass_maximum_preclip_gradient_norm": np.asarray(
            [[[pass_value(scorer, cell, index, "maximum_preclip_gradient_norm")
               for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS],
            dtype=np.float64,
        ),
        "training_pass_permutation_sha256": np.asarray(
            [[[str(training[cell][scorer]["pass_records"][index]["permutation_sha256"])
               for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS],
            dtype="S64",
        ),
        "training_factual_components_recorded": np.asarray(
            [False, True, True], dtype=np.bool_
        ),
        "pruned_state_sha256": np.asarray(pruned_states, dtype="S64"),
        "pruning_mapping_sha256": np.asarray(
            [[_pruning_mapping_sha256(pruning[cell][scorer]["mapping"])
              for cell in CELLS] for scorer in SCORERS],
            dtype="S64",
        ),
        "pruning_active_columns": np.asarray(pb21j.ACTIVE_COLUMNS[ARM_NZ], dtype=np.int16),
        "pruning_parameter_count": np.asarray(
            [[int(pruning[cell][scorer]["mapping"]["parameter_count"])
              for cell in CELLS] for scorer in SCORERS], dtype=np.int32
        ),
        "pruning_padded_logit_sha256": np.asarray(
            [[str(pruning[cell][scorer]["FIT_inference_transfer"]["padded_logit_sha256"])
              for cell in CELLS] for scorer in SCORERS], dtype="S64"
        ),
        "pruning_pruned_logit_sha256": np.asarray(
            [[str(pruning[cell][scorer]["FIT_inference_transfer"]["pruned_logit_sha256"])
              for cell in CELLS] for scorer in SCORERS], dtype="S64"
        ),
        "pruning_maximum_absolute_difference": np.asarray(
            [[float(pruning[cell][scorer]["FIT_inference_transfer"]
                    ["maximum_absolute_difference"])
              for cell in CELLS] for scorer in SCORERS], dtype=np.float64
        ),
        "calibration_raw_logit_sha256": np.asarray(
            [[str(calibration[cell][scorer]["shared_raw_CAL_logit_sha256"])
              for cell in CELLS] for scorer in SCORERS], dtype="S64"
        ),
        "calibration_group_sha256": np.asarray(
            [[value.calibration_group_digest for value in row] for row in provenance],
            dtype="S64",
        ),
        "calibration_fit_group_sha256": np.asarray(
            [[value.upstream_model_fit_group_digest for value in row] for row in provenance],
            dtype="S64",
        ),
        "calibration_source_namespace": np.asarray(
            [[value.source_namespace for value in row] for row in provenance], dtype="S64"
        ),
        "calibration_source_split": np.asarray(
            [[value.source_split for value in row] for row in provenance], dtype="S16"
        ),
        "calibration_source_partition": np.asarray(
            [[value.source_partition for value in row] for row in provenance], dtype="S16"
        ),
        "calibration_upstream_checkpoint_sha256": np.asarray(
            [[value.upstream_checkpoint_sha256 for value in row] for row in provenance],
            dtype="S64",
        ),
        "calibration_dataset_manifest_sha256": np.asarray(
            [[value.dataset_manifest_sha256 for value in row] for row in provenance],
            dtype="S64",
        ),
        "calibration_source_bundle_sha256": np.asarray(
            [[value.source_bundle_sha256 for value in row] for row in provenance],
            dtype="S64",
        ),
        "calibration_partition_algorithm": np.asarray(
            [[value.partition_algorithm for value in row] for row in provenance],
            dtype="S512",
        ),
        "fit_targets": fit_targets,
        "fit_actions": fit_actions,
        "fit_root_ids": frozen["fit_root_ids"][PB21K_FIT_INDICES],
        "fit_group_ids": np.stack(
            [_byte_strings(row, 64) for row in source_lineage.fit_group_ids]
        ),
        "fit_group_ordered_sha256": _byte_strings(
            source_lineage.fit_group_ordered_sha256, 64
        ),
        "fit_nz_feature_sha256": _byte_strings(
            source_lineage.fit_nz_feature_sha256, 64
        ),
        "cal_targets": frozen["cal_targets"][PB21K_CAL_INDICES],
        "cal_actions": frozen["cal_actions"][PB21K_CAL_INDICES],
        "cal_root_ids": frozen["cal_root_ids"][PB21K_CAL_INDICES],
        "cal_group_ids": np.stack(
            [_byte_strings(row, 64) for row in source_lineage.cal_group_ids]
        ),
        "cal_group_ordered_sha256": _byte_strings(
            source_lineage.cal_group_ordered_sha256, 64
        ),
        "dev_targets": frozen["dev_targets"],
        "dev_actions": frozen["dev_actions"],
        "dev_root_ids": frozen["dev_root_ids"],
        "dev_cluster_ordinal": frozen["dev_cluster_ordinal"],
        "dev_cluster_ids": frozen["dev_cluster_ids"],
        "cal_raw_logits": cal_raw_logits,
        "aa_scales": np.asarray(
            [[calibrators[s][c].scales for c in range(len(CELLS))]
             for s in range(len(SCORERS))],
            dtype=np.float64,
        ),
        "aa_biases": np.asarray(
            [[calibrators[s][c].biases for c in range(len(CELLS))]
             for s in range(len(SCORERS))],
            dtype=np.float64,
        ),
        "aa_accepted": np.asarray(
            [[calibrators[s][c].accepted for c in range(len(CELLS))]
             for s in range(len(SCORERS))],
            dtype=np.bool_,
        ),
        "dev_raw_logits": dev_raw_logits,
        "dev_raw_probabilities": dev_raw_probabilities,
        "dev_calibrated_logits": dev_calibrated_logits,
        "dev_calibrated_probabilities": dev_calibrated_probabilities,
        "bootstrap_indices": frozen["bootstrap_indices"],
        "episode_derangements": frozen["episode_derangements"],
    }
    return {name: _canonical_array(value) for name, value in arrays.items()}


def publish_evidence_create_only(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    attempt_sha256: str,
) -> dict[str, object]:
    if set(arrays) != EVIDENCE_KEYS:
        raise ValueError("PB21L evidence key set drifted")
    canonical = {name: _canonical_array(value) for name, value in arrays.items()}
    manifest = _array_manifest(canonical)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("PB21L evidence exists; retry is forbidden")
    temporary = path.with_name(f".{path.name}.{attempt_sha256[:16]}.partial")
    if temporary.exists():
        raise FileExistsError("PB21L partial evidence exists; retry is forbidden")
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
        "array_manifest_sha256": _manifest_sha256(manifest),
        "published_create_only_atomically": True,
    }


def validate_publication_receipt(
    path: Path,
    publication: Mapping[str, object],
    arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    if (
        not path.is_file()
        or _sha256_file(path) != publication.get("sha256")
        or path.stat().st_size != publication.get("byte_length")
        or publication.get("allowed_key_set_sha256") != evidence_key_set_sha256()
    ):
        raise ValueError("PB21L evidence publication file envelope drifted")
    if arrays is None:
        return
    if set(arrays) != EVIDENCE_KEYS:
        raise ValueError("PB21L publication arrays do not have the exact key set")
    manifest = _array_manifest(arrays)
    if (
        publication.get("arrays") != manifest
        or publication.get("array_manifest_sha256") != _manifest_sha256(manifest)
    ):
        raise ValueError("PB21L evidence publication manifest envelope drifted")


def validate_evidence_axes(arrays: Mapping[str, np.ndarray]) -> None:
    if (
        _decode_strings(arrays["scorer_ids"]) != SCORERS
        or _decode_strings(arrays["cell_ids"]) != CELLS
        or not np.array_equal(
            arrays["cell_fit_index"], np.asarray((0, 0, 1, 1), dtype=np.int16)
        )
        or not np.array_equal(
            arrays["cell_init_seed"],
            np.asarray((51_042, 53_042, 51_042, 53_042), dtype=np.int32),
        )
    ):
        raise ValueError("PB21L evidence axes drifted")


def validate_source_lineage_arrays(
    arrays: Mapping[str, np.ndarray],
    expected: ExpectedSourceLineage,
) -> None:
    observed_fit = tuple(
        _decode_strings(arrays["fit_group_ids"][index])
        for index in range(len(FIT_COHORTS))
    )
    observed_cal = tuple(
        _decode_strings(arrays["cal_group_ids"][index])
        for index in range(len(CAL_COHORTS))
    )
    fit_digests = tuple(strict._ordered_sequence_digest(row) for row in observed_fit)
    cal_digests = tuple(strict._ordered_sequence_digest(row) for row in observed_cal)
    if (
        observed_fit != expected.fit_group_ids
        or observed_cal != expected.cal_group_ids
        or fit_digests != expected.fit_group_ordered_sha256
        or cal_digests != expected.cal_group_ordered_sha256
        or _decode_strings(arrays["fit_group_ordered_sha256"]) != fit_digests
        or _decode_strings(arrays["cal_group_ordered_sha256"]) != cal_digests
        or _decode_strings(arrays["fit_nz_feature_sha256"])
        != expected.fit_nz_feature_sha256
    ):
        raise ValueError("PB21L ordered source lineage or FIT NZ features drifted")


def validate_factual_weight_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected_counts = np.stack(
        [
            factual_action_counts(arrays["fit_actions"][index])
            for index in arrays["cell_fit_index"]
        ]
    )
    expected_theoretical = np.stack(
        [
            factual_balance_weights(arrays["fit_actions"][index])
            for index in arrays["cell_fit_index"]
        ]
    )
    expected_consumed = np.ascontiguousarray(expected_theoretical, dtype=np.float32)
    if (
        not np.array_equal(arrays["factual_action_counts"], expected_counts)
        or not np.array_equal(arrays["factual_balance_weights"], expected_theoretical)
        or not np.array_equal(
            arrays["factual_balance_weights_consumed_float32"], expected_consumed
        )
    ):
        raise ValueError("PB21L factual count/theoretical/consumed weight evidence drifted")


def validate_aa_provenance_arrays(
    arrays: Mapping[str, np.ndarray],
    registered: Mapping[str, object],
    decoded_pruned: np.ndarray,
) -> None:
    shape = (len(SCORERS), len(CELLS))
    namespace = np.full(shape, str(registered["source_namespace"]), dtype=object)
    source_split = np.full(shape, str(registered["source_split"]), dtype=object)
    source_partition = np.full(shape, str(registered["source_partition"]), dtype=object)
    partition_algorithm = np.full(shape, str(registered["partition_algorithm"]), dtype=object)
    manifests = np.broadcast_to(
        np.asarray(registered["dataset_manifest_sha256_by_cell"], dtype=object), shape
    )
    source_bundles = np.asarray(
        registered["source_bundle_sha256_by_scorer_and_cell"], dtype=object
    )
    observed = {
        "namespace": np.char.decode(arrays["calibration_source_namespace"], "utf-8"),
        "split": np.char.decode(arrays["calibration_source_split"], "utf-8"),
        "partition": np.char.decode(arrays["calibration_source_partition"], "utf-8"),
        "algorithm": np.char.decode(arrays["calibration_partition_algorithm"], "utf-8"),
        "manifest": np.char.decode(
            arrays["calibration_dataset_manifest_sha256"], "ascii"
        ),
        "source_bundle": np.char.decode(
            arrays["calibration_source_bundle_sha256"], "ascii"
        ),
        "checkpoint": np.char.decode(
            arrays["calibration_upstream_checkpoint_sha256"], "ascii"
        ),
    }
    if (
        not np.array_equal(observed["namespace"], namespace)
        or not np.array_equal(observed["split"], source_split)
        or not np.array_equal(observed["partition"], source_partition)
        or not np.array_equal(observed["algorithm"], partition_algorithm)
        or not np.array_equal(observed["manifest"], manifests)
        or not np.array_equal(observed["source_bundle"], source_bundles)
        or not np.array_equal(observed["checkpoint"], decoded_pruned)
    ):
        raise ValueError("PB21L AA provenance differs from registered/frozen lineage")


def validate_frozen_shared_arrays(
    arrays: Mapping[str, np.ndarray],
    frozen: Mapping[str, np.ndarray],
) -> None:
    exact_cross_links = {
        "fit_targets": frozen["fit_targets"][PB21K_FIT_INDICES],
        "fit_actions": frozen["fit_actions"][PB21K_FIT_INDICES],
        "fit_root_ids": frozen["fit_root_ids"][PB21K_FIT_INDICES],
        "cal_targets": frozen["cal_targets"][PB21K_CAL_INDICES],
        "cal_actions": frozen["cal_actions"][PB21K_CAL_INDICES],
        "cal_root_ids": frozen["cal_root_ids"][PB21K_CAL_INDICES],
        "dev_targets": frozen["dev_targets"],
        "dev_actions": frozen["dev_actions"],
        "dev_root_ids": frozen["dev_root_ids"],
        "dev_cluster_ordinal": frozen["dev_cluster_ordinal"],
        "dev_cluster_ids": frozen["dev_cluster_ids"],
        "bootstrap_indices": frozen["bootstrap_indices"],
        "episode_derangements": frozen["episode_derangements"],
    }
    for name, expected in exact_cross_links.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"PB21L did not preserve frozen PB21K array: {name}")
    if (
        _array_sha256(arrays["bootstrap_indices"]) != EXACT_PB21K_BOOTSTRAP_SHA256
        or _array_sha256(arrays["episode_derangements"])
        != EXACT_PB21K_DERANGEMENT_SHA256
    ):
        raise ValueError("PB21L inference arrays are valid-shaped but not frozen PB21K arrays")


def validate_probability_transforms(arrays: Mapping[str, np.ndarray]) -> None:
    if not np.array_equal(
        arrays["dev_raw_probabilities"], pb21k._sigmoid(arrays["dev_raw_logits"])
    ):
        raise ValueError("raw probabilities do not match logits")
    expected_calibrated = (
        arrays["dev_raw_logits"] * arrays["aa_scales"][:, :, None, :]
        + arrays["aa_biases"][:, :, None, :]
    )
    if not np.array_equal(arrays["dev_calibrated_logits"], expected_calibrated):
        raise ValueError("AA calibrated logits drifted")
    if not np.array_equal(
        arrays["dev_calibrated_probabilities"], pb21k._sigmoid(expected_calibrated)
    ):
        raise ValueError("AA calibrated probabilities drifted")


def validate_base_array_replay(
    arrays: Mapping[str, np.ndarray],
    frozen: Mapping[str, np.ndarray],
) -> None:
    base = SCORERS.index(SCORER_BASE)
    cells = PB21K_CELL_INDICES
    if (
        not np.array_equal(arrays["cal_raw_logits"][base], frozen["cal_nz_raw_logits"][cells])
        or not np.array_equal(
            arrays["aa_scales"][base],
            frozen["calibrator_scales"][PB21K_AA_INDEX, cells],
        )
        or not np.array_equal(
            arrays["aa_biases"][base],
            frozen["calibrator_biases"][PB21K_AA_INDEX, cells],
        )
        or not np.array_equal(
            arrays["aa_accepted"][base],
            frozen["calibrator_accepted"][PB21K_AA_INDEX, cells],
        )
        or not np.array_equal(
            arrays["dev_raw_logits"][base],
            frozen["dev_raw_logits"][cells, PB21K_NZ_INDEX],
        )
        or not np.array_equal(
            arrays["dev_raw_probabilities"][base],
            frozen["dev_raw_probabilities"][cells, PB21K_NZ_INDEX],
        )
        or not np.array_equal(
            arrays["dev_calibrated_logits"][base],
            frozen["nz_calibrated_logits"][PB21K_AA_INDEX, cells],
        )
        or not np.array_equal(
            arrays["dev_calibrated_probabilities"][base],
            frozen["nz_calibrated_probabilities"][PB21K_AA_INDEX, cells],
        )
    ):
        raise ValueError("BASE did not byte-replay frozen PB21K NZ+AA evidence")


def reload_and_validate_evidence(
    path: Path,
    publication: Mapping[str, object],
    *,
    frozen: Mapping[str, np.ndarray],
    pb21k_result: Mapping[str, object],
    expected_source_lineage: ExpectedSourceLineage,
    registered_aa_provenance: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    validate_publication_receipt(path, publication)
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EVIDENCE_KEYS:
            raise ValueError("PB21L evidence keys drifted")
        for name in archive.files:
            value = _canonical_array(archive[name])
            expected = publication["arrays"][name]
            if (
                value.dtype.str != expected["dtype"]
                or list(value.shape) != expected["shape"]
                or _array_sha256(value) != expected["sha256"]
            ):
                raise ValueError(f"PB21L evidence array drifted: {name}")
            arrays[name] = value
    validate_publication_receipt(path, publication, arrays)
    contract = expected_evidence_contract()
    for name, expected in contract.items():
        if (
            arrays[name].dtype.str != expected["dtype"]
            or list(arrays[name].shape) != expected["shape"]
        ):
            raise ValueError(f"PB21L evidence schema drifted: {name}")
    validate_evidence_axes(arrays)
    if not np.array_equal(
        arrays["pruning_active_columns"],
        np.asarray(pb21j.ACTIVE_COLUMNS[ARM_NZ], dtype=np.int16),
    ):
        raise ValueError("PB21L NZ pruning columns drifted")
    if not np.all(
        arrays["pruning_parameter_count"] == pb21j.PRUNED_PARAMETER_COUNTS[ARM_NZ]
    ) or np.any(
        arrays["pruning_maximum_absolute_difference"]
        > pb21j.PRUNING_EQUIVALENCE_MAX_ABS
    ):
        raise ValueError("PB21L pruning size/equivalence contract drifted")
    validate_frozen_shared_arrays(arrays, frozen)
    validate_source_lineage_arrays(arrays, expected_source_lineage)
    validate_factual_weight_arrays(arrays)
    validate_probability_transforms(arrays)
    validate_base_array_replay(arrays, frozen)
    base = SCORERS.index(SCORER_BASE)
    decoded_initial = np.char.decode(arrays["training_initial_state_sha256"], "ascii")
    decoded_final = np.char.decode(arrays["training_final_state_sha256"], "ascii")
    decoded_pruned = np.char.decode(arrays["pruned_state_sha256"], "ascii")
    decoded_pruning_maps = np.char.decode(arrays["pruning_mapping_sha256"], "ascii")
    decoded_permutations = np.char.decode(
        arrays["training_pass_permutation_sha256"], "ascii"
    )
    decoded_cal_raw = np.char.decode(arrays["calibration_raw_logit_sha256"], "ascii")
    validate_aa_provenance_arrays(arrays, registered_aa_provenance, decoded_pruned)
    base_index = SCORERS.index(SCORER_BASE)
    for cell_index, cell in enumerate(CELLS):
        frozen_training = pb21k_result["training"][cell][ARM_NZ]
        frozen_pruning = pb21k_result["pruning"][cell][ARM_NZ]
        frozen_calibration = pb21k_result["calibration"][cell]
        if (
            decoded_initial[base_index, cell_index]
            != frozen_training["initial_state_sha256"]
            or decoded_final[base_index, cell_index]
            != frozen_training["final_state_sha256"]
            or decoded_final[base_index, cell_index] != BASE_FINAL_STATE_SHA256[cell]
            or decoded_pruned[base_index, cell_index]
            != frozen_pruning["mapping"]["pruned_state_sha256"]
            or decoded_cal_raw[base_index, cell_index]
            != frozen_calibration["shared_raw_CAL_logit_sha256"]
        ):
            raise ValueError(f"BASE state/calibration identity drifted for {cell}")
        for pass_index, record in enumerate(frozen_training["pass_records"]):
            if (
                arrays["training_pass_number"][base_index, cell_index, pass_index]
                != record["pass"]
                or arrays["training_pass_mean_all_BCE"][base_index, cell_index, pass_index]
                != record["mean_all_action_BCE"]
                or arrays["training_pass_mean_objective"]
                [base_index, cell_index, pass_index]
                != record["mean_all_action_BCE"]
                or arrays["training_pass_maximum_preclip_gradient_norm"]
                [base_index, cell_index, pass_index]
                != record["maximum_preclip_gradient_norm"]
                or decoded_permutations[base_index, cell_index, pass_index]
                != record["permutation_sha256"]
            ):
                raise ValueError(f"BASE pass-record evidence drifted for {cell}")
        if (
            decoded_pruned[base_index, cell_index]
            != BASE_PRUNED_STATE_SHA256[cell]
            or arrays["pruning_padded_logit_sha256"][base_index, cell_index]
            != frozen_pruning["FIT_inference_transfer"]["padded_logit_sha256"].encode()
            or arrays["pruning_pruned_logit_sha256"][base_index, cell_index]
            != frozen_pruning["FIT_inference_transfer"]["pruned_logit_sha256"].encode()
            or arrays["pruning_maximum_absolute_difference"][base_index, cell_index]
            != frozen_pruning["FIT_inference_transfer"]["maximum_absolute_difference"]
        ):
            raise ValueError(f"BASE pruning transfer evidence drifted for {cell}")
    expected_pass_numbers = np.broadcast_to(
        np.arange(1, PASSES + 1, dtype=np.int16),
        arrays["training_pass_number"].shape,
    )
    if not np.array_equal(arrays["training_pass_number"], expected_pass_numbers):
        raise ValueError("PB21L pass-number sequence drifted")
    if not np.all(decoded_initial == decoded_initial[base_index][None, :]):
        raise ValueError("paired scorer initial states were not byte identical")
    if not np.all(decoded_permutations == decoded_permutations[base_index, 0][None, None, :]):
        raise ValueError("paired scorer permutation hashes drifted")
    for scorer_index in range(len(SCORERS)):
        for cell_index in range(len(CELLS)):
            mapping = {
                "recipe": "masked_superset_train_pruned_deployment_v1",
                "active_columns": arrays["pruning_active_columns"].astype(int).tolist(),
                "input_width": len(arrays["pruning_active_columns"]),
                "parameter_count": int(
                    arrays["pruning_parameter_count"][scorer_index, cell_index]
                ),
                "mapped_tensors_byte_exact": True,
                "pruned_state_sha256": decoded_pruned[scorer_index, cell_index],
            }
            if (
                decoded_pruning_maps[scorer_index, cell_index]
                != _pruning_mapping_sha256(mapping)
            ):
                raise ValueError("PB21L pruning-map digest drifted")
    telemetry_keys = (
        "factual_balance_weights",
        "training_pass_mean_all_BCE",
        "training_pass_mean_factual_component_BCE",
        "training_pass_mean_objective",
        "training_pass_maximum_preclip_gradient_norm",
        "pruning_maximum_absolute_difference",
    )
    if any(not np.isfinite(arrays[name]).all() for name in telemetry_keys):
        raise ValueError("PB21L training/pruning telemetry is non-finite")
    if bool(arrays["training_factual_components_recorded"][base_index]):
        raise ValueError("BASE may not invent factual-component pass telemetry")
    if not np.all(arrays["training_factual_components_recorded"][1:]):
        raise ValueError("UPMIX pass telemetry must retain factual components")
    if not np.array_equal(
        np.char.decode(arrays["calibration_upstream_checkpoint_sha256"], "ascii"),
        decoded_pruned,
    ):
        raise ValueError("AA calibration checkpoint lineage drifted")
    for scorer_index in range(len(SCORERS)):
        for cell_index in range(len(CELLS)):
            if decoded_cal_raw[scorer_index, cell_index] != _array_sha256(
                arrays["cal_raw_logits"][scorer_index, cell_index]
            ):
                raise ValueError("AA raw-CAL digest cross-link drifted")
            cal_index = int(arrays["cell_fit_index"][cell_index])
            raw = arrays["cal_raw_logits"][scorer_index, cell_index]
            targets = arrays["cal_targets"][cal_index]
            action_ids = np.broadcast_to(
                np.arange(ACTION_COUNT, dtype=np.int64), raw.shape
            )
            provenance = pb21k.TrainCalibrationProvenance(
                source_namespace=str(registered_aa_provenance["source_namespace"]),
                source_split=str(registered_aa_provenance["source_split"]),
                source_partition=str(registered_aa_provenance["source_partition"]),
                calibration_group_ids=tuple(
                    group
                    for group in expected_source_lineage.cal_group_ids[cal_index]
                    for _ in range(ACTION_COUNT)
                ),
                upstream_model_fit_group_ids=tuple(
                    sorted(set(expected_source_lineage.fit_group_ids[cal_index]))
                ),
                upstream_checkpoint_sha256=decoded_pruned[scorer_index, cell_index],
                dataset_manifest_sha256=str(
                    registered_aa_provenance["dataset_manifest_sha256_by_cell"]
                    [cell_index]
                ),
                source_bundle_sha256=str(
                    registered_aa_provenance["source_bundle_sha256_by_scorer_and_cell"]
                    [scorer_index][cell_index]
                ),
                partition_algorithm=str(registered_aa_provenance["partition_algorithm"]),
            )
            if (
                provenance.calibration_group_digest
                != np.char.decode(arrays["calibration_group_sha256"], "ascii")
                [scorer_index, cell_index]
                or provenance.upstream_model_fit_group_digest
                != np.char.decode(arrays["calibration_fit_group_sha256"], "ascii")
                [scorer_index, cell_index]
            ):
                raise ValueError("AA calibration group provenance digest drifted")
            refit = pb21k.fit_train_only_per_action_affine(
                torch.from_numpy(raw.reshape(-1)),
                torch.from_numpy(targets.reshape(-1)),
                torch.from_numpy(action_ids.reshape(-1)),
                provenance=provenance,
                action_count=ACTION_COUNT,
                l2_regularization=pb21k.CALIBRATION_L2,
                minimum_scale=pb21k.CALIBRATION_MINIMUM_SCALE,
                minimum_examples_per_action=pb21k.CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
                minimum_class_examples=pb21k.CALIBRATION_MINIMUM_CLASS_EXAMPLES,
                max_iterations=pb21k.CALIBRATION_MAX_ITERATIONS,
                tolerance=pb21k.CALIBRATION_TOLERANCE,
                fit_mode="per_action_affine",
            )
            if (
                not np.array_equal(np.asarray(refit.scales),
                                   arrays["aa_scales"][scorer_index, cell_index])
                or not np.array_equal(np.asarray(refit.biases),
                                      arrays["aa_biases"][scorer_index, cell_index])
                or bool(refit.accepted)
                != bool(arrays["aa_accepted"][scorer_index, cell_index])
            ):
                raise ValueError("deterministic AA refit did not reproduce evidence")
    return arrays, {
        "passed": True,
        "allow_pickle": False,
        "exact_allowed_keys": True,
        "per_array_manifest_verified": True,
        "publication_byte_length_key_set_and_manifest_verified": True,
        "exact_cell_fit_and_initialization_axes_verified": True,
        "all_consumed_source_arrays_cross_linked": True,
        "ordered_FIT_CAL_group_lineage_verified": True,
        "FIT_NZ_feature_digests_verified": True,
        "theoretical_float64_and_consumed_float32_BAL_weights_verified": True,
        "BASE_byte_replayed_PB21K_NZ_AA_arrays": True,
        "BASE_training_pruning_and_pass_records_verified": True,
        "AA_lineage_cross_links_and_deterministic_refits_verified": True,
        "AA_registered_static_provenance_verified": True,
        "PB21K_bootstrap_and_derangements_byte_reused": True,
    }


def _loss_elements(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(targets, dtype=np.float64)
    probability = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12
    )
    if target.shape != probability.shape or target.ndim != 2 or target.shape[1] != ACTION_COUNT:
        raise ValueError("proper-score tables must be aligned Dx5 arrays")
    bce = -(target * np.log(probability) + (1.0 - target) * np.log1p(-probability))
    brier = np.square(target - probability)
    return bce, brier


def _domain_mask(actions: np.ndarray, domain: str) -> np.ndarray:
    factual = np.asarray(actions, dtype=np.int64)
    if factual.ndim != 1 or np.any((factual < 0) | (factual >= ACTION_COUNT)):
        raise ValueError("domain actions must be one-dimensional semantic IDs")
    ids = np.arange(ACTION_COUNT, dtype=np.int64)[None, :]
    if domain == "all_action":
        return np.ones((len(factual), ACTION_COUNT), dtype=np.bool_)
    if domain == "factual":
        return ids == factual[:, None]
    if domain == "nonselected_complement":
        return ids != factual[:, None]
    raise ValueError(f"unknown evaluation domain: {domain}")


def mask_partition_identity(
    targets: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
) -> dict[str, object]:
    bce, brier = _loss_elements(targets, probabilities)
    factual = _domain_mask(actions, "factual")
    complement = _domain_mask(actions, "nonselected_complement")
    all_bce = bce.mean(axis=1)
    all_brier = brier.mean(axis=1)
    factual_bce = bce[factual].reshape(len(actions))
    factual_brier = brier[factual].reshape(len(actions))
    complement_bce = bce[complement].reshape(len(actions), ACTION_COUNT - 1).mean(axis=1)
    complement_brier = (
        brier[complement].reshape(len(actions), ACTION_COUNT - 1).mean(axis=1)
    )
    bce_error = float(
        np.max(np.abs(ACTION_COUNT * all_bce - factual_bce - 4.0 * complement_bce))
    )
    brier_error = float(
        np.max(
            np.abs(ACTION_COUNT * all_brier - factual_brier - 4.0 * complement_brier)
        )
    )
    passed = bce_error <= 1.0e-12 and brier_error <= 1.0e-12
    if not passed:
        raise RuntimeError("5*all=factual+4*complement additive-loss identity failed")
    return {
        "identity": "5*all=factual+4*nonselected_complement",
        "maximum_BCE_absolute_error": bce_error,
        "maximum_Brier_absolute_error": brier_error,
        "passed": True,
    }


def _fit_domain_priors(
    fit_targets: np.ndarray,
    fit_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    targets = np.asarray(fit_targets, dtype=np.float64)
    actions = np.asarray(fit_actions, dtype=np.int64)
    if targets.shape != (ROOTS_PER_FIT, ACTION_COUNT) or actions.shape != (
        ROOTS_PER_FIT,
    ):
        raise ValueError("FIT prior geometry drifted")
    priors: dict[str, np.ndarray] = {}
    priors["all_action"] = targets.mean(axis=0)
    factual = np.empty(ACTION_COUNT, dtype=np.float64)
    complement = np.empty(ACTION_COUNT, dtype=np.float64)
    for action in range(ACTION_COUNT):
        selected = actions == action
        nonselected = ~selected
        if not selected.any() or not nonselected.any():
            raise ValueError("FIT action support is incomplete")
        factual[action] = targets[selected, action].mean()
        complement[action] = targets[nonselected, action].mean()
    priors["factual"] = factual
    priors["nonselected_complement"] = complement
    for values in priors.values():
        if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
            raise ValueError("FIT priors must be finite and nondegenerate")
    return priors


def _per_root_domain_losses(
    targets: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    domain: str,
) -> np.ndarray:
    bce, brier = _loss_elements(targets, probabilities)
    mask = _domain_mask(actions, domain)
    count = mask.sum(axis=1)
    if np.any(count == 0):
        raise ValueError("domain mask left an empty root")
    return np.stack(
        ((bce * mask).sum(axis=1) / count, (brier * mask).sum(axis=1) / count),
        axis=1,
    )


def _episode_means(values: np.ndarray, cluster_ordinal: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    ordinal = np.asarray(cluster_ordinal, dtype=np.int64)
    if data.shape[0] != len(ordinal):
        raise ValueError("episode values and cluster ordinals must align")
    groups = int(ordinal.max()) + 1
    result = np.asarray([data[ordinal == group].mean(axis=0) for group in range(groups)])
    if groups != DEV_EPISODES or any(int((ordinal == group).sum()) != ROOTS_PER_EPISODE
                                     for group in range(groups)):
        raise ValueError("DEV cluster geometry drifted")
    return result


def _bootstrap_means(episode_values: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    return pb21k._bootstrap_means(episode_values, sampled)


def proper_score_report(
    targets: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    cluster_ordinal: np.ndarray,
    sampled: np.ndarray,
    *,
    domain: str,
    priors: np.ndarray,
) -> dict[str, object]:
    baseline = np.broadcast_to(np.asarray(priors, dtype=np.float64), targets.shape)
    model_loss = _per_root_domain_losses(targets, probabilities, actions, domain)
    baseline_loss = _per_root_domain_losses(targets, baseline, actions, domain)
    episode = _episode_means(baseline_loss - model_loss, cluster_ordinal)
    draws = _bootstrap_means(episode, sampled)
    return {
        "domain": domain,
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


def _domain_metric_report(
    targets: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    *,
    domain: str,
    priors: np.ndarray,
) -> dict[str, object]:
    mask = _domain_mask(actions, domain)
    aggregate = binary_probability_metrics(
        targets[mask], probabilities[mask], ece_bins=pb21k.ECE_BINS
    ).as_dict()
    per_action: list[dict[str, object]] = []
    per_action_bias_limit = (
        MAXIMUM_FACTUAL_PER_ACTION_BIAS
        if domain == "factual"
        else MAXIMUM_PER_ACTION_BIAS
    )
    for action in range(ACTION_COUNT):
        rows = mask[:, action]
        action_targets = targets[rows, action]
        action_probability = probabilities[rows, action]
        metrics = binary_probability_metrics(
            action_targets, action_probability, ece_bins=pb21k.ECE_BINS
        ).as_dict()
        baseline_probability = np.full(len(action_targets), priors[action], dtype=np.float64)
        baseline = binary_probability_metrics(
            action_targets, baseline_probability, ece_bins=pb21k.ECE_BINS
        ).as_dict()
        positives = int(action_targets.sum())
        negatives = len(action_targets) - positives
        checks = {
            "ECE": float(metrics["ece_equal_mass"]) <= MAXIMUM_PER_ACTION_ECE,
            "absolute_bias": abs(float(metrics["calibration_bias"]))
            <= per_action_bias_limit,
        }
        if domain == "factual":
            checks["both_classes"] = positives > 0 and negatives > 0
        elif domain == "all_action":
            checks.update(
                {
                    "PR_prevalence_gain": float(metrics["pr_auc"])
                    >= float(metrics["prevalence"]) + MINIMUM_PR_PREVALENCE_GAIN,
                    "Brier_skill_nonnegative": float(metrics["brier"])
                    <= float(baseline["brier"]),
                }
            )
        else:
            checks.update(
                {
                    "both_classes": positives > 0 and negatives > 0,
                    "PR_prevalence_gain": float(metrics["pr_auc"])
                    >= float(metrics["prevalence"]) + MINIMUM_PR_PREVALENCE_GAIN,
                    "Brier_skill_nonnegative": float(metrics["brier"])
                    <= float(baseline["brier"]),
                }
            )
        per_action.append(
            {
                "action_id": action,
                "observations": len(action_targets),
                "positives": positives,
                "negatives": negatives,
                "metrics": metrics,
                "baseline_metrics": baseline,
                "checks": checks,
                "maximum_absolute_bias": per_action_bias_limit,
                "passed": all(checks.values()),
            }
        )
    aggregate_checks = {
        "ECE": float(aggregate["ece_equal_mass"]) <= MAXIMUM_AGGREGATE_ECE,
        "absolute_bias": abs(float(aggregate["calibration_bias"]))
        <= MAXIMUM_AGGREGATE_ECE,
    }
    return {
        "domain": domain,
        "aggregate": aggregate,
        "aggregate_checks": aggregate_checks,
        "per_action": per_action,
        "passed": all(aggregate_checks.values())
        and all(value["passed"] for value in per_action),
    }


def causal_derangement_report(
    targets: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_probabilities: np.ndarray,
    actions: np.ndarray,
    cluster_ordinal: np.ndarray,
    episode_derangements: np.ndarray,
    *,
    domain: str,
) -> dict[str, object]:
    root_donors = pb21k._root_derangements(episode_derangements, cluster_ordinal)
    mask = _domain_mask(actions, domain)
    real_bce = pb21k._binary_bce(targets[mask], calibrated_probabilities[mask])
    real_auc = pb21j._score_auc(targets[mask], raw_scores[mask])
    real_action_auc = [
        pb21j._score_auc(targets[mask[:, action], action],
                         raw_scores[mask[:, action], action])
        for action in range(ACTION_COUNT)
    ]
    shuffled_bce: list[float] = []
    shuffled_auc: list[float] = []
    shuffled_action_auc: list[list[float]] = []
    for donor in root_donors:
        probability = calibrated_probabilities[donor]
        scores = raw_scores[donor]
        shuffled_bce.append(pb21k._binary_bce(targets[mask], probability[mask]))
        shuffled_auc.append(pb21j._score_auc(targets[mask], scores[mask]))
        shuffled_action_auc.append(
            [
                pb21j._score_auc(
                    targets[mask[:, action], action], scores[mask[:, action], action]
                )
                for action in range(ACTION_COUNT)
            ]
        )
    median_bce = float(np.median(shuffled_bce))
    median_auc = float(np.median(shuffled_auc))
    per_action_drop = [
        real_action_auc[action]
        - float(np.median([values[action] for values in shuffled_action_auc]))
        for action in range(ACTION_COUNT)
    ]
    checks = {
        "calibrated_shuffled_BCE_ratio": median_bce / real_bce
        >= pb21j.MINIMUM_SHUFFLED_BCE_RATIO,
        "raw_aggregate_AUC_drop": real_auc - median_auc
        >= pb21j.MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP,
        "every_action_raw_AUC_drop": all(
            value >= pb21j.MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP
            for value in per_action_drop
        ),
    }
    return {
        "domain": domain,
        "algorithm": "whole_episode_row_derangement_restricted_to_fixed_domain_v1",
        "episode_donor_index_sha256": _array_sha256(episode_derangements),
        "root_donor_index_sha256": _array_sha256(root_donors),
        "real_calibrated_BCE": real_bce,
        "median_shuffled_calibrated_BCE": median_bce,
        "calibrated_shuffled_BCE_ratio": median_bce / real_bce,
        "real_raw_AUC": real_auc,
        "median_shuffled_raw_AUC": median_auc,
        "raw_aggregate_AUC_drop": real_auc - median_auc,
        "per_action_raw_AUC_drop": per_action_drop,
        "checks": checks,
        "passed": all(checks.values()),
    }


def absolute_cell_gate(
    *,
    fit_targets: np.ndarray,
    fit_actions: np.ndarray,
    dev_targets: np.ndarray,
    dev_actions: np.ndarray,
    cluster_ordinal: np.ndarray,
    raw_logits: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_logits: np.ndarray,
    calibrated_probabilities: np.ndarray,
    sampled: np.ndarray,
    episode_derangements: np.ndarray,
    calibrator_accepted: bool,
) -> dict[str, object]:
    priors = _fit_domain_priors(fit_targets, fit_actions)
    identity = mask_partition_identity(dev_targets, calibrated_probabilities, dev_actions)
    domains: dict[str, object] = {}
    for domain in DOMAINS:
        metrics = _domain_metric_report(
            dev_targets,
            calibrated_probabilities,
            dev_actions,
            domain=domain,
            priors=priors[domain],
        )
        proper = proper_score_report(
            dev_targets,
            calibrated_probabilities,
            dev_actions,
            cluster_ordinal,
            sampled,
            domain=domain,
            priors=priors[domain],
        )
        causal = causal_derangement_report(
            dev_targets,
            raw_logits,
            calibrated_probabilities,
            dev_actions,
            cluster_ordinal,
            episode_derangements,
            domain=domain,
        )
        proper_pass = (
            proper["observed_BCE_improvement"] > 0.0
            and proper["BCE_one_sided_lower_bound"] > 0.0
            and proper["observed_Brier_improvement"] > 0.0
            and proper["Brier_one_sided_lower_bound"] > 0.0
        )
        domains[domain] = {
            "metrics": metrics,
            "proper_score": proper,
            "causal_derangement": causal,
            "passed": metrics["passed"] is True
            and proper_pass
            and causal["passed"] is True,
        }
    raw_all = binary_probability_metrics(
        dev_targets.reshape(-1), raw_probabilities.reshape(-1), ece_bins=pb21k.ECE_BINS
    ).as_dict()
    calibrated_all = domains["all_action"]["metrics"]["aggregate"]
    all_prior_table = np.broadcast_to(priors["all_action"], dev_targets.shape)
    prior_all = binary_probability_metrics(
        dev_targets.reshape(-1), all_prior_table.reshape(-1), ece_bins=pb21k.ECE_BINS
    ).as_dict()
    raw_ranking = pb21j.score_ranking_gate(
        dev_targets,
        raw_logits,
        prior_auc=float(prior_all["roc_auc"]),
    )
    calibrated_ranking = pb21j.score_ranking_gate(
        dev_targets,
        calibrated_logits,
        prior_auc=float(prior_all["roc_auc"]),
    )
    global_checks = {
        "AA_accepted": bool(calibrator_accepted),
        "calibrated_BCE_not_worse_than_raw": float(calibrated_all["bce"])
        <= float(raw_all["bce"]),
        "all_action_BCE_prior_ratio": float(calibrated_all["bce"])
        <= pb21j.MAX_BASELINE_BCE_RATIO * float(prior_all["bce"]),
        "all_action_Brier_prior_ratio": float(calibrated_all["brier"])
        <= pb21j.MAX_BASELINE_BRIER_RATIO * float(prior_all["brier"]),
        "raw_ranking": raw_ranking["passed"] is True,
        "calibrated_ranking": calibrated_ranking["passed"] is True,
        "mask_partition_identity": identity["passed"] is True,
        "all_domains": all(value["passed"] is True for value in domains.values()),
    }
    payload = {
        "global_checks": global_checks,
        "raw_all_action": raw_all,
        "calibrated_all_action": calibrated_all,
        "FIT_all_action_prior": prior_all,
        "raw_ranking": raw_ranking,
        "calibrated_ranking": calibrated_ranking,
        "mask_partition_identity": identity,
        "domains": domains,
    }
    global_checks["all_metrics_finite"] = pb21j._all_numeric_finite(payload)
    return {**payload, "passed": all(global_checks.values())}


def direct_comparison_report(
    arrays: Mapping[str, np.ndarray],
    *,
    candidate: str,
) -> dict[str, object]:
    candidate_index = SCORERS.index(candidate)
    base_index = SCORERS.index(SCORER_BASE)
    targets = arrays["dev_targets"]
    actions = arrays["dev_actions"]
    ordinal = arrays["dev_cluster_ordinal"]
    sampled = arrays["bootstrap_indices"]
    comparisons: dict[str, object] = {}
    passed = True
    for domain in DOMAINS:
        cell_episode: list[np.ndarray] = []
        cell_points: dict[str, dict[str, float]] = {}
        for cell_index, cell in enumerate(CELLS):
            base_loss = _per_root_domain_losses(
                targets,
                arrays["dev_calibrated_probabilities"][base_index, cell_index],
                actions,
                domain,
            )
            candidate_loss = _per_root_domain_losses(
                targets,
                arrays["dev_calibrated_probabilities"][candidate_index, cell_index],
                actions,
                domain,
            )
            episode = _episode_means(base_loss - candidate_loss, ordinal)
            cell_episode.append(episode)
            cell_points[cell] = {
                "BCE": float(episode[:, 0].mean()),
                "Brier": float(episode[:, 1].mean()),
            }
        grand_episode = np.mean(np.stack(cell_episode), axis=0)
        draws = _bootstrap_means(grand_episode, sampled)
        lower = {
            "BCE": float(
                np.quantile(draws[:, 0], ONE_SIDED_ALPHA, method=QUANTILE_METHOD)
            ),
            "Brier": float(
                np.quantile(draws[:, 1], ONE_SIDED_ALPHA, method=QUANTILE_METHOD)
            ),
        }
        thresholds = {
            "BCE": 0.0 if domain == "factual" else -AA_BCE_MARGIN,
            "Brier": 0.0 if domain == "factual" else -AA_BRIER_MARGIN,
        }
        checks = {
            metric: all(values[metric] > thresholds[metric] for values in cell_points.values())
            and lower[metric] > thresholds[metric]
            for metric in ("BCE", "Brier")
        }
        comparisons[domain] = {
            "direction": f"{SCORER_BASE}_minus_{candidate}",
            "thresholds": thresholds,
            "cell_point_deltas": cell_points,
            "grand_mean_delta": {
                "BCE": float(grand_episode[:, 0].mean()),
                "Brier": float(grand_episode[:, 1].mean()),
            },
            "one_sided_lower_bounds": lower,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "four_cells_averaged_within_episode_before_resampling": True,
            "shared_sample_matrix_sha256": _array_sha256(sampled),
            "checks": checks,
            "passed": all(checks.values()),
        }
        passed = passed and all(checks.values())
    return {
        "candidate": candidate,
        "comparisons": comparisons,
        "passed": passed,
    }


def select_fresh_license(path_passed: Mapping[str, bool]) -> str | None:
    if set(path_passed) != {SCORER_POOL, SCORER_BAL}:
        raise ValueError("fresh-license decision requires exactly both UPMIX paths")
    if path_passed[SCORER_BAL] is True:
        return SCORER_BAL
    if path_passed[SCORER_POOL] is True:
        return SCORER_POOL
    return None


def evaluate_authoritative_evidence(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    absolute: dict[str, dict[str, object]] = {name: {} for name in SCORERS}
    for scorer_index, scorer in enumerate(SCORERS):
        for cell_index, cell in enumerate(CELLS):
            fit_index = int(arrays["cell_fit_index"][cell_index])
            absolute[scorer][cell] = absolute_cell_gate(
                fit_targets=arrays["fit_targets"][fit_index],
                fit_actions=arrays["fit_actions"][fit_index],
                dev_targets=arrays["dev_targets"],
                dev_actions=arrays["dev_actions"],
                cluster_ordinal=arrays["dev_cluster_ordinal"],
                raw_logits=arrays["dev_raw_logits"][scorer_index, cell_index],
                raw_probabilities=arrays["dev_raw_probabilities"][scorer_index, cell_index],
                calibrated_logits=arrays["dev_calibrated_logits"][scorer_index, cell_index],
                calibrated_probabilities=arrays["dev_calibrated_probabilities"]
                [scorer_index, cell_index],
                sampled=arrays["bootstrap_indices"],
                episode_derangements=arrays["episode_derangements"],
                calibrator_accepted=bool(arrays["aa_accepted"][scorer_index, cell_index]),
            )
    direct = {
        scorer: direct_comparison_report(arrays, candidate=scorer)
        for scorer in (SCORER_POOL, SCORER_BAL)
    }
    path_passed = {
        scorer: all(value["passed"] is True for value in absolute[scorer].values())
        and direct[scorer]["passed"] is True
        for scorer in (SCORER_POOL, SCORER_BAL)
    }
    licensed = select_fresh_license(path_passed)
    return {
        "absolute_gates": absolute,
        "paired_direct_inference": direct,
        "path_passed": path_passed,
        "selection": {
            "fresh_data_license": licensed,
            "precedence_applied": list(decision_contract()["precedence"]),
            "diagnosis": (
                f"{licensed}_licensed_for_one_fresh_preregistered_trial"
                if licensed is not None
                else "no_fresh_NZ_upmix_license"
            ),
            "license_is_nomination": False,
            "candidate_publication_allowed": False,
            "checkpoint_emitted": False,
            "qualification_claimed": False,
            "same_consumed_DEV_retry_allowed": False,
        },
    }


def _source_cross_link(
    tape: pb21j.FreshLiveTape,
    frozen_targets: np.ndarray,
    frozen_actions: np.ndarray,
    frozen_root_ids: np.ndarray,
    *,
    label: str,
    manifest_sha256: str,
) -> dict[str, object]:
    if (
        not np.array_equal(tape.hazard_targets, frozen_targets)
        or not np.array_equal(tape.factual_actions, frozen_actions)
        or _byte_strings(tape.root_state_ids, 64).shape != frozen_root_ids.shape
        or not np.array_equal(_byte_strings(tape.root_state_ids, 64), frozen_root_ids)
    ):
        raise RuntimeError(f"{label} live replay differs from frozen PB21K evidence")
    return {
        **pb21j.evidence_manifest(label, tape, manifest_sha256),
        "frozen_PB21K_target_action_root_byte_match": True,
    }


def _publish_attempt(
    project_root: Path,
    registration_record: Mapping[str, object],
) -> dict[str, object]:
    path = canonical_paths(project_root)["attempt"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "attempt_started": True,
        "retry_allowed": False,
        "registration_sha256": registration_record["sha256"],
        "PB21K_parent": registration_record["payload"]["PB21K_parent"],
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "checkpoint_emitted": False,
    }
    digest = development._publish_json_create_only(path, payload)
    return {"path": str(path), "sha256": digest, "payload": payload}


def _run_impl(
    *,
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
        raise RuntimeError("PB21L must hide CUDA")
    root = Path(__file__).resolve().parents[2]
    source_bundle = registration_record["payload"]["source_bundle"]
    if source_bundle != _diagnostic_source_bundle(root):
        raise RuntimeError("registered PB21L source bundle drifted")
    registered_aa_provenance = registration_record["payload"]["AA_provenance_contract"]
    if registered_aa_provenance != aa_provenance_contract(str(source_bundle["sha256"])):
        raise RuntimeError("registered PB21L AA provenance contract drifted")
    frozen, frozen_parent = load_frozen_pb21k_evidence(root)
    if frozen_parent != registration_record["payload"]["PB21K_parent"]:
        raise RuntimeError("PB21K parent drifted after attempt publication")
    pb21k_result = json.loads((root / PB21K_RESULT).read_text(encoding="utf-8"))
    model, _, upstream = preloaded_parent
    model.to(torch.device("cpu"))
    parent_before = development._state_dict_sha256(model.state_dict())
    manifests = pb21k.partition_manifests()

    fit_sources = pb21k.partition_sources(FIT_COHORTS)
    fit_tapes = {
        label: pb21k.collect_fresh_live_tape(
            model, fit_sources[label], partition_label=label
        )
        for label in FIT_COHORTS
    }
    source_evidence: dict[str, object] = {}
    for index, label in enumerate(FIT_COHORTS):
        source_evidence[label] = _source_cross_link(
            fit_tapes[label],
            frozen["fit_targets"][PB21K_FIT_INDICES[index]],
            frozen["fit_actions"][PB21K_FIT_INDICES[index]],
            frozen["fit_root_ids"][PB21K_FIT_INDICES[index]],
            label=label,
            manifest_sha256=manifests[label],
        )

    permutations = pb21k.deterministic_permutations()
    padded: list[list[pb21j.MaskedSupersetHazardHead]] = [
        [None] * len(CELLS) for _ in SCORERS  # type: ignore[list-item]
    ]
    training: dict[str, dict[str, object]] = {}
    features_by_fit = {
        label: pb21k.superset_features(fit_tapes[label], ARM_NZ)
        for label in FIT_COHORTS
    }
    for label in FIT_COHORTS:
        source_evidence[label]["NZ_feature_sha256"] = _array_sha256(
            features_by_fit[label]
        )
    for cell_index, cell in enumerate(CELLS):
        fit_label = cell.split("/", 1)[0]
        seed = int(cell.rsplit("-", 1)[1])
        training[cell] = {}
        initial_digests = set()
        for scorer_index, scorer in enumerate(SCORERS):
            head = initialized_head(seed)
            initial_digests.add(_state_sha256(head))
            record = train_upmix_head(
                head,
                features_by_fit[fit_label],
                fit_tapes[fit_label].hazard_targets,
                fit_tapes[fit_label].factual_actions,
                permutations,
                scorer=scorer,
            )
            if scorer == SCORER_BASE:
                parent_record = pb21k_result["training"][cell][ARM_NZ]
                comparable = {key: record[key] for key in parent_record}
                if (
                    comparable != parent_record
                    or record["final_state_sha256"] != BASE_FINAL_STATE_SHA256[cell]
                ):
                    raise RuntimeError(f"BASE training did not exactly replay PB21K cell {cell}")
                record["PB21K_pass_records_and_final_state_byte_replayed"] = True
            training[cell][scorer] = record
            padded[scorer_index][cell_index] = head
        if len(initial_digests) != 1:
            raise RuntimeError(f"paired scorer initialization drifted in {cell}")

    pruned: list[list[pb21j.PrunedHazardHead]] = [
        [None] * len(CELLS) for _ in SCORERS  # type: ignore[list-item]
    ]
    pruning: dict[str, dict[str, object]] = {}
    for cell_index, cell in enumerate(CELLS):
        fit_label = cell.split("/", 1)[0]
        pruning[cell] = {}
        for scorer_index, scorer in enumerate(SCORERS):
            compact, mapping = pb21j.prune_head(padded[scorer_index][cell_index], ARM_NZ)
            transfer = pb21j.pruning_equivalence_audit(
                padded[scorer_index][cell_index],
                compact,
                features_by_fit[fit_label],
                ARM_NZ,
            )
            if scorer == SCORER_BASE:
                expected = pb21k_result["pruning"][cell][ARM_NZ]
                if mapping != expected["mapping"] or transfer != expected["FIT_inference_transfer"]:
                    raise RuntimeError(f"BASE pruning/FIT transfer did not replay {cell}")
                if mapping["pruned_state_sha256"] != BASE_PRUNED_STATE_SHA256[cell]:
                    raise RuntimeError(f"BASE pruned state digest drifted in {cell}")
            pruning[cell][scorer] = {
                "mapping": mapping,
                "FIT_inference_transfer": transfer,
                "BASE_exact_PB21K_replay": scorer == SCORER_BASE,
            }
            pruned[scorer_index][cell_index] = compact
    del padded

    cal_sources = pb21k.partition_sources(CAL_COHORTS)
    cal_tapes = {
        label: pb21k.collect_fresh_live_tape(
            model, cal_sources[label], partition_label=label
        )
        for label in CAL_COHORTS
    }
    for index, label in enumerate(CAL_COHORTS):
        source_evidence[label] = _source_cross_link(
            cal_tapes[label],
            frozen["cal_targets"][PB21K_CAL_INDICES[index]],
            frozen["cal_actions"][PB21K_CAL_INDICES[index]],
            frozen["cal_root_ids"][PB21K_CAL_INDICES[index]],
            label=label,
            manifest_sha256=manifests[label],
        )
    expected_source_lineage = build_expected_source_lineage(
        fit_tapes=fit_tapes,
        cal_tapes=cal_tapes,
        fit_features=features_by_fit,
        source_evidence=source_evidence,
    )
    pair_map = dict(COHORT_PAIRS)
    calibrators: list[list[object]] = [[None] * len(CELLS) for _ in SCORERS]
    cal_raw = np.empty(
        (len(SCORERS), len(CELLS), ROOTS_PER_CAL, ACTION_COUNT), dtype=np.float64
    )
    calibration: dict[str, dict[str, object]] = {}
    for cell_index, cell in enumerate(CELLS):
        fit_label = cell.split("/", 1)[0]
        cal_label = pair_map[fit_label]
        features = pb21j.compact_features(
            pb21k.superset_features(cal_tapes[cal_label], ARM_NZ), ARM_NZ
        )
        calibration[cell] = {}
        for scorer_index, scorer in enumerate(SCORERS):
            raw = pb21j.raw_logit_table(
                pruned[scorer_index][cell_index], features, batch_size=1
            )
            provenance = pb21k.calibration_provenance(
                pruned[scorer_index][cell_index],
                cal_tapes[cal_label],
                fit_tapes[fit_label],
                cal_manifest_sha256=manifests[cal_label],
                source_bundle_sha256=(
                    EXACT_PB21K_SOURCE_BUNDLE_SHA256
                    if scorer == SCORER_BASE
                    else str(source_bundle["sha256"])
                ),
            )
            calibrator, report = pb21k.fit_aa_calibrator(
                raw, cal_tapes[cal_label], provenance
            )
            if scorer == SCORER_BASE:
                expected = pb21k_result["calibration"][cell]
                if (
                    _array_sha256(raw) != expected["shared_raw_CAL_logit_sha256"]
                    or report != expected["AA"]
                    or _state_sha256(pruned[scorer_index][cell_index])
                    != expected["shared_frozen_scorer_state_sha256"]
                ):
                    raise RuntimeError(f"BASE CAL raw/AA/provenance did not replay {cell}")
            calibrators[scorer_index][cell_index] = calibrator
            cal_raw[scorer_index, cell_index] = raw
            calibration[cell][scorer] = {
                "shared_raw_CAL_logit_sha256": _array_sha256(raw),
                "AA": report,
                "scorer_state_sha256": _state_sha256(pruned[scorer_index][cell_index]),
                "BASE_exact_PB21K_replay": scorer == SCORER_BASE,
            }

    if development._state_dict_sha256(model.state_dict()) != parent_before:
        raise RuntimeError("PB21L mutated parent before consumed DEV")
    dev_source = pb21k.partition_sources(("DEV",))["DEV"]
    dev_tape = pb21k.collect_fresh_live_tape(model, dev_source, partition_label="DEV")
    source_evidence["DEV"] = _source_cross_link(
        dev_tape,
        frozen["dev_targets"],
        frozen["dev_actions"],
        frozen["dev_root_ids"],
        label="DEV",
        manifest_sha256=manifests["DEV"],
    )
    dev_features = pb21j.compact_features(
        pb21k.superset_features(dev_tape, ARM_NZ), ARM_NZ
    )
    shape = (len(SCORERS), len(CELLS), DEV_ROOTS, ACTION_COUNT)
    dev_raw_logits = np.empty(shape, dtype=np.float64)
    dev_raw_probabilities = np.empty(shape, dtype=np.float64)
    dev_calibrated_logits = np.empty(shape, dtype=np.float64)
    dev_calibrated_probabilities = np.empty(shape, dtype=np.float64)
    for scorer_index in range(len(SCORERS)):
        for cell_index in range(len(CELLS)):
            raw = pb21j.raw_logit_table(
                pruned[scorer_index][cell_index], dev_features, batch_size=1
            )
            calibrated_logits, calibrated_probabilities = pb21k.calibrated_output_tables(
                calibrators[scorer_index][cell_index], raw
            )
            dev_raw_logits[scorer_index, cell_index] = raw
            dev_raw_probabilities[scorer_index, cell_index] = pb21k._sigmoid(raw)
            dev_calibrated_logits[scorer_index, cell_index] = calibrated_logits
            dev_calibrated_probabilities[scorer_index, cell_index] = calibrated_probabilities

    arrays = build_evidence_arrays(
        frozen=frozen,
        source_lineage=expected_source_lineage,
        training=training,
        pruning=pruning,
        calibration=calibration,
        cal_raw_logits=cal_raw,
        calibrators=calibrators,
        dev_raw_logits=dev_raw_logits,
        dev_raw_probabilities=dev_raw_probabilities,
        dev_calibrated_logits=dev_calibrated_logits,
        dev_calibrated_probabilities=dev_calibrated_probabilities,
    )
    publication = publish_evidence_create_only(
        evidence_path, arrays, attempt_sha256=str(attempt_record["sha256"])
    )
    del arrays
    authoritative, validation = reload_and_validate_evidence(
        evidence_path,
        publication,
        frozen=frozen,
        pb21k_result=pb21k_result,
        expected_source_lineage=expected_source_lineage,
        registered_aa_provenance=registered_aa_provenance,
    )
    numerical = evaluate_authoritative_evidence(authoritative)
    del authoritative

    parent_after = development._state_dict_sha256(model.state_dict())
    if parent_after != parent_before:
        raise RuntimeError("PB21L mutated the frozen parent")
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "candidate_checkpoint": {"published": False, "checkpoint_emitted": False},
        "nomination": None,
        "device": "cpu",
        "threads": 1,
        "cuda_visible_devices": "-1",
        "registration": {
            "path": registration_record["path"],
            "sha256": registration_record["sha256"],
            "verified_before_attempt": True,
        },
        "attempt": attempt_record,
        "PB21K_parent": frozen_parent,
        "upstream": upstream,
        "source_bundle": source_bundle,
        "objective_contract": objective_contract(),
        "schedule": schedule_record(),
        "inference_contract": inference_contract(),
        "decision_contract": decision_contract(),
        "source_evidence": source_evidence,
        "authoritative_evidence": {
            **publication,
            "validation": validation,
            "numerical_decision_computed_only_after_reload": True,
        },
        "parent_state": {
            "before_sha256": parent_before,
            "after_sha256": parent_after,
            "byte_identical": True,
        },
        "BASE_exact_replay": {
            "valid": True,
            "FIT_identities_and_pass_records": True,
            "initial_and_final_padded_states": True,
            "pruned_state_map_and_FIT_transfer": True,
            "CAL_raw_AA_parameters_and_provenance": True,
            "DEV_raw_and_AA_arrays": True,
        },
        "training": training,
        "pruning": pruning,
        "calibration": calibration,
        **numerical,
        "interpretation_constraints": {
            "permanently_exploratory_consumed_data": True,
            "fresh_data_license_is_not_nomination": True,
            "no_checkpoint_candidate_or_qualification": True,
            "same_consumed_DEV_retry_allowed": False,
        },
    }
    if validate_pb21k_parent(root) != frozen_parent:
        raise RuntimeError("PB21K parent drifted before result publication")
    if _diagnostic_source_bundle(root) != source_bundle:
        raise RuntimeError("PB21L source bundle drifted before result publication")
    if _sha256_file(evidence_path) != publication["sha256"]:
        raise RuntimeError("PB21L evidence drifted before result publication")
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
            raise FileExistsError(f"PB21L canonical {role} exists; retry is forbidden")
    registration_record: Mapping[str, object] | None = None
    attempt_record: Mapping[str, object] | None = None
    try:
        registration_record = validate_registration(registration)
        preloaded_parent = strict.load_exact_parent(upstream_result)
        attempt_record = _publish_attempt(root, registration_record)
        _run_impl(
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
                    "classification": "exploratory_consumed_data_failed_nonqualifying",
                    "qualification_claimed": False,
                    "candidate_publication_allowed": False,
                    "candidate_checkpoint": {
                        "published": False,
                        "checkpoint_emitted": False,
                    },
                    "nomination": None,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--upstream-result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.register_only:
        if args.upstream_result is not None or args.output is not None:
            parser.error("--register-only accepts only --registration")
        register(args.registration)
        return
    if args.upstream_result is None or args.output is None:
        parser.error("run requires --upstream-result and --output")
    run(
        upstream_result=args.upstream_result,
        output=args.output,
        registration=args.registration,
    )


if __name__ == "__main__":
    main()
