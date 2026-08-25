"""PB21M fresh BAL-UPMIX versus BASE, with a sealed CAL futility stage.

This runner consumes new FIT and split TRAIN-CAL cohorts.  It publishes and
reloads cross-fitted CAL evidence before evaluating the preregistered futility
gate.  DEV is not constructed unless that gate passes.  A passing DEV result
can nominate only the BAL-UPMIX+AA recipe for a later scaling qualification;
this run emits no checkpoint and makes no scaling claim.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import tempfile
from time import perf_counter_ns
from typing import Callable, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import Tensor

from irene_brain.data import (
    DatasetSplit,
    MazeChaseDatasetConfig,
    maze_chase_dataset_manifest_sha256,
)
from irene_brain.evaluation.v21_qualification_metrics import binary_probability_metrics
from irene_brain.v2.hazard_calibration import (
    PerActionAffineHazardCalibrator,
    TrainCalibrationProvenance,
    fit_train_only_per_action_affine,
)
from run_provenance import apply_deterministic_mode
import v21i_development_runner as development
import v21i_strict_live_representation_probe_v1 as strict
import v21j_fresh_bz_production_form_diagnostic_v1 as pb21j
import v21k_nz_dualcal_diagnostic_v1 as pb21k
import v21l_nz_upmix_consumed_feasibility_v1 as pb21l


SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = 1
MODE = "pb21m_fresh_bal_cal_first_qualification_v1"
REGISTRATION_MODE = "pb21m_fresh_bal_cal_first_registration_v1"
RUN_ID = "2026-08-25-pb21m-fresh-bal-cal-first-v1"
CLASSIFICATION = "fresh_architecture_qualification_not_checkpoint"

CANONICAL_UPSTREAM_RESULT = pb21k.CANONICAL_UPSTREAM_RESULT
RUN_DIRECTORY = "brain/runs/v21m-qualification"
CANONICAL_REGISTRATION = f"{RUN_DIRECTORY}/{RUN_ID}.registration.json"
CANONICAL_ATTEMPT = f"{RUN_DIRECTORY}/{RUN_ID}.attempt.json"
CANONICAL_CAL_EVIDENCE = f"{RUN_DIRECTORY}/{RUN_ID}.cal-evidence.npz"
CANONICAL_CAL_DECISION = f"{RUN_DIRECTORY}/{RUN_ID}.cal-decision.json"
CANONICAL_DEV_OPEN = f"{RUN_DIRECTORY}/{RUN_ID}.dev-open.json"
CANONICAL_DEV_EVIDENCE = f"{RUN_DIRECTORY}/{RUN_ID}.dev-evidence.npz"
CANONICAL_RESULT = f"{RUN_DIRECTORY}/{RUN_ID}.json"

PB21L_REGISTRATION = pb21l.CANONICAL_REGISTRATION
PB21L_ATTEMPT = pb21l.CANONICAL_ATTEMPT
PB21L_RESULT = pb21l.CANONICAL_RESULT
EXACT_PB21L_REGISTRATION_SHA256 = "069ff428f84b772a7b853637248d45deb225738794f918a5e8887bce9f8937f0"
EXACT_PB21L_ATTEMPT_SHA256 = "7a361cb03cb25c3699ae3b6fe2dcea26e1dd992d3217ea7026dc7b7de8935156"
EXACT_PB21L_RESULT_SHA256 = "eaf7bc1565666e8ddc371f02643b7e79a7b2c7a010798d35731fb251ffa2ad1c"
EXACT_PB21L_REGISTERED_SOURCE_BUNDLE_SHA256 = "21ba2a450fcf2106b60dd596ebc25cf1f3ccbb6cd7f1d2ff935281601f243122"

_PREREGISTRATION = "brain/docs/preregistrations/2026-08-25-pb21m-fresh-bal-cal-first-v1.md"
_TEST_FILE = "brain/tests/test_v21m_fresh_bal_cal_first_qualification_v1.py"
_RUNNER_FILE = "brain/scripts/v21m_fresh_bal_cal_first_qualification_v1.py"
_BOUND_IMPLEMENTATION_FILES = (
    _RUNNER_FILE,
    _PREREGISTRATION,
    _TEST_FILE,
    "brain/scripts/v21j_fresh_bz_production_form_diagnostic_v1.py",
    "brain/scripts/v21k_nz_dualcal_diagnostic_v1.py",
    "brain/scripts/v21l_nz_upmix_consumed_feasibility_v1.py",
    "brain/src/irene_brain/v2/hazard_calibration.py",
)

ACTION_COUNT = pb21j.ACTION_COUNT
SUPERSET_WIDTH = pb21j.SUPERSET_WIDTH
ARM_NZ = pb21j.ARM_NZ
SCORER_BASE = "BASE"
SCORER_BAL = "BAL-UPMIX"
SCORERS = (SCORER_BASE, SCORER_BAL)
FIT_COHORTS = ("F0", "F1", "F2")
CAL_A_COHORTS = ("C0A", "C1A", "C2A")
CAL_B_COHORTS = ("C0B", "C1B", "C2B")
CAL_FOLDS = ("A", "B")
INIT_SEEDS = (91_042, 92_042, 93_042)
PERMUTATION_SEED = 94_042
BOOTSTRAP_SEED = 95_042
DERANGEMENT_SEED = 96_042
CELLS = tuple(f"{fit}/init-{seed}" for fit in FIT_COHORTS for seed in INIT_SEEDS)
PASSES = 32
ROOT_BATCH_SIZE = 24
ROOTS_PER_EPISODE = pb21j.ROOTS_PER_EPISODE
FIT_EPISODES = 256
CAL_FOLD_EPISODES = 256
DEV_EPISODES = 768
ROOTS_PER_FIT = FIT_EPISODES * ROOTS_PER_EPISODE
ROOTS_PER_CAL_FOLD = CAL_FOLD_EPISODES * ROOTS_PER_EPISODE
ROOTS_PER_CAL = 2 * ROOTS_PER_CAL_FOLD
DEV_ROOTS = DEV_EPISODES * ROOTS_PER_EPISODE
STEPS_PER_HEAD = PASSES * ROOTS_PER_FIT // ROOT_BATCH_SIZE
TOTAL_HEADS = len(CELLS) * len(SCORERS)
TOTAL_OPTIMIZER_STEPS = TOTAL_HEADS * STEPS_PER_HEAD
BOOTSTRAP_RESAMPLES = 50_000
DERANGEMENT_REPETITIONS = 20
ONE_SIDED_ALPHA = 0.025
QUANTILE_METHOD = "linear"
DOMAINS = ("all_action", "factual", "nonselected_complement")
AA_BCE_MARGIN = 0.010
AA_BRIER_MARGIN = 0.005
LATENCY_P99_LIMIT_MS = 5.0
MAXIMUM_WORKING_SET_BYTES = 2 * 1024**3
MINIMUM_AVAILABLE_RAM_BYTES = 4 * 1024**3
MAXIMUM_COMMIT_FRACTION = 0.85
MINIMUM_ARTIFACT_FREE_BYTES = 5 * 1024**3


@dataclass(frozen=True)
class FreshPartitionSpec:
    label: str
    role: str
    split: DatasetSplit
    seed_offset: int
    episodes: int


@dataclass(frozen=True)
class PB21MPartitionContract:
    name: str
    dataset_config: MazeChaseDatasetConfig
    burn_in_steps: int = development.BURN_IN_STEPS

    def __post_init__(self) -> None:
        if self.name not in {"TRAIN-FIT", "TRAIN-CAL-A", "TRAIN-CAL-B", "DEV"}:
            raise ValueError("unsupported PB21M partition role")
        expected = DatasetSplit.VALIDATION if self.name == "DEV" else DatasetSplit.TRAIN
        if self.dataset_config.split is not expected:
            raise ValueError("PB21M partition role/split mismatch")
        if (self.dataset_config.counterfactual_targets != "all_actions_v1"
                or self.dataset_config.behavior_policy != "balanced_intervention_v1"
                or self.dataset_config.behavior_intervention_rate != 0.5):
            raise ValueError("PB21M requires the frozen all-action intervention dataset")
        if not 0 <= self.burn_in_steps < self.dataset_config.sequence_length:
            raise ValueError("PB21M burn-in is invalid")


PARTITION_SPECS = (
    FreshPartitionSpec("F0", "TRAIN-FIT", DatasetSplit.TRAIN, 167_772_160, 256),
    FreshPartitionSpec("C0A", "TRAIN-CAL-A", DatasetSplit.TRAIN, 167_772_416, 256),
    FreshPartitionSpec("C0B", "TRAIN-CAL-B", DatasetSplit.TRAIN, 167_772_672, 256),
    FreshPartitionSpec("F1", "TRAIN-FIT", DatasetSplit.TRAIN, 167_772_928, 256),
    FreshPartitionSpec("C1A", "TRAIN-CAL-A", DatasetSplit.TRAIN, 167_773_184, 256),
    FreshPartitionSpec("C1B", "TRAIN-CAL-B", DatasetSplit.TRAIN, 167_773_440, 256),
    FreshPartitionSpec("F2", "TRAIN-FIT", DatasetSplit.TRAIN, 167_773_696, 256),
    FreshPartitionSpec("C2A", "TRAIN-CAL-A", DatasetSplit.TRAIN, 167_773_952, 256),
    FreshPartitionSpec("C2B", "TRAIN-CAL-B", DatasetSplit.TRAIN, 167_774_208, 256),
    FreshPartitionSpec("DEV", "DEV", DatasetSplit.VALIDATION, 184_549_376, 768),
)
PARTITION_ALGORITHM = (
    "pb21m-v1:F0=train[167772160,167772416);C0A=train[167772416,167772672);"
    "C0B=train[167772672,167772928);F1=train[167772928,167773184);"
    "C1A=train[167773184,167773440);C1B=train[167773440,167773696);"
    "F2=train[167773696,167773952);C2A=train[167773952,167774208);"
    "C2B=train[167774208,167774464);DEV=validation[184549376,184550144)"
)

CAL_EVIDENCE_KEYS = frozenset({
    "schema_version", "scorer_ids", "fit_ids", "cal_fold_ids", "cell_ids",
    "cell_fit_index", "cell_init_seed", "fit_targets", "fit_actions", "fit_root_ids",
    "fit_group_ids", "factual_action_counts", "factual_balance_weights_float32",
    "training_initial_state_sha256", "training_final_state_sha256", "training_steps",
    "fit_NZ_feature_sha256", "training_pass_number", "training_pass_mean_all_BCE",
    "training_pass_mean_factual_component_BCE", "training_pass_mean_objective",
    "training_pass_maximum_preclip_gradient_norm", "training_pass_permutation_sha256",
    "pruned_state_sha256", "pruning_mapping_sha256", "pruning_active_columns",
    "pruning_parameter_count", "pruning_padded_logit_sha256",
    "pruning_pruned_logit_sha256", "pruning_maximum_absolute_difference",
    "source_partition_ids", "source_dataset_manifest_sha256",
    "source_targets_sha256", "source_actions_sha256", "source_root_ids_ordered_sha256",
    "source_group_ids_ordered_sha256", "cal_targets", "cal_actions",
    "cal_root_ids", "cal_cluster_ordinal", "cal_cluster_ids", "cal_raw_logits",
    "crossfit_fit_fold_index", "crossfit_eval_fold_index", "oof_row_eval_fold_index",
    "crossfit_scales", "crossfit_biases", "crossfit_accepted", "final_scales",
    "final_biases", "final_accepted", "final_dataset_manifest_sha256",
    "crossfit_report_observations", "crossfit_report_positives",
    "crossfit_report_negatives", "crossfit_report_before_BCE",
    "crossfit_report_after_BCE", "crossfit_report_accepted",
    "final_report_observations", "final_report_positives", "final_report_negatives",
    "final_report_before_BCE", "final_report_after_BCE", "final_report_accepted",
    "oof_calibrated_logits",
    "oof_calibrated_probabilities", "calibration_source_namespace",
    "calibration_source_split", "calibration_source_partition",
    "calibration_group_sha256", "calibration_fit_group_sha256",
    "calibration_upstream_checkpoint_sha256", "calibration_dataset_manifest_sha256",
    "calibration_source_bundle_sha256", "calibration_partition_algorithm",
    "final_calibration_source_namespace", "final_calibration_source_split",
    "final_calibration_source_partition", "final_calibration_group_sha256",
    "final_calibration_fit_group_sha256", "final_calibration_upstream_checkpoint_sha256",
    "final_calibration_dataset_manifest_sha256", "final_calibration_source_bundle_sha256",
    "final_calibration_partition_algorithm",
    "bootstrap_indices", "episode_derangements",
})
DEV_EVIDENCE_KEYS = frozenset({
    "schema_version", "cal_evidence_sha256", "cal_decision_sha256", "dev_open_sha256",
    "scorer_ids", "cell_ids",
    "cell_fit_index", "cell_init_seed", "fit_targets", "fit_actions", "dev_targets",
    "dev_actions", "dev_root_ids", "dev_cluster_ordinal", "dev_cluster_ids",
    "dev_dataset_manifest_sha256", "dev_targets_sha256", "dev_actions_sha256",
    "dev_root_ids_ordered_sha256", "dev_group_ids_ordered_sha256",
    "final_scales", "final_biases", "final_accepted", "dev_raw_logits",
    "dev_raw_probabilities", "dev_calibrated_logits", "dev_calibrated_probabilities",
    "latency_measurements_ms", "bootstrap_indices", "episode_derangements",
})
PREFLIGHT_EVIDENCE_KEYS = frozenset({
    "schema_version", "targets", "actions", "raw_logits", "calibrated_logits",
    "calibrated_probabilities", "scales", "biases", "accepted",
    "calibration_source_namespace", "calibration_source_split",
    "calibration_source_partition", "calibration_group_sha256",
    "calibration_fit_group_sha256", "calibration_upstream_checkpoint_sha256",
    "calibration_dataset_manifest_sha256", "calibration_source_bundle_sha256",
    "calibration_partition_algorithm",
})


def _array_contract(shape: Sequence[int], dtype: str) -> dict[str, object]:
    return {"shape": list(shape), "dtype": np.dtype(dtype).str}


def expected_cal_evidence_contract() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    def put(names: Sequence[str], shape: Sequence[int], dtype: str) -> None:
        for name in names:
            result[name] = _array_contract(shape, dtype)
    put(("schema_version",), (), "<u2")
    put(("scorer_ids",), (2,), "S12"); put(("fit_ids",), (3,), "S2")
    put(("cal_fold_ids",), (2,), "S1"); put(("cell_ids",), (9,), "S16")
    put(("cell_fit_index",), (9,), "<i2"); put(("cell_init_seed",), (9,), "<i4")
    put(("fit_targets",), (3, ROOTS_PER_FIT, ACTION_COUNT), "<f8")
    put(("fit_actions",), (3, ROOTS_PER_FIT), "<i8")
    put(("fit_root_ids", "fit_group_ids"), (3, ROOTS_PER_FIT), "S64")
    put(("factual_action_counts",), (9, ACTION_COUNT), "<i8")
    put(("factual_balance_weights_float32",), (9, ROOTS_PER_FIT), "<f4")
    put(("training_initial_state_sha256", "training_final_state_sha256"), (2, 9), "S64")
    put(("training_steps",), (2, 9), "<i4"); put(("fit_NZ_feature_sha256",), (3,), "S64")
    put(("training_pass_number",), (2, 9, PASSES), "<i2")
    put(("training_pass_mean_all_BCE", "training_pass_mean_factual_component_BCE",
         "training_pass_mean_objective", "training_pass_maximum_preclip_gradient_norm"),
        (2, 9, PASSES), "<f8")
    put(("training_pass_permutation_sha256",), (2, 9, PASSES), "S64")
    put(("pruned_state_sha256", "pruning_mapping_sha256", "pruning_padded_logit_sha256",
         "pruning_pruned_logit_sha256"), (2, 9), "S64")
    put(("pruning_active_columns",), (len(pb21j.ACTIVE_COLUMNS[ARM_NZ]),), "<i2")
    put(("pruning_parameter_count",), (2, 9), "<i4")
    put(("pruning_maximum_absolute_difference",), (2, 9), "<f8")
    put(("source_partition_ids",), (9,), "S3")
    put(("source_dataset_manifest_sha256", "source_targets_sha256", "source_actions_sha256",
         "source_root_ids_ordered_sha256", "source_group_ids_ordered_sha256"), (9,), "S64")
    put(("cal_targets",), (3, ROOTS_PER_CAL, ACTION_COUNT), "<f8")
    put(("cal_actions",), (3, ROOTS_PER_CAL), "<i8")
    put(("cal_root_ids",), (3, ROOTS_PER_CAL), "S64")
    put(("cal_cluster_ordinal",), (3, ROOTS_PER_CAL), "<i4")
    put(("cal_cluster_ids",), (3, 2 * CAL_FOLD_EPISODES), "S64")
    put(("cal_raw_logits",), (2, 9, 2, ROOTS_PER_CAL_FOLD, ACTION_COUNT), "<f8")
    put(("crossfit_fit_fold_index", "crossfit_eval_fold_index"), (2,), "<i2")
    put(("oof_row_eval_fold_index",), (ROOTS_PER_CAL,), "<i2")
    put(("crossfit_scales", "crossfit_biases"), (2, 9, 2, ACTION_COUNT), "<f8")
    put(("crossfit_accepted",), (2, 9, 2), "|b1")
    put(("final_scales", "final_biases"), (2, 9, ACTION_COUNT), "<f8")
    put(("final_accepted",), (2, 9), "|b1")
    put(("final_dataset_manifest_sha256",), (2, 9), "S64")
    put(("crossfit_report_observations", "crossfit_report_positives",
         "crossfit_report_negatives"), (2, 9, 2, ACTION_COUNT), "<i4")
    put(("crossfit_report_before_BCE", "crossfit_report_after_BCE"),
        (2, 9, 2, ACTION_COUNT), "<f8")
    put(("crossfit_report_accepted",), (2, 9, 2, ACTION_COUNT), "|b1")
    put(("final_report_observations", "final_report_positives", "final_report_negatives"),
        (2, 9, ACTION_COUNT), "<i4")
    put(("final_report_before_BCE", "final_report_after_BCE"), (2, 9, ACTION_COUNT), "<f8")
    put(("final_report_accepted",), (2, 9, ACTION_COUNT), "|b1")
    put(("oof_calibrated_logits", "oof_calibrated_probabilities"),
        (2, 9, ROOTS_PER_CAL, ACTION_COUNT), "<f8")
    for prefix, count in (("", 36), ("final_", 18)):
        put((f"{prefix}calibration_source_namespace",), (count,), "S64")
        put((f"{prefix}calibration_source_split", f"{prefix}calibration_source_partition"),
            (count,), "S16")
        put((f"{prefix}calibration_group_sha256", f"{prefix}calibration_fit_group_sha256",
             f"{prefix}calibration_upstream_checkpoint_sha256",
             f"{prefix}calibration_dataset_manifest_sha256",
             f"{prefix}calibration_source_bundle_sha256"), (count,), "S64")
        put((f"{prefix}calibration_partition_algorithm",), (count,), "S512")
    put(("bootstrap_indices",),
        (BOOTSTRAP_RESAMPLES, 3, 2, CAL_FOLD_EPISODES), "<u2")
    put(("episode_derangements",),
        (DERANGEMENT_REPETITIONS, 3, 2, CAL_FOLD_EPISODES), "<u2")
    if set(result) != CAL_EVIDENCE_KEYS:
        raise RuntimeError("internal CAL evidence contract key drift")
    return result


def expected_dev_evidence_contract() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    def put(names: Sequence[str], shape: Sequence[int], dtype: str) -> None:
        for name in names:
            result[name] = _array_contract(shape, dtype)
    put(("schema_version",), (), "<u2")
    put(("cal_evidence_sha256", "cal_decision_sha256", "dev_open_sha256",
         "dev_dataset_manifest_sha256", "dev_targets_sha256", "dev_actions_sha256",
         "dev_root_ids_ordered_sha256", "dev_group_ids_ordered_sha256"), (), "S64")
    put(("scorer_ids",), (2,), "S12"); put(("cell_ids",), (9,), "S16")
    put(("cell_fit_index",), (9,), "<i2"); put(("cell_init_seed",), (9,), "<i4")
    put(("fit_targets",), (3, ROOTS_PER_FIT, ACTION_COUNT), "<f8")
    put(("fit_actions",), (3, ROOTS_PER_FIT), "<i8")
    put(("dev_targets",), (DEV_ROOTS, ACTION_COUNT), "<f8")
    put(("dev_actions",), (DEV_ROOTS,), "<i8"); put(("dev_root_ids",), (DEV_ROOTS,), "S64")
    put(("dev_cluster_ordinal",), (DEV_ROOTS,), "<i4")
    put(("dev_cluster_ids",), (DEV_EPISODES,), "S64")
    put(("final_scales", "final_biases"), (2, 9, ACTION_COUNT), "<f8")
    put(("final_accepted",), (2, 9), "|b1")
    put(("dev_raw_logits", "dev_raw_probabilities", "dev_calibrated_logits",
         "dev_calibrated_probabilities"), (2, 9, DEV_ROOTS, ACTION_COUNT), "<f8")
    put(("latency_measurements_ms",), (2, 9, 256), "<f8")
    put(("bootstrap_indices",), (BOOTSTRAP_RESAMPLES, DEV_EPISODES), "<u2")
    put(("episode_derangements",), (DERANGEMENT_REPETITIONS, DEV_EPISODES), "<u2")
    if set(result) != DEV_EVIDENCE_KEYS:
        raise RuntimeError("internal DEV evidence contract key drift")
    return result


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return strict._array_sha256(np.asarray(values))


def _canonical_array(value: np.ndarray) -> np.ndarray:
    return pb21k._canonical_evidence_array(np.asarray(value))


def _byte_strings(values: Sequence[str], width: int) -> np.ndarray:
    encoded = [value.encode("utf-8") for value in values]
    if any(len(value) > width for value in encoded):
        raise ValueError(f"identifier exceeds frozen S{width} width")
    return np.asarray(encoded, dtype=f"S{width}")


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.dtype.kind != "S" or array.ndim != 1:
        raise ValueError("expected a one-dimensional byte-string array")
    return tuple(value.rstrip(b"\x00").decode("utf-8") for value in array.tolist())


def _state_sha256(module: torch.nn.Module) -> str:
    return development._state_dict_sha256(module.state_dict())


def resource_guard(artifact_path: Path, *, phase: str) -> dict[str, object]:
    if torch.get_num_threads() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("PB21M resource guard requires one Torch thread and hidden CUDA")
    if torch.get_num_interop_threads() != 1:
        raise RuntimeError("PB21M resource guard requires one Torch inter-op thread")
    working_set = 0
    available = 1 << 63
    commit_fraction = 0.0
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(status)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        counters = PROCESS_MEMORY_COUNTERS(); counters.cb = ctypes.sizeof(counters)
        process_handle = kernel32.GetCurrentProcess()
        if not process_handle or not psapi.GetProcessMemoryInfo(
            process_handle, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        working_set = int(counters.WorkingSetSize)
        available = int(status.ullAvailPhys)
        commit_fraction = 1.0 - int(status.ullAvailPageFile) / max(1, int(status.ullTotalPageFile))
    disk_anchor = artifact_path.parent
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    free_disk = shutil.disk_usage(disk_anchor).free
    snapshot = {"phase": phase, "process_working_set_bytes": working_set,
                "available_physical_RAM_bytes": available,
                "committed_memory_fraction": commit_fraction,
                "artifact_disk_free_bytes": free_disk, "foreground_processes": 1,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "CUDA_visible_devices": "-1"}
    checks = {"working_set_below_2_GiB": working_set < MAXIMUM_WORKING_SET_BYTES,
              "available_RAM_at_least_4_GiB": available >= MINIMUM_AVAILABLE_RAM_BYTES,
              "commit_below_85_percent": commit_fraction < MAXIMUM_COMMIT_FRACTION,
              "artifact_disk_free_at_least_5_GiB": free_disk >= MINIMUM_ARTIFACT_FREE_BYTES}
    snapshot["checks"] = checks; snapshot["passed"] = all(checks.values())
    if not snapshot["passed"]:
        raise RuntimeError(f"PB21M resource guard failed before {phase}: {checks}")
    return snapshot


def _key_set_sha256(keys: Sequence[str], stage: str) -> str:
    encoded = json.dumps(sorted(keys), separators=(",", ":")).encode("ascii")
    return sha256(f"IRPB21M{stage}KEYS\x01".encode("ascii") + encoded).hexdigest()


def _json_builtin(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_builtin(item) for item in value]
    return value


def partition_contracts() -> dict[str, PB21MPartitionContract]:
    common = pb21j._common_dataset_config()
    return {
        spec.label: PB21MPartitionContract(
            spec.role,
            MazeChaseDatasetConfig(
                split=spec.split,
                seed_offset=spec.seed_offset,
                sequence_count=spec.episodes,
                **common,
            ),
        )
        for spec in PARTITION_SPECS
    }


def partition_manifests() -> dict[str, str]:
    return {
        label: maze_chase_dataset_manifest_sha256(contract.dataset_config)
        for label, contract in partition_contracts().items()
    }


def combined_cal_manifest_sha256(left: str, right: str) -> str:
    if len(bytes.fromhex(left)) != 32 or len(bytes.fromhex(right)) != 32:
        raise ValueError("CAL fold manifests must be SHA-256")
    return sha256(b"IRPB21MCALABMANIFEST\x01" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def occupied_range_registry() -> list[dict[str, object]]:
    records = list(pb21k.occupied_range_registry())
    for spec in PARTITION_SPECS:
        effective_start, effective_stop = pb21j._effective_interval(
            spec.split.value, spec.seed_offset, spec.seed_offset + spec.episodes
        )
        records.append({
            "dataset_family": "maze_chase", "split": spec.split.value,
            "local_start": spec.seed_offset,
            "local_stop_exclusive": spec.seed_offset + spec.episodes,
            "effective_start": effective_start, "effective_stop_exclusive": effective_stop,
            "owner": f"PB21M {spec.label}",
        })
    return records


def assert_no_range_collisions(
    records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    frozen = list(occupied_range_registry() if records is None else records)
    collisions: list[tuple[str, str]] = []
    for index, left in enumerate(frozen):
        for right in frozen[index + 1:]:
            if left["dataset_family"] != right["dataset_family"]:
                continue
            if max(int(left["effective_start"]), int(right["effective_start"])) < min(
                int(left["effective_stop_exclusive"]), int(right["effective_stop_exclusive"])
            ):
                collisions.append((str(left["owner"]), str(right["owner"])))
    if collisions:
        raise RuntimeError(f"occupied seed ranges collide: {collisions}")
    return {"algorithm": "split-prefix-effective-u64-interval-audit-v1",
            "records": frozen, "collisions": [], "passed": True}


def partition_sources(
    labels: Sequence[str],
    *,
    source_factory: Callable[[PB21MPartitionContract], object] = development.PartitionSource,
) -> dict[str, object]:
    allowed = {spec.label for spec in PARTITION_SPECS}
    if not labels or len(set(labels)) != len(labels) or any(label not in allowed for label in labels):
        raise ValueError("only unique registered PB21M partition labels may be constructed")
    contracts = partition_contracts()
    manifests = partition_manifests()
    sources = {label: source_factory(contracts[label]) for label in labels}
    for label, source in sources.items():
        if getattr(source, "manifest_sha256", None) != manifests[label]:
            raise RuntimeError(f"constructed {label} manifest differs from registration")
    return sources


def validate_scientific_parent(project_root: Path) -> dict[str, object]:
    pb21k_parent = pb21l.validate_pb21k_parent(project_root)
    expected = (
        (project_root / PB21L_REGISTRATION, EXACT_PB21L_REGISTRATION_SHA256, "registration"),
        (project_root / PB21L_ATTEMPT, EXACT_PB21L_ATTEMPT_SHA256, "attempt"),
        (project_root / PB21L_RESULT, EXACT_PB21L_RESULT_SHA256, "result"),
    )
    for path, digest, role in expected:
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"exact immutable PB21L {role} artifact absent or drifted")
    registration = json.loads((project_root / PB21L_REGISTRATION).read_text(encoding="utf-8"))
    result = json.loads((project_root / PB21L_RESULT).read_text(encoding="utf-8"))
    if registration.get("source_bundle", {}).get("sha256") != EXACT_PB21L_REGISTERED_SOURCE_BUNDLE_SHA256:
        raise ValueError("PB21L registered source-bundle identity drifted")
    if (
        result.get("classification") != "exploratory_consumed_data_failed_nonqualifying"
        or result.get("authoritative_evidence") is not None
        or result.get("retry_allowed") is not False
        or result.get("failure", {}).get("type") != "AttributeError"
    ):
        raise ValueError("PB21L failure envelope drifted")
    return {
        "PB21K": pb21k_parent,
        "PB21L_registration_sha256": EXACT_PB21L_REGISTRATION_SHA256,
        "PB21L_attempt_sha256": EXACT_PB21L_ATTEMPT_SHA256,
        "PB21L_failure_result_sha256": EXACT_PB21L_RESULT_SHA256,
        "PB21L_authoritative_metrics_available": False,
        "PB21L_retry_allowed": False,
    }


def _source_bundle(project_root: Path) -> dict[str, object]:
    digest = sha256(b"IRPB21MFRESHBAL\x01")
    files: dict[str, str] = {}
    for relative in _BOUND_IMPLEMENTATION_FILES:
        observed = _sha256_file(project_root / relative)
        files[relative] = observed
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(observed))
    for relative, expected in (
        (PB21L_REGISTRATION, EXACT_PB21L_REGISTRATION_SHA256),
        (PB21L_ATTEMPT, EXACT_PB21L_ATTEMPT_SHA256),
        (PB21L_RESULT, EXACT_PB21L_RESULT_SHA256),
    ):
        observed = _sha256_file(project_root / relative)
        if observed != expected:
            raise ValueError(f"immutable parent artifact drifted: {relative}")
        files[relative] = observed
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(observed))
    return {"schema_version": 1, "sha256": digest.hexdigest(), "bound_files": files}


def objective_contract() -> dict[str, object]:
    return {
        "BASE": "mean_all_root_action_BCE",
        "BAL-UPMIX": (
            "0.5*mean_all_root_action_BCE+0.5*(1/5)*sum_a(mean_matching_factual_root_BCE_a)"
        ),
        "BAL_minibatch_weight": "ROOTS_PER_FIT/(5*full_FIT_factual_count[action])",
        "factual_selector": "loss_only; current applied action is never an input feature",
        "no_adaptive_arm_or_weight_selection": True,
    }


def schedule_record() -> dict[str, object]:
    return {
        "fit_cohorts": list(FIT_COHORTS), "cal_A_cohorts": list(CAL_A_COHORTS),
        "cal_B_cohorts": list(CAL_B_COHORTS), "initialization_seeds": list(INIT_SEEDS),
        "scorers": list(SCORERS), "cells": list(CELLS), "head_count": TOTAL_HEADS,
        "passes": PASSES, "roots_per_fit": ROOTS_PER_FIT,
        "root_batch_size": ROOT_BATCH_SIZE, "steps_per_head": STEPS_PER_HEAD,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS, "permutation_seed": PERMUTATION_SEED,
        "ordering": [
            "publish_attempt", "construct_FIT_and_both_CAL_folds",
            "validate_geometry_support_and_9_partition_disjointness_before_training",
            "fit_and_prune_all_18_heads", "fit_crossfold_and_final_AA",
            "publish_reload_refit_validate_CAL_evidence", "compute_CAL_futility_gate",
            "only_if_CAL_passes_construct_DEV", "publish_reload_validate_DEV_evidence",
            "compute_DEV_gate_and_recipe_nomination",
        ],
        "device": "cpu", "threads": 1, "cuda_visible_devices": "-1",
    }


def inference_contract() -> dict[str, object]:
    return {
        "domains": list(DOMAINS), "crossfit_order": "A predictions from fit-B then B predictions from fit-A",
        "candidate": SCORER_BAL, "control": SCORER_BASE, "AA_only": True,
        "factual_thresholds": {"BCE": 0.0, "Brier": 0.0},
        "all_and_complement_thresholds": {"BCE": -AA_BCE_MARGIN, "Brier": -AA_BRIER_MARGIN},
        "every_cell_every_cohort_and_grand_97_5_percent_LCB": True,
        "CAL_futility_before_DEV_construction": True,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "derangement_seed": DERANGEMENT_SEED, "one_sided_alpha": ONE_SIDED_ALPHA,
        "head_only_latency_P99_diagnostic_limit_ms": LATENCY_P99_LIMIT_MS,
        "head_only_latency_is_not_full_live_workload_but_is_operational_nomination_gate": True,
    }


def decision_contract() -> dict[str, object]:
    return {
        "sole_candidate": SCORER_BAL, "fallback": None, "adaptive_retry": False,
        "CAL_failure": "publish_CAL-only_negative_result_and_never_construct_DEV",
        "DEV_pass": "nominate_recipe_for_later_scaling_qualification_only",
        "DEV_operational_latency_gate": "pruned_hazard_head_CPU_batch1_p99<=5ms",
        "checkpoint_emitted": False, "full_model_training_authorized": False,
        "scalability_claimed": False, "same_data_retry_allowed": False,
    }


def canonical_paths(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve()
    return {
        "upstream_result": (root / CANONICAL_UPSTREAM_RESULT).resolve(),
        "registration": (root / CANONICAL_REGISTRATION).resolve(),
        "attempt": (root / CANONICAL_ATTEMPT).resolve(),
        "cal_evidence": (root / CANONICAL_CAL_EVIDENCE).resolve(),
        "cal_decision": (root / CANONICAL_CAL_DECISION).resolve(),
        "dev_open": (root / CANONICAL_DEV_OPEN).resolve(),
        "dev_evidence": (root / CANONICAL_DEV_EVIDENCE).resolve(),
        "result": (root / CANONICAL_RESULT).resolve(),
    }


def _require_canonical_path(path: Path, expected: Path, *, role: str) -> Path:
    if path.resolve() != expected.resolve():
        raise ValueError(f"{role} must use the single canonical path {expected}")
    return path.resolve()


def _calibrator_provenance_arrays(
    calibrators: Sequence[PerActionAffineHazardCalibrator],
) -> dict[str, np.ndarray]:
    """Package real calibrator fields; no synthetic `.provenance` attribute."""
    values = tuple(calibrators)
    if not values or any(not isinstance(value, PerActionAffineHazardCalibrator) for value in values):
        raise TypeError("only real PerActionAffineHazardCalibrator instances are packageable")
    fields = {
        "calibration_source_namespace": [value.source_namespace for value in values],
        "calibration_source_split": [value.source_split for value in values],
        "calibration_source_partition": [value.source_partition for value in values],
        "calibration_group_sha256": [value.calibration_group_digest for value in values],
        "calibration_fit_group_sha256": [value.upstream_model_fit_group_digest for value in values],
        "calibration_upstream_checkpoint_sha256": [value.upstream_checkpoint_sha256 for value in values],
        "calibration_dataset_manifest_sha256": [value.dataset_manifest_sha256 for value in values],
        "calibration_source_bundle_sha256": [value.source_bundle_sha256 for value in values],
        "calibration_partition_algorithm": [value.partition_algorithm for value in values],
    }
    widths = {"calibration_source_namespace": 64, "calibration_source_split": 16,
              "calibration_source_partition": 16, "calibration_partition_algorithm": 512}
    return {name: _byte_strings(items, widths.get(name, 64)) for name, items in fields.items()}


def real_calibrator_packaging_preflight(project_root: Path) -> dict[str, object]:
    """Reduced synthetic builder->writer->reload->refit->evaluator integration gate."""
    lifecycle = canonical_paths(project_root)
    protected = tuple(lifecycle[name] for name in (
        "attempt", "cal_evidence", "cal_decision", "dev_open", "dev_evidence", "result"
    ))
    before = tuple(path.exists() for path in protected)
    rows = 40
    logits = np.linspace(-2.0, 2.0, rows * ACTION_COUNT, dtype=np.float64).reshape(rows, ACTION_COUNT)
    targets = ((np.arange(rows)[:, None] + np.arange(ACTION_COUNT)[None, :]) % 3 == 0).astype(np.float64)
    actions = np.arange(rows, dtype=np.int64) % ACTION_COUNT
    action_ids = np.broadcast_to(np.arange(ACTION_COUNT, dtype=np.int64), logits.shape)
    group_ids = tuple(f"fixture:{row}" for row in range(rows) for _ in range(ACTION_COUNT))
    fit_group_ids = tuple(f"fit:{row}" for row in range(rows))
    provenance = TrainCalibrationProvenance(
        source_namespace="pb21m.synthetic-preflight.v1", source_split="TRAIN",
        source_partition="TRAIN-CAL", calibration_group_ids=group_ids,
        upstream_model_fit_group_ids=fit_group_ids,
        upstream_checkpoint_sha256="1" * 64, dataset_manifest_sha256="2" * 64,
        source_bundle_sha256="3" * 64,
        partition_algorithm="synthetic-no-reserved-data-packaging-preflight-v1",
    )

    def fit() -> PerActionAffineHazardCalibrator:
        return fit_train_only_per_action_affine(
            torch.from_numpy(logits.reshape(-1)), torch.from_numpy(targets.reshape(-1)),
            torch.from_numpy(action_ids.reshape(-1)), provenance=provenance,
            action_count=ACTION_COUNT, l2_regularization=pb21j.CALIBRATION_L2,
            minimum_scale=pb21j.CALIBRATION_MINIMUM_SCALE,
            minimum_examples_per_action=pb21j.CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
            minimum_class_examples=pb21j.CALIBRATION_MINIMUM_CLASS_EXAMPLES,
            max_iterations=pb21j.CALIBRATION_MAX_ITERATIONS,
            tolerance=pb21j.CALIBRATION_TOLERANCE, fit_mode="per_action_affine",
        )

    calibrator = fit()
    calibrated_logits, probabilities = _apply_calibrator(calibrator, logits)
    packed = _calibrator_provenance_arrays((calibrator,))
    arrays = {
        "schema_version": np.asarray(1, dtype=np.uint16), "targets": targets,
        "actions": actions, "raw_logits": logits, "calibrated_logits": calibrated_logits,
        "calibrated_probabilities": probabilities,
        "scales": np.asarray(calibrator.scales, dtype=np.float64),
        "biases": np.asarray(calibrator.biases, dtype=np.float64),
        "accepted": np.asarray(calibrator.accepted, dtype=np.bool_), **packed,
    }
    with tempfile.TemporaryDirectory(prefix="pb21m-preflight-") as directory:
        path = Path(directory) / "preflight.npz"
        publication = publish_evidence_create_only(
            path, arrays, stage="PREFLIGHT", attempt_sha256="0" * 64
        )
        reloaded = reload_evidence(path, publication, stage="PREFLIGHT")
    refitted = fit()
    if (not np.array_equal(reloaded["scales"], np.asarray(refitted.scales))
            or not np.array_equal(reloaded["biases"], np.asarray(refitted.biases))):
        raise RuntimeError("real-calibrator deterministic refit failed after evidence reload")
    expected_probability = 1.0 / (1.0 + np.exp(-np.clip(reloaded["calibrated_logits"], -700, 700)))
    if not np.array_equal(expected_probability, reloaded["calibrated_probabilities"]):
        raise RuntimeError("preflight evaluator probability transform failed")
    metrics = binary_probability_metrics(
        reloaded["targets"].reshape(-1), reloaded["calibrated_probabilities"].reshape(-1),
        ece_bins=pb21k.ECE_BINS,
    ).as_dict()
    after = tuple(path.exists() for path in protected)
    if after != before:
        raise RuntimeError("synthetic preflight touched a canonical lifecycle path")
    passed = bool(calibrator.accepted and refitted.accepted and pb21j._all_numeric_finite(metrics))
    return {
        "uses_real_fitted_PerActionAffineHazardCalibrator": True,
        "builder_writer_allow_pickle_false_reload_refit_evaluator_traversed": True,
        "synthetic_rows": rows, "reserved_partition_constructed": False,
        "canonical_lifecycle_paths_untouched": True,
        "packaged_fields": sorted(packed), "evidence_sha256": publication["sha256"],
        "array_manifest_sha256": publication["array_manifest_sha256"],
        "accepted": bool(calibrator.accepted), "refit_accepted": bool(refitted.accepted),
        "passed": passed,
    }


def registration_payload(project_root: Path) -> dict[str, object]:
    preflight = real_calibrator_packaging_preflight(project_root)
    if not preflight["passed"]:
        raise RuntimeError("real calibrator packaging preflight failed")
    production_preflight = production_cal_pipeline_preflight(project_root)
    if not production_preflight["passed"]:
        raise RuntimeError("full production CAL pipeline preflight failed")
    source_bundle = _source_bundle(project_root)
    manifests = partition_manifests()
    return {
        "schema_version": SCHEMA_VERSION, "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": REGISTRATION_MODE, "diagnostic_mode": MODE,
        "classification": "fresh_architecture_qualification_registration",
        "create_only": True, "qualification_claimed": False,
        "checkpoint_emitted": False, "scientific_parent": validate_scientific_parent(project_root),
        "source_bundle": source_bundle, "real_calibrator_packaging_preflight": preflight,
        "production_CAL_pipeline_preflight": production_preflight,
        "partitions": [
            {"label": spec.label, "role": spec.role, "split": spec.split.value,
             "seed_offset": spec.seed_offset, "episodes": spec.episodes,
             "dataset_manifest_sha256": manifests[spec.label]}
            for spec in PARTITION_SPECS
        ],
        "partition_algorithm": PARTITION_ALGORITHM,
        "occupied_range_audit": assert_no_range_collisions(),
        "objective": objective_contract(), "schedule": schedule_record(),
        "inference": inference_contract(), "decision": decision_contract(),
        "evidence_contracts": {
            "CAL": {"path": CANONICAL_CAL_EVIDENCE,
                    "allowed_key_set_sha256": _key_set_sha256(CAL_EVIDENCE_KEYS, "CAL"),
                    "arrays": expected_cal_evidence_contract()},
            "CAL_decision": {"path": CANONICAL_CAL_DECISION, "create_only": True},
            "DEV_open_receipt": {"path": CANONICAL_DEV_OPEN, "create_only": True,
                                 "requires_passing_CAL_decision": True},
            "DEV": {"path": CANONICAL_DEV_EVIDENCE,
                    "allowed_key_set_sha256": _key_set_sha256(DEV_EVIDENCE_KEYS, "DEV"),
                    "arrays": expected_dev_evidence_contract(),
                    "must_bind_CAL_evidence_sha256": True},
        },
    }


def register(registration: Path) -> None:
    apply_deterministic_mode()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[2]
    expected = canonical_paths(root)["registration"]
    _require_canonical_path(registration, expected, role="registration")
    development._publish_json_create_only(registration, registration_payload(root))


def validate_registration(registration: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    _require_canonical_path(registration, canonical_paths(root)["registration"], role="registration")
    payload = json.loads(registration.read_text(encoding="utf-8"))
    expected = registration_payload(root)
    if payload != expected:
        raise ValueError("PB21M registration does not match current frozen contract")
    return {"path": str(registration), "sha256": _sha256_file(registration), "payload": payload}


def collect_fresh_live_tape(model: object, source: object, *, partition_label: str) -> pb21j.FreshLiveTape:
    registered = {spec.label: spec for spec in PARTITION_SPECS}
    if partition_label not in registered:
        raise ValueError("collector accepts only registered PB21M partitions")
    spec = registered[partition_label]
    contract = source.contract
    if (contract.name != spec.role or contract.dataset_config.split is not spec.split
            or contract.dataset_config.seed_offset != spec.seed_offset
            or contract.dataset_config.sequence_count != spec.episodes):
        raise ValueError("source does not match its registered PB21M partition")
    was_training = model.training
    model.eval()
    beliefs: list[np.ndarray] = []
    updater_inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    parent_logits: list[np.ndarray] = []
    factual_actions: list[np.ndarray] = []
    groups: list[str] = []
    roots: list[str] = []
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
                dtype=torch.long, device=device,
            )
            step = strict.strict_live_reference_step(
                model, tuple(value.observation.rgb for value in transitions), state, prior_action,
                first_tick=tick == 0,
            )
            state = step.state
            table = step.output.outcome_table
            if tick >= batch.burn_in_steps:
                beliefs.append(step.output.belief.detach().to(torch.float64).cpu().numpy())
                updater_inputs.append(step.updater_input.detach().to(torch.float64).cpu().numpy())
                parent_logits.append(step.canonical_raw_hazard_logits)
                factual_actions.append(applied.detach().cpu().numpy().astype(np.int64, copy=False))
                targets.append(np.asarray([
                    [strict.transition_hazard(branch.event_targets)
                     for branch in transition.counterfactual_targets]
                    for transition in transitions
                ], dtype=np.float64))
                assignments.extend(int(value) for value in assignment_count)
                disagreements.extend(int(value) for value in disagreement_count)
                for sequence, transition in zip(batch.sequences, transitions, strict=True):
                    groups.append(f"{partition_label}:episode:{sequence.episode_seed}")
                    roots.append(transition.root_state_sha256)
            strict._install_applied_pending(state, table, applied)
            for index, (sequence, transition) in enumerate(zip(batch.sequences, transitions, strict=True)):
                assignment_count[index] += int(strict.behavior_assignment(
                    source.manifest_sha256, episode_seed=sequence.episode_seed, tick=tick
                ))
                disagreement_count[index] += int(
                    strict.control_action_class(transition.applied_control)
                    != strict.control_action_class(transition.action_target)
                )
            prior_action = applied
    if was_training:
        model.train()
    tape = pb21j.FreshLiveTape(
        beliefs=np.concatenate(beliefs), updater_inputs=np.concatenate(updater_inputs),
        hazard_targets=np.concatenate(targets), parent_raw_logits=np.concatenate(parent_logits),
        factual_actions=np.concatenate(factual_actions), episode_group_ids=tuple(groups),
        root_state_ids=tuple(roots), prior_assignment_count=np.asarray(assignments, dtype=np.int64),
        prior_disagreement_count=np.asarray(disagreements, dtype=np.int64),
    )
    expected = spec.episodes * ROOTS_PER_EPISODE
    if len(tape.hazard_targets) != expected or len(set(tape.root_state_ids)) != expected:
        raise RuntimeError("fresh strict-live tape geometry or uniqueness drifted")
    return tape


def deterministic_permutations() -> tuple[Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PERMUTATION_SEED)
    values = tuple(torch.randperm(ROOTS_PER_FIT, generator=generator) for _ in range(PASSES))
    if any(int(value.unique().numel()) != ROOTS_PER_FIT for value in values):
        raise RuntimeError("FIT permutation is not a bijection")
    return values


def initialized_head(seed: int) -> pb21j.MaskedSupersetHazardHead:
    if seed not in INIT_SEEDS:
        raise ValueError("PB21M initialization seed is not preregistered")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return pb21j.MaskedSupersetHazardHead()


def _features(tape: pb21j.FreshLiveTape) -> np.ndarray:
    return pb21j.superset_features(tape, ARM_NZ)


def _fit_calibrator(
    raw_logits: np.ndarray,
    tape: pb21j.FreshLiveTape,
    fit_tape: pb21j.FreshLiveTape,
    head: pb21j.PrunedHazardHead,
    *,
    manifest_sha256: str,
    source_bundle_sha256: str,
    calibration_role: str,
) -> PerActionAffineHazardCalibrator:
    logits = np.asarray(raw_logits, dtype=np.float64)
    action_ids = np.broadcast_to(np.arange(ACTION_COUNT, dtype=np.int64), logits.shape)
    provenance = TrainCalibrationProvenance(
        source_namespace=f"maze_chase.pb21m.train-only.{calibration_role}.v1",
        source_split="TRAIN", source_partition="TRAIN-CAL",
        calibration_group_ids=tuple(group for group in tape.episode_group_ids for _ in range(ACTION_COUNT)),
        upstream_model_fit_group_ids=tuple(sorted(set(fit_tape.episode_group_ids))),
        upstream_checkpoint_sha256=_state_sha256(head), dataset_manifest_sha256=manifest_sha256,
        source_bundle_sha256=source_bundle_sha256,
        partition_algorithm=f"{PARTITION_ALGORITHM};calibration_role={calibration_role}",
    )
    provenance.validate(len(tape.hazard_targets) * ACTION_COUNT)
    return fit_train_only_per_action_affine(
        torch.from_numpy(logits.reshape(-1)), torch.from_numpy(tape.hazard_targets.reshape(-1)),
        torch.from_numpy(action_ids.reshape(-1)), provenance=provenance,
        action_count=ACTION_COUNT, l2_regularization=pb21j.CALIBRATION_L2,
        minimum_scale=pb21j.CALIBRATION_MINIMUM_SCALE,
        minimum_examples_per_action=pb21j.CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
        minimum_class_examples=pb21j.CALIBRATION_MINIMUM_CLASS_EXAMPLES,
        max_iterations=pb21j.CALIBRATION_MAX_ITERATIONS, tolerance=pb21j.CALIBRATION_TOLERANCE,
        fit_mode="per_action_affine",
    )


def _apply_calibrator(
    calibrator: PerActionAffineHazardCalibrator, raw_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_logits, dtype=np.float64)
    logits = raw * np.asarray(calibrator.scales)[None, :] + np.asarray(calibrator.biases)[None, :]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
    return logits, probabilities


def _cluster_geometry(tape: pb21j.FreshLiveTape) -> tuple[np.ndarray, tuple[str, ...]]:
    ids = tuple(dict.fromkeys(tape.episode_group_ids))
    lookup = {value: index for index, value in enumerate(ids)}
    ordinal = np.asarray([lookup[value] for value in tape.episode_group_ids], dtype=np.int32)
    if any(int((ordinal == index).sum()) != ROOTS_PER_EPISODE for index in range(len(ids))):
        raise ValueError("episode cluster geometry drifted")
    return ordinal, ids


def cal_bootstrap_indices() -> np.ndarray:
    """Independent whole-episode resamples for each cohort and CAL fold."""
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, CAL_FOLD_EPISODES,
        size=(BOOTSTRAP_RESAMPLES, len(FIT_COHORTS), len(CAL_FOLDS), CAL_FOLD_EPISODES),
        dtype=np.uint16,
    )


def dev_bootstrap_indices() -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, DEV_EPISODES, size=(BOOTSTRAP_RESAMPLES, DEV_EPISODES), dtype=np.uint16
    )


def cal_episode_derangements() -> np.ndarray:
    rng = np.random.default_rng(DERANGEMENT_SEED)
    result = np.empty(
        (DERANGEMENT_REPETITIONS, len(FIT_COHORTS), len(CAL_FOLDS), CAL_FOLD_EPISODES),
        dtype=np.uint16,
    )
    identity = np.arange(CAL_FOLD_EPISODES, dtype=np.uint16)
    for repetition in range(DERANGEMENT_REPETITIONS):
        for cohort in range(len(FIT_COHORTS)):
            for fold in range(len(CAL_FOLDS)):
                value = rng.permutation(CAL_FOLD_EPISODES).astype(np.uint16, copy=False)
                while np.any(value == identity):
                    value = rng.permutation(CAL_FOLD_EPISODES).astype(np.uint16, copy=False)
                result[repetition, cohort, fold] = value
    return result


def dev_episode_derangements() -> np.ndarray:
    rng = np.random.default_rng(DERANGEMENT_SEED)
    result = np.empty((DERANGEMENT_REPETITIONS, DEV_EPISODES), dtype=np.uint16)
    identity = np.arange(DEV_EPISODES, dtype=np.uint16)
    for repetition in range(DERANGEMENT_REPETITIONS):
        value = rng.permutation(DEV_EPISODES).astype(np.uint16, copy=False)
        while np.any(value == identity):
            value = rng.permutation(DEV_EPISODES).astype(np.uint16, copy=False)
        result[repetition] = value
    return result


def _flatten_cal_derangements(values: np.ndarray, cohort: int) -> np.ndarray:
    foldwise = np.asarray(values)[:, cohort].astype(np.int32, copy=False)
    return np.concatenate((foldwise[:, 0], foldwise[:, 1] + CAL_FOLD_EPISODES), axis=1)


def _manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return pb21k._array_manifest(arrays)


def _manifest_sha256(manifest: Mapping[str, Mapping[str, object]], stage: str) -> str:
    encoded = json.dumps(
        {name: dict(manifest[name]) for name in sorted(manifest)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return sha256(f"IRPB21M{stage}MANIFEST\x01".encode("ascii") + encoded).hexdigest()


def publish_evidence_create_only(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    stage: str,
    attempt_sha256: str,
) -> dict[str, object]:
    expected = {"CAL": CAL_EVIDENCE_KEYS, "DEV": DEV_EVIDENCE_KEYS,
                "PREFLIGHT": PREFLIGHT_EVIDENCE_KEYS}.get(stage)
    if expected is None or set(arrays) != expected:
        raise ValueError(f"PB21M {stage} evidence key set drifted")
    canonical = {name: _canonical_array(value) for name, value in arrays.items()}
    contract = (expected_cal_evidence_contract() if stage == "CAL"
                else expected_dev_evidence_contract() if stage == "DEV" else None)
    if contract is not None:
        for name, item in contract.items():
            if list(canonical[name].shape) != item["shape"] or canonical[name].dtype.str != item["dtype"]:
                raise ValueError(f"PB21M {stage} evidence contract drifted: {name}")
    manifest = _manifest(canonical)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"PB21M {stage} evidence exists; retry is forbidden")
    temporary = path.with_name(f".{path.name}.{attempt_sha256[:16]}.partial")
    if temporary.exists():
        raise FileExistsError(f"PB21M {stage} partial evidence exists")
    with zipfile.ZipFile(temporary, mode="x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=True) as archive:
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
        "path": str(path), "sha256": _sha256_file(path), "byte_length": path.stat().st_size,
        "format": "deterministic_ZIP_DEFLATED_NPZ_v1", "stage": stage,
        "allowed_key_set_sha256": _key_set_sha256(expected, stage), "arrays": manifest,
        "array_manifest_sha256": _manifest_sha256(manifest, stage),
        "published_create_only_atomically": True,
    }


def reload_evidence(
    path: Path, publication: Mapping[str, object], *, stage: str,
) -> dict[str, np.ndarray]:
    expected = {"CAL": CAL_EVIDENCE_KEYS, "DEV": DEV_EVIDENCE_KEYS,
                "PREFLIGHT": PREFLIGHT_EVIDENCE_KEYS}.get(stage)
    if expected is None:
        raise ValueError("unknown PB21M evidence stage")
    if (_sha256_file(path) != publication.get("sha256")
            or path.stat().st_size != publication.get("byte_length")
            or publication.get("allowed_key_set_sha256") != _key_set_sha256(expected, stage)):
        raise ValueError(f"PB21M {stage} evidence publication envelope drifted")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != expected:
            raise ValueError(f"PB21M {stage} loaded key set drifted")
        arrays = {name: _canonical_array(loaded[name]) for name in loaded.files}
    contract = (expected_cal_evidence_contract() if stage == "CAL"
                else expected_dev_evidence_contract() if stage == "DEV" else None)
    if contract is not None:
        for name, item in contract.items():
            if list(arrays[name].shape) != item["shape"] or arrays[name].dtype.str != item["dtype"]:
                raise ValueError(f"PB21M {stage} reloaded dtype/shape drifted: {name}")
    manifest = _manifest(arrays)
    if (manifest != publication.get("arrays")
            or _manifest_sha256(manifest, stage) != publication.get("array_manifest_sha256")):
        raise ValueError(f"PB21M {stage} loaded manifest drifted")
    return arrays


def _pruning_mapping_sha256(mapping: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(mapping), sort_keys=True, separators=(",", ":")).encode("ascii")
    return sha256(b"IRPB21MPRUNING\x01" + encoded).hexdigest()


def build_cal_evidence_arrays(
    *,
    fit_tapes: Mapping[str, pb21j.FreshLiveTape],
    cal_tapes: Mapping[str, pb21j.FreshLiveTape],
    training: Mapping[str, Mapping[str, object]],
    pruning: Mapping[str, Mapping[str, object]],
    raw: np.ndarray,
    crossfit: Sequence[PerActionAffineHazardCalibrator],
    final: Sequence[PerActionAffineHazardCalibrator],
    oof_logits: np.ndarray,
    oof_probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    if raw.shape != (len(SCORERS), len(CELLS), 2, ROOTS_PER_CAL_FOLD, ACTION_COUNT):
        raise ValueError("CAL raw-logit axes drifted")
    expected_calibrators = len(SCORERS) * len(CELLS) * 2
    if len(crossfit) != expected_calibrators or len(final) != len(SCORERS) * len(CELLS):
        raise ValueError("CAL calibrator axes drifted")
    fit_index = np.asarray([FIT_COHORTS.index(cell.split("/", 1)[0]) for cell in CELLS], dtype=np.int16)
    fit_targets = np.stack([fit_tapes[label].hazard_targets for label in FIT_COHORTS])
    fit_actions = np.stack([fit_tapes[label].factual_actions for label in FIT_COHORTS])
    cal_targets: list[np.ndarray] = []
    cal_actions: list[np.ndarray] = []
    cal_roots: list[np.ndarray] = []
    cal_ordinals: list[np.ndarray] = []
    cal_ids: list[np.ndarray] = []
    for cohort in range(len(FIT_COHORTS)):
        tapes = (cal_tapes[CAL_A_COHORTS[cohort]], cal_tapes[CAL_B_COHORTS[cohort]])
        targets = np.concatenate([value.hazard_targets for value in tapes])
        actions = np.concatenate([value.factual_actions for value in tapes])
        roots = _byte_strings(tuple(item for value in tapes for item in value.root_state_ids), 64)
        groups = tuple(item for value in tapes for item in value.episode_group_ids)
        ordered = tuple(dict.fromkeys(groups))
        lookup = {value: index for index, value in enumerate(ordered)}
        ordinal = np.asarray([lookup[value] for value in groups], dtype=np.int32)
        cal_targets.append(targets); cal_actions.append(actions); cal_roots.append(roots)
        cal_ordinals.append(ordinal); cal_ids.append(_byte_strings(ordered, 64))
    flat_crossfit = tuple(crossfit)
    flat_final = tuple(final)
    provenance = _calibrator_provenance_arrays(flat_crossfit)
    final_provenance = {
        f"final_{name}": value
        for name, value in _calibrator_provenance_arrays(flat_final).items()
    }
    def pass_value(scorer: str, cell: str, index: int, key: str) -> float:
        record = training[cell][scorer]["pass_records"][index]
        if key == "mean_factual_component_BCE" and key not in record:
            return 0.0
        if key == "mean_objective" and key not in record:
            return float(record["mean_all_action_BCE"])
        return float(record[key])

    source_labels = (*FIT_COHORTS, *CAL_A_COHORTS, *CAL_B_COHORTS)
    source_tapes = {**fit_tapes, **cal_tapes}
    manifests = partition_manifests()
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.uint16),
        "scorer_ids": _byte_strings(SCORERS, 12), "fit_ids": _byte_strings(FIT_COHORTS, 2),
        "cal_fold_ids": _byte_strings(CAL_FOLDS, 1), "cell_ids": _byte_strings(CELLS, 16),
        "cell_fit_index": fit_index,
        "cell_init_seed": np.asarray([int(cell.rsplit("-", 1)[1]) for cell in CELLS], dtype=np.int32),
        "fit_targets": fit_targets, "fit_actions": fit_actions,
        "fit_root_ids": np.stack([_byte_strings(fit_tapes[label].root_state_ids, 64) for label in FIT_COHORTS]),
        "fit_group_ids": np.stack([_byte_strings(fit_tapes[label].episode_group_ids, 64) for label in FIT_COHORTS]),
        "factual_action_counts": np.stack([pb21l.factual_action_counts(fit_actions[index]) for index in fit_index]),
        "factual_balance_weights_float32": np.stack([
            pb21l.consumed_factual_balance_weights(fit_actions[index]) for index in fit_index
        ]),
        "training_initial_state_sha256": np.asarray([
            [training[cell][scorer]["initial_state_sha256"] for cell in CELLS] for scorer in SCORERS
        ], dtype="S64"),
        "training_final_state_sha256": np.asarray([
            [training[cell][scorer]["final_state_sha256"] for cell in CELLS] for scorer in SCORERS
        ], dtype="S64"),
        "training_steps": np.asarray([
            [training[cell][scorer]["optimizer_steps"] for cell in CELLS] for scorer in SCORERS
        ], dtype=np.int32),
        "fit_NZ_feature_sha256": np.asarray([
            _array_sha256(_features(fit_tapes[label])) for label in FIT_COHORTS
        ], dtype="S64"),
        "training_pass_number": np.asarray([[[
            int(training[cell][scorer]["pass_records"][index]["pass"])
            for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS], dtype=np.int16),
        "training_pass_mean_all_BCE": np.asarray([[[
            pass_value(scorer, cell, index, "mean_all_action_BCE")
            for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS]),
        "training_pass_mean_factual_component_BCE": np.asarray([[[
            pass_value(scorer, cell, index, "mean_factual_component_BCE")
            for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS]),
        "training_pass_mean_objective": np.asarray([[[
            pass_value(scorer, cell, index, "mean_objective")
            for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS]),
        "training_pass_maximum_preclip_gradient_norm": np.asarray([[[
            pass_value(scorer, cell, index, "maximum_preclip_gradient_norm")
            for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS]),
        "training_pass_permutation_sha256": np.asarray([[[
            training[cell][scorer]["pass_records"][index]["permutation_sha256"]
            for index in range(PASSES)] for cell in CELLS] for scorer in SCORERS], dtype="S64"),
        "pruned_state_sha256": np.asarray([
            [pruning[cell][scorer]["mapping"]["pruned_state_sha256"] for cell in CELLS]
            for scorer in SCORERS
        ], dtype="S64"),
        "pruning_active_columns": np.asarray(pb21j.ACTIVE_COLUMNS[ARM_NZ], dtype=np.int16),
        "pruning_parameter_count": np.asarray([[
            pruning[cell][scorer]["mapping"]["parameter_count"] for cell in CELLS
        ] for scorer in SCORERS], dtype=np.int32),
        "pruning_padded_logit_sha256": np.asarray([[
            pruning[cell][scorer]["FIT_inference_transfer"]["padded_logit_sha256"] for cell in CELLS
        ] for scorer in SCORERS], dtype="S64"),
        "pruning_pruned_logit_sha256": np.asarray([[
            pruning[cell][scorer]["FIT_inference_transfer"]["pruned_logit_sha256"] for cell in CELLS
        ] for scorer in SCORERS], dtype="S64"),
        "pruning_maximum_absolute_difference": np.asarray([[
            pruning[cell][scorer]["FIT_inference_transfer"]["maximum_absolute_difference"]
            for cell in CELLS] for scorer in SCORERS], dtype=np.float64),
        "source_partition_ids": _byte_strings(source_labels, 3),
        "source_dataset_manifest_sha256": np.asarray([manifests[label] for label in source_labels], dtype="S64"),
        "source_targets_sha256": np.asarray([_array_sha256(source_tapes[label].hazard_targets) for label in source_labels], dtype="S64"),
        "source_actions_sha256": np.asarray([_array_sha256(source_tapes[label].factual_actions) for label in source_labels], dtype="S64"),
        "source_root_ids_ordered_sha256": np.asarray([_array_sha256(_byte_strings(source_tapes[label].root_state_ids, 64)) for label in source_labels], dtype="S64"),
        "source_group_ids_ordered_sha256": np.asarray([_array_sha256(_byte_strings(source_tapes[label].episode_group_ids, 64)) for label in source_labels], dtype="S64"),
        "pruning_mapping_sha256": np.asarray([
            [_pruning_mapping_sha256(pruning[cell][scorer]["mapping"]) for cell in CELLS]
            for scorer in SCORERS
        ], dtype="S64"),
        "cal_targets": np.stack(cal_targets), "cal_actions": np.stack(cal_actions),
        "cal_root_ids": np.stack(cal_roots), "cal_cluster_ordinal": np.stack(cal_ordinals),
        "cal_cluster_ids": np.stack(cal_ids), "cal_raw_logits": raw,
        "crossfit_fit_fold_index": np.asarray((0, 1), dtype=np.int16),
        "crossfit_eval_fold_index": np.asarray((1, 0), dtype=np.int16),
        "oof_row_eval_fold_index": np.repeat(
            np.asarray((0, 1), dtype=np.int16), ROOTS_PER_CAL_FOLD
        ),
        "crossfit_scales": np.asarray([value.scales for value in flat_crossfit], dtype=np.float64).reshape(len(SCORERS), len(CELLS), 2, ACTION_COUNT),
        "crossfit_biases": np.asarray([value.biases for value in flat_crossfit], dtype=np.float64).reshape(len(SCORERS), len(CELLS), 2, ACTION_COUNT),
        "crossfit_accepted": np.asarray([value.accepted for value in flat_crossfit], dtype=np.bool_).reshape(len(SCORERS), len(CELLS), 2),
        "crossfit_report_observations": np.asarray([[report.observations for report in value.per_action] for value in flat_crossfit], dtype=np.int32).reshape(2, 9, 2, 5),
        "crossfit_report_positives": np.asarray([[report.positives for report in value.per_action] for value in flat_crossfit], dtype=np.int32).reshape(2, 9, 2, 5),
        "crossfit_report_negatives": np.asarray([[report.negatives for report in value.per_action] for value in flat_crossfit], dtype=np.int32).reshape(2, 9, 2, 5),
        "crossfit_report_before_BCE": np.asarray([[report.before_bce for report in value.per_action] for value in flat_crossfit], dtype=np.float64).reshape(2, 9, 2, 5),
        "crossfit_report_after_BCE": np.asarray([[report.after_bce for report in value.per_action] for value in flat_crossfit], dtype=np.float64).reshape(2, 9, 2, 5),
        "crossfit_report_accepted": np.asarray([[report.accepted for report in value.per_action] for value in flat_crossfit], dtype=np.bool_).reshape(2, 9, 2, 5),
        "final_scales": np.asarray([value.scales for value in flat_final], dtype=np.float64).reshape(len(SCORERS), len(CELLS), ACTION_COUNT),
        "final_biases": np.asarray([value.biases for value in flat_final], dtype=np.float64).reshape(len(SCORERS), len(CELLS), ACTION_COUNT),
        "final_accepted": np.asarray([value.accepted for value in flat_final], dtype=np.bool_).reshape(len(SCORERS), len(CELLS)),
        "final_report_observations": np.asarray([[report.observations for report in value.per_action] for value in flat_final], dtype=np.int32).reshape(2, 9, 5),
        "final_report_positives": np.asarray([[report.positives for report in value.per_action] for value in flat_final], dtype=np.int32).reshape(2, 9, 5),
        "final_report_negatives": np.asarray([[report.negatives for report in value.per_action] for value in flat_final], dtype=np.int32).reshape(2, 9, 5),
        "final_report_before_BCE": np.asarray([[report.before_bce for report in value.per_action] for value in flat_final], dtype=np.float64).reshape(2, 9, 5),
        "final_report_after_BCE": np.asarray([[report.after_bce for report in value.per_action] for value in flat_final], dtype=np.float64).reshape(2, 9, 5),
        "final_report_accepted": np.asarray([[report.accepted for report in value.per_action] for value in flat_final], dtype=np.bool_).reshape(2, 9, 5),
        "final_dataset_manifest_sha256": np.asarray(
            [value.dataset_manifest_sha256 for value in flat_final], dtype="S64"
        ).reshape(len(SCORERS), len(CELLS)),
        "oof_calibrated_logits": oof_logits, "oof_calibrated_probabilities": oof_probabilities,
        "bootstrap_indices": cal_bootstrap_indices(),
        "episode_derangements": cal_episode_derangements(),
        **provenance,
        **final_provenance,
    }
    return {name: _canonical_array(value) for name, value in arrays.items()}


def validate_cal_evidence(
    arrays: Mapping[str, np.ndarray],
    *,
    source_bundle_sha256: str,
) -> None:
    expected_fit_index = np.repeat(np.arange(len(FIT_COHORTS), dtype=np.int16), len(INIT_SEEDS))
    expected_init_seed = np.tile(np.asarray(INIT_SEEDS, dtype=np.int32), len(FIT_COHORTS))
    if (int(arrays["schema_version"]) != SCHEMA_VERSION
            or _decode_strings(arrays["scorer_ids"]) != SCORERS
            or _decode_strings(arrays["fit_ids"]) != FIT_COHORTS
            or _decode_strings(arrays["cal_fold_ids"]) != CAL_FOLDS
            or _decode_strings(arrays["cell_ids"]) != CELLS
            or not np.array_equal(arrays["cell_fit_index"], expected_fit_index)
            or not np.array_equal(arrays["cell_init_seed"], expected_init_seed)):
        raise ValueError("CAL evidence canonical axes drifted")
    for target_name in ("fit_targets", "cal_targets"):
        if not np.all((arrays[target_name] == 0.0) | (arrays[target_name] == 1.0)):
            raise ValueError(f"CAL evidence {target_name} must be exactly binary")
    for action_name in ("fit_actions", "cal_actions"):
        if not np.all((arrays[action_name] >= 0) & (arrays[action_name] < ACTION_COUNT)):
            raise ValueError(f"CAL evidence {action_name} is outside the semantic action range")
    expected_initial = np.asarray([
        _state_sha256(initialized_head(int(cell.rsplit("-", 1)[1]))) for cell in CELLS
    ], dtype="S64")
    if not np.array_equal(arrays["training_initial_state_sha256"],
                          np.broadcast_to(expected_initial, (len(SCORERS), len(CELLS)))):
        raise ValueError("CAL paired initialization hashes drifted from the seeded head constructor")
    expected_shapes = {
        "cal_targets": (3, ROOTS_PER_CAL, ACTION_COUNT),
        "cal_actions": (3, ROOTS_PER_CAL),
        "cal_raw_logits": (2, 9, 2, ROOTS_PER_CAL_FOLD, ACTION_COUNT),
        "crossfit_scales": (2, 9, 2, ACTION_COUNT),
        "oof_calibrated_probabilities": (2, 9, ROOTS_PER_CAL, ACTION_COUNT),
        "bootstrap_indices": (BOOTSTRAP_RESAMPLES, 3, 2, CAL_FOLD_EPISODES),
        "episode_derangements": (DERANGEMENT_REPETITIONS, 3, 2, CAL_FOLD_EPISODES),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"CAL evidence {name} shape drifted")
    for cohort in range(len(FIT_COHORTS)):
        if (len(set(bytes(value) for value in arrays["cal_root_ids"][cohort])) != ROOTS_PER_CAL
                or len(set(bytes(value) for value in arrays["cal_cluster_ids"][cohort]))
                != 2 * CAL_FOLD_EPISODES
                or any(int((arrays["cal_cluster_ordinal"][cohort] == group).sum())
                       != ROOTS_PER_EPISODE for group in range(2 * CAL_FOLD_EPISODES))):
            raise ValueError("CAL root/group identity geometry drifted")
    if arrays["bootstrap_indices"].dtype != np.dtype("<u2"):
        raise ValueError("CAL bootstrap indices must remain little-endian uint16")
    regenerated_bootstrap = cal_bootstrap_indices()
    if not np.array_equal(arrays["bootstrap_indices"], regenerated_bootstrap):
        raise ValueError("CAL bootstrap evidence differs from preregistered generator")
    del regenerated_bootstrap
    regenerated_derangements = cal_episode_derangements()
    if not np.array_equal(arrays["episode_derangements"], regenerated_derangements):
        raise ValueError("CAL derangement evidence differs from preregistered generator")
    identity = np.arange(CAL_FOLD_EPISODES, dtype=np.uint16)
    if np.any(arrays["episode_derangements"] == identity[None, None, None, :]):
        raise ValueError("CAL derangements contain a fixed point")
    if (not np.all(arrays["training_steps"] == STEPS_PER_HEAD)
            or not np.array_equal(arrays["training_pass_number"],
                                  np.broadcast_to(np.arange(1, PASSES + 1), (2, 9, PASSES)))):
        raise ValueError("CAL training schedule evidence drifted")
    expected_permutations = np.asarray([
        _array_sha256(value.numpy().astype(np.int64, copy=False))
        for value in deterministic_permutations()
    ], dtype="S64")
    if not np.array_equal(arrays["training_pass_permutation_sha256"],
                          np.broadcast_to(expected_permutations, (2, 9, PASSES))):
        raise ValueError("CAL training permutation lineage drifted")
    if (not np.array_equal(arrays["pruning_active_columns"],
                           np.asarray(pb21j.ACTIVE_COLUMNS[ARM_NZ], dtype=np.int16))
            or not np.all(arrays["pruning_parameter_count"] == pb21j.PRUNED_PARAMETER_COUNTS[ARM_NZ])
            or not np.all(arrays["pruning_maximum_absolute_difference"] == 0.0)):
        raise ValueError("CAL pruning transfer evidence drifted")
    for cell_index in range(len(CELLS)):
        cohort = int(arrays["cell_fit_index"][cell_index])
        expected_counts = pb21l.factual_action_counts(arrays["fit_actions"][cohort])
        if not np.array_equal(arrays["factual_action_counts"][cell_index], expected_counts):
            raise ValueError("CAL factual action counts drifted")
        expected_weights = pb21l.consumed_factual_balance_weights(arrays["fit_actions"][cohort])
        if not np.array_equal(arrays["factual_balance_weights_float32"][cell_index], expected_weights):
            raise ValueError("CAL factual balance weights drifted")
    for cohort in range(len(FIT_COHORTS)):
        if len(set(bytes(value) for value in arrays["fit_root_ids"][cohort])) != ROOTS_PER_FIT:
            raise ValueError("FIT root identity geometry drifted")
        fit_groups = arrays["fit_group_ids"][cohort]
        unique_fit_groups, fit_group_counts = np.unique(fit_groups, return_counts=True)
        if (len(unique_fit_groups) != FIT_EPISODES
                or not np.all(fit_group_counts == ROOTS_PER_EPISODE)):
            raise ValueError("FIT episode-group geometry drifted")
        fit_actions = arrays["fit_actions"][cohort]
        fit_targets = arrays["fit_targets"][cohort]
        for action in range(ACTION_COUNT):
            selected = fit_targets[fit_actions == action, action]
            positives = int(selected.sum()); negatives = int(len(selected) - positives)
            if len(selected) < 20 or positives < 2 or negatives < 2:
                raise ValueError("FIT factual-action BAL support is insufficient")
        try:
            priors = pb21l._fit_domain_priors(fit_targets, fit_actions)
        except (ValueError, FloatingPointError) as error:
            raise ValueError("FIT domain-prior construction failed") from error
        if any(
            np.asarray(priors[domain]).shape != (ACTION_COUNT,)
            or not np.isfinite(priors[domain]).all()
            or not np.all((np.asarray(priors[domain]) > 0.0)
                          & (np.asarray(priors[domain]) < 1.0))
            for domain in DOMAINS
        ):
            raise ValueError("FIT all/factual/complement priors are nonfinite or degenerate")
    source_labels = (*FIT_COHORTS, *CAL_A_COHORTS, *CAL_B_COHORTS)
    if (_decode_strings(arrays["source_partition_ids"]) != source_labels
            or _decode_strings(arrays["source_dataset_manifest_sha256"])
            != tuple(partition_manifests()[label] for label in source_labels)):
        raise ValueError("CAL source manifest axes drifted")
    expected_target_digests: list[str] = []
    expected_action_digests: list[str] = []
    expected_root_digests: list[str] = []
    expected_group_digests: list[str] = []
    for cohort in range(len(FIT_COHORTS)):
        expected_target_digests.append(_array_sha256(arrays["fit_targets"][cohort]))
        expected_action_digests.append(_array_sha256(arrays["fit_actions"][cohort]))
        expected_root_digests.append(_array_sha256(arrays["fit_root_ids"][cohort]))
        expected_group_digests.append(_array_sha256(arrays["fit_group_ids"][cohort]))
    for fold in range(2):
        start = fold * ROOTS_PER_CAL_FOLD; stop = start + ROOTS_PER_CAL_FOLD
        for cohort in range(len(FIT_COHORTS)):
            cluster_ids = tuple(bytes(value).rstrip(b"\x00").decode("utf-8")
                                for value in arrays["cal_cluster_ids"][cohort])
            group_rows = _byte_strings(tuple(
                cluster_ids[int(index)] for index in arrays["cal_cluster_ordinal"][cohort, start:stop]
            ), 64)
            expected_target_digests.append(_array_sha256(arrays["cal_targets"][cohort, start:stop]))
            expected_action_digests.append(_array_sha256(arrays["cal_actions"][cohort, start:stop]))
            expected_root_digests.append(_array_sha256(arrays["cal_root_ids"][cohort, start:stop]))
            expected_group_digests.append(_array_sha256(group_rows))
    for name, expected_values in (
        ("source_targets_sha256", expected_target_digests),
        ("source_actions_sha256", expected_action_digests),
        ("source_root_ids_ordered_sha256", expected_root_digests),
        ("source_group_ids_ordered_sha256", expected_group_digests),
    ):
        if _decode_strings(arrays[name]) != tuple(expected_values):
            raise ValueError(f"CAL source ordered digest drifted: {name}")
    root_sets = [set(bytes(value) for value in arrays["fit_root_ids"][cohort])
                 for cohort in range(len(FIT_COHORTS))]
    group_sets = [set(bytes(value) for value in arrays["fit_group_ids"][cohort])
                  for cohort in range(len(FIT_COHORTS))]
    for fold in range(2):
        start = fold * ROOTS_PER_CAL_FOLD; stop = start + ROOTS_PER_CAL_FOLD
        for cohort in range(len(FIT_COHORTS)):
            root_sets.append(set(bytes(value) for value in arrays["cal_root_ids"][cohort, start:stop]))
            ids = arrays["cal_cluster_ids"][cohort, fold * CAL_FOLD_EPISODES:(fold + 1) * CAL_FOLD_EPISODES]
            group_sets.append(set(bytes(value) for value in ids))
    for collections, role in ((root_sets, "root"), (group_sets, "group")):
        for index, left in enumerate(collections):
            if any(left & right for right in collections[index + 1:]):
                raise ValueError(f"CAL/FIT partition {role} IDs overlap")
    feature_digests = _decode_strings(arrays["fit_NZ_feature_sha256"])
    if len(feature_digests) != len(FIT_COHORTS) or any(
        len(bytes.fromhex(value)) != 32 for value in feature_digests
    ):
        raise ValueError("FIT NZ feature digest evidence drifted")
    for name, value in arrays.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"CAL evidence contains non-finite values: {name}")
    expected_probability = 1.0 / (1.0 + np.exp(-np.clip(arrays["oof_calibrated_logits"], -700, 700)))
    if not np.array_equal(expected_probability, arrays["oof_calibrated_probabilities"]):
        raise ValueError("CAL probability transform drifted")
    for scorer in range(len(SCORERS)):
        for cell in range(len(CELLS)):
            cohort = cell // len(INIT_SEEDS)
            pb21l.mask_partition_identity(
                arrays["cal_targets"][cohort], arrays["oof_calibrated_probabilities"][scorer, cell],
                arrays["cal_actions"][cohort],
            )
    if (not np.array_equal(arrays["crossfit_fit_fold_index"], np.asarray((0, 1), dtype=np.int16))
            or not np.array_equal(arrays["crossfit_eval_fold_index"], np.asarray((1, 0), dtype=np.int16))
            or not np.array_equal(arrays["oof_row_eval_fold_index"],
                                  np.repeat(np.asarray((0, 1), dtype=np.int16), ROOTS_PER_CAL_FOLD))):
        raise ValueError("CAL cross-fit direction axes drifted")
    scales = arrays["crossfit_scales"]
    biases = arrays["crossfit_biases"]
    raw = arrays["cal_raw_logits"]
    expected_a = raw[:, :, 0] * scales[:, :, 1, None, :] + biases[:, :, 1, None, :]
    expected_b = raw[:, :, 1] * scales[:, :, 0, None, :] + biases[:, :, 0, None, :]
    expected_oof = np.concatenate((expected_a, expected_b), axis=2)
    if not np.array_equal(expected_oof, arrays["oof_calibrated_logits"]):
        raise ValueError("CAL OOF predictions do not exactly respect fit-half/eval-half direction")
    decoded_bundles = _decode_strings(arrays["calibration_source_bundle_sha256"])
    if any(value != source_bundle_sha256 for value in decoded_bundles):
        raise ValueError("CAL source-bundle provenance drifted")
    decoded_final_bundles = _decode_strings(arrays["final_calibration_source_bundle_sha256"])
    if any(value != source_bundle_sha256 for value in decoded_final_bundles):
        raise ValueError("final CAL source-bundle provenance drifted")
    manifests = partition_manifests()
    expected_roles = tuple(role for _ in SCORERS for _ in CELLS for role in ("A", "B"))
    namespaces = _decode_strings(arrays["calibration_source_namespace"])
    algorithms = _decode_strings(arrays["calibration_partition_algorithm"])
    if any(not namespace.endswith(f".{role}.v1") for namespace, role in zip(namespaces, expected_roles, strict=True)):
        raise ValueError("crossfit provenance role order drifted")
    if any(not algorithm.endswith(f";calibration_role={role}")
           for algorithm, role in zip(algorithms, expected_roles, strict=True)):
        raise ValueError("crossfit partition-algorithm role order drifted")
    if any(value != "TRAIN" for value in _decode_strings(arrays["calibration_source_split"])):
        raise ValueError("crossfit source split must be TRAIN")
    if any(value != "TRAIN-CAL" for value in _decode_strings(arrays["calibration_source_partition"])):
        raise ValueError("crossfit source partition must be TRAIN-CAL")
    expected_manifests = tuple(
        manifests[(CAL_A_COHORTS if fold == 0 else CAL_B_COHORTS)[cell // 3]]
        for _ in SCORERS for cell in range(len(CELLS)) for fold in range(2)
    )
    if _decode_strings(arrays["calibration_dataset_manifest_sha256"]) != expected_manifests:
        raise ValueError("crossfit CAL manifest direction drifted")
    final_namespaces = _decode_strings(arrays["final_calibration_source_namespace"])
    final_algorithms = _decode_strings(arrays["final_calibration_partition_algorithm"])
    if (any(not value.endswith(".A+B.v1") for value in final_namespaces)
            or any(not value.endswith(";calibration_role=A+B") for value in final_algorithms)
            or any(value != "TRAIN-CAL" for value in _decode_strings(
                arrays["final_calibration_source_partition"]
            ))):
        raise ValueError("final A+B calibrator provenance drifted")
    expected_final_manifests = tuple(
        combined_cal_manifest_sha256(manifests[CAL_A_COHORTS[cell // 3]],
                                     manifests[CAL_B_COHORTS[cell // 3]])
        for _ in SCORERS for cell in range(len(CELLS))
    )
    if _decode_strings(arrays["final_calibration_dataset_manifest_sha256"]) != expected_final_manifests:
        raise ValueError("final combined CAL manifest drifted")
    if not np.array_equal(
        arrays["final_dataset_manifest_sha256"],
        np.asarray(expected_final_manifests, dtype="S64").reshape(2, 9),
    ):
        raise ValueError("final combined CAL manifest axis drifted")


def deterministic_refit_from_reloaded_cal(
    arrays: Mapping[str, np.ndarray],
    *,
    manifests: Mapping[str, str],
    source_bundle_sha256: str,
) -> dict[str, object]:
    def decoded_row(values: np.ndarray) -> tuple[str, ...]:
        return tuple(bytes(value).rstrip(b"\x00").decode("utf-8") for value in values)

    def fit_from_arrays(
        raw: np.ndarray, targets: np.ndarray, root_groups: tuple[str, ...],
        fit_groups: tuple[str, ...], checkpoint_sha256: str, manifest_sha256: str,
        role: str,
    ) -> PerActionAffineHazardCalibrator:
        action_ids = np.broadcast_to(np.arange(ACTION_COUNT, dtype=np.int64), raw.shape)
        provenance = TrainCalibrationProvenance(
            source_namespace=f"maze_chase.pb21m.train-only.{role}.v1",
            source_split="TRAIN", source_partition="TRAIN-CAL",
            calibration_group_ids=tuple(group for group in root_groups for _ in range(ACTION_COUNT)),
            upstream_model_fit_group_ids=tuple(sorted(set(fit_groups))),
            upstream_checkpoint_sha256=checkpoint_sha256,
            dataset_manifest_sha256=manifest_sha256,
            source_bundle_sha256=source_bundle_sha256,
            partition_algorithm=f"{PARTITION_ALGORITHM};calibration_role={role}",
        )
        return fit_train_only_per_action_affine(
            torch.from_numpy(np.asarray(raw).reshape(-1)),
            torch.from_numpy(np.asarray(targets).reshape(-1)),
            torch.from_numpy(action_ids.reshape(-1)), provenance=provenance,
            action_count=ACTION_COUNT, l2_regularization=pb21j.CALIBRATION_L2,
            minimum_scale=pb21j.CALIBRATION_MINIMUM_SCALE,
            minimum_examples_per_action=pb21j.CALIBRATION_MINIMUM_EXAMPLES_PER_ACTION,
            minimum_class_examples=pb21j.CALIBRATION_MINIMUM_CLASS_EXAMPLES,
            max_iterations=pb21j.CALIBRATION_MAX_ITERATIONS,
            tolerance=pb21j.CALIBRATION_TOLERANCE, fit_mode="per_action_affine",
        )

    crossfit: list[PerActionAffineHazardCalibrator] = []
    final: list[PerActionAffineHazardCalibrator] = []
    for scorer_index, _scorer in enumerate(SCORERS):
        for cell_index, _cell in enumerate(CELLS):
            cohort = int(arrays["cell_fit_index"][cell_index])
            raw_a = arrays["cal_raw_logits"][scorer_index, cell_index, 0]
            raw_b = arrays["cal_raw_logits"][scorer_index, cell_index, 1]
            targets = arrays["cal_targets"][cohort]
            cluster_ids = decoded_row(arrays["cal_cluster_ids"][cohort])
            root_groups = tuple(cluster_ids[int(index)] for index in arrays["cal_cluster_ordinal"][cohort])
            fit_groups = decoded_row(arrays["fit_group_ids"][cohort])
            checkpoint = bytes(arrays["pruned_state_sha256"][scorer_index, cell_index]).decode("ascii")
            crossfit.append(fit_from_arrays(
                raw_a, targets[:ROOTS_PER_CAL_FOLD], root_groups[:ROOTS_PER_CAL_FOLD],
                fit_groups, checkpoint, manifests[CAL_A_COHORTS[cohort]], "A",
            ))
            crossfit.append(fit_from_arrays(
                raw_b, targets[ROOTS_PER_CAL_FOLD:], root_groups[ROOTS_PER_CAL_FOLD:],
                fit_groups, checkpoint, manifests[CAL_B_COHORTS[cohort]], "B",
            ))
            final.append(fit_from_arrays(
                np.concatenate((raw_a, raw_b)), targets, root_groups, fit_groups, checkpoint,
                combined_cal_manifest_sha256(manifests[CAL_A_COHORTS[cohort]],
                                             manifests[CAL_B_COHORTS[cohort]]), "A+B",
            ))
    observed = {
        "crossfit_scales": np.asarray([value.scales for value in crossfit]).reshape(2, 9, 2, 5),
        "crossfit_biases": np.asarray([value.biases for value in crossfit]).reshape(2, 9, 2, 5),
        "crossfit_accepted": np.asarray([value.accepted for value in crossfit]).reshape(2, 9, 2),
        "final_scales": np.asarray([value.scales for value in final]).reshape(2, 9, 5),
        "final_biases": np.asarray([value.biases for value in final]).reshape(2, 9, 5),
        "final_accepted": np.asarray([value.accepted for value in final]).reshape(2, 9),
    }
    for prefix, values, shape in (("crossfit", crossfit, (2, 9, 2, 5)),
                                  ("final", final, (2, 9, 5))):
        observed[f"{prefix}_report_observations"] = np.asarray([
            [report.observations for report in value.per_action] for value in values
        ], dtype=np.int32).reshape(shape)
        observed[f"{prefix}_report_positives"] = np.asarray([
            [report.positives for report in value.per_action] for value in values
        ], dtype=np.int32).reshape(shape)
        observed[f"{prefix}_report_negatives"] = np.asarray([
            [report.negatives for report in value.per_action] for value in values
        ], dtype=np.int32).reshape(shape)
        observed[f"{prefix}_report_before_BCE"] = np.asarray([
            [report.before_bce for report in value.per_action] for value in values
        ], dtype=np.float64).reshape(shape)
        observed[f"{prefix}_report_after_BCE"] = np.asarray([
            [report.after_bce for report in value.per_action] for value in values
        ], dtype=np.float64).reshape(shape)
        observed[f"{prefix}_report_accepted"] = np.asarray([
            [report.accepted for report in value.per_action] for value in values
        ], dtype=np.bool_).reshape(shape)
    if any(not np.array_equal(arrays[name], value) for name, value in observed.items()):
        raise RuntimeError("deterministic AA refit from reloaded CAL evidence drifted")
    observed_crossfit_provenance = _calibrator_provenance_arrays(crossfit)
    observed_final_provenance = {
        f"final_{name}": value for name, value in _calibrator_provenance_arrays(final).items()
    }
    for name, value in {**observed_crossfit_provenance, **observed_final_provenance}.items():
        if not np.array_equal(arrays[name], value):
            raise RuntimeError(f"deterministic AA refit provenance drifted: {name}")
    return {"crossfit_refits": len(crossfit), "final_refits": len(final),
            "parameters_and_acceptance_byte_identical": True, "passed": True}


def build_dev_evidence_arrays(
    *,
    cal_arrays: Mapping[str, np.ndarray],
    cal_evidence_sha256: str,
    cal_decision_sha256: str,
    dev_open_sha256: str,
    dev_tape: pb21j.FreshLiveTape,
    raw_logits: np.ndarray,
    calibrated_logits: np.ndarray,
    calibrated_probabilities: np.ndarray,
    latency_measurements_ms: np.ndarray,
) -> dict[str, np.ndarray]:
    ordinal, groups = _cluster_geometry(dev_tape)
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.uint16),
        "cal_evidence_sha256": np.asarray(cal_evidence_sha256.encode("ascii"), dtype="S64"),
        "cal_decision_sha256": np.asarray(cal_decision_sha256.encode("ascii"), dtype="S64"),
        "dev_open_sha256": np.asarray(dev_open_sha256.encode("ascii"), dtype="S64"),
        "scorer_ids": cal_arrays["scorer_ids"], "cell_ids": cal_arrays["cell_ids"],
        "cell_fit_index": cal_arrays["cell_fit_index"], "cell_init_seed": cal_arrays["cell_init_seed"],
        "fit_targets": cal_arrays["fit_targets"], "fit_actions": cal_arrays["fit_actions"],
        "dev_targets": dev_tape.hazard_targets, "dev_actions": dev_tape.factual_actions,
        "dev_root_ids": _byte_strings(dev_tape.root_state_ids, 64), "dev_cluster_ordinal": ordinal,
        "dev_cluster_ids": _byte_strings(groups, 64), "final_scales": cal_arrays["final_scales"],
        "dev_dataset_manifest_sha256": np.asarray(
            partition_manifests()["DEV"].encode("ascii"), dtype="S64"
        ),
        "dev_targets_sha256": np.asarray(_array_sha256(dev_tape.hazard_targets).encode("ascii"), dtype="S64"),
        "dev_actions_sha256": np.asarray(_array_sha256(dev_tape.factual_actions).encode("ascii"), dtype="S64"),
        "dev_root_ids_ordered_sha256": np.asarray(
            _array_sha256(_byte_strings(dev_tape.root_state_ids, 64)).encode("ascii"), dtype="S64"
        ),
        "dev_group_ids_ordered_sha256": np.asarray(
            _array_sha256(_byte_strings(dev_tape.episode_group_ids, 64)).encode("ascii"), dtype="S64"
        ),
        "final_biases": cal_arrays["final_biases"], "final_accepted": cal_arrays["final_accepted"],
        "dev_raw_logits": raw_logits,
        "dev_raw_probabilities": 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -700, 700))),
        "dev_calibrated_logits": calibrated_logits,
        "dev_calibrated_probabilities": calibrated_probabilities,
        "latency_measurements_ms": latency_measurements_ms,
        "bootstrap_indices": dev_bootstrap_indices(),
        "episode_derangements": dev_episode_derangements(),
    }
    return {name: _canonical_array(value) for name, value in arrays.items()}


def validate_dev_evidence(
    arrays: Mapping[str, np.ndarray], *, cal_arrays: Mapping[str, np.ndarray],
    project_root: Path, attempt_sha256: str, cal_evidence_sha256: str,
) -> None:
    validate_canonical_cal_evidence_file(
        project_root, expected_sha256=cal_evidence_sha256,
    )
    cal_decision = validate_cal_decision_receipt(
        project_root, attempt_sha256=attempt_sha256,
        cal_evidence_sha256=cal_evidence_sha256,
    )
    recomputed_cal_decision = evaluate_stage(cal_arrays, stage="CAL")
    if (cal_decision["payload"]["passed"] is not True
            or cal_decision["payload"]["decision"] != recomputed_cal_decision
            or cal_decision["payload"]["decision_sha256"]
            != _decision_sha256(recomputed_cal_decision)):
        raise ValueError("DEV lineage does not bind an exact passing authoritative CAL decision")
    dev_open = validate_dev_open_receipt(
        project_root, attempt_sha256=attempt_sha256,
        cal_decision_sha256=cal_decision["sha256"],
    )
    if bytes(arrays["cal_evidence_sha256"]).rstrip(b"\x00").decode("ascii") != cal_evidence_sha256:
        raise ValueError("DEV evidence does not bind authoritative CAL evidence")
    if bytes(arrays["cal_decision_sha256"]).rstrip(b"\x00").decode("ascii") != cal_decision["sha256"]:
        raise ValueError("DEV evidence does not bind sealed CAL decision")
    if bytes(arrays["dev_open_sha256"]).rstrip(b"\x00").decode("ascii") != dev_open["sha256"]:
        raise ValueError("DEV evidence does not bind DEV-open receipt")
    for name in ("scorer_ids", "cell_ids", "cell_fit_index", "cell_init_seed",
                 "fit_targets", "fit_actions", "final_scales", "final_biases",
                 "final_accepted"):
        if not np.array_equal(arrays[name], cal_arrays[name]):
            raise ValueError(f"DEV {name} is not an exact CAL-evidence cross-link")
    expected = (len(SCORERS), len(CELLS), DEV_ROOTS, ACTION_COUNT)
    for name in ("dev_raw_logits", "dev_raw_probabilities", "dev_calibrated_logits",
                 "dev_calibrated_probabilities"):
        if arrays[name].shape != expected:
            raise ValueError(f"DEV evidence {name} shape drifted")
    if (int(arrays["schema_version"]) != SCHEMA_VERSION
            or _decode_strings(arrays["scorer_ids"]) != SCORERS
            or _decode_strings(arrays["cell_ids"]) != CELLS
            or arrays["dev_targets"].shape != (DEV_ROOTS, ACTION_COUNT)
            or arrays["dev_actions"].shape != (DEV_ROOTS,)
            or arrays["bootstrap_indices"].shape != (BOOTSTRAP_RESAMPLES, DEV_EPISODES)
            or arrays["episode_derangements"].shape != (DERANGEMENT_REPETITIONS, DEV_EPISODES)):
        raise ValueError("DEV frozen axes drifted")
    if not np.all((arrays["dev_targets"] == 0.0) | (arrays["dev_targets"] == 1.0)):
        raise ValueError("DEV targets must be exactly binary")
    if not np.all((arrays["dev_actions"] >= 0) & (arrays["dev_actions"] < ACTION_COUNT)):
        raise ValueError("DEV actions are outside the semantic action range")
    if arrays["bootstrap_indices"].dtype != np.dtype("<u2"):
        raise ValueError("DEV bootstrap must remain little-endian uint16")
    regenerated_bootstrap = dev_bootstrap_indices()
    if not np.array_equal(arrays["bootstrap_indices"], regenerated_bootstrap):
        raise ValueError("DEV bootstrap differs from preregistered generator")
    del regenerated_bootstrap
    regenerated_derangements = dev_episode_derangements()
    if not np.array_equal(arrays["episode_derangements"], regenerated_derangements):
        raise ValueError("DEV derangements differ from preregistered generator")
    if np.any(arrays["episode_derangements"] == np.arange(DEV_EPISODES, dtype=np.uint16)[None, :]):
        raise ValueError("DEV derangements contain a fixed point")
    raw_probability = 1.0 / (1.0 + np.exp(-np.clip(arrays["dev_raw_logits"], -700, 700)))
    calibrated_probability = 1.0 / (1.0 + np.exp(-np.clip(arrays["dev_calibrated_logits"], -700, 700)))
    if (not np.array_equal(raw_probability, arrays["dev_raw_probabilities"])
            or not np.array_equal(calibrated_probability, arrays["dev_calibrated_probabilities"])):
        raise ValueError("DEV probability transform drifted")
    expected_logits = (
        arrays["dev_raw_logits"] * arrays["final_scales"][:, :, None, :]
        + arrays["final_biases"][:, :, None, :]
    )
    if not np.array_equal(expected_logits, arrays["dev_calibrated_logits"]):
        raise ValueError("DEV logits are not exact reloaded-final-AA transforms")
    if (len(set(_decode_strings(arrays["dev_root_ids"]))) != DEV_ROOTS
            or len(_decode_strings(arrays["dev_cluster_ids"])) != DEV_EPISODES
            or len(set(_decode_strings(arrays["dev_cluster_ids"]))) != DEV_EPISODES
            or any(int((arrays["dev_cluster_ordinal"] == group).sum()) != ROOTS_PER_EPISODE
                   for group in range(DEV_EPISODES))):
        raise ValueError("DEV source identity/cluster geometry drifted")
    cluster_ids = _decode_strings(arrays["dev_cluster_ids"])
    ordered_group_rows = _byte_strings(tuple(
        cluster_ids[int(index)] for index in arrays["dev_cluster_ordinal"]
    ), 64)
    expected_lineage = {
        "dev_dataset_manifest_sha256": partition_manifests()["DEV"],
        "dev_targets_sha256": _array_sha256(arrays["dev_targets"]),
        "dev_actions_sha256": _array_sha256(arrays["dev_actions"]),
        "dev_root_ids_ordered_sha256": _array_sha256(arrays["dev_root_ids"]),
        "dev_group_ids_ordered_sha256": _array_sha256(ordered_group_rows),
    }
    for name, expected_value in expected_lineage.items():
        observed = bytes(arrays[name]).rstrip(b"\x00").decode("ascii")
        if observed != expected_value:
            raise ValueError(f"DEV source lineage drifted: {name}")
    train_roots = set(bytes(value) for value in cal_arrays["fit_root_ids"].reshape(-1))
    train_roots.update(bytes(value) for value in cal_arrays["cal_root_ids"].reshape(-1))
    train_groups = set(bytes(value) for value in cal_arrays["fit_group_ids"].reshape(-1))
    train_groups.update(bytes(value) for value in cal_arrays["cal_cluster_ids"].reshape(-1))
    if (train_roots & set(bytes(value) for value in arrays["dev_root_ids"])
            or train_groups & set(bytes(value) for value in arrays["dev_cluster_ids"])):
        raise ValueError("DEV source overlaps FIT/CAL identity namespace")
    for scorer in range(len(SCORERS)):
        for cell in range(len(CELLS)):
            pb21l.mask_partition_identity(
                arrays["dev_targets"], arrays["dev_calibrated_probabilities"][scorer, cell],
                arrays["dev_actions"],
            )
    if (arrays["latency_measurements_ms"].shape != (2, 9, 256)
            or not np.isfinite(arrays["latency_measurements_ms"]).all()
            or np.any(arrays["latency_measurements_ms"] < 0.0)):
        raise ValueError("DEV latency evidence geometry drifted")


def _episode_means(values: np.ndarray, ordinal: np.ndarray, groups: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    result = np.asarray([data[ordinal == group].mean(axis=0) for group in range(groups)])
    if any(int((ordinal == group).sum()) != ROOTS_PER_EPISODE for group in range(groups)):
        raise ValueError("episode mean geometry drifted")
    return result


def _bootstrap_draws(episode_values: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    episodes = np.asarray(episode_values, dtype=np.float64)
    indices = np.asarray(sampled)
    if indices.ndim == 2:
        indices = indices[:, None, :]
        episodes = episodes.reshape(1, indices.shape[2], *episodes.shape[1:])
    elif indices.ndim == 3:
        episodes = episodes.reshape(indices.shape[1], indices.shape[2], *episodes.shape[1:])
    else:
        raise ValueError("bootstrap indices must be resample x [fold x] episode")
    result = np.empty((len(indices), *episodes.shape[2:]), dtype=np.float64)
    for start in range(0, len(indices), 1024):
        stop = min(start + 1024, len(indices))
        block = indices[start:stop].astype(np.int64, copy=False)
        gathered = np.stack(
            [episodes[fold][block[:, fold]] for fold in range(indices.shape[1])], axis=1
        )
        result[start:stop] = gathered.mean(axis=(1, 2))
    return result


def _proper_report(
    fit_targets: np.ndarray,
    fit_actions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    ordinal: np.ndarray,
    sampled: np.ndarray,
    *,
    domain: str,
) -> dict[str, float]:
    priors = pb21l._fit_domain_priors(fit_targets, fit_actions)[domain]
    baseline = np.broadcast_to(priors, targets.shape)
    model_loss = pb21l._per_root_domain_losses(targets, probabilities, actions, domain)
    base_loss = pb21l._per_root_domain_losses(targets, baseline, actions, domain)
    episode = _episode_means(
        base_loss - model_loss, ordinal, sampled.shape[1] * sampled.shape[2]
    )
    draws = _bootstrap_draws(episode, sampled)
    return {
        "BCE": float(episode[:, 0].mean()),
        "BCE_LCB": float(np.quantile(draws[:, 0], ONE_SIDED_ALPHA, method=QUANTILE_METHOD)),
        "Brier": float(episode[:, 1].mean()),
        "Brier_LCB": float(np.quantile(draws[:, 1], ONE_SIDED_ALPHA, method=QUANTILE_METHOD)),
    }


def _support_report(targets: np.ndarray, actions: np.ndarray, *, domain: str) -> dict[str, object]:
    mask = pb21l._domain_mask(actions, domain)
    reports: list[dict[str, object]] = []
    for action in range(ACTION_COUNT):
        rows = mask[:, action]
        selected = np.asarray(targets)[rows, action]
        positives = int(selected.sum()); negatives = int(len(selected) - positives)
        checks = {"observations_at_least_20": len(selected) >= 20,
                  "positives_at_least_2": positives >= 2,
                  "negatives_at_least_2": negatives >= 2}
        reports.append({"action_id": action, "observations": len(selected),
                        "positives": positives, "negatives": negatives,
                        "checks": checks, "passed": all(checks.values())})
    return {"domain": domain, "per_action": reports,
            "passed": all(value["passed"] for value in reports)}


def _root_derangements(
    episode_derangement: np.ndarray, ordinal: np.ndarray, *, expected_clusters: int | None = None,
) -> np.ndarray:
    episodes = np.asarray(episode_derangement, dtype=np.int64)
    ordinal = np.asarray(ordinal, dtype=np.int64)
    if episodes.ndim != 2 or ordinal.ndim != 1:
        raise ValueError("PB21M derangement arrays have invalid rank")
    groups = episodes.shape[1]
    if expected_clusters is not None and groups != expected_clusters:
        raise ValueError("PB21M derangement cluster count drifted")
    if set(np.unique(ordinal).tolist()) != set(range(groups)):
        raise ValueError("PB21M cluster ordinals must be contiguous and complete")
    rows = tuple(np.flatnonzero(ordinal == group) for group in range(groups))
    if any(len(value) != ROOTS_PER_EPISODE for value in rows):
        raise ValueError("PB21M derangement cluster geometry drifted")
    donors = np.empty((len(episodes), len(ordinal)), dtype=np.int32)
    for repetition, permutation in enumerate(episodes):
        if sorted(permutation.tolist()) != list(range(groups)) or np.any(permutation == np.arange(groups)):
            raise ValueError("PB21M episode donor is not a fixed-point-free bijection")
        for target_group, donor_group in enumerate(permutation):
            donors[repetition, rows[target_group]] = rows[int(donor_group)]
    return donors


def causal_derangement_report(
    targets: np.ndarray, raw_scores: np.ndarray, probabilities: np.ndarray,
    actions: np.ndarray, ordinal: np.ndarray, episode_derangement: np.ndarray,
    *, domain: str,
) -> dict[str, object]:
    donors = _root_derangements(episode_derangement, ordinal)
    mask = pb21l._domain_mask(actions, domain)
    real_bce = pb21k._binary_bce(targets[mask], probabilities[mask])
    real_auc = pb21j._score_auc(targets[mask], raw_scores[mask])
    real_per_action = [pb21j._score_auc(targets[mask[:, action], action],
                                        raw_scores[mask[:, action], action])
                       for action in range(ACTION_COUNT)]
    shuffled_bce: list[float] = []
    shuffled_auc: list[float] = []
    shuffled_per_action: list[list[float]] = []
    for donor in donors:
        shuffled_bce.append(pb21k._binary_bce(targets[mask], probabilities[donor][mask]))
        shuffled_auc.append(pb21j._score_auc(targets[mask], raw_scores[donor][mask]))
        shuffled_per_action.append([
            pb21j._score_auc(targets[mask[:, action], action],
                             raw_scores[donor][mask[:, action], action])
            for action in range(ACTION_COUNT)
        ])
    median_bce = float(np.median(shuffled_bce)); median_auc = float(np.median(shuffled_auc))
    per_action_drop = [
        real_per_action[action] - float(np.median([value[action] for value in shuffled_per_action]))
        for action in range(ACTION_COUNT)
    ]
    checks = {
        "calibrated_shuffled_BCE_ratio": median_bce / real_bce >= pb21j.MINIMUM_SHUFFLED_BCE_RATIO,
        "raw_aggregate_AUC_drop": real_auc - median_auc >= pb21j.MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP,
        "every_action_raw_AUC_drop": all(
            value >= pb21j.MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP for value in per_action_drop
        ),
    }
    return {"domain": domain, "algorithm": "PB21M_geometry_generic_fold_restricted_episode_derangement_v1",
            "episode_donor_index_sha256": _array_sha256(episode_derangement),
            "root_donor_index_sha256": _array_sha256(donors), "real_calibrated_BCE": real_bce,
            "median_shuffled_calibrated_BCE": median_bce,
            "calibrated_shuffled_BCE_ratio": median_bce / real_bce,
            "real_raw_AUC": real_auc, "median_shuffled_raw_AUC": median_auc,
            "raw_aggregate_AUC_drop": real_auc - median_auc,
            "per_action_raw_AUC_drop": per_action_drop, "checks": checks,
            "passed": all(checks.values())}


def _absolute_cell_gate(
    *,
    fit_targets: np.ndarray,
    fit_actions: np.ndarray,
    targets: np.ndarray,
    actions: np.ndarray,
    ordinal: np.ndarray,
    raw_logits: np.ndarray,
    calibrated_logits: np.ndarray,
    probabilities: np.ndarray,
    sampled: np.ndarray,
    derangements: np.ndarray,
    calibrator_accepted: bool,
) -> dict[str, object]:
    support_reports = {
        domain: _support_report(targets, actions, domain=domain) for domain in DOMAINS
    }
    if not all(report["passed"] for report in support_reports.values()):
        return {
            "domains": {domain: {"support": report, "passed": False}
                        for domain, report in support_reports.items()},
            "checks": {"AA_accepted": bool(calibrator_accepted),
                       "support_precheck": False, "metric_and_ranking_evaluation_completed": False},
            "evaluation_skipped": "insufficient_per_action_class_support",
            "passed": False,
        }
    priors = pb21l._fit_domain_priors(fit_targets, fit_actions)
    raw_probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -700, 700)))
    domains: dict[str, object] = {}
    for domain in DOMAINS:
        metrics = pb21l._domain_metric_report(targets, probabilities, actions,
                                               domain=domain, priors=priors[domain])
        proper = _proper_report(fit_targets, fit_actions, targets, probabilities, actions,
                                ordinal, sampled, domain=domain)
        causal = causal_derangement_report(
            targets, raw_logits, probabilities, actions, ordinal, derangements, domain=domain
        )
        support = support_reports[domain]
        proper_passed = all(proper[name] > 0.0 for name in ("BCE", "BCE_LCB", "Brier", "Brier_LCB"))
        domains[domain] = {"metrics": metrics, "proper_score": proper, "causal": causal,
                           "support": support,
                           "passed": metrics["passed"] and proper_passed
                           and causal["passed"] and support["passed"]}
    raw_metrics = binary_probability_metrics(targets.reshape(-1), raw_probabilities.reshape(-1),
                                               ece_bins=pb21k.ECE_BINS).as_dict()
    calibrated_metrics = domains["all_action"]["metrics"]["aggregate"]
    prior_table = np.broadcast_to(priors["all_action"], targets.shape)
    prior_metrics = binary_probability_metrics(targets.reshape(-1), prior_table.reshape(-1),
                                                ece_bins=pb21k.ECE_BINS).as_dict()
    raw_ranking = pb21j.score_ranking_gate(targets, raw_logits, prior_auc=float(prior_metrics["roc_auc"]))
    calibrated_ranking = pb21j.score_ranking_gate(
        targets, calibrated_logits, prior_auc=float(prior_metrics["roc_auc"])
    )
    checks = {
        "AA_accepted": bool(calibrator_accepted),
        "calibrated_BCE_not_worse_than_raw": calibrated_metrics["bce"] <= raw_metrics["bce"],
        "all_action_BCE_prior_ratio": calibrated_metrics["bce"] <= pb21j.MAX_BASELINE_BCE_RATIO * prior_metrics["bce"],
        "all_action_Brier_prior_ratio": calibrated_metrics["brier"] <= pb21j.MAX_BASELINE_BRIER_RATIO * prior_metrics["brier"],
        "raw_ranking": raw_ranking["passed"], "calibrated_ranking": calibrated_ranking["passed"],
        "all_domains": all(value["passed"] for value in domains.values()),
    }
    return {"domains": domains, "checks": checks, "passed": all(checks.values())}


def _direct_report(
    *,
    targets_by_cohort: np.ndarray,
    actions_by_cohort: np.ndarray,
    probabilities: np.ndarray,
    ordinals: np.ndarray,
    sampled: np.ndarray,
    cell_fit_index: np.ndarray,
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    passed = True
    for domain in DOMAINS:
        cell_episodes: list[np.ndarray] = []
        cell_reports: dict[str, object] = {}
        cohort_reports: dict[str, object] = {}
        for cell_index, cell in enumerate(CELLS):
            cohort = int(cell_fit_index[cell_index])
            base = pb21l._per_root_domain_losses(
                targets_by_cohort[cohort], probabilities[0, cell_index], actions_by_cohort[cohort], domain
            )
            candidate = pb21l._per_root_domain_losses(
                targets_by_cohort[cohort], probabilities[1, cell_index], actions_by_cohort[cohort], domain
            )
            episode = _episode_means(
                base - candidate, ordinals[cohort], sampled.shape[2] * sampled.shape[3]
            )
            cell_episodes.append(episode)
            draws = _bootstrap_draws(episode, sampled[:, cohort])
            threshold_bce = 0.0 if domain == "factual" else -AA_BCE_MARGIN
            threshold_brier = 0.0 if domain == "factual" else -AA_BRIER_MARGIN
            point = episode.mean(axis=0)
            lower = np.quantile(draws, ONE_SIDED_ALPHA, axis=0, method=QUANTILE_METHOD)
            checks = {"BCE": point[0] > threshold_bce and lower[0] > threshold_bce,
                      "Brier": point[1] > threshold_brier and lower[1] > threshold_brier}
            fold_reports: dict[str, object] = {}
            if sampled.shape[2] == 2:
                root_delta = base - candidate
                for fold, fold_name in enumerate(CAL_FOLDS):
                    start = fold * ROOTS_PER_CAL_FOLD; stop = start + ROOTS_PER_CAL_FOLD
                    fold_point = root_delta[start:stop].mean(axis=0)
                    fold_checks = {"BCE": fold_point[0] > threshold_bce,
                                   "Brier": fold_point[1] > threshold_brier}
                    fold_reports[fold_name] = {
                        "point": {"BCE": float(fold_point[0]), "Brier": float(fold_point[1])},
                        "checks": fold_checks, "passed": all(fold_checks.values()),
                    }
                checks["each_heldout_fold_point"] = all(
                    value["passed"] for value in fold_reports.values()
                )
            cell_reports[cell] = {"point": {"BCE": float(point[0]), "Brier": float(point[1])},
                                  "LCB": {"BCE": float(lower[0]), "Brier": float(lower[1])},
                                  "heldout_fold_points": fold_reports,
                                  "checks": checks, "passed": all(checks.values())}
        cohort_draws: list[np.ndarray] = []
        for cohort, fit_label in enumerate(FIT_COHORTS):
            cohort_cells = [cell_episodes[index] for index in range(len(CELLS))
                            if int(cell_fit_index[index]) == cohort]
            if len(cohort_cells) != len(INIT_SEEDS):
                raise ValueError("direct-report cell/FIT axis drifted")
            episode = np.mean(np.stack(cohort_cells), axis=0)
            draws = _bootstrap_draws(episode, sampled[:, cohort])
            cohort_draws.append(draws)
            lower = np.quantile(draws, ONE_SIDED_ALPHA, axis=0, method=QUANTILE_METHOD)
            threshold = np.asarray((0.0, 0.0) if domain == "factual" else (-AA_BCE_MARGIN, -AA_BRIER_MARGIN))
            point = episode.mean(axis=0)
            checks = {"BCE": point[0] > threshold[0] and lower[0] > threshold[0],
                      "Brier": point[1] > threshold[1] and lower[1] > threshold[1]}
            cohort_reports[fit_label] = {"point": {"BCE": float(point[0]), "Brier": float(point[1])},
                                         "LCB": {"BCE": float(lower[0]), "Brier": float(lower[1])},
                                         "checks": checks, "passed": all(checks.values())}
        grand_draws = np.mean(np.stack(cohort_draws), axis=0)
        grand_point = np.mean(np.stack(cell_episodes), axis=(0, 1))
        grand_lower = np.quantile(grand_draws, ONE_SIDED_ALPHA, axis=0, method=QUANTILE_METHOD)
        threshold = np.asarray((0.0, 0.0) if domain == "factual" else (-AA_BCE_MARGIN, -AA_BRIER_MARGIN))
        grand_checks = {"BCE": grand_point[0] > threshold[0] and grand_lower[0] > threshold[0],
                        "Brier": grand_point[1] > threshold[1] and grand_lower[1] > threshold[1]}
        domain_passed = (all(value["passed"] for value in cell_reports.values())
                         and all(value["passed"] for value in cohort_reports.values())
                         and all(grand_checks.values()))
        comparisons[domain] = {
            "direction": "BASE_minus_BAL-UPMIX", "cell": cell_reports, "cohort": cohort_reports,
            "grand": {"point": {"BCE": float(grand_point[0]), "Brier": float(grand_point[1])},
                      "LCB": {"BCE": float(grand_lower[0]), "Brier": float(grand_lower[1])},
                      "checks": grand_checks}, "passed": domain_passed,
        }
        passed = passed and domain_passed
    return {"comparisons": comparisons, "passed": passed}


def evaluate_stage(arrays: Mapping[str, np.ndarray], *, stage: str) -> dict[str, object]:
    if stage == "CAL":
        targets = arrays["cal_targets"]; actions = arrays["cal_actions"]
        ordinals = arrays["cal_cluster_ordinal"]
        raw = arrays["cal_raw_logits"].reshape(
            len(SCORERS), len(CELLS), ROOTS_PER_CAL, ACTION_COUNT
        )
        probabilities = arrays["oof_calibrated_probabilities"]
        calibrated_logits = arrays["oof_calibrated_logits"]
        crossfit_accepted = np.all(arrays["crossfit_accepted"], axis=2)
        final_accepted = arrays["final_accepted"]
        accepted = crossfit_accepted
        sampled = arrays["bootstrap_indices"]
        derangements = np.stack([
            _flatten_cal_derangements(arrays["episode_derangements"], cohort)
            for cohort in range(len(FIT_COHORTS))
        ])
    elif stage == "DEV":
        targets = np.broadcast_to(arrays["dev_targets"], (3, DEV_ROOTS, ACTION_COUNT))
        actions = np.broadcast_to(arrays["dev_actions"], (3, DEV_ROOTS))
        ordinals = np.broadcast_to(arrays["dev_cluster_ordinal"], (3, DEV_ROOTS))
        raw = arrays["dev_raw_logits"]; probabilities = arrays["dev_calibrated_probabilities"]
        calibrated_logits = arrays["dev_calibrated_logits"]
        final_accepted = arrays["final_accepted"]
        crossfit_accepted = None
        accepted = final_accepted
        shared = arrays["bootstrap_indices"]
        sampled = np.broadcast_to(shared[:, None, None, :], (len(shared), 3, 1, shared.shape[1]))
        derangements = np.broadcast_to(arrays["episode_derangements"],
                                       (3, *arrays["episode_derangements"].shape))
    else:
        raise ValueError("stage must be CAL or DEV")
    absolute: dict[str, dict[str, object]] = {scorer: {} for scorer in SCORERS}
    for scorer_index, scorer in enumerate(SCORERS):
        for cell_index, cell in enumerate(CELLS):
            cohort = cell_index // len(INIT_SEEDS)
            absolute[scorer][cell] = _absolute_cell_gate(
                fit_targets=arrays["fit_targets"][cohort], fit_actions=arrays["fit_actions"][cohort],
                targets=targets[cohort], actions=actions[cohort], ordinal=ordinals[cohort],
                raw_logits=raw[scorer_index, cell_index],
                calibrated_logits=calibrated_logits[scorer_index, cell_index],
                probabilities=probabilities[scorer_index, cell_index],
                sampled=sampled[:, cohort], derangements=derangements[cohort],
                calibrator_accepted=bool(accepted[scorer_index, cell_index]),
            )
    direct = _direct_report(targets_by_cohort=targets, actions_by_cohort=actions,
                            probabilities=probabilities, ordinals=ordinals, sampled=sampled,
                            cell_fit_index=arrays["cell_fit_index"])
    control_checks = {
        cell: bool(accepted[0, cell_index])
        and np.isfinite(raw[0, cell_index]).all()
        and np.isfinite(probabilities[0, cell_index]).all()
        for cell_index, cell in enumerate(CELLS)
    }
    if stage == "CAL":
        aa_acceptance = {
            "crossfit_by_scorer_cell": crossfit_accepted.tolist(),
            "final_by_scorer_cell": final_accepted.tolist(),
            "all_crossfit_accepted": bool(np.all(crossfit_accepted)),
            "all_final_accepted": bool(np.all(final_accepted)),
            "passed": bool(np.all(crossfit_accepted) and np.all(final_accepted)),
        }
    else:
        aa_acceptance = {
            "final_by_scorer_cell": final_accepted.tolist(),
            "all_final_accepted": bool(np.all(final_accepted)),
            "passed": bool(np.all(final_accepted)),
        }
    passed = (all(value["passed"] for value in absolute[SCORER_BAL].values())
              and all(control_checks.values()) and aa_acceptance["passed"] and direct["passed"])
    return _json_builtin({
        "stage": stage, "absolute_gates": absolute, "direct_BAL_vs_BASE": direct,
        "AA_acceptance": aa_acceptance,
        "BASE_paired_control_finite_and_AA_accepted": control_checks,
        "BASE_absolute_deployability_required": False, "passed": passed,
    })


def production_cal_pipeline_preflight(project_root: Path) -> dict[str, object]:
    """Exercise the real CAL schema/builder/writer/reload/refit/evaluator at reduced size."""
    roots = 100; roots_per_episode = 5; episodes = 20
    resamples = 8; repetitions = 4; source_bundle = "7" * 64
    manifests = {spec.label: f"{index + 1:064x}" for index, spec in enumerate(PARTITION_SPECS)}
    bootstrap = np.random.default_rng(12_345).integers(
        0, episodes, size=(resamples, 3, 2, episodes), dtype=np.uint16
    )
    derangements = np.empty((repetitions, 3, 2, episodes), dtype=np.uint16)
    identity = np.arange(episodes, dtype=np.uint16)
    for repetition in range(repetitions):
        for cohort in range(3):
            for fold in range(2):
                derangements[repetition, cohort, fold] = np.roll(
                    identity, 1 + (repetition + cohort + fold) % (episodes - 1)
                )

    def tape(label: str, offset: int) -> pb21j.FreshLiveTape:
        row = np.arange(roots, dtype=np.int64)
        action = np.arange(ACTION_COUNT, dtype=np.int64)
        targets = ((row[:, None] + 2 * action[None, :] + offset) % 4 < 2).astype(np.float64)
        updater = np.zeros((roots, strict.CAPACITY_STATE_WIDTH), dtype=np.float64)
        updater[:, 365] = 1.0
        return pb21j.FreshLiveTape(
            beliefs=np.zeros((roots, pb21j.WIDTH), dtype=np.float64), updater_inputs=updater,
            hazard_targets=targets, parent_raw_logits=np.zeros((roots, ACTION_COUNT), dtype=np.float64),
            factual_actions=((row + offset) % ACTION_COUNT).astype(np.int64),
            episode_group_ids=tuple(f"{label}:episode:{index // roots_per_episode}" for index in range(roots)),
            root_state_ids=tuple(f"{label}:root:{index}" for index in range(roots)),
            prior_assignment_count=(row % 3).astype(np.int64),
            prior_disagreement_count=(row % 2).astype(np.int64),
        )

    lifecycle = canonical_paths(project_root)
    protected = tuple(lifecycle[name] for name in (
        "attempt", "cal_evidence", "cal_decision", "dev_open", "dev_evidence", "result"
    ))
    before = tuple(path.exists() for path in protected)
    patched_names = {
        "ROOTS_PER_EPISODE": roots_per_episode, "FIT_EPISODES": episodes,
        "CAL_FOLD_EPISODES": episodes, "ROOTS_PER_FIT": roots,
        "ROOTS_PER_CAL_FOLD": roots, "ROOTS_PER_CAL": 2 * roots,
        "PASSES": 2, "ROOT_BATCH_SIZE": 20, "STEPS_PER_HEAD": 10,
        "BOOTSTRAP_RESAMPLES": resamples, "DERANGEMENT_REPETITIONS": repetitions,
    }
    saved = {name: globals()[name] for name in patched_names}
    saved_helpers = (partition_manifests, cal_bootstrap_indices, cal_episode_derangements)
    saved_pb21l_roots = pb21l.ROOTS_PER_FIT
    try:
        globals().update(patched_names)
        pb21l.ROOTS_PER_FIT = roots
        globals()["partition_manifests"] = lambda: dict(manifests)
        globals()["cal_bootstrap_indices"] = lambda: bootstrap.copy()
        globals()["cal_episode_derangements"] = lambda: derangements.copy()
        fit_tapes = {label: tape(label, index) for index, label in enumerate(FIT_COHORTS)}
        cal_labels = (*CAL_A_COHORTS, *CAL_B_COHORTS)
        cal_tapes = {label: tape(label, 10 + index) for index, label in enumerate(cal_labels)}
        permutations = deterministic_permutations()
        permutation_hashes = tuple(_array_sha256(value.numpy().astype(np.int64, copy=False))
                                   for value in permutations)
        heads: dict[str, dict[str, pb21j.PrunedHazardHead]] = {}
        training: dict[str, dict[str, object]] = {}
        pruning: dict[str, dict[str, object]] = {}
        for cell_index, cell in enumerate(CELLS):
            heads[cell] = {}; training[cell] = {}; pruning[cell] = {}
            initial_state = _state_sha256(initialized_head(int(cell.rsplit("-", 1)[1])))
            for scorer_index, scorer in enumerate(SCORERS):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(1_000 + 10 * cell_index + scorer_index)
                    head = pb21j.PrunedHazardHead(input_width=1)
                heads[cell][scorer] = head
                state = _state_sha256(head)
                training[cell][scorer] = {
                    "initial_state_sha256": initial_state,
                    "final_state_sha256": state, "optimizer_steps": 10,
                    "pass_records": [{
                        "pass": index + 1, "mean_all_action_BCE": 0.8 - 0.1 * index,
                        "mean_factual_component_BCE": 0.7 - 0.1 * index,
                        "mean_objective": 0.75 - 0.1 * index,
                        "maximum_preclip_gradient_norm": 1.0,
                        "permutation_sha256": permutation_hashes[index],
                    } for index in range(2)],
                }
                mapping = {"arm": ARM_NZ, "active_columns": list(pb21j.ACTIVE_COLUMNS[ARM_NZ]),
                           "parameter_count": pb21j.PRUNED_PARAMETER_COUNTS[ARM_NZ],
                           "pruned_state_sha256": state}
                pruning[cell][scorer] = {"mapping": mapping, "FIT_inference_transfer": {
                    "padded_logit_sha256": state, "pruned_logit_sha256": state,
                    "maximum_absolute_difference": 0.0}}
        raw = np.empty((2, 9, 2, roots, ACTION_COUNT), dtype=np.float64)
        crossfit: list[PerActionAffineHazardCalibrator] = []
        final: list[PerActionAffineHazardCalibrator] = []
        oof_logits = np.empty((2, 9, 2 * roots, ACTION_COUNT), dtype=np.float64)
        oof_probabilities = np.empty_like(oof_logits)
        for scorer_index, scorer in enumerate(SCORERS):
            for cell_index, cell in enumerate(CELLS):
                cohort = cell_index // 3; fit_label = FIT_COHORTS[cohort]
                tape_a = cal_tapes[CAL_A_COHORTS[cohort]]
                tape_b = cal_tapes[CAL_B_COHORTS[cohort]]
                for fold, fold_tape in enumerate((tape_a, tape_b)):
                    sign = 2.0 * fold_tape.hazard_targets - 1.0
                    raw[scorer_index, cell_index, fold] = (
                        sign * (0.8 + 0.1 * scorer_index)
                        + np.linspace(-0.15, 0.15, roots)[:, None] + 0.001 * cell_index
                    )
                fitted_a = _fit_calibrator(
                    raw[scorer_index, cell_index, 0], tape_a, fit_tapes[fit_label],
                    heads[cell][scorer], manifest_sha256=manifests[CAL_A_COHORTS[cohort]],
                    source_bundle_sha256=source_bundle, calibration_role="A")
                fitted_b = _fit_calibrator(
                    raw[scorer_index, cell_index, 1], tape_b, fit_tapes[fit_label],
                    heads[cell][scorer], manifest_sha256=manifests[CAL_B_COHORTS[cohort]],
                    source_bundle_sha256=source_bundle, calibration_role="B")
                crossfit.extend((fitted_a, fitted_b))
                logits_a, probability_a = _apply_calibrator(fitted_b, raw[scorer_index, cell_index, 0])
                logits_b, probability_b = _apply_calibrator(fitted_a, raw[scorer_index, cell_index, 1])
                oof_logits[scorer_index, cell_index] = np.concatenate((logits_a, logits_b))
                oof_probabilities[scorer_index, cell_index] = np.concatenate((probability_a, probability_b))
                combined = pb21j.FreshLiveTape(
                    beliefs=np.concatenate((tape_a.beliefs, tape_b.beliefs)),
                    updater_inputs=np.concatenate((tape_a.updater_inputs, tape_b.updater_inputs)),
                    hazard_targets=np.concatenate((tape_a.hazard_targets, tape_b.hazard_targets)),
                    parent_raw_logits=np.concatenate((tape_a.parent_raw_logits, tape_b.parent_raw_logits)),
                    factual_actions=np.concatenate((tape_a.factual_actions, tape_b.factual_actions)),
                    episode_group_ids=(*tape_a.episode_group_ids, *tape_b.episode_group_ids),
                    root_state_ids=(*tape_a.root_state_ids, *tape_b.root_state_ids),
                    prior_assignment_count=np.concatenate((tape_a.prior_assignment_count, tape_b.prior_assignment_count)),
                    prior_disagreement_count=np.concatenate((tape_a.prior_disagreement_count, tape_b.prior_disagreement_count)))
                final.append(_fit_calibrator(
                    np.concatenate((raw[scorer_index, cell_index, 0], raw[scorer_index, cell_index, 1])),
                    combined, fit_tapes[fit_label], heads[cell][scorer],
                    manifest_sha256=combined_cal_manifest_sha256(
                        manifests[CAL_A_COHORTS[cohort]], manifests[CAL_B_COHORTS[cohort]]),
                    source_bundle_sha256=source_bundle, calibration_role="A+B"))
        arrays = build_cal_evidence_arrays(
            fit_tapes=fit_tapes, cal_tapes=cal_tapes, training=training, pruning=pruning,
            raw=raw, crossfit=crossfit, final=final, oof_logits=oof_logits,
            oof_probabilities=oof_probabilities)
        with tempfile.TemporaryDirectory(prefix="pb21m-production-preflight-") as directory:
            evidence = Path(directory) / "cal-evidence.npz"
            publication = publish_evidence_create_only(
                evidence, arrays, stage="CAL", attempt_sha256="8" * 64)
            authoritative = reload_evidence(evidence, publication, stage="CAL")
        validate_cal_evidence(authoritative, source_bundle_sha256=source_bundle)
        refit = deterministic_refit_from_reloaded_cal(
            authoritative, manifests=manifests, source_bundle_sha256=source_bundle)
        decision = evaluate_stage(authoritative, stage="CAL")
        json.dumps(decision, allow_nan=False)
        after = tuple(path.exists() for path in protected)
        if after != before:
            raise RuntimeError("production CAL preflight touched canonical lifecycle paths")
        passed = len(crossfit) == 36 and len(final) == 18 and refit["passed"] is True
        return {"uses_full_CAL_evidence_schema": True, "real_crossfit_calibrators": len(crossfit),
                "real_final_calibrators": len(final),
                "builder_writer_reload_validate_NPZ_only_refit_real_evaluator": True,
                "allow_pickle": False, "canonical_lifecycle_paths_untouched": True,
                "evidence_sha256": publication["sha256"],
                "array_manifest_sha256": publication["array_manifest_sha256"],
                "decision_sha256": _decision_sha256(decision), "passed": passed}
    finally:
        globals().update(saved)
        pb21l.ROOTS_PER_FIT = saved_pb21l_roots
        (globals().__setitem__("partition_manifests", saved_helpers[0]),
         globals().__setitem__("cal_bootstrap_indices", saved_helpers[1]),
         globals().__setitem__("cal_episode_derangements", saved_helpers[2]))


def latency_probe(heads: Mapping[str, Mapping[str, pb21j.PrunedHazardHead]],
                  dev_tape: pb21j.FreshLiveTape) -> np.ndarray:
    features = torch.from_numpy(np.ascontiguousarray(
        pb21j.compact_features(_features(dev_tape), ARM_NZ)[:256], dtype=np.float32
    ))
    values = np.empty((len(SCORERS), len(CELLS), 256), dtype=np.float64)
    with torch.inference_mode():
        for scorer_index, scorer in enumerate(SCORERS):
            for cell_index, cell in enumerate(CELLS):
                head = heads[cell][scorer]
                for index in range(16):
                    head(features[index:index + 1])
                for index in range(256):
                    started = perf_counter_ns(); head(features[index:index + 1]); ended = perf_counter_ns()
                    values[scorer_index, cell_index, index] = (ended - started) / 1.0e6
    return values


def latency_report(values: np.ndarray) -> dict[str, object]:
    measurements = np.asarray(values, dtype=np.float64)
    p99 = float(np.quantile(measurements, 0.99, method=QUANTILE_METHOD))
    return {"scope": "pruned_hazard_head_batch1_CPU_single_thread",
            "not_end_to_end_live_latency": True, "human_speed_claimed": False,
            "source_order": "scorer_then_cell_then_first_256_canonical_DEV_roots",
            "warmup_per_head": 16, "timed_roots_per_head": 256,
            "samples": int(measurements.size), "median_ms": float(np.median(measurements)),
            "p99_ms": p99, "limit_ms": LATENCY_P99_LIMIT_MS,
            "operational_nomination_gate": True, "passed": p99 <= LATENCY_P99_LIMIT_MS}


def _publish_attempt(project_root: Path, registration: Mapping[str, object],
                     resource_snapshot: Mapping[str, object]) -> dict[str, object]:
    path = canonical_paths(project_root)["attempt"]
    if path.exists():
        raise FileExistsError("PB21M attempt already exists; retry is forbidden")
    payload = {
        "schema_version": 1, "mode": "pb21m_fresh_bal_cal_first_attempt_v1", "run_id": RUN_ID,
        "classification": "fresh_attempt_consumed_no_retry", "registration_sha256": registration["sha256"],
        "source_bundle_sha256": registration["payload"]["source_bundle"]["sha256"],
        "CAL_evidence_path": CANONICAL_CAL_EVIDENCE, "CAL_decision_path": CANONICAL_CAL_DECISION,
        "DEV_open_path": CANONICAL_DEV_OPEN, "DEV_evidence_path": CANONICAL_DEV_EVIDENCE,
        "result_path": CANONICAL_RESULT, "published_before_any_partition_source": True,
        "resource_snapshot": dict(resource_snapshot), "retry_allowed": False,
    }
    digest = development._publish_json_create_only(path, payload)
    return {"path": str(path), "sha256": digest, "payload": payload}


def _publish_result(path: Path, payload: Mapping[str, object]) -> None:
    development._publish_json_create_only(path, dict(payload))


def _decision_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(value), ensure_ascii=True, sort_keys=True,
                         separators=(",", ":")).encode("ascii")
    return sha256(b"IRPB21MCALDECISION\x01" + encoded).hexdigest()


def _reload_json_receipt(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
               allow_nan=False).encode("ascii")
    return {"path": str(path), "sha256": _sha256_file(path), "payload": payload}


def validate_cal_decision_receipt(
    project_root: Path, *, attempt_sha256: str, cal_evidence_sha256: str,
) -> dict[str, object]:
    path = canonical_paths(project_root)["cal_decision"]
    record = _reload_json_receipt(path)
    payload = record["payload"]
    decision = payload.get("decision")
    if (payload.get("schema_version") != 1
            or payload.get("mode") != "pb21m_cal_decision_receipt_v1"
            or payload.get("attempt_sha256") != attempt_sha256
            or payload.get("cal_evidence_sha256") != cal_evidence_sha256
            or payload.get("terminal_if_failed") is not True
            or not isinstance(decision, dict)
            or payload.get("decision_sha256") != _decision_sha256(decision)
            or payload.get("passed") is not decision.get("passed")):
        raise ValueError("sealed CAL decision receipt failed reload/hash/lineage validation")
    return record


def validate_dev_open_receipt(
    project_root: Path, *, attempt_sha256: str, cal_decision_sha256: str,
) -> dict[str, object]:
    path = canonical_paths(project_root)["dev_open"]
    record = _reload_json_receipt(path)
    payload = record["payload"]
    if (payload.get("schema_version") != 1
            or payload.get("mode") != "pb21m_dev_open_receipt_v1"
            or payload.get("attempt_sha256") != attempt_sha256
            or payload.get("cal_decision_sha256") != cal_decision_sha256
            or payload.get("published_before_DEV_partition_source") is not True):
        raise ValueError("sealed DEV-open receipt failed reload/hash/lineage validation")
    return record


def validate_canonical_cal_evidence_file(
    project_root: Path, *, expected_sha256: str,
) -> dict[str, object]:
    path = canonical_paths(project_root)["cal_evidence"]
    if not path.is_file():
        raise ValueError("canonical CAL evidence file is absent before DEV validation")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise ValueError("canonical CAL evidence file hash drifted before DEV validation")
    return {"path": str(path), "sha256": observed, "byte_length": path.stat().st_size}


def publish_cal_decision(
    project_root: Path,
    *,
    attempt_sha256: str,
    cal_evidence_sha256: str,
    decision: Mapping[str, object],
    resource_snapshot: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": 1, "mode": "pb21m_cal_decision_receipt_v1",
        "attempt_sha256": attempt_sha256, "cal_evidence_sha256": cal_evidence_sha256,
        "decision_sha256": _decision_sha256(decision), "passed": bool(decision["passed"]),
        "decision": dict(decision), "terminal_if_failed": True,
        "resource_snapshot": dict(resource_snapshot),
    }
    path = canonical_paths(project_root)["cal_decision"]
    development._publish_json_create_only(path, payload)
    return validate_cal_decision_receipt(
        project_root, attempt_sha256=attempt_sha256,
        cal_evidence_sha256=cal_evidence_sha256,
    )


def publish_dev_open(
    project_root: Path,
    *,
    attempt_sha256: str,
    cal_evidence_sha256: str,
    resource_snapshot: Mapping[str, object],
) -> dict[str, object]:
    cal_decision = validate_cal_decision_receipt(
        project_root, attempt_sha256=attempt_sha256,
        cal_evidence_sha256=cal_evidence_sha256,
    )
    if cal_decision["payload"]["passed"] is not True:
        raise RuntimeError("DEV cannot open without a passing sealed CAL decision")
    payload = {
        "schema_version": 1, "mode": "pb21m_dev_open_receipt_v1",
        "attempt_sha256": attempt_sha256,
        "cal_decision_sha256": cal_decision["sha256"],
        "published_before_DEV_partition_source": True,
        "resource_snapshot": dict(resource_snapshot),
    }
    path = canonical_paths(project_root)["dev_open"]
    development._publish_json_create_only(path, payload)
    return validate_dev_open_receipt(
        project_root, attempt_sha256=attempt_sha256,
        cal_decision_sha256=cal_decision["sha256"],
    )


def assert_partition_disjointness(tapes: Mapping[str, pb21j.FreshLiveTape]) -> dict[str, object]:
    labels = tuple(tapes)
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            if set(tapes[left].root_state_ids) & set(tapes[right].root_state_ids):
                raise RuntimeError(f"partition root-ID overlap: {left}/{right}")
            if set(tapes[left].episode_group_ids) & set(tapes[right].episode_group_ids):
                raise RuntimeError(f"partition group-ID overlap: {left}/{right}")
    return {"labels": list(labels), "pairwise_root_ID_disjoint": True,
            "pairwise_episode_group_ID_disjoint": True, "passed": True}


def pretraining_support_report(
    fit_tapes: Mapping[str, pb21j.FreshLiveTape],
    cal_tapes: Mapping[str, pb21j.FreshLiveTape],
) -> dict[str, object]:
    def tape_checks(
        tape: pb21j.FreshLiveTape, roots: int, episodes: int,
    ) -> dict[str, bool]:
        features = _features(tape)
        group_ids = np.asarray(tape.episode_group_ids)
        unique_groups, group_counts = np.unique(group_ids, return_counts=True)
        return {
            "root_geometry": len(tape.hazard_targets) == roots
            and tape.hazard_targets.shape == (roots, ACTION_COUNT)
            and tape.factual_actions.shape == (roots,),
            "targets_exactly_binary": bool(np.all(
                (tape.hazard_targets == 0.0) | (tape.hazard_targets == 1.0)
            )),
            "unique_root_identity_geometry": len(tape.root_state_ids) == roots
            and len(set(tape.root_state_ids)) == roots,
            "episode_group_geometry": len(tape.episode_group_ids) == roots
            and len(unique_groups) == episodes
            and bool(np.all(group_counts == ROOTS_PER_EPISODE)),
            "all_numeric_finite": all(np.isfinite(value).all() for value in (
                tape.beliefs, tape.updater_inputs, tape.hazard_targets, tape.parent_raw_logits,
                tape.prior_assignment_count, tape.prior_disagreement_count,
            )),
            "semantic_actions": bool(np.all((tape.factual_actions >= 0)
                                             & (tape.factual_actions < ACTION_COUNT))),
            "NZ_feature_geometry_and_finite": features.shape == (roots, SUPERSET_WIDTH)
            and bool(np.isfinite(features).all()),
        }

    fit: dict[str, object] = {}
    for label in FIT_COHORTS:
        tape = fit_tapes[label]
        actions = tape.factual_actions; targets = tape.hazard_targets
        checks: list[dict[str, object]] = []
        for action in range(ACTION_COUNT):
            selected = targets[actions == action, action]
            positives = int(selected.sum()); negatives = int(len(selected) - positives)
            passed = len(selected) >= 20 and positives >= 2 and negatives >= 2
            checks.append({"action_id": action, "observations": len(selected),
                           "positives": positives, "negatives": negatives,
                           "passed": passed})
        geometry = tape_checks(tape, ROOTS_PER_FIT, FIT_EPISODES)
        prior_checks: dict[str, object] = {}
        try:
            priors = pb21l._fit_domain_priors(targets, actions)
        except (ValueError, FloatingPointError) as error:
            for domain in DOMAINS:
                prior_checks[domain] = {
                    "values": None, "finite": False,
                    "strictly_between_zero_and_one": False, "passed": False,
                    "construction_failure": {"type": type(error).__name__, "message": str(error)},
                }
        else:
            for domain in DOMAINS:
                values = np.asarray(priors[domain], dtype=np.float64)
                finite = bool(values.shape == (ACTION_COUNT,) and np.isfinite(values).all())
                nondegenerate = bool(finite and np.all((values > 0.0) & (values < 1.0)))
                prior_checks[domain] = {
                    "values": values.tolist(), "finite": finite,
                    "strictly_between_zero_and_one": nondegenerate,
                    "passed": bool(finite and nondegenerate),
                }
        fit[label] = {"per_factual_action": checks, "geometry": geometry,
                      "domain_priors": prior_checks,
                      "passed": all(geometry.values())
                      and all(value["passed"] for value in checks)
                      and all(value["passed"] for value in prior_checks.values())}
    cal: dict[str, object] = {}
    for label in (*CAL_A_COHORTS, *CAL_B_COHORTS):
        tape = cal_tapes[label]
        reports = {domain: _support_report(tape.hazard_targets, tape.factual_actions, domain=domain)
                   for domain in DOMAINS}
        geometry = tape_checks(tape, ROOTS_PER_CAL_FOLD, CAL_FOLD_EPISODES)
        cal[label] = {"domains": reports, "geometry": geometry,
                      "passed": all(geometry.values())
                      and all(value["passed"] for value in reports.values())}
    return _json_builtin({"FIT": fit, "CAL_folds": cal,
                          "passed": all(value["passed"] for value in fit.values())
                          and all(value["passed"] for value in cal.values())})


def _run_impl(
    *,
    output: Path,
    registration: Mapping[str, object],
    attempt: Mapping[str, object],
    preloaded_parent: tuple[object, dict[str, object], dict[str, object]],
) -> None:
    if torch.get_num_threads() != 1:
        raise RuntimeError("PB21M run entered implementation with thread drift")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("PB21M must hide CUDA")
    root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(root)
    source_bundle = registration["payload"]["source_bundle"]
    if source_bundle != _source_bundle(root):
        raise RuntimeError("PB21M source bundle drifted")
    model, _, upstream = preloaded_parent
    model.to(torch.device("cpu")); parent_before = development._state_dict_sha256(model.state_dict())
    resource_snapshots: list[dict[str, object]] = []
    resource_snapshots.append(resource_guard(paths["cal_evidence"], phase="FIT_source_collection"))
    fit_sources = partition_sources(FIT_COHORTS)
    fit_tapes = {label: collect_fresh_live_tape(model, fit_sources[label], partition_label=label)
                 for label in FIT_COHORTS}
    resource_snapshots.append(resource_guard(paths["cal_decision"], phase="pretraining_CAL_support"))
    cal_labels = (*CAL_A_COHORTS, *CAL_B_COHORTS)
    cal_sources = partition_sources(cal_labels)
    cal_tapes = {label: collect_fresh_live_tape(model, cal_sources[label], partition_label=label)
                 for label in cal_labels}
    train_partition_disjointness = assert_partition_disjointness({**fit_tapes, **cal_tapes})
    support = pretraining_support_report(fit_tapes, cal_tapes)
    if not support["passed"]:
        parent_after_support = development._state_dict_sha256(model.state_dict())
        parent_state_support = {
            "before_sha256": parent_before, "after_sha256": parent_after_support,
            "byte_identical": parent_after_support == parent_before,
        }
        if not parent_state_support["byte_identical"]:
            raise RuntimeError("PB21M mutated the frozen parent during pretraining support checks")
        support_payload = {
            "schema_version": 1, "mode": "pb21m_pretraining_support_decision_v1",
            "attempt_sha256": attempt["sha256"], "passed": False,
            "terminal_if_failed": True, "support": support,
            "parent_state": parent_state_support,
            "resource_snapshot": resource_snapshots[-1],
        }
        support_sha256 = development._publish_json_create_only(paths["cal_decision"], support_payload)
        resource_snapshots.append(resource_guard(output, phase="support_negative_result_publication"))
        _publish_result(output, {
            "schema_version": SCHEMA_VERSION, "implementation_revision": IMPLEMENTATION_REVISION,
            "mode": MODE, "classification": "fresh_pretraining_support_negative_no_training_no_DEV",
            "qualification_claimed": False, "checkpoint_emitted": False,
            "registration": registration, "attempt": attempt, "upstream": upstream,
            "source_bundle": source_bundle, "pretraining_support": support,
            "TRAIN_partition_disjointness": train_partition_disjointness,
            "parent_state": parent_state_support,
            "CAL_authoritative_evidence": None,
            "CAL_decision_receipt": {"path": str(paths["cal_decision"]),
                                     "sha256": support_sha256, "payload": support_payload},
            "DEV_constructed": False, "resource_snapshots": resource_snapshots,
            "retry_allowed": False,
        })
        return
    resource_snapshots.append(resource_guard(paths["cal_evidence"], phase="FIT_training"))
    permutations = deterministic_permutations()
    padded: dict[str, dict[str, pb21j.MaskedSupersetHazardHead]] = {}
    training: dict[str, dict[str, object]] = {}
    fit_features = {label: _features(fit_tapes[label]) for label in FIT_COHORTS}
    for fit_label in FIT_COHORTS:
        for seed in INIT_SEEDS:
            cell = f"{fit_label}/init-{seed}"; padded[cell] = {}; training[cell] = {}
            initial = set()
            for scorer in SCORERS:
                head = initialized_head(seed); initial.add(_state_sha256(head))
                training[cell][scorer] = pb21l.train_upmix_head(
                    head, fit_features[fit_label], fit_tapes[fit_label].hazard_targets,
                    fit_tapes[fit_label].factual_actions, permutations, scorer=scorer,
                )
                padded[cell][scorer] = head
            if len(initial) != 1:
                raise RuntimeError("paired scorers did not share byte-identical initialization")
    pruned: dict[str, dict[str, pb21j.PrunedHazardHead]] = {}
    pruning: dict[str, dict[str, object]] = {}
    for cell in CELLS:
        fit_label = cell.split("/", 1)[0]; pruned[cell] = {}; pruning[cell] = {}
        for scorer in SCORERS:
            compact, mapping = pb21j.prune_head(padded[cell][scorer], ARM_NZ)
            audit = pb21j.pruning_equivalence_audit(
                padded[cell][scorer], compact, fit_features[fit_label], ARM_NZ
            )
            pruned[cell][scorer] = compact
            pruning[cell][scorer] = {"mapping": mapping, "FIT_inference_transfer": audit}
    del padded
    resource_snapshots.append(resource_guard(paths["cal_evidence"], phase="CAL_fitting"))
    manifests = partition_manifests()
    raw = np.empty((2, 9, 2, ROOTS_PER_CAL_FOLD, ACTION_COUNT), dtype=np.float64)
    crossfit: list[PerActionAffineHazardCalibrator] = []
    final: list[PerActionAffineHazardCalibrator] = []
    oof_logits = np.empty((2, 9, ROOTS_PER_CAL, ACTION_COUNT), dtype=np.float64)
    oof_probabilities = np.empty_like(oof_logits)
    for scorer_index, scorer in enumerate(SCORERS):
        for cell_index, cell in enumerate(CELLS):
            cohort = cell_index // 3; fit_label = FIT_COHORTS[cohort]
            fold_tapes = (cal_tapes[CAL_A_COHORTS[cohort]], cal_tapes[CAL_B_COHORTS[cohort]])
            for fold, tape in enumerate(fold_tapes):
                raw[scorer_index, cell_index, fold] = pb21j.raw_logit_table(
                    pruned[cell][scorer], pb21j.compact_features(_features(tape), ARM_NZ), batch_size=1
                )
            fitted_a = _fit_calibrator(raw[scorer_index, cell_index, 0], fold_tapes[0],
                                       fit_tapes[fit_label], pruned[cell][scorer],
                                       manifest_sha256=manifests[CAL_A_COHORTS[cohort]],
                                       source_bundle_sha256=source_bundle["sha256"],
                                       calibration_role="A")
            fitted_b = _fit_calibrator(raw[scorer_index, cell_index, 1], fold_tapes[1],
                                       fit_tapes[fit_label], pruned[cell][scorer],
                                       manifest_sha256=manifests[CAL_B_COHORTS[cohort]],
                                       source_bundle_sha256=source_bundle["sha256"],
                                       calibration_role="B")
            crossfit.extend((fitted_a, fitted_b))
            logits_a, probs_a = _apply_calibrator(fitted_b, raw[scorer_index, cell_index, 0])
            logits_b, probs_b = _apply_calibrator(fitted_a, raw[scorer_index, cell_index, 1])
            oof_logits[scorer_index, cell_index] = np.concatenate((logits_a, logits_b))
            oof_probabilities[scorer_index, cell_index] = np.concatenate((probs_a, probs_b))
            combined_tape = pb21j.FreshLiveTape(
                beliefs=np.concatenate([value.beliefs for value in fold_tapes]),
                updater_inputs=np.concatenate([value.updater_inputs for value in fold_tapes]),
                hazard_targets=np.concatenate([value.hazard_targets for value in fold_tapes]),
                parent_raw_logits=np.concatenate([value.parent_raw_logits for value in fold_tapes]),
                factual_actions=np.concatenate([value.factual_actions for value in fold_tapes]),
                episode_group_ids=tuple(item for value in fold_tapes for item in value.episode_group_ids),
                root_state_ids=tuple(item for value in fold_tapes for item in value.root_state_ids),
                prior_assignment_count=np.concatenate([value.prior_assignment_count for value in fold_tapes]),
                prior_disagreement_count=np.concatenate([value.prior_disagreement_count for value in fold_tapes]),
            )
            final.append(_fit_calibrator(
                np.concatenate((raw[scorer_index, cell_index, 0], raw[scorer_index, cell_index, 1])),
                combined_tape, fit_tapes[fit_label], pruned[cell][scorer],
                manifest_sha256=combined_cal_manifest_sha256(
                    manifests[CAL_A_COHORTS[cohort]], manifests[CAL_B_COHORTS[cohort]]
                ),
                source_bundle_sha256=source_bundle["sha256"], calibration_role="A+B",
            ))
    cal_arrays = build_cal_evidence_arrays(
        fit_tapes=fit_tapes, cal_tapes=cal_tapes, training=training, pruning=pruning,
        raw=raw, crossfit=crossfit, final=final, oof_logits=oof_logits,
        oof_probabilities=oof_probabilities,
    )
    resource_snapshots.append(resource_guard(paths["cal_evidence"], phase="CAL_evidence_publication"))
    cal_publication = publish_evidence_create_only(
        paths["cal_evidence"], cal_arrays, stage="CAL", attempt_sha256=attempt["sha256"]
    )
    del cal_arrays
    resource_snapshots.append(resource_guard(paths["cal_evidence"], phase="CAL_reload_refit_evaluation"))
    authoritative_cal = reload_evidence(paths["cal_evidence"], cal_publication, stage="CAL")
    validate_cal_evidence(authoritative_cal, source_bundle_sha256=source_bundle["sha256"])
    refit_validation = deterministic_refit_from_reloaded_cal(
        authoritative_cal, manifests=manifests, source_bundle_sha256=source_bundle["sha256"],
    )
    cal_decision = evaluate_stage(authoritative_cal, stage="CAL")
    parent_after_cal = development._state_dict_sha256(model.state_dict())
    parent_state_cal = {
        "before_sha256": parent_before, "after_sha256": parent_after_cal,
        "byte_identical": parent_after_cal == parent_before,
    }
    if not parent_state_cal["byte_identical"]:
        raise RuntimeError("PB21M mutated the frozen parent before the terminal CAL decision")
    resource_snapshots.append(resource_guard(paths["cal_decision"], phase="CAL_decision_publication"))
    cal_decision_receipt = publish_cal_decision(
        root, attempt_sha256=attempt["sha256"], cal_evidence_sha256=cal_publication["sha256"],
        decision=cal_decision, resource_snapshot=resource_snapshots[-1],
    )
    common_result = {
        "schema_version": SCHEMA_VERSION, "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": MODE, "classification": CLASSIFICATION, "qualification_claimed": False,
        "checkpoint_emitted": False, "registration": registration, "attempt": attempt,
        "scientific_parent": registration["payload"]["scientific_parent"], "upstream": upstream,
        "source_bundle": source_bundle, "schedule": schedule_record(), "objective": objective_contract(),
        "inference": inference_contract(), "CAL_authoritative_evidence": cal_publication,
        "CAL_decision": cal_decision, "CAL_decision_receipt": cal_decision_receipt,
        "CAL_deterministic_refit_validation": refit_validation,
        "TRAIN_partition_disjointness": train_partition_disjointness,
        "parent_state_at_CAL_terminal_boundary": parent_state_cal,
        "resource_snapshots": resource_snapshots,
    }
    if not cal_decision["passed"]:
        resource_snapshots.append(resource_guard(output, phase="CAL_negative_result_publication"))
        _publish_result(output, {
            **common_result, "classification": "fresh_CAL_futility_negative_no_DEV",
            "DEV_constructed": False, "DEV_authoritative_evidence": None,
            "selection": {"recipe_nominated_for_scaling": False, "reason": "CAL_futility_gate_failed"},
            "retry_allowed": False,
        })
        return
    if development._state_dict_sha256(model.state_dict()) != parent_before:
        raise RuntimeError("PB21M mutated the frozen parent before DEV construction")
    resource_snapshots.append(resource_guard(paths["dev_open"], phase="DEV_open"))
    dev_open = publish_dev_open(
        root, attempt_sha256=attempt["sha256"],
        cal_evidence_sha256=cal_publication["sha256"],
        resource_snapshot=resource_snapshots[-1],
    )
    dev_source = partition_sources(("DEV",))["DEV"]
    dev_tape = collect_fresh_live_tape(model, dev_source, partition_label="DEV")
    all_partition_disjointness = assert_partition_disjointness(
        {**fit_tapes, **cal_tapes, "DEV": dev_tape}
    )
    dev_raw = np.empty((2, 9, DEV_ROOTS, ACTION_COUNT), dtype=np.float64)
    dev_logits = np.empty_like(dev_raw); dev_probabilities = np.empty_like(dev_raw)
    for scorer_index, scorer in enumerate(SCORERS):
        for cell_index, cell in enumerate(CELLS):
            dev_raw[scorer_index, cell_index] = pb21j.raw_logit_table(
                pruned[cell][scorer], pb21j.compact_features(_features(dev_tape), ARM_NZ), batch_size=1
            )
            logits = (
                dev_raw[scorer_index, cell_index]
                * authoritative_cal["final_scales"][scorer_index, cell_index][None, :]
                + authoritative_cal["final_biases"][scorer_index, cell_index][None, :]
            )
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
            dev_logits[scorer_index, cell_index] = logits
            dev_probabilities[scorer_index, cell_index] = probabilities
    latency_measurements = latency_probe(pruned, dev_tape)
    dev_arrays = build_dev_evidence_arrays(
        cal_arrays=authoritative_cal, cal_evidence_sha256=cal_publication["sha256"],
        cal_decision_sha256=cal_decision_receipt["sha256"], dev_open_sha256=dev_open["sha256"],
        dev_tape=dev_tape, raw_logits=dev_raw, calibrated_logits=dev_logits,
        calibrated_probabilities=dev_probabilities,
        latency_measurements_ms=latency_measurements,
    )
    resource_snapshots.append(resource_guard(paths["dev_evidence"], phase="DEV_evidence_publication"))
    dev_publication = publish_evidence_create_only(
        paths["dev_evidence"], dev_arrays, stage="DEV", attempt_sha256=attempt["sha256"]
    )
    del dev_arrays
    resource_snapshots.append(resource_guard(paths["dev_evidence"], phase="DEV_reload_evaluation"))
    authoritative_dev = reload_evidence(paths["dev_evidence"], dev_publication, stage="DEV")
    validate_dev_evidence(
        authoritative_dev, cal_arrays=authoritative_cal,
        project_root=root, attempt_sha256=attempt["sha256"],
        cal_evidence_sha256=cal_publication["sha256"],
    )
    dev_decision = evaluate_stage(authoritative_dev, stage="DEV")
    latency = latency_report(authoritative_dev["latency_measurements_ms"])
    nominated = bool(dev_decision["passed"] and latency["passed"])
    parent_after = development._state_dict_sha256(model.state_dict())
    if parent_after != parent_before:
        raise RuntimeError("PB21M mutated the frozen parent")
    resource_snapshots.append(resource_guard(output, phase="DEV_result_publication"))
    _publish_result(output, {
        **common_result, "classification": ("fresh_recipe_nominated_for_scaling_not_checkpoint"
                                             if nominated else "fresh_DEV_negative_not_candidate"),
        "DEV_constructed": True, "DEV_authoritative_evidence": dev_publication,
        "DEV_open_receipt": dev_open, "all_partition_disjointness": all_partition_disjointness,
        "DEV_decision": dev_decision, "latency": latency,
        "parent_state": {"before_sha256": parent_before, "after_sha256": parent_after,
                         "byte_identical": True},
        "selection": {"recipe_nominated_for_scaling": nominated,
                      "recipe": "NZ_masked485_to_pruned240_BAL-UPMIX_plus_standard_AA" if nominated else None,
                      "checkpoint_emitted": False, "scalability_claimed": False,
                      "full_model_training_authorized": False},
        "retry_allowed": False,
    })


def run(*, upstream_result: Path, output: Path, registration: Path) -> None:
    apply_deterministic_mode()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[2]
    paths = canonical_paths(root)
    upstream_result = _require_canonical_path(upstream_result, paths["upstream_result"], role="upstream result")
    output = _require_canonical_path(output, paths["result"], role="result")
    registration = _require_canonical_path(registration, paths["registration"], role="registration")
    for role in ("attempt", "cal_evidence", "cal_decision", "dev_open", "dev_evidence", "result"):
        if paths[role].exists():
            raise FileExistsError(f"PB21M canonical {role} exists; retry is forbidden")
    registration_record: Mapping[str, object] | None = None
    attempt: Mapping[str, object] | None = None
    loaded_parent: tuple[object, dict[str, object], dict[str, object]] | None = None
    wrapper_parent_before: str | None = None
    try:
        registration_record = validate_registration(registration)
        preflight = real_calibrator_packaging_preflight(root)
        if not preflight["passed"]:
            raise RuntimeError("real calibrator packaging preflight failed before attempt")
        loaded_parent = strict.load_exact_parent(upstream_result)
        wrapper_parent_before = development._state_dict_sha256(loaded_parent[0].state_dict())
        attempt_guard = resource_guard(paths["attempt"], phase="attempt_publication")
        attempt = _publish_attempt(root, registration_record, attempt_guard)
        _run_impl(output=output, registration=registration_record, attempt=attempt,
                  preloaded_parent=loaded_parent)
    except BaseException as error:
        if attempt is not None and not output.exists():
            failure_parent_state = None
            if loaded_parent is not None and wrapper_parent_before is not None:
                wrapper_parent_after = development._state_dict_sha256(loaded_parent[0].state_dict())
                failure_parent_state = {
                    "before_sha256": wrapper_parent_before,
                    "after_sha256": wrapper_parent_after,
                    "byte_identical": wrapper_parent_before == wrapper_parent_after,
                }
            unvalidated_publications = {}
            for stage, role in (("CAL", "cal_evidence"), ("DEV", "dev_evidence")):
                path = paths[role]
                unvalidated_publications[stage] = (
                    {"path": str(path), "sha256": _sha256_file(path),
                     "byte_length": path.stat().st_size,
                     "authoritative": False,
                     "reason": "exception_path_cannot_prove_reload_validate_refit_completed"}
                    if path.is_file() else None
                )
            receipts = {}
            for name, role in (("CAL_decision", "cal_decision"), ("DEV_open", "dev_open")):
                path = paths[role]
                receipts[name] = ({"path": str(path), "sha256": _sha256_file(path)}
                                  if path.is_file() else None)
            _publish_result(output, {
                "schema_version": SCHEMA_VERSION, "implementation_revision": IMPLEMENTATION_REVISION,
                "mode": MODE, "classification": "fresh_run_failed_not_candidate",
                "qualification_claimed": False, "checkpoint_emitted": False,
                "registration": registration_record, "attempt": attempt,
                "authoritative_evidence": None,
                "published_but_unvalidated_evidence": unvalidated_publications,
                "lifecycle_receipts_unvalidated_on_exception_path": receipts,
                "parent_state_on_failure_exit": failure_parent_state,
                "retry_allowed": False,
                "failure": {"type": type(error).__name__, "message": str(error)},
            })
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
        register(registration); return
    if args.upstream_result is None or args.output is None:
        parser.error("run requires --upstream-result, --output, and --registration")
    run(upstream_result=Path(args.upstream_result).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(), registration=registration)


if __name__ == "__main__":
    main()
