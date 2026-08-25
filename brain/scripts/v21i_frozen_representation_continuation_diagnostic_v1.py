"""TRAIN-only continuation diagnostic for the exact V2.1i seed-42 v3 parent.

This script cannot publish a candidate. It freezes the recurrent representation,
continues only the existing nonlinear hazard path on TRAIN-FIT, and evaluates a
fixed milestone schedule on TRAIN-FIT and TRAIN-CAL. DEV, CPU-QUAL, and TEST are
never constructed.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from irene_brain.data import DatasetSplit, MazeChaseDatasetConfig
from irene_brain.evaluation.v21_qualification_metrics import (
    all_action_probability_metrics,
    binary_probability_metrics,
    evaluate_cross_episode_derangements,
    gather_factual_rows,
)
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model
from run_provenance import apply_deterministic_mode
from v21i_development_runner import (
    ACTION_COUNT,
    BURN_IN_STEPS,
    PARTITION_ALGORITHM,
    SEQUENCE_LENGTH,
    TRAIN_CAL_EPISODES,
    TRAIN_CAL_OFFSET,
    TRAIN_FIT_EPISODES,
    TRAIN_FIT_OFFSET,
    CollectedEvidence,
    PartitionContract,
    PartitionSource,
    _canonicalize_table,
    _publish_checkpoint_create_only,
    _publish_json_create_only,
    _source_bundle,
    _state_dict_sha256,
    audit_model_finite,
    collect_evidence,
    model_config,
)


SCHEMA_VERSION = 1
MODE = "v21i_frozen_representation_continuation_diagnostic_v1"
CLASSIFICATION = "diagnostic_not_candidate"
CANDIDATE_PUBLICATION_ALLOWED = False

EXACT_V3_RESULT_SHA256 = (
    "3f91921b7ca63148addd3204cdb1e96b130f26240f49277a269ac8f079b13936"
)
EXACT_V3_PARENT_SHA256 = (
    "e5a4c1e92cdf5649679f344ce9a032397e69a0a5c93bb9d3a7bebc4bc1352081"
)
EXACT_V3_PARENT_STATE_SHA256 = (
    "2619b5b0a180bdd55296d3471d89725f4346110e74ef093f84c3068a32e6576d"
)
EXACT_V3_SOURCE_BUNDLE_SHA256 = (
    "681f3f45422316ec40c299b6c449a90a4790f544ea21311597416cfa7fbb3862"
)
EXACT_TRAIN_FIT_MANIFEST_SHA256 = (
    "835fa63a0388c23eff17713b55836626f42ddd361ec2829948359d2187f65179"
)
EXACT_TRAIN_CAL_MANIFEST_SHA256 = (
    "aed8e64a9889fab82eea2b2551422c4a7f44cf96ea2b80864f857512218a2a83"
)

UPSTREAM_REFINEMENT_PASSES = 8
ADDED_PASSES = 16
MILESTONES = (0, 1, 2, 4, 8, 16)
ROOT_BATCH_SIZE = 24
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
CLIP_NORM = 1.0
PERMUTATION_SEED = 30_042
SHUFFLE_REPETITIONS = 20
SHUFFLE_SEED = 31_042
TRAINABLE_PARAMETER_COUNT = 44_161

FIT_BCE_IMPROVEMENT_FOR_DIAGNOSIS = 0.02
CAL_BCE_IMPROVEMENT_FOR_HEAD = 0.01
CAL_AUC_GAIN_FOR_HEAD = 0.02
CAL_WORST_ACTION_AUC_GAIN_FOR_HEAD = 0.01
CAL_SHUFFLE_RATIO_GAIN_FOR_HEAD = 0.02
CAL_AUC_GAIN_MAX_FOR_REPRESENTATION = 0.01
CAL_WORST_ACTION_GAIN_MAX_FOR_REPRESENTATION = 0.005
FIT_PLATEAU_IMPROVEMENT = 0.005

_HAZARD_PREFIXES = (
    "hazard_state_trunk.",
    "hazard_action_embedding.",
    "hazard_outcome_trunk.",
    "hazard_head.",
)
_PREREGISTRATION = (
    "brain/docs/preregistrations/"
    "2026-08-24-v21i-frozen-representation-continuation-diagnostic-v1.md"
)
_DIAGNOSTIC_SOURCE_FILES = (
    "brain/scripts/v21i_frozen_representation_continuation_diagnostic_v1.py",
    _PREREGISTRATION,
    "brain/tests/test_v21i_frozen_representation_continuation_diagnostic_v1.py",
)


def scope_limitation_record() -> dict[str, object]:
    return {
        "registered_recurrence_uses_prior_actual_reward": True,
        "registered_recurrence_uses_prior_actual_hazard": True,
        "live_act_rgb_provides_those_outcomes": False,
        "live_signature_matched": False,
        "live_belief_sufficiency_claim_allowed": False,
        "live_qualification_implication": "none",
        "required_followup": "separate_exact_live_signature_matched_probe",
    }


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _diagnostic_source_bundle(project_root: Path) -> dict[str, object]:
    production = _source_bundle(project_root)
    if production.get("sha256") != EXACT_V3_SOURCE_BUNDLE_SHA256:
        raise RuntimeError("production source differs from the exact v3 source bundle")
    diagnostic_files: dict[str, str] = {}
    digest = sha256(b"IRV21IFROZENCONTINUATION\x01")
    digest.update(bytes.fromhex(str(production["sha256"])))
    for relative in _DIAGNOSTIC_SOURCE_FILES:
        content = (project_root / relative).read_bytes()
        file_sha = sha256(content).hexdigest()
        diagnostic_files[relative] = file_sha
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha))
    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "production_v3_source_bundle": production,
        "diagnostic_files": diagnostic_files,
    }


def _preregistration_receipt_payload(
    project_root: Path,
    *,
    source_bundle_factory: Callable[[Path], Mapping[str, object]] = _diagnostic_source_bundle,
) -> dict[str, object]:
    source_bundle = dict(source_bundle_factory(project_root))
    diagnostic_files = source_bundle.get("diagnostic_files")
    if not isinstance(diagnostic_files, Mapping) or set(diagnostic_files) != set(
        _DIAGNOSTIC_SOURCE_FILES
    ):
        raise ValueError("preregistration source bundle is incomplete")
    return {
        "schema_version": 1,
        "mode": MODE,
        "classification": "diagnostic_preregistration_receipt_not_candidate",
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "must_preexist_execution": True,
        "source_bundle": source_bundle,
        "scope_limitation": scope_limitation_record(),
    }


def register_preregistration(
    receipt_path: Path,
    *,
    project_root: Path,
    source_bundle_factory: Callable[[Path], Mapping[str, object]] = _diagnostic_source_bundle,
) -> str:
    """Create the immutable source receipt; never overwrite an existing receipt."""

    payload = _preregistration_receipt_payload(
        project_root,
        source_bundle_factory=source_bundle_factory,
    )
    return _publish_json_create_only(receipt_path, payload)


def validate_preregistration_receipt(
    receipt_path: Path,
    *,
    project_root: Path,
    source_bundle_factory: Callable[[Path], Mapping[str, object]] = _diagnostic_source_bundle,
) -> dict[str, object]:
    if not receipt_path.is_file():
        raise FileNotFoundError("create-only preregistration receipt is required")
    receipt_sha256 = _sha256_file(receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("preregistration receipt must contain a JSON object")
    expected = _preregistration_receipt_payload(
        project_root,
        source_bundle_factory=source_bundle_factory,
    )
    if payload != expected:
        raise ValueError("preregistration receipt or diagnostic source drifted")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": receipt_sha256,
        "source_bundle": expected["source_bundle"],
        "classification": expected["classification"],
        "must_preexist_execution": True,
    }


def validate_upstream_result_payload(
    payload: Mapping[str, object],
    *,
    artifact_sha256: str,
) -> dict[str, object]:
    """Fail closed unless this is the exact frozen seed-42 v3 result."""

    if artifact_sha256 != EXACT_V3_RESULT_SHA256:
        raise ValueError("upstream result is not the exact pinned v3 artifact")
    if (
        payload.get("schema_version") != 1
        or payload.get("mode") != "v21i_development_only"
        or payload.get("classification") != "development_candidate_not_qualified"
        or payload.get("qualification_claimed") is not False
        or payload.get("model_seed") != 42
        or payload.get("epochs") != 3
    ):
        raise ValueError("upstream v3 result envelope drifted")
    source_bundle = payload.get("source_bundle")
    if (
        not isinstance(source_bundle, Mapping)
        or source_bundle.get("sha256") != EXACT_V3_SOURCE_BUNDLE_SHA256
    ):
        raise ValueError("upstream v3 source bundle drifted")
    gate = payload.get("development_gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not False:
        raise ValueError("the pinned v3 result must be the recorded gate failure")
    namespace = payload.get("namespace_policy")
    if not isinstance(namespace, Mapping) or (
        namespace.get("test_split_opened") is not False
        or namespace.get("cpu_qual_opened") is not False
        or namespace.get("dev_used_for_training_or_calibration") is not False
    ):
        raise ValueError("upstream v3 namespace policy drifted")
    partitions = payload.get("dataset_partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != {
        "TRAIN-FIT",
        "TRAIN-CAL",
        "DEV",
    }:
        raise ValueError("upstream v3 partitions are incomplete")
    expected_manifests = {
        "TRAIN-FIT": EXACT_TRAIN_FIT_MANIFEST_SHA256,
        "TRAIN-CAL": EXACT_TRAIN_CAL_MANIFEST_SHA256,
    }
    for name, record in partitions.items():
        if not isinstance(record, Mapping):
            raise TypeError("upstream partition record must be a mapping")
        manifest = record.get("dataset_manifest")
        if not isinstance(manifest, Mapping):
            raise TypeError("upstream dataset manifest must be a mapping")
        if (
            manifest.get("behavior_policy") != "balanced_intervention_v1"
            or manifest.get("behavior_intervention_rate_hex")
            != float(0.5).hex()
            or manifest.get("counterfactual_targets") != "all_actions_v1"
            or record.get("burn_in_steps") != BURN_IN_STEPS
        ):
            raise ValueError(f"upstream {name} behavior protocol drifted")
        if name in expected_manifests and (
            record.get("dataset_manifest_sha256") != expected_manifests[name]
        ):
            raise ValueError(f"upstream {name} manifest digest drifted")
    parent = payload.get("upstream_uncalibrated_checkpoint")
    if not isinstance(parent, Mapping):
        raise TypeError("upstream parent record must be a mapping")
    if (
        parent.get("sha256") != EXACT_V3_PARENT_SHA256
        or parent.get("state_sha256") != EXACT_V3_PARENT_STATE_SHA256
        or not isinstance(parent.get("path"), str)
    ):
        raise ValueError("upstream parent binding drifted")
    return {
        "result_sha256": artifact_sha256,
        "source_bundle_sha256": source_bundle["sha256"],
        "parent_path": parent["path"],
        "parent_sha256": parent["sha256"],
        "parent_state_sha256": parent["state_sha256"],
        "train_fit_manifest_sha256": expected_manifests["TRAIN-FIT"],
        "train_cal_manifest_sha256": expected_manifests["TRAIN-CAL"],
        "partition_algorithm": PARTITION_ALGORITHM,
    }


def load_exact_v3_parent(
    result_path: Path,
) -> tuple[CoreV2Model, dict[str, object], dict[str, object]]:
    result_path = result_path.resolve()
    artifact_sha = _sha256_file(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("upstream result must be a JSON object")
    provenance = validate_upstream_result_payload(
        result,
        artifact_sha256=artifact_sha,
    )
    parent_path = Path(str(provenance["parent_path"])).resolve()
    if _sha256_file(parent_path) != EXACT_V3_PARENT_SHA256:
        raise ValueError("exact v3 parent file hash mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, dict):
        raise TypeError("exact v3 parent checkpoint must be a mapping")
    if (
        parent.get("schema_version") != 1
        or parent.get("mode") != "v21i_development_only"
        or parent.get("checkpoint_role")
        != "upstream_joint_and_hazard_refined_uncalibrated_parent"
        or parent.get("classification") != "development_candidate_not_qualified"
        or parent.get("model_seed") != 42
        or parent.get("state_sha256") != EXACT_V3_PARENT_STATE_SHA256
        or parent.get("source_bundle") != result.get("source_bundle")
        or parent.get("config") != asdict(model_config())
        or parent.get("flags") != asdict(CONFIG_B_PREDICTIVE)
    ):
        raise ValueError("exact v3 parent checkpoint provenance drifted")
    training = parent.get("training")
    if not isinstance(training, Mapping) or (
        training.get("joint_training", {}).get("optimizer_steps") != 48
        or training.get("hazard_refinement", {}).get("passes")
        != UPSTREAM_REFINEMENT_PASSES
        or training.get("hazard_refinement", {}).get("optimizer_steps") != 512
    ):
        raise ValueError("exact v3 parent schedule drifted")
    state_dict = parent.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("exact v3 parent has no model state")
    if _state_dict_sha256(state_dict) != EXACT_V3_PARENT_STATE_SHA256:
        raise ValueError("exact v3 parent state digest mismatch")
    model = CoreV2Model(config=model_config(), flags=CONFIG_B_PREDICTIVE)
    model.load_state_dict(state_dict, strict=True)
    finite = audit_model_finite(model)
    if finite["state_sha256"] != EXACT_V3_PARENT_STATE_SHA256:
        raise ValueError("loaded exact v3 parent state drifted")
    outcome_model = model.world_model.outcome_model if model.world_model else None
    if outcome_model is None:
        raise RuntimeError("exact v3 parent has no outcome model")
    if not torch.equal(
        outcome_model.hazard_calibration_scale,
        torch.ones_like(outcome_model.hazard_calibration_scale),
    ) or not torch.equal(
        outcome_model.hazard_calibration_bias,
        torch.zeros_like(outcome_model.hazard_calibration_bias),
    ):
        raise ValueError("exact v3 parent is not raw and uncalibrated")
    return model, result, provenance


def select_hazard_path_parameters(
    model: CoreV2Model,
) -> tuple[list[nn.Parameter], tuple[str, ...]]:
    model.requires_grad_(False)
    outcome_model = model.world_model.outcome_model if model.world_model else None
    if outcome_model is None:
        raise RuntimeError("hazard continuation requires an outcome model")
    selected: list[nn.Parameter] = []
    names: list[str] = []
    for name, parameter in outcome_model.named_parameters():
        if name.startswith(_HAZARD_PREFIXES):
            parameter.requires_grad_(True)
            selected.append(parameter)
            names.append(f"world_model.outcome_model.{name}")
    count = sum(parameter.numel() for parameter in selected)
    if count != TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError(
            f"hazard continuation selected {count} params, expected "
            f"{TRAINABLE_PARAMETER_COUNT}"
        )
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in set(names)
    ):
        raise RuntimeError("a non-hazard parameter remains trainable")
    return selected, tuple(sorted(names))


def deterministic_permutations(
    root_count: int,
    *,
    passes: int = ADDED_PASSES,
    seed: int = PERMUTATION_SEED,
) -> tuple[Tensor, ...]:
    if root_count < 1 or passes < 1:
        raise ValueError("root_count and passes must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutations = tuple(torch.randperm(root_count, generator=generator) for _ in range(passes))
    if any(int(permutation.unique().numel()) != root_count for permutation in permutations):
        raise RuntimeError("diagnostic permutation contains replacement")
    return permutations


def _permutation_sha256(permutation: Tensor) -> str:
    values = permutation.detach().to(device="cpu", dtype=torch.int64).numpy()
    return sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def preregistered_permutation_digests() -> tuple[str, ...]:
    """Derive the only accepted +16 pass order from the frozen seed and root count."""

    permutations = deterministic_permutations(
        TRAIN_FIT_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS),
        passes=ADDED_PASSES,
        seed=PERMUTATION_SEED,
    )
    return tuple(_permutation_sha256(permutation) for permutation in permutations)


def diagnostic_partition_contracts(
    *,
    config_factory: Callable[..., object] = MazeChaseDatasetConfig,
    contract_factory: Callable[..., object] = PartitionContract,
) -> dict[str, object]:
    """Construct only the two authorized TRAIN contracts for this diagnostic."""

    common: dict[str, object] = {
        "sequence_length": SEQUENCE_LENGTH,
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
    specifications = (
        ("TRAIN-FIT", TRAIN_FIT_EPISODES, TRAIN_FIT_OFFSET),
        ("TRAIN-CAL", TRAIN_CAL_EPISODES, TRAIN_CAL_OFFSET),
    )
    return {
        name: contract_factory(
            name,
            config_factory(
                split=DatasetSplit.TRAIN,
                sequence_count=episodes,
                seed_offset=offset,
                **common,
            ),
        )
        for name, episodes, offset in specifications
    }


def diagnostic_partition_sources(
    *,
    config_factory: Callable[..., object] = MazeChaseDatasetConfig,
    contract_factory: Callable[..., object] = PartitionContract,
    source_factory: Callable[[object], object] = PartitionSource,
) -> dict[str, object]:
    """Instantiate sources only after the local TRAIN-only contract boundary."""

    contracts = diagnostic_partition_contracts(
        config_factory=config_factory,
        contract_factory=contract_factory,
    )
    return {name: source_factory(contract) for name, contract in contracts.items()}


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256(b"IRV21IDIAGNOSTICARRAY\x01")
    digest.update(array.dtype.str.encode("ascii"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_string_sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(b"IRV21IDIAGNOSTICSEQUENCE\x01" + encoded).hexdigest()


def evidence_manifest(
    partition: str,
    evidence: CollectedEvidence,
    *,
    dataset_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "partition": partition,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "roots": int(evidence.hazard_targets.shape[0]),
        "actions_per_root": int(evidence.hazard_targets.shape[1]),
        "belief_sha256": _array_sha256(evidence.beliefs),
        "target_sha256": _array_sha256(evidence.hazard_targets),
        "factual_action_sha256": _array_sha256(evidence.factual_actions),
        "ordered_root_sha256": _ordered_string_sequence_sha256(
            evidence.root_state_ids
        ),
        "ordered_episode_group_sha256": _ordered_string_sequence_sha256(
            evidence.episode_group_ids
        ),
    }


@torch.no_grad()
def raw_logits_from_cached_beliefs(
    model: CoreV2Model,
    beliefs: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    outcome_model = model.world_model.outcome_model if model.world_model else None
    if outcome_model is None:
        raise RuntimeError("raw evaluation requires an outcome model")
    was_training = outcome_model.training
    outcome_model.eval()
    tables: list[np.ndarray] = []
    contexts = torch.from_numpy(beliefs).to(dtype=torch.float32)
    for start in range(0, contexts.shape[0], batch_size):
        table = outcome_model(contexts[start : start + batch_size])
        if table.raw_hazard_logits is None:
            raise RuntimeError("diagnostic requires raw semantic hazard logits")
        tables.append(_canonicalize_table(table.raw_hazard_logits, table.action_ids))
    if was_training:
        outcome_model.train()
    return np.concatenate(tables, axis=0)


def evaluate_cached_partition(
    model: CoreV2Model,
    evidence: CollectedEvidence,
    *,
    partition: str,
) -> dict[str, object]:
    logits = raw_logits_from_cached_beliefs(model, evidence.beliefs)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    action_ids = tuple(range(ACTION_COUNT))
    factual_targets, factual_probabilities = gather_factual_rows(
        evidence.hazard_targets,
        probabilities,
        evidence.factual_actions,
        action_ids=action_ids,
    )
    shuffle = evaluate_cross_episode_derangements(
        evidence.hazard_targets,
        probabilities,
        evidence.episode_group_ids,
        action_ids=action_ids,
        repetitions=SHUFFLE_REPETITIONS,
        seed=SHUFFLE_SEED,
    ).as_dict(include_repetitions=True)
    all_action = all_action_probability_metrics(
        evidence.hazard_targets,
        probabilities,
        action_ids=action_ids,
    ).as_dict()
    model_bce = _finite(all_action["aggregate"]["bce"], name="model BCE")
    shuffled_bce = _finite(shuffle["median_shuffled_bce"], name="shuffled BCE")
    return {
        "partition": partition,
        "raw_uncalibrated": True,
        "raw_logit_sha256": _array_sha256(logits),
        "all_action": all_action,
        "factual": binary_probability_metrics(
            factual_targets,
            factual_probabilities,
        ).as_dict(),
        "cross_episode_derangements": shuffle,
        "shuffle_to_model_bce_ratio": shuffled_bce / model_bce,
    }


def _worst_action_auc(report: Mapping[str, object]) -> float:
    per_action = report["all_action"]["per_action"]
    return min(_finite(entry["metrics"]["roc_auc"], name="action AUC") for entry in per_action)


def interpretation_report(
    initial: Mapping[str, object],
    final: Mapping[str, object],
) -> dict[str, object]:
    initial_fit = initial["TRAIN-FIT"]
    final_fit = final["TRAIN-FIT"]
    initial_cal = initial["TRAIN-CAL"]
    final_cal = final["TRAIN-CAL"]
    fit_bce_improvement = _finite(
        initial_fit["all_action"]["aggregate"]["bce"],
        name="initial FIT BCE",
    ) - _finite(final_fit["all_action"]["aggregate"]["bce"], name="final FIT BCE")
    cal_bce_improvement = _finite(
        initial_cal["all_action"]["aggregate"]["bce"],
        name="initial CAL BCE",
    ) - _finite(final_cal["all_action"]["aggregate"]["bce"], name="final CAL BCE")
    cal_auc_gain = _finite(
        final_cal["all_action"]["aggregate"]["roc_auc"],
        name="final CAL AUC",
    ) - _finite(
        initial_cal["all_action"]["aggregate"]["roc_auc"],
        name="initial CAL AUC",
    )
    worst_action_gain = _worst_action_auc(final_cal) - _worst_action_auc(initial_cal)
    initial_action_auc = [
        _finite(entry["metrics"]["roc_auc"], name="initial action AUC")
        for entry in initial_cal["all_action"]["per_action"]
    ]
    final_action_auc = [
        _finite(entry["metrics"]["roc_auc"], name="final action AUC")
        for entry in final_cal["all_action"]["per_action"]
    ]
    nondeclining_actions = sum(
        after >= before
        for before, after in zip(initial_action_auc, final_action_auc, strict=True)
    )
    shuffle_ratio_gain = _finite(
        final_cal["shuffle_to_model_bce_ratio"],
        name="final shuffle ratio",
    ) - _finite(
        initial_cal["shuffle_to_model_bce_ratio"],
        name="initial shuffle ratio",
    )
    head_checks = {
        "train_fit_bce_improvement_at_least_0_02": (
            fit_bce_improvement >= FIT_BCE_IMPROVEMENT_FOR_DIAGNOSIS
        ),
        "train_cal_bce_improvement_at_least_0_01": (
            cal_bce_improvement >= CAL_BCE_IMPROVEMENT_FOR_HEAD
        ),
        "train_cal_auc_gain_at_least_0_02": cal_auc_gain >= CAL_AUC_GAIN_FOR_HEAD,
        "train_cal_worst_action_auc_gain_at_least_0_01": (
            worst_action_gain >= CAL_WORST_ACTION_AUC_GAIN_FOR_HEAD
        ),
        "at_least_four_action_aucs_nondeclining": nondeclining_actions >= 4,
        "train_cal_shuffle_ratio_gain_at_least_0_02": (
            shuffle_ratio_gain >= CAL_SHUFFLE_RATIO_GAIN_FOR_HEAD
        ),
    }
    representation_checks = {
        "train_fit_bce_improvement_at_least_0_02": (
            fit_bce_improvement >= FIT_BCE_IMPROVEMENT_FOR_DIAGNOSIS
        ),
        "train_cal_auc_gain_below_0_01": (
            cal_auc_gain < CAL_AUC_GAIN_MAX_FOR_REPRESENTATION
        ),
        "train_cal_worst_action_auc_gain_below_0_005": (
            worst_action_gain < CAL_WORST_ACTION_GAIN_MAX_FOR_REPRESENTATION
        ),
    }
    if all(head_checks.values()):
        conclusion = "hazard_path_undertraining_supported"
    elif all(representation_checks.values()):
        conclusion = "frozen_belief_generalization_limitation_supported"
    else:
        conclusion = "inconclusive"
    return {
        "schema_version": 1,
        "comparison": "added_pass_16_minus_added_pass_0_only",
        "metrics": {
            "train_fit_bce_improvement": fit_bce_improvement,
            "train_cal_bce_improvement": cal_bce_improvement,
            "train_cal_aggregate_auc_gain": cal_auc_gain,
            "train_cal_worst_action_auc_gain": worst_action_gain,
            "train_cal_nondeclining_action_auc_count": nondeclining_actions,
            "train_cal_shuffle_bce_ratio_gain": shuffle_ratio_gain,
        },
        "head_undertraining_checks": head_checks,
        "representation_limitation_checks": representation_checks,
        "fit_plateau_below_0_005": fit_bce_improvement < FIT_PLATEAU_IMPROVEMENT,
        "conclusion": conclusion,
        "candidate_decision": "prohibited",
        "scope_limitation": scope_limitation_record(),
    }


def _optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not bool(torch.isfinite(value).all()):
                return False
    return True


def _digest_structured_value(digest: object, value: object) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(b"T")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(len(tensor.shape).to_bytes(4, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        items = sorted(
            value.items(),
            key=lambda item: (type(item[0]).__name__, repr(item[0])),
        )
        digest.update(len(items).to_bytes(8, "big"))
        for key, nested in items:
            _digest_structured_value(digest, key)
            _digest_structured_value(digest, nested)
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"L" if isinstance(value, list) else b"Q")
        digest.update(len(value).to_bytes(8, "big"))
        for nested in value:
            _digest_structured_value(digest, nested)
        return
    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if type(value) is int:
        encoded = str(value).encode("ascii")
        digest.update(b"I" + len(encoded).to_bytes(4, "big") + encoded)
        return
    if type(value) is float:
        encoded = value.hex().encode("ascii")
        digest.update(b"F" + len(encoded).to_bytes(4, "big") + encoded)
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
        return
    raise TypeError(f"unsupported optimizer-state value: {type(value).__name__}")


def _optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = sha256(b"IRV21IDIAGNOSTICOPTIMIZER\x01")
    _digest_structured_value(digest, optimizer.state_dict())
    return digest.hexdigest()


def _checkpoint_paths(checkpoint_dir: Path) -> dict[int, Path]:
    return {
        milestone: checkpoint_dir / f"added-pass-{milestone:02d}.diagnostic.pt"
        for milestone in MILESTONES
    }


def _assert_targets_available(output: Path, checkpoint_paths: Mapping[int, Path]) -> None:
    targets = [output.resolve(), *(path.resolve() for path in checkpoint_paths.values())]
    if len(set(targets)) != len(targets):
        raise ValueError("diagnostic output/checkpoint targets must be distinct")
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to replace an existing diagnostic artifact")


def _existing_checkpoint_receipts(
    checkpoint_paths: Mapping[int, Path],
) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for milestone, path in checkpoint_paths.items():
        if path.is_file():
            receipts.append(
                {
                    "added_pass": milestone,
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                }
            )
    return receipts


def validate_pass_chain(
    pass_records: Sequence[Mapping[str, object]],
    *,
    initial_optimizer_state_sha256: str,
) -> dict[str, object]:
    if len(pass_records) != ADDED_PASSES:
        raise ValueError("pass chain must contain exactly 16 records")
    if not _is_sha256(initial_optimizer_state_sha256):
        raise ValueError("initial optimizer state digest is invalid")
    expected_state = EXACT_V3_PARENT_STATE_SHA256
    expected_optimizer_state = initial_optimizer_state_sha256
    expected_step = 0
    expected_permutation_digests = preregistered_permutation_digests()
    if len(expected_permutation_digests) != ADDED_PASSES:
        raise RuntimeError("preregistered permutation schedule is incomplete")
    for added_pass, record in enumerate(pass_records, start=1):
        if (
            record.get("added_pass") != added_pass
            or record.get("total_refinement_pass")
            != UPSTREAM_REFINEMENT_PASSES + added_pass
            or record.get("roots") != 1_536
            or record.get("optimizer_steps") != 64
            or record.get("optimizer_step_start") != expected_step
            or record.get("optimizer_step_end") != expected_step + 64
            or record.get("state_sha256_before") != expected_state
            or record.get("optimizer_state_sha256_before")
            != expected_optimizer_state
        ):
            raise ValueError("pass chain schedule/state continuity drifted")
        after = record.get("state_sha256_after")
        optimizer_after = record.get("optimizer_state_sha256_after")
        permutation = record.get("root_permutation_sha256")
        if not _is_sha256(after) or not _is_sha256(optimizer_after):
            raise ValueError("pass chain contains an invalid SHA-256")
        if permutation != expected_permutation_digests[added_pass - 1]:
            raise ValueError("pass chain permutation differs from preregistration")
        expected_state = str(after)
        expected_optimizer_state = str(optimizer_after)
        expected_step += 64
    if expected_step != 1_024:
        raise ValueError("pass chain does not end at optimizer step 1,024")
    return {
        "valid": True,
        "initial_state_sha256": EXACT_V3_PARENT_STATE_SHA256,
        "final_state_sha256": expected_state,
        "initial_optimizer_state_sha256": initial_optimizer_state_sha256,
        "final_optimizer_state_sha256": expected_optimizer_state,
        "optimizer_step_start": 0,
        "optimizer_step_end": expected_step,
        "passes": ADDED_PASSES,
        "exact_preregistered_permutations": True,
        "permutation_digest_vector_sha256": _ordered_string_sequence_sha256(
            expected_permutation_digests
        ),
    }


def _run_impl(
    *,
    upstream_result: Path,
    output: Path,
    checkpoint_dir: Path,
    preregistration_receipt_path: Path,
) -> None:
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    project_root = Path(__file__).resolve().parents[2]
    preregistration_receipt = validate_preregistration_receipt(
        preregistration_receipt_path,
        project_root=project_root,
    )
    source_bundle = preregistration_receipt["source_bundle"]
    model, upstream_payload, upstream = load_exact_v3_parent(upstream_result)
    parent_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    selected, selected_names = select_hazard_path_parameters(model)
    selected_name_set = set(selected_names)
    optimizer = torch.optim.AdamW(
        selected,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    initial_optimizer_state_sha256 = _optimizer_state_sha256(optimizer)
    optimizer_restart = {
        "optimizer": "AdamW",
        "fresh_restart": True,
        "upstream_optimizer_state_available": False,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "initial_state_sha256": initial_optimizer_state_sha256,
    }

    sources = diagnostic_partition_sources()
    if set(sources) != {"TRAIN-FIT", "TRAIN-CAL"}:
        raise RuntimeError("diagnostic source factory escaped the TRAIN-only boundary")
    fit_source = sources["TRAIN-FIT"]
    cal_source = sources["TRAIN-CAL"]
    if fit_source.manifest_sha256 != EXACT_TRAIN_FIT_MANIFEST_SHA256:
        raise RuntimeError("live TRAIN-FIT manifest differs from exact v3")
    if cal_source.manifest_sha256 != EXACT_TRAIN_CAL_MANIFEST_SHA256:
        raise RuntimeError("live TRAIN-CAL manifest differs from exact v3")
    parent_sha_before_cache = _state_dict_sha256(model.state_dict())
    fit_evidence = collect_evidence(model, fit_source)
    cal_evidence = collect_evidence(model, cal_source)
    if _state_dict_sha256(model.state_dict()) != parent_sha_before_cache:
        raise RuntimeError("TRAIN cache construction mutated the exact parent")
    evidence_manifests = {
        "TRAIN-FIT": evidence_manifest(
            "TRAIN-FIT",
            fit_evidence,
            dataset_manifest_sha256=fit_source.manifest_sha256,
        ),
        "TRAIN-CAL": evidence_manifest(
            "TRAIN-CAL",
            cal_evidence,
            dataset_manifest_sha256=cal_source.manifest_sha256,
        ),
    }
    if evidence_manifests["TRAIN-FIT"]["roots"] != 1_536:
        raise RuntimeError("TRAIN-FIT cache must contain exactly 1,536 roots")
    if evidence_manifests["TRAIN-CAL"]["roots"] != 768:
        raise RuntimeError("TRAIN-CAL cache must contain exactly 768 roots")

    permutations = deterministic_permutations(fit_evidence.beliefs.shape[0])
    checkpoint_paths = _checkpoint_paths(checkpoint_dir)
    milestone_reports: dict[str, object] = {}
    checkpoint_receipts: list[dict[str, object]] = []
    pass_records: list[dict[str, object]] = []
    optimizer_steps_completed = 0
    contexts = torch.from_numpy(fit_evidence.beliefs).to(dtype=torch.float32)
    targets = torch.from_numpy(fit_evidence.hazard_targets).to(dtype=torch.float32)
    outcome_model = model.world_model.outcome_model if model.world_model else None
    if outcome_model is None:
        raise RuntimeError("diagnostic model has no outcome model")

    def publish_milestone(added_pass: int) -> None:
        state_before_evaluation = _state_dict_sha256(model.state_dict())
        optimizer_state_sha256 = _optimizer_state_sha256(optimizer)
        expected_optimizer_state_sha256 = (
            initial_optimizer_state_sha256
            if not pass_records
            else pass_records[-1]["optimizer_state_sha256_after"]
        )
        if optimizer_state_sha256 != expected_optimizer_state_sha256:
            raise RuntimeError("milestone optimizer state is not contiguous")
        evaluations = {
            "TRAIN-FIT": evaluate_cached_partition(
                model,
                fit_evidence,
                partition="TRAIN-FIT",
            ),
            "TRAIN-CAL": evaluate_cached_partition(
                model,
                cal_evidence,
                partition="TRAIN-CAL",
            ),
        }
        if _state_dict_sha256(model.state_dict()) != state_before_evaluation:
            raise RuntimeError("milestone evaluation mutated diagnostic state")
        finite = audit_model_finite(model)
        nonselected_equal = all(
            torch.equal(parent_state[name], value)
            for name, value in model.state_dict().items()
            if name not in selected_name_set
        )
        if not nonselected_equal:
            raise RuntimeError("diagnostic changed a frozen non-hazard tensor")
        if not _optimizer_finite(optimizer):
            raise FloatingPointError("diagnostic optimizer state is non-finite")
        report = {
            "added_pass": added_pass,
            "total_refinement_pass": UPSTREAM_REFINEMENT_PASSES + added_pass,
            "optimizer_steps": optimizer_steps_completed,
            "evaluations": evaluations,
            "finite_state_audit": finite,
            "optimizer_state_finite": True,
            "optimizer_state_sha256": optimizer_state_sha256,
            "all_nonhazard_tensors_equal_to_parent": True,
        }
        checkpoint_payload = {
            "schema_version": SCHEMA_VERSION,
            "mode": MODE,
            "classification": CLASSIFICATION,
            "qualification_claimed": False,
            "candidate_publication_allowed": CANDIDATE_PUBLICATION_ALLOWED,
            "checkpoint_role": "diagnostic_milestone_not_candidate",
            "scope_limitation": scope_limitation_record(),
            "upstream": upstream,
            "preregistration_receipt": preregistration_receipt,
            "source_bundle": source_bundle,
            "namespace_policy": {
                "opened": ["TRAIN-FIT", "TRAIN-CAL"],
                "dev_opened": False,
                "cpu_qual_opened": False,
                "test_opened": False,
                "train_cal_used_for_optimization_or_selection": False,
            },
            "schedule": schedule_record(),
            "optimizer_restart": optimizer_restart,
            "selected_parameter_names": list(selected_names),
            "selected_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "evidence_manifests": evidence_manifests,
            "pass_records": list(pass_records),
            "milestone": report,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_sha256": finite["state_sha256"],
            "optimizer_state_sha256": optimizer_state_sha256,
        }
        checkpoint_path = checkpoint_paths[added_pass]
        checkpoint_sha = _publish_checkpoint_create_only(
            checkpoint_path,
            checkpoint_payload,
        )
        receipt = {
            "added_pass": added_pass,
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "state_sha256": finite["state_sha256"],
            "optimizer_state_sha256": optimizer_state_sha256,
            "classification": CLASSIFICATION,
            "candidate_publication_allowed": False,
        }
        checkpoint_receipts.append(receipt)
        milestone_reports[str(added_pass)] = report

    publish_milestone(0)
    for pass_index, permutation in enumerate(permutations, start=1):
        state_sha_before_pass = _state_dict_sha256(model.state_dict())
        optimizer_state_sha_before_pass = _optimizer_state_sha256(optimizer)
        if pass_index == 1:
            if state_sha_before_pass != EXACT_V3_PARENT_STATE_SHA256:
                raise RuntimeError("continuation does not start at the exact v3 parent")
            if optimizer_state_sha_before_pass != initial_optimizer_state_sha256:
                raise RuntimeError("continuation does not start at the fresh optimizer")
        elif state_sha_before_pass != pass_records[-1]["state_sha256_after"]:
            raise RuntimeError("continuation state chain is not contiguous")
        elif (
            optimizer_state_sha_before_pass
            != pass_records[-1]["optimizer_state_sha256_after"]
        ):
            raise RuntimeError("continuation optimizer chain is not contiguous")
        optimizer_step_start = optimizer_steps_completed
        outcome_model.train()
        loss_sum = 0.0
        roots_seen = 0
        max_gradient = 0.0
        for start in range(0, len(permutation), ROOT_BATCH_SIZE):
            indices = permutation[start : start + ROOT_BATCH_SIZE]
            table = outcome_model(contexts[indices])
            if table.raw_hazard_logits is None:
                raise RuntimeError("continuation requires raw hazard logits")
            logits = table.raw_hazard_logits.squeeze(-1)
            aligned_targets = torch.gather(targets[indices], 1, table.action_ids)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                aligned_targets,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite continuation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(selected, CLIP_NORM))
            if not math.isfinite(gradient):
                raise FloatingPointError("non-finite continuation gradient")
            optimizer.step()
            optimizer_steps_completed += 1
            roots = len(indices)
            loss_sum += float(loss.detach()) * roots
            roots_seen += roots
            max_gradient = max(max_gradient, gradient)
        permutation_sha = _permutation_sha256(permutation)
        state_sha_after_pass = _state_dict_sha256(model.state_dict())
        optimizer_state_sha_after_pass = _optimizer_state_sha256(optimizer)
        pass_records.append(
            {
                "added_pass": pass_index,
                "total_refinement_pass": UPSTREAM_REFINEMENT_PASSES + pass_index,
                "roots": roots_seen,
                "optimizer_steps": math.ceil(roots_seen / ROOT_BATCH_SIZE),
                "optimizer_step_start": optimizer_step_start,
                "optimizer_step_end": optimizer_steps_completed,
                "mean_unweighted_all_action_bce": loss_sum / roots_seen,
                "maximum_preclip_gradient_norm": max_gradient,
                "root_permutation_sha256": permutation_sha,
                "state_sha256_before": state_sha_before_pass,
                "state_sha256_after": state_sha_after_pass,
                "optimizer_state_sha256_before": optimizer_state_sha_before_pass,
                "optimizer_state_sha256_after": optimizer_state_sha_after_pass,
            }
        )
        if pass_index in MILESTONES:
            publish_milestone(pass_index)

    if len(pass_records) != ADDED_PASSES:
        raise RuntimeError("diagnostic did not complete all preregistered passes")
    if optimizer_steps_completed != 1_024:
        raise RuntimeError("diagnostic optimizer-step chain did not end at 1,024")
    pass_chain_audit = validate_pass_chain(
        pass_records,
        initial_optimizer_state_sha256=initial_optimizer_state_sha256,
    )
    initial = milestone_reports["0"]["evaluations"]
    final = milestone_reports[str(ADDED_PASSES)]["evaluations"]
    interpretation = interpretation_report(initial, final)
    current_source_bundle = _diagnostic_source_bundle(project_root)
    if current_source_bundle != source_bundle:
        raise RuntimeError("diagnostic source bundle changed during execution")
    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "qualification_claimed": False,
        "candidate_publication_allowed": CANDIDATE_PUBLICATION_ALLOWED,
        "publication_effect": "diagnostic_evidence_only",
        "scope_limitation": scope_limitation_record(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "upstream": upstream,
        "upstream_result_path": str(upstream_result.resolve()),
        "upstream_gate_passed": upstream_payload["development_gate"]["passed"],
        "preregistration_receipt": preregistration_receipt,
        "source_bundle": source_bundle,
        "namespace_policy": {
            "opened": ["TRAIN-FIT", "TRAIN-CAL"],
            "dev_opened": False,
            "cpu_qual_opened": False,
            "test_opened": False,
            "train_fit_used_for_optimization": True,
            "train_cal_used_for_optimization_or_selection": False,
            "early_stopping_or_model_selection": False,
        },
        "schedule": schedule_record(),
        "optimizer_restart": optimizer_restart,
        "selected_parameter_names": list(selected_names),
        "selected_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "evidence_manifests": evidence_manifests,
        "pass_records": pass_records,
        "pass_chain_audit": pass_chain_audit,
        "milestones": milestone_reports,
        "checkpoint_receipts": checkpoint_receipts,
        "interpretation": interpretation,
        "candidate_checkpoint": {
            "published": False,
            "publication_allowed": False,
            "reason": "diagnostic script has no candidate publication path",
        },
    }
    _publish_json_create_only(output, result)
    print(f"DONE diagnostic-only evidence -> {output.resolve()}", flush=True)


def schedule_record() -> dict[str, object]:
    return {
        "upstream_refinement_passes": UPSTREAM_REFINEMENT_PASSES,
        "added_passes": ADDED_PASSES,
        "total_refinement_passes": UPSTREAM_REFINEMENT_PASSES + ADDED_PASSES,
        "milestones_added_passes": list(MILESTONES),
        "milestones_total_passes": [
            UPSTREAM_REFINEMENT_PASSES + milestone for milestone in MILESTONES
        ],
        "train_partition": "TRAIN-FIT",
        "evaluation_partitions": ["TRAIN-FIT", "TRAIN-CAL"],
        "root_batch_size": ROOT_BATCH_SIZE,
        "roots_per_pass": TRAIN_FIT_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS),
        "optimizer_steps_per_pass": 64,
        "optimizer_steps": 1_024,
        "action_branches_per_root": ACTION_COUNT,
        "sampling": "deterministic_complete_root_permutation_without_replacement",
        "permutation_seed": PERMUTATION_SEED,
        "shuffle_repetitions": SHUFFLE_REPETITIONS,
        "shuffle_seed": SHUFFLE_SEED,
        "early_stopping": False,
        "model_selection": False,
        "calibration_fit_or_install": False,
    }


def _run_with_failure_receipt(
    *,
    upstream_result: Path,
    preregistration_receipt_path: Path,
    output: Path,
    checkpoint_paths: Mapping[int, Path],
    operation: Callable[[], None],
) -> None:
    try:
        operation()
    except BaseException as error:
        if not output.exists():
            preregistration_observation: dict[str, object] = {
                "path": str(preregistration_receipt_path.resolve()),
                "present": preregistration_receipt_path.is_file(),
                "sha256": None,
                "validated": False,
            }
            if preregistration_receipt_path.is_file():
                preregistration_observation["sha256"] = _sha256_file(
                    preregistration_receipt_path
                )
                try:
                    validate_preregistration_receipt(
                        preregistration_receipt_path,
                        project_root=Path(__file__).resolve().parents[2],
                    )
                except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
                    pass
                else:
                    preregistration_observation["validated"] = True
            failure = {
                "schema_version": SCHEMA_VERSION,
                "mode": MODE,
                "classification": "diagnostic_run_failed_not_candidate",
                "qualification_claimed": False,
                "candidate_publication_allowed": False,
                "scope_limitation": scope_limitation_record(),
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "upstream_result_path": str(upstream_result.resolve()),
                "preregistration_receipt": preregistration_observation,
                "diagnostic_checkpoint_receipts": _existing_checkpoint_receipts(
                    checkpoint_paths
                ),
                "candidate_checkpoint": {
                    "published": False,
                    "publication_allowed": False,
                },
            }
            _publish_json_create_only(output, failure)
        raise


def run(
    *,
    upstream_result: Path,
    output: Path,
    checkpoint_dir: Path,
    preregistration_receipt_path: Path,
) -> None:
    checkpoint_paths = _checkpoint_paths(checkpoint_dir.resolve())

    def preflight_and_run() -> None:
        _assert_targets_available(output, checkpoint_paths)
        _run_impl(
            upstream_result=upstream_result,
            output=output,
            checkpoint_dir=checkpoint_dir,
            preregistration_receipt_path=preregistration_receipt_path,
        )

    _run_with_failure_receipt(
        upstream_result=upstream_result,
        preregistration_receipt_path=preregistration_receipt_path,
        output=output,
        checkpoint_paths=checkpoint_paths,
        operation=preflight_and_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--preregistration-receipt", required=True)
    parser.add_argument("--upstream-result")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    preregistration_receipt_path = (
        Path(args.preregistration_receipt).expanduser().resolve()
    )
    if args.register_only:
        if args.upstream_result or args.output or args.checkpoint_dir:
            parser.error("--register-only accepts only --preregistration-receipt")
        project_root = Path(__file__).resolve().parents[2]
        receipt_sha256 = register_preregistration(
            preregistration_receipt_path,
            project_root=project_root,
        )
        print(
            "DONE create-only diagnostic preregistration receipt "
            f"sha256={receipt_sha256} -> {preregistration_receipt_path}",
            flush=True,
        )
        return
    if not args.upstream_result or not args.output:
        parser.error("execution requires --upstream-result and --output")
    upstream_result = Path(args.upstream_result).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else Path(f"{output}.diagnostic-checkpoints")
    )
    run(
        upstream_result=upstream_result,
        output=output,
        checkpoint_dir=checkpoint_dir,
        preregistration_receipt_path=preregistration_receipt_path,
    )


if __name__ == "__main__":
    main()
