"""Frozen data and scoring contract for the first matched held-out suites.

This module is deliberately framework-light.  It materializes the already
available deterministic MovingShapes sequences and aggregates the existing
final-exit ``movement_exact_match`` metric.  It does not load a model, launch
training, or select a checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

from irene_brain.data import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequenceDataset,
    dataset_manifest_sha256,
)
from irene_brain.training.batches import TrajectoryBatch


SAMPLE_HASH_DOMAIN = b"IRFIRSTMATCHEDSAMPLES\x01"
EVALUATION_CODE_HASH_DOMAIN = b"IRFIRSTMATCHEDEVAL\x01"
MOVEMENT_KEYS = (
    {"name": "w", "hid_usage_id": 26},
    {"name": "a", "hid_usage_id": 4},
    {"name": "s", "hid_usage_id": 22},
    {"name": "d", "hid_usage_id": 7},
)
EVALUATION_CODE_FILES = (
    "scripts/first_matched_suite_contract.py",
    "src/irene_brain/data/moving_shapes_dataset.py",
    "src/irene_brain/environments/moving_shapes.py",
    "src/irene_brain/training/batches.py",
    "src/irene_brain/training/checkpoint.py",
    "src/irene_brain/training/objective.py",
    "src/irene_brain/training/torch_system.py",
    "src/irene_brain/types.py",
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(value: bytes, *, name: str) -> dict[str, object]:
    try:
        decoded = value.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _exact_fields(
    value: Mapping[str, object], *, expected: set[str], name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields differ; missing={missing}, extra={extra}")


def evaluation_code_manifest(brain_root: Path) -> dict[str, object]:
    root = brain_root.resolve()
    source_files: dict[str, str] = {}
    for relative in EVALUATION_CODE_FILES:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("evaluation source escapes the brain root") from error
        source_files[relative] = _sha256_bytes(path.read_bytes())
    digest_payload = {
        "schema_version": 1,
        "source_files": source_files,
        "entrypoints": {
            "batch_iteration": (
                "scripts/first_matched_suite_contract.py:iter_suite_batches"
            ),
            "score_aggregation": (
                "scripts/first_matched_suite_contract.py:"
                "aggregate_movement_exact_match"
            ),
            "per_batch_metric": (
                "src/irene_brain/training/objective.py:"
                "ThoughtFieldObjective.movement_exact_match"
            ),
        },
    }
    digest = _sha256_bytes(
        EVALUATION_CODE_HASH_DOMAIN + canonical_json(digest_payload).encode("utf-8")
    )
    return {**digest_payload, "evaluation_code_sha256": digest}


def _validate_evaluation_code(value: Mapping[str, object]) -> None:
    _exact_fields(
        value,
        expected={
            "schema_version",
            "source_files",
            "entrypoints",
            "evaluation_code_sha256",
        },
        name="evaluation_code",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("evaluation_code.schema_version must be integer 1")
    source_files = value["source_files"]
    entrypoints = value["entrypoints"]
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("evaluation_code.source_files must be a nonempty object")
    if not isinstance(entrypoints, Mapping) or not entrypoints:
        raise ValueError("evaluation_code.entrypoints must be a nonempty object")
    for path, digest in source_files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("evaluation source paths must be nonempty strings")
        _sha256_text(digest, name=f"evaluation source {path}")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(entrypoint, str)
        or not entrypoint
        for name, entrypoint in entrypoints.items()
    ):
        raise ValueError("evaluation entrypoints must map nonempty strings")
    _sha256_text(
        value["evaluation_code_sha256"], name="evaluation_code_sha256"
    )
    payload = {
        "schema_version": value["schema_version"],
        "source_files": source_files,
        "entrypoints": entrypoints,
    }
    actual = _sha256_bytes(
        EVALUATION_CODE_HASH_DOMAIN + canonical_json(payload).encode("utf-8")
    )
    if actual != value["evaluation_code_sha256"]:
        raise ValueError("suite evaluation code self-digest mismatch")


def sample_manifest_sha256(sample_manifest: Mapping[str, object]) -> str:
    return _sha256_bytes(
        SAMPLE_HASH_DOMAIN + canonical_json(sample_manifest).encode("utf-8")
    )


def suite_manifest_sha256(payload: Mapping[str, object]) -> str:
    value = dict(payload)
    value.pop("suite_manifest_sha256", None)
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def dataset_config_from_suite(
    suite: Mapping[str, object],
) -> MovingShapesDatasetConfig:
    sample = suite.get("sample_manifest")
    if not isinstance(sample, Mapping):
        raise ValueError("suite sample_manifest must be an object")
    dataset = sample.get("dataset_manifest")
    if not isinstance(dataset, Mapping):
        raise ValueError("suite dataset_manifest must be an object")
    _exact_fields(
        dataset,
        expected={
            "schema_version",
            "generator_id",
            "origin",
            "license_record_id",
            "environment_family",
            "split",
            "seed_namespace_tag",
            "seed_offset",
            "sequence_count",
            "sequence_length",
            "total_transitions",
            "hazard_count",
            "tick_period_ns",
            "discount_hex",
            "teacher_inputs",
            "world_target",
            "value_target",
        },
        name="dataset_manifest",
    )
    if dataset["split"] != DatasetSplit.TEST.value:
        raise ValueError("held-out suite must use the test seed namespace")
    integer_fields = (
        "schema_version",
        "seed_namespace_tag",
        "seed_offset",
        "sequence_count",
        "sequence_length",
        "total_transitions",
        "hazard_count",
        "tick_period_ns",
    )
    if any(type(dataset[field]) is not int for field in integer_fields):
        raise ValueError("dataset manifest integer fields must be exact integers")
    string_fields = (
        "generator_id",
        "origin",
        "license_record_id",
        "environment_family",
        "split",
        "discount_hex",
        "teacher_inputs",
        "world_target",
        "value_target",
    )
    if any(not isinstance(dataset[field], str) for field in string_fields):
        raise ValueError("dataset manifest string fields must be strings")
    try:
        config = MovingShapesDatasetConfig(
            split=DatasetSplit.TEST,
            sequence_count=dataset["sequence_count"],
            sequence_length=dataset["sequence_length"],
            seed_offset=dataset["seed_offset"],
            hazard_count=dataset["hazard_count"],
            tick_period_ns=dataset["tick_period_ns"],
            discount=float.fromhex(dataset["discount_hex"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("suite dataset configuration is invalid") from error
    if config.manifest_dict() != dict(dataset):
        raise ValueError("suite dataset manifest is not derived from MovingShapes")
    expected_dataset_digest = sample.get("dataset_manifest_sha256")
    _sha256_text(expected_dataset_digest, name="dataset_manifest_sha256")
    if dataset_manifest_sha256(config) != expected_dataset_digest:
        raise ValueError("suite dataset manifest digest mismatch")
    return config


def validate_suite_manifest(
    suite: Mapping[str, object], *, brain_root: Path | None = None
) -> dict[str, object]:
    _exact_fields(
        suite,
        expected={
            "schema_version",
            "suite_id",
            "scope",
            "weight",
            "evaluated_samples",
            "sample_manifest",
            "evaluation_sample_manifest_sha256",
            "metric",
            "evaluation_code",
            "suite_manifest_sha256",
        },
        name="suite manifest",
    )
    if type(suite["schema_version"]) is not int or suite["schema_version"] != 1:
        raise ValueError("suite schema_version must be integer 1")
    if not isinstance(suite["suite_id"], str) or not suite["suite_id"]:
        raise ValueError("suite_id must be a nonempty string")
    if not isinstance(suite["scope"], str) or not suite["scope"]:
        raise ValueError("suite scope must be a nonempty string")
    if type(suite["weight"]) is not float or suite["weight"] != 1.0:
        raise ValueError("the preregistered suites must have equal unit weight")
    if type(suite["evaluated_samples"]) is not int or suite["evaluated_samples"] < 1:
        raise ValueError("suite evaluated_samples must be a positive integer")
    config = dataset_config_from_suite(suite)
    sample = suite["sample_manifest"]
    assert isinstance(sample, Mapping)
    _exact_fields(
        sample,
        expected={
            "schema_version",
            "dataset_manifest",
            "dataset_manifest_sha256",
            "burn_in_steps",
            "evaluated_transition_indices",
            "evaluated_samples",
        },
        name="sample_manifest",
    )
    if type(sample["schema_version"]) is not int or sample["schema_version"] != 1:
        raise ValueError("sample schema_version must be integer 1")
    burn_in = sample["burn_in_steps"]
    if type(burn_in) is not int or not 0 <= burn_in < config.sequence_length:
        raise ValueError("suite burn-in is invalid")
    transition_indices = sample["evaluated_transition_indices"]
    if transition_indices != {
        "start_inclusive": burn_in,
        "stop_exclusive": config.sequence_length,
    }:
        raise ValueError("suite transition selection is not the post-burn-in suffix")
    evaluated = config.sequence_count * (config.sequence_length - burn_in)
    if type(sample["evaluated_samples"]) is not int:
        raise ValueError("sample evaluated_samples must be an integer")
    if sample["evaluated_samples"] != evaluated or suite["evaluated_samples"] != evaluated:
        raise ValueError("suite evaluated sample count mismatch")
    sample_digest = sample_manifest_sha256(sample)
    _sha256_text(
        suite["evaluation_sample_manifest_sha256"],
        name="evaluation_sample_manifest_sha256",
    )
    if suite["evaluation_sample_manifest_sha256"] != sample_digest:
        raise ValueError("suite sample selection digest mismatch")
    metric = suite["metric"]
    if not isinstance(metric, Mapping):
        raise ValueError("suite metric must be an object")
    _exact_fields(
        metric,
        expected={
            "metric_id",
            "model_exit",
            "prediction_rule",
            "target_rule",
            "movement_keys",
            "per_sample_score",
            "aggregation",
            "normalized_range",
        },
        name="metric",
    )
    expected_metric = {
        "metric_id": "movement_exact_match",
        "model_exit": "final_anytime_exit",
        "prediction_rule": "button_logit > 0.0",
        "target_rule": "target_control > 0.5",
        "movement_keys": list(MOVEMENT_KEYS),
        "per_sample_score": (
            "1 iff predicted and target active sets agree exactly on W/A/S/D; else 0"
        ),
        "aggregation": "sample-weighted arithmetic mean over post-burn-in decisions",
        "normalized_range": [0.0, 1.0],
    }
    movement_keys = metric["movement_keys"]
    normalized_range = metric["normalized_range"]
    if not isinstance(movement_keys, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"name", "hid_usage_id"}
        or not isinstance(item["name"], str)
        or type(item["hid_usage_id"]) is not int
        for item in movement_keys
    ):
        raise ValueError("metric movement_keys have incompatible types")
    if (
        not isinstance(normalized_range, list)
        or len(normalized_range) != 2
        or any(type(value) is not float for value in normalized_range)
    ):
        raise ValueError("metric normalized_range must contain exact floats")
    if dict(metric) != expected_metric:
        raise ValueError("suite metric differs from movement_exact_match contract")
    evaluation_code = suite["evaluation_code"]
    if not isinstance(evaluation_code, Mapping):
        raise ValueError("suite evaluation_code must be an object")
    _validate_evaluation_code(evaluation_code)
    if brain_root is not None and dict(evaluation_code) != evaluation_code_manifest(brain_root):
        raise ValueError("suite evaluation code differs from the pinned workspace bytes")
    _sha256_text(suite["suite_manifest_sha256"], name="suite_manifest_sha256")
    if suite_manifest_sha256(suite) != suite["suite_manifest_sha256"]:
        raise ValueError("suite manifest self-digest mismatch")
    return dict(suite)


def load_suite_manifest(path: Path, *, brain_root: Path | None = None) -> dict[str, object]:
    raw = path.read_bytes()
    suite = strict_json_bytes(raw, name=str(path))
    if raw != (canonical_json(suite) + "\n").encode("utf-8"):
        raise ValueError("suite manifest file is not canonical JSON plus one LF")
    return validate_suite_manifest(suite, brain_root=brain_root)


def iter_suite_batches(
    suite: Mapping[str, object], *, batch_size: int = 1
) -> Iterator[TrajectoryBatch]:
    """Yield the exact registered sequence matrix without loading a model."""

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    validated = validate_suite_manifest(suite)
    config = dataset_config_from_suite(validated)
    sample = validated["sample_manifest"]
    assert isinstance(sample, Mapping)
    burn_in = sample["burn_in_steps"]
    assert isinstance(burn_in, int)
    dataset = MovingShapesSequenceDataset(config)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        yield TrajectoryBatch(
            split=DatasetSplit.TEST.value,
            burn_in_steps=burn_in,
            sequences=tuple(dataset[index] for index in range(start, stop)),
        )


def aggregate_movement_exact_match(results: Iterable[object]) -> float:
    """Sample-weight the existing per-batch final-exit exact-match metric."""

    weighted = 0.0
    samples = 0
    for result in results:
        result_samples = getattr(result, "samples", None)
        metrics = getattr(result, "metrics", None)
        if type(result_samples) is not int or result_samples < 1:
            raise ValueError("each evaluation result must have positive integer samples")
        if not isinstance(metrics, Mapping) or "movement_exact_match" not in metrics:
            raise ValueError("evaluation result lacks movement_exact_match")
        score = metrics["movement_exact_match"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("movement_exact_match must be numeric")
        score = float(score)
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("movement_exact_match must be finite and in [0, 1]")
        weighted += score * result_samples
        samples += result_samples
    if samples == 0:
        raise ValueError("at least one evaluation result is required")
    return weighted / samples


__all__ = [
    "EVALUATION_CODE_FILES",
    "MOVEMENT_KEYS",
    "aggregate_movement_exact_match",
    "canonical_json",
    "dataset_config_from_suite",
    "evaluation_code_manifest",
    "iter_suite_batches",
    "load_suite_manifest",
    "sample_manifest_sha256",
    "strict_json_bytes",
    "suite_manifest_sha256",
    "validate_suite_manifest",
]
