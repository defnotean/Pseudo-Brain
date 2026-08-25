"""Integrated held-out verification for a hazard-calibrated V2.1h checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from irene_brain.training.batches import MazeChaseBatchSource
from irene_brain.v2 import CONFIG_B_PREDICTIVE
from irene_brain.v2.trajectory_objective import AllActionV2TrajectoryObjective
from run_provenance import apply_deterministic_mode
from stage_v21_predictive_trajectory_smoke import dataset_config, evaluate
from v21f_hazard_head_probe import (
    _checkpoint_model,
    _sha256_file,
    _write_atomic,
)
from v21_cpu_heldout_learning_diagnostic import _source_bundle


_ALLOWED_HAZARD_STATE_PREFIXES = (
    "world_model.outcome_model.hazard_state_trunk.",
    "world_model.outcome_model.hazard_action_embedding.",
    "world_model.outcome_model.hazard_outcome_trunk.",
    "world_model.outcome_model.hazard_head.",
)


class _AllActionSourceAdapter:
    def __init__(self, source: MazeChaseBatchSource) -> None:
        self._source = source
        self.config = source.config

    @property
    def manifest_sha256(self) -> str:
        return self._source.manifest_sha256

    def iter_batches(self, **kwargs: object):
        return self._source.iter_all_action_batches(**kwargs)


def _publish_result(path: Path, payload: dict[str, object]) -> None:
    """Write evidence first, then return failure to automation when unqualified."""

    _write_atomic(path, payload)
    qualification = payload.get("qualification")
    passed = isinstance(qualification, dict) and qualification.get("passed") is True
    if not passed:
        print(f"FAIL -> {path.resolve()}", flush=True)
        raise SystemExit(1)
    print(f"DONE -> {path.resolve()}", flush=True)


def _assert_output_available(path: Path) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {path}")


def _validate_parent_checkpoint_sha(
    parent_path: Path,
    calibration: dict[str, object],
) -> str:
    parent_sha256 = _sha256_file(parent_path)
    if calibration.get("parent_checkpoint_sha256") != parent_sha256:
        raise ValueError("parent checkpoint hash does not match calibration provenance")
    return parent_sha256


def _source_bundle_matches(
    child_source_bundle: object,
    parent_source_bundle: object,
    current_source_bundle: dict[str, object],
    calibration: dict[str, object],
) -> bool:
    return bool(
        isinstance(child_source_bundle, dict)
        and child_source_bundle.get("schema_version") == 1
        and isinstance(child_source_bundle.get("sha256"), str)
        and len(child_source_bundle["sha256"]) == 64
        and child_source_bundle == parent_source_bundle
        and child_source_bundle == current_source_bundle
        and calibration.get("source_bundle_sha256")
        == child_source_bundle.get("sha256")
    )


def _validate_calibrated_state_dicts(
    parent_state: dict[str, torch.Tensor],
    child_state: dict[str, torch.Tensor],
    recorded_changed_state_keys: object,
) -> tuple[str, ...]:
    if parent_state.keys() != child_state.keys():
        raise ValueError("parent and calibrated state dictionaries differ in keys")
    changed = tuple(
        name
        for name in parent_state
        if not torch.equal(parent_state[name], child_state[name])
    )
    if not changed or any(
        not name.startswith(_ALLOWED_HAZARD_STATE_PREFIXES) for name in changed
    ):
        raise ValueError("calibration changed a non-hazard tensor")
    if list(changed) != recorded_changed_state_keys:
        raise ValueError("recorded changed_state_keys do not match tensor diff")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    args = parser.parse_args()

    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    parent_path = Path(args.parent_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    _assert_output_available(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    if not isinstance(parent, dict):
        raise ValueError("parent checkpoint root must be a mapping")
    calibration = checkpoint.get("hazard_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("checkpoint lacks hazard_calibration provenance")
    if calibration.get("schema_version") != 1:
        raise ValueError("unsupported hazard calibration schema")
    if calibration.get("target_mode") != "all_actions_v1":
        raise ValueError("calibration target_mode must be all_actions_v1")
    if calibration.get("training_scope") != "all_hazard_path":
        raise ValueError("calibration training_scope must be all_hazard_path")
    child_source_bundle = checkpoint.get("source_bundle")
    parent_source_bundle = parent.get("source_bundle")
    current_source_bundle = _source_bundle(Path(args.source_root).resolve())
    source_bundle_valid = _source_bundle_matches(
        child_source_bundle,
        parent_source_bundle,
        current_source_bundle,
        calibration,
    )
    _validate_parent_checkpoint_sha(parent_path, calibration)
    model, config = _checkpoint_model(checkpoint)
    parent_model, parent_config = _checkpoint_model(parent)
    if config != parent_config:
        raise ValueError("calibration changed the model configuration")
    if config.hazard_outcome_path != "dedicated_stopgrad_v1":
        raise ValueError("calibrated checkpoint must use dedicated_stopgrad_v1")

    source = MazeChaseBatchSource(
        replace(dataset_config(mini=False), counterfactual_targets="all_actions_v1")
    )
    if checkpoint.get("dataset_manifest") != parent.get("dataset_manifest"):
        raise ValueError("calibration changed the checkpoint dataset manifest")
    if checkpoint.get("dataset_config") != parent.get("dataset_config"):
        raise ValueError("calibration changed the checkpoint dataset configuration")
    if checkpoint.get("dataset_manifest") != source.manifest_sha256:
        raise ValueError("calibrated checkpoint dataset manifest mismatch")
    if calibration.get("dataset_manifest_sha256") != source.manifest_sha256:
        raise ValueError("calibration dataset manifest mismatch")

    parent_state = parent_model.state_dict()
    child_state = model.state_dict()
    recomputed_changed = _validate_calibrated_state_dicts(
        parent_state,
        child_state,
        calibration.get("changed_state_keys"),
    )

    objective = AllActionV2TrajectoryObjective(
        model,
        flags=CONFIG_B_PREDICTIVE,
        prediction_error_intervention="normal",
    )
    adapter = _AllActionSourceAdapter(source)
    validation = evaluate(
        objective,
        adapter,
        torch.device("cpu"),
        split="validation",
    )
    parent_validation = evaluate(
        AllActionV2TrajectoryObjective(
            parent_model,
            flags=CONFIG_B_PREDICTIVE,
            prediction_error_intervention="normal",
        ),
        adapter,
        torch.device("cpu"),
        split="validation",
    )
    parent_components = parent_validation["components"]
    child_components = validation["components"]
    parent_metrics = parent_validation["metrics"]
    child_metrics = validation["metrics"]
    checks = {
        "calibration_screen_passed": calibration.get(
            "calibration_screen_passed"
        )
        is True,
        "only_dedicated_hazard_tensors_changed": True,
        "source_bundle_bound_and_parent_matched": source_bundle_valid,
        "hazard_loss_not_worse": (
            child_components["hazard"] <= parent_components["hazard"]
        ),
        "next_loss_preserved_within_1pct": (
            child_components["next_latent"]
            <= parent_components["next_latent"] * 1.01
        ),
        "reward_loss_preserved_within_1pct": (
            child_components["reward"] <= parent_components["reward"] * 1.01
        ),
        "decision_loss_preserved_within_1pct": (
            child_components["decision"] <= parent_components["decision"] * 1.01
        ),
        "next_cosine_preserved_within_0_01": (
            child_metrics["next_latent_cosine"]
            >= parent_metrics["next_latent_cosine"] - 0.01
        ),
        "reward_mae_preserved_within_2pct": (
            child_metrics["reward_mae"] <= parent_metrics["reward_mae"] * 1.02
        ),
        "factual_hazard_margin_at_least_0_02": (
            child_metrics["hazard_positive_probability"]
            >= child_metrics["hazard_negative_probability"] + 0.02
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "v21h_calibrated_checkpoint_development_verification",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "test_split_opened": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "parent_sha256": calibration.get("parent_checkpoint_sha256"),
            "non_hazard_tensors_byte_equal": calibration.get(
                "non_hazard_tensors_byte_equal"
            ),
            "changed_state_keys": calibration.get("changed_state_keys"),
        },
        "dataset_manifest": source.manifest_sha256,
        "evaluation_namespace": "development_validation_reused_after_calibration",
        "confirmatory": False,
        "parent_development_validation": parent_validation,
        "development_validation": validation,
        "qualification": {"checks": checks, "passed": all(checks.values())},
    }
    _publish_result(output_path, payload)


if __name__ == "__main__":
    main()
