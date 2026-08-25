"""CPU-only one-arm diagnostic for held-out V2.1 predictive learning."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from irene_brain.training.batches import MazeChaseBatchSource
from irene_brain.v2.trajectory_objective import AllActionV2TrajectoryObjective
from run_provenance import apply_deterministic_mode
import stage_v21_predictive_trajectory_smoke as stage
from stage_v21_predictive_trajectory_smoke import (
    DATASET_MANIFEST,
    EPOCHS,
    dataset_config,
)


_SOURCE_BUNDLE_PIPELINE_FILES = (
    "brain/scripts/run_provenance.py",
    "brain/scripts/stage_v20_core_v2_deployed_loss_baseline.py",
    "brain/scripts/v21_cpu_heldout_learning_diagnostic.py",
    "brain/scripts/stage_v21_predictive_trajectory_smoke.py",
    "brain/scripts/v21f_hazard_head_probe.py",
    "brain/scripts/v21h_verify_calibrated_checkpoint.py",
)


def _source_bundle_files(project_root: Path) -> tuple[str, ...]:
    """Return the complete, stable set of Python sources for this pipeline."""

    package_root = project_root / "brain" / "src" / "irene_brain"
    package_files = (
        path.relative_to(project_root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file()
    )
    return tuple(sorted({*_SOURCE_BUNDLE_PIPELINE_FILES, *package_files}))


def _hash_source_bundle_files(
    project_root: Path,
    relative_files: tuple[str, ...],
) -> dict[str, object]:
    digest = sha256(b"IRV21SOURCEBUNDLE\x01")
    files: dict[str, str] = {}
    for relative in relative_files:
        path = project_root / relative
        content = path.read_bytes()
        file_sha = sha256(content).hexdigest()
        files[relative] = file_sha
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha))
    return {"schema_version": 1, "sha256": digest.hexdigest(), "files": files}


def _source_bundle(project_root: Path) -> dict[str, object]:
    return _hash_source_bundle_files(
        project_root,
        _source_bundle_files(project_root),
    )


def _assert_diagnostic_targets_available(output: Path) -> None:
    checkpoint_directory = Path(f"{output}.checkpoints")
    if output.exists() or checkpoint_directory.exists():
        raise FileExistsError(
            "refusing to replace an existing diagnostic artifact or checkpoint directory"
        )


def _write_json_create_only(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finalize_checkpoint_provenance(
    checkpoint_record: dict[str, object],
    *,
    source_bundle: dict[str, object],
    dataset_config_dict: dict[str, object],
) -> None:
    checkpoint_path = Path(str(checkpoint_record["path"])).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("diagnostic checkpoint root must be a mapping")
    payload["source_bundle"] = source_bundle
    payload["dataset_config"] = dataset_config_dict
    temporary = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.provenance.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    checkpoint_record["sha256"] = sha256(checkpoint_path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "decision_only_zero_pe",
            "predictive_zero_pe",
            "predictive_normal_pe",
        ),
        default="predictive_normal_pe",
    )
    parser.add_argument("--train-seed", type=int, default=stage.TRAIN_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--latent-comparison",
        choices=("raw_mse_v0", "cosine_distance_v1"),
        default="raw_mse_v0",
    )
    parser.add_argument(
        "--reward-comparison",
        choices=("raw_mse_v0", "symlog_mse_v1"),
        default="raw_mse_v0",
    )
    parser.add_argument(
        "--behavior-policy",
        choices=("teacher", "balanced_intervention_v1"),
        default="teacher",
    )
    parser.add_argument("--behavior-intervention-rate", type=float, default=0.0)
    parser.add_argument(
        "--outcome-action-conditioning",
        choices=("thought_only_v0", "factual_or_proposal_v1"),
        default="thought_only_v0",
    )
    parser.add_argument(
        "--outcome-architecture",
        choices=("legacy_hypothesis_world_v0", "all_action_table_v1"),
        default="legacy_hypothesis_world_v0",
    )
    parser.add_argument(
        "--reward-prediction",
        choices=("scalar_v0", "symlog_twohot_v1"),
        default="scalar_v0",
    )
    parser.add_argument(
        "--prediction-error-fusion",
        choices=("latent_only_v0", "latent_outcome_surprise_v1"),
        default="latent_only_v0",
    )
    parser.add_argument(
        "--counterfactual-targets",
        choices=("none", "all_actions_v1"),
        default="none",
    )
    parser.add_argument(
        "--hazard-outcome-path",
        choices=("shared_outcome_v0", "dedicated_stopgrad_v1"),
        default="shared_outcome_v0",
    )
    parser.add_argument("--hazard-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 32:
        parser.error("--epochs must be in [1, 32]")
    if not 0.0 < args.hazard_weight <= 10.0:
        parser.error("--hazard-weight must be in (0, 10]")
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    candidate_dataset_config = replace(
        dataset_config(mini=False),
        behavior_policy=args.behavior_policy,
        behavior_intervention_rate=args.behavior_intervention_rate,
        counterfactual_targets=args.counterfactual_targets,
    )
    source = MazeChaseBatchSource(candidate_dataset_config)
    if (
        args.behavior_policy == "teacher"
        and args.behavior_intervention_rate == 0.0
        and args.counterfactual_targets == "none"
        and source.manifest_sha256 != DATASET_MANIFEST
    ):
        raise RuntimeError("dataset manifest drift")
    output_path = Path(args.output).expanduser().resolve()
    _assert_diagnostic_targets_available(output_path)
    output = str(output_path)
    Path(f"{output}.checkpoints").mkdir(parents=True, exist_ok=False)
    original_model_config = stage.model_config
    predictive = args.arm != "decision_only_zero_pe"
    intervention = "normal" if args.arm == "predictive_normal_pe" else "zero"

    class DiagnosticSourceAdapter:
        """Expose the explicit all-action iterator to the unchanged loop."""

        def __init__(self, wrapped: MazeChaseBatchSource) -> None:
            self._wrapped = wrapped
            self.config = wrapped.config

        @property
        def manifest_sha256(self) -> str:
            return self._wrapped.manifest_sha256

        def iter_batches(self, **kwargs: object):
            return self._wrapped.iter_all_action_batches(**kwargs)

    training_source = (
        DiagnosticSourceAdapter(source)
        if args.counterfactual_targets == "all_actions_v1"
        else source
    )

    def diagnostic_model_config(predictive: bool):
        return replace(
            original_model_config(predictive),
            latent_comparison=args.latent_comparison,
            reward_comparison=args.reward_comparison,
            outcome_action_conditioning=args.outcome_action_conditioning,
            outcome_architecture=args.outcome_architecture,
            reward_prediction=args.reward_prediction,
            prediction_error_fusion=args.prediction_error_fusion,
            hazard_outcome_path=args.hazard_outcome_path,
            hazard_weight=args.hazard_weight,
        )

    with (
        patch.object(stage, "model_config", diagnostic_model_config),
        patch.object(stage, "TRAIN_SEED", args.train_seed),
        patch.object(
            stage,
            "V2TrajectoryObjective",
            AllActionV2TrajectoryObjective
            if args.counterfactual_targets == "all_actions_v1"
            else stage.V2TrajectoryObjective,
        ),
    ):
        arm = stage.run_arm(
            args.arm,
            predictive,
            intervention,
            training_source,
            torch.device("cpu"),
            args.epochs,
            output + ".checkpoints",
        )
    project_root = Path(__file__).resolve().parents[2]
    source_bundle = _source_bundle(project_root)
    _finalize_checkpoint_provenance(
        arm["checkpoint"],
        source_bundle=source_bundle,
        dataset_config_dict=asdict(candidate_dataset_config),
    )
    initial = arm["initial_validation"]["components"]
    final = arm["final_validation"]["components"]
    result = {
        "schema_version": 1,
        "mode": "cpu_heldout_learning_diagnostic",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "train_seed": args.train_seed,
        "epochs": args.epochs,
        "dataset_manifest": source.manifest_sha256,
        "latent_comparison": args.latent_comparison,
        "reward_comparison": args.reward_comparison,
        "behavior_policy": args.behavior_policy,
        "behavior_intervention_rate": args.behavior_intervention_rate,
        "outcome_action_conditioning": args.outcome_action_conditioning,
        "outcome_architecture": args.outcome_architecture,
        "reward_prediction": args.reward_prediction,
        "prediction_error_fusion": args.prediction_error_fusion,
        "counterfactual_targets": args.counterfactual_targets,
        "hazard_outcome_path": args.hazard_outcome_path,
        "hazard_weight": args.hazard_weight,
        "test_split_opened": False,
        "source_bundle": source_bundle,
        "arm": arm,
        "heldout_loss_ratios": {
            name: round(final[name] / initial[name], 6)
            for name in ("next_latent", "reward", "hazard")
        },
    }
    _write_json_create_only(output_path, result)
    print(f"DONE -> {output}", flush=True)


if __name__ == "__main__":
    main()
