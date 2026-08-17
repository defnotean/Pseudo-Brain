"""Target-blind, create-only preregistration for RCQ-v2 final qualification.

This builder hashes source and configuration identities and derives procedural
dataset *configuration* manifests only.  It never constructs a sequence
dataset, reads a generated example, computes a target, or opens a sealed final
TEST range.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import os
from pathlib import Path
import stat
from typing import Sequence

from ..data import DatasetSplit
from ..runtime.policy import Capability
from ..training.config import TrainingConfig, load_training_config
from .rcq_v2 import (
    DEVELOPMENT_GATE_ID,
    VALUE_ABSOLUTE_MAX_MSE,
    VALUE_DEVELOPMENT_GATE_ID,
    VALUE_DEVELOPMENT_TARGET_VARIANCE,
    VALUE_IMPROVEMENT_RATIO,
    VALUE_TRAIN_CONDITIONAL_BASELINE_MSE,
    RCQInputError,
)
from .rcq_v2_final import (
    BOOTSTRAP_LOWER_INDEX,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DONOR_OFFSET,
    DONOR_MATCHING_SEED,
    DONOR_SEQUENCES,
    FINAL_DECISIONS,
    FINAL_EVALUATOR_ID,
    FINAL_STAGE_INDEX,
    FINAL_STEP,
    FUTURE_CAMPAIGN_OFFSET,
    GUARD_END,
    RECIPIENT_OFFSET,
    RECIPIENT_SEQUENCES,
    RETIRED_END,
    RCQRegistration,
    _canonical,
    _final_range_claim_id,
    _publish_no_replace,
    _registered_thresholds,
    _validate_registration,
)
from .rcq_v2_torch import (
    _file_sha256,
    _inside_file,
    _dataset_manifest_from_training_config,
    _metadata_only_batch_source_manifest_sha256,
    _plain_resolve,
    _require_registered_config,
    _source_tree_sha256_exact,
    _REGISTERED_CONFIG_RELATIVE_PATH,
    _REGISTRATION_RELATIVE_PATH,
    evaluator_bundle_sha256,
)


def _resolve_inputs(
    training_release_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> tuple[Path, Path, Path]:
    release = _plain_resolve(
        training_release_root,
        name="training_release_root",
        kind="directory",
    )
    source = _plain_resolve(
        release / "brain" / "src" / "irene_brain",
        name="training source root",
        kind="directory",
    )
    config = _inside_file(config_path, release, name="config_path")
    expected_config = _plain_resolve(
        release / _REGISTERED_CONFIG_RELATIVE_PATH,
        name="registered training config",
        kind="file",
    )
    if config != expected_config:
        raise RCQInputError("config_path is not the fixed RCQ-v2 reference config")
    package = importlib.import_module("irene_brain")
    package_file = getattr(package, "__file__", None)
    if not package_file or _plain_resolve(
        package_file,
        name="imported irene_brain package",
        kind="file",
    ).parent != source:
        raise RCQInputError("imported irene_brain package is not the identified release")
    return release, source, config


def _dataset_manifest(
    config: TrainingConfig,
    *,
    split: DatasetSplit,
    sequence_count: int,
    seed_offset: int,
) -> str:
    """Hash virtual-dataset geometry without constructing the virtual dataset."""

    return _dataset_manifest_from_training_config(
        config,
        split=split,
        sequence_count=sequence_count,
        seed_offset=seed_offset,
    )


def _registration_payload(
    *,
    config: TrainingConfig,
    config_file: Path,
    source_root: Path,
) -> dict[str, object]:
    scored_per_sequence = config.dataset.sequence_length - config.dataset.burn_in_steps
    train_manifest = _dataset_manifest(
        config,
        split=DatasetSplit.TRAIN,
        sequence_count=config.dataset.train_sequences,
        seed_offset=config.dataset.seed_offset,
    )
    development_manifest = _dataset_manifest(
        config,
        split=DatasetSplit.VALIDATION,
        sequence_count=config.dataset.validation_sequences,
        seed_offset=config.dataset.seed_offset,
    )
    recipient_manifest = _dataset_manifest(
        config,
        split=DatasetSplit.TEST,
        sequence_count=RECIPIENT_SEQUENCES,
        seed_offset=RECIPIENT_OFFSET,
    )
    donor_manifest = _dataset_manifest(
        config,
        split=DatasetSplit.TEST,
        sequence_count=DONOR_SEQUENCES,
        seed_offset=DONOR_OFFSET,
    )
    return {
        "schema_version": 2,
        "qualification_id": "rcq_v2_reference_v2",
        "evaluator_id": FINAL_EVALUATOR_ID,
        "config_canonical_sha256": config.config_sha256,
        "config_raw_sha256": _file_sha256(config_file),
        "source_tree_sha256": _source_tree_sha256_exact(source_root),
        "evaluator_bundle_sha256": evaluator_bundle_sha256(source_root),
        "batch_source_manifest_sha256": (
            _metadata_only_batch_source_manifest_sha256(config)
        ),
        "run_seed": config.run.seed,
        "run_id": f"{config.run.name}-seed-{config.run.seed}",
        "model_factory": config.run.model_factory,
        "joint_end_step": config.stages[0].end_optimizer_step,
        "final_step": config.run.max_optimizer_steps,
        "sequence_length": config.dataset.sequence_length,
        "burn_in_steps": config.dataset.burn_in_steps,
        "hazard_count": config.dataset.hazard_count,
        "tick_period_ns": config.dataset.tick_period_ns,
        "discount_hex": config.dataset.discount.hex(),
        "slices": {
            "train": {
                "split": "train",
                "local_start": config.dataset.seed_offset,
                "local_end": (
                    config.dataset.seed_offset + config.dataset.train_sequences
                ),
                "sequences": config.dataset.train_sequences,
                "scored_decisions": (
                    config.dataset.train_sequences * scored_per_sequence
                ),
                "manifest_sha256": train_manifest,
            },
            "development": {
                "split": "validation",
                "local_start": config.dataset.seed_offset,
                "local_end": (
                    config.dataset.seed_offset + config.dataset.validation_sequences
                ),
                "sequences": config.dataset.validation_sequences,
                "scored_decisions": (
                    config.dataset.validation_sequences * scored_per_sequence
                ),
                "manifest_sha256": development_manifest,
            },
            "final_recipient": {
                "split": "test",
                "local_start": RECIPIENT_OFFSET,
                "local_end": RECIPIENT_OFFSET + RECIPIENT_SEQUENCES,
                "sequences": RECIPIENT_SEQUENCES,
                "scored_decisions": FINAL_DECISIONS,
                "manifest_sha256": recipient_manifest,
            },
            "final_donor_only": {
                "split": "test",
                "local_start": DONOR_OFFSET,
                "local_end": DONOR_OFFSET + DONOR_SEQUENCES,
                "sequences": DONOR_SEQUENCES,
                "scored_decisions": 0,
                "manifest_sha256": donor_manifest,
            },
        },
        "bootstrap": {
            "cluster": (
                "unique_recipient_donor_sequence_pair_with_all_six_paired_decisions"
            ),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "one_sided_confidence": "0x1.e666666666666p-1",
            "sorted_lower_index_zero_based": BOOTSTRAP_LOWER_INDEX,
        },
        "thresholds": _registered_thresholds(),
        "guard_band": {
            "local_start": RETIRED_END,
            "local_end": GUARD_END,
            "status": "unused",
        },
        "future_campaign_offset": FUTURE_CAMPAIGN_OFFSET,
        "runtime_protocol": {
            "device": "cuda",
            "precision": "bfloat16",
            "allow_tf32": True,
            "deterministic_algorithms": True,
            "compile_model": False,
            "num_workers": 0,
            "world_size": 1,
            "autocast": "torch.autocast(device_type=cuda,dtype=bfloat16)",
            "actuator_exit": "final",
            "button_activation": "logit_strictly_greater_than_zero",
            "target_button_activation": "target_control_strictly_greater_than_0.5",
            "continuous_output": "final_action_control",
            "recurrent_conditions": "independent_normal_and_rgb_deranged",
            "rgb_intervention": (
                "replace_rgb_only_from_one_donor_sequence_all_timesteps"
            ),
            "donor_assignment": (
                "target_blind_one_to_one_sequence_minimum_total_previous_wasd_bit_hamming"
            ),
            "donor_tie_break": (
                "sha256_seeded_donor_order_then_lowest_hungarian_column"
            ),
            "donor_matching_seed": DONOR_MATCHING_SEED,
            "donor_rgb_constraint": (
                "all_eight_corresponding_frame_sha256_values_distinct"
            ),
        },
        "development_gates": {
            "entry": {
                "gate": DEVELOPMENT_GATE_ID,
                "optimizer_step": config.stages[0].end_optimizer_step,
            },
            "completion": {
                "gate": VALUE_DEVELOPMENT_GATE_ID,
                "optimizer_step": FINAL_STEP,
                "entry_gate": DEVELOPMENT_GATE_ID,
                "entry_optimizer_step": config.stages[0].end_optimizer_step,
                "train_timestep_previous_wasd_dev_mse_hex": (
                    VALUE_TRAIN_CONDITIONAL_BASELINE_MSE.hex()
                ),
                "absolute_max_mse_hex": VALUE_ABSOLUTE_MAX_MSE.hex(),
                "dev_target_variance_hex": VALUE_DEVELOPMENT_TARGET_VARIANCE.hex(),
                "entry_improvement_ratio_hex": VALUE_IMPROVEMENT_RATIO.hex(),
            },
        },
        "workspace_protocol": {
            "contract": "pseudo-brain-workspace-v2",
            "host_account_home_relative_path": "projects/pseudo-brain",
            "host_marker_relative_path": ".pseudo-brain-workspace-v2",
            "host_marker_exact_utf8": "pseudo-brain-workspace-v2\n",
            "dedicated_dispatcher_action": "rcq_v2_final_once_v1",
            "preclaim_dispatcher_action": "rcq_v2_preclaim_v1",
            "receipt_verifier_dispatcher_action": "rcq_v2_verify_receipt_v1",
            "container_release_root": "/workspace/repo",
            "container_run_root": "/workspace/run",
            "container_claim_registry_root": "/workspace/final-claims",
            "container_pin_root": "/workspace/pins",
            "host_pin_directory_relative_path": (
                "qualification-pins/rcq-v2-reference-v2"
            ),
            "pretraining_pin_filename": "pretraining.json",
            "final_authorization_filename": "final-authorization.json",
            "registration_release_relative_path": (
                "registrations/rcq-v2-reference-v2.json"
            ),
            "readiness_receipt_relative_path": (
                "preclaim-readiness/rcq-v2-reference-v2.json"
            ),
        },
        "receipt_directory": f"final-claims/{_final_range_claim_id()}",
        "config_test_field_status": (
            "disabled_retired_placeholder_generic_trainer_test_forbidden"
        ),
        "scope": (
            "one-seed open-loop teacher-forced reference-policy qualification; "
            "not closed-loop gameplay, architecture superiority, or causal thought use"
        ),
    }


def build_target_blind_registration(
    *,
    training_release_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> tuple[bytes, str]:
    """Return canonical registration bytes and digest without opening TEST data."""

    _release, source_root, config_file = _resolve_inputs(
        training_release_root,
        config_path,
    )
    config = load_training_config(config_file)
    payload = _registration_payload(
        config=config,
        config_file=config_file,
        source_root=source_root,
    )
    _validate_registration(payload)
    encoded = (_canonical(payload) + "\n").encode("utf-8")
    digest = sha256(encoded).hexdigest()
    registration = RCQRegistration(payload=payload, sha256=digest)
    _require_registered_config(config, config_file, registration)
    if (
        _file_sha256(config_file) != payload["config_raw_sha256"]
        or _source_tree_sha256_exact(source_root) != payload["source_tree_sha256"]
        or evaluator_bundle_sha256(source_root) != payload["evaluator_bundle_sha256"]
    ):
        raise RCQInputError("a preregistration input changed during target-blind hashing")
    return encoded, digest


def _safe_registration_parent(value: str | os.PathLike[str]) -> Path:
    """Resolve an existing registration directory that may hold historical files.

    Claim receipt roots must be empty and exclusively owned. Registration
    parents are different: only the live filename is create-once, and sibling
    historical JSON files must remain.
    """

    absolute = Path(value).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists():
            metadata = current.stat(follow_symlinks=False)
            reparse = getattr(metadata, "st_file_attributes", 0) & getattr(
                stat,
                "FILE_ATTRIBUTE_REPARSE_POINT",
                0,
            )
            if current.is_symlink() or reparse:
                raise RCQInputError(
                    "registration parent cannot traverse a link or reparse point"
                )
    if not absolute.exists() or not absolute.is_dir():
        raise RCQInputError("registration parent must be an existing directory")
    resolved = absolute.resolve(strict=True)
    metadata = resolved.stat(follow_symlinks=False)
    reparse = getattr(metadata, "st_file_attributes", 0) & getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0,
    )
    if not resolved.is_dir() or resolved.is_symlink() or reparse:
        raise RCQInputError(
            "registration parent must resolve to a non-symlink directory"
        )
    return resolved


def write_target_blind_registration(
    *,
    training_release_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> tuple[Path, str]:
    """Create one immutable registration file and return its external pin."""

    release, source_root, config_file = _resolve_inputs(
        training_release_root,
        config_path,
    )
    target = release / _REGISTRATION_RELATIVE_PATH
    if target.exists() or target.is_symlink():
        raise RCQInputError("registration output already exists; replacement is forbidden")
    encoded, digest = build_target_blind_registration(
        training_release_root=release,
        config_path=config_file,
    )
    config = load_training_config(config_file)
    config.resource_policy.require(Capability.ARTIFACT_WRITE)
    parent = _safe_registration_parent(target.parent)
    target = parent / target.name
    if target.exists() or target.is_symlink():
        raise RCQInputError("registration output already exists; replacement is forbidden")
    _publish_no_replace(target, encoded, policy=config.resource_policy)
    if _file_sha256(target) != digest:
        raise RCQInputError("published registration differs from its external SHA-256 pin")
    return target, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build target-blind RCQ-v2 final preregistration"
    )
    parser.add_argument("--training-release-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path, digest = write_target_blind_registration(
        training_release_root=arguments.training_release_root,
        config_path=arguments.config,
    )
    print(
        _canonical(
            {
                "status": "created_target_blind_registration",
                "registration": str(path),
                "registration_sha256": digest,
                "external_pin_required": True,
                "sealed_test_examples_opened": 0,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_target_blind_registration",
    "write_target_blind_registration",
]
