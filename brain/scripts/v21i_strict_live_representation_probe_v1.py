"""Strict-live matched V2.1i representation/head probe revision 2 (development only).

This diagnostic is deliberately unable to publish a model.  It replays only
the exact seed-42 v3 TRAIN-FIT and DEV populations through the same sensory
signature as ``OutcomeAwareCoreV2MazePolicy.act_rgb``: RGB plus the
controller-owned prior applied action, with no actual reward or hazard.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from struct import pack
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from irene_brain.data import DatasetSplit, MazeChaseDatasetConfig
from irene_brain.evaluation.v21_qualification_metrics import (
    all_action_probability_metrics,
    binary_probability_metrics,
)
from irene_brain.training.objective import _rgb_tensor
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model
from irene_brain.v2.outcome_model import ActionOutcomeTable
from irene_brain.v2.maze_policy import OUTCOME_AWARE_MODEL_SEED_V1
from irene_brain.v2.state import PendingPrediction
from irene_brain.v2.trajectory_objective import control_action_class, transition_hazard
from run_provenance import apply_deterministic_mode
from v21i_development_runner import (
    ACTION_COUNT,
    BURN_IN_STEPS,
    DEV_EPISODES,
    DEV_OFFSET,
    PARTITION_ALGORITHM,
    SEQUENCE_LENGTH,
    TRAIN_FIT_EPISODES,
    TRAIN_FIT_OFFSET,
    PartitionContract,
    PartitionSource,
    _canonicalize_table,
    _publish_json_create_only,
    _source_bundle,
    _state_dict_sha256,
    audit_model_finite,
    model_config,
)


SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = 2
MODE = "v21i_strict_live_representation_probe_v2"
CLASSIFICATION = "diagnostic_not_candidate"
CANDIDATE_PUBLICATION_ALLOWED = False
REGISTRATION_MODE = "v21i_strict_live_representation_probe_registration_v2"

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
EXACT_DEV_MANIFEST_SHA256 = (
    "f2cd82e9a799107800744ee9350e5ebdd4bd795d454bf975280c252f7f78bd64"
)
EXACT_FAILED_V1_ARTIFACT_SHA256 = (
    "3ed153224ee1d038392e0f4e524c69582413b58051fac2dae073420efc748e57"
)
FAILED_V1_ARTIFACT = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-probe-v1.json"
)
EXACT_FAILED_V1_REGISTRATION_SHA256 = (
    "5138da4ef4fe3f9bf9734e80d8a97eabfbd1e7ed7e0a514172b453af8fa7eba9"
)
FAILED_V1_REGISTRATION = (
    "brain/runs/v21i-diagnostics/"
    "2026-08-24-seed42-v3-strict-live-probe-v1.registration.json"
)
FAILED_V1_CAUSE = "cpu_gemm_batch_shape_rounding"
FAILED_V1_DIAGNOSIS = {
    "parent_collection_batch_size": 1,
    "actual_failed_guard_batch_size": 1536,
    "independent_reproduction_batch_size": 257,
    "independent_reproduction_max_abs_delta": 4.768371582e-7,
    "independent_reproduction_mean_abs_delta": 8.426e-8,
    "independent_reproduction_mismatch_count": 953,
    "independent_matched_batch_size_one_mismatch_count": 0,
    "semantic_action_order_mismatch": False,
}

UPSTREAM_REFINEMENT_PASSES = 8
TOTAL_PROBE_PASSES = 32
O_WARM_ADDED_PASSES = TOTAL_PROBE_PASSES - UPSTREAM_REFINEMENT_PASSES
ROOT_BATCH_SIZE = 24
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
CLIP_NORM = 1.0
PROBE_SEED = 30_042
BOOTSTRAP_SEED = 31_042
BOOTSTRAP_RESAMPLES = 10_000
STRICT_LIVE_REPLAY_BATCH_SIZE = 1
PARENT_LOGIT_EQUIVALENCE_BATCH_SIZE = 1
CURRENT_HEAD_PARAMETERS = 44_161
CAPACITY_STATE_WIDTH = 366
CAPACITY_HIDDEN = 240
CAPACITY_HEAD_PARAMETERS = 262_801

MINIMUM_AGGREGATE_ROC_AUC = 0.65
MINIMUM_BASELINE_ROC_AUC_GAIN = 0.10
MINIMUM_PER_ACTION_ROC_AUC = 0.60

ARM_O_WARM = "O-warm"
ARM_O_SCRATCH = "O-scratch"
ARM_C_BELIEF = "C-belief"
ARM_C_UPDATER = "C-updater"
ARMS = (ARM_O_WARM, ARM_O_SCRATCH, ARM_C_BELIEF, ARM_C_UPDATER)

_PREREGISTRATION = (
    "brain/docs/preregistrations/"
    "2026-08-24-v21i-strict-live-representation-probe-v1.md"
)
_TEST_FILE = "brain/tests/test_v21i_strict_live_representation_probe.py"
_DIAGNOSTIC_FILES = (
    "brain/scripts/v21i_strict_live_representation_probe_v1.py",
    _PREREGISTRATION,
    _TEST_FILE,
)
_FORBIDDEN_FEATURES = (
    "actual_reward",
    "actual_hazard",
    "teacher_action",
    "event_target",
    "intervention_assignment",
    "environment_seed",
    "episode_id",
    "root_state_id",
    "observation.previous_control",
)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256(b"IRV21ISTRICTLIVEARRAY\x01")
    digest.update(array.dtype.str.encode("ascii"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_sequence_digest(values: Sequence[str]) -> str:
    """Hash exact UTF-8 sequence order and multiplicity without canonicalizing."""

    digest = sha256(b"IRV21ISTRICTLIVESEQUENCE\x01")
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        if not isinstance(value, str):
            raise TypeError("ordered sequence digest values must be strings")
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _diagnostic_source_bundle(project_root: Path) -> dict[str, object]:
    production = _source_bundle(project_root)
    if production.get("sha256") != EXACT_V3_SOURCE_BUNDLE_SHA256:
        raise RuntimeError("production source differs from the exact v3 source bundle")
    digest = sha256(b"IRV21ISTRICTLIVEPROBE\x01")
    digest.update(bytes.fromhex(EXACT_V3_SOURCE_BUNDLE_SHA256))
    files: dict[str, str] = {}
    for relative in _DIAGNOSTIC_FILES:
        file_sha = _sha256_file(project_root / relative)
        files[relative] = file_sha
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha))
    return {
        "schema_version": 1,
        "sha256": digest.hexdigest(),
        "production_v3_source_bundle": production,
        "diagnostic_files": files,
    }


def failed_v1_record(project_root: Path) -> dict[str, object]:
    """Verify and bind the immutable failed revision-1 diagnostic artifact."""

    artifact = (project_root / FAILED_V1_ARTIFACT).resolve()
    if not artifact.is_file() or _sha256_file(artifact) != EXACT_FAILED_V1_ARTIFACT_SHA256:
        raise ValueError("exact failed strict-live v1 artifact is absent or drifted")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    registration = (project_root / FAILED_V1_REGISTRATION).resolve()
    if (
        not registration.is_file()
        or _sha256_file(registration) != EXACT_FAILED_V1_REGISTRATION_SHA256
    ):
        raise ValueError("exact failed strict-live v1 registration is absent or drifted")
    failure = payload.get("failure")
    registration_binding = payload.get("registration")
    if (
        payload.get("mode") != "v21i_strict_live_representation_probe_v1"
        or payload.get("classification") != "diagnostic_run_failed_not_candidate"
        or not isinstance(failure, Mapping)
        or failure.get("type") != "RuntimeError"
        or failure.get("message")
        != "O-warm does not exactly reproduce parent TRAIN-FIT logits"
        or not isinstance(registration_binding, Mapping)
        or registration_binding.get("sha256")
        != EXACT_FAILED_V1_REGISTRATION_SHA256
    ):
        raise ValueError("failed strict-live v1 artifact envelope drifted")
    return {
        "artifact": FAILED_V1_ARTIFACT,
        "sha256": EXACT_FAILED_V1_ARTIFACT_SHA256,
        "mode": payload["mode"],
        "registration": {
            "artifact": FAILED_V1_REGISTRATION,
            "sha256": EXACT_FAILED_V1_REGISTRATION_SHA256,
        },
        "failure": dict(failure),
        "cause": FAILED_V1_CAUSE,
        "diagnosis": dict(FAILED_V1_DIAGNOSIS),
    }


def diagnostic_scope() -> dict[str, object]:
    return {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "sensory_signature": "rgb_plus_internally_owned_prior_applied_action_v1",
        "actual_reward": None,
        "actual_hazard": None,
        "pending_prediction": "semantic_gather_of_current_applied_action_after_tick",
        "parent_logit_equivalence": (
            "byte_exact_at_matched_batch_size_one_no_tolerance"
        ),
        "feature_denylist": list(_FORBIDDEN_FEATURES),
        "constructed_partitions": ["TRAIN-FIT", "DEV"],
        "fit_partition": "TRAIN-FIT",
        "dev_role": "fixed_parent_and_final_endpoint_evaluation_only",
        "train_cal_constructed": False,
        "cpu_qual_opened": False,
        "test_split_opened": False,
        "candidate_publication_allowed": False,
        "checkpoint_emitted": False,
    }


def registration_payload(project_root: Path) -> dict[str, object]:
    """Return the complete immutable pre-run registration receipt payload."""

    return {
        "schema_version": 1,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": REGISTRATION_MODE,
        "classification": "development_diagnostic_registration_not_candidate",
        "create_only": True,
        "diagnostic_mode": MODE,
        "qualification_claimed": False,
        "candidate_publication_allowed": False,
        "checkpoint_emitted": False,
        "supersedes_failed_v1": failed_v1_record(project_root),
        "upstream": {
            "result_sha256": EXACT_V3_RESULT_SHA256,
            "parent_sha256": EXACT_V3_PARENT_SHA256,
            "parent_state_sha256": EXACT_V3_PARENT_STATE_SHA256,
            "production_source_bundle_sha256": EXACT_V3_SOURCE_BUNDLE_SHA256,
            "train_fit_manifest_sha256": EXACT_TRAIN_FIT_MANIFEST_SHA256,
            "dev_manifest_sha256": EXACT_DEV_MANIFEST_SHA256,
        },
        "diagnostic_source_bundle": _diagnostic_source_bundle(project_root),
        "schedule": schedule_record(),
        "scope": diagnostic_scope(),
    }


def register(
    registration: Path,
    *,
    project_root: Path | None = None,
) -> None:
    """Create the one immutable registration receipt; never overwrite it."""

    root = (
        Path(__file__).resolve().parents[2]
        if project_root is None
        else project_root.resolve()
    )
    _publish_json_create_only(registration.resolve(), registration_payload(root))


def validate_registration(
    registration: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Reject absent, malformed, or source-drifted pre-run registration."""

    registration = registration.resolve()
    if not registration.is_file():
        raise FileNotFoundError("strict-live probe registration receipt is required")
    root = (
        Path(__file__).resolve().parents[2]
        if project_root is None
        else project_root.resolve()
    )
    payload = json.loads(registration.read_text(encoding="utf-8"))
    expected = registration_payload(root)
    if payload != expected:
        raise ValueError("strict-live probe registration or registered sources drifted")
    return {
        "path": str(registration),
        "sha256": _sha256_file(registration),
        "mode": REGISTRATION_MODE,
        "implementation_revision": payload["implementation_revision"],
        "supersedes_failed_v1": payload["supersedes_failed_v1"],
        "diagnostic_source_bundle": payload["diagnostic_source_bundle"],
        "scope": payload["scope"],
    }


def validate_upstream_result(
    payload: Mapping[str, object],
    *,
    artifact_sha256: str,
) -> dict[str, object]:
    """Fail closed unless ``payload`` binds the exact failed seed-42 v3 parent."""

    if artifact_sha256 != EXACT_V3_RESULT_SHA256:
        raise ValueError("upstream result is not the exact pinned v3 artifact")
    if (
        payload.get("schema_version") != 1
        or payload.get("mode") != "v21i_development_only"
        or payload.get("classification") != "development_candidate_not_qualified"
        or payload.get("qualification_claimed") is not False
        or payload.get("model_seed") != 42
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
        raise ValueError("the exact v3 upstream must remain a failed development run")
    namespace = payload.get("namespace_policy")
    if not isinstance(namespace, Mapping) or (
        namespace.get("cpu_qual_opened") is not False
        or namespace.get("test_split_opened") is not False
    ):
        raise ValueError("upstream namespace boundary drifted")
    partitions = payload.get("dataset_partitions")
    if not isinstance(partitions, Mapping):
        raise TypeError("upstream dataset partitions must be a mapping")
    expected = {
        "TRAIN-FIT": EXACT_TRAIN_FIT_MANIFEST_SHA256,
        "DEV": EXACT_DEV_MANIFEST_SHA256,
    }
    for name, manifest_sha in expected.items():
        record = partitions.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"upstream {name} partition is absent")
        manifest = record.get("dataset_manifest")
        if not isinstance(manifest, Mapping) or (
            record.get("dataset_manifest_sha256") != manifest_sha
            or record.get("burn_in_steps") != BURN_IN_STEPS
            or manifest.get("behavior_policy") != "balanced_intervention_v1"
            or manifest.get("behavior_intervention_rate_hex") != float(0.5).hex()
            or manifest.get("counterfactual_targets") != "all_actions_v1"
        ):
            raise ValueError(f"upstream {name} manifest drifted")
    parent = payload.get("upstream_uncalibrated_checkpoint")
    if not isinstance(parent, Mapping) or (
        parent.get("sha256") != EXACT_V3_PARENT_SHA256
        or parent.get("state_sha256") != EXACT_V3_PARENT_STATE_SHA256
        or not isinstance(parent.get("path"), str)
    ):
        raise ValueError("upstream uncalibrated parent binding drifted")
    return {
        "result_sha256": artifact_sha256,
        "parent_path": parent["path"],
        "parent_sha256": parent["sha256"],
        "parent_state_sha256": parent["state_sha256"],
        "source_bundle_sha256": source_bundle["sha256"],
        "train_fit_manifest_sha256": EXACT_TRAIN_FIT_MANIFEST_SHA256,
        "dev_manifest_sha256": EXACT_DEV_MANIFEST_SHA256,
        "partition_algorithm": PARTITION_ALGORITHM,
    }


def strict_live_partition_contracts() -> dict[str, PartitionContract]:
    """Construct exactly TRAIN-FIT and DEV; no unused namespace is instantiated."""

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
    return {
        "TRAIN-FIT": PartitionContract(
            "TRAIN-FIT",
            MazeChaseDatasetConfig(
                split=DatasetSplit.TRAIN,
                sequence_count=TRAIN_FIT_EPISODES,
                seed_offset=TRAIN_FIT_OFFSET,
                **common,
            ),
        ),
        "DEV": PartitionContract(
            "DEV",
            MazeChaseDatasetConfig(
                split=DatasetSplit.VALIDATION,
                sequence_count=DEV_EPISODES,
                seed_offset=DEV_OFFSET,
                **common,
            ),
        ),
    }


def strict_live_partition_sources(
    source_factory: Callable[[PartitionContract], object] = PartitionSource,
) -> dict[str, object]:
    contracts = strict_live_partition_contracts()
    return {name: source_factory(contract) for name, contract in contracts.items()}


def load_exact_parent(
    result_path: Path,
) -> tuple[CoreV2Model, dict[str, object], dict[str, object]]:
    result_path = result_path.resolve()
    artifact_sha = _sha256_file(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("upstream result must be a JSON object")
    provenance = validate_upstream_result(result, artifact_sha256=artifact_sha)
    parent_path = Path(str(provenance["parent_path"])).resolve()
    if _sha256_file(parent_path) != EXACT_V3_PARENT_SHA256:
        raise ValueError("exact v3 parent file hash mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping) or (
        parent.get("schema_version") != 1
        or parent.get("mode") != "v21i_development_only"
        or parent.get("checkpoint_role")
        != "upstream_joint_and_hazard_refined_uncalibrated_parent"
        or parent.get("classification") != "development_candidate_not_qualified"
        or parent.get("model_seed") != 42
        or parent.get("state_sha256") != EXACT_V3_PARENT_STATE_SHA256
        or parent.get("config") != asdict(model_config())
        or parent.get("flags") != asdict(CONFIG_B_PREDICTIVE)
    ):
        raise ValueError("exact v3 parent checkpoint drifted")
    training = parent.get("training")
    if not isinstance(training, Mapping) or (
        training.get("joint_training", {}).get("optimizer_steps") != 48
        or training.get("hazard_refinement", {}).get("passes")
        != UPSTREAM_REFINEMENT_PASSES
        or training.get("hazard_refinement", {}).get("optimizer_steps") != 512
    ):
        raise ValueError("exact v3 parent training schedule drifted")
    state_dict = parent.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or (
        _state_dict_sha256(state_dict) != EXACT_V3_PARENT_STATE_SHA256
    ):
        raise ValueError("exact v3 parent state digest mismatch")
    model = CoreV2Model(config=model_config(), flags=CONFIG_B_PREDICTIVE)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    finite = audit_model_finite(model)
    if finite["state_sha256"] != EXACT_V3_PARENT_STATE_SHA256:
        raise ValueError("loaded exact v3 parent state drifted")
    outcome = model.world_model.outcome_model if model.world_model else None
    if outcome is None or not torch.equal(
        outcome.hazard_calibration_scale,
        torch.ones_like(outcome.hazard_calibration_scale),
    ) or not torch.equal(
        outcome.hazard_calibration_bias,
        torch.zeros_like(outcome.hazard_calibration_bias),
    ):
        raise ValueError("exact v3 parent must be raw and uncalibrated")
    return model, result, provenance


@dataclass(frozen=True)
class InterventionHistory:
    """Diagnostic-only causal counts, never an input to a probe head."""

    prior_assignment_count: np.ndarray
    prior_disagreement_count: np.ndarray

    def __post_init__(self) -> None:
        if self.prior_assignment_count.ndim != 1 or (
            self.prior_disagreement_count.shape != self.prior_assignment_count.shape
        ):
            raise ValueError("intervention history arrays must be aligned vectors")
        if np.any(self.prior_assignment_count < 0) or np.any(
            self.prior_disagreement_count < 0
        ):
            raise ValueError("intervention history counts must be nonnegative")


@dataclass(frozen=True)
class StrictLiveFeatureTape:
    beliefs: np.ndarray
    updater_inputs: np.ndarray
    hazard_targets: np.ndarray
    parent_raw_logits: np.ndarray
    episode_group_ids: tuple[str, ...]
    root_state_ids: tuple[str, ...]
    history: InterventionHistory

    def __post_init__(self) -> None:
        roots = self.hazard_targets.shape[0]
        if self.beliefs.shape != (roots, model_config().width):
            raise ValueError("belief tape shape drifted")
        if self.updater_inputs.shape != (roots, CAPACITY_STATE_WIDTH):
            raise ValueError("updater-input tape shape drifted")
        if self.hazard_targets.shape != (roots, ACTION_COUNT):
            raise ValueError("hazard target table shape drifted")
        if self.parent_raw_logits.shape != (roots, ACTION_COUNT):
            raise ValueError("parent logit table shape drifted")
        if len(self.episode_group_ids) != roots or len(self.root_state_ids) != roots:
            raise ValueError("tape provenance does not cover every root")
        if self.history.prior_assignment_count.shape != (roots,):
            raise ValueError("history does not cover every root")
        for array in (
            self.beliefs,
            self.updater_inputs,
            self.hazard_targets,
            self.parent_raw_logits,
        ):
            if not np.isfinite(array).all():
                raise ValueError("strict-live tape contains non-finite values")

    def model_features(self, arm: str) -> np.ndarray:
        """Return only preregistered causal model features for ``arm``."""

        if arm in {ARM_O_WARM, ARM_O_SCRATCH}:
            return self.beliefs.copy()
        if arm == ARM_C_BELIEF:
            padding = np.zeros(
                (self.beliefs.shape[0], CAPACITY_STATE_WIDTH - self.beliefs.shape[1]),
                dtype=self.beliefs.dtype,
            )
            return np.concatenate((self.beliefs, padding), axis=1)
        if arm == ARM_C_UPDATER:
            return self.updater_inputs.copy()
        raise ValueError(f"unknown probe arm: {arm}")


@dataclass(frozen=True)
class StrictLiveStepSnapshot:
    """Hazard-relevant state from one exact strict-live model tick."""

    output: object
    state: object
    updater_input: Tensor
    canonical_raw_hazard_logits: np.ndarray


@contextmanager
def _seeded_first_tick_rng(device: torch.device):
    """Match OutcomeAwareCoreV2MazePolicy's fixed first-tick RNG scope."""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            torch.cuda.current_device() if device.index is None else device.index
        )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(OUTCOME_AWARE_MODEL_SEED_V1)
        if device.type == "cuda":
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(OUTCOME_AWARE_MODEL_SEED_V1)
        yield


@torch.no_grad()
def strict_live_reference_step(
    model: CoreV2Model,
    rgb_frames: Sequence[object],
    state: object,
    prior_action: Tensor,
    *,
    first_tick: bool,
) -> StrictLiveStepSnapshot:
    """Execute the hazard-relevant path with the exact act_rgb call signature."""

    if prior_action.dim() != 1 or len(prior_action) != len(rgb_frames):
        raise ValueError("prior action must align with RGB frames")
    device = prior_action.device
    old_belief = state.fast.belief.detach().clone()
    has_pending = torch.full(
        (len(rgb_frames), 1),
        float(state.fast.pending_prediction is not None),
        dtype=torch.float32,
        device=device,
    )
    pixels = _rgb_tensor(
        tuple(rgb_frames),
        device=device,
        resolution=(32, 32),
    )
    rng_context = _seeded_first_tick_rng(device) if first_tick else nullcontext()
    with rng_context:
        output, state = model(
            pixels,
            state,
            prev_action=prior_action,
            actual_reward=None,
            actual_hazard=None,
        )
    table = output.outcome_table
    cognitive_error = output.prediction_error.cognitive_error
    if table is None or table.raw_hazard_logits is None or cognitive_error is None:
        raise RuntimeError("strict-live replay requires table logits and cognitive error")
    if cognitive_error.dim() == 3:
        cognitive_error = cognitive_error.mean(dim=1)
    updater = torch.cat(
        (
            old_belief,
            output.latent.detach(),
            cognitive_error.detach(),
            F.one_hot(prior_action, num_classes=ACTION_COUNT).to(torch.float32),
            has_pending,
        ),
        dim=-1,
    )
    if updater.shape != (len(rgb_frames), CAPACITY_STATE_WIDTH):
        raise RuntimeError("strict-live updater feature width drifted")
    return StrictLiveStepSnapshot(
        output=output,
        state=state,
        updater_input=updater,
        canonical_raw_hazard_logits=_canonicalize_table(
            table.raw_hazard_logits,
            table.action_ids,
        ),
    )


def behavior_assignment(
    manifest_sha256: str,
    *,
    episode_seed: int,
    tick: int,
) -> bool:
    """Reconstruct the exact balanced-policy assignment for one causal tick."""

    if len(manifest_sha256) != 64:
        raise ValueError("manifest_sha256 must be a SHA-256 digest")
    if type(episode_seed) is not int or episode_seed < 0:
        raise ValueError("episode_seed must be a nonnegative integer")
    if type(tick) is not int or tick < 0:
        raise ValueError("tick must be a nonnegative integer")
    digest = sha256(
        b"IRMCBEHAVIOR\x01"
        + bytes.fromhex(manifest_sha256)
        + pack(">Q", episode_seed)
        + pack(">Q", tick)
    ).digest()
    return int.from_bytes(digest[:8], "big") < int(0.5 * (1 << 64))


def prior_only_history_counts(values: Sequence[bool]) -> np.ndarray:
    """Return prefix counts that strictly exclude each current element."""

    result = np.zeros(len(values), dtype=np.int64)
    count = 0
    for index, value in enumerate(values):
        if type(value) is not bool:
            raise TypeError("history values must be booleans")
        result[index] = count
        count += int(value)
    return result


def _install_applied_pending(
    model_state: object,
    table: ActionOutcomeTable,
    applied_action: Tensor,
) -> None:
    """Mirror act_rgb's post-selection pending-prediction semantic gather."""

    selected = table.gather(applied_action)
    model_state.fast.pending_prediction = PendingPrediction(
        predicted_next_latent=selected.predicted_next_latent.detach(),
        predicted_reward=selected.predicted_reward.detach(),
        predicted_reward_logits=(
            None
            if selected.predicted_reward_logits is None
            else selected.predicted_reward_logits.detach()
        ),
        predicted_hazard=selected.predicted_hazard.detach(),
        predicted_confidence=(
            None
            if selected.predicted_confidence is None
            else selected.predicted_confidence.detach()
        ),
        predicted_branch_logit=(
            None
            if selected.predicted_branch_logit is None
            else selected.predicted_branch_logit.detach()
        ),
    )


@torch.no_grad()
def collect_strict_live_tape(
    model: CoreV2Model,
    source: PartitionSource,
) -> StrictLiveFeatureTape:
    """Replay a partition through RGB + internally owned prior action only."""

    if source.contract.name not in {"TRAIN-FIT", "DEV"}:
        raise ValueError("strict-live probe may construct only TRAIN-FIT and DEV")
    was_training = model.training
    model.eval()
    beliefs: list[np.ndarray] = []
    updater_inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    parent_logits: list[np.ndarray] = []
    episode_groups: list[str] = []
    root_ids: list[str] = []
    assignment_histories: list[int] = []
    disagreement_histories: list[int] = []
    device = torch.device("cpu")

    # A batch size of one preserves the policy's per-episode first-tick RNG
    # contract exactly.  Batched episode initialization would consume one RNG
    # stream across multiple episodes and would therefore not be act_rgb-equivalent.
    for batch in source.iter_all_action_batches(
        epoch=0,
        batch_size=STRICT_LIVE_REPLAY_BATCH_SIZE,
    ):
        state = model.init_state(batch.batch_size, device)
        prior_action = torch.zeros(batch.batch_size, dtype=torch.long, device=device)
        assignment_count = np.zeros(batch.batch_size, dtype=np.int64)
        disagreement_count = np.zeros(batch.batch_size, dtype=np.int64)
        for tick in range(batch.sequence_length):
            transitions = tuple(sequence.transitions[tick] for sequence in batch.sequences)
            applied = torch.tensor(
                [control_action_class(transition.applied_control) for transition in transitions],
                dtype=torch.long,
                device=device,
            )
            step = strict_live_reference_step(
                model,
                tuple(transition.observation.rgb for transition in transitions),
                state,
                prior_action,
                first_tick=tick == 0,
            )
            output = step.output
            state = step.state
            table = output.outcome_table

            if tick >= batch.burn_in_steps:
                beliefs.append(output.belief.detach().to(torch.float64).cpu().numpy())
                updater_inputs.append(
                    step.updater_input.detach().to(torch.float64).cpu().numpy()
                )
                parent_logits.append(step.canonical_raw_hazard_logits)
                targets.append(
                    np.asarray(
                        [
                            [
                                transition_hazard(branch.event_targets)
                                for branch in transition.counterfactual_targets
                            ]
                            for transition in transitions
                        ],
                        dtype=np.float64,
                    )
                )
                assignment_histories.extend(int(value) for value in assignment_count)
                disagreement_histories.extend(int(value) for value in disagreement_count)
                for sequence, transition in zip(batch.sequences, transitions, strict=True):
                    episode_groups.append(
                        f"{source.contract.name}:episode:{sequence.episode_seed}"
                    )
                    root_ids.append(transition.root_state_sha256)

            # Applied action becomes private controller history only after the
            # current observation/features have been consumed.
            _install_applied_pending(state, table, applied)
            for index, (sequence, transition) in enumerate(
                zip(batch.sequences, transitions, strict=True)
            ):
                assignment_count[index] += int(
                    behavior_assignment(
                        source.manifest_sha256,
                        episode_seed=sequence.episode_seed,
                        tick=tick,
                    )
                )
                disagreement_count[index] += int(
                    control_action_class(transition.applied_control)
                    != control_action_class(transition.action_target)
                )
            prior_action = applied

    if was_training:
        model.train()
    tape = StrictLiveFeatureTape(
        beliefs=np.concatenate(beliefs, axis=0),
        updater_inputs=np.concatenate(updater_inputs, axis=0),
        hazard_targets=np.concatenate(targets, axis=0),
        parent_raw_logits=np.concatenate(parent_logits, axis=0),
        episode_group_ids=tuple(episode_groups),
        root_state_ids=tuple(root_ids),
        history=InterventionHistory(
            prior_assignment_count=np.asarray(assignment_histories, dtype=np.int64),
            prior_disagreement_count=np.asarray(disagreement_histories, dtype=np.int64),
        ),
    )
    expected_roots = source.contract.dataset_config.sequence_count * (
        SEQUENCE_LENGTH - BURN_IN_STEPS
    )
    if tape.hazard_targets.shape[0] != expected_roots:
        raise RuntimeError("strict-live replay root count drifted")
    if len(set(tape.root_state_ids)) != len(tape.root_state_ids):
        raise RuntimeError("strict-live replay contains duplicate roots")
    return tape


class CurrentHazardProbe(nn.Module):
    """Exact 44,161-parameter dedicated v3 hazard-head topology."""

    def __init__(self) -> None:
        super().__init__()
        width = model_config().width
        hidden = model_config().world_model_hidden
        self.state_trunk = nn.Sequential(nn.Linear(width, hidden), nn.ReLU())
        self.action_embedding = nn.Embedding(ACTION_COUNT, hidden)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)
        if sum(parameter.numel() for parameter in self.parameters()) != CURRENT_HEAD_PARAMETERS:
            raise RuntimeError("current hazard probe parameter count drifted")

    def load_parent(self, model: CoreV2Model) -> None:
        outcome = model.world_model.outcome_model if model.world_model else None
        if outcome is None or outcome.hazard_state_trunk is None:
            raise RuntimeError("parent lacks the dedicated hazard path")
        self.state_trunk.load_state_dict(outcome.hazard_state_trunk.state_dict())
        self.action_embedding.load_state_dict(outcome.hazard_action_embedding.state_dict())
        self.outcome_trunk.load_state_dict(outcome.hazard_outcome_trunk.state_dict())
        self.head.load_state_dict(outcome.hazard_head.state_dict())

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() != 2 or state.shape[1] != model_config().width:
            raise ValueError("current probe state shape drifted")
        ids = torch.arange(ACTION_COUNT, device=state.device).expand(state.shape[0], -1)
        encoded = F.layer_norm(self.state_trunk(state), (model_config().width,))
        encoded = encoded.unsqueeze(1).expand(-1, ACTION_COUNT, -1)
        action = self.action_embedding(ids)
        action = F.layer_norm(action, (action.shape[-1],))
        return self.head(self.outcome_trunk(torch.cat((encoded, action), dim=-1))).squeeze(-1)


def current_hazard_weight_copy_identity(
    head: CurrentHazardProbe,
    model: CoreV2Model,
) -> dict[str, object]:
    """Prove every copied parent hazard tensor is byte-identical by local name."""

    outcome = model.world_model.outcome_model if model.world_model else None
    if (
        outcome is None
        or outcome.hazard_state_trunk is None
        or outcome.hazard_action_embedding is None
        or outcome.hazard_outcome_trunk is None
    ):
        raise RuntimeError("parent lacks the dedicated hazard path")
    expected: dict[str, Tensor] = {}
    for local_name, parent_module in (
        ("state_trunk", outcome.hazard_state_trunk),
        ("action_embedding", outcome.hazard_action_embedding),
        ("outcome_trunk", outcome.hazard_outcome_trunk),
        ("head", outcome.hazard_head),
    ):
        for name, value in parent_module.state_dict().items():
            expected[f"{local_name}.{name}"] = value
    actual = head.state_dict()
    names_match = tuple(sorted(expected)) == tuple(sorted(actual))
    mismatched = tuple(
        name
        for name in sorted(set(expected) & set(actual))
        if not torch.equal(expected[name], actual[name])
    )
    expected_sha = _state_dict_sha256(expected)
    actual_sha = _state_dict_sha256(actual)
    return {
        "all_tensors_byte_equal": names_match and not mismatched,
        "tensor_names_match": names_match,
        "tensor_count": len(actual),
        "mismatched_tensor_names": list(mismatched),
        "parent_projected_state_sha256": expected_sha,
        "probe_state_sha256": actual_sha,
        "state_sha256_equal": expected_sha == actual_sha,
    }


class CapacityHazardProbe(nn.Module):
    """Fixed higher-capacity head used identically for belief/updater arms."""

    def __init__(self) -> None:
        super().__init__()
        self.state_trunk = nn.Sequential(
            nn.Linear(CAPACITY_STATE_WIDTH, CAPACITY_HIDDEN),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(ACTION_COUNT, CAPACITY_HIDDEN)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(CAPACITY_HIDDEN * 2, CAPACITY_HIDDEN),
            nn.ReLU(),
            nn.Linear(CAPACITY_HIDDEN, CAPACITY_HIDDEN),
            nn.ReLU(),
        )
        self.head = nn.Linear(CAPACITY_HIDDEN, 1)
        if sum(parameter.numel() for parameter in self.parameters()) != CAPACITY_HEAD_PARAMETERS:
            raise RuntimeError("capacity hazard probe parameter count drifted")

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() != 2 or state.shape[1] != CAPACITY_STATE_WIDTH:
            raise ValueError("capacity probe state shape drifted")
        ids = torch.arange(ACTION_COUNT, device=state.device).expand(state.shape[0], -1)
        encoded = F.layer_norm(self.state_trunk(state), (CAPACITY_HIDDEN,))
        encoded = encoded.unsqueeze(1).expand(-1, ACTION_COUNT, -1)
        action = self.action_embedding(ids)
        action = F.layer_norm(action, (CAPACITY_HIDDEN,))
        return self.head(self.outcome_trunk(torch.cat((encoded, action), dim=-1))).squeeze(-1)


def initialized_probe(factory: Callable[[], nn.Module], *, seed: int = PROBE_SEED) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return factory()


def deterministic_permutations(
    root_count: int,
    *,
    passes: int = TOTAL_PROBE_PASSES,
    seed: int = PROBE_SEED,
) -> tuple[Tensor, ...]:
    if root_count < 1 or passes < 1:
        raise ValueError("root_count and passes must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutations = tuple(torch.randperm(root_count, generator=generator) for _ in range(passes))
    if any(int(value.unique().numel()) != root_count for value in permutations):
        raise RuntimeError("probe permutation contains replacement")
    return permutations


def train_probe_head(
    head: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    permutations: Sequence[Tensor],
) -> dict[str, object]:
    """Fit one head on TRAIN-FIT only; no evaluation input is accepted."""

    if len(features) != len(targets):
        raise ValueError("feature and target root counts differ")
    state = torch.from_numpy(features).to(dtype=torch.float32)
    labels = torch.from_numpy(targets).to(dtype=torch.float32)
    parameters = tuple(head.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    initial_sha = _state_dict_sha256(head.state_dict())
    records: list[dict[str, object]] = []
    optimizer_steps = 0
    head.train()
    for pass_index, permutation in enumerate(permutations, start=1):
        if len(permutation) != len(state) or int(permutation.unique().numel()) != len(state):
            raise ValueError("training permutation is not a complete root bijection")
        loss_sum = 0.0
        maximum_gradient_norm = 0.0
        for start in range(0, len(state), ROOT_BATCH_SIZE):
            indices = permutation[start : start + ROOT_BATCH_SIZE]
            logits = head(state[indices])
            loss = F.binary_cross_entropy_with_logits(logits, labels[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(parameters, CLIP_NORM))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite probe gradient")
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach()) * len(indices)
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        records.append(
            {
                "pass": pass_index,
                "roots": len(state),
                "optimizer_steps": math.ceil(len(state) / ROOT_BATCH_SIZE),
                "mean_unweighted_all_action_bce": loss_sum / len(state),
                "maximum_preclip_gradient_norm": maximum_gradient_norm,
                "root_permutation_sha256": _array_sha256(
                    permutation.to(torch.int64).cpu().numpy()
                ),
            }
        )
    return {
        "initial_state_sha256": initial_sha,
        "final_state_sha256": _state_dict_sha256(head.state_dict()),
        "passes": len(permutations),
        "optimizer_steps": optimizer_steps,
        "pass_records": records,
        "optimizer": "AdamW_fresh_restart",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "root_batch_size": ROOT_BATCH_SIZE,
        "loss": "unweighted_binary_cross_entropy_with_logits_all_five_actions",
        "early_stopping": False,
        "model_selection": False,
    }


@torch.no_grad()
def raw_logit_table(
    head: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    head.eval()
    if batch_size is None:
        batch_size = len(features)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("raw-logit batch_size must be a positive integer")
    logits = torch.cat(
        tuple(
            head(torch.from_numpy(features[start : start + batch_size]).to(torch.float32))
            for start in range(0, len(features), batch_size)
        ),
        dim=0,
    )
    if logits.shape != (len(features), ACTION_COUNT) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("probe produced an invalid logit table")
    return logits.to(torch.float64).cpu().numpy()


@torch.no_grad()
def probability_table(head: nn.Module, features: np.ndarray) -> np.ndarray:
    logits = torch.from_numpy(raw_logit_table(head, features))
    return torch.sigmoid(logits).numpy()


def fixed_baseline_reports(
    train_tape: StrictLiveFeatureTape,
    dev_tape: StrictLiveFeatureTape,
) -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Evaluate immutable parent and TRAIN-FIT-fitted prior on both endpoints."""

    tapes = {"TRAIN-FIT": train_tape, "DEV": dev_tape}
    parent_probabilities = {
        partition: 1.0 / (1.0 + np.exp(-tape.parent_raw_logits))
        for partition, tape in tapes.items()
    }
    train_prior = train_tape.hazard_targets.mean(axis=0)
    prior_probabilities = {
        partition: np.broadcast_to(train_prior, tape.hazard_targets.shape).copy()
        for partition, tape in tapes.items()
    }
    parent_reports = {
        partition: all_action_probability_metrics(
            tape.hazard_targets,
            parent_probabilities[partition],
        ).as_dict()
        for partition, tape in tapes.items()
    }
    prior_reports = {
        partition: all_action_probability_metrics(
            tape.hazard_targets,
            prior_probabilities[partition],
        ).as_dict()
        for partition, tape in tapes.items()
    }
    return (
        {
            "parent_strict_live": parent_reports,
            "train_fitted_action_prior": {
                "fit_partition": "TRAIN-FIT",
                "per_action_probability": train_prior.tolist(),
                "partitions": prior_reports,
            },
        },
        parent_probabilities,
        prior_probabilities,
    )


def ranking_gate(
    report: Mapping[str, object],
    *,
    prior_aggregate_auc: float,
) -> dict[str, object]:
    aggregate = report["aggregate"]
    per_action = report["per_action"]
    aggregate_auc = float(aggregate["roc_auc"])
    checks = {
        "aggregate_roc_auc": aggregate_auc >= MINIMUM_AGGREGATE_ROC_AUC,
        "aggregate_prior_gain": (
            aggregate_auc - prior_aggregate_auc >= MINIMUM_BASELINE_ROC_AUC_GAIN
        ),
        "every_action_roc_auc": all(
            float(entry["metrics"]["roc_auc"]) >= MINIMUM_PER_ACTION_ROC_AUC
            for entry in per_action
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_aggregate_roc_auc": MINIMUM_AGGREGATE_ROC_AUC,
            "minimum_baseline_roc_auc_gain": MINIMUM_BASELINE_ROC_AUC_GAIN,
            "minimum_per_action_roc_auc": MINIMUM_PER_ACTION_ROC_AUC,
        },
        "prior_aggregate_roc_auc": prior_aggregate_auc,
        "aggregate_roc_auc_gain": aggregate_auc - prior_aggregate_auc,
    }


def paired_episode_auc_delta(
    targets: np.ndarray,
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray,
    episode_group_ids: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Paired two-sided 95% episode-clustered interval for aggregate AUC delta."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not (
        targets.shape == left_probabilities.shape == right_probabilities.shape
        and len(episode_group_ids) == targets.shape[0]
    ):
        raise ValueError("paired bootstrap inputs are not aligned")
    groups = tuple(dict.fromkeys(episode_group_ids))
    group_rows = {
        group: np.flatnonzero(np.asarray(episode_group_ids, dtype=object) == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([group_rows[groups[int(value)]] for value in sampled])
        left_auc = binary_probability_metrics(
            targets[rows].reshape(-1), left_probabilities[rows].reshape(-1)
        ).roc_auc
        right_auc = binary_probability_metrics(
            targets[rows].reshape(-1), right_probabilities[rows].reshape(-1)
        ).roc_auc
        deltas[index] = left_auc - right_auc
    point = (
        binary_probability_metrics(targets.reshape(-1), left_probabilities.reshape(-1)).roc_auc
        - binary_probability_metrics(targets.reshape(-1), right_probabilities.reshape(-1)).roc_auc
    )
    return {
        "point_delta": point,
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "resamples": resamples,
        "seed": seed,
        "clustering": "episode_paired_with_replacement",
    }


def intervention_stratification(
    targets: np.ndarray,
    probabilities: np.ndarray,
    counts: np.ndarray,
) -> dict[str, object]:
    """Report exact prior-count strata; these values never affect a gate."""

    if len(counts) != len(targets) or probabilities.shape != targets.shape:
        raise ValueError("stratification inputs are not aligned")
    strata: list[dict[str, object]] = []
    for count in sorted(int(value) for value in np.unique(counts)):
        rows = counts == count
        flat_targets = targets[rows].reshape(-1)
        flat_probabilities = probabilities[rows].reshape(-1)
        positives = int(flat_targets.sum())
        record: dict[str, object] = {
            "prior_count": count,
            "roots": int(rows.sum()),
            "branch_observations": len(flat_targets),
            "positives": positives,
            "negatives": len(flat_targets) - positives,
            "used_as_feature": False,
            "used_as_gate": False,
        }
        if positives and positives != len(flat_targets):
            record["aggregate_metrics"] = binary_probability_metrics(
                flat_targets, flat_probabilities
            ).as_dict()
        else:
            record["aggregate_metrics"] = None
            record["limitation"] = "stratum lacks both hazard classes"
        strata.append(record)
    return {
        "causal_scope": "strictly prior ticks; current tick excluded",
        "used_as_feature": False,
        "used_as_gate": False,
        "strata": strata,
    }


def evidence_manifest(
    partition: str,
    tape: StrictLiveFeatureTape,
    *,
    dataset_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "partition": partition,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "roots": len(tape.beliefs),
        "actions_per_root": ACTION_COUNT,
        "belief_sha256": _array_sha256(tape.beliefs),
        "updater_input_sha256": _array_sha256(tape.updater_inputs),
        "target_sha256": _array_sha256(tape.hazard_targets),
        "parent_raw_logit_sha256": _array_sha256(tape.parent_raw_logits),
        "prior_assignment_count_sha256": _array_sha256(
            tape.history.prior_assignment_count
        ),
        "prior_disagreement_count_sha256": _array_sha256(
            tape.history.prior_disagreement_count
        ),
        "ordered_root_sha256": _ordered_sequence_digest(tape.root_state_ids),
        "ordered_episode_group_sha256": _ordered_sequence_digest(
            tape.episode_group_ids
        ),
        "feature_allowlist": [
            "post_update_belief",
            "old_belief",
            "current_rgb_encoder_latent",
            "live_only_cognitive_prediction_error",
            "internally_owned_prior_applied_action_onehot",
            "has_pending_prediction",
            "semantic_candidate_action_id",
        ],
        "feature_denylist": list(_FORBIDDEN_FEATURES),
        "actual_reward_input": None,
        "actual_hazard_input": None,
    }


def schedule_record() -> dict[str, object]:
    roots = TRAIN_FIT_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS)
    steps_per_pass = math.ceil(roots / ROOT_BATCH_SIZE)
    return {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "train_partition": "TRAIN-FIT",
        "evaluation_partitions": ["TRAIN-FIT", "DEV"],
        "train_cal_constructed": False,
        "dev_evaluated_only_at_fixed_parent_and_final_endpoints": True,
        "dev_used_for_stopping": False,
        "dev_used_for_model_selection": False,
        "calibration_fit_or_install": False,
        "strict_live_replay_batch_size": STRICT_LIVE_REPLAY_BATCH_SIZE,
        "model_initialization_seed": OUTCOME_AWARE_MODEL_SEED_V1,
        "first_tick_rng_restarted_per_episode": True,
        "parent_logit_equivalence_batch_size": PARENT_LOGIT_EQUIVALENCE_BATCH_SIZE,
        "parent_logit_equivalence": "byte_exact_no_tolerance",
        "v1_guard_correction_cause": FAILED_V1_CAUSE,
        "root_batch_size": ROOT_BATCH_SIZE,
        "roots_per_pass": roots,
        "steps_per_pass": steps_per_pass,
        "sampling": "deterministic_complete_root_permutation_without_replacement",
        "permutation_seed": PROBE_SEED,
        "arms": {
            ARM_O_WARM: {
                "upstream_passes": UPSTREAM_REFINEMENT_PASSES,
                "added_passes": O_WARM_ADDED_PASSES,
                "nominal_total_passes_across_signatures": TOTAL_PROBE_PASSES,
                "strict_live_tape_passes": O_WARM_ADDED_PASSES,
                "optimizer_steps": O_WARM_ADDED_PASSES * steps_per_pass,
                "optimizer_restart": "fresh_AdamW_no_parent_optimizer_state",
                "warm_start_source_signature": (
                    "v3_balanced_replay_with_prior_actual_reward_and_hazard"
                ),
                "adaptation_signature": (
                    "strict_live_rgb_plus_internal_prior_action_without_actual_outcomes"
                ),
                "cross_signature_caveat": (
                    "the upstream eight passes did not use the strict-live belief tape; "
                    "32 is a nominal warm-start count, not 32 matched-distribution passes"
                ),
            },
            ARM_O_SCRATCH: {
                "from_scratch": True,
                "passes": TOTAL_PROBE_PASSES,
                "optimizer_steps": TOTAL_PROBE_PASSES * steps_per_pass,
            },
            ARM_C_BELIEF: {
                "from_scratch": True,
                "passes": TOTAL_PROBE_PASSES,
                "optimizer_steps": TOTAL_PROBE_PASSES * steps_per_pass,
            },
            ARM_C_UPDATER: {
                "from_scratch": True,
                "passes": TOTAL_PROBE_PASSES,
                "optimizer_steps": TOTAL_PROBE_PASSES * steps_per_pass,
            },
        },
        "optimizer": "AdamW_fresh_restart",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "initialization_and_permutation_seed": PROBE_SEED,
        "early_stopping": False,
        "model_selection": False,
    }


def interpretation_report(
    gates: Mapping[str, Mapping[str, object]],
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    warm_pass = gates[ARM_O_WARM]["passed"] is True
    scratch_pass = gates[ARM_O_SCRATCH]["passed"] is True
    belief_pass = gates[ARM_C_BELIEF]["passed"] is True
    updater_pass = gates[ARM_C_UPDATER]["passed"] is True
    optimization = (
        warm_pass and comparisons["O-warm_minus_parent"]["lower_95"] > 0.0
    ) or (
        scratch_pass and comparisons["O-scratch_minus_parent"]["lower_95"] > 0.0
    )
    capacity = (
        not warm_pass
        and not scratch_pass
        and belief_pass
        and comparisons["C-belief_minus_O-warm"]["lower_95"] > 0.0
        and comparisons["C-belief_minus_O-scratch"]["lower_95"] > 0.0
    )
    current_tick_belief_update_compression = (
        not belief_pass
        and updater_pass
        and comparisons["C-updater_minus_C-belief"]["lower_95"] > 0.0
    )
    if optimization:
        diagnosis = "current_head_optimization_or_initialization_underfit_supported"
    elif capacity:
        diagnosis = "current_head_capacity_limit_supported"
    elif current_tick_belief_update_compression:
        diagnosis = "current_tick_belief_update_compression_or_accessibility_supported"
    else:
        diagnosis = "inconclusive_across_upstream_sensory_or_longer_history_compression"
    return {
        "diagnosis": diagnosis,
        "optimization_underfit_supported": optimization,
        "head_capacity_limit_supported": capacity,
        "current_tick_belief_update_compression_supported": (
            current_tick_belief_update_compression
        ),
        "longer_history_information_loss_ruled_out": False,
        "c_updater_scope": (
            "C-updater includes old_belief and can test current-tick update "
            "compression/accessibility only; it cannot recover history already "
            "discarded before old_belief"
        ),
        "c_updater_failure_interpretation": (
            "inconclusive across upstream sensory compression and longer-history "
            "information loss"
        ),
        "threshold_source": "unchanged_v21i_frozen_ranking_gates_plus_paired_CI_lower_bound_gt_zero",
        "candidate_claim": False,
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
    device = torch.device("cpu")
    if device.type != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("strict-live probe is CPU-only with CUDA hidden")

    project_root = Path(__file__).resolve().parents[2]
    source_bundle = registration_record["diagnostic_source_bundle"]
    if source_bundle != _diagnostic_source_bundle(project_root):
        raise RuntimeError("registered diagnostic source bundle drifted before execution")
    model, _, provenance = load_exact_parent(upstream_result)
    model.to(device)
    parent_state_before = _state_dict_sha256(model.state_dict())

    sources = strict_live_partition_sources()
    if sources["TRAIN-FIT"].manifest_sha256 != EXACT_TRAIN_FIT_MANIFEST_SHA256:
        raise RuntimeError("TRAIN-FIT manifest differs from v3")
    if sources["DEV"].manifest_sha256 != EXACT_DEV_MANIFEST_SHA256:
        raise RuntimeError("DEV manifest differs from v3")
    train_tape = collect_strict_live_tape(model, sources["TRAIN-FIT"])
    dev_tape = collect_strict_live_tape(model, sources["DEV"])
    if _state_dict_sha256(model.state_dict()) != parent_state_before:
        raise RuntimeError("strict-live feature collection mutated the parent")

    o_warm = CurrentHazardProbe()
    o_warm.load_parent(model)
    weight_copy_identity = current_hazard_weight_copy_identity(o_warm, model)
    if weight_copy_identity["all_tensors_byte_equal"] is not True:
        raise RuntimeError("O-warm did not byte-exactly copy every parent hazard tensor")
    o_scratch = initialized_probe(CurrentHazardProbe)
    c_belief = initialized_probe(CapacityHazardProbe)
    c_updater = initialized_probe(CapacityHazardProbe)
    heads = {
        ARM_O_WARM: o_warm,
        ARM_O_SCRATCH: o_scratch,
        ARM_C_BELIEF: c_belief,
        ARM_C_UPDATER: c_updater,
    }
    if _state_dict_sha256(c_belief.state_dict()) != _state_dict_sha256(
        c_updater.state_dict()
    ):
        raise RuntimeError("matched capacity arms did not share initialization")
    warm_start_equivalence: dict[str, object] = {}
    for name, tape in (("TRAIN-FIT", train_tape), ("DEV", dev_tape)):
        reproduced = raw_logit_table(
            o_warm,
            tape.model_features(ARM_O_WARM),
            batch_size=PARENT_LOGIT_EQUIVALENCE_BATCH_SIZE,
        )
        if not np.array_equal(reproduced, tape.parent_raw_logits):
            raise RuntimeError(f"O-warm does not exactly reproduce parent {name} logits")
        warm_start_equivalence[name] = {
            "byte_equal": True,
            "comparison_batch_size": PARENT_LOGIT_EQUIVALENCE_BATCH_SIZE,
            "tolerance": None,
            "comparison_contract": "byte_exact_matched_parent_collection_batch_shape",
            "raw_logit_sha256": _array_sha256(reproduced),
        }
    baseline_metrics, parent_probabilities, _prior_probabilities = (
        fixed_baseline_reports(train_tape, dev_tape)
    )

    permutations = deterministic_permutations(len(train_tape.beliefs))
    training_records: dict[str, object] = {}
    arm_probabilities: dict[str, dict[str, np.ndarray]] = {}
    arm_metrics: dict[str, object] = {}
    for arm in ARMS:
        used_permutations = (
            permutations[:O_WARM_ADDED_PASSES] if arm == ARM_O_WARM else permutations
        )
        training_records[arm] = train_probe_head(
            heads[arm],
            train_tape.model_features(arm),
            train_tape.hazard_targets,
            used_permutations,
        )
        train_probabilities = probability_table(
            heads[arm], train_tape.model_features(arm)
        )
        dev_probabilities = probability_table(heads[arm], dev_tape.model_features(arm))
        arm_probabilities[arm] = {
            "TRAIN-FIT": train_probabilities,
            "DEV": dev_probabilities,
        }
        arm_metrics[arm] = {
            "TRAIN-FIT": all_action_probability_metrics(
                train_tape.hazard_targets, train_probabilities
            ).as_dict(),
            "DEV": all_action_probability_metrics(
                dev_tape.hazard_targets, dev_probabilities
            ).as_dict(),
            "probability_sha256": {
                "TRAIN-FIT": _array_sha256(train_probabilities),
                "DEV": _array_sha256(dev_probabilities),
            },
        }

    prior_auc = float(
        baseline_metrics["train_fitted_action_prior"]["partitions"]["DEV"][
            "aggregate"
        ]["roc_auc"]
    )
    gates = {
        arm: ranking_gate(arm_metrics[arm]["DEV"], prior_aggregate_auc=prior_auc)
        for arm in ARMS
    }
    comparisons = {
        "O-warm_minus_parent": paired_episode_auc_delta(
            dev_tape.hazard_targets,
            arm_probabilities[ARM_O_WARM]["DEV"],
            parent_probabilities["DEV"],
            dev_tape.episode_group_ids,
        ),
        "O-scratch_minus_parent": paired_episode_auc_delta(
            dev_tape.hazard_targets,
            arm_probabilities[ARM_O_SCRATCH]["DEV"],
            parent_probabilities["DEV"],
            dev_tape.episode_group_ids,
        ),
        "C-belief_minus_O-warm": paired_episode_auc_delta(
            dev_tape.hazard_targets,
            arm_probabilities[ARM_C_BELIEF]["DEV"],
            arm_probabilities[ARM_O_WARM]["DEV"],
            dev_tape.episode_group_ids,
        ),
        "C-belief_minus_O-scratch": paired_episode_auc_delta(
            dev_tape.hazard_targets,
            arm_probabilities[ARM_C_BELIEF]["DEV"],
            arm_probabilities[ARM_O_SCRATCH]["DEV"],
            dev_tape.episode_group_ids,
        ),
        "C-updater_minus_C-belief": paired_episode_auc_delta(
            dev_tape.hazard_targets,
            arm_probabilities[ARM_C_UPDATER]["DEV"],
            arm_probabilities[ARM_C_BELIEF]["DEV"],
            dev_tape.episode_group_ids,
        ),
    }
    strata = {
        arm: {
            "prior_randomized_assignment_count": intervention_stratification(
                dev_tape.hazard_targets,
                arm_probabilities[arm]["DEV"],
                dev_tape.history.prior_assignment_count,
            ),
            "prior_actual_teacher_disagreement_count": intervention_stratification(
                dev_tape.hazard_targets,
                arm_probabilities[arm]["DEV"],
                dev_tape.history.prior_disagreement_count,
            ),
        }
        for arm in ARMS
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "qualification_claimed": False,
        "candidate_publication_allowed": CANDIDATE_PUBLICATION_ALLOWED,
        "candidate_checkpoint": {
            "published": False,
            "publication_allowed": False,
            "checkpoint_emitted": False,
        },
        "device": "cpu",
        "threads": 1,
        "cuda_visible_devices": "-1",
        "upstream": provenance,
        "registration": {
            "path": registration_record["path"],
            "sha256": registration_record["sha256"],
            "mode": registration_record["mode"],
            "implementation_revision": registration_record[
                "implementation_revision"
            ],
            "verified_before_run": True,
        },
        "supersedes_failed_v1": registration_record["supersedes_failed_v1"],
        "revision_correction": {
            "cause": FAILED_V1_CAUSE,
            "parent_logit_equivalence_batch_size": (
                PARENT_LOGIT_EQUIVALENCE_BATCH_SIZE
            ),
            "loose_numeric_tolerance_used": False,
        },
        "source_bundle": source_bundle,
        "namespace_policy": {
            "constructed_partitions": ["TRAIN-FIT", "DEV"],
            "train_cal_constructed": False,
            "cpu_qual_opened": False,
            "test_split_opened": False,
            "dev_used_for_training_or_stopping_or_selection": False,
        },
        "schedule": schedule_record(),
        "feature_contract": {
            "sensory_signature": "rgb_plus_internally_owned_prior_applied_action_v1",
            "actual_reward": None,
            "actual_hazard": None,
            "feature_denylist": list(_FORBIDDEN_FEATURES),
            "intervention_history_is_diagnostic_only": True,
            "c_updater_interpretation_scope": (
                "current-tick belief-update compression/accessibility only; "
                "old_belief cannot restore previously discarded history"
            ),
        },
        "evidence": {
            "TRAIN-FIT": evidence_manifest(
                "TRAIN-FIT",
                train_tape,
                dataset_manifest_sha256=EXACT_TRAIN_FIT_MANIFEST_SHA256,
            ),
            "DEV": evidence_manifest(
                "DEV",
                dev_tape,
                dataset_manifest_sha256=EXACT_DEV_MANIFEST_SHA256,
            ),
        },
        "parent_state": {
            "before_sha256": parent_state_before,
            "after_sha256": _state_dict_sha256(model.state_dict()),
            "byte_identical": _state_dict_sha256(model.state_dict()) == parent_state_before,
        },
        "o_warm_parent_logit_equivalence": warm_start_equivalence,
        "o_warm_parent_weight_copy_identity": weight_copy_identity,
        "training": training_records,
        "metrics": {
            **baseline_metrics,
            "arms": arm_metrics,
        },
        "ranking_gates": gates,
        "paired_episode_auc_deltas": comparisons,
        "intervention_history_stratification": strata,
        "interpretation": interpretation_report(gates, comparisons),
    }
    _publish_json_create_only(output.resolve(), result)


def run(*, upstream_result: Path, output: Path, registration: Path) -> None:
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
            receipt_sha = (
                _sha256_file(registration.resolve())
                if registration.resolve().is_file()
                else None
            )
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
                        "sha256": receipt_sha,
                        "verified_before_run": registration_record is not None,
                    },
                    "supersedes_failed_v1": (
                        registration_record["supersedes_failed_v1"]
                        if registration_record is not None
                        else {
                            "artifact": FAILED_V1_ARTIFACT,
                            "sha256": EXACT_FAILED_V1_ARTIFACT_SHA256,
                            "registration": {
                                "artifact": FAILED_V1_REGISTRATION,
                                "sha256": EXACT_FAILED_V1_REGISTRATION_SHA256,
                            },
                            "cause": FAILED_V1_CAUSE,
                            "diagnosis": dict(FAILED_V1_DIAGNOSIS),
                        }
                    ),
                    "revision_correction": {
                        "cause": FAILED_V1_CAUSE,
                        "parent_logit_equivalence_batch_size": (
                            PARENT_LOGIT_EQUIVALENCE_BATCH_SIZE
                        ),
                        "loose_numeric_tolerance_used": False,
                    },
                    "strict_sensory_and_namespace_scope": diagnostic_scope(),
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
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
