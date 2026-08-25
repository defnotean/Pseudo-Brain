"""Production-only V2.1i closed-loop PLAY-QUAL runner.

This entry point has two deliberately separate lifecycle operations:

``preregister``
    Validate and bind one exact V2.1i checkpoint, freeze the fixed unseen
    PLAY-QUAL seed bank, and publish a canonical create-only receipt.  It does
    not instantiate an environment or advance a model.

``qualify``
    Revalidate that receipt and every bound file, require a passing real
    wall-clock receipt for the same checkpoint/source/workload, construct the
    exact outcome-aware policy and MazeChase evaluator internally, and publish
    one create-only pass/failure artifact.

There is intentionally no controller, policy, environment, episode-runner, or
configuration injection surface in the production operation.
"""
from __future__ import annotations

import os

# These must be set before Torch is imported.  PLAY-QUAL must not initialize a
# GPU or compete with a foreground game through an unconstrained CPU pool.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import inspect
import json
from pathlib import Path
from typing import Mapping, Sequence

from torch import Tensor

from irene_brain.evaluation.v21_live_play_qualification import (
    LivePlayProvenance,
    LivePlayQualificationConfig,
    exact_maze_environment_config,
    exact_maze_environment_config_sha256,
    live_play_cohort_binding_sha256,
    qualify_live_play_create_only,
)
from irene_brain.evaluation.v21_wallclock_latency import (
    POST_DGX_ENVELOPE,
    LoadedV21ICheckpoint,
    PostDgxReleaseReceipt,
    load_exact_v21i_checkpoint_cpu,
    load_post_dgx_release_receipt,
    validate_post_dgx_release_binding,
    validate_wallclock_artifact,
    wallclock_evaluator_bundle_sha256,
    wallclock_workload_sha256,
)
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.maze_policy import (
    OUTCOME_AWARE_MODEL_SEED_V1,
    OutcomeAwareCoreV2MazePolicy,
)


SCHEMA_VERSION = 1
QUALIFICATION_ID = "v21i-play-qual-world-model-v1"
PREREGISTRATION_STATUS = "sealed_unopened"
PLAY_QUAL_EPISODES = 64
PLAY_QUAL_MAX_TICKS = 600
PLAY_QUAL_RANDOM_STREAMS = 8
PLAY_QUAL_DETERMINISTIC_REPEATS = 2
PLAY_QUAL_BOOTSTRAP_RESAMPLES = 10_000
PLAY_QUAL_CONFIDENCE_LEVEL = 0.95
PLAY_QUAL_ENVIRONMENT_SEED_TOP_BITS = 0b11
PLAY_QUAL_RANDOM_BASELINE_SEED = 2_126_202_681
PLAY_QUAL_BOOTSTRAP_SEED = 2_126_202_699
PLAY_QUAL_SEED_DERIVATION = "uint64-prefix-or-contiguous-index-v1"
PLAY_QUAL_RANDOM_STREAM_DERIVATION = "live-policy-xor-stream-v1"
_PREREGISTRATION_BODY_DOMAIN = b"IRV21IPLAYQUALPREREG\x01"
_MODEL_CONFIG_DOMAIN = b"IRV21IMODELCONFIG\x01"
_FEATURE_FLAGS_DOMAIN = b"IRV21IFEATUREFLAGS\x01"
_HAZARD_CALIBRATION_DOMAIN = b"IRV21IHAZARDCALIBRATION\x01"
_EVALUATOR_BUNDLE_DOMAIN = b"IRV21IPLAYQUALEVALUATOR\x01"
_TRAINING_SOURCE_BUNDLE_DOMAIN = b"IRV21IDEVSOURCE\x01"
_STATE_DICT_DOMAIN = b"IRV21ISTATE\x01"

_TRAINING_SOURCE_PIPELINE_FILES = (
    "brain/scripts/run_provenance.py",
    "brain/scripts/v21i_development_runner.py",
)
_PRODUCTION_EVALUATOR_FILES = (
    "brain/scripts/v21i_live_qualification_runner.py",
    "brain/src/irene_brain/environments/maze_chase.py",
    "brain/src/irene_brain/evaluation/closed_loop_play.py",
    "brain/src/irene_brain/evaluation/v21_live_play_qualification.py",
    "brain/src/irene_brain/evaluation/v21_wallclock_latency.py",
    "brain/src/irene_brain/training/objective.py",
    "brain/src/irene_brain/types.py",
    "brain/src/irene_brain/v2/config.py",
    "brain/src/irene_brain/v2/core.py",
    "brain/src/irene_brain/v2/maze_policy.py",
    "brain/src/irene_brain/v2/outcome_model.py",
    "brain/src/irene_brain/v2/state.py",
)
class PlayQualificationError(RuntimeError):
    """A production PLAY-QUAL invariant or artifact is invalid."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PlayQualificationError("value is not canonical JSON") from error


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = sha256(domain)
    digest.update(_canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise PlayQualificationError(f"required file is missing: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PlayQualificationError(f"{name} must be a lowercase SHA-256")
    return value


def _require_plain_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = (1 << 64) - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlayQualificationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise PlayQualificationError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _strict_json_file(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    if not path.is_file():
        raise PlayQualificationError(f"{name} is missing")
    encoded = path.read_bytes()

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PlayQualificationError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        decoded = encoded.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PlayQualificationError(f"{name} contains non-finite {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlayQualificationError(f"{name} must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise PlayQualificationError(f"{name} root must be an object")
    if encoded != (_canonical_json(payload) + "\n").encode("utf-8"):
        raise PlayQualificationError(f"{name} must be one canonical JSON line")
    return payload, encoded


def _publish_json_create_only(path: Path, payload: Mapping[str, object]) -> str:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    resolved = path.resolve()
    if not resolved.parent.is_dir():
        raise PlayQualificationError("artifact parent directory must already exist")
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, resolved)
    except FileExistsError as error:
        raise PlayQualificationError(
            f"refusing to overwrite existing artifact: {resolved}"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256(encoded).hexdigest()


def _bundle(
    project_root: Path,
    relative_paths: Sequence[str],
    *,
    domain: bytes,
) -> dict[str, object]:
    resolved_root = project_root.resolve(strict=True)
    normalized = tuple(sorted(set(relative_paths)))
    if not normalized or len(normalized) != len(tuple(relative_paths)):
        raise PlayQualificationError("source bundle paths must be nonempty and unique")
    digest = sha256(domain)
    files: dict[str, str] = {}
    for relative in normalized:
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise PlayQualificationError("source bundle paths must be POSIX relative paths")
        candidate = (resolved_root / Path(relative)).resolve(strict=True)
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise PlayQualificationError("source bundle path escapes project root") from error
        file_sha = _file_sha256(candidate)
        files[relative] = file_sha
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(bytes.fromhex(file_sha))
    return {"schema_version": 1, "sha256": digest.hexdigest(), "files": files}


def training_source_bundle(project_root: Path) -> dict[str, object]:
    package_root = project_root / "brain" / "src" / "irene_brain"
    package_files = tuple(
        path.relative_to(project_root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file()
    )
    return _bundle(
        project_root,
        (*_TRAINING_SOURCE_PIPELINE_FILES, *package_files),
        domain=_TRAINING_SOURCE_BUNDLE_DOMAIN,
    )


def production_evaluator_bundle(project_root: Path) -> dict[str, object]:
    return _bundle(
        project_root,
        _PRODUCTION_EVALUATOR_FILES,
        domain=_EVALUATOR_BUNDLE_DOMAIN,
    )


def environment_source_sha256(project_root: Path) -> str:
    return _file_sha256(
        project_root / "brain" / "src" / "irene_brain" / "environments" / "maze_chase.py"
    )


def _expected_model_config() -> CoreV2Config:
    return CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
        hazard_parameterization="probability_sigmoid_v1",
        latent_comparison="cosine_distance_v1",
        reward_comparison="raw_mse_v0",
        outcome_action_conditioning="thought_only_v0",
        outcome_architecture="all_action_table_v1",
        hazard_outcome_path="dedicated_stopgrad_v1",
        reward_prediction="symlog_twohot_v1",
        prediction_error_fusion="latent_outcome_surprise_v1",
        next_weight=0.5,
        reward_weight=0.25,
        hazard_weight=1.0,
    )


def _state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    digest = sha256(_STATE_DICT_DOMAIN)
    for name, value in sorted(state_dict.items()):
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise PlayQualificationError("state_dict must map names to tensors")
        tensor = value.detach().to(device="cpu").contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(2, "big"))
        digest.update(encoded_dtype)
        digest.update(len(tensor.shape).to_bytes(2, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: CoreV2Model
    checkpoint_sha256: str
    state_sha256: str
    model_config_sha256: str
    feature_flags_sha256: str
    hazard_calibration_sha256: str
    source_bundle_sha256: str
    model_seed: int
    post_dgx_release: PostDgxReleaseReceipt
    training_runtime: Mapping[str, object]


def load_exact_v21i_checkpoint(
    checkpoint_path: Path,
    *,
    project_root: Path,
    expected_checkpoint_sha256: str,
    expected_source_bundle_sha256: str,
    expected_model_seed: int,
    post_dgx_release: PostDgxReleaseReceipt,
) -> LoadedCheckpoint:
    """Load only a bounded-DGX checkpoint with its external 5/5 release."""

    try:
        loaded: LoadedV21ICheckpoint = load_exact_v21i_checkpoint_cpu(
            checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_source_bundle_sha256=expected_source_bundle_sha256,
            expected_model_seed=expected_model_seed,
            project_root=project_root,
            required_envelope=POST_DGX_ENVELOPE,
            post_dgx_release=post_dgx_release,
        )
    except Exception as error:
        raise PlayQualificationError(
            "checkpoint is not an exact post-DGX 5/5-qualified V2.1i model"
        ) from error
    if loaded.post_dgx_release is None or loaded.training_runtime is None:
        raise PlayQualificationError("post-DGX checkpoint lineage is incomplete")
    return LoadedCheckpoint(
        model=loaded.model,
        checkpoint_sha256=loaded.checkpoint_sha256,
        state_sha256=loaded.state_sha256,
        model_config_sha256=loaded.model_config_sha256,
        feature_flags_sha256=loaded.feature_flags_sha256,
        hazard_calibration_sha256=loaded.hazard_calibration_sha256,
        source_bundle_sha256=loaded.source_bundle_sha256,
        model_seed=loaded.model_seed,
        post_dgx_release=loaded.post_dgx_release,
        training_runtime=loaded.training_runtime,
    )


def fixed_play_qual_environment_seeds() -> tuple[int, ...]:
    prefix = PLAY_QUAL_ENVIRONMENT_SEED_TOP_BITS << 62
    return tuple(prefix | index for index in range(PLAY_QUAL_EPISODES))


def fixed_random_stream_seed_bank() -> tuple[tuple[int, ...], ...]:
    uint64_mask = (1 << 64) - 1
    result: list[tuple[int, ...]] = []
    for environment_seed in fixed_play_qual_environment_seeds():
        streams: list[int] = []
        for stream in range(PLAY_QUAL_RANDOM_STREAMS):
            campaign_seed = (
                PLAY_QUAL_RANDOM_BASELINE_SEED
                ^ ((stream + 1) * 0x9E3779B97F4A7C15)
            )
            streams.append(
                (
                    campaign_seed
                    ^ environment_seed
                    ^ 0x4952454E455F524E
                )
                & uint64_mask
            )
        result.append(tuple(streams))
    frozen = tuple(result)
    if len({seed for row in frozen for seed in row}) != (
        PLAY_QUAL_EPISODES * PLAY_QUAL_RANDOM_STREAMS
    ):
        raise PlayQualificationError("fixed PLAY-QUAL random stream seeds collided")
    return frozen


def _cohort_record() -> dict[str, object]:
    environment_seeds = fixed_play_qual_environment_seeds()
    random_seed_bank = fixed_random_stream_seed_bank()
    return {
        "cluster_ids": [f"play-qual-environment-{index:02d}" for index in range(64)],
        "environment_seed_count": PLAY_QUAL_EPISODES,
        "environment_seed_derivation": PLAY_QUAL_SEED_DERIVATION,
        "environment_seed_top_bits": "11",
        "environment_seed_encoding": "hex-u64be-lower-without-prefix",
        "environment_seeds": [f"{value:016x}" for value in environment_seeds],
        "paired_environment_random_stream_count": (
            PLAY_QUAL_EPISODES * PLAY_QUAL_RANDOM_STREAMS
        ),
        "random_stream_derivation": PLAY_QUAL_RANDOM_STREAM_DERIVATION,
        "random_stream_ids": list(range(PLAY_QUAL_RANDOM_STREAMS)),
        "random_stream_seed_bank": [
            [f"{value:016x}" for value in row] for row in random_seed_bank
        ],
        "random_streams_per_environment_seed": PLAY_QUAL_RANDOM_STREAMS,
    }


def _workload_record() -> dict[str, object]:
    return {
        "action_semantics": ["idle", "W", "A", "S", "D"],
        "bootstrap_resamples": PLAY_QUAL_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": PLAY_QUAL_BOOTSTRAP_SEED,
        "confidence_level": PLAY_QUAL_CONFIDENCE_LEVEL,
        "deterministic_repeats": PLAY_QUAL_DETERMINISTIC_REPEATS,
        "environment": exact_maze_environment_config(PLAY_QUAL_MAX_TICKS),
        "max_ticks": PLAY_QUAL_MAX_TICKS,
        "minimum_catch_advantage": 0.0,
        "minimum_reward_advantage": 0.0,
        "model_reset_seed": OUTCOME_AWARE_MODEL_SEED_V1,
        "policy_identity": OutcomeAwareCoreV2MazePolicy.identity,
        "random_baseline_seed": PLAY_QUAL_RANDOM_BASELINE_SEED,
        "random_streams_per_environment_seed": PLAY_QUAL_RANDOM_STREAMS,
        "wallclock_workload_sha256": wallclock_workload_sha256(),
    }


def _post_dgx_release_record(checkpoint: LoadedCheckpoint) -> dict[str, object]:
    release = checkpoint.post_dgx_release
    return {
        "artifact": dict(release.payload),
        "body_sha256": release.body_sha256,
        "file_sha256": release.file_sha256,
        "training_runtime": dict(checkpoint.training_runtime),
    }


def build_play_qual_preregistration(
    *,
    checkpoint: LoadedCheckpoint,
    project_root: Path,
) -> dict[str, object]:
    evaluator = production_evaluator_bundle(project_root)
    body: dict[str, object] = {
        "checkpoint": {
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "envelope": POST_DGX_ENVELOPE,
            "feature_flags_sha256": checkpoint.feature_flags_sha256,
            "hazard_calibration_sha256": checkpoint.hazard_calibration_sha256,
            "model_config_sha256": checkpoint.model_config_sha256,
            "model_seed": checkpoint.model_seed,
            "state_sha256": checkpoint.state_sha256,
        },
        "seed_bank": _cohort_record(),
        "environment_config_sha256": exact_maze_environment_config_sha256(
            PLAY_QUAL_MAX_TICKS
        ),
        "environment_source_sha256": environment_source_sha256(project_root),
        "evaluator_bundle": evaluator,
        "full_source_bundle_sha256": checkpoint.source_bundle_sha256,
        "lifecycle": [
            "publish_preregistration_create_only",
            "measure_wallclock_create_only",
            "qualify_exact_cohort_once",
            "publish_result_create_only",
        ],
        "namespace_policy": {
            "cpu_qual_or_test_materialized_by_this_runner": False,
            "environment_seed_namespace": (
                "uint64_top_bits_11_unassigned_not_train_validation_test"
            ),
            "training_or_calibration_allowed": False,
        },
        "post_dgx_release": _post_dgx_release_record(checkpoint),
        "qualification_id": QUALIFICATION_ID,
        "schema_version": SCHEMA_VERSION,
        "status": PREREGISTRATION_STATUS,
        "workload": _workload_record(),
    }
    body["preregistration_sha256"] = _json_sha256(
        body, domain=_PREREGISTRATION_BODY_DOMAIN
    )
    return body


@dataclass(frozen=True, slots=True)
class SealedPlayQualification:
    path: Path
    file_sha256: str
    preregistration_sha256: str
    payload: Mapping[str, object]


def validate_play_qual_preregistration(
    payload: Mapping[str, object],
    *,
    project_root: Path,
) -> None:
    expected_keys = {
        "checkpoint",
        "environment_config_sha256",
        "environment_source_sha256",
        "evaluator_bundle",
        "full_source_bundle_sha256",
        "lifecycle",
        "namespace_policy",
        "post_dgx_release",
        "preregistration_sha256",
        "qualification_id",
        "schema_version",
        "seed_bank",
        "status",
        "workload",
    }
    if set(payload) != expected_keys:
        raise PlayQualificationError("PLAY-QUAL preregistration fields drifted")
    if payload["schema_version"] != SCHEMA_VERSION or isinstance(
        payload["schema_version"], bool
    ):
        raise PlayQualificationError("PLAY-QUAL preregistration schema is invalid")
    if payload["qualification_id"] != QUALIFICATION_ID:
        raise PlayQualificationError("PLAY-QUAL qualification ID drifted")
    if payload["status"] != PREREGISTRATION_STATUS:
        raise PlayQualificationError("PLAY-QUAL preregistration is not sealed")
    recorded_body_sha = _require_sha256(
        payload["preregistration_sha256"], name="preregistration_sha256"
    )
    body = dict(payload)
    del body["preregistration_sha256"]
    if _json_sha256(body, domain=_PREREGISTRATION_BODY_DOMAIN) != recorded_body_sha:
        raise PlayQualificationError("PLAY-QUAL preregistration body digest mismatch")
    if payload["seed_bank"] != _cohort_record():
        raise PlayQualificationError("PLAY-QUAL unseen seed bank drifted")
    if payload["workload"] != _workload_record():
        raise PlayQualificationError("PLAY-QUAL workload drifted")
    if payload["lifecycle"] != [
        "publish_preregistration_create_only",
        "measure_wallclock_create_only",
        "qualify_exact_cohort_once",
        "publish_result_create_only",
    ]:
        raise PlayQualificationError("PLAY-QUAL lifecycle drifted")
    if payload["namespace_policy"] != {
        "cpu_qual_or_test_materialized_by_this_runner": False,
        "environment_seed_namespace": (
            "uint64_top_bits_11_unassigned_not_train_validation_test"
        ),
        "training_or_calibration_allowed": False,
    }:
        raise PlayQualificationError("PLAY-QUAL namespace policy drifted")
    current_source = training_source_bundle(project_root)
    if payload["full_source_bundle_sha256"] != current_source["sha256"]:
        raise PlayQualificationError("full model source changed after sealing")
    evaluator = payload["evaluator_bundle"]
    if not isinstance(evaluator, Mapping) or dict(evaluator) != production_evaluator_bundle(
        project_root
    ):
        raise PlayQualificationError("PLAY-QUAL evaluator source changed after sealing")
    if payload["environment_source_sha256"] != environment_source_sha256(project_root):
        raise PlayQualificationError("MazeChase source changed after sealing")
    if payload["environment_config_sha256"] != exact_maze_environment_config_sha256(
        PLAY_QUAL_MAX_TICKS
    ):
        raise PlayQualificationError("MazeChase configuration changed after sealing")
    checkpoint = payload["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "checkpoint_sha256",
        "envelope",
        "feature_flags_sha256",
        "hazard_calibration_sha256",
        "model_config_sha256",
        "model_seed",
        "state_sha256",
    }:
        raise PlayQualificationError("PLAY-QUAL checkpoint binding is malformed")
    if checkpoint["envelope"] != POST_DGX_ENVELOPE:
        raise PlayQualificationError("PLAY-QUAL requires the post-DGX envelope")
    for name in (
        "checkpoint_sha256",
        "feature_flags_sha256",
        "hazard_calibration_sha256",
        "model_config_sha256",
        "state_sha256",
    ):
        _require_sha256(checkpoint[name], name=name)
    _require_plain_int(checkpoint["model_seed"], name="model_seed")
    _require_sha256(payload["full_source_bundle_sha256"], name="full_source_bundle_sha256")
    expected_config_sha = _json_sha256(
        asdict(_expected_model_config()), domain=_MODEL_CONFIG_DOMAIN
    )
    expected_flags_sha = _json_sha256(
        asdict(CONFIG_B_PREDICTIVE), domain=_FEATURE_FLAGS_DOMAIN
    )
    if checkpoint["model_config_sha256"] != expected_config_sha:
        raise PlayQualificationError("sealed model configuration is not exact V2.1i")
    if checkpoint["feature_flags_sha256"] != expected_flags_sha:
        raise PlayQualificationError("sealed feature flags are not exact V2.1i")
    release = payload["post_dgx_release"]
    if not isinstance(release, Mapping) or set(release) != {
        "artifact",
        "body_sha256",
        "file_sha256",
        "training_runtime",
    }:
        raise PlayQualificationError("post-DGX release binding is malformed")
    artifact = release["artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "body",
        "body_sha256",
        "qualification_id",
        "schema_version",
        "status",
    }:
        raise PlayQualificationError("post-DGX release artifact fields drifted")
    body = artifact["body"]
    if not isinstance(body, Mapping):
        raise PlayQualificationError("post-DGX release body is missing")
    expected_body_sha = _json_sha256(
        dict(body), domain=b"IRV21IPOSTDGXRELEASEBODY\x01"
    )
    if (
        artifact["body_sha256"] != expected_body_sha
        or release["body_sha256"] != expected_body_sha
    ):
        raise PlayQualificationError("post-DGX release body digest mismatch")
    expected_file_sha = sha256(
        (_canonical_json(dict(artifact)) + "\n").encode("utf-8")
    ).hexdigest()
    if release["file_sha256"] != expected_file_sha:
        raise PlayQualificationError("post-DGX release file digest mismatch")
    if artifact["qualification_id"] != "v21i-post-dgx-cpu-qual-release-v1":
        raise PlayQualificationError("post-DGX release qualification ID drifted")
    if artifact["schema_version"] != 1 or artifact["status"] != "passed":
        raise PlayQualificationError("post-DGX release did not pass")
    if release["training_runtime"] != body.get("dgx_lineage"):
        raise PlayQualificationError("post-DGX lineage binding drifted")
    embedded_release = PostDgxReleaseReceipt(
        path=Path("embedded-post-dgx-release.json"),
        payload=dict(artifact),
        body=dict(body),
        body_sha256=expected_body_sha,
        file_sha256=expected_file_sha,
    )
    try:
        validate_post_dgx_release_binding(
            embedded_release,
            checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
            state_sha256=str(checkpoint["state_sha256"]),
            model_config_sha256=str(checkpoint["model_config_sha256"]),
            feature_flags_sha256=str(checkpoint["feature_flags_sha256"]),
            hazard_calibration_sha256=str(
                checkpoint["hazard_calibration_sha256"]
            ),
            source_bundle_sha256=str(payload["full_source_bundle_sha256"]),
            model_seed=int(checkpoint["model_seed"]),
            training_runtime=release["training_runtime"],
        )
    except Exception as error:
        raise PlayQualificationError(
            "post-DGX release lacks exact 5/5 CPU-QUAL authorization"
        ) from error


def publish_play_qual_preregistration_create_only(
    path: Path,
    payload: Mapping[str, object],
    *,
    project_root: Path,
) -> str:
    validate_play_qual_preregistration(payload, project_root=project_root)
    return _publish_json_create_only(path, payload)


def load_sealed_play_qual_preregistration(
    path: Path,
    *,
    project_root: Path,
) -> SealedPlayQualification:
    payload, encoded = _strict_json_file(path, name="PLAY-QUAL preregistration")
    validate_play_qual_preregistration(payload, project_root=project_root)
    return SealedPlayQualification(
        path=path.resolve(),
        file_sha256=sha256(encoded).hexdigest(),
        preregistration_sha256=str(payload["preregistration_sha256"]),
        payload=payload,
    )


def _checkpoint_binding(payload: Mapping[str, object]) -> Mapping[str, object]:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise PlayQualificationError("PLAY-QUAL checkpoint binding is missing")
    return checkpoint


def live_cohort_receipt(
    registration: SealedPlayQualification,
) -> dict[str, object]:
    """Adapt the create-only seed-bank seal to the live evaluator contract."""

    seed_bank = registration.payload.get("seed_bank")
    if seed_bank != _cohort_record():
        raise PlayQualificationError("sealed PLAY-QUAL seed bank is invalid")
    return {
        "cluster_ids": [
            f"play-qual-environment-{index:02d}"
            for index in range(PLAY_QUAL_EPISODES)
        ],
        "episode_seeds": list(fixed_play_qual_environment_seeds()),
        "max_ticks": PLAY_QUAL_MAX_TICKS,
        "namespace": "uint64_top_bits_11_unassigned_play_qual_v1",
        "preregistration_sha256": registration.preregistration_sha256,
        "qualification_id": QUALIFICATION_ID,
        "random_baseline_streams": PLAY_QUAL_RANDOM_STREAMS,
        "sealed": True,
        "seed_bank": seed_bank,
        "test_accessed": False,
    }


def _verify_loaded_checkpoint_binding(
    checkpoint: LoadedCheckpoint,
    registration: SealedPlayQualification,
) -> None:
    binding = _checkpoint_binding(registration.payload)
    expected = {
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "envelope": POST_DGX_ENVELOPE,
        "feature_flags_sha256": checkpoint.feature_flags_sha256,
        "hazard_calibration_sha256": checkpoint.hazard_calibration_sha256,
        "model_config_sha256": checkpoint.model_config_sha256,
        "model_seed": checkpoint.model_seed,
        "state_sha256": checkpoint.state_sha256,
    }
    if dict(binding) != expected:
        raise PlayQualificationError("loaded checkpoint differs from PLAY-QUAL seal")
    if registration.payload["full_source_bundle_sha256"] != (
        checkpoint.source_bundle_sha256
    ):
        raise PlayQualificationError("loaded source differs from PLAY-QUAL seal")
    if registration.payload["post_dgx_release"] != _post_dgx_release_record(
        checkpoint
    ):
        raise PlayQualificationError("loaded post-DGX release differs from PLAY-QUAL seal")


def _load_wallclock_artifact(
    path: Path,
    *,
    checkpoint: LoadedCheckpoint,
    registration: SealedPlayQualification,
) -> Mapping[str, object]:
    artifact, _encoded = _strict_json_file(path, name="wall-clock qualification artifact")
    try:
        receipt = validate_wallclock_artifact(
            artifact,
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
            expected_source_bundle_sha256=checkpoint.source_bundle_sha256,
            expected_preregistration_sha256=registration.preregistration_sha256,
            required_checkpoint_envelope=POST_DGX_ENVELOPE,
            expected_post_dgx_release_receipt_sha256=(
                checkpoint.post_dgx_release.file_sha256
            ),
        )
    except Exception as error:
        raise PlayQualificationError("wall-clock qualification receipt is invalid") from error
    if receipt.get("model_config_sha256") != checkpoint.model_config_sha256:
        raise PlayQualificationError("wall-clock receipt model config mismatch")
    if receipt.get("feature_flags_sha256") != checkpoint.feature_flags_sha256:
        raise PlayQualificationError("wall-clock receipt feature flags mismatch")
    if receipt.get("hazard_calibration_sha256") != checkpoint.hazard_calibration_sha256:
        raise PlayQualificationError("wall-clock receipt calibration mismatch")
    if receipt.get("model_state_sha256") != checkpoint.state_sha256:
        raise PlayQualificationError("wall-clock receipt state mismatch")
    if receipt.get("model_seed") != checkpoint.model_seed:
        raise PlayQualificationError("wall-clock receipt model seed mismatch")
    if receipt.get("evaluator_bundle_sha256") != wallclock_evaluator_bundle_sha256():
        raise PlayQualificationError("wall-clock evaluator source mismatch")
    if receipt.get("checkpoint_envelope") != POST_DGX_ENVELOPE:
        raise PlayQualificationError("wall-clock checkpoint envelope mismatch")
    if receipt.get("post_dgx_release_body_sha256") != (
        checkpoint.post_dgx_release.body_sha256
    ):
        raise PlayQualificationError("wall-clock post-DGX release body mismatch")
    if receipt.get("post_dgx_release_file_sha256") != (
        checkpoint.post_dgx_release.file_sha256
    ):
        raise PlayQualificationError("wall-clock post-DGX release file mismatch")
    return artifact


def _production_live_config() -> LivePlayQualificationConfig:
    """Construct the only live-play workload accepted by this entry point."""

    return LivePlayQualificationConfig(
        episode_seeds=fixed_play_qual_environment_seeds(),
        cluster_ids=tuple(
            f"play-qual-environment-{index:02d}" for index in range(PLAY_QUAL_EPISODES)
        ),
        max_ticks=PLAY_QUAL_MAX_TICKS,
        model_reset_seed=OUTCOME_AWARE_MODEL_SEED_V1,
        random_baseline_seed=PLAY_QUAL_RANDOM_BASELINE_SEED,
        bootstrap_seed=PLAY_QUAL_BOOTSTRAP_SEED,
        bootstrap_resamples=PLAY_QUAL_BOOTSTRAP_RESAMPLES,
        confidence_level=PLAY_QUAL_CONFIDENCE_LEVEL,
        minimum_reward_advantage=0.0,
        minimum_catch_advantage=0.0,
        deterministic_repeats=PLAY_QUAL_DETERMINISTIC_REPEATS,
    )


def _publish_preflight_failure(
    output_path: Path,
    *,
    error: BaseException,
    checkpoint_path: Path,
    preregistration_path: Path,
    post_dgx_release_path: Path,
    wallclock_path: Path,
) -> None:
    payload = {
        "classification": "production_play_qualification_preflight_failed",
        "failure": {"message": str(error), "type": type(error).__name__},
        "inputs": {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "preregistration_path": str(preregistration_path.resolve()),
            "post_dgx_release_path": str(post_dgx_release_path.resolve()),
            "wallclock_path": str(wallclock_path.resolve()),
        },
        "qualification_id": QUALIFICATION_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
    }
    _publish_json_create_only(output_path, payload)


def run_production_play_qualification(
    *,
    checkpoint_path: Path,
    preregistration_path: Path,
    post_dgx_release_path: Path,
    wallclock_path: Path,
    output_path: Path,
    project_root: Path,
) -> None:
    """Run exact PLAY-QUAL with no injectable controller or environment."""

    if output_path.exists():
        raise PlayQualificationError(f"refusing to overwrite existing artifact: {output_path}")
    try:
        registration = load_sealed_play_qual_preregistration(
            preregistration_path, project_root=project_root
        )
        binding = _checkpoint_binding(registration.payload)
        release_binding = registration.payload["post_dgx_release"]
        if not isinstance(release_binding, Mapping):
            raise PlayQualificationError("post-DGX release binding is missing")
        try:
            post_dgx_release = load_post_dgx_release_receipt(
                post_dgx_release_path,
                expected_file_sha256=str(release_binding["file_sha256"]),
            )
        except Exception as error:
            raise PlayQualificationError("post-DGX release receipt is invalid") from error
        if dict(post_dgx_release.payload) != release_binding["artifact"]:
            raise PlayQualificationError("post-DGX release differs from PLAY-QUAL seal")
        checkpoint = load_exact_v21i_checkpoint(
            checkpoint_path,
            project_root=project_root,
            expected_checkpoint_sha256=str(binding["checkpoint_sha256"]),
            expected_source_bundle_sha256=str(
                registration.payload["full_source_bundle_sha256"]
            ),
            expected_model_seed=int(binding["model_seed"]),
            post_dgx_release=post_dgx_release,
        )
        _verify_loaded_checkpoint_binding(checkpoint, registration)
        wallclock_artifact = _load_wallclock_artifact(
            wallclock_path,
            checkpoint=checkpoint,
            registration=registration,
        )
        cohort_receipt = live_cohort_receipt(registration)
        provenance = LivePlayProvenance(
            qualification_id=QUALIFICATION_ID,
            model_identity=OutcomeAwareCoreV2MazePolicy.identity,
            checkpoint_envelope=POST_DGX_ENVELOPE,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            model_state_sha256=checkpoint.state_sha256,
            model_config_sha256=checkpoint.model_config_sha256,
            feature_flags_sha256=checkpoint.feature_flags_sha256,
            source_bundle_sha256=checkpoint.source_bundle_sha256,
            evaluator_bundle_sha256=str(
                registration.payload["evaluator_bundle"]["sha256"]
            ),
            environment_source_sha256=str(
                registration.payload["environment_source_sha256"]
            ),
            preregistration_sha256=registration.preregistration_sha256,
            cohort_binding_sha256=live_play_cohort_binding_sha256(cohort_receipt),
            post_dgx_release_body_sha256=(
                checkpoint.post_dgx_release.body_sha256
            ),
            post_dgx_release_file_sha256=(
                checkpoint.post_dgx_release.file_sha256
            ),
        )
        controller = OutcomeAwareCoreV2MazePolicy(checkpoint.model)
        # The production function must not grow an injected runner surface.
        if "episode_runner" in inspect.signature(qualify_live_play_create_only).parameters:
            raise PlayQualificationError(
                "live qualifier still exposes an arbitrary episode runner"
            )
        qualify_live_play_create_only(
            controller,
            config=_production_live_config(),
            provenance=provenance,
            cohort_receipt=cohort_receipt,
            wall_clock_artifact=wallclock_artifact,
            output_path=output_path,
        )
    except BaseException as error:
        if not output_path.exists():
            _publish_preflight_failure(
                output_path,
                error=error,
                checkpoint_path=checkpoint_path,
                preregistration_path=preregistration_path,
                post_dgx_release_path=post_dgx_release_path,
                wallclock_path=wallclock_path,
            )
        raise


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--checkpoint", required=True)
    preregister.add_argument("--post-dgx-release", required=True)
    preregister.add_argument("--post-dgx-release-sha256", required=True)
    preregister.add_argument("--output", required=True)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--checkpoint", required=True)
    qualify.add_argument("--preregistration", required=True)
    qualify.add_argument("--post-dgx-release", required=True)
    qualify.add_argument("--wallclock-receipt", required=True)
    qualify.add_argument("--output", required=True)
    args = parser.parse_args()
    project_root = _project_root()
    if args.operation == "preregister":
        post_dgx_release = load_post_dgx_release_receipt(
            Path(args.post_dgx_release).expanduser().resolve(),
            expected_file_sha256=args.post_dgx_release_sha256,
        )
        release_checkpoint = post_dgx_release.body.get("checkpoint")
        if not isinstance(release_checkpoint, Mapping):
            parser.error("post-DGX release checkpoint binding is missing")
        checkpoint = load_exact_v21i_checkpoint(
            Path(args.checkpoint).expanduser().resolve(),
            project_root=project_root,
            expected_checkpoint_sha256=str(
                release_checkpoint["checkpoint_sha256"]
            ),
            expected_source_bundle_sha256=str(
                release_checkpoint["source_bundle_sha256"]
            ),
            expected_model_seed=int(release_checkpoint["model_seed"]),
            post_dgx_release=post_dgx_release,
        )
        payload = build_play_qual_preregistration(
            checkpoint=checkpoint,
            project_root=project_root,
        )
        destination = Path(args.output).expanduser().resolve()
        publish_play_qual_preregistration_create_only(
            destination,
            payload,
            project_root=project_root,
        )
        print(f"SEALED PLAY-QUAL preregistration -> {destination}", flush=True)
        return
    run_production_play_qualification(
        checkpoint_path=Path(args.checkpoint).expanduser().resolve(),
        preregistration_path=Path(args.preregistration).expanduser().resolve(),
        post_dgx_release_path=Path(args.post_dgx_release).expanduser().resolve(),
        wallclock_path=Path(args.wallclock_receipt).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        project_root=project_root,
    )
    print(f"PASSED production PLAY-QUAL -> {Path(args.output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
