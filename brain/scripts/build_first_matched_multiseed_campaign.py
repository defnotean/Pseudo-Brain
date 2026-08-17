"""Build the immutable inputs for the first matched multi-seed campaign.

The architecture campaign is intentionally blocked: the historical v1
architecture manifest registers 500-step recipes whose reference candidate did
not pass qualification.  This tool can freeze the real held-out suites and
provides reusable config/execution helpers, but it refuses to emit a campaign
registration, effective configs, or an execution record.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import sys
import tomllib
from types import MappingProxyType
from typing import Mapping


BRAIN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC = BRAIN_ROOT / "src"
for path in (str(SRC), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from first_matched_suite_contract import (  # noqa: E402
    MOVEMENT_KEYS,
    canonical_json,
    evaluation_code_manifest,
    sample_manifest_sha256,
    strict_json_bytes,
    suite_manifest_sha256,
    validate_suite_manifest,
)
from irene_brain.data import (  # noqa: E402
    DatasetSplit,
    MovingShapesDatasetConfig,
    dataset_manifest_sha256,
)
from irene_brain.evaluation.multiseed_comparison import (  # noqa: E402
    CampaignExecution,
    CampaignRegistration,
    ExecutedRun,
    RegisteredRun,
    RegisteredSuite,
)
from irene_brain.project_paths import resolve_workspace_path  # noqa: E402


HISTORICAL_ARCHITECTURE_MANIFEST_SHA256 = (
    "52bba6a9b723dda51da18d2842c5eaf360453db011e7f97bbb066e4034905421"
)
TARGET_OPTIMIZER_STEPS = 2_048
HISTORICAL_OPTIMIZER_STEPS = 500
TRAINING_SEEDS = tuple(range(1701, 1709))
SUITE_SET_ID = "moving-shapes-heldout-hazards-1-3-5-v1"
SUITE_ROOT_RELATIVE = Path("configs/evaluation") / SUITE_SET_ID
BLOCKER_RELATIVE = (
    Path("configs/campaigns") / "first-matched-8x3-v1" / "campaign-blocker.json"
)
SUITE_SPECS = (
    ("moving-shapes-test-hazards-1-v1", 1, 2_097_152),
    ("moving-shapes-test-hazards-3-v1", 3, 2_097_408),
    ("moving-shapes-test-hazards-5-v1", 5, 2_097_664),
)
FORBIDDEN_WHILE_BLOCKED = (
    Path("configs/campaigns/first-matched-8x3-v1/campaign-registration.json"),
    Path("configs/campaigns/first-matched-8x3-v1/comparison-criterion.json"),
    Path("configs/campaigns/first-matched-8x3-v1/campaign-execution.json"),
    Path("configs/campaigns/first-matched-8x3-v1/effective-configs"),
)
_RUN_NAME = re.compile(r'^name = "[^"]*"(?P<newline>\r\n|\n)?$')
_RUN_SEED = re.compile(r"^seed = [0-9]+(?P<newline>\r\n|\n)?$")


class CampaignBlockedError(RuntimeError):
    """Raised when code attempts to register the scientifically blocked run."""


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    raw_toml: str
    raw_sha256: str
    canonical_json: str
    canonical_sha256: str
    training_budget_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedSuiteSet:
    manifest: Mapping[str, object]
    suites: tuple[Mapping[str, object], ...]
    registrations: tuple[RegisteredSuite, ...]


@dataclass(frozen=True, slots=True)
class RuntimeRegistration:
    schema_version: int
    campaign_sha256: str
    runtime_fingerprint: Mapping[str, str | int | bool]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("runtime registration schema_version must be integer 1")
        _digest(self.campaign_sha256, name="campaign_sha256")
        fingerprint = _runtime_fingerprint(self.runtime_fingerprint)
        object.__setattr__(self, "runtime_fingerprint", MappingProxyType(fingerprint))

    @property
    def canonical_json(self) -> str:
        return canonical_json(
            {
                "schema_version": self.schema_version,
                "campaign_sha256": self.campaign_sha256,
                "runtime_fingerprint": dict(self.runtime_fingerprint),
            }
        )

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.canonical_json.encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _runtime_fingerprint(
    value: object,
) -> dict[str, str | int | bool]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("runtime_fingerprint must be a nonempty object")
    result: dict[str, str | int | bool] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("runtime fingerprint names must be nonempty strings")
        if type(item) not in {str, int, bool}:
            raise ValueError(
                "runtime fingerprint values must be strings, integers, or booleans"
            )
        result[name] = item
    return dict(sorted(result.items()))


def _with_self_digest(payload: Mapping[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = _sha256_bytes(canonical_json(payload).encode("utf-8"))
    return result


def _canonical_file(payload: Mapping[str, object]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact path must be a nonempty POSIX relative path")
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("artifact path escapes the brain root") from error
    return candidate


def _exact_dataclass_fields(
    value: Mapping[str, object], dataclass_type: type[object], *, name: str
) -> None:
    expected = {field.name for field in fields(dataclass_type)}
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _suite_manifest(
    *, suite_id: str, hazard_count: int, seed_offset: int, brain_root: Path
) -> dict[str, object]:
    config = MovingShapesDatasetConfig(
        split=DatasetSplit.TEST,
        sequence_count=256,
        sequence_length=8,
        seed_offset=seed_offset,
        hazard_count=hazard_count,
        tick_period_ns=16_666_667,
        discount=0.99,
    )
    burn_in_steps = 2
    evaluated_samples = config.sequence_count * (config.sequence_length - burn_in_steps)
    sample_manifest = {
        "schema_version": 1,
        "dataset_manifest": config.manifest_dict(),
        "dataset_manifest_sha256": dataset_manifest_sha256(config),
        "burn_in_steps": burn_in_steps,
        "evaluated_transition_indices": {
            "start_inclusive": burn_in_steps,
            "stop_exclusive": config.sequence_length,
        },
        "evaluated_samples": evaluated_samples,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "suite_id": suite_id,
        "scope": (
            "Deterministic in-repository MovingShapes test-namespace trajectories "
            f"with hazard_count={hazard_count}; this is same-family held-out "
            "control evaluation reserved for the future architecture campaign, "
            "not general game intelligence."
        ),
        "weight": 1.0,
        "evaluated_samples": evaluated_samples,
        "sample_manifest": sample_manifest,
        "evaluation_sample_manifest_sha256": sample_manifest_sha256(sample_manifest),
        "metric": {
            "metric_id": "movement_exact_match",
            "model_exit": "final_anytime_exit",
            "prediction_rule": "button_logit > 0.0",
            "target_rule": "target_control > 0.5",
            "movement_keys": list(MOVEMENT_KEYS),
            "per_sample_score": (
                "1 iff predicted and target active sets agree exactly on W/A/S/D; "
                "else 0"
            ),
            "aggregation": (
                "sample-weighted arithmetic mean over post-burn-in decisions"
            ),
            "normalized_range": [0.0, 1.0],
        },
        "evaluation_code": evaluation_code_manifest(brain_root),
    }
    payload["suite_manifest_sha256"] = suite_manifest_sha256(payload)
    validate_suite_manifest(payload, brain_root=brain_root)
    return payload


def build_suite_artifacts(brain_root: Path = BRAIN_ROOT) -> dict[Path, bytes]:
    root = brain_root.resolve()
    artifacts: dict[Path, bytes] = {}
    suite_entries: list[dict[str, object]] = []
    for suite_id, hazard_count, seed_offset in SUITE_SPECS:
        suite = _suite_manifest(
            suite_id=suite_id,
            hazard_count=hazard_count,
            seed_offset=seed_offset,
            brain_root=root,
        )
        relative = SUITE_ROOT_RELATIVE / "suites" / f"{suite_id}.json"
        encoded = _canonical_file(suite)
        artifacts[relative] = encoded
        suite_entries.append(
            {
                "suite_id": suite_id,
                "path": relative.as_posix(),
                "raw_sha256": _sha256_bytes(encoded),
                "suite_manifest_sha256": suite["suite_manifest_sha256"],
                "evaluation_sample_manifest_sha256": suite[
                    "evaluation_sample_manifest_sha256"
                ],
                "evaluation_code_sha256": suite["evaluation_code"][
                    "evaluation_code_sha256"
                ],
                "evaluated_samples": suite["evaluated_samples"],
                "weight": suite["weight"],
            }
        )
    set_payload = {
        "schema_version": 1,
        "suite_set_id": SUITE_SET_ID,
        "status": "frozen_before_architecture_campaign",
        "scope": (
            "Three equally weighted, disjoint deterministic MovingShapes test "
            "slices at hazard counts 1, 3, and 5, scored only by final-exit "
            "movement_exact_match."
        ),
        "reservation": {
            "purpose": "future matched-architecture campaign only",
            "local_half_open_ranges": [
                [2_097_152, 2_097_408],
                [2_097_408, 2_097_664],
                [2_097_664, 2_097_920],
            ],
            "disjoint_from": [
                "RCQ TEST recipients",
                "RCQ TEST donor-only pool",
                "RCQ TEST guard band",
                "opened historical Stage-A slices",
            ],
        },
        "suites": suite_entries,
    }
    suite_set = _with_self_digest(set_payload, "suite_set_sha256")
    artifacts[SUITE_ROOT_RELATIVE / "suite-set.json"] = _canonical_file(suite_set)
    return artifacts


def campaign_blocker() -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "status": "blocked_non_launchable",
        "historical_architecture_manifest_sha256": (
            HISTORICAL_ARCHITECTURE_MANIFEST_SHA256
        ),
        "historical_manifest_role": (
            "Preserved design-history manifest only; its registered 500-step "
            "recipes are not authorized for the first multi-seed claim campaign."
        ),
        "reference_candidate_qualification": "not_passed",
        "historical_recipe_optimizer_steps": HISTORICAL_OPTIMIZER_STEPS,
        "required_recipe_optimizer_steps": TARGET_OPTIMIZER_STEPS,
        "planned_training_seeds": list(TRAINING_SEEDS),
        "planned_suite_set_id": SUITE_SET_ID,
        "reserved_test_local_half_open_range": [2_097_152, 2_097_920],
        "required_before_campaign_registration": [
            "Reference Candidate Qualification passes under its frozen gate.",
            "A new versioned architecture manifest freezes the qualified source bundle.",
            "That new manifest registers matched 2048-step staged recipes for all variants.",
            "Every effective per-seed config is derived by changing only run.name and run.seed.",
            "A common runtime fingerprint is preregistered before training.",
            "A multiseed evaluator with CampaignExecution schema 2 binds that "
            "runtime identity into execution evidence and observations.",
        ],
        "forbidden_outputs_while_blocked": [
            path.as_posix() for path in FORBIDDEN_WHILE_BLOCKED
        ],
        "checkpoint_binding_rule": (
            "No CampaignExecution exists before all registered runs finish. The "
            "post-run binder must hash checkpoint bytes and verify config, data, "
            "source, run identity through config hash, and optimizer cursor."
        ),
        "current_evaluator_runtime_limitation": (
            "CampaignExecution schema 1 does not carry runtime identity. Until "
            "a multiseed evaluator with CampaignExecution schema 2 binds the "
            "preregistered runtime into execution and observations, results "
            "cannot support a runtime-matched claim."
        ),
    }
    return _with_self_digest(payload, "blocker_sha256")


def build_static_artifacts(brain_root: Path = BRAIN_ROOT) -> dict[Path, bytes]:
    artifacts = build_suite_artifacts(brain_root)
    artifacts[BLOCKER_RELATIVE] = _canonical_file(campaign_blocker())
    return dict(sorted(artifacts.items(), key=lambda item: item[0].as_posix()))


def _replace_run_fields(template_raw_toml: str, *, run_id: str, seed: int) -> str:
    if not isinstance(run_id, str) or not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id
    ):
        raise ValueError("run_id must use lowercase ASCII letters, digits, and hyphens")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    lines = template_raw_toml.splitlines(keepends=True)
    in_run = False
    replaced_name = 0
    replaced_seed = 0
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_run = stripped == "[run]"
        if in_run:
            name_match = _RUN_NAME.fullmatch(line)
            seed_match = _RUN_SEED.fullmatch(line)
            if name_match is not None:
                line = f'name = "{run_id}"{name_match.group("newline")}'
                replaced_name += 1
            elif seed_match is not None:
                line = f'seed = {seed}{seed_match.group("newline")}'
                replaced_seed += 1
        result.append(line)
    if replaced_name != 1 or replaced_seed != 1:
        raise ValueError("template must contain exactly one run name and seed")
    return "".join(result)


def derive_effective_config(
    template_raw_toml: bytes, *, run_id: str, training_seed: int
) -> EffectiveConfig:
    try:
        template_text = template_raw_toml.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("template config is not UTF-8") from error
    if template_text.encode("utf-8") != template_raw_toml:
        raise ValueError("template config bytes do not round-trip losslessly")
    try:
        template_payload = tomllib.loads(template_text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError("template config is invalid TOML") from error
    if template_payload.get("schema_version") not in {2, 3}:
        raise ValueError("campaign templates must use training-config schema 2 or 3")
    effective_text = _replace_run_fields(
        template_text,
        run_id=run_id,
        seed=training_seed,
    )
    effective_payload = tomllib.loads(effective_text)
    template_normalized = json.loads(canonical_json(template_payload))
    effective_normalized = json.loads(canonical_json(effective_payload))
    for payload in (template_normalized, effective_normalized):
        del payload["run"]["name"]
        del payload["run"]["seed"]
    if canonical_json(template_normalized) != canonical_json(effective_normalized):
        raise RuntimeError("effective config changed a field beyond run.name/run.seed")
    budget_payload = json.loads(canonical_json(effective_payload))
    for field in ("name", "seed", "model_factory"):
        del budget_payload["run"][field]
    effective_canonical = canonical_json(effective_payload)
    return EffectiveConfig(
        raw_toml=effective_text,
        raw_sha256=_sha256_bytes(effective_text.encode("utf-8")),
        canonical_json=effective_canonical,
        canonical_sha256=_sha256_bytes(effective_canonical.encode("utf-8")),
        training_budget_sha256=_sha256_bytes(
            canonical_json(budget_payload).encode("utf-8")
        ),
    )


def build_campaign_artifacts(*_args: object, **_kwargs: object) -> dict[Path, bytes]:
    raise CampaignBlockedError(
        "campaign generation is blocked: manifest v1 registers unqualified "
        "500-step recipes; wait for passed RCQ and a new versioned 2048-step manifest"
    )


def load_suite_set(brain_root: Path = BRAIN_ROOT) -> LoadedSuiteSet:
    root = brain_root.resolve()
    expected = build_suite_artifacts(root)
    for relative, encoded in expected.items():
        path = root / relative
        if not path.is_file() or path.read_bytes() != encoded:
            raise ValueError(f"checked-in suite artifact differs from generator: {relative}")
    set_path = root / SUITE_ROOT_RELATIVE / "suite-set.json"
    raw_set = set_path.read_bytes()
    suite_set = strict_json_bytes(raw_set, name=str(set_path))
    if raw_set != _canonical_file(suite_set):
        raise ValueError("suite-set file is not canonical JSON plus one LF")
    expected_digest = suite_set.get("suite_set_sha256")
    payload = dict(suite_set)
    payload.pop("suite_set_sha256", None)
    if _sha256_bytes(canonical_json(payload).encode("utf-8")) != expected_digest:
        raise ValueError("suite-set self-digest mismatch")
    entries = suite_set.get("suites")
    if not isinstance(entries, list) or len(entries) != len(SUITE_SPECS):
        raise ValueError("suite-set has the wrong suite matrix")
    suites: list[Mapping[str, object]] = []
    registrations: list[RegisteredSuite] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("suite-set entries must be objects")
        path = _safe_path(root, entry.get("path"))
        raw = path.read_bytes()
        if _sha256_bytes(raw) != entry.get("raw_sha256"):
            raise ValueError("suite-set raw suite digest mismatch")
        suite = strict_json_bytes(raw, name=str(path))
        if raw != _canonical_file(suite):
            raise ValueError("suite file is not canonical JSON plus one LF")
        validated = validate_suite_manifest(suite, brain_root=root)
        for name in (
            "suite_id",
            "suite_manifest_sha256",
            "evaluation_sample_manifest_sha256",
            "evaluated_samples",
            "weight",
        ):
            if entry.get(name) != validated.get(name):
                raise ValueError(f"suite-set {name} differs from suite manifest")
        evaluation_code = validated["evaluation_code"]
        assert isinstance(evaluation_code, Mapping)
        if entry.get("evaluation_code_sha256") != evaluation_code.get(
            "evaluation_code_sha256"
        ):
            raise ValueError("suite-set evaluation code digest mismatch")
        suites.append(validated)
        registrations.append(
            RegisteredSuite(
                suite_id=validated["suite_id"],
                weight=validated["weight"],
                evaluation_sample_manifest_sha256=validated[
                    "evaluation_sample_manifest_sha256"
                ],
                evaluation_code_sha256=evaluation_code["evaluation_code_sha256"],
                evaluated_samples=validated["evaluated_samples"],
            )
        )
    return LoadedSuiteSet(
        manifest=suite_set,
        suites=tuple(suites),
        registrations=tuple(registrations),
    )


def load_campaign_registration(
    path: Path, *, expected_campaign_sha256: str
) -> CampaignRegistration:
    raw = path.read_bytes()
    payload = strict_json_bytes(raw, name=str(path))
    if raw != _canonical_file(payload):
        raise ValueError("campaign registration is not canonical JSON plus one LF")
    _exact_dataclass_fields(payload, CampaignRegistration, name="campaign")
    suites = payload["suites"]
    runs = payload["runs"]
    if not isinstance(suites, list) or not isinstance(runs, list):
        raise ValueError("campaign suites and runs must be arrays")
    registered_suites: list[RegisteredSuite] = []
    for suite in suites:
        if not isinstance(suite, Mapping):
            raise ValueError("campaign suite entries must be objects")
        _exact_dataclass_fields(suite, RegisteredSuite, name="registered suite")
        registered_suites.append(RegisteredSuite(**suite))
    registered_runs: list[RegisteredRun] = []
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("campaign run entries must be objects")
        _exact_dataclass_fields(run, RegisteredRun, name="registered run")
        registered_runs.append(RegisteredRun(**run))
    campaign_fields = dict(payload)
    campaign_fields["training_seeds"] = tuple(payload["training_seeds"])
    campaign_fields["suites"] = tuple(registered_suites)
    campaign_fields["runs"] = tuple(registered_runs)
    campaign = CampaignRegistration(**campaign_fields)
    if campaign.canonical_json.encode("utf-8") + b"\n" != raw:
        raise ValueError("campaign dataclass does not reproduce registered bytes")
    if campaign.sha256 != expected_campaign_sha256:
        raise ValueError("campaign registration differs from externally pinned digest")
    return campaign


def load_runtime_registration(
    path: Path,
    *,
    expected_runtime_registration_sha256: str,
) -> RuntimeRegistration:
    raw = path.read_bytes()
    payload = strict_json_bytes(raw, name=str(path))
    if raw != _canonical_file(payload):
        raise ValueError("runtime registration is not canonical JSON plus one LF")
    _exact_dataclass_fields(payload, RuntimeRegistration, name="runtime registration")
    registration = RuntimeRegistration(**payload)
    _digest(
        expected_runtime_registration_sha256,
        name="expected_runtime_registration_sha256",
    )
    if registration.sha256 != expected_runtime_registration_sha256:
        raise ValueError("runtime registration differs from externally pinned digest")
    return registration


def _is_reparse(path: Path) -> bool:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _resolve_checkpoint_path(checkpoint_root: Path, relative: object) -> Path:
    unresolved_root = Path(os.path.abspath(checkpoint_root))
    if not unresolved_root.is_dir() or _is_reparse(unresolved_root):
        raise ValueError("checkpoint_root must be a real, non-reparse directory")
    root = unresolved_root.resolve(strict=True)
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("checkpoint_path must be a nonempty POSIX relative path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
    ):
        raise ValueError("checkpoint_path must stay beneath checkpoint_root")
    candidate = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if _is_reparse(cursor):
            raise ValueError("checkpoint_path cannot traverse a symlink or reparse point")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("checkpoint_path escapes checkpoint_root") from error
    if not resolved.is_file() or not stat.S_ISREG(os.lstat(resolved).st_mode):
        raise ValueError("checkpoint_path must identify a regular file")
    return resolved


def _load_verified_checkpoint(
    checkpoint_path: Path,
    run: RegisteredRun,
    runtime_fingerprint: Mapping[str, str | int | bool],
) -> object:
    from irene_brain.training.checkpoint import file_sha256, load_checkpoint

    before = file_sha256(checkpoint_path)
    loaded = load_checkpoint(
        checkpoint_path,
        expected_config_sha256=run.canonical_config_sha256,
        expected_data_sha256=run.training_data_sha256,
        expected_code_sha256=run.training_code_sha256,
        expected_runtime_fingerprint=runtime_fingerprint,
        expected_checkpoint_sha256=before,
    )
    after = file_sha256(checkpoint_path)
    if after != before or loaded.checkpoint_sha256 != before:
        raise ValueError("checkpoint bytes changed while binding execution")
    return loaded


def bind_campaign_execution(
    campaign: CampaignRegistration,
    checkpoint_spec_path: Path,
    *,
    expected_campaign_sha256: str,
    checkpoint_root: Path,
    runtime_registration: RuntimeRegistration,
    expected_runtime_registration_sha256: str,
) -> CampaignExecution:
    """Post-run only: verify exact checkpoints before constructing execution data."""

    if not isinstance(campaign, CampaignRegistration):
        raise ValueError("campaign must be a CampaignRegistration")
    _digest(expected_campaign_sha256, name="expected_campaign_sha256")
    if campaign.sha256 != expected_campaign_sha256:
        raise ValueError("campaign registration differs from externally pinned digest")
    if not isinstance(runtime_registration, RuntimeRegistration):
        raise ValueError("runtime_registration must be a RuntimeRegistration")
    _digest(
        expected_runtime_registration_sha256,
        name="expected_runtime_registration_sha256",
    )
    if runtime_registration.sha256 != expected_runtime_registration_sha256:
        raise ValueError("runtime registration differs from externally pinned digest")
    if runtime_registration.campaign_sha256 != campaign.sha256:
        raise ValueError("runtime registration targets a different campaign")
    spec_raw = checkpoint_spec_path.read_bytes()
    spec = strict_json_bytes(spec_raw, name=str(checkpoint_spec_path))
    if spec_raw != _canonical_file(spec):
        raise ValueError("checkpoint binding spec is not canonical JSON plus one LF")
    if set(spec) != {
        "schema_version",
        "campaign_sha256",
        "runtime_registration_sha256",
        "runs",
    }:
        raise ValueError("checkpoint binding spec has incompatible fields")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("checkpoint binding schema_version must be integer 1")
    if spec["campaign_sha256"] != campaign.sha256:
        raise ValueError("checkpoint binding spec targets a different campaign")
    if spec["runtime_registration_sha256"] != runtime_registration.sha256:
        raise ValueError("checkpoint binding spec targets a different runtime")
    entries = spec["runs"]
    if not isinstance(entries, list):
        raise ValueError("checkpoint binding runs must be an array")
    by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "variant_id",
            "training_seed",
            "run_id",
            "checkpoint_path",
        }:
            raise ValueError("checkpoint binding run has incompatible fields")
        if (
            not isinstance(entry["variant_id"], str)
            or not entry["variant_id"]
            or type(entry["training_seed"]) is not int
            or entry["training_seed"] < 0
        ):
            raise ValueError("checkpoint binding variant and seed are invalid")
        key = (entry["variant_id"], entry["training_seed"])
        if key in by_key:
            raise ValueError("checkpoint binding variant/seed entries must be unique")
        by_key[key] = entry
    planned = {(run.variant_id, run.training_seed): run for run in campaign.runs}
    if set(by_key) != set(planned):
        raise ValueError("checkpoint binding is not the exact registered run matrix")
    executed: list[ExecutedRun] = []
    for run in campaign.runs:
        entry = by_key[(run.variant_id, run.training_seed)]
        if entry["run_id"] != run.run_id:
            raise ValueError("checkpoint binding run_id differs from registration")
        checkpoint_path = _resolve_checkpoint_path(
            checkpoint_root,
            entry["checkpoint_path"],
        )
        loaded = _load_verified_checkpoint(
            checkpoint_path,
            run,
            runtime_registration.runtime_fingerprint,
        )
        if loaded.cursor.optimizer_step != run.expected_optimizer_step:
            raise ValueError("checkpoint optimizer cursor differs from registration")
        executed.append(
            ExecutedRun(
                variant_id=run.variant_id,
                training_seed=run.training_seed,
                run_id=run.run_id,
                checkpoint_sha256=loaded.checkpoint_sha256,
                optimizer_step=loaded.cursor.optimizer_step,
            )
        )
    return CampaignExecution(
        schema_version=1,
        campaign_sha256=campaign.sha256,
        runs=tuple(executed),
    )


def _write_new_or_identical(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)


def write_static_artifacts(brain_root: Path = BRAIN_ROOT) -> None:
    root = brain_root.resolve()
    for relative in FORBIDDEN_WHILE_BLOCKED:
        if (root / relative).exists():
            raise CampaignBlockedError(
                f"blocked campaign output must not exist: {relative.as_posix()}"
            )
    artifacts = build_static_artifacts(root)
    for relative, content in artifacts.items():
        path = root / relative
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    for relative, content in artifacts.items():
        _write_new_or_identical(root / relative, content)


def check_static_artifacts(brain_root: Path = BRAIN_ROOT) -> None:
    root = brain_root.resolve()
    for relative, content in build_static_artifacts(root).items():
        path = root / relative
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"artifact differs from deterministic generator: {relative}")
    blocker_path = root / BLOCKER_RELATIVE
    blocker_raw = blocker_path.read_bytes()
    blocker = strict_json_bytes(blocker_raw, name=str(blocker_path))
    if blocker_raw != _canonical_file(blocker):
        raise ValueError("campaign blocker is not canonical JSON plus one LF")
    recorded = blocker.get("blocker_sha256")
    blocker_payload = dict(blocker)
    blocker_payload.pop("blocker_sha256", None)
    if _sha256_bytes(canonical_json(blocker_payload).encode("utf-8")) != recorded:
        raise ValueError("campaign blocker self-digest mismatch")
    load_suite_set(root)
    for relative in FORBIDDEN_WHILE_BLOCKED:
        if (root / relative).exists():
            raise CampaignBlockedError(
                f"blocked campaign output must not exist: {relative.as_posix()}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write-static", action="store_true")
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--emit-static-json", action="store_true")
    actions.add_argument("--attempt-campaign", action="store_true")
    actions.add_argument("--bind-execution", action="store_true")
    parser.add_argument("--campaign-registration", type=Path)
    parser.add_argument("--expected-campaign-sha256")
    parser.add_argument("--runtime-registration", type=Path)
    parser.add_argument("--expected-runtime-registration-sha256")
    parser.add_argument("--checkpoint-spec", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--execution-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_static:
        write_static_artifacts()
        return 0
    if args.check:
        check_static_artifacts()
        return 0
    if args.emit_static_json:
        print(
            json.dumps(
                {
                    path.as_posix(): content.decode("utf-8")
                    for path, content in build_static_artifacts().items()
                },
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.attempt_campaign:
        build_campaign_artifacts()
        return 2  # pragma: no cover - fail-closed call above always raises
    required = {
        "--campaign-registration": args.campaign_registration,
        "--expected-campaign-sha256": args.expected_campaign_sha256,
        "--runtime-registration": args.runtime_registration,
        "--expected-runtime-registration-sha256": (
            args.expected_runtime_registration_sha256
        ),
        "--checkpoint-spec": args.checkpoint_spec,
        "--checkpoint-root": args.checkpoint_root,
        "--execution-output": args.execution_output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"--bind-execution requires: {', '.join(missing)}")
    campaign = load_campaign_registration(
        args.campaign_registration,
        expected_campaign_sha256=args.expected_campaign_sha256,
    )
    runtime_registration = load_runtime_registration(
        args.runtime_registration,
        expected_runtime_registration_sha256=(
            args.expected_runtime_registration_sha256
        ),
    )
    execution = bind_campaign_execution(
        campaign,
        args.checkpoint_spec,
        expected_campaign_sha256=args.expected_campaign_sha256,
        checkpoint_root=args.checkpoint_root,
        runtime_registration=runtime_registration,
        expected_runtime_registration_sha256=(
            args.expected_runtime_registration_sha256
        ),
    )
    _write_new_or_identical(
        resolve_workspace_path(args.execution_output),
        (execution.canonical_json + "\n").encode("utf-8"),
    )
    print(execution.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCKER_RELATIVE",
    "CampaignBlockedError",
    "EffectiveConfig",
    "FORBIDDEN_WHILE_BLOCKED",
    "HISTORICAL_ARCHITECTURE_MANIFEST_SHA256",
    "RuntimeRegistration",
    "SUITE_ROOT_RELATIVE",
    "SUITE_SET_ID",
    "SUITE_SPECS",
    "TARGET_OPTIMIZER_STEPS",
    "TRAINING_SEEDS",
    "bind_campaign_execution",
    "build_campaign_artifacts",
    "build_static_artifacts",
    "build_suite_artifacts",
    "campaign_blocker",
    "check_static_artifacts",
    "derive_effective_config",
    "load_campaign_registration",
    "load_runtime_registration",
    "load_suite_set",
    "main",
    "write_static_artifacts",
]
