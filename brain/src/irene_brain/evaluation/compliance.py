"""Machine-readable checks for the project's one-brain deployment rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from string import hexdigits
from typing import Any, Mapping

from irene_brain.data.records import PRIVILEGED_STEP_FIELDS
from irene_brain.types import MODEL_OBSERVATION_FIELDS

_GAME_ID_FIELD_NAMES = frozenset(
    {
        "environment_id",
        "env_id",
        "game_id",
        "level_id",
        "rom_id",
        "rom_hash",
        "environment_family",
        "control_mapping_hash",
        "split",
        "world_id",
        "world_lineage_id",
    }
)
_PRIVILEGED_FIELD_NAMES = frozenset(
    {
        "action",
        "applied_action",
        "current_action",
        "current_control",
        "done",
        "enemy_position",
        "events",
        "future",
        "future_events",
        "future_frames",
        "ground_truth",
        "lives",
        "object_positions",
        "player_position",
        "requested_action",
        "ram",
        "reward",
        "rng_state",
        "return",
        "score",
        "seed",
        "simulator_state",
        "state_snapshot",
        "terminated",
        "truncated",
        "world_state",
    }
) | PRIVILEGED_STEP_FIELDS


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list of strings")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise TypeError(f"{name} must contain only strings")
    if any(not item.strip() for item in result):
        raise ValueError(f"{name} cannot contain blank strings")
    return result


def _bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in hexdigits for character in normalized):
        raise ValueError("checkpoint hashes must be 64 hexadecimal SHA-256 characters")
    return normalized


def _field_leaf(value: str) -> str:
    """Normalize ``observation.game_id`` and similar schema paths."""

    return value.strip().lower().replace("-", "_").rsplit(".", 1)[-1]


def _merge_evidence(explicit: tuple[str, ...], inferred: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*explicit, *inferred)))


@dataclass(frozen=True, slots=True)
class DeploymentManifest:
    """Auditable facts about one evaluated deployment.

    ``checkpoint_hashes`` may repeat the same hash for several environments;
    compliance requires exactly one *distinct* checkpoint.  Forbidden features
    are represented as explicit evidence rather than inferred from a model
    name, keeping the audit deterministic and reviewable.
    """

    checkpoint_hashes: tuple[str, ...]
    model_input_fields: tuple[str, ...] = ()
    game_id_inputs: tuple[str, ...] = ()
    per_game_adapters: tuple[str, ...] = ()
    privileged_inputs: tuple[str, ...] = ()
    uses_external_planner: bool = False
    uses_remote_inference_in_deadline_loop: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        hashes = _string_tuple(self.checkpoint_hashes, name="checkpoint_hashes")
        object.__setattr__(self, "checkpoint_hashes", tuple(_sha256(item) for item in hashes))
        for field_name in (
            "model_input_fields",
            "game_id_inputs",
            "per_game_adapters",
            "privileged_inputs",
        ):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "uses_external_planner",
            _bool(self.uses_external_planner, name="uses_external_planner"),
        )
        object.__setattr__(
            self,
            "uses_remote_inference_in_deadline_loop",
            _bool(
                self.uses_remote_inference_in_deadline_loop,
                name="uses_remote_inference_in_deadline_loop",
            ),
        )
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != 1:
            raise ValueError("unsupported deployment manifest schema_version")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeploymentManifest:
        if not isinstance(value, Mapping):
            raise TypeError("deployment manifest must be a mapping")
        known_fields = {
            "checkpoint_hashes",
            "model_input_fields",
            "game_id_inputs",
            "per_game_adapters",
            "privileged_inputs",
            "uses_external_planner",
            "uses_remote_inference_in_deadline_loop",
            "schema_version",
        }
        unknown_fields = sorted(str(item) for item in set(value) - known_fields)
        if unknown_fields:
            joined = ", ".join(unknown_fields)
            raise ValueError(f"unknown deployment manifest fields: {joined}")
        if "checkpoint_hashes" not in value:
            raise ValueError("deployment manifest requires checkpoint_hashes")
        return cls(
            checkpoint_hashes=_string_tuple(
                value["checkpoint_hashes"],
                name="checkpoint_hashes",
            ),
            model_input_fields=_string_tuple(
                value.get("model_input_fields", ()),
                name="model_input_fields",
            ),
            game_id_inputs=_string_tuple(
                value.get("game_id_inputs", ()),
                name="game_id_inputs",
            ),
            per_game_adapters=_string_tuple(
                value.get("per_game_adapters", ()),
                name="per_game_adapters",
            ),
            privileged_inputs=_string_tuple(
                value.get("privileged_inputs", ()),
                name="privileged_inputs",
            ),
            uses_external_planner=_bool(
                value.get("uses_external_planner", False),
                name="uses_external_planner",
            ),
            uses_remote_inference_in_deadline_loop=_bool(
                value.get("uses_remote_inference_in_deadline_loop", False),
                name="uses_remote_inference_in_deadline_loop",
            ),
            schema_version=value.get("schema_version", 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_hashes": list(self.checkpoint_hashes),
            "model_input_fields": list(self.model_input_fields),
            "game_id_inputs": list(self.game_id_inputs),
            "per_game_adapters": list(self.per_game_adapters),
            "privileged_inputs": list(self.privileged_inputs),
            "uses_external_planner": self.uses_external_planner,
            "uses_remote_inference_in_deadline_loop": (
                self.uses_remote_inference_in_deadline_loop
            ),
        }

    def audit(self) -> ComplianceReport:
        return audit_deployment(self)


class ComplianceCode(StrEnum):
    MISSING_CHECKPOINT = "missing_checkpoint"
    MULTIPLE_CHECKPOINTS = "multiple_checkpoints"
    GAME_ID_INPUT = "game_id_input"
    PER_GAME_ADAPTER = "per_game_adapter"
    EXTERNAL_PLANNER = "external_planner"
    PRIVILEGED_INPUT = "privileged_input"
    REMOTE_DEADLINE_INFERENCE = "remote_deadline_inference"


@dataclass(frozen=True, slots=True)
class ComplianceViolation:
    code: ComplianceCode
    message: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    violations: tuple[ComplianceViolation, ...] = field(default_factory=tuple)

    @property
    def compliant(self) -> bool:
        return not self.violations

    def require_compliant(self) -> None:
        if self.violations:
            raise DeploymentComplianceError(self.violations)


class DeploymentComplianceError(ValueError):
    """Raised when a deployment is structurally valid but rule-incompatible."""

    def __init__(self, violations: tuple[ComplianceViolation, ...]) -> None:
        self.violations = violations
        summary = "; ".join(violation.message for violation in violations)
        super().__init__(summary)


def audit_deployment(manifest: DeploymentManifest) -> ComplianceReport:
    """Return all one-brain rule violations in stable rule order."""

    if not isinstance(manifest, DeploymentManifest):
        raise TypeError("manifest must be a DeploymentManifest")

    violations: list[ComplianceViolation] = []
    unique_checkpoints = tuple(dict.fromkeys(manifest.checkpoint_hashes))
    if not unique_checkpoints:
        violations.append(
            ComplianceViolation(
                ComplianceCode.MISSING_CHECKPOINT,
                "deployment must declare one checkpoint",
            )
        )
    elif len(unique_checkpoints) > 1:
        violations.append(
            ComplianceViolation(
                ComplianceCode.MULTIPLE_CHECKPOINTS,
                "all evaluation environments must use one checkpoint",
                unique_checkpoints,
            )
        )

    inferred_game_ids = tuple(
        field_name
        for field_name in manifest.model_input_fields
        if _field_leaf(field_name) in _GAME_ID_FIELD_NAMES
    )
    game_id_evidence = _merge_evidence(manifest.game_id_inputs, inferred_game_ids)
    if game_id_evidence:
        violations.append(
            ComplianceViolation(
                ComplianceCode.GAME_ID_INPUT,
                "game identity cannot be supplied to the model",
                game_id_evidence,
            )
        )
    if manifest.per_game_adapters:
        violations.append(
            ComplianceViolation(
                ComplianceCode.PER_GAME_ADAPTER,
                "per-game adapters are not part of a one-brain deployment",
                manifest.per_game_adapters,
            )
        )
    if manifest.uses_external_planner:
        violations.append(
            ComplianceViolation(
                ComplianceCode.EXTERNAL_PLANNER,
                "the deployed model cannot delegate decisions to an external planner",
            )
        )
    noncanonical_inputs = tuple(
        field_name
        for field_name in manifest.model_input_fields
        if _field_leaf(field_name) not in MODEL_OBSERVATION_FIELDS
    )
    inferred_privileged_inputs = tuple(
        field_name
        for field_name in manifest.model_input_fields
        if _field_leaf(field_name) in _PRIVILEGED_FIELD_NAMES
    )
    privileged_evidence = _merge_evidence(
        manifest.privileged_inputs,
        _merge_evidence(inferred_privileged_inputs, noncanonical_inputs),
    )
    if privileged_evidence:
        violations.append(
            ComplianceViolation(
                ComplianceCode.PRIVILEGED_INPUT,
                "model inputs must use only canonical ModelObservation fields",
                privileged_evidence,
            )
        )
    if manifest.uses_remote_inference_in_deadline_loop:
        violations.append(
            ComplianceViolation(
                ComplianceCode.REMOTE_DEADLINE_INFERENCE,
                "deadline-bound inference must execute locally",
            )
        )
    return ComplianceReport(tuple(violations))


def check_compliance(manifest: DeploymentManifest) -> tuple[ComplianceViolation, ...]:
    """Convenience API returning only the immutable violation sequence."""

    return audit_deployment(manifest).violations


def require_compliant(manifest: DeploymentManifest) -> DeploymentManifest:
    """Raise on violations and otherwise return the validated manifest."""

    audit_deployment(manifest).require_compliant()
    return manifest


__all__ = [
    "ComplianceCode",
    "ComplianceReport",
    "ComplianceViolation",
    "DeploymentComplianceError",
    "DeploymentManifest",
    "audit_deployment",
    "check_compliance",
    "require_compliant",
]
