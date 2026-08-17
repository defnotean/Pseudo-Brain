"""Frozen gates for Reference Candidate Qualification v2.

The development gate is dependency-free and is safe to call from the generic
trainer.  Final-test consumption and receipt handling live in this module too
so the untouched recipient slice has one fail-closed entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import sys
from types import MappingProxyType
from typing import Mapping

from ..training.protocol import TrainingStepResult


DEVELOPMENT_GATE_ID = "rcq_v2_development_v1"
VALUE_DEVELOPMENT_GATE_ID = "rcq_v2_value_development_v1"
DEVELOPMENT_SAMPLES = 1_536
DEVELOPMENT_TARGET_ACTIVE = 2_275
DEVELOPMENT_CHANGED = 467
DEVELOPMENT_PREVIOUS_EXACT = 1_069
_COUNT_TOLERANCE = 1e-4
VALUE_TRAIN_CONDITIONAL_BASELINE_MSE = float.fromhex("0x1.754d5eea85785p-2")
VALUE_ABSOLUTE_MAX_MSE = float.fromhex("0x1.4ff8d56cab52bp-2")
VALUE_DEVELOPMENT_TARGET_VARIANCE = float.fromhex("0x1.be77815f41fbfp-2")
VALUE_IMPROVEMENT_RATIO = float.fromhex("0x1.ccccccccccccdp-1")


class RCQInputError(ValueError):
    """An RCQ artifact is missing, malformed, inconsistent, or already used."""


@dataclass(frozen=True, slots=True)
class RCQCheck:
    name: str
    passed: bool
    observed: int | float
    requirement: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("RCQ check name must be a non-empty string")
        if type(self.passed) is not bool:
            raise ValueError("RCQ check passed must be a boolean")
        if type(self.observed) not in {int, float}:
            raise ValueError("RCQ check observation must be an integer or float")
        if type(self.observed) is float:
            if not isfinite(self.observed):
                raise ValueError("RCQ check observation must be finite")
            if self.observed == 0.0:
                object.__setattr__(self, "observed", 0.0)
        if type(self.requirement) is not str or not self.requirement:
            raise ValueError("RCQ check requirement must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class RCQDevelopmentReport:
    checks: tuple[RCQCheck, ...]
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checks, tuple)
            or not self.checks
            or any(type(check) is not RCQCheck for check in self.checks)
        ):
            raise ValueError("development checks must be a non-empty tuple")
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(_finite_metrics(self.metrics)),
        )

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "gate": DEVELOPMENT_GATE_ID,
            "scope": (
                "fresh open-loop teacher-forced reference-policy development gate; "
                "not closed-loop control or architecture evidence"
            ),
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "metrics": dict(sorted(self.metrics.items())),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class RCQValueDevelopmentReport:
    """Final dev report binding value competence to the passed entry gate."""

    checks: tuple[RCQCheck, ...]
    metrics: Mapping[str, float]
    entry_gate_report_sha256: str
    entry_value_loss: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checks, tuple)
            or not self.checks
            or any(type(check) is not RCQCheck for check in self.checks)
        ):
            raise ValueError("value development checks must be a non-empty tuple")
        metrics = _finite_metrics(self.metrics)
        _sha256_string(
            self.entry_gate_report_sha256,
            name="entry_gate_report_sha256",
        )
        if self.entry_gate_report_sha256 == "0" * 64:
            raise ValueError("entry gate report SHA-256 cannot be null")
        if type(self.entry_value_loss) not in {int, float}:
            raise ValueError("entry value loss must be finite")
        entry_value_loss = float(self.entry_value_loss)
        if not isfinite(entry_value_loss) or entry_value_loss < 0.0:
            raise ValueError("entry value loss must be finite and nonnegative")
        if entry_value_loss == 0.0:
            entry_value_loss = 0.0
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "entry_value_loss", entry_value_loss)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "gate": VALUE_DEVELOPMENT_GATE_ID,
            "scope": (
                "fresh open-loop teacher-forced reference-policy value-development "
                "gate; not closed-loop control or architecture evidence"
            ),
            "passed": self.passed,
            "entry_gate": {
                "gate": DEVELOPMENT_GATE_ID,
                "optimizer_step": 1_536,
                "report_sha256": self.entry_gate_report_sha256,
                "value_loss": self.entry_value_loss,
            },
            "registered_value_baselines": {
                "train_timestep_previous_wasd_dev_mse_hex": (
                    VALUE_TRAIN_CONDITIONAL_BASELINE_MSE.hex()
                ),
                "absolute_max_mse_hex": VALUE_ABSOLUTE_MAX_MSE.hex(),
                "dev_target_variance_hex": (
                    VALUE_DEVELOPMENT_TARGET_VARIANCE.hex()
                ),
                "entry_improvement_ratio_hex": VALUE_IMPROVEMENT_RATIO.hex(),
            },
            "checks": [check.to_dict() for check in self.checks],
            "metrics": dict(sorted(self.metrics.items())),
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _finite_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise RCQInputError("development metrics must be a mapping")
    result: dict[str, float] = {}
    for name, raw in value.items():
        if type(name) is not str or not name:
            raise RCQInputError("development metric names must be non-empty strings")
        if type(raw) not in {int, float}:
            raise RCQInputError(f"development metric {name!r} must be finite")
        try:
            normalized = float(raw)
        except (OverflowError, ValueError) as error:
            raise RCQInputError(
                f"development metric {name!r} must be finite"
            ) from error
        if not isfinite(normalized):
            raise RCQInputError(f"development metric {name!r} must be finite")
        if normalized == 0.0:
            normalized = 0.0
        result[name] = normalized
    return result


def _sha256_string(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RCQInputError(f"{name} must be a lowercase SHA-256 string")
    return value


def _integer_count(metrics: Mapping[str, float], name: str, *, scale: int) -> int:
    try:
        raw = metrics[name]
    except KeyError as error:
        raise RCQInputError(f"development result is missing {name!r}") from error
    scaled = raw * scale
    nearest = round(scaled)
    if abs(scaled - nearest) > _COUNT_TOLERANCE:
        raise RCQInputError(f"development metric {name!r} is off its count lattice")
    return int(nearest)


def evaluate_rcq_v2_development(
    result: TrainingStepResult,
) -> RCQDevelopmentReport:
    """Evaluate the exact fresh 256-sequence development slice at step 1536."""

    if not isinstance(result, TrainingStepResult):
        raise RCQInputError("development gate requires a TrainingStepResult")
    if result.samples != DEVELOPMENT_SAMPLES:
        raise RCQInputError("development gate requires exactly 1,536 decisions")
    metrics = _finite_metrics(result.metrics)
    required = {
        "movement_target_active_count",
        "movement_changed_samples_per_sample",
        "previous_control_movement_exact_match",
        "movement_exact_match",
        "movement_changed_exact_matches_per_sample",
        "positive_key_recall",
        "movement_false_positive_count",
        "movement_opposite_conflict_rate",
        "non_movement_key_false_positive_count",
        "off_support_button_positive_count",
        "off_support_button_target_active_count",
        "continuous_target_nonzero_count",
        "continuous_action_squared_magnitude",
        "continuous_action_outside_0_05_count",
        "total_loss",
        "value_loss",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise RCQInputError(
            "development result is missing required metrics: " + ", ".join(missing)
        )
    for name in (
        "movement_changed_samples_per_sample",
        "previous_control_movement_exact_match",
        "movement_exact_match",
        "movement_changed_exact_matches_per_sample",
        "positive_key_recall",
        "movement_opposite_conflict_rate",
    ):
        if not 0.0 <= metrics[name] <= 1.0:
            raise RCQInputError(f"development metric {name!r} is outside [0, 1]")
    for name, maximum in (
        ("movement_target_active_count", 4.0),
        ("movement_false_positive_count", 4.0),
        ("non_movement_key_false_positive_count", 252.0),
        ("off_support_button_positive_count", 292.0),
        ("off_support_button_target_active_count", 292.0),
        ("continuous_target_nonzero_count", 11.0),
        ("continuous_action_outside_0_05_count", 11.0),
    ):
        if not 0.0 <= metrics[name] <= maximum:
            raise RCQInputError(
                f"development metric {name!r} is outside [0, {maximum}]"
            )
    if metrics["value_loss"] < 0.0:
        raise RCQInputError("development value_loss must be nonnegative")

    counts = {
        name: _integer_count(metrics, name, scale=DEVELOPMENT_SAMPLES)
        for name in (
            "movement_target_active_count",
            "movement_changed_samples_per_sample",
            "previous_control_movement_exact_match",
            "movement_exact_match",
            "movement_changed_exact_matches_per_sample",
            "movement_false_positive_count",
            "movement_opposite_conflict_rate",
            "non_movement_key_false_positive_count",
            "off_support_button_positive_count",
            "off_support_button_target_active_count",
            "continuous_target_nonzero_count",
            "continuous_action_outside_0_05_count",
        )
    }
    if counts["movement_target_active_count"] != DEVELOPMENT_TARGET_ACTIVE:
        raise RCQInputError("development target-active invariant changed")
    if counts["movement_changed_samples_per_sample"] != DEVELOPMENT_CHANGED:
        raise RCQInputError("development changed-decision invariant changed")
    if counts["previous_control_movement_exact_match"] != DEVELOPMENT_PREVIOUS_EXACT:
        raise RCQInputError("development previous-control invariant changed")
    if DEVELOPMENT_CHANGED + DEVELOPMENT_PREVIOUS_EXACT != DEVELOPMENT_SAMPLES:
        raise RCQInputError("registered development invariants are inconsistent")
    exact = counts["movement_exact_match"]
    changed_exact = counts["movement_changed_exact_matches_per_sample"]
    if changed_exact > DEVELOPMENT_CHANGED or changed_exact > exact:
        raise RCQInputError("development changed exact matches are impossible")
    if exact - changed_exact > DEVELOPMENT_PREVIOUS_EXACT:
        raise RCQInputError("development unchanged exact matches are impossible")

    checks = (
        RCQCheck("movement_exact", exact >= 1_229, exact, ">= 1229/1536 (0.80)"),
        RCQCheck(
            "changed_movement_exact",
            changed_exact >= 234,
            changed_exact,
            ">= 234/467 (0.50)",
        ),
        RCQCheck(
            "positive_key_recall",
            metrics["positive_key_recall"] >= 0.90,
            metrics["positive_key_recall"],
            ">= 0.90",
        ),
        RCQCheck(
            "movement_false_positives",
            counts["movement_false_positive_count"] <= 230,
            counts["movement_false_positive_count"],
            "<= 230 keys (0.15 per decision)",
        ),
        RCQCheck(
            "opposite_conflicts",
            counts["movement_opposite_conflict_rate"] <= 7,
            counts["movement_opposite_conflict_rate"],
            "<= 7/1536 (< 0.005)",
        ),
        RCQCheck(
            "nonmovement_keyboard_false_positives",
            counts["non_movement_key_false_positive_count"] == 0,
            counts["non_movement_key_false_positive_count"],
            "= 0",
        ),
        RCQCheck(
            "all_296_buttons_off_support",
            counts["off_support_button_positive_count"] == 0,
            counts["off_support_button_positive_count"],
            "= 0 predicted-active buttons outside W/A/S/D",
        ),
        RCQCheck(
            "off_support_targets_inactive",
            counts["off_support_button_target_active_count"] == 0,
            counts["off_support_button_target_active_count"],
            "= 0 target-active buttons outside W/A/S/D",
        ),
        RCQCheck(
            "continuous_targets_zero",
            counts["continuous_target_nonzero_count"] == 0,
            counts["continuous_target_nonzero_count"],
            "= 0 nonzero continuous targets",
        ),
        RCQCheck(
            "continuous_outputs_in_deadzone",
            counts["continuous_action_outside_0_05_count"] == 0,
            counts["continuous_action_outside_0_05_count"],
            "= 0 of 16,896 continuous outputs with |value| > 0.05",
        ),
    )
    return RCQDevelopmentReport(checks=checks, metrics=metrics)


def evaluate_rcq_v2_value_development(
    result: TrainingStepResult,
    *,
    entry_report: RCQDevelopmentReport,
    entry_report_sha256: str,
) -> RCQValueDevelopmentReport:
    """Gate step 2048 value competence before one-shot TEST authorization."""

    if not isinstance(entry_report, RCQDevelopmentReport):
        raise RCQInputError("value completion requires the entry development report")
    entry_digest = _sha256_string(
        entry_report_sha256,
        name="entry_report_sha256",
    )
    if entry_digest == "0" * 64:
        raise RCQInputError("value completion entry report digest cannot be null")
    try:
        reconstructed_entry = evaluate_rcq_v2_development(
            TrainingStepResult(
                loss=0.0,
                metrics=entry_report.metrics,
                samples=DEVELOPMENT_SAMPLES,
            )
        )
    except (TypeError, ValueError) as error:
        raise RCQInputError("value completion entry report cannot be reconstructed") from error
    if reconstructed_entry.canonical_json != entry_report.canonical_json:
        raise RCQInputError("value completion entry report is internally inconsistent")
    reconstructed_entry_digest = sha256(
        (entry_report.canonical_json + "\n").encode("utf-8")
    ).hexdigest()
    if reconstructed_entry_digest != entry_digest:
        raise RCQInputError(
            "value completion entry report differs from its supplied SHA-256"
        )
    if not entry_report.passed:
        raise RCQInputError("value completion requires a passed entry development gate")

    action_report = evaluate_rcq_v2_development(result)
    entry_value_loss = float(entry_report.metrics["value_loss"])
    final_value_loss = float(action_report.metrics["value_loss"])
    if entry_value_loss < 0.0 or final_value_loss < 0.0:
        raise RCQInputError("value losses must be nonnegative")
    development_r2 = 1.0 - (
        final_value_loss / VALUE_DEVELOPMENT_TARGET_VARIANCE
    )
    if not isfinite(development_r2):
        # A finite but enormous MSE is valid scientific evidence of failure.
        # Keep the receipt strict-JSON serializable instead of converting that
        # failure into an input/serialization error and losing terminal proof.
        development_r2 = -sys.float_info.max
    value_checks = (
        RCQCheck(
            "value_loss_absolute",
            final_value_loss <= VALUE_ABSOLUTE_MAX_MSE,
            final_value_loss,
            f"<= {VALUE_ABSOLUTE_MAX_MSE.hex()} (0.90 x frozen TRAIN-fit baseline)",
        ),
        RCQCheck(
            "value_loss_entry_improvement",
            final_value_loss <= VALUE_IMPROVEMENT_RATIO * entry_value_loss,
            final_value_loss,
            (
                "<= 0x1.ccccccccccccdp-1 x step-1536 entry value_loss "
                f"({entry_value_loss.hex()})"
            ),
        ),
        RCQCheck(
            "value_development_r2",
            development_r2 >= 0.20,
            development_r2,
            (">= 0.20 against frozen development target variance "
             "(redundant with the stronger absolute-MSE check)"),
        ),
    )
    return RCQValueDevelopmentReport(
        checks=(*action_report.checks, *value_checks),
        metrics=action_report.metrics,
        entry_gate_report_sha256=entry_digest,
        entry_value_loss=entry_value_loss,
    )


__all__ = [
    "DEVELOPMENT_CHANGED",
    "DEVELOPMENT_GATE_ID",
    "DEVELOPMENT_PREVIOUS_EXACT",
    "DEVELOPMENT_SAMPLES",
    "DEVELOPMENT_TARGET_ACTIVE",
    "VALUE_ABSOLUTE_MAX_MSE",
    "VALUE_DEVELOPMENT_GATE_ID",
    "VALUE_DEVELOPMENT_TARGET_VARIANCE",
    "VALUE_IMPROVEMENT_RATIO",
    "VALUE_TRAIN_CONDITIONAL_BASELINE_MSE",
    "RCQCheck",
    "RCQDevelopmentReport",
    "RCQInputError",
    "RCQValueDevelopmentReport",
    "evaluate_rcq_v2_development",
    "evaluate_rcq_v2_value_development",
]
