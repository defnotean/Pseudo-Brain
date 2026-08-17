"""Fail-closed evaluator for the broad Stage A continuation gate.

This gate checks whether a 500-step procedural MovingShapes run is worth
continuing.  It is deliberately not a causal thought-use certification: rank,
diversity, and actuator-attention breadth can stay healthy even when thought
slots do not make distinct causal contributions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping, Sequence


GATE_NAME = "stagea_continuation_v1"
GATE_SCOPE = (
    "broad final-exit continuation screen; not causal thought-use or anytime-exit "
    "certification"
)
EXPECTED_TRAIN_STEPS = (1, 100, 200, 300, 400, 500)
EXPECTED_VALIDATION_STEPS = (100, 200, 300, 400, 500)
EXPECTED_RECORD_ORDER = tuple(
    [("train", 1)]
    + [
        item
        for step in EXPECTED_VALIDATION_STEPS
        for item in (("train", step), ("validation", step))
    ]
)
FINAL_STEP = 500
FINAL_VALIDATION_SAMPLES = 96
_COUNT_TOLERANCE = 1e-4
_INVARIANT_COUNTS = MappingProxyType(
    {
        "movement_target_active_count": 137,
        "movement_changed_samples_per_sample": 26,
        "previous_control_movement_exact_match": 70,
    }
)
_FINAL_REQUIRED_METRICS = frozenset(
    {
        "samples",
        "movement_target_active_count",
        "movement_changed_samples_per_sample",
        "previous_control_movement_exact_match",
        "movement_exact_match",
        "movement_changed_exact_matches_per_sample",
        "positive_key_recall",
        "movement_false_positive_count",
        "movement_predicted_active_count",
        "movement_opposite_conflict_rate",
        "non_movement_key_false_positive_count",
        "thought_summary_rank_proxy",
        "thought_full_register_rank_proxy",
        "movement_query_effective_slot_count",
        "diversity_loss",
        "value_loss",
        "world_loss",
    }
)


class StageAGateInputError(ValueError):
    """The metrics artifact cannot safely be interpreted as this gate run."""


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    observed: int | float
    requirement: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class StageAContinuationGateReport:
    checks: tuple[GateCheck, ...]
    world_sanity: Mapping[str, float]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "gate": GATE_NAME,
            "scope": GATE_SCOPE,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "world_sanity": {
                **dict(self.world_sanity),
                "interpretation": (
                    "diagnostic loss comparison only; not evidence for multiple world models"
                ),
            },
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
class _MetricRecord:
    step: int
    epoch: int
    split: str
    metrics: Mapping[str, float]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StageAGateInputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> object:
    raise StageAGateInputError(f"non-finite JSON number is forbidden: {value}")


def _parse_record(line: str, *, line_number: int) -> _MetricRecord:
    try:
        raw = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except StageAGateInputError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise StageAGateInputError(
            f"line {line_number} is not strict JSON: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise StageAGateInputError(f"line {line_number} must contain a JSON object")
    expected_fields = {"schema_version", "step", "epoch", "split", "metrics"}
    if set(raw) != expected_fields:
        missing = sorted(expected_fields - set(raw))
        unknown = sorted(set(raw) - expected_fields)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise StageAGateInputError(
            f"line {line_number} has the wrong record schema ({'; '.join(details)})"
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise StageAGateInputError(f"line {line_number} schema_version must be integer 1")
    step = raw["step"]
    epoch = raw["epoch"]
    split = raw["split"]
    metrics = raw["metrics"]
    if type(step) is not int or step < 0:
        raise StageAGateInputError(f"line {line_number} step must be a nonnegative integer")
    if type(epoch) is not int or epoch != 0:
        raise StageAGateInputError(
            f"line {line_number} epoch must be integer 0 for this 500-step recipe"
        )
    if type(split) is not str or split not in {"train", "validation"}:
        raise StageAGateInputError(
            f"line {line_number} split must be train or validation"
        )
    if not isinstance(metrics, dict) or not metrics:
        raise StageAGateInputError(f"line {line_number} metrics must be a non-empty object")
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if type(name) is not str or not name:
            raise StageAGateInputError(
                f"line {line_number} metric names must be non-empty strings"
            )
        if type(value) not in {int, float}:
            raise StageAGateInputError(
                f"line {line_number} metric {name!r} must be a finite number"
            )
        try:
            normalized_value = float(value)
        except (OverflowError, ValueError) as error:
            raise StageAGateInputError(
                f"line {line_number} metric {name!r} must be a finite number"
            ) from error
        if not isfinite(normalized_value):
            raise StageAGateInputError(
                f"line {line_number} metric {name!r} must be a finite number"
            )
        normalized[name] = normalized_value
    return _MetricRecord(
        step=step,
        epoch=epoch,
        split=split,
        metrics=MappingProxyType(normalized),
    )


def load_stagea_continuation_metrics(path: str | Path) -> tuple[_MetricRecord, ...]:
    """Read and strictly validate the exact metrics schedule without writing."""

    source = Path(path)
    try:
        encoded = source.read_bytes()
    except OSError as error:
        raise StageAGateInputError(f"cannot read metrics artifact: {error}") from error
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise StageAGateInputError("metrics artifact must be UTF-8") from error
    lines = text.splitlines()
    if not lines:
        raise StageAGateInputError("metrics artifact is empty")
    if any(not line.strip() for line in lines):
        raise StageAGateInputError("metrics artifact contains a blank record")
    records = tuple(
        _parse_record(line, line_number=index)
        for index, line in enumerate(lines, start=1)
    )
    observed_order = tuple((record.split, record.step) for record in records)
    if len(set(observed_order)) != len(observed_order):
        raise StageAGateInputError("metrics artifact contains a duplicate split/step record")
    if observed_order != EXPECTED_RECORD_ORDER:
        raise StageAGateInputError(
            "metrics schedule must be exactly "
            + ", ".join(f"{split}:{step}" for split, step in EXPECTED_RECORD_ORDER)
        )
    return records


def _required_metric(metrics: Mapping[str, float], name: str) -> float:
    try:
        return metrics[name]
    except KeyError as error:
        raise StageAGateInputError(f"final validation is missing metric {name!r}") from error


def _require_range(
    metrics: Mapping[str, float],
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> None:
    value = _required_metric(metrics, name)
    if not minimum <= value <= maximum:
        raise StageAGateInputError(
            f"final validation metric {name!r} is outside its valid range "
            f"[{minimum}, {maximum}]: {value}"
        )


def _count(value: float) -> float:
    return value * FINAL_VALIDATION_SAMPLES


def _count_equal(observed_per_sample: float, expected: int) -> bool:
    return abs(_count(observed_per_sample) - expected) <= _COUNT_TOLERANCE


def _integer_lattice(value: float, *, scale: int, name: str) -> int:
    scaled = value * scale
    nearest = round(scaled)
    if abs(scaled - nearest) > _COUNT_TOLERANCE:
        raise StageAGateInputError(
            f"final validation metric {name!r} is off its 1/{scale} count lattice: "
            f"{value}"
        )
    return int(nearest)


def evaluate_stagea_continuation_gate(
    path: str | Path,
) -> StageAContinuationGateReport:
    """Evaluate a completed 500-step run against the predeclared gate."""

    records = load_stagea_continuation_metrics(path)
    by_key = {(record.split, record.step): record for record in records}
    final = by_key[("validation", FINAL_STEP)].metrics
    missing = sorted(_FINAL_REQUIRED_METRICS - set(final))
    if missing:
        raise StageAGateInputError(
            "final validation is missing required metrics: " + ", ".join(missing)
        )

    for name in (
        "movement_exact_match",
        "movement_changed_samples_per_sample",
        "movement_changed_exact_matches_per_sample",
        "positive_key_recall",
        "movement_opposite_conflict_rate",
        "previous_control_movement_exact_match",
        "thought_summary_rank_proxy",
        "thought_full_register_rank_proxy",
        "diversity_loss",
    ):
        _require_range(final, name, minimum=0.0, maximum=1.0)
    for name, maximum in (
        ("movement_target_active_count", 4.0),
        ("movement_predicted_active_count", 4.0),
        ("movement_false_positive_count", 4.0),
        ("non_movement_key_false_positive_count", 252.0),
        ("movement_query_effective_slot_count", 32.0),
    ):
        _require_range(final, name, minimum=0.0, maximum=maximum)
    for name in ("value_loss", "world_loss"):
        _require_range(final, name, minimum=0.0, maximum=float("inf"))

    sample_count = final["samples"]
    target_active = final["movement_target_active_count"]
    changed_samples = final["movement_changed_samples_per_sample"]
    previous_exact = final["previous_control_movement_exact_match"]
    exact = final["movement_exact_match"]
    changed_exact = final["movement_changed_exact_matches_per_sample"]
    false_positives = final["movement_false_positive_count"]
    predicted_active = final["movement_predicted_active_count"]
    conflicts = final["movement_opposite_conflict_rate"]
    nonmovement_false_positives = final["non_movement_key_false_positive_count"]

    count_metrics = {
        name: _integer_lattice(final[name], scale=FINAL_VALIDATION_SAMPLES, name=name)
        for name in (
            "movement_target_active_count",
            "movement_changed_samples_per_sample",
            "previous_control_movement_exact_match",
            "movement_exact_match",
            "movement_changed_exact_matches_per_sample",
            "movement_false_positive_count",
            "movement_predicted_active_count",
            "movement_opposite_conflict_rate",
            "non_movement_key_false_positive_count",
        )
    }
    recall_half_count = _integer_lattice(
        final["positive_key_recall"],
        scale=2 * FINAL_VALIDATION_SAMPLES,
        name="positive_key_recall",
    )
    target_count = count_metrics["movement_target_active_count"]
    changed_count = count_metrics["movement_changed_samples_per_sample"]
    previous_count = count_metrics["previous_control_movement_exact_match"]
    exact_count = count_metrics["movement_exact_match"]
    changed_exact_count = count_metrics["movement_changed_exact_matches_per_sample"]
    false_positive_count = count_metrics["movement_false_positive_count"]
    predicted_count = count_metrics["movement_predicted_active_count"]
    if changed_count + previous_count != FINAL_VALIDATION_SAMPLES:
        raise StageAGateInputError(
            "changed-sample and previous-control-exact counts must sum to 96"
        )
    if changed_exact_count > changed_count or changed_exact_count > exact_count:
        raise StageAGateInputError(
            "changed exact matches cannot exceed changed samples or all exact matches"
        )
    if exact_count - changed_exact_count > previous_count:
        raise StageAGateInputError(
            "exact unchanged decisions cannot exceed the unchanged-sample count"
        )
    if predicted_count < false_positive_count:
        raise StageAGateInputError(
            "predicted movement keys cannot be fewer than movement false positives"
        )
    true_positive_count = predicted_count - false_positive_count
    if true_positive_count > target_count:
        raise StageAGateInputError(
            "implied movement true positives cannot exceed target movement keys"
        )
    false_negative_count = target_count - true_positive_count
    nonexact_count = FINAL_VALIDATION_SAMPLES - exact_count
    conflict_count = count_metrics["movement_opposite_conflict_rate"]
    if recall_half_count < (2 * exact_count):
        raise StageAGateInputError(
            "positive-key recall cannot be lower than the exact-match contribution"
        )
    if false_positive_count + false_negative_count < nonexact_count:
        raise StageAGateInputError(
            "movement errors cannot account for every nonexact decision"
        )
    if false_positive_count + false_negative_count > (4 * nonexact_count):
        raise StageAGateInputError(
            "movement errors exceed the four-key capacity of nonexact decisions"
        )
    if false_positive_count > (3 * nonexact_count):
        raise StageAGateInputError(
            "movement false positives exceed the capacity of nonexact decisions"
        )
    if false_negative_count > (2 * nonexact_count):
        raise StageAGateInputError(
            "movement false negatives exceed the capacity of nonexact decisions"
        )
    if conflict_count > nonexact_count or conflict_count > false_positive_count:
        raise StageAGateInputError(
            "opposite-direction conflicts must be nonexact and include a false positive"
        )
    if (
        sample_count == float(FINAL_VALIDATION_SAMPLES)
        and target_count == _INVARIANT_COUNTS["movement_target_active_count"]
    ):
        two_key_samples = target_count - FINAL_VALIDATION_SAMPLES
        one_key_samples = FINAL_VALIDATION_SAMPLES - two_key_samples
        one_key_true_positives = recall_half_count - true_positive_count
        two_key_true_positives = (2 * true_positive_count) - recall_half_count
        if not 0 <= one_key_true_positives <= one_key_samples:
            raise StageAGateInputError(
                "positive-key recall is inconsistent with one-key true positives"
            )
        if not 0 <= two_key_true_positives <= (2 * two_key_samples):
            raise StageAGateInputError(
                "positive-key recall is inconsistent with two-key true positives"
            )
    expected_invariants = (
        (sample_count, float(FINAL_VALIDATION_SAMPLES), "samples must equal 96"),
        (target_count, 137, "target active count must equal 137"),
        (changed_count, 26, "changed-sample count must equal 26"),
        (previous_count, 70, "previous-control exact count must equal 70"),
    )
    for observed, expected, message in expected_invariants:
        if observed != expected:
            raise StageAGateInputError(
                f"final validation slice invariant failed: {message}; observed {observed}"
            )

    checks = (
        GateCheck(
            "validation_samples",
            sample_count == float(FINAL_VALIDATION_SAMPLES),
            sample_count,
            "= 96",
        ),
        GateCheck(
            "target_active_count",
            _count_equal(target_active, _INVARIANT_COUNTS["movement_target_active_count"]),
            _count(target_active),
            "= 137 across 96 decisions",
        ),
        GateCheck(
            "changed_sample_count",
            _count_equal(
                changed_samples,
                _INVARIANT_COUNTS["movement_changed_samples_per_sample"],
            ),
            _count(changed_samples),
            "= 26 across 96 decisions",
        ),
        GateCheck(
            "previous_control_exact_count",
            _count_equal(
                previous_exact,
                _INVARIANT_COUNTS["previous_control_movement_exact_match"],
            ),
            _count(previous_exact),
            "= 70 across 96 decisions",
        ),
        GateCheck(
            "movement_exact_count",
            _count(exact) + _COUNT_TOLERANCE >= 80.0,
            _count(exact),
            ">= 80 across 96 decisions",
        ),
        GateCheck(
            "changed_movement_exact_count",
            _count(changed_exact) + _COUNT_TOLERANCE >= 13.0,
            _count(changed_exact),
            ">= 13 of the 26 changed decisions",
        ),
        GateCheck(
            "positive_key_recall",
            final["positive_key_recall"] >= 0.90,
            final["positive_key_recall"],
            ">= 0.90",
        ),
        GateCheck(
            "movement_false_positive_count",
            _count(false_positives) <= 12.0 + _COUNT_TOLERANCE,
            _count(false_positives),
            "<= 12 across 96 decisions",
        ),
        GateCheck(
            "predicted_active_count_per_decision",
            abs(predicted_active - (137.0 / 96.0)) <= 0.15,
            predicted_active,
            "within 0.15 of 137/96",
        ),
        GateCheck(
            "opposite_direction_conflicts",
            conflicts == 0.0,
            _count(conflicts),
            "= 0 across 96 decisions",
        ),
        GateCheck(
            "nonmovement_key_false_positives",
            nonmovement_false_positives == 0.0,
            _count(nonmovement_false_positives),
            "= 0 across 96 decisions",
        ),
        GateCheck(
            "thought_summary_normalized_rank",
            final["thought_summary_rank_proxy"] >= 0.125,
            final["thought_summary_rank_proxy"],
            ">= 0.125",
        ),
        GateCheck(
            "thought_register_normalized_rank",
            final["thought_full_register_rank_proxy"] >= 0.125,
            final["thought_full_register_rank_proxy"],
            ">= 0.125",
        ),
        GateCheck(
            "movement_query_effective_slots",
            final["movement_query_effective_slot_count"] >= 4.0,
            final["movement_query_effective_slot_count"],
            ">= 4",
        ),
        GateCheck(
            "thought_diversity_loss",
            final["diversity_loss"] <= (7.0 / 31.0),
            final["diversity_loss"],
            "<= 7/31",
        ),
        GateCheck(
            "value_loss",
            final["value_loss"] < 0.35806523,
            final["value_loss"],
            "< 0.35806523",
        ),
    )

    train_first = by_key[("train", 1)].metrics
    train_final = by_key[("train", FINAL_STEP)].metrics
    for label, metrics in (("train step 1", train_first), ("train step 500", train_final)):
        value = metrics.get("world_loss")
        if value is None:
            raise StageAGateInputError(f"{label} is missing metric 'world_loss'")
        if not isfinite(value) or value < 0.0:
            raise StageAGateInputError(f"{label} world_loss must be finite and nonnegative")
    world_sanity = MappingProxyType(
        {
            "train_step_1_world_loss": train_first["world_loss"],
            "train_step_500_world_loss": train_final["world_loss"],
            "validation_step_500_world_loss": final["world_loss"],
            "train_change_step_500_minus_step_1": (
                train_final["world_loss"] - train_first["world_loss"]
            ),
            "validation_minus_final_train": (
                final["world_loss"] - train_final["world_loss"]
            ),
        }
    )
    world_checks = (
        GateCheck(
            "world_validation_beats_initial_train",
            final["world_loss"] < train_first["world_loss"],
            final["world_loss"],
            "< train step-1 world_loss",
        ),
        GateCheck(
            "world_validation_tracks_final_train",
            final["world_loss"] <= (1.5 * train_final["world_loss"]),
            final["world_loss"],
            "<= 1.5 * train step-500 world_loss",
        ),
    )
    return StageAContinuationGateReport(
        checks=(*checks, *world_checks),
        world_sanity=world_sanity,
    )


def _error_document(message: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "gate": GATE_NAME,
        "scope": GATE_SCOPE,
        "passed": False,
        "input_error": message,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one completed 500-step Stage A continuation metrics artifact."
        ),
        epilog=(
            "Passing authorizes only the next research stage; it does not certify "
            "causal use of thought slots."
        ),
    )
    parser.add_argument("metrics", type=Path, help="read-only metrics.jsonl path")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent JSON output for human inspection",
    )
    args = parser.parse_args(argv)
    indent = 2 if args.pretty else None
    try:
        report = evaluate_stagea_continuation_gate(args.metrics)
    except StageAGateInputError as error:
        print(
            json.dumps(
                _error_document(str(error)),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
            )
        )
        return 2
    print(
        json.dumps(
            report.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper/CLI.
    sys.exit(main())


__all__ = [
    "EXPECTED_RECORD_ORDER",
    "GATE_NAME",
    "GATE_SCOPE",
    "GateCheck",
    "StageAContinuationGateReport",
    "StageAGateInputError",
    "evaluate_stagea_continuation_gate",
    "load_stagea_continuation_metrics",
    "main",
]
