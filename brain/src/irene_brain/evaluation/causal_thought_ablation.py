"""Deterministic final-readout interventions for Stage-A thought slots.

This evaluator asks one deliberately narrow question: on the fixed Stage-A
validation slice, do the final movement decisions beneficially depend on the
sample-specific thought tensor presented to the final actuator readout?

The recurrent trajectory is always advanced with the unmodified model.  A
temporary pre-hook captures the exact final actuator inputs from that full
pass.  The actuator is then replayed with either zero thoughts or a
time-aligned, cyclically deranged donor thought tensor from another validation
sequence.  Sensors, belief, working memory, retrieved memory, goal context,
and all future recurrent state therefore remain those of the full pass.

The module intentionally uses absolute package imports inside functions.  It
can consequently be executed as a standalone file while ``PYTHONPATH`` points
at an older immutable training release that did not yet contain this evaluator.
That arrangement keeps every imported model/data/checkpoint implementation
covered by the checkpoint's original source-tree hash.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import random
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_SEQUENCES = 16
EXPECTED_SEQUENCE_LENGTH = 8
EXPECTED_BURN_IN_STEPS = 2
EXPECTED_SAMPLES = 96
EXPECTED_TARGET_ACTIVE = 137
EXPECTED_CHANGED_SAMPLES = 26
EXPECTED_PREVIOUS_EXACT = 70
EXPECTED_CHECKPOINT_STEP = 500
EXPECTED_CHECKPOINT_EPOCH = 0
EXPECTED_CHECKPOINT_NEXT_BATCH = 4000
EXPECTED_DONOR_TARGET_OVERLAP = 15
EXPECTED_CHANGED_DONOR_TARGET_OVERLAP = 3
REGISTERED_A_RECIPE_ID = "stagea_continuation_cosine_a_schema1"
REGISTERED_A_CONFIG_SHA256 = (
    "d6f8c8ba3caaaab4643c430ff75747c0025f866363ddfbbb2114fc80969c8fcd"
)
REGISTERED_A_CONFIG_FILE_SHA256 = (
    "47ef30ba199f0b83b5719caa938e3eb77dfd1695dd5e10cda615ab503dc4a162"
)
REGISTERED_B_RECIPE_ID = "stagea_continuation_constant_lr_b_schema2"
REGISTERED_B_CONFIG_SHA256 = (
    "8e8e4cc12123f55ee22aace1d65517dbb7586465bbf704c77cc7fe45bb3fc1f9"
)
REGISTERED_B_CONFIG_FILE_SHA256 = (
    "f013e3e14dfb854da1131aa726dd141bbe22e5766db8a48ed7cdb1aff6b9a6bf"
)
REGISTERED_DATA_SHA256 = (
    "9dbc93218e26e5b522c0d1f2bddd9172499bf66ebdc2855ef51570fce26de70e"
)
# Compatibility aliases retain the original schema-1 A API and exact values.
REGISTERED_CONFIG_SHA256 = REGISTERED_A_CONFIG_SHA256
REGISTERED_CONFIG_FILE_SHA256 = REGISTERED_A_CONFIG_FILE_SHA256
REGISTERED_VALIDATION_MANIFEST_SHA256 = (
    "a9cdc763e85e9bb13d21fb2ecf1098af11912d51bc3c8b9801f03d125347c1e5"
)
MIN_FULL_MOVEMENT_EXACT = 80
MIN_FULL_CHANGED_EXACT = 13
MIN_OVERALL_EXACT_DROP = 5
MIN_CHANGED_EXACT_DROP = 3
MOVEMENT_CONTROL_INDICES = (4, 7, 22, 26)
INTERVENTIONS = ("zeroed", "batch_shuffled")
SCOPE = (
    "conditional causal contribution of the complete final thought tensor to "
    "final W/A/S/D decisions on one fixed 96-decision validation slice"
)
LIMITATIONS = (
    "This isolates the final actuator readout; it does not measure recurrent or "
    "long-horizon effects of thoughts.",
    "It tests the complete thought tensor, not whether individual slots are "
    "specialized, independent, or each causally necessary.",
    "The fixed deterministic slice is an engineering gate, not a statistical "
    "generalization estimate.",
    "The registered 5/96 and 3/26 effect sizes are not significance tests; the "
    "six decisions within each trajectory are correlated and this validation "
    "slice was visible to training-time model selection.",
    "The result is not evidence of human-like thought or multiple world models.",
)


class CausalThoughtEvaluationInputError(ValueError):
    """The requested checkpoint evaluation cannot be interpreted safely."""


def _sha256_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CausalThoughtEvaluationInputError(
            f"{name} must be 64 lowercase hexadecimal SHA-256 characters"
        )
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise CausalThoughtEvaluationInputError(f"{name} must be a positive integer")
    return value


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _deep_plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RegisteredRecipe:
    recipe_id: str
    config_filename: str
    config_schema_version: int
    run_name: str
    scheduler_kind: str
    config_sha256: str
    config_file_sha256: str
    data_sha256: str
    optimizer_step: int = EXPECTED_CHECKPOINT_STEP
    epoch: int = EXPECTED_CHECKPOINT_EPOCH
    next_batch: int = EXPECTED_CHECKPOINT_NEXT_BATCH

    def __post_init__(self) -> None:
        for name in (
            "recipe_id",
            "config_filename",
            "run_name",
            "scheduler_kind",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"registered recipe {name} must be a non-empty string")
        if self.config_schema_version not in {1, 2}:
            raise ValueError("registered recipe schema must be 1 or 2")
        for name in ("config_sha256", "config_file_sha256", "data_sha256"):
            _sha256_string(getattr(self, name), name=f"registered_recipe.{name}")
        if (
            self.optimizer_step,
            self.epoch,
            self.next_batch,
        ) != (
            EXPECTED_CHECKPOINT_STEP,
            EXPECTED_CHECKPOINT_EPOCH,
            EXPECTED_CHECKPOINT_NEXT_BATCH,
        ):
            raise ValueError("registered recipe has an unsupported cursor boundary")

    def to_dict(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "config_filename": self.config_filename,
            "config_schema_version": self.config_schema_version,
            "run_name": self.run_name,
            "scheduler_kind": self.scheduler_kind,
            "config_sha256": self.config_sha256,
            "config_file_sha256": self.config_file_sha256,
            "data_sha256": self.data_sha256,
            "optimizer_step": self.optimizer_step,
            "epoch": self.epoch,
            "next_batch": self.next_batch,
        }


REGISTERED_RECIPES: Mapping[str, RegisteredRecipe] = MappingProxyType(
    {
        REGISTERED_A_RECIPE_ID: RegisteredRecipe(
            recipe_id=REGISTERED_A_RECIPE_ID,
            config_filename="dgx-stagea-continuation-gate.toml",
            config_schema_version=1,
            run_name="dgx-stagea-continuation-gate",
            scheduler_kind="cosine_after_warmup",
            config_sha256=REGISTERED_A_CONFIG_SHA256,
            config_file_sha256=REGISTERED_A_CONFIG_FILE_SHA256,
            data_sha256=REGISTERED_DATA_SHA256,
        ),
        REGISTERED_B_RECIPE_ID: RegisteredRecipe(
            recipe_id=REGISTERED_B_RECIPE_ID,
            config_filename="dgx-stagea-continuation-gate-b.toml",
            config_schema_version=2,
            run_name="dgx-stagea-continuation-gate-b",
            scheduler_kind="constant_after_warmup",
            config_sha256=REGISTERED_B_CONFIG_SHA256,
            config_file_sha256=REGISTERED_B_CONFIG_FILE_SHA256,
            data_sha256=REGISTERED_DATA_SHA256,
        ),
    }
)


def _registered_recipe(
    *,
    config_sha256: str,
    config_file_sha256: str,
    data_sha256: str,
) -> RegisteredRecipe:
    matches = tuple(
        recipe
        for recipe in REGISTERED_RECIPES.values()
        if (
            recipe.config_sha256,
            recipe.config_file_sha256,
            recipe.data_sha256,
        )
        == (config_sha256, config_file_sha256, data_sha256)
    )
    if len(matches) != 1:
        raise CausalThoughtEvaluationInputError(
            "config, raw TOML, and data hashes do not identify exactly one "
            "registered Stage-A A/B recipe"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    checkpoint_sha256: str
    config_sha256: str
    config_file_sha256: str
    data_sha256: str
    training_code_sha256: str
    evaluator_sha256: str
    optimizer_step: int
    epoch: int
    next_batch: int
    runtime_fingerprint: Mapping[str, str | int | bool]
    training_source_root: str

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "config_sha256",
            "config_file_sha256",
            "data_sha256",
            "training_code_sha256",
            "evaluator_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256_string(getattr(self, name), name=name),
            )
        recipe = _registered_recipe(
            config_sha256=self.config_sha256,
            config_file_sha256=self.config_file_sha256,
            data_sha256=self.data_sha256,
        )
        for name in ("optimizer_step", "epoch", "next_batch"):
            if type(getattr(self, name)) is not int:
                raise CausalThoughtEvaluationInputError(f"{name} must be an integer")
        if self.optimizer_step != recipe.optimizer_step:
            raise CausalThoughtEvaluationInputError(
                f"optimizer_step must be the registered boundary {recipe.optimizer_step}"
            )
        if self.epoch != recipe.epoch:
            raise CausalThoughtEvaluationInputError(
                f"epoch must be {recipe.epoch} at the registered boundary"
            )
        if self.next_batch != recipe.next_batch:
            raise CausalThoughtEvaluationInputError(
                f"next_batch must be {recipe.next_batch} at the registered boundary"
            )
        if not isinstance(self.runtime_fingerprint, Mapping) or not self.runtime_fingerprint:
            raise CausalThoughtEvaluationInputError(
                "runtime_fingerprint must be a non-empty mapping"
            )
        normalized: dict[str, str | int | bool] = {}
        for key, value in self.runtime_fingerprint.items():
            if type(key) is not str or not key:
                raise CausalThoughtEvaluationInputError(
                    "runtime fingerprint names must be non-empty strings"
                )
            if type(value) not in {str, int, bool}:
                raise CausalThoughtEvaluationInputError(
                    "runtime fingerprint values must be strings, integers, or booleans"
                )
            normalized[key] = value
        object.__setattr__(
            self,
            "runtime_fingerprint",
            MappingProxyType(dict(sorted(normalized.items()))),
        )
        if not isinstance(self.training_source_root, str) or not self.training_source_root:
            raise CausalThoughtEvaluationInputError(
                "training_source_root must be a non-empty path string"
            )

    def to_dict(self) -> dict[str, object]:
        recipe = self.registered_recipe
        return {
            "registered_recipe": recipe.recipe_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "config_file_sha256": self.config_file_sha256,
            "data_sha256": self.data_sha256,
            "training_code_sha256": self.training_code_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "optimizer_step": self.optimizer_step,
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "runtime_fingerprint": dict(self.runtime_fingerprint),
            "training_source_root": self.training_source_root,
        }

    @property
    def registered_recipe(self) -> RegisteredRecipe:
        return _registered_recipe(
            config_sha256=self.config_sha256,
            config_file_sha256=self.config_file_sha256,
            data_sha256=self.data_sha256,
        )


@dataclass(frozen=True, slots=True)
class DecisionCounts:
    movement_exact: int
    changed_movement_exact: int

    def __post_init__(self) -> None:
        for name in ("movement_exact", "changed_movement_exact"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CausalThoughtEvaluationInputError(
                    f"{name} must be a nonnegative integer"
                )
        if self.movement_exact > EXPECTED_SAMPLES:
            raise CausalThoughtEvaluationInputError(
                "movement_exact exceeds the fixed sample count"
            )
        if self.changed_movement_exact > EXPECTED_CHANGED_SAMPLES:
            raise CausalThoughtEvaluationInputError(
                "changed_movement_exact exceeds the fixed changed-sample count"
            )
        if self.changed_movement_exact > self.movement_exact:
            raise CausalThoughtEvaluationInputError(
                "changed_movement_exact cannot exceed movement_exact"
            )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "movement_exact_count": self.movement_exact,
            "movement_exact_rate": self.movement_exact / EXPECTED_SAMPLES,
            "changed_movement_exact_count": self.changed_movement_exact,
            "changed_movement_exact_rate": (
                self.changed_movement_exact / EXPECTED_CHANGED_SAMPLES
            ),
        }


@dataclass(frozen=True, slots=True)
class PairedCorrectnessCounts:
    full_correct_intervention_wrong: int
    full_wrong_intervention_correct: int
    changed_full_correct_intervention_wrong: int
    changed_full_wrong_intervention_correct: int

    def __post_init__(self) -> None:
        for name in (
            "full_correct_intervention_wrong",
            "full_wrong_intervention_correct",
            "changed_full_correct_intervention_wrong",
            "changed_full_wrong_intervention_correct",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= EXPECTED_SAMPLES:
                raise CausalThoughtEvaluationInputError(
                    f"{name} is outside the decision-count lattice"
                )
        if (
            self.changed_full_correct_intervention_wrong
            > self.full_correct_intervention_wrong
            or self.changed_full_wrong_intervention_correct
            > self.full_wrong_intervention_correct
        ):
            raise CausalThoughtEvaluationInputError(
                "changed paired counts cannot exceed their all-sample counts"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "full_correct_intervention_wrong": (
                self.full_correct_intervention_wrong
            ),
            "full_wrong_intervention_correct": (
                self.full_wrong_intervention_correct
            ),
            "changed_full_correct_intervention_wrong": (
                self.changed_full_correct_intervention_wrong
            ),
            "changed_full_wrong_intervention_correct": (
                self.changed_full_wrong_intervention_correct
            ),
        }


def _preregistered_checks(
    conditions: Mapping[str, DecisionCounts],
) -> tuple[dict[str, object], ...]:
    if not isinstance(conditions, Mapping) or set(conditions) != {
        "full",
        *INTERVENTIONS,
    } or any(not isinstance(value, DecisionCounts) for value in conditions.values()):
        raise CausalThoughtEvaluationInputError(
            "gate checks require full, zeroed, and batch_shuffled DecisionCounts"
        )
    full = conditions["full"]
    result: list[dict[str, object]] = [
        {
            "name": "full_movement_exact_count",
            "actual_count": full.movement_exact,
            "minimum_count": MIN_FULL_MOVEMENT_EXACT,
            "passed": full.movement_exact >= MIN_FULL_MOVEMENT_EXACT,
        },
        {
            "name": "full_changed_movement_exact_count",
            "actual_count": full.changed_movement_exact,
            "minimum_count": MIN_FULL_CHANGED_EXACT,
            "passed": full.changed_movement_exact >= MIN_FULL_CHANGED_EXACT,
        },
    ]
    for intervention in INTERVENTIONS:
        altered = conditions[intervention]
        overall_drop = full.movement_exact - altered.movement_exact
        changed_drop = full.changed_movement_exact - altered.changed_movement_exact
        result.extend(
            (
                {
                    "name": f"full_vs_{intervention}_movement_exact_drop",
                    "actual_count": overall_drop,
                    "minimum_count": MIN_OVERALL_EXACT_DROP,
                    "passed": overall_drop >= MIN_OVERALL_EXACT_DROP,
                },
                {
                    "name": f"full_vs_{intervention}_changed_exact_drop",
                    "actual_count": changed_drop,
                    "minimum_count": MIN_CHANGED_EXACT_DROP,
                    "passed": changed_drop >= MIN_CHANGED_EXACT_DROP,
                },
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CausalThoughtAblationReport:
    identity: CheckpointIdentity
    validation_slice_sha256: str
    conditions: Mapping[str, DecisionCounts]
    decisions_changed_from_full: Mapping[str, int]
    paired_correctness: Mapping[str, PairedCorrectnessCounts]
    per_sequence_outcomes: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CheckpointIdentity):
            raise CausalThoughtEvaluationInputError(
                "identity must be a CheckpointIdentity"
            )
        object.__setattr__(
            self,
            "validation_slice_sha256",
            _sha256_string(
                self.validation_slice_sha256,
                name="validation_slice_sha256",
            ),
        )
        expected_conditions = {"full", *INTERVENTIONS}
        if not isinstance(self.conditions, Mapping) or set(self.conditions) != expected_conditions:
            raise CausalThoughtEvaluationInputError(
                "conditions must contain full, zeroed, and batch_shuffled exactly"
            )
        normalized_conditions: dict[str, DecisionCounts] = {}
        for name, counts in self.conditions.items():
            if not isinstance(counts, DecisionCounts):
                raise CausalThoughtEvaluationInputError(
                    f"conditions.{name} must be DecisionCounts"
                )
            normalized_conditions[name] = counts
        object.__setattr__(
            self,
            "conditions",
            MappingProxyType(dict(sorted(normalized_conditions.items()))),
        )
        if not isinstance(self.decisions_changed_from_full, Mapping) or set(
            self.decisions_changed_from_full
        ) != set(INTERVENTIONS):
            raise CausalThoughtEvaluationInputError(
                "decision-change counts must contain both interventions exactly"
            )
        normalized_changes: dict[str, int] = {}
        for name, value in self.decisions_changed_from_full.items():
            if type(value) is not int or not 0 <= value <= EXPECTED_SAMPLES:
                raise CausalThoughtEvaluationInputError(
                    f"decisions_changed_from_full.{name} is outside the sample lattice"
                )
            normalized_changes[name] = value
        object.__setattr__(
            self,
            "decisions_changed_from_full",
            MappingProxyType(dict(sorted(normalized_changes.items()))),
        )
        if not isinstance(self.paired_correctness, Mapping) or set(
            self.paired_correctness
        ) != set(INTERVENTIONS):
            raise CausalThoughtEvaluationInputError(
                "paired correctness must contain both interventions exactly"
            )
        normalized_pairs: dict[str, PairedCorrectnessCounts] = {}
        full = normalized_conditions["full"]
        for name, counts in self.paired_correctness.items():
            if not isinstance(counts, PairedCorrectnessCounts):
                raise CausalThoughtEvaluationInputError(
                    f"paired_correctness.{name} must be PairedCorrectnessCounts"
                )
            altered = normalized_conditions[name]
            if (
                counts.full_correct_intervention_wrong
                - counts.full_wrong_intervention_correct
                != full.movement_exact - altered.movement_exact
                or counts.changed_full_correct_intervention_wrong
                - counts.changed_full_wrong_intervention_correct
                != (
                    full.changed_movement_exact
                    - altered.changed_movement_exact
                )
            ):
                raise CausalThoughtEvaluationInputError(
                    f"paired correctness counts contradict condition totals for {name}"
                )
            if (
                counts.full_correct_intervention_wrong
                + counts.full_wrong_intervention_correct
                > normalized_changes[name]
            ):
                raise CausalThoughtEvaluationInputError(
                    f"correctness discordance exceeds decision changes for {name}"
                )
            normalized_pairs[name] = counts
        object.__setattr__(
            self,
            "paired_correctness",
            MappingProxyType(dict(sorted(normalized_pairs.items()))),
        )
        if not isinstance(self.per_sequence_outcomes, tuple) or len(
            self.per_sequence_outcomes
        ) != EXPECTED_SEQUENCES:
            raise CausalThoughtEvaluationInputError(
                "per_sequence_outcomes must contain exactly 16 records"
            )
        normalized_sequence_records: list[Mapping[str, object]] = []
        aggregate_exact = {name: 0 for name in expected_conditions}
        aggregate_changed_exact = {name: 0 for name in expected_conditions}
        aggregate_changed_samples = 0
        aggregate_decision_changes = {name: 0 for name in INTERVENTIONS}
        aggregate_sequence_pairs = {
            name: {key: 0 for key in normalized_pairs[name].to_dict()}
            for name in INTERVENTIONS
        }
        for expected_index, record in enumerate(self.per_sequence_outcomes):
            if (
                not isinstance(record, Mapping)
                or set(record)
                != {
                    "sequence_index",
                    "decision_count",
                    "changed_decision_count",
                    "conditions",
                    "effects",
                }
                or record.get("sequence_index") != expected_index
                or record.get("decision_count")
                != EXPECTED_SEQUENCE_LENGTH - EXPECTED_BURN_IN_STEPS
            ):
                raise CausalThoughtEvaluationInputError(
                    "per-sequence outcomes have incompatible identity or fields"
                )
            changed_samples = record["changed_decision_count"]
            if type(changed_samples) is not int or not 0 <= changed_samples <= 6:
                raise CausalThoughtEvaluationInputError(
                    "per-sequence changed decision count is outside 0..6"
                )
            aggregate_changed_samples += changed_samples
            sequence_conditions = record["conditions"]
            if not isinstance(sequence_conditions, Mapping) or set(
                sequence_conditions
            ) != expected_conditions:
                raise CausalThoughtEvaluationInputError(
                    "per-sequence conditions are incompatible"
                )
            for name, raw_counts in sequence_conditions.items():
                if not isinstance(raw_counts, Mapping) or set(raw_counts) != {
                    "movement_exact_count",
                    "changed_movement_exact_count",
                }:
                    raise CausalThoughtEvaluationInputError(
                        "per-sequence condition counts are incompatible"
                    )
                exact = raw_counts["movement_exact_count"]
                changed_exact = raw_counts["changed_movement_exact_count"]
                if (
                    type(exact) is not int
                    or type(changed_exact) is not int
                    or not 0 <= changed_exact <= changed_samples
                    or not changed_exact <= exact <= 6
                ):
                    raise CausalThoughtEvaluationInputError(
                        "per-sequence exact counts are outside their lattices"
                    )
                aggregate_exact[name] += exact
                aggregate_changed_exact[name] += changed_exact
            sequence_effects = record["effects"]
            if not isinstance(sequence_effects, Mapping) or set(
                sequence_effects
            ) != set(INTERVENTIONS):
                raise CausalThoughtEvaluationInputError(
                    "per-sequence effects are incompatible"
                )
            for name, raw_effect in sequence_effects.items():
                if not isinstance(raw_effect, Mapping) or set(raw_effect) != {
                    "decisions_changed_from_full_count",
                    "paired_correctness",
                }:
                    raise CausalThoughtEvaluationInputError(
                        "per-sequence effect fields are incompatible"
                    )
                changes = raw_effect["decisions_changed_from_full_count"]
                raw_pairs = raw_effect["paired_correctness"]
                if type(changes) is not int or not 0 <= changes <= 6:
                    raise CausalThoughtEvaluationInputError(
                        "per-sequence decision changes are outside 0..6"
                    )
                if not isinstance(raw_pairs, Mapping) or set(raw_pairs) != set(
                    aggregate_sequence_pairs[name]
                ):
                    raise CausalThoughtEvaluationInputError(
                        "per-sequence paired correctness fields are incompatible"
                    )
                full_only = raw_pairs["full_correct_intervention_wrong"]
                intervention_only = raw_pairs["full_wrong_intervention_correct"]
                changed_full_only = raw_pairs[
                    "changed_full_correct_intervention_wrong"
                ]
                changed_intervention_only = raw_pairs[
                    "changed_full_wrong_intervention_correct"
                ]
                full_condition = sequence_conditions["full"]
                altered_condition = sequence_conditions[name]
                if (
                    full_only - intervention_only
                    != (
                        full_condition["movement_exact_count"]
                        - altered_condition["movement_exact_count"]
                    )
                    or changed_full_only - changed_intervention_only
                    != (
                        full_condition["changed_movement_exact_count"]
                        - altered_condition["changed_movement_exact_count"]
                    )
                    or changed_full_only > full_only
                    or changed_intervention_only > intervention_only
                    or full_only + intervention_only > changes
                ):
                    raise CausalThoughtEvaluationInputError(
                        f"per-sequence paired ledger contradicts {name} outcomes"
                    )
                aggregate_decision_changes[name] += changes
                for pair_name, value in raw_pairs.items():
                    if type(value) is not int or not 0 <= value <= 6:
                        raise CausalThoughtEvaluationInputError(
                            "per-sequence paired correctness is outside 0..6"
                        )
                    aggregate_sequence_pairs[name][pair_name] += value
            normalized_sequence_records.append(
                _deep_freeze_json(record)
            )
        if aggregate_changed_samples != EXPECTED_CHANGED_SAMPLES:
            raise CausalThoughtEvaluationInputError(
                "per-sequence changed counts do not sum to the fixed slice"
            )
        for name in expected_conditions:
            if (
                aggregate_exact[name] != normalized_conditions[name].movement_exact
                or aggregate_changed_exact[name]
                != normalized_conditions[name].changed_movement_exact
            ):
                raise CausalThoughtEvaluationInputError(
                    f"per-sequence condition counts do not sum to {name} totals"
                )
        for name in INTERVENTIONS:
            if (
                aggregate_decision_changes[name] != normalized_changes[name]
                or aggregate_sequence_pairs[name] != normalized_pairs[name].to_dict()
            ):
                raise CausalThoughtEvaluationInputError(
                    f"per-sequence effect counts do not sum to {name} totals"
                )
        object.__setattr__(
            self,
            "per_sequence_outcomes",
            tuple(normalized_sequence_records),
        )

    @property
    def checks(self) -> tuple[dict[str, object], ...]:
        return _preregistered_checks(self.conditions)

    @property
    def passed(self) -> bool:
        return all(bool(check["passed"]) for check in self.checks)

    @property
    def causal_decision_effect_observed(self) -> bool:
        return all(
            self.decisions_changed_from_full[name] > 0 for name in INTERVENTIONS
        )

    def to_dict(self) -> dict[str, object]:
        full = self.conditions["full"]
        effects = {}
        for intervention in INTERVENTIONS:
            altered = self.conditions[intervention]
            effects[intervention] = {
                "movement_exact_drop_count": (
                    full.movement_exact - altered.movement_exact
                ),
                "changed_movement_exact_drop_count": (
                    full.changed_movement_exact
                    - altered.changed_movement_exact
                ),
                "decisions_changed_from_full_count": (
                    self.decisions_changed_from_full[intervention]
                ),
                "paired_correctness": self.paired_correctness[
                    intervention
                ].to_dict(),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "scope": SCOPE,
            "passed": self.passed,
            "causal_decision_effect_observed": self.causal_decision_effect_observed,
            "registered_recipe": self.identity.registered_recipe.to_dict(),
            "identity": self.identity.to_dict(),
            "validation_slice": {
                "split": "validation",
                "epoch": 0,
                "sequence_indices": list(range(EXPECTED_SEQUENCES)),
                "sequence_count": EXPECTED_SEQUENCES,
                "sequence_length": EXPECTED_SEQUENCE_LENGTH,
                "burn_in_steps": EXPECTED_BURN_IN_STEPS,
                "decision_count": EXPECTED_SAMPLES,
                "changed_decision_count": EXPECTED_CHANGED_SAMPLES,
                "sha256": self.validation_slice_sha256,
                "batch_shuffled_donor_sequence_index_by_receiver": [
                    *range(1, EXPECTED_SEQUENCES),
                    0,
                ],
                "batch_shuffled_target_overlap_count": (
                    EXPECTED_DONOR_TARGET_OVERLAP
                ),
                "batch_shuffled_changed_target_overlap_count": (
                    EXPECTED_CHANGED_DONOR_TARGET_OVERLAP
                ),
            },
            "interventions": {
                "zeroed": "replace the complete final thought tensor with zeros",
                "batch_shuffled": (
                    "replace sequence i's complete final thought tensor with the "
                    "same-timestep tensor from sequence (i + 1) mod 16"
                ),
                "recurrent_state_policy": (
                    "advance only the unmodified full-pass next_state"
                ),
            },
            "conditions": {
                name: counts.to_dict()
                for name, counts in sorted(self.conditions.items())
            },
            "effects": effects,
            "per_sequence_outcomes": [
                _deep_plain_json(record) for record in self.per_sequence_outcomes
            ],
            "thresholds": {
                "minimum_full_movement_exact_count": MIN_FULL_MOVEMENT_EXACT,
                "minimum_full_changed_movement_exact_count": MIN_FULL_CHANGED_EXACT,
                "minimum_movement_exact_drop_count_against_each_intervention": (
                    MIN_OVERALL_EXACT_DROP
                ),
                "minimum_changed_exact_drop_count_against_each_intervention": (
                    MIN_CHANGED_EXACT_DROP
                ),
            },
            "checks": list(self.checks),
            "limitations": list(LIMITATIONS),
        }


def _validation_slice(batch_source: object, config: object) -> tuple[object, str]:
    """Return and fingerprint the exact 16-sequence, 96-decision slice."""

    from irene_brain.training.batches import TrajectoryBatch

    optimization = getattr(config, "optimization", None)
    logging = getattr(config, "logging", None)
    dataset = getattr(config, "dataset", None)
    if (
        getattr(optimization, "batch_size", None) != 1
        or getattr(logging, "validation_batches", None) != EXPECTED_SEQUENCES
        or getattr(dataset, "sequence_length", None) != EXPECTED_SEQUENCE_LENGTH
        or getattr(dataset, "burn_in_steps", None) != EXPECTED_BURN_IN_STEPS
    ):
        raise CausalThoughtEvaluationInputError(
            "configuration does not identify the fixed 16x(8-2) validation slice"
        )
    batches = tuple(
        batch_source.iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=1,
            max_batches=EXPECTED_SEQUENCES,
        )
    )
    if len(batches) != EXPECTED_SEQUENCES or any(
        batch.batch_size != 1 for batch in batches
    ):
        raise CausalThoughtEvaluationInputError(
            "batch source did not emit exactly 16 singleton validation batches"
        )
    sequences = tuple(batch.sequences[0] for batch in batches)
    if tuple(sequence.sequence_index for sequence in sequences) != tuple(
        range(EXPECTED_SEQUENCES)
    ):
        raise CausalThoughtEvaluationInputError(
            "validation slice must contain sequence indices 0 through 15 in order"
        )
    if any(
        sequence.manifest_sha256 != REGISTERED_VALIDATION_MANIFEST_SHA256
        for sequence in sequences
    ):
        raise CausalThoughtEvaluationInputError(
            "validation sequences do not match the registered validation manifest"
        )
    combined = TrajectoryBatch(
        split="validation",
        burn_in_steps=EXPECTED_BURN_IN_STEPS,
        sequences=sequences,
    )
    if combined.sample_count != EXPECTED_SAMPLES:
        raise CausalThoughtEvaluationInputError(
            "validation slice does not contain exactly 96 optimized decisions"
        )

    target_active = 0
    changed_samples = 0
    previous_exact = 0
    donor_target_overlap = 0
    changed_donor_target_overlap = 0
    for sequence in sequences:
        for transition in sequence.transitions[EXPECTED_BURN_IN_STEPS:]:
            target = set(transition.action_target.keys_down) & set(
                MOVEMENT_CONTROL_INDICES
            )
            previous = set(transition.observation.previous_control.keys_down) & set(
                MOVEMENT_CONTROL_INDICES
            )
            target_active += len(target)
            changed_samples += target != previous
            previous_exact += target == previous
    for sequence_index, sequence in enumerate(sequences):
        donor = sequences[(sequence_index + 1) % EXPECTED_SEQUENCES]
        for time_index in range(EXPECTED_BURN_IN_STEPS, EXPECTED_SEQUENCE_LENGTH):
            transition = sequence.transitions[time_index]
            donor_transition = donor.transitions[time_index]
            target = set(transition.action_target.keys_down) & set(
                MOVEMENT_CONTROL_INDICES
            )
            previous = set(transition.observation.previous_control.keys_down) & set(
                MOVEMENT_CONTROL_INDICES
            )
            donor_target = set(donor_transition.action_target.keys_down) & set(
                MOVEMENT_CONTROL_INDICES
            )
            same_target = target == donor_target
            donor_target_overlap += same_target
            changed_donor_target_overlap += (target != previous) and same_target
    actual_invariants = (target_active, changed_samples, previous_exact)
    expected_invariants = (
        EXPECTED_TARGET_ACTIVE,
        EXPECTED_CHANGED_SAMPLES,
        EXPECTED_PREVIOUS_EXACT,
    )
    if actual_invariants != expected_invariants:
        raise CausalThoughtEvaluationInputError(
            "validation target invariants do not match the registered Stage-A slice"
        )
    if (
        donor_target_overlap,
        changed_donor_target_overlap,
    ) != (
        EXPECTED_DONOR_TARGET_OVERLAP,
        EXPECTED_CHANGED_DONOR_TARGET_OVERLAP,
    ):
        raise CausalThoughtEvaluationInputError(
            "registered cyclic donor overlap invariants do not match the slice"
        )

    digest = sha256(b"IRENECAUSALTHOUGHTSLICE\x01")
    digest.update(bytes.fromhex(batch_source.manifest_sha256))
    for sequence in sequences:
        digest.update(bytes.fromhex(sequence.content_sha256))
    return combined, digest.hexdigest()


def _capture_final_actuator_context(
    model: object,
    *model_args: object,
    **model_kwargs: object,
) -> tuple[object, Mapping[str, object]]:
    """Run a full pass and capture only the last actuator's keyword inputs."""

    calls: list[dict[str, object]] = []

    def capture(_module: object, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        if args:
            raise CausalThoughtEvaluationInputError(
                "actuator invocation unexpectedly used positional arguments"
            )
        calls.append(dict(kwargs))

    handle = model.actuator.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        output = model(*model_args, **model_kwargs)
    finally:
        handle.remove()
    expected_calls = model.config.cognitive_cycles + 1
    if len(calls) != expected_calls:
        raise CausalThoughtEvaluationInputError(
            "model did not invoke one actuator exit per configured cycle plus exit zero"
        )
    context = calls[-1]
    expected_fields = {
        "sensors",
        "belief",
        "thoughts",
        "working_memory",
        "retrieved_memory",
        "goal_context",
    }
    if set(context) != expected_fields:
        raise CausalThoughtEvaluationInputError(
            "final actuator context has incompatible fields"
        )
    if context["thoughts"] is not output.next_state.thoughts:
        raise CausalThoughtEvaluationInputError(
            "captured final thoughts are not the full pass's recurrent next_state"
        )
    return output, MappingProxyType(context)


def _movement_prediction(action: object) -> tuple[bool, bool, bool, bool]:
    import torch

    logits = action.button_logits
    if tuple(logits.shape) != (1, 296):
        raise CausalThoughtEvaluationInputError(
            "final actuator button_logits must have shape [1, 296]"
        )
    if not bool(torch.isfinite(logits).all()):
        raise CausalThoughtEvaluationInputError(
            "final actuator button_logits must all be finite"
        )
    indices = torch.tensor(
        MOVEMENT_CONTROL_INDICES,
        dtype=torch.long,
        device=logits.device,
    )
    values = logits.index_select(1, indices).float() > 0.0
    return tuple(bool(value) for value in values[0].cpu().tolist())


def _rng_snapshot(torch: object) -> tuple[object, object, tuple[object, ...]]:
    cpu = torch.get_rng_state().clone()
    cuda = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return random.getstate(), cpu, cuda


def _same_rng_snapshot(torch: object, left: object, right: object) -> bool:
    left_python, left_cpu, left_cuda = left
    right_python, right_cpu, right_cuda = right
    return (
        left_python == right_python
        and bool(torch.equal(left_cpu, right_cpu))
        and len(left_cuda) == len(right_cuda)
        and all(
            bool(torch.equal(left_state, right_state))
            for left_state, right_state in zip(left_cuda, right_cuda)
        )
    )


def evaluate_causal_thought_ablation(
    *,
    model: object,
    validation_batch: object,
    validation_slice_sha256: str,
    identity: CheckpointIdentity,
    autocast_context_factory: Callable[[], object] = nullcontext,
) -> CausalThoughtAblationReport:
    """Evaluate full, zeroed, and deranged-donor final thought readouts."""

    import torch
    from irene_brain.training.batches import control_to_vector
    from irene_brain.training.objective import (
        _rgb_tensor,
        deterministic_eval_thought_noise,
    )

    if getattr(validation_batch, "split", None) != "validation":
        raise CausalThoughtEvaluationInputError("evaluation requires validation data")
    if getattr(validation_batch, "burn_in_steps", None) != EXPECTED_BURN_IN_STEPS:
        raise CausalThoughtEvaluationInputError("validation burn-in is incompatible")
    sequences = getattr(validation_batch, "sequences", None)
    if not isinstance(sequences, tuple) or len(sequences) != EXPECTED_SEQUENCES:
        raise CausalThoughtEvaluationInputError(
            "evaluation requires exactly 16 validation sequences"
        )
    if tuple(sequence.sequence_index for sequence in sequences) != tuple(
        range(EXPECTED_SEQUENCES)
    ):
        raise CausalThoughtEvaluationInputError(
            "evaluation sequences are not the registered ordered slice"
        )
    if any(len(sequence.transitions) != EXPECTED_SEQUENCE_LENGTH for sequence in sequences):
        raise CausalThoughtEvaluationInputError(
            "evaluation sequences must each contain eight transitions"
        )
    _sha256_string(validation_slice_sha256, name="validation_slice_sha256")
    if not isinstance(identity, CheckpointIdentity):
        raise CausalThoughtEvaluationInputError("identity must be a CheckpointIdentity")
    if not callable(autocast_context_factory):
        raise CausalThoughtEvaluationInputError(
            "autocast_context_factory must be callable"
        )

    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as error:
        raise CausalThoughtEvaluationInputError(
            "model must expose at least one parameter"
        ) from error
    device = parameter.device
    model.eval()
    thought_noise = deterministic_eval_thought_noise(
        thoughtlets=model.config.thoughtlets,
        width=model.config.core_width,
        batch_size=1,
        device=device,
    )
    states: list[object | None] = [None] * EXPECTED_SEQUENCES
    exact_counts = {name: 0 for name in ("full", *INTERVENTIONS)}
    changed_exact_counts = {name: 0 for name in ("full", *INTERVENTIONS)}
    decisions_changed = {name: 0 for name in INTERVENTIONS}
    paired_counts = {
        name: {
            "full_correct_intervention_wrong": 0,
            "full_wrong_intervention_correct": 0,
            "changed_full_correct_intervention_wrong": 0,
            "changed_full_wrong_intervention_correct": 0,
        }
        for name in INTERVENTIONS
    }
    sequence_exact = [
        {name: 0 for name in ("full", *INTERVENTIONS)}
        for _ in range(EXPECTED_SEQUENCES)
    ]
    sequence_changed_exact = [
        {name: 0 for name in ("full", *INTERVENTIONS)}
        for _ in range(EXPECTED_SEQUENCES)
    ]
    sequence_changed_samples = [0] * EXPECTED_SEQUENCES
    sequence_decisions_changed = [
        {name: 0 for name in INTERVENTIONS}
        for _ in range(EXPECTED_SEQUENCES)
    ]
    sequence_paired = [
        {
            name: {
                "full_correct_intervention_wrong": 0,
                "full_wrong_intervention_correct": 0,
                "changed_full_correct_intervention_wrong": 0,
                "changed_full_wrong_intervention_correct": 0,
            }
            for name in INTERVENTIONS
        }
        for _ in range(EXPECTED_SEQUENCES)
    ]
    observed_samples = 0
    observed_changed = 0
    rng_before = _rng_snapshot(torch)

    with torch.no_grad():
        for time_index in range(EXPECTED_SEQUENCE_LENGTH):
            outputs: list[object] = []
            contexts: list[Mapping[str, object]] = []
            transitions: list[object] = []
            for sequence_index, sequence in enumerate(sequences):
                transition = sequence.transitions[time_index]
                transitions.append(transition)
                pixels = _rgb_tensor(
                    (transition.observation.rgb,),
                    device=device,
                    resolution=getattr(model, "input_resolution", None),
                )
                previous_control = torch.tensor(
                    [control_to_vector(transition.observation.previous_control)],
                    dtype=torch.float32,
                    device=device,
                )
                elapsed_seconds = torch.tensor(
                    [
                        0.0
                        if time_index == 0
                        else (
                            transition.observation.elapsed_ns
                            - sequence.transitions[
                                time_index - 1
                            ].observation.elapsed_ns
                        )
                        / 1_000_000_000.0
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                with autocast_context_factory():
                    output, context = _capture_final_actuator_context(
                        model,
                        pixels,
                        previous_control,
                        elapsed_seconds,
                        states[sequence_index],
                        thought_noise=thought_noise,
                    )
                outputs.append(output)
                contexts.append(context)
                states[sequence_index] = (
                    output.next_state.detach()
                    if time_index < EXPECTED_BURN_IN_STEPS
                    else output.next_state
                )

            if time_index < EXPECTED_BURN_IN_STEPS:
                continue

            for sequence_index, (output, context, transition) in enumerate(
                zip(outputs, contexts, transitions)
            ):
                zero_context = dict(context)
                zero_context["thoughts"] = torch.zeros_like(context["thoughts"])
                shuffled_context = dict(context)
                shuffled_context["thoughts"] = contexts[
                    (sequence_index + 1) % EXPECTED_SEQUENCES
                ]["thoughts"]
                with autocast_context_factory():
                    replayed_full_action = model.actuator(**dict(context))
                    zero_action = model.actuator(**zero_context)
                    shuffled_action = model.actuator(**shuffled_context)

                replayed_logits = replayed_full_action.button_logits
                original_logits = output.action.button_logits
                if tuple(replayed_logits.shape) != tuple(original_logits.shape) or not bool(
                    torch.equal(replayed_logits, original_logits)
                ):
                    raise CausalThoughtEvaluationInputError(
                        "unmodified final actuator replay did not reproduce the full pass"
                    )

                predictions = {
                    "full": _movement_prediction(output.action),
                    "zeroed": _movement_prediction(zero_action),
                    "batch_shuffled": _movement_prediction(shuffled_action),
                }
                target_keys = set(transition.action_target.keys_down)
                previous_keys = set(
                    transition.observation.previous_control.keys_down
                )
                target = tuple(
                    index in target_keys for index in MOVEMENT_CONTROL_INDICES
                )
                previous = tuple(
                    index in previous_keys for index in MOVEMENT_CONTROL_INDICES
                )
                changed = target != previous
                observed_samples += 1
                observed_changed += changed
                sequence_changed_samples[sequence_index] += changed
                exact_by_condition: dict[str, bool] = {}
                for name, prediction in predictions.items():
                    exact = prediction == target
                    exact_by_condition[name] = exact
                    exact_counts[name] += exact
                    changed_exact_counts[name] += changed and exact
                    sequence_exact[sequence_index][name] += exact
                    sequence_changed_exact[sequence_index][name] += changed and exact
                for name in INTERVENTIONS:
                    decision_differs = predictions[name] != predictions["full"]
                    decisions_changed[name] += decision_differs
                    sequence_decisions_changed[sequence_index][name] += decision_differs
                    full_win = exact_by_condition["full"] and not exact_by_condition[name]
                    intervention_win = (
                        not exact_by_condition["full"] and exact_by_condition[name]
                    )
                    paired_counts[name][
                        "full_correct_intervention_wrong"
                    ] += full_win
                    paired_counts[name][
                        "full_wrong_intervention_correct"
                    ] += intervention_win
                    sequence_paired[sequence_index][name][
                        "full_correct_intervention_wrong"
                    ] += full_win
                    sequence_paired[sequence_index][name][
                        "full_wrong_intervention_correct"
                    ] += intervention_win
                    if changed:
                        paired_counts[name][
                            "changed_full_correct_intervention_wrong"
                        ] += full_win
                        paired_counts[name][
                            "changed_full_wrong_intervention_correct"
                        ] += intervention_win
                        sequence_paired[sequence_index][name][
                            "changed_full_correct_intervention_wrong"
                        ] += full_win
                        sequence_paired[sequence_index][name][
                            "changed_full_wrong_intervention_correct"
                        ] += intervention_win

    rng_after = _rng_snapshot(torch)
    if not _same_rng_snapshot(torch, rng_before, rng_after):
        raise CausalThoughtEvaluationInputError(
            "causal evaluation unexpectedly consumed a global RNG stream"
        )
    if observed_samples != EXPECTED_SAMPLES or observed_changed != EXPECTED_CHANGED_SAMPLES:
        raise CausalThoughtEvaluationInputError(
            "evaluated decision counts do not match the registered validation slice"
        )
    conditions = {
        name: DecisionCounts(
            movement_exact=exact_counts[name],
            changed_movement_exact=changed_exact_counts[name],
        )
        for name in ("full", *INTERVENTIONS)
    }
    paired = {
        name: PairedCorrectnessCounts(**paired_counts[name])
        for name in INTERVENTIONS
    }
    per_sequence_outcomes = tuple(
        {
            "sequence_index": sequence_index,
            "decision_count": EXPECTED_SEQUENCE_LENGTH - EXPECTED_BURN_IN_STEPS,
            "changed_decision_count": sequence_changed_samples[sequence_index],
            "conditions": {
                name: {
                    "movement_exact_count": sequence_exact[sequence_index][name],
                    "changed_movement_exact_count": sequence_changed_exact[
                        sequence_index
                    ][name],
                }
                for name in ("full", *INTERVENTIONS)
            },
            "effects": {
                name: {
                    "decisions_changed_from_full_count": (
                        sequence_decisions_changed[sequence_index][name]
                    ),
                    "paired_correctness": dict(
                        sequence_paired[sequence_index][name]
                    ),
                }
                for name in INTERVENTIONS
            },
        }
        for sequence_index in range(EXPECTED_SEQUENCES)
    )
    return CausalThoughtAblationReport(
        identity=identity,
        validation_slice_sha256=validation_slice_sha256,
        conditions=conditions,
        decisions_changed_from_full=decisions_changed,
        paired_correctness=paired,
        per_sequence_outcomes=per_sequence_outcomes,
    )


def _resolve_release_inputs(
    *,
    training_release_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    expected_training_code_sha256: str,
) -> tuple[Path, Path, Path]:
    """Bind imports and configuration to one immutable training release."""

    from irene_brain.training.checkpoint import source_tree_sha256

    expected_code = _sha256_string(
        expected_training_code_sha256,
        name="expected_training_code_sha256",
    )
    release_root = Path(training_release_root).resolve(strict=True)
    if not release_root.is_dir():
        raise CausalThoughtEvaluationInputError(
            "training_release_root must be a directory"
        )
    source_root = (release_root / "brain" / "src" / "irene_brain").resolve(
        strict=True
    )
    if not source_root.is_dir():
        raise CausalThoughtEvaluationInputError(
            "training release does not contain brain/src/irene_brain"
        )
    config = Path(config_path).resolve(strict=True)
    if not config.is_file() or not config.is_relative_to(release_root):
        raise CausalThoughtEvaluationInputError(
            "configuration must be a file inside the identified training release"
        )
    package = importlib.import_module("irene_brain")
    package_file = getattr(package, "__file__", None)
    if not package_file:
        raise CausalThoughtEvaluationInputError(
            "imported irene_brain package has no filesystem identity"
        )
    imported_source_root = Path(package_file).resolve().parent
    if imported_source_root != source_root:
        raise CausalThoughtEvaluationInputError(
            "PYTHONPATH imported irene_brain from a different release"
        )
    actual_code = source_tree_sha256(source_root)
    if actual_code != expected_code:
        raise CausalThoughtEvaluationInputError(
            "training release source tree does not match the expected code SHA-256"
        )
    return release_root, source_root, config


def _factory(path: str) -> Callable[[object], object]:
    try:
        module_name, attribute = path.split(":", 1)
    except ValueError as error:
        raise CausalThoughtEvaluationInputError(
            "model factory must use module.path:callable syntax"
        ) from error
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise CausalThoughtEvaluationInputError(
            "configured model factory is not callable"
        )
    return factory


def evaluate_checkpoint(
    *,
    training_release_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
    expected_config_file_sha256: str,
    expected_data_sha256: str,
    expected_training_code_sha256: str,
    expected_evaluator_sha256: str,
    expected_optimizer_step: int,
) -> CausalThoughtAblationReport:
    """Strictly load one checkpoint and run the fixed causal readout gate."""

    expected_checkpoint = _sha256_string(
        expected_checkpoint_sha256,
        name="expected_checkpoint_sha256",
    )
    expected_config = _sha256_string(
        expected_config_sha256,
        name="expected_config_sha256",
    )
    expected_config_file = _sha256_string(
        expected_config_file_sha256,
        name="expected_config_file_sha256",
    )
    expected_data = _sha256_string(
        expected_data_sha256,
        name="expected_data_sha256",
    )
    recipe = _registered_recipe(
        config_sha256=expected_config,
        config_file_sha256=expected_config_file,
        data_sha256=expected_data,
    )
    expected_code = _sha256_string(
        expected_training_code_sha256,
        name="expected_training_code_sha256",
    )
    expected_evaluator = _sha256_string(
        expected_evaluator_sha256,
        name="expected_evaluator_sha256",
    )
    expected_step = _positive_integer(
        expected_optimizer_step,
        name="expected_optimizer_step",
    )
    if expected_step != recipe.optimizer_step:
        raise CausalThoughtEvaluationInputError(
            f"expected_optimizer_step must be {recipe.optimizer_step} for "
            f"{recipe.recipe_id}"
        )
    _, source_root, config_file = _resolve_release_inputs(
        training_release_root=training_release_root,
        config_path=config_path,
        expected_training_code_sha256=expected_code,
    )

    from irene_brain.training.checkpoint import file_sha256
    from irene_brain.training.config import load_training_config

    if config_file.name != recipe.config_filename:
        raise CausalThoughtEvaluationInputError(
            "configuration filename does not match the selected registered recipe"
        )
    if file_sha256(config_file) != expected_config_file:
        raise CausalThoughtEvaluationInputError(
            "raw TOML artifact does not match the expected SHA-256"
        )
    if file_sha256(Path(__file__).resolve()) != expected_evaluator:
        raise CausalThoughtEvaluationInputError(
            "evaluator artifact does not match the expected SHA-256"
        )
    checkpoint_file = Path(checkpoint_path).resolve(strict=True)
    if not checkpoint_file.is_file():
        raise CausalThoughtEvaluationInputError("checkpoint must be a regular file")
    if file_sha256(checkpoint_file) != expected_checkpoint:
        raise CausalThoughtEvaluationInputError(
            "checkpoint artifact does not match the expected SHA-256"
        )

    config = load_training_config(config_file)
    if config.config_sha256 != expected_config:
        raise CausalThoughtEvaluationInputError(
            "configuration does not match the expected SHA-256"
        )
    if (
        config.schema_version,
        config.run.name,
        config.optimization.scheduler_kind,
    ) != (
        recipe.config_schema_version,
        recipe.run_name,
        recipe.scheduler_kind,
    ):
        raise CausalThoughtEvaluationInputError(
            "configuration semantics do not match the selected registered recipe"
        )
    for name, value in config.resource_policy.process_environment().items():
        os.environ[name] = value
    if config.precision.device == "cuda" and config.determinism.enabled:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in {":4096:8", ":16:8"}:
            raise CausalThoughtEvaluationInputError(
                "deterministic CUDA requires a supported CUBLAS workspace setting"
            )

    import torch
    from irene_brain.training.batches import MovingShapesBatchSource
    from irene_brain.training.checkpoint import load_checkpoint
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    random.seed(config.run.seed)
    torch.manual_seed(config.run.seed)
    if config.precision.device == "cuda":
        torch.cuda.manual_seed_all(config.run.seed)
    model = _factory(config.run.model_factory)(config)
    objective = ThoughtFieldObjective(
        model,
        action_loss_kind=config.objective.action_loss_kind,
        button_support_control_indices=config.objective.button_support_control_indices,
        button_support_weight=config.objective.button_support_weight,
        button_background_weight=config.objective.button_background_weight,
        button_background_tail_mix=config.objective.button_background_tail_mix,
        button_background_tail_temperature=(
            config.objective.button_background_tail_temperature
        ),
        continuous_action_weight=config.objective.continuous_action_weight,
        action_weight=config.objective.action_weight,
        value_weight=config.objective.value_weight,
        world_weight=config.objective.world_weight,
        diversity_weight=config.objective.diversity_weight,
    )
    system = TorchTrainingSystem(objective, config)
    source = MovingShapesBatchSource(config.dataset)
    if source.manifest_sha256 != expected_data:
        raise CausalThoughtEvaluationInputError(
            "dataset manifest does not match the expected SHA-256"
        )
    validation_batch, slice_sha256 = _validation_slice(source, config)
    loaded = load_checkpoint(
        checkpoint_file,
        expected_config_sha256=expected_config,
        expected_data_sha256=expected_data,
        expected_code_sha256=expected_code,
        expected_runtime_fingerprint=system.runtime_fingerprint,
        expected_checkpoint_sha256=expected_checkpoint,
    )
    if (
        loaded.cursor.optimizer_step,
        loaded.cursor.epoch,
        loaded.cursor.next_batch,
    ) != (
        expected_step,
        recipe.epoch,
        recipe.next_batch,
    ):
        raise CausalThoughtEvaluationInputError(
            f"checkpoint cursor does not match {recipe.recipe_id}'s registered boundary"
        )
    system.restore_checkpoint_state(loaded.system_state)
    system.restore_rng_state(loaded.rng_state)

    def autocast_context() -> object:
        if config.precision.mode == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if config.precision.mode == "bfloat16" else torch.float16
        return torch.autocast(device_type=system.device.type, dtype=dtype)

    identity = CheckpointIdentity(
        checkpoint_sha256=loaded.checkpoint_sha256,
        config_sha256=config.config_sha256,
        config_file_sha256=expected_config_file,
        data_sha256=source.manifest_sha256,
        training_code_sha256=expected_code,
        evaluator_sha256=expected_evaluator,
        optimizer_step=loaded.cursor.optimizer_step,
        epoch=loaded.cursor.epoch,
        next_batch=loaded.cursor.next_batch,
        runtime_fingerprint=system.runtime_fingerprint,
        training_source_root=str(source_root),
    )
    return evaluate_causal_thought_ablation(
        model=system.objective.model,
        validation_batch=validation_batch,
        validation_slice_sha256=slice_sha256,
        identity=identity,
        autocast_context_factory=autocast_context,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed Stage-A final-thought causal ablation"
    )
    parser.add_argument("--training-release-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--config-file-sha256", required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--training-code-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--optimizer-step", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = evaluate_checkpoint(
            training_release_root=arguments.training_release_root,
            config_path=arguments.config,
            checkpoint_path=arguments.checkpoint,
            expected_checkpoint_sha256=arguments.checkpoint_sha256,
            expected_config_sha256=arguments.config_sha256,
            expected_config_file_sha256=arguments.config_file_sha256,
            expected_data_sha256=arguments.data_sha256,
            expected_training_code_sha256=arguments.training_code_sha256,
            expected_evaluator_sha256=arguments.evaluator_sha256,
            expected_optimizer_step=arguments.optimizer_step,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "passed": False,
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(
        json.dumps(report.to_dict(), allow_nan=False, sort_keys=True),
        flush=True,
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CausalThoughtAblationReport",
    "CausalThoughtEvaluationInputError",
    "CheckpointIdentity",
    "DecisionCounts",
    "PairedCorrectnessCounts",
    "REGISTERED_A_CONFIG_FILE_SHA256",
    "REGISTERED_A_CONFIG_SHA256",
    "REGISTERED_A_RECIPE_ID",
    "REGISTERED_B_CONFIG_FILE_SHA256",
    "REGISTERED_B_CONFIG_SHA256",
    "REGISTERED_B_RECIPE_ID",
    "REGISTERED_DATA_SHA256",
    "REGISTERED_RECIPES",
    "RegisteredRecipe",
    "evaluate_causal_thought_ablation",
    "evaluate_checkpoint",
    "main",
]
