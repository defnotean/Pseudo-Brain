"""Dependency-free RCQ-v2 registration, scoring, and retirement primitives.

The production trust boundary and sole receipt-authoring CLI live in
:mod:`irene_brain.evaluation.rcq_v2_torch`.  This module deliberately exports no
authorization constructor, evidence-factory entry point, receipt writer, or
unreceipted scorer.  The CLI publishes its authoritative claim before trusted
code may construct either frozen TEST dataset.  A surviving claim retires both
ranges, including after a crash or failure, only within the preserved canonical
DGX host registry.  Preventing copied/reset host registries is delegated to the
fixed dispatcher and external registration/claim/receipt pins.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum, isfinite, sqrt
import os
from pathlib import Path
import random
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

from ..data import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequence,
    MovingShapesSequenceDataset,
)
from ..runtime.policy import Capability, ResourcePolicy
from ..training.batches import BUTTON_TARGET_INDICES, CONTINUOUS_TARGET_INDICES
from ..types import HidKey
from .rcq_v2 import RCQCheck, RCQInputError


FINAL_EVALUATOR_ID = "rcq_v2_final_v1"
FINAL_RANGE_CLAIM_PROTOCOL = "irene_moving_shapes_test_range_retirement_v1"
RECIPIENT_OFFSET = 3_145_728
RECIPIENT_SEQUENCES = 512
DONOR_OFFSET = 3_146_240
DONOR_SEQUENCES = 512
RETIRED_END = 3_146_752
GUARD_END = 3_147_264
FUTURE_CAMPAIGN_OFFSET = 2_097_152
FINAL_STEP = 2_048
FINAL_STAGE_INDEX = 1
FINAL_DECISIONS = 3_072
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1702
BOOTSTRAP_LOWER_INDEX = 499
DONOR_MATCHING_SEED = 1702
CONTINUOUS_DEADZONE = 0.05
EVALUATOR_BUNDLE_FILES = (
    "evaluation/rcq_v2.py",
    "evaluation/rcq_v2_final.py",
    "evaluation/rcq_v2_torch.py",
)

_MOVEMENT_INDICES = (
    int(HidKey.W),
    int(HidKey.A),
    int(HidKey.S),
    int(HidKey.D),
)
_BUTTON_INDEX_SET = frozenset(BUTTON_TARGET_INDICES)
_CONTINUOUS_INDEX_SET = frozenset(CONTINUOUS_TARGET_INDICES)


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RCQInputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise RCQInputError(f"non-finite JSON number is forbidden: {value}")


def _canonical(value: object) -> str:
    return json.dumps(
        _deep_thaw(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RCQInputError("immutable payload keys must be strings")
            frozen[key] = _deep_freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise RCQInputError("immutable payload contains a non-JSON value")


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(item) for item in value]
    return value


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(  # type: ignore[arg-type]
            _strict_equal(left[key], right[key]) for key in left  # type: ignore[index]
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _strict_equal(a, b) for a, b in zip(left, right)  # type: ignore[arg-type]
        )
    return left == right


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _digest_payload(domain: bytes, value: object) -> str:
    return sha256(domain + _canonical(value).encode("utf-8")).hexdigest()


def _final_range_claim_id() -> str:
    """Return the canonical-registry key for these TEST ranges and split protocol.

    ``FINAL_RANGE_CLAIM_PROTOCOL`` is deliberately independent of the evaluator,
    registration bytes, run, and checkpoint.  Future evaluator versions must
    therefore collide with this retirement key if they try to reuse the same
    sealed split ranges.
    """

    return _digest_payload(
        b"IRENERCQRANGECLAIM\x01",
        {
            "retirement_protocol": FINAL_RANGE_CLAIM_PROTOCOL,
            "split": "test",
            "recipient": [RECIPIENT_OFFSET, RECIPIENT_OFFSET + RECIPIENT_SEQUENCES],
            "donor": [DONOR_OFFSET, DONOR_OFFSET + DONOR_SEQUENCES],
        },
    )


def _hash_string(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RCQInputError(f"{name} must be a lowercase SHA-256 string")
    return value


@dataclass(frozen=True, slots=True)
class RCQRegistration:
    payload: Mapping[str, object]
    sha256: str

    def __post_init__(self) -> None:
        _hash_string(self.sha256, name="registration sha256")
        if not isinstance(self.payload, Mapping):
            raise RCQInputError("registration payload must be a mapping")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))

    @property
    def config_sha256(self) -> str:
        return str(self.payload["config_canonical_sha256"])


def load_rcq_v2_registration(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> RCQRegistration:
    """Load and strictly verify the externally pinned frozen registration."""

    expected = _hash_string(expected_sha256, name="expected registration sha256")
    try:
        encoded = Path(path).read_bytes()
    except OSError as error:
        raise RCQInputError(f"cannot read RCQ registration: {error}") from error
    actual = _digest_bytes(encoded)
    if actual != expected:
        raise RCQInputError("RCQ registration does not match its external SHA-256 pin")
    try:
        raw = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except RCQInputError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RCQInputError(f"RCQ registration is not strict JSON: {error}") from error
    _validate_registration(raw)
    assert isinstance(raw, dict)
    if encoded != (_canonical(raw) + "\n").encode("utf-8"):
        raise RCQInputError("RCQ registration must be one byte-canonical JSON line")
    return RCQRegistration(payload=raw, sha256=actual)


def _validate_registration(raw: object) -> None:
    required = {
        "schema_version",
        "qualification_id",
        "evaluator_id",
        "config_canonical_sha256",
        "config_raw_sha256",
        "source_tree_sha256",
        "evaluator_bundle_sha256",
        "batch_source_manifest_sha256",
        "run_seed",
        "run_id",
        "model_factory",
        "joint_end_step",
        "final_step",
        "sequence_length",
        "burn_in_steps",
        "hazard_count",
        "tick_period_ns",
        "discount_hex",
        "slices",
        "bootstrap",
        "thresholds",
        "guard_band",
        "future_campaign_offset",
        "runtime_protocol",
        "development_gates",
        "workspace_protocol",
        "receipt_directory",
        "config_test_field_status",
        "scope",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise RCQInputError("RCQ registration has incompatible fields")
    exact = {
        "schema_version": 2,
        "qualification_id": "rcq_v2_reference_v1",
        "evaluator_id": FINAL_EVALUATOR_ID,
        "run_seed": 1702,
        "run_id": "dgx-rcq-v2-reference-seed-1702",
        "model_factory": "irene_brain.training.factory:build_thesis_model",
        "joint_end_step": 1536,
        "final_step": FINAL_STEP,
        "sequence_length": 8,
        "burn_in_steps": 2,
        "hazard_count": 3,
        "tick_period_ns": 16_666_667,
        "discount_hex": float(0.99).hex(),
        "future_campaign_offset": FUTURE_CAMPAIGN_OFFSET,
        "receipt_directory": f"final-claims/{_final_range_claim_id()}",
        "config_test_field_status": "disabled_retired_placeholder_generic_trainer_test_forbidden",
        "scope": (
            "one-seed open-loop teacher-forced reference-policy qualification; "
            "not closed-loop gameplay, architecture superiority, or causal thought use"
        ),
    }
    for name, expected in exact.items():
        if type(raw[name]) is not type(expected) or raw[name] != expected:
            raise RCQInputError(f"RCQ registration field {name!r} is not frozen")
    _hash_string(raw["config_canonical_sha256"], name="config canonical sha256")
    _hash_string(raw["config_raw_sha256"], name="config raw sha256")
    _hash_string(raw["source_tree_sha256"], name="source tree sha256")
    _hash_string(raw["evaluator_bundle_sha256"], name="evaluator bundle sha256")
    _hash_string(
        raw["batch_source_manifest_sha256"],
        name="batch source manifest sha256",
    )
    if not _strict_equal(raw["runtime_protocol"], {
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
        "rgb_intervention": "replace_rgb_only_from_one_donor_sequence_all_timesteps",
        "donor_assignment": (
            "target_blind_one_to_one_sequence_minimum_total_previous_wasd_bit_hamming"
        ),
        "donor_tie_break": "sha256_seeded_donor_order_then_lowest_hungarian_column",
        "donor_matching_seed": DONOR_MATCHING_SEED,
        "donor_rgb_constraint": "all_eight_corresponding_frame_sha256_values_distinct",
    }):
        raise RCQInputError("RCQ runtime protocol registration changed")
    if not _strict_equal(raw["development_gates"], {
        "entry": {
            "gate": "rcq_v2_development_v1",
            "optimizer_step": 1_536,
        },
        "completion": {
            "gate": "rcq_v2_value_development_v1",
            "optimizer_step": FINAL_STEP,
            "entry_gate": "rcq_v2_development_v1",
            "entry_optimizer_step": 1_536,
            "train_timestep_previous_wasd_dev_mse_hex": "0x1.754d5eea85785p-2",
            "absolute_max_mse_hex": "0x1.4ff8d56cab52bp-2",
            "dev_target_variance_hex": "0x1.be77815f41fbfp-2",
            "entry_improvement_ratio_hex": "0x1.ccccccccccccdp-1",
        },
    }):
        raise RCQInputError("RCQ development-gate registration changed")
    if not _strict_equal(raw["workspace_protocol"], {
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
            "qualification-pins/rcq-v2-reference-v1"
        ),
        "pretraining_pin_filename": "pretraining.json",
        "final_authorization_filename": "final-authorization.json",
        "registration_release_relative_path": (
            "registrations/rcq-v2-reference-v1.json"
        ),
        "readiness_receipt_relative_path": (
            "preclaim-readiness/rcq-v2-reference-v1.json"
        ),
    }):
        raise RCQInputError("RCQ canonical workspace protocol changed")
    expected_slices = {
        "train": ("train", 1_048_576, 8_192),
        "development": ("validation", 1_048_576, 256),
        "final_recipient": ("test", RECIPIENT_OFFSET, RECIPIENT_SEQUENCES),
        "final_donor_only": ("test", DONOR_OFFSET, DONOR_SEQUENCES),
    }
    slices = raw["slices"]
    if not isinstance(slices, dict) or set(slices) != set(expected_slices):
        raise RCQInputError("RCQ registration slice set is incompatible")
    for name, (split, start, count) in expected_slices.items():
        value = slices[name]
        if not isinstance(value, dict) or set(value) != {
            "split",
            "local_start",
            "local_end",
            "sequences",
            "scored_decisions",
            "manifest_sha256",
        }:
            raise RCQInputError(f"registered slice {name!r} has incompatible fields")
        expected_slice = {
            "split": split,
            "local_start": start,
            "local_end": start + count,
            "sequences": count,
            "scored_decisions": 0 if name == "final_donor_only" else count * 6,
        }
        for field, expected in expected_slice.items():
            if type(value[field]) is not type(expected) or value[field] != expected:
                raise RCQInputError(f"registered slice {name!r}.{field} changed")
        _hash_string(value["manifest_sha256"], name=f"{name} manifest")
    bootstrap = raw["bootstrap"]
    if not _strict_equal(bootstrap, {
        "cluster": "unique_recipient_donor_sequence_pair_with_all_six_paired_decisions",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "one_sided_confidence": "0x1.e666666666666p-1",
        "sorted_lower_index_zero_based": BOOTSTRAP_LOWER_INDEX,
    }):
        raise RCQInputError("RCQ bootstrap registration changed")
    thresholds = raw["thresholds"]
    if not _strict_equal(thresholds, _registered_thresholds()):
        raise RCQInputError("RCQ threshold registration changed")
    if not _strict_equal(raw["guard_band"], {
        "local_start": RETIRED_END,
        "local_end": GUARD_END,
        "status": "unused",
    }):
        raise RCQInputError("RCQ guard-band registration changed")


def _registered_thresholds() -> dict[str, object]:
    return {
        "movement_exact_min_fraction_hex": float(0.85).hex(),
        "changed_exact_min_fraction_hex": float(0.60).hex(),
        "sample_macro_positive_recall_min_hex": float(0.93).hex(),
        "movement_false_positive_max_per_decision_hex": float(0.10).hex(),
        "opposite_conflict_max_fraction_hex": float(0.002).hex(),
        "off_support_button_positive_max_count": 0,
        "continuous_deadzone_abs_max_hex": CONTINUOUS_DEADZONE.hex(),
        "continuous_deadzone_violation_max_count": 0,
        "action_baseline_margin_min_hex": float(0.10).hex(),
        "rgb_deranged_exact_delta_min_hex": float(0.10).hex(),
        "rgb_deranged_changed_delta_min_hex": float(0.20).hex(),
        "rgb_deranged_exact_bootstrap_lower_min_hex": float(0.05).hex(),
        "rgb_deranged_changed_bootstrap_lower_min_hex": float(0.10).hex(),
        "value_r2_min_hex": float(0.10).hex(),
        "value_train_baseline_mse_ratio_max_hex": float(0.90).hex(),
        "value_pearson_min_hex": float(0.30).hex(),
        "value_rgb_deranged_mse_ratio_max_hex": float(0.90).hex(),
    }


def _movement_mask(keys_down: Sequence[int]) -> int:
    key_set = set(keys_down)
    return sum((1 << index) for index, key in enumerate(_MOVEMENT_INDICES) if key in key_set)


def _mask_tuple(mask: int) -> tuple[bool, bool, bool, bool]:
    return tuple(bool(mask & (1 << index)) for index in range(4))  # type: ignore[return-value]


def _mode_mask(counter: Counter[int]) -> int:
    if not counter:
        raise RCQInputError("cannot fit a mode from an empty training stratum")
    maximum = max(counter.values())
    candidates = [mask for mask, count in counter.items() if count == maximum]
    return min(candidates, key=_mask_tuple)


@dataclass(frozen=True, slots=True)
class TrainLookupReceipt:
    payload: Mapping[str, object]
    sha256: str

    def __post_init__(self) -> None:
        _hash_string(self.sha256, name="train lookup sha256")
        if not isinstance(self.payload, Mapping):
            raise RCQInputError("train lookup payload must be a mapping")
        if _digest_payload(b"IRENERCQLOOKUP\x01", self.payload) != self.sha256:
            raise RCQInputError("train lookup digest differs from its payload")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))


def fit_rcq_v2_train_lookup(
    registration: RCQRegistration,
) -> TrainLookupReceipt:
    """Fit action and RGB-blind value baselines from registered TRAIN only."""

    if not isinstance(registration, RCQRegistration):
        raise RCQInputError("registration must be an RCQRegistration")
    train = registration.payload["slices"]["train"]  # type: ignore[index]
    assert isinstance(train, Mapping)
    dataset = MovingShapesSequenceDataset(
        MovingShapesDatasetConfig(
            split=DatasetSplit.TRAIN,
            sequence_count=train["sequences"],
            sequence_length=registration.payload["sequence_length"],
            seed_offset=train["local_start"],
            hazard_count=registration.payload["hazard_count"],
            tick_period_ns=registration.payload["tick_period_ns"],
            discount=float.fromhex(str(registration.payload["discount_hex"])),
        )
    )
    if dataset.manifest_sha256 != train["manifest_sha256"]:
        raise RCQInputError("TRAIN dataset manifest differs from registration")
    conditional_actions: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    timestep_actions: dict[int, Counter[int]] = defaultdict(Counter)
    global_actions: Counter[int] = Counter()
    conditional_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    timestep_values: dict[int, list[float]] = defaultdict(list)
    global_values: list[float] = []
    decisions = 0
    burn_in = int(registration.payload["burn_in_steps"])
    for sequence in dataset:
        for time_index, transition in enumerate(sequence.transitions[burn_in:], start=burn_in):
            previous = _movement_mask(transition.observation.previous_control.keys_down)
            target = _movement_mask(transition.action_target.keys_down)
            key = (time_index, previous)
            conditional_actions[key][target] += 1
            timestep_actions[time_index][target] += 1
            global_actions[target] += 1
            conditional_values[key].append(transition.value_target)
            timestep_values[time_index].append(transition.value_target)
            global_values.append(transition.value_target)
            decisions += 1
    if decisions != train["scored_decisions"]:
        raise RCQInputError("TRAIN baseline exposure count differs from registration")
    payload: dict[str, object] = {
        "schema_version": 1,
        "registration_sha256": registration.sha256,
        "train_manifest_sha256": dataset.manifest_sha256,
        "train_decisions": decisions,
        "action_conditional_mode": {
            f"{time_index}:{previous}": _mode_mask(counter)
            for (time_index, previous), counter in sorted(conditional_actions.items())
        },
        "action_timestep_mode": {
            str(time_index): _mode_mask(counter)
            for time_index, counter in sorted(timestep_actions.items())
        },
        "action_global_mode": _mode_mask(global_actions),
        "value_conditional_mean_hex": {
            f"{time_index}:{previous}": (fsum(values) / len(values)).hex()
            for (time_index, previous), values in sorted(conditional_values.items())
        },
        "value_timestep_mean_hex": {
            str(time_index): (fsum(values) / len(values)).hex()
            for time_index, values in sorted(timestep_values.items())
        },
        "value_global_mean_hex": (fsum(global_values) / len(global_values)).hex(),
        "action_tie_break": "lexicographically_smallest_boolean_tuple_W_A_S_D",
        "backoff": "conditional_then_timestep_then_global",
    }
    digest = _digest_payload(b"IRENERCQLOOKUP\x01", payload)
    return TrainLookupReceipt(payload=payload, sha256=digest)


def publish_train_lookup_receipt(
    receipt: TrainLookupReceipt,
    path: str | os.PathLike[str],
    *,
    policy: ResourcePolicy,
) -> None:
    if not isinstance(receipt, TrainLookupReceipt):
        raise RCQInputError("receipt must be a TrainLookupReceipt")
    payload = {**_deep_thaw(receipt.payload), "receipt_sha256": receipt.sha256}
    _publish_no_replace(
        Path(path),
        (_canonical(payload) + "\n").encode("utf-8"),
        policy=policy,
    )


@dataclass(frozen=True, slots=True)
class DonorFrameAssignment:
    recipient_episode_seed: int
    time_index: int
    donor_episode_seed: int
    recipient_rgb_sha256: str
    donor_rgb_sha256: str
    recipient_previous_wasd_mask: int
    donor_previous_wasd_mask: int
    previous_wasd_bit_hamming: int

    def __post_init__(self) -> None:
        if (
            type(self.recipient_episode_seed) is not int
            or not 0 <= self.recipient_episode_seed < (1 << 64)
            or type(self.donor_episode_seed) is not int
            or not 0 <= self.donor_episode_seed < (1 << 64)
        ):
            raise RCQInputError("donor assignment episode seeds must be uint64 integers")
        if type(self.time_index) is not int or not 0 <= self.time_index <= 7:
            raise RCQInputError("donor assignment time_index must be an integer in [0, 7]")
        _hash_string(self.recipient_rgb_sha256, name="recipient RGB sha256")
        _hash_string(self.donor_rgb_sha256, name="donor RGB sha256")
        if self.recipient_rgb_sha256 == self.donor_rgb_sha256:
            raise RCQInputError("donor assignment must replace RGB with a different frame")
        if (
            type(self.recipient_previous_wasd_mask) is not int
            or not 0 <= self.recipient_previous_wasd_mask <= 15
            or type(self.donor_previous_wasd_mask) is not int
            or not 0 <= self.donor_previous_wasd_mask <= 15
            or type(self.previous_wasd_bit_hamming) is not int
        ):
            raise RCQInputError("donor assignment previous-WASD masks are invalid")
        expected_hamming = (
            self.recipient_previous_wasd_mask ^ self.donor_previous_wasd_mask
        ).bit_count()
        if self.previous_wasd_bit_hamming != expected_hamming:
            raise RCQInputError("donor assignment previous-WASD Hamming cost changed")

    def to_dict(self) -> dict[str, object]:
        return {
            "recipient_episode_seed": self.recipient_episode_seed,
            "time_index": self.time_index,
            "donor_episode_seed": self.donor_episode_seed,
            "recipient_rgb_sha256": self.recipient_rgb_sha256,
            "donor_rgb_sha256": self.donor_rgb_sha256,
            "recipient_previous_wasd_mask": self.recipient_previous_wasd_mask,
            "donor_previous_wasd_mask": self.donor_previous_wasd_mask,
            "previous_wasd_bit_hamming": self.previous_wasd_bit_hamming,
        }


@dataclass(frozen=True, slots=True)
class DonorMapping:
    assignments: tuple[DonorFrameAssignment, ...]
    sha256: str

    def __post_init__(self) -> None:
        if type(self.assignments) is not tuple or not self.assignments:
            raise RCQInputError("donor mapping cannot be empty")
        if any(type(item) is not DonorFrameAssignment for item in self.assignments):
            raise RCQInputError("donor mapping contains an incompatible assignment")
        ordered = tuple(
            sorted(
                self.assignments,
                key=lambda item: (item.recipient_episode_seed, item.time_index),
            )
        )
        if ordered != self.assignments:
            raise RCQInputError("donor mapping assignments are not in canonical order")
        keys = {
            (item.recipient_episode_seed, item.time_index)
            for item in self.assignments
        }
        if len(keys) != len(self.assignments):
            raise RCQInputError("donor mapping contains duplicate recipient/time keys")
        _hash_string(self.sha256, name="donor mapping sha256")
        expected = _digest_payload(
            b"IRENERCQDONOR\x01",
            [item.to_dict() for item in self.assignments],
        )
        if self.sha256 != expected:
            raise RCQInputError("donor mapping SHA-256 differs from its assignments")

    def by_recipient_time(self) -> dict[tuple[int, int], DonorFrameAssignment]:
        return {
            (item.recipient_episode_seed, item.time_index): item
            for item in self.assignments
        }

    def assignment_cost_summary(self) -> dict[str, object]:
        grouped: dict[tuple[int, int], int] = defaultdict(int)
        for item in self.assignments:
            grouped[
                (item.recipient_episode_seed, item.donor_episode_seed)
            ] += item.previous_wasd_bit_hamming
        pairs = [
            {
                "recipient_episode_seed": recipient,
                "donor_episode_seed": donor,
                "previous_wasd_bit_hamming": cost,
            }
            for (recipient, donor), cost in sorted(grouped.items())
        ]
        return {
            "cost": "sum_corresponding_previous_wasd_mask_bit_hamming_over_8_frames",
            "donor_matching_seed": DONOR_MATCHING_SEED,
            "pairs": pairs,
            "pair_count": len(pairs),
            "total_previous_wasd_bit_hamming": sum(grouped.values()),
        }


def build_target_blind_donor_mapping(
    recipients: Sequence[MovingShapesSequence],
    donors: Sequence[MovingShapesSequence],
    *,
    seed: int = DONOR_MATCHING_SEED,
) -> DonorMapping:
    """Build a minimum-cost one-to-one sequence permutation without targets."""

    if type(seed) is not int or not 0 <= seed < (1 << 64):
        raise RCQInputError("donor matching seed must be a uint64 integer")
    if not recipients or len(recipients) != len(donors):
        raise RCQInputError("recipient and donor sequence sets must be equal and non-empty")
    if len({sequence.episode_seed for sequence in recipients}) != len(recipients):
        raise RCQInputError("recipient episode seeds must be unique")
    if len({sequence.episode_seed for sequence in donors}) != len(donors):
        raise RCQInputError("donor episode seeds must be unique")
    ordered_recipients = tuple(sorted(recipients, key=lambda item: item.episode_seed))
    ordered_donors = tuple(
        sorted(
            donors,
            key=lambda item: (
                sha256(
                    b"IRENERCQSEQUENCETIE\x01"
                    + seed.to_bytes(8, "big")
                    + item.episode_seed.to_bytes(8, "big")
                ).digest(),
                item.episode_seed,
            ),
        )
    )
    if any(len(sequence.transitions) != 8 for sequence in (*recipients, *donors)):
        raise RCQInputError("donor permutation requires exact eight-frame sequences")

    def signature(sequence: MovingShapesSequence) -> tuple[int, ...]:
        return tuple(
            _movement_mask(transition.observation.previous_control.keys_down)
            for transition in sequence.transitions
        )

    recipient_signatures = tuple(signature(sequence) for sequence in ordered_recipients)
    donor_signatures = tuple(signature(sequence) for sequence in ordered_donors)
    forbidden = 1_000_000
    costs: list[list[int]] = []
    for recipient, recipient_signature in zip(
        ordered_recipients,
        recipient_signatures,
    ):
        recipient_rgb = tuple(
            transition.observation.rgb.sha256 for transition in recipient.transitions
        )
        row: list[int] = []
        for donor, donor_signature in zip(ordered_donors, donor_signatures):
            donor_rgb = tuple(
                transition.observation.rgb.sha256 for transition in donor.transitions
            )
            if any(left == right for left, right in zip(recipient_rgb, donor_rgb)):
                row.append(forbidden)
            else:
                row.append(
                    sum(
                        (left ^ right).bit_count()
                        for left, right in zip(
                            recipient_signature,
                            donor_signature,
                        )
                    )
                )
        costs.append(row)
    permutation = _minimum_cost_permutation(costs=costs, forbidden=forbidden)
    assignments: list[DonorFrameAssignment] = []
    for recipient_index, donor_index in enumerate(permutation):
        recipient = ordered_recipients[recipient_index]
        donor = ordered_donors[donor_index]
        for time_index, (recipient_transition, donor_transition) in enumerate(
            zip(recipient.transitions, donor.transitions)
        ):
            recipient_rgb = recipient_transition.observation.rgb.sha256
            donor_rgb = donor_transition.observation.rgb.sha256
            recipient_previous = _movement_mask(
                recipient_transition.observation.previous_control.keys_down
            )
            donor_previous = _movement_mask(
                donor_transition.observation.previous_control.keys_down
            )
            if recipient_rgb == donor_rgb:
                raise RCQInputError("donor mapping retained an identical RGB frame")
            assignments.append(
                DonorFrameAssignment(
                    recipient_episode_seed=recipient.episode_seed,
                    time_index=time_index,
                    donor_episode_seed=donor.episode_seed,
                    recipient_rgb_sha256=recipient_rgb,
                    donor_rgb_sha256=donor_rgb,
                    recipient_previous_wasd_mask=recipient_previous,
                    donor_previous_wasd_mask=donor_previous,
                    previous_wasd_bit_hamming=(
                        recipient_previous ^ donor_previous
                    ).bit_count(),
                )
            )
    ordered = tuple(
        sorted(assignments, key=lambda item: (item.recipient_episode_seed, item.time_index))
    )
    payload = [item.to_dict() for item in ordered]
    return DonorMapping(
        assignments=ordered,
        sha256=_digest_payload(b"IRENERCQDONOR\x01", payload),
    )


def _minimum_cost_permutation(
    *,
    costs: Sequence[Sequence[int]],
    forbidden: int,
) -> tuple[int, ...]:
    """Solve a deterministic square integer assignment (Hungarian algorithm)."""

    size = len(costs)
    if type(forbidden) is not int or forbidden < 1:
        raise RCQInputError("donor assignment forbidden cost must be a positive integer")
    if size < 1 or any(
        len(row) != size
        or any(
            type(value) is not int or not 0 <= value <= forbidden
            for value in row
        )
        for row in costs
    ):
        raise RCQInputError("donor assignment costs must be a nonempty square integer matrix")
    infinity = forbidden * (size + 1)
    row_potential = [0] * (size + 1)
    column_potential = [0] * (size + 1)
    column_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    for row in range(1, size + 1):
        column_row[0] = row
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            active_row = column_row[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            if delta >= infinity:
                raise RCQInputError("no complete target-blind donor permutation exists")
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[column_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if column_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            column_row[column] = column_row[prior]
            column = prior
            if column == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[column_row[column] - 1] = column - 1
    if any(
        donor < 0 or costs[row][donor] >= forbidden
        for row, donor in enumerate(assignment)
    ):
        raise RCQInputError("no RGB-distinct one-to-one donor permutation exists")
    return tuple(assignment)


@dataclass(frozen=True, slots=True)
class RCQDecisionEvidence:
    recipient_episode_seed: int
    time_index: int
    donor_episode_seed: int
    recipient_rgb_sha256: str
    donor_rgb_sha256: str
    previous_movement_mask: int
    target_movement_mask: int
    target_active_buttons: tuple[int, ...]
    normal_active_buttons: tuple[int, ...]
    deranged_active_buttons: tuple[int, ...]
    target_continuous: tuple[float, ...]
    normal_continuous: tuple[float, ...]
    deranged_continuous: tuple[float, ...]
    value_target: float
    normal_value: float
    deranged_value: float
    has_event: bool
    nonzero_return: bool

    def __post_init__(self) -> None:
        if (
            type(self.recipient_episode_seed) is not int
            or not 0 <= self.recipient_episode_seed < (1 << 64)
        ):
            raise RCQInputError("recipient episode seed must be a uint64 integer")
        if (
            type(self.donor_episode_seed) is not int
            or not 0 <= self.donor_episode_seed < (1 << 64)
        ):
            raise RCQInputError("donor episode seed must be a uint64 integer")
        if type(self.time_index) is not int or not 2 <= self.time_index <= 7:
            raise RCQInputError("final evidence time_index must be in [2, 7]")
        for name in ("recipient_rgb_sha256", "donor_rgb_sha256"):
            _hash_string(getattr(self, name), name=name)
        if self.recipient_rgb_sha256 == self.donor_rgb_sha256:
            raise RCQInputError("recipient and donor RGB frames must differ")
        for name in ("previous_movement_mask", "target_movement_mask"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 15:
                raise RCQInputError(f"{name} must be a four-bit integer")
        for name in (
            "target_active_buttons",
            "normal_active_buttons",
            "deranged_active_buttons",
        ):
            raw = getattr(self, name)
            if type(raw) is not tuple or raw != tuple(sorted(set(raw))):
                raise RCQInputError(f"{name} must be a sorted unique tuple")
            if any(type(index) is not int or index not in _BUTTON_INDEX_SET for index in raw):
                raise RCQInputError(f"{name} contains a non-button control index")
        if _movement_mask(self.target_active_buttons) != self.target_movement_mask:
            raise RCQInputError("target movement mask disagrees with target buttons")
        for name in (
            "target_continuous",
            "normal_continuous",
            "deranged_continuous",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 11:
                raise RCQInputError(f"{name} must contain eleven values")
            if any(type(value) not in {int, float} or not isfinite(float(value)) for value in values):
                raise RCQInputError(f"{name} must contain only finite numbers")
        for name in ("value_target", "normal_value", "deranged_value"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not isfinite(float(value)):
                raise RCQInputError(f"{name} must be finite")
        if type(self.has_event) is not bool or type(self.nonzero_return) is not bool:
            raise RCQInputError("event and nonzero-return flags must be booleans")

    def to_dict(self) -> dict[str, object]:
        return {
            "recipient_episode_seed": self.recipient_episode_seed,
            "time_index": self.time_index,
            "donor_episode_seed": self.donor_episode_seed,
            "recipient_rgb_sha256": self.recipient_rgb_sha256,
            "donor_rgb_sha256": self.donor_rgb_sha256,
            "previous_movement_mask": self.previous_movement_mask,
            "target_movement_mask": self.target_movement_mask,
            "target_active_buttons": list(self.target_active_buttons),
            "normal_active_buttons": list(self.normal_active_buttons),
            "deranged_active_buttons": list(self.deranged_active_buttons),
            "target_continuous_hex": [float(value).hex() for value in self.target_continuous],
            "normal_continuous_hex": [float(value).hex() for value in self.normal_continuous],
            "deranged_continuous_hex": [float(value).hex() for value in self.deranged_continuous],
            "value_target_hex": float(self.value_target).hex(),
            "normal_value_hex": float(self.normal_value).hex(),
            "deranged_value_hex": float(self.deranged_value).hex(),
            "has_event": self.has_event,
            "nonzero_return": self.nonzero_return,
        }


@dataclass(frozen=True, slots=True)
class RCQFinalEvidence:
    registration_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_step: int
    checkpoint_stage_index: int
    recipient_manifest_sha256: str
    donor_manifest_sha256: str
    donor_mapping_sha256: str
    invariance_reference: Mapping[str, object]
    invariance_current: Mapping[str, object]
    decisions: tuple[RCQDecisionEvidence, ...]

    def __post_init__(self) -> None:
        for name in (
            "registration_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "recipient_manifest_sha256",
            "donor_manifest_sha256",
            "donor_mapping_sha256",
        ):
            _hash_string(getattr(self, name), name=name)
        if (
            type(self.checkpoint_step) is not int
            or type(self.checkpoint_stage_index) is not int
            or self.checkpoint_step != FINAL_STEP
            or self.checkpoint_stage_index != FINAL_STAGE_INDEX
        ):
            raise RCQInputError("final evidence must come from stage 1 at step 2048")
        if not isinstance(self.invariance_reference, Mapping) or not isinstance(
            self.invariance_current, Mapping
        ):
            raise RCQInputError("final evidence invariance snapshots must be mappings")
        object.__setattr__(
            self,
            "invariance_reference",
            _deep_freeze(self.invariance_reference),
        )
        object.__setattr__(
            self,
            "invariance_current",
            _deep_freeze(self.invariance_current),
        )
        if (
            type(self.decisions) is not tuple
            or len(self.decisions) != FINAL_DECISIONS
            or any(type(item) is not RCQDecisionEvidence for item in self.decisions)
        ):
            raise RCQInputError("final evidence must contain exactly 3,072 decisions")


@dataclass(frozen=True, slots=True)
class _FinalAuthorization:
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_step: int
    checkpoint_stage_index: int
    development_gate_sha256: str
    final_development_sha256: str
    invariance_report_sha256: str
    source_sha256: str
    runtime_fingerprint_sha256: str
    receipt_root_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "checkpoint_sha256",
            "development_gate_sha256",
            "final_development_sha256",
            "invariance_report_sha256",
            "source_sha256",
            "runtime_fingerprint_sha256",
            "receipt_root_sha256",
        ):
            _hash_string(getattr(self, name), name=name)
        if (
            type(self.checkpoint_step) is not int
            or type(self.checkpoint_stage_index) is not int
            or self.checkpoint_step != FINAL_STEP
            or self.checkpoint_stage_index != FINAL_STAGE_INDEX
        ):
            raise RCQInputError("final authorization requires stage 1 at step 2048")


@dataclass(frozen=True, slots=True)
class RCQFinalReceipt:
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise RCQInputError("final receipt payload must be a mapping")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def canonical_json(self) -> str:
        return _canonical(self.payload)


def _condition_metrics(
    decisions: Sequence[RCQDecisionEvidence],
    *,
    condition: str,
) -> dict[str, object]:
    if condition not in {"normal", "deranged"}:
        raise RCQInputError("condition must be normal or deranged")
    exact = changed_exact = target_active = predicted_active = true_positive = 0
    false_positive = conflicts = full_button_exact = off_support = 0
    target_off_support = continuous_target_nonzero = continuous_violations = 0
    continuous_squares: list[float] = []
    continuous_abs_max = 0.0
    recall_values: list[float] = []
    direction_target = [0, 0, 0, 0]
    direction_predicted = [0, 0, 0, 0]
    direction_true_positive = [0, 0, 0, 0]
    for item in decisions:
        active_buttons = getattr(item, f"{condition}_active_buttons")
        continuous = getattr(item, f"{condition}_continuous")
        prediction = _movement_mask(active_buttons)
        target = item.target_movement_mask
        is_exact = prediction == target
        changed = item.previous_movement_mask != target
        exact += int(is_exact)
        changed_exact += int(changed and is_exact)
        target_active += target.bit_count()
        predicted_active += prediction.bit_count()
        true_positive += (prediction & target).bit_count()
        false_positive += (prediction & ~target & 0xF).bit_count()
        conflicts += int(
            bool((prediction & 0b0101) == 0b0101)
            or bool((prediction & 0b1010) == 0b1010)
        )
        full_button_exact += int(active_buttons == item.target_active_buttons)
        off_support += sum(index not in _MOVEMENT_INDICES for index in active_buttons)
        target_off_support += sum(
            index not in _MOVEMENT_INDICES for index in item.target_active_buttons
        )
        continuous_target_nonzero += sum(value != 0.0 for value in item.target_continuous)
        continuous_violations += sum(abs(value) > CONTINUOUS_DEADZONE for value in continuous)
        continuous_squares.extend(float(value) ** 2 for value in continuous)
        continuous_abs_max = max(
            continuous_abs_max,
            max(abs(float(value)) for value in continuous),
        )
        positives = target.bit_count()
        if positives == 0:
            raise RCQInputError("RCQ final target unexpectedly has no movement key")
        recall_values.append((prediction & target).bit_count() / positives)
        for index in range(4):
            bit = 1 << index
            direction_target[index] += int(bool(target & bit))
            direction_predicted[index] += int(bool(prediction & bit))
            direction_true_positive[index] += int(bool(prediction & target & bit))
    return {
        "decision_count": len(decisions),
        "changed_decision_count": sum(
            item.previous_movement_mask != item.target_movement_mask
            for item in decisions
        ),
        "movement_exact_count": exact,
        "changed_movement_exact_count": changed_exact,
        "movement_target_active_count": target_active,
        "movement_predicted_active_count": predicted_active,
        "movement_true_positive_count": true_positive,
        "movement_false_positive_count": false_positive,
        "opposite_conflict_count": conflicts,
        "full_button_exact_count": full_button_exact,
        "off_support_button_positive_count": off_support,
        "off_support_button_target_active_count": target_off_support,
        "sample_macro_positive_key_recall": fsum(recall_values) / len(recall_values),
        "direction_target_counts": direction_target,
        "direction_predicted_counts": direction_predicted,
        "direction_true_positive_counts": direction_true_positive,
        "direction_recalls": [
            direction_true_positive[index] / direction_target[index]
            for index in range(4)
        ],
        "continuous_value_count": len(continuous_squares),
        "continuous_target_nonzero_count": continuous_target_nonzero,
        "continuous_sse": fsum(continuous_squares),
        "continuous_rmse": sqrt(fsum(continuous_squares) / len(continuous_squares)),
        "continuous_abs_max": continuous_abs_max,
        "continuous_deadzone_violation_count": continuous_violations,
    }


def _value_statistics(
    targets: Sequence[float],
    predictions: Sequence[float],
) -> dict[str, float | int]:
    if not targets or len(targets) != len(predictions):
        raise RCQInputError("value statistics require equal non-empty vectors")
    if any(not isfinite(value) for value in (*targets, *predictions)):
        raise RCQInputError("value statistics contain non-finite values")
    count = len(targets)
    target_mean = fsum(targets) / count
    prediction_mean = fsum(predictions) / count
    target_centered = [value - target_mean for value in targets]
    prediction_centered = [value - prediction_mean for value in predictions]
    target_ss = fsum(value * value for value in target_centered)
    prediction_ss = fsum(value * value for value in prediction_centered)
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    sse = fsum(error * error for error in errors)
    if target_ss <= 0.0 or prediction_ss <= 0.0:
        pearson = 0.0
    else:
        pearson = fsum(
            left * right
            for left, right in zip(target_centered, prediction_centered)
        ) / sqrt(target_ss * prediction_ss)
    return {
        "count": count,
        "target_mean": target_mean,
        "target_std": sqrt(target_ss / count),
        "prediction_mean": prediction_mean,
        "prediction_std": sqrt(prediction_ss / count),
        "mse": sse / count,
        "mae": fsum(abs(error) for error in errors) / count,
        "pearson": pearson,
        "r2": 0.0 if target_ss <= 0.0 else 1.0 - (sse / target_ss),
    }


def _closed_finite_ratio(numerator: float, denominator: float) -> float:
    """Return a finite diagnostic ratio while leaving zero-denominator checks false."""

    if not isfinite(numerator) or not isfinite(denominator) or numerator < 0.0:
        raise RCQInputError("ratio inputs must be finite and nonnegative")
    if denominator <= 0.0:
        return sys.float_info.max
    return numerator / denominator


def _lookup_action(receipt: TrainLookupReceipt, time_index: int, previous: int) -> int:
    payload = receipt.payload
    conditional = payload["action_conditional_mode"]
    timestep = payload["action_timestep_mode"]
    assert isinstance(conditional, Mapping) and isinstance(timestep, Mapping)
    key = f"{time_index}:{previous}"
    if key in conditional:
        return int(conditional[key])
    if str(time_index) in timestep:
        return int(timestep[str(time_index)])
    return int(payload["action_global_mode"])


def _lookup_value(receipt: TrainLookupReceipt, time_index: int, previous: int) -> float:
    payload = receipt.payload
    conditional = payload["value_conditional_mean_hex"]
    timestep = payload["value_timestep_mean_hex"]
    assert isinstance(conditional, Mapping) and isinstance(timestep, Mapping)
    key = f"{time_index}:{previous}"
    if key in conditional:
        return float.fromhex(str(conditional[key]))
    if str(time_index) in timestep:
        return float.fromhex(str(timestep[str(time_index)]))
    return float.fromhex(str(payload["value_global_mean_hex"]))


def _bootstrap_effect_bounds(
    decisions: Sequence[RCQDecisionEvidence],
) -> dict[str, float | int]:
    grouped: dict[int, list[RCQDecisionEvidence]] = defaultdict(list)
    for item in decisions:
        grouped[item.recipient_episode_seed].append(item)
    if len(grouped) != RECIPIENT_SEQUENCES or any(
        len(items) != 6 for items in grouped.values()
    ):
        raise RCQInputError("bootstrap requires 512 complete recipient sequences")
    clusters: list[tuple[int, int, int, int, int, int]] = []
    for seed in sorted(grouped):
        items = grouped[seed]
        normal = deranged = normal_changed = deranged_changed = changed = 0
        for item in items:
            normal_mask = _movement_mask(item.normal_active_buttons)
            deranged_mask = _movement_mask(item.deranged_active_buttons)
            is_changed = item.previous_movement_mask != item.target_movement_mask
            normal_exact = normal_mask == item.target_movement_mask
            deranged_exact = deranged_mask == item.target_movement_mask
            normal += int(normal_exact)
            deranged += int(deranged_exact)
            changed += int(is_changed)
            normal_changed += int(is_changed and normal_exact)
            deranged_changed += int(is_changed and deranged_exact)
        clusters.append((normal, deranged, normal_changed, deranged_changed, 6, changed))
    rng = random.Random(BOOTSTRAP_SEED)
    overall: list[float] = []
    changed_effect: list[float] = []
    count = len(clusters)
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [clusters[rng.randrange(count)] for _ in range(count)]
        decisions_total = sum(item[4] for item in sampled)
        changed_total = sum(item[5] for item in sampled)
        if changed_total == 0:
            raise RCQInputError("a sequence bootstrap replicate has no changed decisions")
        overall.append(
            (sum(item[0] for item in sampled) - sum(item[1] for item in sampled))
            / decisions_total
        )
        changed_effect.append(
            (
                sum(item[2] for item in sampled)
                - sum(item[3] for item in sampled)
            )
            / changed_total
        )
    overall.sort()
    changed_effect.sort()
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "one_sided_confidence": 0.95,
        "sorted_lower_index_zero_based": BOOTSTRAP_LOWER_INDEX,
        "overall_exact_delta_lower": overall[BOOTSTRAP_LOWER_INDEX],
        "changed_exact_delta_lower": changed_effect[BOOTSTRAP_LOWER_INDEX],
    }


def _validate_evidence(
    evidence: RCQFinalEvidence,
    registration: RCQRegistration,
    authorization: _FinalAuthorization,
) -> tuple[RCQDecisionEvidence, ...]:
    if not isinstance(evidence, RCQFinalEvidence):
        raise RCQInputError("evidence factory must return RCQFinalEvidence")
    comparisons = (
        ("registration", evidence.registration_sha256, registration.sha256),
        ("config", evidence.config_sha256, authorization.config_sha256),
        ("checkpoint", evidence.checkpoint_sha256, authorization.checkpoint_sha256),
    )
    for name, observed, expected in comparisons:
        if observed != expected:
            raise RCQInputError(f"final evidence {name} identity differs")
    slices = registration.payload["slices"]
    assert isinstance(slices, Mapping)
    if evidence.recipient_manifest_sha256 != slices["final_recipient"]["manifest_sha256"]:
        raise RCQInputError("recipient TEST manifest differs from registration")
    if evidence.donor_manifest_sha256 != slices["final_donor_only"]["manifest_sha256"]:
        raise RCQInputError("donor-only TEST manifest differs from registration")
    if dict(evidence.invariance_reference) != dict(evidence.invariance_current):
        raise RCQInputError("final checkpoint records failed action/state invariance")
    recipient_start = (2 << 62) | RECIPIENT_OFFSET
    recipient_end = (2 << 62) | (RECIPIENT_OFFSET + RECIPIENT_SEQUENCES)
    donor_start = (2 << 62) | DONOR_OFFSET
    donor_end = (2 << 62) | (DONOR_OFFSET + DONOR_SEQUENCES)
    ordered = tuple(
        sorted(
            evidence.decisions,
            key=lambda item: (item.recipient_episode_seed, item.time_index),
        )
    )
    if ordered != evidence.decisions:
        raise RCQInputError("final decision ledger is not in canonical order")
    expected_keys = {
        (seed, time_index)
        for seed in range(recipient_start, recipient_end)
        for time_index in range(2, 8)
    }
    observed_keys = {
        (item.recipient_episode_seed, item.time_index) for item in ordered
    }
    if observed_keys != expected_keys or len(observed_keys) != len(ordered):
        raise RCQInputError("final decision ledger does not cover the exact recipient slice")
    if any(not donor_start <= item.donor_episode_seed < donor_end for item in ordered):
        raise RCQInputError("final decision ledger uses a donor outside its retired slice")
    if any(any(value != 0.0 for value in item.target_continuous) for item in ordered):
        raise RCQInputError("final continuous targets are not quiescent")
    if any(
        any(index not in _MOVEMENT_INDICES for index in item.target_active_buttons)
        for item in ordered
    ):
        raise RCQInputError("final discrete targets contain an off-support button")
    return ordered


def _evaluate_final_evidence(
    evidence: RCQFinalEvidence,
    *,
    registration: RCQRegistration,
    authorization: _FinalAuthorization,
    lookup: TrainLookupReceipt,
) -> dict[str, object]:
    decisions = _validate_evidence(evidence, registration, authorization)
    if lookup.payload.get("registration_sha256") != registration.sha256:
        raise RCQInputError("train lookup receipt belongs to another registration")
    normal = _condition_metrics(decisions, condition="normal")
    deranged = _condition_metrics(decisions, condition="deranged")
    if normal["decision_count"] != FINAL_DECISIONS:
        raise RCQInputError("final decision count invariant changed")
    changed_decisions = int(normal["changed_decision_count"])
    target_active = int(normal["movement_target_active_count"])
    if changed_decisions <= 0:
        raise RCQInputError("final recipient slice has a zero changed-decision denominator")
    if target_active <= 0:
        raise RCQInputError("final recipient slice has a zero target-active denominator")
    previous_exact = sum(
        item.previous_movement_mask == item.target_movement_mask for item in decisions
    )
    if previous_exact + changed_decisions != FINAL_DECISIONS:
        raise RCQInputError("final previous-control count lattice is inconsistent")

    copy_exact = sum(
        item.previous_movement_mask == item.target_movement_mask for item in decisions
    )
    lookup_predictions = [
        _lookup_action(lookup, item.time_index, item.previous_movement_mask)
        for item in decisions
    ]
    lookup_exact = sum(
        prediction == item.target_movement_mask
        for prediction, item in zip(lookup_predictions, decisions)
    )
    normal_exact = int(normal["movement_exact_count"])
    deranged_exact = int(deranged["movement_exact_count"])
    normal_changed = int(normal["changed_movement_exact_count"])
    deranged_changed = int(deranged["changed_movement_exact_count"])
    action_baseline_rate = max(copy_exact, lookup_exact) / FINAL_DECISIONS
    normal_rate = normal_exact / FINAL_DECISIONS
    deranged_rate = deranged_exact / FINAL_DECISIONS
    changed_normal_rate = normal_changed / changed_decisions
    changed_deranged_rate = deranged_changed / changed_decisions
    bootstrap = _bootstrap_effect_bounds(decisions)

    targets = [float(item.value_target) for item in decisions]
    normal_values = [float(item.normal_value) for item in decisions]
    deranged_values = [float(item.deranged_value) for item in decisions]
    global_values = [
        float.fromhex(str(lookup.payload["value_global_mean_hex"]))
        for _item in decisions
    ]
    timestep_values = [
        float.fromhex(
            str(lookup.payload["value_timestep_mean_hex"][str(item.time_index)])  # type: ignore[index]
        )
        for item in decisions
    ]
    conditional_values = [
        _lookup_value(lookup, item.time_index, item.previous_movement_mask)
        for item in decisions
    ]
    value_normal = _value_statistics(targets, normal_values)
    value_deranged = _value_statistics(targets, deranged_values)
    baseline_stats = {
        "train_global": _value_statistics(targets, global_values),
        "train_timestep": _value_statistics(targets, timestep_values),
        "train_timestep_previous_wasd": _value_statistics(targets, conditional_values),
    }
    best_baseline_mse = min(float(value["mse"]) for value in baseline_stats.values())
    deranged_mse = float(value_deranged["mse"])
    normal_mse = float(value_normal["mse"])
    value_strata: dict[str, object] = {}
    for time_index in range(2, 8):
        selected = [index for index, item in enumerate(decisions) if item.time_index == time_index]
        value_strata[f"timestep_{time_index}"] = _value_statistics(
            [targets[index] for index in selected],
            [normal_values[index] for index in selected],
        )
    for name, predicate in (
        ("event", lambda item: item.has_event),
        ("nonzero_return", lambda item: item.nonzero_return),
    ):
        selected = [index for index, item in enumerate(decisions) if predicate(item)]
        if not selected:
            raise RCQInputError(f"final value stratum {name!r} is empty")
        value_strata[name] = _value_statistics(
            [targets[index] for index in selected],
            [normal_values[index] for index in selected],
        )

    checks = (
        RCQCheck(
            "movement_exact",
            normal_exact / FINAL_DECISIONS >= 0.85,
            normal_exact / FINAL_DECISIONS,
            ">= 0.85 of post-claim decisions",
        ),
        RCQCheck(
            "changed_movement_exact",
            normal_changed / changed_decisions >= 0.60,
            normal_changed / changed_decisions,
            ">= 0.60 of post-claim changed decisions",
        ),
        RCQCheck(
            "sample_macro_positive_recall",
            float(normal["sample_macro_positive_key_recall"]) >= 0.93,
            float(normal["sample_macro_positive_key_recall"]),
            ">= 0.93",
        ),
        RCQCheck(
            "movement_false_positives",
            int(normal["movement_false_positive_count"]) / FINAL_DECISIONS <= 0.10,
            int(normal["movement_false_positive_count"]) / FINAL_DECISIONS,
            "<= 0.10 per post-claim decision",
        ),
        RCQCheck(
            "opposite_conflicts",
            int(normal["opposite_conflict_count"]) / FINAL_DECISIONS <= 0.002,
            int(normal["opposite_conflict_count"]) / FINAL_DECISIONS,
            "<= 0.002 of post-claim decisions",
        ),
        RCQCheck(
            "all_296_buttons_off_support",
            int(normal["off_support_button_positive_count"]) == 0,
            int(normal["off_support_button_positive_count"]),
            "= 0",
        ),
        RCQCheck(
            "continuous_deadzone",
            int(normal["continuous_deadzone_violation_count"]) == 0,
            int(normal["continuous_deadzone_violation_count"]),
            "= 0 values with |output| > 0.05",
        ),
        RCQCheck(
            "state_blind_action_margin",
            normal_rate - action_baseline_rate >= 0.10,
            normal_rate - action_baseline_rate,
            ">= 0.10 over max(copy previous, train lookup)",
        ),
        RCQCheck(
            "rgb_deranged_action_delta",
            normal_rate - deranged_rate >= 0.10,
            normal_rate - deranged_rate,
            ">= 0.10",
        ),
        RCQCheck(
            "rgb_deranged_changed_delta",
            changed_normal_rate - changed_deranged_rate >= 0.20,
            changed_normal_rate - changed_deranged_rate,
            ">= 0.20",
        ),
        RCQCheck(
            "rgb_deranged_action_bootstrap_lower",
            float(bootstrap["overall_exact_delta_lower"]) > 0.05,
            float(bootstrap["overall_exact_delta_lower"]),
            "> 0.05 one-sided 95% sequence-paired bound",
        ),
        RCQCheck(
            "rgb_deranged_changed_bootstrap_lower",
            float(bootstrap["changed_exact_delta_lower"]) > 0.10,
            float(bootstrap["changed_exact_delta_lower"]),
            "> 0.10 one-sided 95% sequence-paired bound",
        ),
        RCQCheck(
            "value_r2",
            float(value_normal["r2"]) >= 0.10,
            float(value_normal["r2"]),
            ">= 0.10",
        ),
        RCQCheck(
            "value_train_baseline_mse",
            best_baseline_mse > 0.0 and normal_mse <= 0.90 * best_baseline_mse,
            _closed_finite_ratio(normal_mse, best_baseline_mse),
            "model MSE / best frozen train baseline MSE <= 0.90",
        ),
        RCQCheck(
            "value_pearson",
            float(value_normal["pearson"]) >= 0.30,
            float(value_normal["pearson"]),
            ">= 0.30",
        ),
        RCQCheck(
            "value_rgb_deranged_mse",
            deranged_mse > 0.0 and normal_mse <= 0.90 * deranged_mse,
            _closed_finite_ratio(normal_mse, deranged_mse),
            "normal MSE / RGB-deranged MSE <= 0.90",
        ),
    )
    ledger_payload = [item.to_dict() for item in decisions]
    return {
        "schema_version": 1,
        "evaluator": FINAL_EVALUATOR_ID,
        "scope": registration.payload["scope"],
        "passed": all(check.passed for check in checks),
        "checks": [check.to_dict() for check in checks],
        "normal_action": normal,
        "deranged_action": deranged,
        "post_claim_denominators": {
            "recipient_sequences": RECIPIENT_SEQUENCES,
            "decisions": FINAL_DECISIONS,
            "changed_decisions": changed_decisions,
            "previous_control_exact_decisions": previous_exact,
            "movement_target_active": target_active,
        },
        "action_baselines": {
            "copy_previous_exact_count": copy_exact,
            "train_timestep_previous_wasd_exact_count": lookup_exact,
        },
        "bootstrap": bootstrap,
        "normal_value": value_normal,
        "deranged_value": value_deranged,
        "value_baselines": baseline_stats,
        "value_strata": value_strata,
        "donor_mapping_sha256": evidence.donor_mapping_sha256,
        "decision_ledger_sha256": _digest_payload(
            b"IRENERCQLEDGER\x01",
            ledger_payload,
        ),
    }


def _authorization_payload(
    authorization: _FinalAuthorization,
    *,
    registration: RCQRegistration,
    lookup: TrainLookupReceipt,
) -> dict[str, object]:
    if not isinstance(authorization, _FinalAuthorization):
        raise RCQInputError("authorization must be internally reconstructed")
    if authorization.config_sha256 != registration.config_sha256:
        raise RCQInputError("final authorization config differs from registration")
    if lookup.payload.get("registration_sha256") != registration.sha256:
        raise RCQInputError("train lookup receipt differs from registration")
    return {
        "schema_version": 1,
        "registration_sha256": registration.sha256,
        "config_sha256": authorization.config_sha256,
        "checkpoint_sha256": authorization.checkpoint_sha256,
        "checkpoint_step": authorization.checkpoint_step,
        "checkpoint_stage_index": authorization.checkpoint_stage_index,
        "development_gate_sha256": authorization.development_gate_sha256,
        "final_development_sha256": authorization.final_development_sha256,
        "invariance_report_sha256": authorization.invariance_report_sha256,
        "train_lookup_sha256": lookup.sha256,
        "source_sha256": authorization.source_sha256,
        "runtime_fingerprint_sha256": authorization.runtime_fingerprint_sha256,
        "receipt_root_sha256": authorization.receipt_root_sha256,
    }


def _reserved_excluded_ranges() -> list[dict[str, object]]:
    return [
        {
            "id": "opened_rcq_recipient_v0",
            "split": "test",
            "local_start": 1_048_576,
            "local_end": 1_049_088,
            "status": "opened_and_retired_excluded",
        },
        {
            "id": "opened_rcq_donor_v0",
            "split": "test",
            "local_start": 1_049_088,
            "local_end": 1_049_600,
            "status": "opened_and_retired_excluded",
        },
        {
            "id": "opened_rcq_guard_v0",
            "split": "test",
            "local_start": 1_049_600,
            "local_end": 1_050_112,
            "status": "retired_design_guard_excluded",
        },
        {
            "id": "first_matched_hazard_1_suite",
            "split": "test",
            "local_start": 2_097_152,
            "local_end": 2_097_408,
            "status": "reserved_first_matched_evaluation_suite_excluded_from_rcq_v2",
        },
        {
            "id": "first_matched_hazard_3_suite",
            "split": "test",
            "local_start": 2_097_408,
            "local_end": 2_097_664,
            "status": "reserved_first_matched_evaluation_suite_excluded_from_rcq_v2",
        },
        {
            "id": "first_matched_hazard_5_suite",
            "split": "test",
            "local_start": 2_097_664,
            "local_end": 2_097_920,
            "status": "reserved_first_matched_evaluation_suite_excluded_from_rcq_v2",
        },
        {
            "id": "rcq_v2_final_recipient",
            "split": "test",
            "local_start": RECIPIENT_OFFSET,
            "local_end": RECIPIENT_OFFSET + RECIPIENT_SEQUENCES,
            "status": "claimed_and_retired",
        },
        {
            "id": "rcq_v2_final_donor",
            "split": "test",
            "local_start": DONOR_OFFSET,
            "local_end": DONOR_OFFSET + DONOR_SEQUENCES,
            "status": "claimed_and_retired",
        },
        {
            "id": "rcq_v2_final_guard",
            "split": "test",
            "local_start": RETIRED_END,
            "local_end": GUARD_END,
            "status": "sealed_unused_excluded",
        },
    ]


def _raw_jsonl(values: Sequence[Mapping[str, object]]) -> bytes:
    if not values:
        raise RCQInputError("a raw final ledger cannot be empty")
    return b"".join(
        (_canonical(dict(value)) + "\n").encode("utf-8") for value in values
    )


def _validate_complete_donor_mapping(
    mapping: DonorMapping,
    evidence: RCQFinalEvidence,
) -> None:
    payload = [item.to_dict() for item in mapping.assignments]
    if _digest_payload(b"IRENERCQDONOR\x01", payload) != mapping.sha256:
        raise RCQInputError("donor mapping digest is inconsistent with its ledger")
    if mapping.sha256 != evidence.donor_mapping_sha256:
        raise RCQInputError("evidence and donor mapping identities differ")
    recipient_start = (2 << 62) | RECIPIENT_OFFSET
    recipient_end = (2 << 62) | (RECIPIENT_OFFSET + RECIPIENT_SEQUENCES)
    donor_start = (2 << 62) | DONOR_OFFSET
    donor_end = (2 << 62) | (DONOR_OFFSET + DONOR_SEQUENCES)
    expected = {
        (seed, time_index)
        for seed in range(recipient_start, recipient_end)
        for time_index in range(8)
    }
    observed = {
        (item.recipient_episode_seed, item.time_index)
        for item in mapping.assignments
    }
    if observed != expected or len(observed) != len(mapping.assignments):
        raise RCQInputError("donor ledger does not cover every recipient RGB frame once")
    if any(
        not donor_start <= item.donor_episode_seed < donor_end
        for item in mapping.assignments
    ):
        raise RCQInputError("donor ledger contains an out-of-slice donor")
    expected_donor_frames = {
        (seed, time_index)
        for seed in range(donor_start, donor_end)
        for time_index in range(8)
    }
    observed_donor_frames = {
        (item.donor_episode_seed, item.time_index)
        for item in mapping.assignments
    }
    if observed_donor_frames != expected_donor_frames:
        raise RCQInputError("donor ledger is not an exact one-to-one frame bijection")
    recipient_donors: dict[int, set[int]] = defaultdict(set)
    for item in mapping.assignments:
        recipient_donors[item.recipient_episode_seed].add(item.donor_episode_seed)
    if (
        len(recipient_donors) != RECIPIENT_SEQUENCES
        or any(len(donors) != 1 for donors in recipient_donors.values())
        or len({next(iter(donors)) for donors in recipient_donors.values()})
        != DONOR_SEQUENCES
    ):
        raise RCQInputError("donor ledger is not a one-to-one sequence permutation")
    recipient_frames: dict[int, dict[int, tuple[int, str]]] = defaultdict(dict)
    donor_frames: dict[int, dict[int, tuple[int, str]]] = defaultdict(dict)
    for item in mapping.assignments:
        recipient_frames[item.recipient_episode_seed][item.time_index] = (
            item.recipient_previous_wasd_mask,
            item.recipient_rgb_sha256,
        )
        donor_frames[item.donor_episode_seed][item.time_index] = (
            item.donor_previous_wasd_mask,
            item.donor_rgb_sha256,
        )
    recipient_seeds = sorted(recipient_frames)
    donor_seeds = sorted(
        donor_frames,
        key=lambda seed: (
            sha256(
                b"IRENERCQSEQUENCETIE\x01"
                + DONOR_MATCHING_SEED.to_bytes(8, "big")
                + seed.to_bytes(8, "big")
            ).digest(),
            seed,
        ),
    )
    forbidden = 1_000_000
    costs: list[list[int]] = []
    for recipient_seed in recipient_seeds:
        row: list[int] = []
        recipient = recipient_frames[recipient_seed]
        for donor_seed in donor_seeds:
            donor = donor_frames[donor_seed]
            if any(recipient[index][1] == donor[index][1] for index in range(8)):
                row.append(forbidden)
            else:
                row.append(
                    sum(
                        (recipient[index][0] ^ donor[index][0]).bit_count()
                        for index in range(8)
                    )
                )
        costs.append(row)
    expected_permutation = _minimum_cost_permutation(
        costs=costs,
        forbidden=forbidden,
    )
    observed_pairing = {
        recipient: next(iter(donors))
        for recipient, donors in recipient_donors.items()
    }
    expected_pairing = {
        recipient_seeds[index]: donor_seeds[donor_index]
        for index, donor_index in enumerate(expected_permutation)
    }
    if observed_pairing != expected_pairing:
        raise RCQInputError("donor ledger is not the preregistered global minimum")
    indexed = mapping.by_recipient_time()
    for item in evidence.decisions:
        assignment = indexed[(item.recipient_episode_seed, item.time_index)]
        if (
            item.donor_episode_seed,
            item.recipient_rgb_sha256,
            item.donor_rgb_sha256,
            item.previous_movement_mask,
        ) != (
            assignment.donor_episode_seed,
            assignment.recipient_rgb_sha256,
            assignment.donor_rgb_sha256,
            assignment.recipient_previous_wasd_mask,
        ):
            raise RCQInputError("decision ledger differs from its donor assignment")


def _safe_receipt_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value)
    absolute = root.absolute()
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
                raise RCQInputError("receipt_root cannot traverse a link or reparse point")
    missing: list[Path] = []
    probe = absolute
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    absolute.mkdir(parents=True, exist_ok=True)
    resolved = absolute.resolve(strict=True)
    metadata = resolved.stat(follow_symlinks=False)
    reparse = getattr(metadata, "st_file_attributes", 0) & getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0,
    )
    if not resolved.is_dir() or resolved.is_symlink() or reparse:
        raise RCQInputError("receipt_root must resolve to a non-symlink directory")
    if any(resolved.iterdir()):
        raise RCQInputError("receipt_root must be a new or empty exclusively owned directory")
    for created in reversed(missing):
        _fsync_directory(created)
        _fsync_directory(created.parent)
    return resolved


def _fsync_directory(path: Path) -> None:
    """Make linked artifact names durable on POSIX; Windows has no directory fd."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(
    path: Path,
    content: bytes,
    *,
    policy: ResourcePolicy,
) -> None:
    policy.require(Capability.ARTIFACT_WRITE)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "BOOTSTRAP_LOWER_INDEX",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONTINUOUS_DEADZONE",
    "DONOR_OFFSET",
    "DONOR_MATCHING_SEED",
    "DONOR_SEQUENCES",
    "DonorFrameAssignment",
    "DonorMapping",
    "FINAL_DECISIONS",
    "FINAL_EVALUATOR_ID",
    "FINAL_RANGE_CLAIM_PROTOCOL",
    "FINAL_STAGE_INDEX",
    "FINAL_STEP",
    "FUTURE_CAMPAIGN_OFFSET",
    "GUARD_END",
    "RCQRegistration",
    "RECIPIENT_OFFSET",
    "RECIPIENT_SEQUENCES",
    "RETIRED_END",
    "TrainLookupReceipt",
    "build_target_blind_donor_mapping",
    "fit_rcq_v2_train_lookup",
    "load_rcq_v2_registration",
    "publish_train_lookup_receipt",
]
