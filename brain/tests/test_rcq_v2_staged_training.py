from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
from typing import Iterator, Mapping, Sequence
import unittest
from unittest.mock import patch

from irene_brain.data import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequenceDataset,
)
from irene_brain.data.moving_shapes_dataset import _overlaps_rcq_v2_sealed_test_range
from irene_brain.training.checkpoint import (
    TrainerCursor,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
)
from irene_brain.training.batches import MovingShapesBatchSource
from irene_brain.training.config import (
    DatasetConfig,
    DeterminismConfig,
    LoggingConfig,
    ObjectiveConfig,
    OptimizationConfig,
    PrecisionConfig,
    ResourceConfig,
    RunConfig,
    TrainingConfig,
    TrainingStageConfig,
    load_training_config,
)
from irene_brain.training.protocol import StagedTrainingSystem, TrainingStepResult
from irene_brain.training.metrics import MetricRecord
from irene_brain.training.trainer import Trainer
from irene_brain.evaluation.rcq_v2 import (
    DEVELOPMENT_CHANGED,
    DEVELOPMENT_PREVIOUS_EXACT,
    DEVELOPMENT_SAMPLES,
    DEVELOPMENT_TARGET_ACTIVE,
    RCQCheck,
    RCQDevelopmentReport,
    RCQInputError,
    RCQValueDevelopmentReport,
    VALUE_ABSOLUTE_MAX_MSE,
    VALUE_DEVELOPMENT_TARGET_VARIANCE,
    VALUE_DEVELOPMENT_GATE_ID,
    VALUE_IMPROVEMENT_RATIO,
    VALUE_TRAIN_CONDITIONAL_BASELINE_MSE,
    evaluate_rcq_v2_development,
    evaluate_rcq_v2_value_development,
)
from irene_brain.types import HidKey


try:
    import torch
except ModuleNotFoundError:  # The dependency-free test environment remains supported.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.training.factory import build_smoke_model
    from irene_brain.training.objective import LossOutput, ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem


_NULL_SHA256 = "0" * 64
_CODE_SHA256 = "c" * 64
_DATA_SHA256 = "d" * 64
_VALUE_PARAMETERS = (
    "model.value_per_thought.bias",
    "model.value_per_thought.weight",
)
_BRAIN_ROOT = Path(__file__).resolve().parents[1]


def _stage_config(
    *,
    final_step: int = 4,
    transition_gate: str = "none",
    completion_gate: str = "none",
    validation_batches: int = 1,
) -> TrainingConfig:
    stage_zero = TrainingStageConfig(
        index=0,
        name="joint",
        start_optimizer_step=0,
        end_optimizer_step=1,
        trainable_parameters=("*",),
        reset_optimizer=True,
        learning_rate=1e-2,
        weight_decay=1e-2,
        max_gradient_norm=1.0,
        warmup_steps=0,
        scheduler_kind="constant_after_warmup",
        action_weight=1.0,
        value_weight=0.25,
        world_weight=0.0,
        diversity_weight=0.0,
        transition_gate=transition_gate,
        completion_gate="none",
        invariance_audit="none",
    )
    stage_one = TrainingStageConfig(
        index=1,
        name="value-only",
        start_optimizer_step=1,
        end_optimizer_step=final_step,
        trainable_parameters=_VALUE_PARAMETERS,
        reset_optimizer=True,
        learning_rate=3e-3,
        weight_decay=0.0,
        max_gradient_norm=1.0,
        warmup_steps=0,
        scheduler_kind="constant_after_warmup",
        action_weight=0.0,
        value_weight=1.0,
        world_weight=0.0,
        diversity_weight=0.0,
        transition_gate="none",
        completion_gate=completion_gate,
        invariance_audit="none",
    )
    return TrainingConfig(
        schema_version=3,
        run=RunConfig(
            name="schema3-unit",
            seed=1702,
            model_factory="irene_brain.training.factory:build_smoke_model",
            max_optimizer_steps=final_step,
        ),
        dataset=DatasetConfig(
            kind="moving_shapes",
            train_sequences=4,
            validation_sequences=max(2, validation_batches),
            test_sequences=2,
            sequence_length=2,
            burn_in_steps=1,
            seed_offset=0,
            hazard_count=1,
            tick_period_ns=33_333_333,
            discount=0.99,
        ),
        optimization=OptimizationConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=stage_zero.learning_rate,
            weight_decay=stage_zero.weight_decay,
            max_gradient_norm=stage_zero.max_gradient_norm,
            warmup_steps=stage_zero.warmup_steps,
            scheduler_kind=stage_zero.scheduler_kind,
        ),
        precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
        determinism=DeterminismConfig(
            enabled=True,
            num_workers=0,
            compile_model=False,
        ),
        logging=LoggingConfig(
            log_every_steps=1,
            evaluate_every_steps=1,
            validation_batches=validation_batches,
            checkpoint_every_steps=1,
            keep_last_checkpoints=4,
        ),
        resources=ResourceConfig(
            allow_gpu=False,
            allow_capture=False,
            allow_hid_output=False,
            allow_background_threads=False,
            allow_network=False,
            allow_subprocess=False,
            write_artifacts=True,
            cpu_threads=1,
        ),
        objective=ObjectiveConfig(
            action_weight=stage_zero.action_weight,
            value_weight=stage_zero.value_weight,
            world_weight=stage_zero.world_weight,
            diversity_weight=stage_zero.diversity_weight,
        ),
        stages=(stage_zero, stage_one),
    )


def _rcq_cpu_config() -> TrainingConfig:
    """Use the frozen gate identity with only the runtime resource set made CPU-safe."""

    registered = load_training_config(
        _BRAIN_ROOT / "configs" / "training" / "dgx-rcq-v2-reference.toml"
    )
    return replace(
        registered,
        precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
        resources=replace(
            registered.resources,
            allow_gpu=False,
            cpu_threads=1,
        ),
    )


def _gate_metrics(
    *,
    samples: int = DEVELOPMENT_SAMPLES,
    target_active: int = DEVELOPMENT_TARGET_ACTIVE,
    changed: int = DEVELOPMENT_CHANGED,
    previous_exact: int = DEVELOPMENT_PREVIOUS_EXACT,
    exact: int = 1_300,
    changed_exact: int = 300,
    false_positives: int = 100,
    conflicts: int = 2,
    value_loss: float = 0.30,
) -> dict[str, float]:
    return {
        "movement_target_active_count": target_active / samples,
        "movement_changed_samples_per_sample": changed / samples,
        "previous_control_movement_exact_match": previous_exact / samples,
        "movement_exact_match": exact / samples,
        "movement_changed_exact_matches_per_sample": changed_exact / samples,
        "positive_key_recall": 0.95,
        "movement_false_positive_count": false_positives / samples,
        "movement_opposite_conflict_rate": conflicts / samples,
        "non_movement_key_false_positive_count": 0.0,
        "off_support_button_positive_count": 0.0,
        "off_support_button_target_active_count": 0.0,
        "continuous_target_nonzero_count": 0.0,
        "continuous_action_squared_magnitude": 0.0,
        "continuous_action_outside_0_05_count": 0.0,
        "total_loss": 0.25,
        "value_loss": value_loss,
    }


def _gate_result(
    *,
    passed: bool = True,
    value_loss: float = 0.30,
) -> TrainingStepResult:
    metrics = _gate_metrics(
        exact=1_300 if passed else 1_228,
        value_loss=value_loss,
    )
    return TrainingStepResult(loss=0.25, metrics=metrics, samples=DEVELOPMENT_SAMPLES)


class _TinyBatchSource:
    def __init__(self, *, validation_batches: int = 1) -> None:
        self.validation_batches = validation_batches
        self.requested_splits: list[str] = []

    @property
    def manifest_sha256(self) -> str:
        return _DATA_SHA256

    def batches_per_epoch(self, *, split: str, batch_size: int) -> int:
        if batch_size != 1:
            raise ValueError("tiny source requires batch_size=1")
        return self.validation_batches if split == "validation" else 4

    def iter_batches(
        self,
        *,
        split: str,
        epoch: int,
        start_batch: int,
        batch_size: int,
        max_batches: int | None = None,
    ) -> Iterator[object]:
        self.requested_splits.append(split)
        total = self.batches_per_epoch(split=split, batch_size=batch_size)
        stop = total if max_batches is None else min(total, start_batch + max_batches)
        for batch_index in range(start_batch, stop):
            yield SimpleNamespace(
                sample_count=1,
                split=split,
                epoch=epoch,
                batch_index=batch_index,
            )


class _StagedSystemDouble:
    """Small protocol-complete double for foreground trainer boundary tests."""

    def __init__(
        self,
        config: TrainingConfig,
        validation_results: Sequence[TrainingStepResult],
    ) -> None:
        self.config = config
        self.validation_results = tuple(validation_results)
        self.validation_index = 0
        self.train_calls = 0
        self._active_stage_index = 0
        self._optimizer_parameter_names = ("model.joint",)
        self.entry_gate = ("none", _NULL_SHA256, True, 0)
        self.completion_gate: tuple[str, str, bool | None, int] = (
            "none",
            _NULL_SHA256,
            None,
            0,
        )

    @property
    def runtime_fingerprint(self) -> Mapping[str, str | int | bool]:
        return {"runtime": "schema3-double-cpu", "world_size": 1}

    @property
    def active_stage_index(self) -> int:
        return self._active_stage_index

    @property
    def optimizer_parameter_names(self) -> tuple[str, ...]:
        return self._optimizer_parameter_names

    def train_optimizer_step(
        self,
        microbatches: Sequence[object],
    ) -> TrainingStepResult:
        self.train_calls += 1
        return TrainingStepResult(
            loss=float(self.train_calls),
            metrics={"fake_train": float(self.train_calls)},
            samples=sum(int(getattr(batch, "sample_count")) for batch in microbatches),
        )

    def evaluate_batch(self, _batch: object) -> TrainingStepResult:
        if self.validation_index >= len(self.validation_results):
            raise RuntimeError("the test supplied too few validation results")
        result = self.validation_results[self.validation_index]
        self.validation_index += 1
        return result

    def checkpoint_state(self) -> Mapping[str, object]:
        return {"train_calls": self.train_calls}

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        train_calls = state.get("train_calls")
        if type(train_calls) is not int:
            raise ValueError("invalid train call count")
        self.train_calls = train_calls

    def capture_rng_state(self) -> Mapping[str, object]:
        return {"counter": self.train_calls}

    def restore_rng_state(self, _state: Mapping[str, object]) -> None:
        return None

    def transition_to_stage(
        self,
        stage_index: int,
        *,
        invariance_batches: Sequence[object],
        entry_gate_report_sha256: str,
        entry_gate_passed: bool,
        entry_gate_step: int,
    ) -> Mapping[str, object]:
        if stage_index != self._active_stage_index + 1 or entry_gate_passed is not True:
            raise ValueError("invalid stage transition")
        prior = self.config.stages[self._active_stage_index]
        self.entry_gate = (
            prior.transition_gate,
            entry_gate_report_sha256,
            True,
            entry_gate_step,
        )
        self._active_stage_index = stage_index
        self._optimizer_parameter_names = _VALUE_PARAMETERS
        stage = self.config.stages[stage_index]
        self.completion_gate = (stage.completion_gate, _NULL_SHA256, None, 0)
        return self.stage_checkpoint_state(global_optimizer_step=entry_gate_step)

    def prepare_stage_for_resume(self, stage_state: Mapping[str, object]) -> None:
        stage_index = stage_state.get("stage_index")
        if type(stage_index) is not int:
            raise ValueError("invalid stage index")
        self._active_stage_index = stage_index

    def validate_stage_invariance(
        self,
        *,
        invariance_batches: Sequence[object],
    ) -> Mapping[str, object]:
        stage = self.config.stages[self._active_stage_index]
        if stage.invariance_audit != "none" and not invariance_batches:
            raise ValueError("an active audit requires its deterministic batches")
        return {
            "schema_version": 1,
            "stage_index": stage.index,
            "audit": stage.invariance_audit,
            "passed": True,
        }

    def bind_completion_gate(
        self,
        *,
        report_sha256: str,
        passed: bool,
        global_optimizer_step: int,
    ) -> None:
        if type(passed) is not bool:
            raise ValueError("completion status must be boolean")
        stage = self.config.stages[self._active_stage_index]
        self.completion_gate = (
            stage.completion_gate,
            report_sha256,
            passed,
            global_optimizer_step,
        )

    def stage_checkpoint_state(
        self,
        *,
        global_optimizer_step: int,
    ) -> Mapping[str, object]:
        stage = self.config.stages[self._active_stage_index]
        entry_id, entry_sha, entry_passed, entry_step = self.entry_gate
        completion_id, completion_sha, completion_passed, completion_step = (
            self.completion_gate
        )
        names_json = json.dumps(
            list(self._optimizer_parameter_names),
            separators=(",", ":"),
        ).encode("ascii")
        return {
            "schema_version": 1,
            "stage_index": stage.index,
            "stage_name": stage.name,
            "stage_start_optimizer_step": stage.start_optimizer_step,
            "stage_end_optimizer_step": stage.end_optimizer_step,
            "stage_local_optimizer_step": (
                global_optimizer_step - stage.start_optimizer_step
            ),
            "stage_config_sha256": self.config.config_sha256,
            "trainable_parameter_names": list(stage.trainable_parameters),
            "optimizer_parameter_names": list(self._optimizer_parameter_names),
            "optimizer_parameter_names_sha256": sha256(names_json).hexdigest(),
            "optimizer_reset": stage.reset_optimizer,
            "invariance_audit": stage.invariance_audit,
            "invariance_reference": None,
            "invariance_current": None,
            "entry_gate_id": entry_id,
            "entry_gate_report_sha256": entry_sha,
            "entry_gate_passed": entry_passed,
            "entry_gate_step": entry_step,
            "completion_gate_id": completion_id,
            "completion_gate_report_sha256": completion_sha,
            "completion_gate_passed": completion_passed,
            "completion_gate_step": completion_step,
        }

    def restore_model_for_evaluation(
        self,
        system_state: Mapping[str, object],
        stage_state: Mapping[str, object],
    ) -> None:
        self.restore_checkpoint_state(system_state)
        self.prepare_stage_for_resume(stage_state)


if torch is not None:

    class _TinyStageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Linear(1, 2)
            self.value_per_thought = torch.nn.Linear(2, 1)


    class _TinyStageObjective(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _TinyStageModel()
            self.set_loss_weights(
                action_weight=1.0,
                value_weight=0.25,
                world_weight=0.0,
                diversity_weight=0.0,
            )

        def set_loss_weights(
            self,
            *,
            action_weight: float,
            value_weight: float,
            world_weight: float,
            diversity_weight: float,
        ) -> None:
            self.action_weight = float(action_weight)
            self.value_weight = float(value_weight)
            self.world_weight = float(world_weight)
            self.diversity_weight = float(diversity_weight)

        def forward(self, batch: object) -> LossOutput:
            parameter = next(self.parameters())
            feature = float(getattr(batch, "feature", 0.75))
            target = float(getattr(batch, "target", -0.25))
            inputs = parameter.new_tensor([[feature]])
            hidden = torch.tanh(self.model.trunk(inputs))
            value = self.model.value_per_thought(hidden)
            action_loss = hidden.square().mean() + 0.1 * value.square().mean()
            value_loss = (value - target).square().mean()
            loss = self.action_weight * action_loss + self.value_weight * value_loss
            return LossOutput(
                loss=loss,
                metrics={"tiny_value_loss": value_loss},
                samples=int(getattr(batch, "sample_count")),
            )


def _tiny_batch() -> object:
    return SimpleNamespace(sample_count=1, feature=0.75, target=-0.25)


def _assert_nested_equal(
    case: unittest.TestCase,
    left: object,
    right: object,
) -> None:
    if torch is not None and isinstance(left, torch.Tensor):
        case.assertIsInstance(right, torch.Tensor)
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, Mapping):
        case.assertIsInstance(right, Mapping)
        assert isinstance(right, Mapping)
        case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertIsInstance(right, type(left))
        assert isinstance(right, (list, tuple))
        case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(case, left_item, right_item)
    else:
        case.assertEqual(left, right)


class RCQV2CountLatticeTests(unittest.TestCase):
    def test_static_guard_identifies_sealed_rcq_v2_test_range_overlap(
        self,
    ) -> None:
        def config(*, split: DatasetSplit, start: int, count: int = 1) -> object:
            return MovingShapesDatasetConfig(
                split=split,
                sequence_count=count,
                sequence_length=8,
                seed_offset=start,
                hazard_count=3,
                tick_period_ns=16_666_667,
                discount=0.99,
            )

        for name, start, count, expected in (
            ("recipient", 3_145_728, 1, True),
            ("donor", 3_146_240, 1, True),
            ("guard", 3_146_752, 1, True),
            ("overlap", 3_145_727, 2, True),
            ("adjacent", 3_147_264, 1, False),
        ):
            with self.subTest(name=name):
                self.assertIs(
                    _overlaps_rcq_v2_sealed_test_range(
                        config(split=DatasetSplit.TEST, start=start, count=count)
                    ),
                    expected,
                )

        self.assertFalse(
            _overlaps_rcq_v2_sealed_test_range(
                config(split=DatasetSplit.TRAIN, start=3_145_728)
            )
        )

    def test_report_dataclasses_reject_ambiguous_values_and_normalize_zero(
        self,
    ) -> None:
        normalized = RCQCheck("check", True, -0.0, "= 0")
        self.assertEqual(normalized.observed, 0.0)
        assert isinstance(normalized.observed, float)
        self.assertEqual(normalized.observed.hex(), "0x0.0p+0")

        invalid_checks = (
            {"name": "", "passed": True, "observed": 0, "requirement": "= 0"},
            {"name": "check", "passed": 1, "observed": 0, "requirement": "= 0"},
            {"name": "check", "passed": True, "observed": True, "requirement": "= 0"},
            {"name": "check", "passed": True, "observed": math.inf, "requirement": "= 0"},
            {"name": "check", "passed": True, "observed": math.nan, "requirement": "= 0"},
            {"name": "check", "passed": True, "observed": 0, "requirement": ""},
        )
        for values in invalid_checks:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RCQCheck(**values)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            RCQDevelopmentReport(
                checks=(object(),),  # type: ignore[arg-type]
                metrics={"value_loss": 0.0},
            )
        with self.assertRaises(RCQInputError):
            RCQDevelopmentReport(
                checks=(normalized,),
                metrics={"value_loss": math.inf},
            )

        entry = evaluate_rcq_v2_development(_gate_result(value_loss=-0.0))
        self.assertEqual(entry.metrics["value_loss"].hex(), "0x0.0p+0")
        entry_digest = sha256(
            (entry.canonical_json + "\n").encode("utf-8")
        ).hexdigest()
        report = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=-0.0),
            entry_report=entry,
            entry_report_sha256=entry_digest,
        )
        rebuilt = RCQValueDevelopmentReport(
            checks=report.checks,
            metrics={**report.metrics, "value_loss": -0.0},
            entry_gate_report_sha256=entry_digest,
            entry_value_loss=-0.0,
        )
        self.assertEqual(rebuilt.entry_value_loss.hex(), "0x0.0p+0")
        self.assertEqual(rebuilt.metrics["value_loss"].hex(), "0x0.0p+0")
        self.assertEqual(rebuilt.canonical_json, report.canonical_json)

    def test_value_completion_gate_binds_entry_digest_and_both_thresholds(self) -> None:
        entry = evaluate_rcq_v2_development(
            _gate_result(value_loss=0.40)
        )
        entry_digest = sha256(
            (entry.canonical_json + "\n").encode("utf-8")
        ).hexdigest()
        passing = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=0.30),
            entry_report=entry,
            entry_report_sha256=entry_digest,
        )
        self.assertTrue(passing.passed)
        self.assertEqual(passing.to_dict()["gate"], VALUE_DEVELOPMENT_GATE_ID)
        self.assertEqual(
            passing.to_dict()["entry_gate"]["report_sha256"],
            entry_digest,
        )

        absolute_failure = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=VALUE_ABSOLUTE_MAX_MSE + 1e-6),
            entry_report=entry,
            entry_report_sha256=entry_digest,
        )
        self.assertFalse(absolute_failure.passed)
        relative_entry = evaluate_rcq_v2_development(
            _gate_result(value_loss=0.32)
        )
        relative_digest = sha256(
            (relative_entry.canonical_json + "\n").encode("utf-8")
        ).hexdigest()
        relative_failure = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=0.30),
            entry_report=relative_entry,
            entry_report_sha256=relative_digest,
        )
        self.assertFalse(relative_failure.passed)
        with self.assertRaisesRegex(RCQInputError, "supplied SHA-256"):
            evaluate_rcq_v2_value_development(
                _gate_result(value_loss=0.30),
                entry_report=entry,
                entry_report_sha256="f" * 64,
            )

    def test_value_completion_exact_absolute_and_relative_thresholds_pass(self) -> None:
        absolute_entry = evaluate_rcq_v2_development(
            _gate_result(value_loss=1.0)
        )
        absolute_digest = sha256(
            (absolute_entry.canonical_json + "\n").encode("utf-8")
        ).hexdigest()
        absolute = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=VALUE_ABSOLUTE_MAX_MSE),
            entry_report=absolute_entry,
            entry_report_sha256=absolute_digest,
        )
        absolute_checks = {check.name: check for check in absolute.checks}
        self.assertTrue(absolute.passed)
        self.assertTrue(absolute_checks["value_loss_absolute"].passed)
        self.assertEqual(
            absolute_checks["value_loss_absolute"].observed,
            VALUE_ABSOLUTE_MAX_MSE,
        )
        absolute_just_over = evaluate_rcq_v2_value_development(
            _gate_result(
                value_loss=math.nextafter(VALUE_ABSOLUTE_MAX_MSE, math.inf)
            ),
            entry_report=absolute_entry,
            entry_report_sha256=absolute_digest,
        )
        absolute_just_over_checks = {
            check.name: check for check in absolute_just_over.checks
        }
        self.assertFalse(absolute_just_over.passed)
        self.assertFalse(
            absolute_just_over_checks["value_loss_absolute"].passed
        )
        self.assertTrue(
            absolute_just_over_checks["value_loss_entry_improvement"].passed
        )
        self.assertTrue(absolute_just_over_checks["value_development_r2"].passed)

        relative_entry_loss = 0.32
        relative_threshold = VALUE_IMPROVEMENT_RATIO * relative_entry_loss
        relative_entry = evaluate_rcq_v2_development(
            _gate_result(value_loss=relative_entry_loss)
        )
        relative_digest = sha256(
            (relative_entry.canonical_json + "\n").encode("utf-8")
        ).hexdigest()
        relative = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=relative_threshold),
            entry_report=relative_entry,
            entry_report_sha256=relative_digest,
        )
        relative_checks = {check.name: check for check in relative.checks}
        self.assertTrue(relative.passed)
        self.assertTrue(relative_checks["value_loss_entry_improvement"].passed)
        self.assertEqual(
            relative_checks["value_loss_entry_improvement"].observed,
            relative_threshold,
        )
        relative_just_over = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=math.nextafter(relative_threshold, math.inf)),
            entry_report=relative_entry,
            entry_report_sha256=relative_digest,
        )
        relative_just_over_checks = {
            check.name: check for check in relative_just_over.checks
        }
        self.assertFalse(relative_just_over.passed)
        self.assertTrue(relative_just_over_checks["value_loss_absolute"].passed)
        self.assertFalse(
            relative_just_over_checks["value_loss_entry_improvement"].passed
        )
        self.assertTrue(relative_just_over_checks["value_development_r2"].passed)

    def test_value_completion_preserves_action_gate_failure(self) -> None:
        entry = evaluate_rcq_v2_development(_gate_result(value_loss=0.40))
        entry_digest = sha256(
            (entry.canonical_json + "\n").encode("utf-8")
        ).hexdigest()
        report = evaluate_rcq_v2_value_development(
            _gate_result(passed=False, value_loss=0.30),
            entry_report=entry,
            entry_report_sha256=entry_digest,
        )
        checks = {check.name: check for check in report.checks}
        self.assertFalse(report.passed)
        self.assertFalse(checks["movement_exact"].passed)
        self.assertTrue(checks["value_loss_absolute"].passed)
        self.assertTrue(checks["value_loss_entry_improvement"].passed)
        self.assertTrue(checks["value_development_r2"].passed)

    def test_max_float_value_loss_is_a_receiptable_scientific_failure(self) -> None:
        entry = evaluate_rcq_v2_development(_gate_result(value_loss=0.40))
        entry_digest = sha256(
            (entry.canonical_json + "\n").encode("utf-8")
        ).hexdigest()
        report = evaluate_rcq_v2_value_development(
            _gate_result(value_loss=sys.float_info.max),
            entry_report=entry,
            entry_report_sha256=entry_digest,
        )
        checks = {check.name: check for check in report.checks}
        self.assertFalse(report.passed)
        self.assertEqual(
            checks["value_development_r2"].observed,
            -sys.float_info.max,
        )
        self.assertFalse(checks["value_development_r2"].passed)
        self.assertIn('"passed":false', report.canonical_json)

    def test_value_loss_missing_negative_nonfinite_and_malformed_are_rejected(
        self,
    ) -> None:
        missing_metrics = _gate_metrics()
        del missing_metrics["value_loss"]
        with self.assertRaisesRegex(RCQInputError, "missing.*value_loss"):
            evaluate_rcq_v2_development(
                TrainingStepResult(
                    loss=0.0,
                    metrics=missing_metrics,
                    samples=DEVELOPMENT_SAMPLES,
                )
            )

        with self.assertRaisesRegex(RCQInputError, "value_loss must be nonnegative"):
            evaluate_rcq_v2_development(_gate_result(value_loss=-0.01))

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                _gate_result(value_loss=value)

        for value in (True, "not-a-number"):
            malformed = _gate_metrics()
            malformed["value_loss"] = value  # type: ignore[assignment]
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                TrainingStepResult(
                    loss=0.0,
                    metrics=malformed,
                    samples=DEVELOPMENT_SAMPLES,
                )

    def test_registered_fixture_passes_and_threshold_failure_is_not_input_error(
        self,
    ) -> None:
        passing = evaluate_rcq_v2_development(_gate_result(passed=True))
        failing = evaluate_rcq_v2_development(_gate_result(passed=False))

        self.assertTrue(passing.passed)
        self.assertFalse(failing.passed)
        self.assertFalse(
            next(check for check in failing.checks if check.name == "movement_exact").passed
        )

    def test_fractional_count_and_impossible_cross_count_are_rejected(self) -> None:
        fractional = _gate_metrics()
        fractional["movement_false_positive_count"] = 100.5 / DEVELOPMENT_SAMPLES
        impossible = _gate_metrics(exact=1_300, changed_exact=1_301)

        with self.assertRaisesRegex(RCQInputError, "count lattice"):
            evaluate_rcq_v2_development(
                TrainingStepResult(
                    loss=0.0,
                    metrics=fractional,
                    samples=DEVELOPMENT_SAMPLES,
                )
            )
        with self.assertRaisesRegex(RCQInputError, "impossible"):
            evaluate_rcq_v2_development(
                TrainingStepResult(
                    loss=0.0,
                    metrics=impossible,
                    samples=DEVELOPMENT_SAMPLES,
                )
            )

    def test_weighted_batch_aggregation_lands_on_the_registered_count_lattice(
        self,
    ) -> None:
        config = _stage_config(final_step=2, validation_batches=2)
        first = TrainingStepResult(
            loss=0.2,
            metrics=_gate_metrics(
                samples=768,
                target_active=1_137,
                changed=233,
                previous_exact=535,
                exact=650,
                changed_exact=150,
                false_positives=50,
                conflicts=1,
            ),
            samples=768,
        )
        second = TrainingStepResult(
            loss=0.3,
            metrics=_gate_metrics(
                samples=768,
                target_active=1_138,
                changed=234,
                previous_exact=534,
                exact=650,
                changed_exact=150,
                false_positives=50,
                conflicts=1,
            ),
            samples=768,
        )
        source = _TinyBatchSource(validation_batches=2)
        system = _StagedSystemDouble(config, (first, second))

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                config,
                system,
                source,
                directory,
                code_sha256=_CODE_SHA256,
            )
            aggregate = trainer.evaluate(split="validation")

        self.assertEqual(aggregate.samples, DEVELOPMENT_SAMPLES)
        self.assertTrue(evaluate_rcq_v2_development(aggregate).passed)


class Schema3StrictnessTests(unittest.TestCase):
    def test_value_baseline_constants_regenerate_from_train_and_development_only(
        self,
    ) -> None:
        config = load_training_config(
            _BRAIN_ROOT / "configs" / "training" / "dgx-rcq-v2-reference.toml"
        )

        def dataset(split: DatasetSplit, count: int) -> MovingShapesSequenceDataset:
            return MovingShapesSequenceDataset(
                MovingShapesDatasetConfig(
                    split=split,
                    sequence_count=count,
                    sequence_length=config.dataset.sequence_length,
                    seed_offset=config.dataset.seed_offset,
                    hazard_count=config.dataset.hazard_count,
                    tick_period_ns=config.dataset.tick_period_ns,
                    discount=config.dataset.discount,
                )
            )

        movement_bits = {
            int(HidKey.W): 1,
            int(HidKey.A): 2,
            int(HidKey.S): 4,
            int(HidKey.D): 8,
        }

        def previous_mask(keys: Sequence[int]) -> int:
            return sum(movement_bits.get(int(key), 0) for key in keys)

        conditional_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        timestep_values: dict[int, list[float]] = defaultdict(list)
        global_values: list[float] = []
        for sequence in dataset(DatasetSplit.TRAIN, config.dataset.train_sequences):
            for time_index, transition in enumerate(
                sequence.transitions[config.dataset.burn_in_steps :],
                start=config.dataset.burn_in_steps,
            ):
                key = (
                    time_index,
                    previous_mask(transition.observation.previous_control.keys_down),
                )
                conditional_values[key].append(transition.value_target)
                timestep_values[time_index].append(transition.value_target)
                global_values.append(transition.value_target)

        conditional_means = {
            key: math.fsum(values) / len(values)
            for key, values in conditional_values.items()
        }
        timestep_means = {
            key: math.fsum(values) / len(values)
            for key, values in timestep_values.items()
        }
        global_mean = math.fsum(global_values) / len(global_values)
        targets: list[float] = []
        global_predictions: list[float] = []
        timestep_predictions: list[float] = []
        conditional_predictions: list[float] = []
        for sequence in dataset(
            DatasetSplit.VALIDATION,
            config.dataset.validation_sequences,
        ):
            for time_index, transition in enumerate(
                sequence.transitions[config.dataset.burn_in_steps :],
                start=config.dataset.burn_in_steps,
            ):
                key = (
                    time_index,
                    previous_mask(transition.observation.previous_control.keys_down),
                )
                targets.append(transition.value_target)
                global_predictions.append(global_mean)
                timestep_predictions.append(timestep_means.get(time_index, global_mean))
                conditional_predictions.append(
                    conditional_means.get(
                        key,
                        timestep_means.get(time_index, global_mean),
                    )
                )

        def mse(predictions: Sequence[float]) -> float:
            return math.fsum(
                (prediction - target) * (prediction - target)
                for prediction, target in zip(predictions, targets)
            ) / len(targets)

        target_mean = math.fsum(targets) / len(targets)
        target_variance = math.fsum(
            (target - target_mean) * (target - target_mean) for target in targets
        ) / len(targets)
        conditional_mse = mse(conditional_predictions)
        self.assertEqual(len(targets), 1_536)
        self.assertEqual(target_mean, 0.6259369390854818)
        self.assertEqual(target_variance.hex(), VALUE_DEVELOPMENT_TARGET_VARIANCE.hex())
        self.assertEqual(mse([0.0] * len(targets)), 0.8277998034808363)
        self.assertEqual(mse(global_predictions), 0.4361457901076875)
        self.assertEqual(mse(timestep_predictions), 0.3714460879257104)
        self.assertEqual(
            conditional_mse.hex(),
            VALUE_TRAIN_CONDITIONAL_BASELINE_MSE.hex(),
        )
        self.assertEqual(
            (VALUE_IMPROVEMENT_RATIO * conditional_mse).hex(),
            VALUE_ABSOLUTE_MAX_MSE.hex(),
        )

    def test_schema3_numeric_aliases_have_one_canonical_hash(self) -> None:
        base = _stage_config()
        float_stage_zero = replace(
            base.stages[0],
            learning_rate=1.0,
            weight_decay=0.0,
            max_gradient_norm=1.0,
        )
        float_config = replace(
            base,
            dataset=replace(base.dataset, discount=1.0),
            optimization=replace(
                base.optimization,
                learning_rate=1.0,
                weight_decay=0.0,
                max_gradient_norm=1.0,
            ),
            stages=(float_stage_zero, base.stages[1]),
        )
        integer_stage_zero = replace(
            base.stages[0],
            learning_rate=1,
            weight_decay=0,
            max_gradient_norm=1,
        )
        integer_config = replace(
            base,
            dataset=replace(base.dataset, discount=1),
            optimization=replace(
                base.optimization,
                learning_rate=1,
                weight_decay=0,
                max_gradient_norm=1,
            ),
            stages=(integer_stage_zero, base.stages[1]),
        )
        negative_zero_stage = replace(float_stage_zero, weight_decay=-0.0)
        negative_zero_config = replace(
            float_config,
            dataset=replace(float_config.dataset, discount=-0.0),
            optimization=replace(float_config.optimization, weight_decay=-0.0),
            stages=(negative_zero_stage, float_config.stages[1]),
        )
        positive_zero_config = replace(
            float_config,
            dataset=replace(float_config.dataset, discount=0.0),
        )

        self.assertEqual(float_config.canonical_json, integer_config.canonical_json)
        self.assertEqual(float_config.config_sha256, integer_config.config_sha256)
        self.assertEqual(
            negative_zero_config.canonical_json,
            positive_zero_config.canonical_json,
        )
        self.assertEqual(
            negative_zero_config.config_sha256,
            positive_zero_config.config_sha256,
        )

    def test_historical_schema1_and_schema2_canonical_hashes_are_unchanged(self) -> None:
        schema_one = load_training_config(
            _BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate.toml"
        )
        schema_two = load_training_config(
            _BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate-b.toml"
        )

        self.assertEqual(
            schema_one.config_sha256,
            "d6f8c8ba3caaaab4643c430ff75747c0025f866363ddfbbb2114fc80969c8fcd",
        )
        self.assertEqual(
            schema_two.config_sha256,
            "8e8e4cc12123f55ee22aace1d65517dbb7586465bbf704c77cc7fe45bb3fc1f9",
        )
        self.assertNotIn("stages", schema_one.to_dict())
        self.assertNotIn("stages", schema_two.to_dict())

    def test_rcq_gate_id_rejects_every_wrong_registered_geometry_axis(self) -> None:
        registered = _rcq_cpu_config()
        stage_zero, stage_one = registered.stages
        mutations = {
            "seed": lambda: replace(
                registered,
                run=replace(registered.run, seed=1703),
            ),
            "dataset": lambda: replace(
                registered,
                dataset=replace(registered.dataset, hazard_count=2),
            ),
            "batch": lambda: replace(
                registered,
                optimization=replace(registered.optimization, batch_size=2),
            ),
            "evaluate cadence": lambda: replace(
                registered,
                logging=replace(registered.logging, evaluate_every_steps=512),
            ),
            "checkpoint cadence": lambda: replace(
                registered,
                logging=replace(registered.logging, checkpoint_every_steps=512),
            ),
            "stage boundary": lambda: replace(
                registered,
                stages=(
                    replace(stage_zero, end_optimizer_step=1_280),
                    replace(stage_one, start_optimizer_step=1_280),
                ),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "RCQ-v2|rcq_v2"):
                mutate()

        self.assertEqual(registered.schema_version, 3)
        self.assertEqual(
            (registered.stages[0].end_optimizer_step, registered.stages[1].end_optimizer_step),
            (1_536, 2_048),
        )

    def test_schema3_configuration_rejects_boolean_and_float_integer_fields(self) -> None:
        config = _stage_config()
        stage = config.stages[0]
        for changes in (
            {"index": True},
            {"start_optimizer_step": False},
            {"end_optimizer_step": 1.0},
            {"warmup_steps": 0.0},
            {"reset_optimizer": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(stage, **changes)

    def test_checkpoint_trainer_state_rejects_boolean_and_float_counts(self) -> None:
        config = _stage_config()
        stage_state = {
            "schema_version": 1,
            "stage_index": 0,
        }
        valid = {
            "schema_version": 1,
            "metrics_byte_length": 4,
            "metrics_record_count": 1,
            "metrics_sha256": sha256(b"x\n").hexdigest(),
        }
        for field, value in (
            ("schema_version", True),
            ("metrics_byte_length", True),
            ("metrics_byte_length", 4.0),
            ("metrics_record_count", False),
            ("metrics_record_count", 1.0),
        ):
            trainer_state = {**valid, field: value}
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaises(ValueError),
            ):
                save_checkpoint(
                    Path(directory) / "bad.pt",
                    cursor=TrainerCursor(),
                    system_state={},
                    rng_state={},
                    config_sha256=config.config_sha256,
                    data_sha256=_DATA_SHA256,
                    code_sha256=_CODE_SHA256,
                    runtime_fingerprint={"runtime": "cpu"},
                    policy=config.resource_policy,
                    stage_state=stage_state,
                    trainer_state=trainer_state,
                )


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class Schema3TorchSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    @staticmethod
    def _system(config: TrainingConfig) -> object:
        assert torch is not None
        return TorchTrainingSystem(_TinyStageObjective(), config)

    def _reach_value_stage(self, config: TrainingConfig) -> object:
        system = self._system(config)
        system.train_optimizer_step((_tiny_batch(),))
        system.transition_to_stage(
            1,
            invariance_batches=(),
            entry_gate_report_sha256=_NULL_SHA256,
            entry_gate_passed=True,
            entry_gate_step=1,
        )
        return system

    def test_runtime_protocol_reset_exact_freeze_mask_and_schema2_checkpoint(
        self,
    ) -> None:
        assert torch is not None
        config = _stage_config()
        system = self._system(config)
        self.assertIsInstance(system, StagedTrainingSystem)
        self.assertEqual(system.active_stage_index, 0)
        self.assertGreater(len(system.optimizer_parameter_names), 2)
        initial_scheduler_state = system.scheduler.state_dict()
        self.assertEqual(
            set(initial_scheduler_state),
            {
                "base_lrs",
                "last_epoch",
                "_step_count",
                "_is_initial",
                "_get_lr_called_within_step",
                "_last_lr",
                "lr_lambdas",
            },
        )
        self.assertIs(initial_scheduler_state["_is_initial"], False)
        self.assertEqual(initial_scheduler_state["lr_lambdas"], [{}])
        self.assertEqual(
            system._scheduler_state_keys,
            frozenset(initial_scheduler_state),
        )

        system.train_optimizer_step((_tiny_batch(),))
        self.assertTrue(system.optimizer.state)
        trunk_before = {
            name: parameter.detach().clone()
            for name, parameter in system.objective.named_parameters()
            if name.startswith("model.trunk.")
        }
        boundary = system.transition_to_stage(
            1,
            invariance_batches=(),
            entry_gate_report_sha256=_NULL_SHA256,
            entry_gate_passed=True,
            entry_gate_step=1,
        )

        self.assertEqual(system.active_stage_index, 1)
        self.assertEqual(system.optimizer_parameter_names, _VALUE_PARAMETERS)
        self.assertEqual(system.optimizer.state, {})
        self.assertEqual(len(system.optimizer.param_groups), 1)
        self.assertEqual(system.optimizer.param_groups[0]["lr"], 3e-3)
        self.assertEqual(system.optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(system.scheduler.last_epoch, 0)
        self.assertEqual(boundary["stage_index"], 1)
        self.assertEqual(boundary["stage_local_optimizer_step"], 0)
        self.assertEqual(tuple(boundary["optimizer_parameter_names"]), _VALUE_PARAMETERS)
        self.assertEqual(
            {
                name
                for name, parameter in system.objective.named_parameters()
                if parameter.requires_grad
            },
            set(_VALUE_PARAMETERS),
        )

        trainer_state = {
            "schema_version": 1,
            "metrics_byte_length": 2,
            "metrics_record_count": 1,
            "metrics_sha256": sha256(b"x\n").hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.pt"
            digest = save_checkpoint(
                path,
                cursor=TrainerCursor(epoch=0, next_batch=1, optimizer_step=1),
                system_state=system.checkpoint_state(),
                rng_state=system.capture_rng_state(),
                config_sha256=config.config_sha256,
                data_sha256=_DATA_SHA256,
                code_sha256=_CODE_SHA256,
                runtime_fingerprint=system.runtime_fingerprint,
                policy=config.resource_policy,
                stage_state=boundary,
                trainer_state=trainer_state,
            )
            loaded = load_checkpoint(
                path,
                expected_config_sha256=config.config_sha256,
                expected_data_sha256=_DATA_SHA256,
                expected_code_sha256=_CODE_SHA256,
                expected_runtime_fingerprint=system.runtime_fingerprint,
                expected_checkpoint_sha256=digest,
            )

        self.assertEqual(loaded.cursor.optimizer_step, 1)
        self.assertIsNotNone(loaded.stage_state)
        self.assertIsNotNone(loaded.trainer_state)
        assert loaded.stage_state is not None and loaded.trainer_state is not None
        self.assertEqual(loaded.stage_state["stage_index"], 1)
        self.assertEqual(loaded.stage_state["stage_local_optimizer_step"], 0)
        self.assertEqual(loaded.trainer_state, trainer_state)

        resumed = self._system(config)
        resumed.prepare_stage_for_resume(loaded.stage_state)
        resumed.restore_checkpoint_state(loaded.system_state)
        resumed.restore_rng_state(loaded.rng_state)
        self.assertEqual(resumed.active_stage_index, 1)
        self.assertEqual(resumed.optimizer_parameter_names, _VALUE_PARAMETERS)
        self.assertEqual(resumed.optimizer.state, {})
        self.assertEqual(resumed.optimizer.param_groups[0]["lr"], 3e-3)
        self.assertEqual(resumed.optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(resumed.scheduler.last_epoch, 0)
        _assert_nested_equal(self, system.checkpoint_state(), resumed.checkpoint_state())
        _assert_nested_equal(
            self,
            system.capture_rng_state(),
            resumed.capture_rng_state(),
        )

        source = _TinyBatchSource()
        uninterrupted_batch = next(
            source.iter_batches(
                split="train",
                epoch=loaded.cursor.epoch,
                start_batch=loaded.cursor.next_batch,
                batch_size=1,
                max_batches=1,
            )
        )
        resumed_batch = next(
            source.iter_batches(
                split="train",
                epoch=loaded.cursor.epoch,
                start_batch=loaded.cursor.next_batch,
                batch_size=1,
                max_batches=1,
            )
        )
        self.assertEqual(
            (
                uninterrupted_batch.epoch,
                uninterrupted_batch.batch_index,
            ),
            (resumed_batch.epoch, resumed_batch.batch_index),
        )
        self.assertEqual(uninterrupted_batch.batch_index, loaded.cursor.next_batch)

        uninterrupted_result = system.train_optimizer_step((uninterrupted_batch,))
        resumed_result = resumed.train_optimizer_step((resumed_batch,))
        self.assertEqual(uninterrupted_result, resumed_result)
        expected_cursor = TrainerCursor(epoch=0, next_batch=2, optimizer_step=2)
        self.assertEqual(
            expected_cursor,
            TrainerCursor(
                epoch=loaded.cursor.epoch,
                next_batch=loaded.cursor.next_batch + 1,
                optimizer_step=loaded.cursor.optimizer_step + 1,
            ),
        )
        _assert_nested_equal(self, system.checkpoint_state(), resumed.checkpoint_state())
        _assert_nested_equal(
            self,
            system.capture_rng_state(),
            resumed.capture_rng_state(),
        )
        self.assertEqual(
            system.stage_checkpoint_state(global_optimizer_step=2),
            resumed.stage_checkpoint_state(global_optimizer_step=2),
        )
        for candidate in (system, resumed):
            for name, parameter in candidate.objective.named_parameters():
                if name in trunk_before:
                    self.assertTrue(torch.equal(parameter, trunk_before[name]))

    def test_stage_zero_midstage_resume_matches_exact_next_update_before_transition(
        self,
    ) -> None:
        base = _stage_config()
        stage_zero = replace(base.stages[0], end_optimizer_step=2)
        stage_one = replace(base.stages[1], start_optimizer_step=2)
        config = replace(base, stages=(stage_zero, stage_one))
        source = _TinyBatchSource()
        uninterrupted = self._system(config)
        first_batch = next(
            source.iter_batches(
                split="train",
                epoch=0,
                start_batch=0,
                batch_size=1,
                max_batches=1,
            )
        )
        uninterrupted.train_optimizer_step((first_batch,))
        stage_state = uninterrupted.stage_checkpoint_state(global_optimizer_step=1)
        cursor = TrainerCursor(epoch=0, next_batch=1, optimizer_step=1)
        trainer_state = {
            "schema_version": 1,
            "metrics_byte_length": 2,
            "metrics_record_count": 1,
            "metrics_sha256": sha256(b"x\n").hexdigest(),
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage-zero-midpoint.pt"
            digest = save_checkpoint(
                path,
                cursor=cursor,
                system_state=uninterrupted.checkpoint_state(),
                rng_state=uninterrupted.capture_rng_state(),
                config_sha256=config.config_sha256,
                data_sha256=_DATA_SHA256,
                code_sha256=_CODE_SHA256,
                runtime_fingerprint=uninterrupted.runtime_fingerprint,
                policy=config.resource_policy,
                stage_state=stage_state,
                trainer_state=trainer_state,
            )
            resumed = self._system(config)
            loaded = load_checkpoint(
                path,
                expected_config_sha256=config.config_sha256,
                expected_data_sha256=_DATA_SHA256,
                expected_code_sha256=_CODE_SHA256,
                expected_runtime_fingerprint=resumed.runtime_fingerprint,
                expected_checkpoint_sha256=digest,
            )
            assert loaded.stage_state is not None
            resumed.prepare_stage_for_resume(loaded.stage_state)
            resumed.restore_checkpoint_state(loaded.system_state)
            resumed.restore_rng_state(loaded.rng_state)

        self.assertEqual(loaded.cursor, cursor)
        self.assertEqual(uninterrupted.active_stage_index, 0)
        self.assertEqual(resumed.active_stage_index, 0)
        self.assertEqual(uninterrupted.scheduler.last_epoch, 1)
        self.assertEqual(resumed.scheduler.last_epoch, 1)
        _assert_nested_equal(
            self,
            uninterrupted.checkpoint_state(),
            resumed.checkpoint_state(),
        )
        _assert_nested_equal(
            self,
            uninterrupted.capture_rng_state(),
            resumed.capture_rng_state(),
        )

        uninterrupted_batch = next(
            source.iter_batches(
                split="train",
                epoch=cursor.epoch,
                start_batch=cursor.next_batch,
                batch_size=1,
                max_batches=1,
            )
        )
        resumed_batch = next(
            source.iter_batches(
                split="train",
                epoch=cursor.epoch,
                start_batch=cursor.next_batch,
                batch_size=1,
                max_batches=1,
            )
        )
        self.assertEqual(
            (uninterrupted_batch.epoch, uninterrupted_batch.batch_index),
            (resumed_batch.epoch, resumed_batch.batch_index),
        )
        uninterrupted_result = uninterrupted.train_optimizer_step(
            (uninterrupted_batch,)
        )
        resumed_result = resumed.train_optimizer_step((resumed_batch,))

        self.assertEqual(uninterrupted_result, resumed_result)
        expected_cursor = TrainerCursor(epoch=0, next_batch=2, optimizer_step=2)
        self.assertEqual(
            expected_cursor,
            TrainerCursor(
                epoch=cursor.epoch,
                next_batch=cursor.next_batch + 1,
                optimizer_step=cursor.optimizer_step + 1,
            ),
        )
        _assert_nested_equal(
            self,
            uninterrupted.checkpoint_state(),
            resumed.checkpoint_state(),
        )
        _assert_nested_equal(
            self,
            uninterrupted.capture_rng_state(),
            resumed.capture_rng_state(),
        )
        self.assertEqual(
            uninterrupted.stage_checkpoint_state(global_optimizer_step=2),
            resumed.stage_checkpoint_state(global_optimizer_step=2),
        )

    def test_midstage_resume_matches_model_optimizer_scheduler_rng_and_next_update(
        self,
    ) -> None:
        assert torch is not None
        config = _stage_config()
        uninterrupted = self._reach_value_stage(config)
        uninterrupted.train_optimizer_step((_tiny_batch(),))
        stage_state = uninterrupted.stage_checkpoint_state(global_optimizer_step=2)
        cursor = TrainerCursor(epoch=0, next_batch=2, optimizer_step=2)
        trainer_state = {
            "schema_version": 1,
            "metrics_byte_length": 2,
            "metrics_record_count": 1,
            "metrics_sha256": sha256(b"x\n").hexdigest(),
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "midstage.pt"
            digest = save_checkpoint(
                path,
                cursor=cursor,
                system_state=uninterrupted.checkpoint_state(),
                rng_state=uninterrupted.capture_rng_state(),
                config_sha256=config.config_sha256,
                data_sha256=_DATA_SHA256,
                code_sha256=_CODE_SHA256,
                runtime_fingerprint=uninterrupted.runtime_fingerprint,
                policy=config.resource_policy,
                stage_state=stage_state,
                trainer_state=trainer_state,
            )
            resumed = self._system(config)
            loaded = load_checkpoint(
                path,
                expected_config_sha256=config.config_sha256,
                expected_data_sha256=_DATA_SHA256,
                expected_code_sha256=_CODE_SHA256,
                expected_runtime_fingerprint=resumed.runtime_fingerprint,
                expected_checkpoint_sha256=digest,
            )
            assert loaded.stage_state is not None
            resumed.prepare_stage_for_resume(loaded.stage_state)
            resumed.restore_checkpoint_state(loaded.system_state)
            resumed.restore_rng_state(loaded.rng_state)

        self.assertEqual(loaded.cursor, cursor)
        self.assertEqual(resumed.active_stage_index, 1)
        self.assertEqual(resumed.scheduler.last_epoch, 1)
        _assert_nested_equal(
            self,
            uninterrupted.checkpoint_state(),
            resumed.checkpoint_state(),
        )
        _assert_nested_equal(
            self,
            uninterrupted.capture_rng_state(),
            resumed.capture_rng_state(),
        )

        uninterrupted_result = uninterrupted.train_optimizer_step((_tiny_batch(),))
        resumed_result = resumed.train_optimizer_step((_tiny_batch(),))
        self.assertEqual(uninterrupted_result, resumed_result)
        _assert_nested_equal(
            self,
            uninterrupted.checkpoint_state(),
            resumed.checkpoint_state(),
        )
        self.assertEqual(
            uninterrupted.stage_checkpoint_state(global_optimizer_step=3),
            resumed.stage_checkpoint_state(global_optimizer_step=3),
        )

    def test_stage_state_strictly_rejects_bool_float_missing_and_mask_tampering(
        self,
    ) -> None:
        config = _stage_config()
        system = self._reach_value_stage(config)
        system.train_optimizer_step((_tiny_batch(),))
        valid = dict(system.stage_checkpoint_state(global_optimizer_step=2))
        cases: tuple[tuple[str, object], ...] = (
            ("schema_version", True),
            ("stage_index", True),
            ("stage_start_optimizer_step", False),
            ("stage_end_optimizer_step", 4.0),
            ("stage_local_optimizer_step", 1.0),
            ("optimizer_reset", 1),
            ("entry_gate_passed", 1),
            ("entry_gate_step", 1.0),
            ("completion_gate_step", False),
            ("optimizer_parameter_names", ["model.value_per_thought.bias"]),
        )
        for field, value in cases:
            tampered = copy.deepcopy(valid)
            tampered[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self._system(config).prepare_stage_for_resume(tampered)

        missing = copy.deepcopy(valid)
        del missing["stage_config_sha256"]
        with self.assertRaisesRegex(ValueError, "incompatible fields"):
            self._system(config).prepare_stage_for_resume(missing)

    def test_restore_rejects_scheduler_and_optimizer_contract_tampering(self) -> None:
        config = _stage_config()
        source = self._reach_value_stage(config)
        source.train_optimizer_step((_tiny_batch(),))
        stage_state = source.stage_checkpoint_state(global_optimizer_step=2)
        valid_state = source.checkpoint_state()

        scheduler_tamper = copy.deepcopy(valid_state)
        scheduler_tamper["scheduler"]["last_epoch"] = 2
        resumed = self._system(config)
        resumed.prepare_stage_for_resume(stage_state)
        with self.assertRaises(ValueError):
            resumed.restore_checkpoint_state(scheduler_tamper)

        optimizer_tamper = copy.deepcopy(valid_state)
        optimizer_tamper["optimizer"]["param_groups"][0]["weight_decay"] = 0.5
        resumed = self._system(config)
        resumed.prepare_stage_for_resume(stage_state)
        with self.assertRaises(ValueError):
            resumed.restore_checkpoint_state(optimizer_tamper)

        empty_optimizer_state = copy.deepcopy(valid_state)
        empty_optimizer_state["optimizer"]["state"] = {}
        resumed = self._system(config)
        resumed.prepare_stage_for_resume(stage_state)
        with self.assertRaises(ValueError):
            resumed.restore_checkpoint_state(empty_optimizer_state)

        step_tamper = copy.deepcopy(valid_state)
        first_parameter = next(iter(step_tamper["optimizer"]["state"]))
        adam_state = step_tamper["optimizer"]["state"][first_parameter]
        adam_state["step"] = adam_state["step"] + 1
        resumed = self._system(config)
        resumed.prepare_stage_for_resume(stage_state)
        with self.assertRaises(ValueError):
            resumed.restore_checkpoint_state(step_tamper)

    def test_final_evaluation_restore_validates_full_training_state(self) -> None:
        config = _stage_config(final_step=2)
        source = self._reach_value_stage(config)
        source.train_optimizer_step((_tiny_batch(),))
        stage_state = source.stage_checkpoint_state(global_optimizer_step=2)
        valid_state = source.checkpoint_state()

        restored = self._system(config)
        restored.restore_model_for_evaluation(valid_state, stage_state)
        self.assertFalse(restored.objective.training)
        _assert_nested_equal(
            self,
            valid_state["objective"],
            restored.checkpoint_state()["objective"],
        )
        self.assertEqual(restored.optimizer.state, {})
        self.assertEqual(restored.scheduler.last_epoch, 0)
        self.assertEqual(restored.scaler.state_dict(), {})

        optimizer_tamper = copy.deepcopy(valid_state)
        optimizer_tamper["optimizer"]["param_groups"][0]["weight_decay"] = 0.5
        scheduler_tamper = copy.deepcopy(valid_state)
        scheduler_tamper["scheduler"]["last_epoch"] = 0
        scheduler_missing_static_tamper = copy.deepcopy(valid_state)
        del scheduler_missing_static_tamper["scheduler"]["_is_initial"]
        empty_adam_tamper = copy.deepcopy(valid_state)
        empty_adam_tamper["optimizer"]["state"] = {}
        optimizer_extra_tamper = copy.deepcopy(valid_state)
        optimizer_extra_tamper["optimizer"]["unexpected"] = True
        negative_second_moment_tamper = copy.deepcopy(valid_state)
        first_parameter = next(iter(negative_second_moment_tamper["optimizer"]["state"]))
        second_moment = negative_second_moment_tamper["optimizer"]["state"][
            first_parameter
        ]["exp_avg_sq"]
        negative_second_moment_tamper["optimizer"]["state"][first_parameter][
            "exp_avg_sq"
        ] = -torch.ones_like(second_moment)
        scaler_tamper = copy.deepcopy(valid_state)
        scaler_tamper["scaler"] = {"scale": 1.0}
        for name, tampered in (
            ("optimizer", optimizer_tamper),
            ("scheduler", scheduler_tamper),
            ("scheduler_missing_static", scheduler_missing_static_tamper),
            ("adam", empty_adam_tamper),
            ("optimizer_extra", optimizer_extra_tamper),
            ("negative_second_moment", negative_second_moment_tamper),
            ("scaler", scaler_tamper),
        ):
            candidate = self._system(config)
            objective_before = copy.deepcopy(candidate.objective.state_dict())
            with self.subTest(name=name), self.assertRaises(ValueError):
                candidate.restore_model_for_evaluation(
                    tampered,
                    stage_state,
                )
            _assert_nested_equal(
                self,
                objective_before,
                candidate.objective.state_dict(),
            )

    def test_all_future_stage_masks_are_resolved_at_construction(self) -> None:
        config = _stage_config()
        invalid_stage = replace(
            config.stages[1],
            trainable_parameters=("model.value_per_thought.typo",),
        )
        invalid = replace(config, stages=(config.stages[0], invalid_stage))
        with self.assertRaisesRegex(ValueError, "do not exist"):
            self._system(invalid)

    def test_real_smoke_joint_update_can_emit_stage_checkpoint_state(self) -> None:
        """Exercise optimizer coverage with the production model/objective graph."""

        registered = load_training_config(
            _BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-rcq-v2-staging-canary.toml"
        )
        config = replace(
            registered,
            precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
            resources=replace(
                registered.resources,
                allow_gpu=False,
                cpu_threads=1,
            ),
        )
        objective_config = config.objective
        objective = ThoughtFieldObjective(
            build_smoke_model(config),
            action_loss_kind=objective_config.action_loss_kind,
            button_support_control_indices=(
                objective_config.button_support_control_indices
            ),
            button_support_weight=objective_config.button_support_weight,
            button_background_weight=objective_config.button_background_weight,
            button_background_tail_mix=objective_config.button_background_tail_mix,
            button_background_tail_temperature=(
                objective_config.button_background_tail_temperature
            ),
            continuous_action_weight=objective_config.continuous_action_weight,
            action_weight=objective_config.action_weight,
            value_weight=objective_config.value_weight,
            world_weight=objective_config.world_weight,
            diversity_weight=objective_config.diversity_weight,
        )
        system = TorchTrainingSystem(objective, config)
        source = MovingShapesBatchSource(config.dataset)
        batch = next(
            source.iter_batches(
                split="train",
                epoch=0,
                start_batch=0,
                batch_size=1,
                max_batches=1,
            )
        )

        system.train_optimizer_step((batch,))
        observed_state_fields = {
            tuple(sorted(parameter_state))
            for parameter_state in system.optimizer.state.values()
        }
        self.assertEqual(
            observed_state_fields,
            {("exp_avg", "exp_avg_sq", "step")},
        )
        state = system.stage_checkpoint_state(global_optimizer_step=1)

        self.assertEqual(state["stage_index"], 0)
        self.assertEqual(state["stage_local_optimizer_step"], 1)


class Schema3TrainerBoundaryTests(unittest.TestCase):
    def _run(
        self,
        validation_results: Sequence[TrainingStepResult],
    ) -> tuple[TrainingConfig, _TinyBatchSource, _StagedSystemDouble, object, Path, object]:
        config = _rcq_cpu_config()
        source = _TinyBatchSource()
        system = _StagedSystemDouble(config, validation_results)
        temporary = tempfile.TemporaryDirectory()
        run_dir = Path(temporary.name)
        trainer = Trainer(
            config,
            system,
            source,
            run_dir,
            code_sha256=_CODE_SHA256,
        )
        # Begin immediately before the registered joint-stage boundary. This is
        # still the exact cursor exposure implied by the schema-3 contract and
        # keeps the orchestration test CPU-small without weakening gate geometry.
        start_step = config.stages[0].end_optimizer_step - 1
        batches_per_epoch = source.batches_per_epoch(split="train", batch_size=1)
        consumed = start_step * config.optimization.gradient_accumulation_steps
        epoch, next_batch = divmod(consumed, batches_per_epoch)
        trainer._prepare_run(resuming=False)
        trainer.cursor = TrainerCursor(
            epoch=epoch,
            next_batch=next_batch,
            optimizer_step=start_step,
        )
        assert trainer._metrics is not None
        for split, step, record_epoch in trainer._expected_metric_records(start_step):
            trainer._metrics.append(
                MetricRecord(
                    step=step,
                    epoch=record_epoch,
                    split=split,
                    metrics={"seeded_cadence_fixture": 1.0},
                )
            )
        system.train_calls = start_step
        with patch.object(trainer, "_prepare_run", return_value=None):
            summary = trainer.run()
        return config, source, system, summary, run_dir, temporary

    @staticmethod
    def _load(
        config: TrainingConfig,
        system: _StagedSystemDouble,
        path: Path,
    ) -> object:
        return load_checkpoint(
            path,
            expected_config_sha256=config.config_sha256,
            expected_data_sha256=_DATA_SHA256,
            expected_code_sha256=_CODE_SHA256,
            expected_runtime_fingerprint=system.runtime_fingerprint,
            expected_checkpoint_sha256=file_sha256(path),
        )

    def test_passed_transition_checkpoints_new_stage_at_local_zero_and_final_pass(
        self,
    ) -> None:
        config, _source, system, summary, run_dir, temporary = self._run(
            (
                _gate_result(value_loss=0.40),
                _gate_result(value_loss=0.30),
                _gate_result(value_loss=0.30),
            )
        )
        try:
            self.assertTrue(summary.completed)
            self.assertIsNone(summary.stop_reason)
            boundary = self._load(
                config,
                system,
                run_dir / "checkpoints" / "step-00001536.pt",
            )
            final = self._load(config, system, summary.last_checkpoint)
            assert boundary.stage_state is not None and final.stage_state is not None
            self.assertEqual(boundary.stage_state["stage_index"], 1)
            self.assertEqual(boundary.stage_state["stage_local_optimizer_step"], 0)
            self.assertEqual(
                tuple(boundary.stage_state["optimizer_parameter_names"]),
                _VALUE_PARAMETERS,
            )
            self.assertTrue(boundary.stage_state["entry_gate_passed"])
            self.assertNotEqual(
                boundary.stage_state["entry_gate_report_sha256"],
                _NULL_SHA256,
            )
            self.assertEqual(final.stage_state["stage_index"], 1)
            self.assertEqual(final.stage_state["stage_local_optimizer_step"], 512)
            self.assertIs(final.stage_state["completion_gate_passed"], True)
            self.assertEqual(final.stage_state["completion_gate_step"], 2_048)
            self.assertEqual(
                final.stage_state["completion_gate_report_sha256"],
                file_sha256(run_dir / "final-development-step-00002048.json"),
            )
        finally:
            temporary.cleanup()

    def test_failed_transition_is_terminal_and_checkpoints_stage_zero_boundary(
        self,
    ) -> None:
        config, _source, system, summary, run_dir, temporary = self._run(
            (_gate_result(passed=False, value_loss=0.40),)
        )
        try:
            self.assertFalse(summary.completed)
            self.assertEqual(summary.stop_reason, "development_gate_failed")
            self.assertEqual(summary.cursor.optimizer_step, 1_536)
            loaded = self._load(config, system, summary.last_checkpoint)
            assert loaded.stage_state is not None
            self.assertEqual(loaded.stage_state["stage_index"], 0)
            self.assertEqual(loaded.stage_state["stage_local_optimizer_step"], 1_536)
            self.assertEqual(system.active_stage_index, 0)
            payload = json.loads(
                (run_dir / "development-gate-step-00001536.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIs(payload["passed"], False)
            self.assertFalse((run_dir / "stage-transition-01.json").exists())
        finally:
            temporary.cleanup()

    def test_failed_completion_is_terminal_and_checkpoint_binds_failed_receipt(
        self,
    ) -> None:
        config, _source, system, summary, run_dir, temporary = self._run(
            (
                _gate_result(value_loss=0.40),
                _gate_result(value_loss=0.30),
                _gate_result(passed=False, value_loss=0.30),
            )
        )
        try:
            self.assertFalse(summary.completed)
            self.assertEqual(summary.stop_reason, "final_development_gate_failed")
            self.assertEqual(summary.cursor.optimizer_step, 2_048)
            loaded = self._load(config, system, summary.last_checkpoint)
            assert loaded.stage_state is not None
            self.assertEqual(loaded.stage_state["stage_index"], 1)
            self.assertEqual(loaded.stage_state["stage_local_optimizer_step"], 512)
            self.assertIs(loaded.stage_state["completion_gate_passed"], False)
            self.assertEqual(loaded.stage_state["completion_gate_step"], 2_048)
            receipt = run_dir / "final-development-step-00002048.json"
            self.assertEqual(
                loaded.stage_state["completion_gate_report_sha256"],
                file_sha256(receipt),
            )
            self.assertIs(json.loads(receipt.read_text(encoding="utf-8"))["passed"], False)
        finally:
            temporary.cleanup()

    def test_value_threshold_failure_is_terminal_and_resume_reconstructs_receipt(
        self,
    ) -> None:
        final_value_loss = VALUE_ABSOLUTE_MAX_MSE + 1e-6
        config, source, system, summary, run_dir, temporary = self._run(
            (
                _gate_result(value_loss=0.40),
                _gate_result(value_loss=0.30),
                _gate_result(value_loss=final_value_loss),
            )
        )
        try:
            self.assertFalse(summary.completed)
            self.assertEqual(summary.stop_reason, "final_development_gate_failed")
            self.assertEqual(summary.cursor.optimizer_step, 2_048)
            receipt = run_dir / "final-development-step-00002048.json"
            receipt_bytes = receipt.read_bytes()
            receipt_digest = file_sha256(receipt)
            payload = json.loads(receipt_bytes.decode("utf-8"))
            self.assertEqual(payload["gate"], VALUE_DEVELOPMENT_GATE_ID)
            self.assertIs(payload["passed"], False)
            checks = {check["name"]: check for check in payload["checks"]}
            value_check_names = {
                "value_loss_absolute",
                "value_loss_entry_improvement",
                "value_development_r2",
            }
            self.assertTrue(
                all(
                    check["passed"]
                    for name, check in checks.items()
                    if name not in value_check_names
                )
            )
            self.assertIs(checks["value_loss_absolute"]["passed"], False)
            self.assertIs(checks["value_loss_entry_improvement"]["passed"], True)

            loaded = self._load(config, system, summary.last_checkpoint)
            assert loaded.stage_state is not None
            self.assertEqual(loaded.stage_state["stage_index"], 1)
            self.assertEqual(loaded.stage_state["stage_local_optimizer_step"], 512)
            self.assertIs(loaded.stage_state["completion_gate_passed"], False)
            self.assertEqual(
                loaded.stage_state["completion_gate_report_sha256"],
                receipt_digest,
            )
            self.assertEqual(file_sha256(summary.last_checkpoint), summary.checkpoint_sha256)

            resumed_system = _StagedSystemDouble(config, ())
            resumed = Trainer(
                config,
                resumed_system,
                source,
                run_dir,
                code_sha256=_CODE_SHA256,
            )
            resumed._prepare_run(resuming=True)
            resumed._resume(
                summary.last_checkpoint,
                expected_checkpoint_sha256=summary.checkpoint_sha256,
            )
            self.assertEqual(resumed.cursor, summary.cursor)
            self.assertEqual(resumed_system.active_stage_index, 1)
            self.assertEqual(resumed._development_gate_path, summary.development_gate_path)
            self.assertEqual(resumed._final_development_path, receipt)
            self.assertEqual(receipt.read_bytes(), receipt_bytes)
        finally:
            temporary.cleanup()

    def test_gate_metrics_must_match_checkpoint_bound_validation_record(self) -> None:
        config, source, system, summary, run_dir, temporary = self._run(
            (
                _gate_result(value_loss=0.40),
                _gate_result(value_loss=0.30),
                _gate_result(value_loss=0.28),
            )
        )
        try:
            loaded = self._load(config, system, summary.last_checkpoint)
            assert loaded.trainer_state is not None
            verifier = Trainer(
                config,
                _StagedSystemDouble(config, ()),
                source,
                run_dir,
                code_sha256=_CODE_SHA256,
            )
            verifier._prepare_run(resuming=True)
            verifier._assert_gate_metrics_match_durable_validation(
                loaded.trainer_state,
                gate_step=1_536,
                gate_metrics=_gate_metrics(value_loss=0.40),
                gate_name="entry development gate",
            )
            with self.assertRaisesRegex(ValueError, "differs from durable validation"):
                verifier._assert_gate_metrics_match_durable_validation(
                    loaded.trainer_state,
                    gate_step=1_536,
                    gate_metrics=_gate_metrics(value_loss=0.41),
                    gate_name="entry development gate",
                )

            rewritten: list[bytes] = []
            for line in verifier.metrics_path.read_bytes().splitlines():
                payload = json.loads(line.decode("utf-8"))
                if payload["split"] == "validation" and payload["step"] == 1_536:
                    payload["metrics"]["loss"] = math.nextafter(0.25, math.inf)
                record = MetricRecord(
                    step=payload["step"],
                    epoch=payload["epoch"],
                    split=payload["split"],
                    metrics=payload["metrics"],
                )
                rewritten.append((record.canonical_json + "\n").encode("utf-8"))
            tampered_bytes = b"".join(rewritten)
            verifier.metrics_path.write_bytes(tampered_bytes)
            tampered_state = {
                **loaded.trainer_state,
                "metrics_byte_length": len(tampered_bytes),
                "metrics_sha256": sha256(tampered_bytes).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "loss differs"):
                verifier._assert_gate_metrics_match_durable_validation(
                    tampered_state,
                    gate_step=1_536,
                    gate_metrics=_gate_metrics(value_loss=0.40),
                    gate_name="entry development gate",
                )
        finally:
            temporary.cleanup()

    def test_max_float_value_failure_preserves_terminal_receipt_and_checkpoint(
        self,
    ) -> None:
        config, _source, system, summary, run_dir, temporary = self._run(
            (
                _gate_result(value_loss=0.40),
                _gate_result(value_loss=0.30),
                _gate_result(value_loss=sys.float_info.max),
            )
        )
        try:
            self.assertFalse(summary.completed)
            self.assertEqual(summary.stop_reason, "final_development_gate_failed")
            self.assertEqual(summary.cursor.optimizer_step, 2_048)
            receipt = run_dir / "final-development-step-00002048.json"
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            checks = {check["name"]: check for check in payload["checks"]}
            self.assertIs(payload["passed"], False)
            self.assertEqual(
                checks["value_development_r2"]["observed"],
                -sys.float_info.max,
            )
            loaded = self._load(config, system, summary.last_checkpoint)
            assert loaded.stage_state is not None
            self.assertIs(loaded.stage_state["completion_gate_passed"], False)
            self.assertEqual(
                loaded.stage_state["completion_gate_report_sha256"],
                file_sha256(receipt),
            )
        finally:
            temporary.cleanup()

    def test_generic_schema3_test_evaluation_is_refused_before_source_access(self) -> None:
        config = _stage_config(final_step=2)
        source = _TinyBatchSource()
        system = _StagedSystemDouble(config, ())
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                config,
                system,
                source,
                directory,
                code_sha256=_CODE_SHA256,
            )
            with self.assertRaisesRegex(ValueError, "once-only final evaluator"):
                trainer.evaluate(split="test")

        self.assertNotIn("test", source.requested_splits)

    def test_metrics_prefix_rejects_same_step_reordering_and_epoch_tampering(
        self,
    ) -> None:
        config = _stage_config(final_step=2)
        source = _TinyBatchSource()
        system = _StagedSystemDouble(config, ())
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                config,
                system,
                source,
                directory,
                code_sha256=_CODE_SHA256,
            )
            trainer._prepare_run(resuming=False)
            trainer.cursor = TrainerCursor(epoch=0, next_batch=1, optimizer_step=1)
            assert trainer._metrics is not None
            records = (
                MetricRecord(
                    step=1,
                    epoch=0,
                    split="train",
                    metrics={"fixture": 1.0},
                ),
                MetricRecord(
                    step=1,
                    epoch=0,
                    split="validation",
                    metrics={"fixture": 1.0},
                ),
            )
            for record in records:
                trainer._metrics.append(record)
            lines = trainer.metrics_path.read_bytes().splitlines(keepends=True)
            trainer.metrics_path.write_bytes(b"".join(reversed(lines)))
            with self.assertRaisesRegex(ValueError, "ordered cadence/epochs"):
                trainer._metrics_prefix_state()

            tampered = (
                records[0].canonical_json
                + "\n"
                + MetricRecord(
                    step=1,
                    epoch=1,
                    split="validation",
                    metrics={"fixture": 1.0},
                ).canonical_json
                + "\n"
            ).encode("utf-8")
            trainer.metrics_path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "ordered cadence/epochs"):
                trainer._metrics_prefix_state()

    def test_terminal_resume_requires_exact_completion_gate_artifact(self) -> None:
        config, source, system, summary, run_dir, temporary = self._run(
            (
                _gate_result(value_loss=0.40),
                _gate_result(value_loss=0.30),
                _gate_result(value_loss=0.30),
            )
        )
        try:
            final_artifact = run_dir / "final-development-step-00002048.json"
            final_artifact.unlink()
            resumed = Trainer(
                config,
                _StagedSystemDouble(config, ()),
                source,
                run_dir,
                code_sha256=_CODE_SHA256,
            )
            with self.assertRaisesRegex(ValueError, "final development artifact"):
                resumed.run(
                    resume_from=summary.last_checkpoint,
                    expected_checkpoint_sha256=summary.checkpoint_sha256,
                )
        finally:
            temporary.cleanup()

    def test_durable_metric_tail_replays_byte_exactly_before_new_append(self) -> None:
        config = _stage_config(final_step=4)
        source = _TinyBatchSource()
        train_one = TrainingStepResult(
            loss=1.0,
            metrics={"fake_train": 1.0},
            samples=1,
        )
        train_two = TrainingStepResult(
            loss=2.0,
            metrics={"fake_train": 2.0},
            samples=1,
        )
        train_three = TrainingStepResult(
            loss=3.0,
            metrics={"fake_train": 3.0},
            samples=1,
        )
        validation = TrainingStepResult(
            loss=0.5,
            metrics={"fake_validation": 0.5},
            samples=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            original_system = _StagedSystemDouble(config, ())
            original = Trainer(
                config,
                original_system,
                source,
                run_dir,
                code_sha256=_CODE_SHA256,
            )
            original._prepare_run(resuming=False)
            original_system.train_calls = 1
            original_system.transition_to_stage(
                1,
                invariance_batches=(),
                entry_gate_report_sha256=_NULL_SHA256,
                entry_gate_passed=True,
                entry_gate_step=1,
            )
            original.cursor = TrainerCursor(
                epoch=0,
                next_batch=1,
                optimizer_step=1,
            )
            original._record("train", train_one)
            original._record("validation", validation)
            checkpoint, checkpoint_sha = original._checkpoint()

            # These records reached durable storage after the checkpoint but
            # before the simulated process failure.
            original_system.train_calls = 2
            original.cursor = TrainerCursor(
                epoch=0,
                next_batch=2,
                optimizer_step=2,
            )
            original._record("train", train_two)
            original._record("validation", validation)
            durable_crash_bytes = original.metrics_path.read_bytes()

            mismatched_system = _StagedSystemDouble(config, ())
            mismatched = Trainer(
                config,
                mismatched_system,
                source,
                run_dir,
                code_sha256=_CODE_SHA256,
            )
            mismatched._prepare_run(resuming=True)
            mismatched._resume(
                checkpoint,
                expected_checkpoint_sha256=checkpoint_sha,
            )
            mismatched.cursor = TrainerCursor(
                epoch=0,
                next_batch=2,
                optimizer_step=2,
            )
            with self.assertRaisesRegex(ValueError, "differs from the durable"):
                mismatched._record(
                    "train",
                    replace(train_two, loss=2.5),
                )
            self.assertEqual(mismatched.metrics_path.read_bytes(), durable_crash_bytes)

            resumed_system = _StagedSystemDouble(config, ())
            resumed = Trainer(
                config,
                resumed_system,
                source,
                run_dir,
                code_sha256=_CODE_SHA256,
            )
            resumed._prepare_run(resuming=True)
            resumed._resume(
                checkpoint,
                expected_checkpoint_sha256=checkpoint_sha,
            )
            resumed_system.train_calls = 2
            resumed.cursor = TrainerCursor(
                epoch=0,
                next_batch=2,
                optimizer_step=2,
            )
            resumed._record("train", train_two)
            with self.assertRaisesRegex(ValueError, "tail is replayed"):
                resumed._checkpoint()
            resumed._record("validation", validation)
            self.assertEqual(resumed.metrics_path.read_bytes(), durable_crash_bytes)

            resumed_system.train_calls = 3
            resumed.cursor = TrainerCursor(
                epoch=0,
                next_batch=3,
                optimizer_step=3,
            )
            resumed._record("train", train_three)
            resumed._record("validation", validation)
            expected_final_bytes = durable_crash_bytes + (
                MetricRecord(
                    step=3,
                    epoch=0,
                    split="train",
                    metrics={
                        "fake_train": 3.0,
                        "loss": 3.0,
                        "samples": 1.0,
                        "stage_index": 1.0,
                    },
                ).canonical_json
                + "\n"
                + MetricRecord(
                    step=3,
                    epoch=0,
                    split="validation",
                    metrics={
                        "fake_validation": 0.5,
                        "loss": 0.5,
                        "samples": 1.0,
                        "stage_index": 1.0,
                    },
                ).canonical_json
                + "\n"
            ).encode("utf-8")
            final_checkpoint, final_sha = resumed._checkpoint()
            self.assertEqual(resumed.metrics_path.read_bytes(), expected_final_bytes)
            loaded = self._load(config, resumed_system, final_checkpoint)
            assert loaded.trainer_state is not None
            self.assertEqual(loaded.checkpoint_sha256, final_sha)
            self.assertEqual(
                loaded.trainer_state["metrics_sha256"],
                sha256(expected_final_bytes).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
