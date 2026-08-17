from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.runtime.policy import ResourceDenied, ResourcePolicy
from irene_brain.training.batches import (
    BUTTON_TARGET_INDICES,
    CONTINUOUS_TARGET_INDICES,
    CONTROL_LAYOUT_ID,
    CONTROL_VECTOR_SIZE,
    MovingShapesBatchSource,
    TrajectoryBatch,
    control_to_vector,
)
from irene_brain.training.config import (
    DatasetConfig,
    DeterminismConfig,
    LoggingConfig,
    OptimizationConfig,
    PrecisionConfig,
    ResourceConfig,
    RunConfig,
    TrainingConfig,
    load_training_config,
)
from irene_brain.training.metrics import JsonlMetricWriter, MetricRecord
from irene_brain.training.checkpoint import LoadedCheckpoint, TrainerCursor, save_checkpoint
from irene_brain.training.protocol import TrainingStepResult
from irene_brain.training.schedules import (
    CONSTANT_AFTER_WARMUP,
    COSINE_AFTER_WARMUP,
    learning_rate_multiplier,
)
from irene_brain.training.trainer import Trainer
from irene_brain.training.train import (
    _bounded_config,
    _environment_value,
    _model_factory,
    _resume_path,
    _run_directory,
)
from irene_brain.types import GenericControl, HidKey

BRAIN_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError:  # Phase 0's stdlib-only environment remains supported.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.training.checkpoint import load_checkpoint
    from irene_brain.training.factory import build_smoke_model, build_thesis_model
    from irene_brain.training.objective import (
        LossOutput,
        ThoughtFieldObjective,
        _button_exact_set_count,
        _final_action_logit_metric_counts,
        _movement_exact_set_count,
        _movement_metric_counts,
        _structured_action_loss,
        deterministic_eval_thought_noise,
    )
    from irene_brain.training.torch_system import TorchTrainingSystem


    class ScalarLossObjective(torch.nn.Module):
        """Minimal deterministic objective for optimizer/scheduler contract tests."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch: object) -> LossOutput:
            samples = getattr(batch, "sample_count")
            return LossOutput(
                loss=self.weight.square() + 1.0,
                metrics={},
                samples=samples,
            )


def training_config(*, max_optimizer_steps: int = 2) -> TrainingConfig:
    return TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name="phase1-test",
            seed=20260816,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=max_optimizer_steps,
        ),
        dataset=DatasetConfig(
            kind="moving_shapes",
            train_sequences=4,
            validation_sequences=2,
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
            learning_rate=1e-3,
            weight_decay=0.0,
            max_gradient_norm=1.0,
            warmup_steps=0,
        ),
        precision=PrecisionConfig(
            device="cpu",
            mode="float32",
            allow_tf32=False,
        ),
        determinism=DeterminismConfig(
            enabled=True,
            num_workers=0,
            compile_model=False,
        ),
        logging=LoggingConfig(
            log_every_steps=1,
            evaluate_every_steps=1,
            validation_batches=1,
            checkpoint_every_steps=1,
            keep_last_checkpoints=1,
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
    )


class FakeTrainingSystem:
    """Dependency-free optimizer-boundary double for trainer orchestration tests."""

    def __init__(self) -> None:
        self.train_calls = 0
        self.evaluate_calls = 0
        self.restored_rng: Mapping[str, object] | None = None
        self.trained_sequence_indices: list[tuple[int, ...]] = []

    @property
    def runtime_fingerprint(self) -> Mapping[str, str | int | bool]:
        return {"runtime": "fake-cpu", "world_size": 1}

    def train_optimizer_step(self, microbatches: Sequence[object]) -> TrainingStepResult:
        self.train_calls += 1
        samples = 0
        for batch in microbatches:
            if not isinstance(batch, TrajectoryBatch):
                raise ValueError("fake trainer requires TrajectoryBatch inputs")
            samples += batch.sample_count
            self.trained_sequence_indices.append(
                tuple(sequence.sequence_index for sequence in batch.sequences)
            )
        return TrainingStepResult(
            loss=float(self.train_calls),
            metrics={"fake_train": float(self.train_calls)},
            samples=samples,
        )

    def evaluate_batch(self, batch: object) -> TrainingStepResult:
        if not isinstance(batch, TrajectoryBatch):
            raise ValueError("fake evaluator requires a TrajectoryBatch")
        self.evaluate_calls += 1
        return TrainingStepResult(
            loss=0.25,
            metrics={"fake_validation": 0.5},
            samples=batch.sample_count,
        )

    def checkpoint_state(self) -> Mapping[str, object]:
        return {"train_calls": self.train_calls}

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        train_calls = state.get("train_calls")
        if isinstance(train_calls, bool) or not isinstance(train_calls, int):
            raise ValueError("fake checkpoint train_calls must be an integer")
        self.train_calls = train_calls

    def capture_rng_state(self) -> Mapping[str, object]:
        return {"counter": self.train_calls}

    def restore_rng_state(self, state: Mapping[str, object]) -> None:
        self.restored_rng = dict(state)


def fake_save_checkpoint(path: str | Path, **_kwargs: object) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"test optimizer-boundary checkpoint")
    return "e" * 64


class TrainingOrchestrationContractTests(unittest.TestCase):
    def test_pseudo_brain_environment_names_are_canonical_with_strict_legacy_aliases(
        self,
    ) -> None:
        config = training_config(max_optimizer_steps=2)
        with patch.dict(
            os.environ,
            {
                "PSEUDO_BRAIN_MAX_STEPS": "1",
                "PSEUDO_BRAIN_RUN_DIR": "pseudo-run",
                "PSEUDO_BRAIN_RESUME": "pseudo-checkpoint.pt",
                "PSEUDO_BRAIN_CHECKPOINT_SHA256": "a" * 64,
            },
            clear=True,
        ):
            self.assertEqual(_bounded_config(config).run.max_optimizer_steps, 1)
            self.assertEqual(_run_directory(None), Path("pseudo-run"))
            self.assertEqual(_resume_path(None), Path("pseudo-checkpoint.pt"))
            self.assertEqual(
                _environment_value(
                    "PSEUDO_BRAIN_CHECKPOINT_SHA256",
                    "IRENE_BRAIN_CHECKPOINT_SHA256",
                ),
                "a" * 64,
            )

        with patch.dict(
            os.environ,
            {"IRENE_BRAIN_RUN_DIR": "legacy-run"},
            clear=True,
        ):
            self.assertEqual(_run_directory(None), Path("legacy-run"))

        with patch.dict(
            os.environ,
            {
                "PSEUDO_BRAIN_RUN_DIR": "same-run",
                "IRENE_BRAIN_RUN_DIR": "same-run",
            },
            clear=True,
        ):
            self.assertEqual(_run_directory(None), Path("same-run"))

        with patch.dict(
            os.environ,
            {
                "PSEUDO_BRAIN_RUN_DIR": "pseudo-run",
                "IRENE_BRAIN_RUN_DIR": "different-legacy-run",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "conflicting PSEUDO_BRAIN_RUN_DIR"):
                _run_directory(None)

    def test_entrypoint_configures_deterministic_cuda_and_seed_before_factory(self) -> None:
        source = (
            BRAIN_ROOT / "src" / "irene_brain" / "training" / "train.py"
        ).read_text(encoding="utf-8")
        cublas_setup = source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]')
        torch_import = source.index("    import torch\n")
        random_seed = source.index("random.seed(config.run.seed)")
        torch_seed = source.index("torch.manual_seed(config.run.seed)")
        factory_call = source.index("_model_factory(config.run.model_factory)(config)")

        self.assertLess(cublas_setup, torch_import)
        self.assertLess(torch_import, random_seed)
        self.assertLess(random_seed, factory_call)
        self.assertLess(torch_seed, factory_call)

    def test_checked_in_dgx_configurations_parse_strictly(self) -> None:
        smoke = load_training_config(BRAIN_ROOT / "configs" / "training" / "dgx-smoke.toml")
        bootstrap = load_training_config(
            BRAIN_ROOT / "configs" / "training" / "phase1-bootstrap.toml"
        )
        overfit = load_training_config(
            BRAIN_ROOT / "configs" / "training" / "dgx-stagea-action-overfit.toml"
        )

        self.assertEqual(smoke.run.max_optimizer_steps, 3)
        self.assertEqual(smoke.dataset.train_sequences, 3)
        self.assertEqual(smoke.run.model_factory, "irene_brain.training.factory:build_smoke_model")
        self.assertEqual(
            bootstrap.run.model_factory,
            "irene_brain.training.factory:build_thesis_model",
        )
        self.assertEqual(overfit.run.max_optimizer_steps, 300)
        self.assertEqual(overfit.dataset.train_sequences, 8)
        self.assertEqual(overfit.dataset.validation_sequences, 8)
        self.assertEqual(overfit.dataset.test_sequences, 8)
        self.assertEqual(overfit.dataset.sequence_length, 8)
        self.assertEqual(overfit.dataset.burn_in_steps, 2)
        self.assertEqual(overfit.optimization.batch_size, 1)
        self.assertEqual(overfit.optimization.gradient_accumulation_steps, 8)
        self.assertEqual(overfit.optimization.learning_rate, 1e-4)
        self.assertEqual(overfit.optimization.warmup_steps, 10)
        self.assertEqual(overfit.optimization.weight_decay, 0.0)
        self.assertEqual(overfit.objective.action_weight, 1.0)
        self.assertEqual(overfit.objective.value_weight, 0.0)
        self.assertEqual(overfit.objective.world_weight, 0.0)
        self.assertEqual(overfit.objective.diversity_weight, 0.0)
        self.assertEqual(overfit.logging.log_every_steps, 10)
        self.assertEqual(overfit.logging.evaluate_every_steps, 100)
        self.assertEqual(overfit.logging.validation_batches, 8)
        self.assertEqual(overfit.logging.checkpoint_every_steps, 100)
        self.assertEqual(
            MovingShapesBatchSource(overfit.dataset).batches_per_epoch(
                split="train",
                batch_size=overfit.optimization.batch_size,
            ),
            overfit.optimization.gradient_accumulation_steps,
        )
        for config in (smoke, bootstrap, overfit):
            self.assertEqual(config.precision.device, "cuda")
            self.assertEqual(config.precision.mode, "bfloat16")
            self.assertTrue(config.resource_policy.allow_gpu)
            self.assertFalse(config.resource_policy.allow_capture)
            self.assertFalse(config.resource_policy.allow_hid_output)
            self.assertFalse(config.resource_policy.allow_network)
            self.assertFalse(config.resource_policy.allow_subprocess)
            self.assertEqual(config.determinism.num_workers, 0)
            self.assertFalse(config.determinism.compile_model)
        self.assertNotEqual(smoke.config_sha256, bootstrap.config_sha256)
        self.assertNotEqual(overfit.config_sha256, bootstrap.config_sha256)

    def test_training_configuration_is_canonical_and_offline_only(self) -> None:
        first = training_config()
        second = training_config()

        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.config_sha256, second.config_sha256)
        self.assertEqual(len(first.config_sha256), 64)
        self.assertEqual(json.loads(first.canonical_json), first.to_dict())
        self.assertFalse(first.resource_policy.allow_gpu)
        self.assertFalse(first.resource_policy.allow_capture)
        self.assertFalse(first.resource_policy.allow_hid_output)
        self.assertFalse(first.resource_policy.allow_network)
        self.assertFalse(first.resource_policy.allow_subprocess)
        self.assertTrue(first.resource_policy.allow_artifact_write)

        with self.assertRaises(ValueError):
            replace(
                first,
                precision=PrecisionConfig(
                    device="cuda",
                    mode="bfloat16",
                    allow_tf32=False,
                ),
            )

    def test_generic_control_has_one_exact_307_channel_training_order(self) -> None:
        axes = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
        control = GenericControl(
            keys_down=(0, 255),
            mouse_buttons=(0, 7),
            mouse_dx=12.5,
            mouse_dy=-3.25,
            mouse_wheel=2.0,
            gamepad_buttons=(0, 31),
            gamepad_axes=axes,
            impulse_sequence=7,
            intended_hold_ns=123_456,
        )
        vector = control_to_vector(control)

        self.assertEqual(CONTROL_LAYOUT_ID, "generic-hid-307-v1")
        self.assertEqual(len(vector), CONTROL_VECTOR_SIZE)
        self.assertEqual((vector[0], vector[255]), (1.0, 1.0))
        self.assertEqual((vector[256], vector[263]), (1.0, 1.0))
        self.assertEqual(vector[264:267], (12.5, -3.25, 2.0))
        self.assertEqual((vector[267], vector[298]), (1.0, 1.0))
        self.assertEqual(vector[299:307], axes)
        self.assertEqual(len(BUTTON_TARGET_INDICES), 296)
        self.assertEqual(len(set(BUTTON_TARGET_INDICES)), 296)
        self.assertEqual(CONTINUOUS_TARGET_INDICES, (264, 265, 266, *range(299, 307)))

        different_protocol_metadata = replace(
            control,
            impulse_sequence=99,
            intended_hold_ns=999_999,
        )
        self.assertNotEqual(
            control.canonical_bytes(),
            different_protocol_metadata.canonical_bytes(),
        )
        self.assertEqual(vector, control_to_vector(different_protocol_metadata))

    def test_batches_repeat_exactly_and_resume_at_a_batch_boundary(self) -> None:
        source = MovingShapesBatchSource(training_config().dataset)
        first_pass = tuple(
            source.iter_batches(
                split="train",
                epoch=4,
                start_batch=0,
                batch_size=2,
            )
        )
        repeated = tuple(
            source.iter_batches(
                split="train",
                epoch=4,
                start_batch=0,
                batch_size=2,
            )
        )
        resumed = tuple(
            source.iter_batches(
                split="train",
                epoch=4,
                start_batch=1,
                batch_size=2,
            )
        )

        def identities(
            batches: tuple[TrajectoryBatch, ...],
        ) -> tuple[tuple[str, ...], ...]:
            return tuple(
                tuple(sequence.content_sha256 for sequence in batch.sequences)
                for batch in batches
            )

        self.assertEqual(identities(first_pass), identities(repeated))
        self.assertEqual(identities(resumed), identities(first_pass)[1:])
        self.assertEqual(source.batches_per_epoch(split="train", batch_size=2), 2)
        self.assertEqual(first_pass[0].sample_count, 2)
        self.assertEqual(len(source.manifest_sha256), 64)

        validation = tuple(
            source.iter_batches(
                split="validation",
                epoch=999,
                start_batch=0,
                batch_size=2,
            )
        )
        self.assertEqual(
            tuple(sequence.sequence_index for sequence in validation[0].sequences),
            (0, 1),
        )

    def test_metric_writes_are_capability_gated_and_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            with self.assertRaises(ResourceDenied):
                JsonlMetricWriter(path, policy=ResourcePolicy.play_safe())
            self.assertFalse(path.exists())

            writer = JsonlMetricWriter(path, policy=training_config().resource_policy)
            record = MetricRecord(
                step=1,
                epoch=0,
                split="train",
                metrics={"z_loss": 2.0, "a_loss": 1.0},
            )
            writer.append(record)
            self.assertEqual(path.read_text(encoding="utf-8"), record.canonical_json + "\n")

    def test_foreground_trainer_advances_only_at_optimizer_boundaries(self) -> None:
        config = training_config(max_optimizer_steps=2)
        source = MovingShapesBatchSource(config.dataset)
        system = FakeTrainingSystem()

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                config,
                system,
                source,
                directory,
                code_sha256="c" * 64,
            )
            with patch(
                "irene_brain.training.trainer.save_checkpoint",
                side_effect=fake_save_checkpoint,
            ) as save_mock:
                summary = trainer.run()

            self.assertEqual(summary.cursor.optimizer_step, 2)
            self.assertEqual(system.train_calls, 2)
            self.assertEqual(system.evaluate_calls, 2)
            self.assertEqual(save_mock.call_count, 2)
            self.assertEqual(summary.checkpoint_sha256, "e" * 64)
            self.assertTrue(summary.last_checkpoint.exists())
            self.assertEqual(
                sorted(path.name for path in trainer.checkpoint_dir.glob("step-????????.pt")),
                ["step-00000002.pt"],
            )
            records = [
                json.loads(line)
                for line in summary.metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["split"] for record in records], [
                "train",
                "validation",
                "train",
                "validation",
            ])
            self.assertEqual([record["step"] for record in records], [1, 1, 2, 2])

    def test_resume_uses_saved_sampler_cursor_without_repeating_a_batch(self) -> None:
        config = training_config(max_optimizer_steps=2)
        source = MovingShapesBatchSource(config.dataset)
        expected_batch = next(
            source.iter_batches(
                split="train",
                epoch=0,
                start_batch=1,
                batch_size=1,
                max_batches=1,
            )
        )
        expected_indices = tuple(
            sequence.sequence_index for sequence in expected_batch.sequences
        )
        loaded = LoadedCheckpoint(
            cursor=TrainerCursor(epoch=0, next_batch=1, optimizer_step=1),
            system_state={"train_calls": 1},
            rng_state={"counter": 1},
            checkpoint_sha256="f" * 64,
        )
        system = FakeTrainingSystem()

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                config,
                system,
                source,
                directory,
                code_sha256="c" * 64,
            )
            with (
                patch(
                    "irene_brain.training.trainer.load_checkpoint",
                    return_value=loaded,
                ) as load_mock,
                patch(
                    "irene_brain.training.trainer.save_checkpoint",
                    side_effect=fake_save_checkpoint,
                ),
            ):
                summary = trainer.run(
                    resume_from=Path(directory) / "incoming.pt",
                    expected_checkpoint_sha256="f" * 64,
                )

        self.assertEqual(load_mock.call_count, 1)
        self.assertEqual(summary.cursor.optimizer_step, 2)
        self.assertEqual(system.train_calls, 2)
        self.assertEqual(system.restored_rng, {"counter": 1})
        self.assertEqual(system.trained_sequence_indices, [expected_indices])

    def test_resumed_run_never_prunes_its_resume_boundary(self) -> None:
        config = training_config(max_optimizer_steps=3)
        source = MovingShapesBatchSource(config.dataset)
        loaded = LoadedCheckpoint(
            cursor=TrainerCursor(epoch=0, next_batch=1, optimizer_step=1),
            system_state={"train_calls": 1},
            rng_state={"counter": 1},
            checkpoint_sha256="f" * 64,
        )

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                config,
                FakeTrainingSystem(),
                source,
                directory,
                code_sha256="c" * 64,
            )
            trainer.checkpoint_dir.mkdir(parents=True)
            resume_boundary = trainer.checkpoint_dir / "step-00000001.pt"
            boundary_bytes = b"immutable resume boundary"
            resume_boundary.write_bytes(boundary_bytes)

            with (
                patch(
                    "irene_brain.training.trainer.load_checkpoint",
                    return_value=loaded,
                ),
                patch(
                    "irene_brain.training.trainer.save_checkpoint",
                    side_effect=fake_save_checkpoint,
                ),
            ):
                summary = trainer.run(
                    resume_from=resume_boundary,
                    expected_checkpoint_sha256="f" * 64,
                )

            self.assertEqual(summary.cursor.optimizer_step, 3)
            self.assertEqual(resume_boundary.read_bytes(), boundary_bytes)
            self.assertEqual(
                sorted(path.name for path in trainer.checkpoint_dir.glob("step-????????.pt")),
                [
                    "step-00000001.pt",
                    "step-00000002.pt",
                    "step-00000003.pt",
                ],
            )

    def test_resume_rejects_checkpoint_at_or_beyond_training_budget(self) -> None:
        config = training_config(max_optimizer_steps=2)
        source = MovingShapesBatchSource(config.dataset)

        for optimizer_step in (2, 3):
            with (
                self.subTest(optimizer_step=optimizer_step),
                tempfile.TemporaryDirectory() as directory,
            ):
                system = FakeTrainingSystem()
                trainer = Trainer(
                    config,
                    system,
                    source,
                    directory,
                    code_sha256="c" * 64,
                )
                loaded = LoadedCheckpoint(
                    cursor=TrainerCursor(
                        epoch=0,
                        next_batch=1,
                        optimizer_step=optimizer_step,
                    ),
                    system_state={"train_calls": optimizer_step},
                    rng_state={"counter": optimizer_step},
                    checkpoint_sha256="f" * 64,
                )
                with (
                    patch(
                        "irene_brain.training.trainer.load_checkpoint",
                        return_value=loaded,
                    ),
                    patch(
                        "irene_brain.training.trainer.save_checkpoint",
                        side_effect=fake_save_checkpoint,
                    ) as save_mock,
                    self.assertRaisesRegex(ValueError, "reached or exceeded"),
                ):
                    trainer.run(
                        resume_from=Path(directory) / "incoming.pt",
                        expected_checkpoint_sha256="f" * 64,
                    )

                save_mock.assert_not_called()
                self.assertEqual(system.evaluate_calls, 0)
                self.assertEqual(system.trained_sequence_indices, [])

    def test_checkpoint_publication_atomically_refuses_a_racing_target(self) -> None:
        config = training_config()
        target_bytes = b"checkpoint published by another writer"

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "step-00000001.pt"

            def publish_racing_target(_payload: object, stream: Any) -> None:
                stream.write(b"candidate checkpoint")
                target.write_bytes(target_bytes)

            fake_torch = SimpleNamespace(save=publish_racing_target)
            with (
                patch(
                    "irene_brain.training.checkpoint._torch",
                    return_value=fake_torch,
                ),
                self.assertRaises(FileExistsError),
            ):
                save_checkpoint(
                    target,
                    cursor=TrainerCursor(optimizer_step=1),
                    system_state={},
                    rng_state={},
                    config_sha256=config.config_sha256,
                    data_sha256="d" * 64,
                    code_sha256="c" * 64,
                    runtime_fingerprint={"runtime": "fake-cpu", "world_size": 1},
                    policy=config.resource_policy,
                )

            self.assertEqual(target.read_bytes(), target_bytes)
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])


@unittest.skipUnless(torch is not None, "PyTorch is not installed in the play-safe runtime")
class TorchTrainingAndCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    @staticmethod
    def scheduler_config(scheduler_kind: str) -> TrainingConfig:
        base = training_config(max_optimizer_steps=500)
        return replace(
            base,
            schema_version=(
                1 if scheduler_kind == COSINE_AFTER_WARMUP else 2
            ),
            optimization=replace(
                base.optimization,
                learning_rate=1e-4,
                warmup_steps=20,
                scheduler_kind=scheduler_kind,
            ),
        )

    @staticmethod
    def scalar_system(config: TrainingConfig) -> Any:
        return TorchTrainingSystem(ScalarLossObjective(), config)

    def test_checkpoint_load_binds_digest_and_deserialization_to_one_inode(self) -> None:
        config = training_config()
        fingerprint = {"runtime": "test-cpu", "world_size": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "checkpoint.pt"
            replacement = root / "replacement.pt"
            digest = save_checkpoint(
                target,
                cursor=TrainerCursor(optimizer_step=1),
                system_state={},
                rng_state={},
                config_sha256=config.config_sha256,
                data_sha256="d" * 64,
                code_sha256="c" * 64,
                runtime_fingerprint=fingerprint,
                policy=config.resource_policy,
            )
            save_checkpoint(
                replacement,
                cursor=TrainerCursor(optimizer_step=2),
                system_state={},
                rng_state={},
                config_sha256=config.config_sha256,
                data_sha256="d" * 64,
                code_sha256="c" * 64,
                runtime_fingerprint=fingerprint,
                policy=config.resource_policy,
            )
            real_open = os.open
            swapped = False

            def swap_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
                nonlocal swapped
                if not swapped and Path(path) == target:
                    swapped = True
                    target.unlink()
                    replacement.replace(target)
                return real_open(path, flags, *args, **kwargs)

            with patch(
                "irene_brain.training.checkpoint.os.open",
                side_effect=swap_before_open,
            ):
                with self.assertRaisesRegex(ValueError, "changed while opening"):
                    load_checkpoint(
                        target,
                        expected_config_sha256=config.config_sha256,
                        expected_data_sha256="d" * 64,
                        expected_code_sha256="c" * 64,
                        expected_runtime_fingerprint=fingerprint,
                        expected_checkpoint_sha256=digest,
                    )

            hardlink = root / "checkpoint-hardlink.pt"
            os.link(target, hardlink)
            with self.assertRaisesRegex(ValueError, "exactly one hard link"):
                load_checkpoint(
                    target,
                    expected_config_sha256=config.config_sha256,
                    expected_data_sha256="d" * 64,
                    expected_code_sha256="c" * 64,
                    expected_runtime_fingerprint=fingerprint,
                )

    def make_system(
        self,
    ) -> tuple[Any, MovingShapesBatchSource, TrainingConfig]:
        assert torch is not None
        config = training_config(max_optimizer_steps=2)
        model_config = replace(
            ThoughtFieldConfig.smoke(),
            core_width=16,
            sensor_tokens=4,
            belief_tokens=2,
            working_memory_tokens=1,
            thoughtlets=4,
            goal_context_tokens=1,
            cognitive_cycles=2,
            brain_cell_blocks=1,
            attention_heads=2,
            routed_neighbors=1,
            episodic_memory_entries=8,
            retrieved_entries_per_thoughtlet=1,
        )
        model = IreneBrainModel(model_config, input_resolution=(8, 8), plan_steps=2)
        objective = ThoughtFieldObjective(model)
        system = TorchTrainingSystem(objective, config)
        source = MovingShapesBatchSource(config.dataset)
        return system, source, config

    def test_learning_rate_metric_is_the_rate_applied_before_scheduler_advance(
        self,
    ) -> None:
        assert torch is not None
        config = self.scheduler_config(CONSTANT_AFTER_WARMUP)
        system = self.scalar_system(config)
        batch = SimpleNamespace(sample_count=1)
        applied: list[float] = []
        handle = system.optimizer.register_step_pre_hook(
            lambda optimizer, _args, _kwargs: applied.append(
                float(optimizer.param_groups[0]["lr"])
            )
        )
        try:
            results = [system.train_optimizer_step((batch,)) for _ in range(21)]
        finally:
            handle.remove()

        expected = [
            config.optimization.learning_rate
            * learning_rate_multiplier(
                scheduler_kind=config.optimization.scheduler_kind,
                warmup_steps=config.optimization.warmup_steps,
                max_optimizer_steps=config.run.max_optimizer_steps,
                step=step,
            )
            for step in range(21)
        ]
        self.assertEqual(applied, expected)
        self.assertEqual(
            [result.metrics["learning_rate"] for result in results],
            expected,
        )
        self.assertEqual(
            {update: applied[update - 1] for update in (1, 10, 19, 20, 21)},
            {
                1: 0.000005,
                10: 0.00005,
                19: 0.000095,
                20: 0.0001,
                21: 0.0001,
            },
        )
        self.assertEqual(system.scheduler.last_epoch, 21)
        self.assertEqual(system.optimizer.param_groups[0]["lr"], 0.0001)

    def test_scheduler_checkpoint_resume_preserves_next_applied_lr_exactly(
        self,
    ) -> None:
        assert torch is not None
        batch = SimpleNamespace(sample_count=1)
        data_hash = "d" * 64
        code_hash = "c" * 64

        def assert_nested_equal(left: object, right: object) -> None:
            if isinstance(left, torch.Tensor):
                self.assertIsInstance(right, torch.Tensor)
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, Mapping):
                self.assertIsInstance(right, Mapping)
                assert isinstance(right, Mapping)
                self.assertEqual(set(left), set(right))
                for key in left:
                    assert_nested_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertIsInstance(right, type(left))
                assert isinstance(right, (list, tuple))
                self.assertEqual(len(left), len(right))
                for left_item, right_item in zip(left, right):
                    assert_nested_equal(left_item, right_item)
            else:
                self.assertEqual(left, right)

        for scheduler_kind in (COSINE_AFTER_WARMUP, CONSTANT_AFTER_WARMUP):
            with self.subTest(scheduler_kind=scheduler_kind):
                config = self.scheduler_config(scheduler_kind)
                uninterrupted = self.scalar_system(config)
                for _ in range(19):
                    uninterrupted.train_optimizer_step((batch,))
                self.assertEqual(uninterrupted.scheduler.last_epoch, 19)
                self.assertEqual(
                    uninterrupted.optimizer.param_groups[0]["lr"],
                    0.0001,
                )

                cursor = TrainerCursor(epoch=0, next_batch=19, optimizer_step=19)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "scheduler-checkpoint.pt"
                    digest = save_checkpoint(
                        path,
                        cursor=cursor,
                        system_state=uninterrupted.checkpoint_state(),
                        rng_state=uninterrupted.capture_rng_state(),
                        config_sha256=config.config_sha256,
                        data_sha256=data_hash,
                        code_sha256=code_hash,
                        runtime_fingerprint=uninterrupted.runtime_fingerprint,
                        policy=config.resource_policy,
                    )
                    resumed = self.scalar_system(config)
                    loaded = load_checkpoint(
                        path,
                        expected_config_sha256=config.config_sha256,
                        expected_data_sha256=data_hash,
                        expected_code_sha256=code_hash,
                        expected_runtime_fingerprint=resumed.runtime_fingerprint,
                        expected_checkpoint_sha256=digest,
                    )
                    self.assertEqual(loaded.cursor, cursor)
                    resumed.restore_checkpoint_state(loaded.system_state)
                    resumed.restore_rng_state(loaded.rng_state)

                self.assertEqual(resumed.scheduler.last_epoch, 19)
                self.assertEqual(resumed.optimizer.param_groups[0]["lr"], 0.0001)
                uninterrupted_applied: list[float] = []
                resumed_applied: list[float] = []
                uninterrupted_handle = uninterrupted.optimizer.register_step_pre_hook(
                    lambda optimizer, _args, _kwargs: uninterrupted_applied.append(
                        float(optimizer.param_groups[0]["lr"])
                    )
                )
                resumed_handle = resumed.optimizer.register_step_pre_hook(
                    lambda optimizer, _args, _kwargs: resumed_applied.append(
                        float(optimizer.param_groups[0]["lr"])
                    )
                )
                try:
                    uninterrupted_results = [
                        uninterrupted.train_optimizer_step((batch,))
                        for _ in range(3)
                    ]
                    resumed_results = [
                        resumed.train_optimizer_step((batch,)) for _ in range(3)
                    ]
                finally:
                    uninterrupted_handle.remove()
                    resumed_handle.remove()

                expected = [
                    config.optimization.learning_rate
                    * learning_rate_multiplier(
                        scheduler_kind=config.optimization.scheduler_kind,
                        warmup_steps=config.optimization.warmup_steps,
                        max_optimizer_steps=config.run.max_optimizer_steps,
                        step=step,
                    )
                    for step in range(19, 22)
                ]
                self.assertEqual(uninterrupted_applied, expected)
                self.assertEqual(resumed_applied, expected)
                self.assertEqual(
                    [
                        result.metrics["learning_rate"]
                        for result in uninterrupted_results
                    ],
                    expected,
                )
                self.assertEqual(
                    [result.metrics["learning_rate"] for result in resumed_results],
                    expected,
                )
                assert_nested_equal(
                    uninterrupted.checkpoint_state(),
                    resumed.checkpoint_state(),
                )

    def test_same_run_seed_constructs_the_exact_same_factory_checkpoint(self) -> None:
        assert torch is not None
        config = training_config()
        torch.manual_seed(config.run.seed)
        first = build_smoke_model(config)
        torch.manual_seed(config.run.seed)
        second = build_smoke_model(config)

        first_state = first.state_dict()
        second_state = second.state_dict()
        self.assertEqual(set(first_state), set(second_state))
        self.assertTrue(
            all(torch.equal(first_state[name], second_state[name]) for name in first_state)
        )

    def test_configured_factories_resolve_to_expected_model_scales(self) -> None:
        config = training_config()
        smoke_factory = _model_factory(
            "irene_brain.training.factory:build_smoke_model"
        )
        thesis_factory = _model_factory(
            "irene_brain.training.factory:build_thesis_model"
        )

        smoke = smoke_factory(config)
        thesis = thesis_factory(config)
        self.assertIsInstance(smoke, IreneBrainModel)
        self.assertIsInstance(thesis, IreneBrainModel)
        self.assertEqual(smoke.input_resolution, (32, 32))
        self.assertEqual(thesis.input_resolution, (64, 64))
        self.assertEqual(smoke.config, ThoughtFieldConfig.smoke())
        self.assertEqual(thesis.config, ThoughtFieldConfig.thesis_mvp())
        self.assertIsInstance(smoke.pixel_encoder.spatial_pool, torch.nn.Identity)
        self.assertIsInstance(thesis.pixel_encoder.spatial_pool, torch.nn.Identity)
        with torch.no_grad():
            smoke_grid = smoke.pixel_encoder.stem(torch.zeros(1, 3, 32, 32))
            thesis_grid = thesis.pixel_encoder.stem(torch.zeros(1, 3, 64, 64))
            smoke_tokens = smoke.pixel_encoder(torch.zeros(1, 3, 32, 32))
            thesis_tokens = thesis.pixel_encoder(torch.zeros(1, 3, 64, 64))
        self.assertEqual(smoke_grid.shape[-2:], (4, 4))
        self.assertEqual(thesis_grid.shape[-2:], (8, 8))
        self.assertEqual(smoke_tokens.shape, (1, 16, smoke.config.core_width))
        self.assertEqual(thesis_tokens.shape, (1, 64, thesis.config.core_width))

    def test_deterministic_eval_noise_is_distinct_without_global_rng_use(self) -> None:
        assert torch is not None
        rng_before = torch.get_rng_state().clone()
        first = deterministic_eval_thought_noise(
            thoughtlets=4,
            width=16,
            batch_size=2,
            device="cpu",
        )
        rng_between = torch.get_rng_state().clone()
        second = deterministic_eval_thought_noise(
            thoughtlets=4,
            width=16,
            batch_size=2,
            device="cpu",
        )
        rng_after = torch.get_rng_state().clone()

        self.assertEqual(first.shape, (2, 4, 16))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0], first[1]))
        self.assertGreater(float((first[:, 1:] - first[:, :1]).abs().max()), 1e-6)
        self.assertTrue(torch.equal(rng_before, rng_between))
        self.assertTrue(torch.equal(rng_before, rng_after))

    @staticmethod
    def action_prediction(button_logits: Any) -> SimpleNamespace:
        assert torch is not None
        batch_size = button_logits.shape[0]
        return SimpleNamespace(
            button_logits=button_logits,
            mouse_mean=torch.zeros(batch_size, 2),
            scroll_mean=torch.zeros(batch_size, 1),
            gamepad_axis_mean=torch.zeros(batch_size, 8),
        )

    def test_sparse_button_loss_rejects_all_off_and_all_four_shortcuts(self) -> None:
        assert torch is not None
        movement = tuple(int(key) for key in (HidKey.W, HidKey.A, HidKey.S, HidKey.D))
        target = torch.zeros(2, CONTROL_VECTOR_SIZE)
        target[0, int(HidKey.W)] = 1.0
        target[0, int(HidKey.D)] = 1.0
        target[1, int(HidKey.A)] = 1.0
        target[1, int(HidKey.S)] = 1.0
        exact_logits = torch.full((2, len(BUTTON_TARGET_INDICES)), -8.0)
        exact_logits[0, int(HidKey.W)] = 8.0
        exact_logits[0, int(HidKey.D)] = 8.0
        exact_logits[1, int(HidKey.A)] = 8.0
        exact_logits[1, int(HidKey.S)] = 8.0
        all_off_logits = torch.full_like(exact_logits, -8.0)
        all_four_logits = torch.full_like(exact_logits, -8.0)
        all_four_logits[:, movement] = 8.0
        indices = torch.tensor(BUTTON_TARGET_INDICES, dtype=torch.long)

        exact_loss = _structured_action_loss(
            self.action_prediction(exact_logits), target, indices
        )
        all_off_loss = _structured_action_loss(
            self.action_prediction(all_off_logits), target, indices
        )
        all_four_loss = _structured_action_loss(
            self.action_prediction(all_four_logits), target, indices
        )

        self.assertLess(float(exact_loss), 0.001)
        self.assertGreater(float(all_off_loss), float(exact_loss) + 3.0)
        self.assertGreater(float(all_four_loss), float(exact_loss) + 3.0)

    def test_movement_metrics_expose_shortcuts_as_additive_counts(self) -> None:
        assert torch is not None
        movement = torch.tensor(
            tuple(int(key) for key in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)),
            dtype=torch.long,
        )
        target = torch.zeros(3, 256, dtype=torch.bool)
        target[0, [int(HidKey.W), int(HidKey.D)]] = True
        target[1, [int(HidKey.A), int(HidKey.S)]] = True
        target[2, int(HidKey.W)] = True
        previous = torch.zeros(3, 256)
        previous[0, [int(HidKey.W), int(HidKey.D)]] = 1.0
        previous[1, [int(HidKey.W), int(HidKey.D)]] = 1.0
        previous[2, int(HidKey.W)] = 1.0

        exact_logits = torch.full((3, 256), -8.0)
        exact_logits[target] = 8.0
        all_four_logits = torch.full_like(exact_logits, -8.0)
        all_four_logits[:, movement] = 8.0
        all_off_logits = torch.full_like(exact_logits, -8.0)
        exact = _movement_metric_counts(exact_logits, target, previous, movement)
        all_four = _movement_metric_counts(all_four_logits, target, previous, movement)
        all_off = _movement_metric_counts(all_off_logits, target, previous, movement)

        self.assertEqual(int(exact["movement_exact_match"]), 3)
        self.assertEqual(int(all_four["movement_exact_match"]), 0)
        self.assertEqual(int(all_off["movement_exact_match"]), 0)
        self.assertEqual(int(all_four["movement_predicted_active_count"]), 12)
        self.assertEqual(int(all_four["movement_false_positive_count"]), 7)
        self.assertEqual(int(all_four["movement_opposite_conflict_rate"]), 3)
        self.assertEqual(int(all_four["non_movement_key_false_positive_count"]), 0)
        self.assertEqual(int(exact["previous_control_movement_exact_match"]), 2)
        self.assertEqual(int(exact["movement_changed_samples_per_sample"]), 1)
        self.assertEqual(int(exact["movement_changed_exact_matches_per_sample"]), 1)
        self.assertEqual(int(exact["all_off_movement_exact_match"]), 0)
        self.assertEqual(int(exact["all_four_movement_exact_match"]), 0)

        first = _movement_metric_counts(
            exact_logits[:1], target[:1], previous[:1], movement
        )
        remainder = _movement_metric_counts(
            exact_logits[1:], target[1:], previous[1:], movement
        )
        self.assertEqual(set(first), set(exact))
        self.assertTrue(
            all(
                torch.equal(first[name] + remainder[name], exact[name])
                for name in exact
            )
        )
        changed_exact = exact["movement_changed_exact_matches_per_sample"] / exact[
            "movement_changed_samples_per_sample"
        ]
        self.assertEqual(float(changed_exact), 1.0)

    def test_button_exact_set_count_covers_every_discrete_actuator_additively(self) -> None:
        assert torch is not None
        indices = torch.tensor(BUTTON_TARGET_INDICES, dtype=torch.long)
        target = torch.zeros(3, CONTROL_VECTOR_SIZE)
        logits = torch.full((3, len(BUTTON_TARGET_INDICES)), -8.0)

        target[0, int(HidKey.W)] = 1.0
        logits[0, int(HidKey.W)] = 8.0
        target[1, 256] = 1.0
        logits[1, 256] = 8.0
        target[2, 267] = 1.0

        combined = _button_exact_set_count(logits, target, indices)
        first = _button_exact_set_count(logits[:1], target[:1], indices)
        remainder = _button_exact_set_count(logits[1:], target[1:], indices)
        self.assertEqual(int(combined), 2)
        self.assertEqual(int(first + remainder), int(combined))

    def test_exact_set_metrics_use_a_strictly_positive_logit_threshold(self) -> None:
        assert torch is not None
        button_indices = torch.tensor(BUTTON_TARGET_INDICES, dtype=torch.long)
        movement = torch.tensor(
            tuple(int(key) for key in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)),
            dtype=torch.long,
        )
        target = torch.zeros(1, CONTROL_VECTOR_SIZE)
        button_logits = torch.full((1, len(BUTTON_TARGET_INDICES)), -8.0)

        button_logits[0, int(HidKey.W)] = 0.0
        self.assertEqual(
            int(_button_exact_set_count(button_logits, target, button_indices)),
            1,
        )
        self.assertEqual(
            int(
                _movement_exact_set_count(
                    button_logits[:, :256],
                    target[:, :256] > 0.5,
                    movement,
                )
            ),
            1,
        )

        target[0, int(HidKey.W)] = 1.0
        self.assertEqual(
            int(_button_exact_set_count(button_logits, target, button_indices)),
            0,
        )
        button_logits[0, int(HidKey.W)] = torch.finfo(button_logits.dtype).eps
        self.assertEqual(
            int(_button_exact_set_count(button_logits, target, button_indices)),
            1,
        )
        self.assertEqual(
            int(
                _movement_exact_set_count(
                    button_logits[:, :256],
                    target[:, :256] > 0.5,
                    movement,
                )
            ),
            1,
        )

    def test_final_logit_diagnostics_are_macro_additive_and_empty_safe(self) -> None:
        assert torch is not None
        movement = torch.tensor(
            tuple(int(key) for key in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)),
            dtype=torch.long,
        )
        target = torch.zeros(3, 256, dtype=torch.bool)
        logits = torch.full((3, 256), -5.0)

        target[0, [int(HidKey.W), int(HidKey.D)]] = True
        logits[0, int(HidKey.W)] = 2.0
        logits[0, int(HidKey.D)] = 4.0
        logits[0, int(HidKey.A)] = -1.0
        logits[0, int(HidKey.S)] = -3.0
        logits[0, 0] = 0.5

        logits[1, movement] = torch.tensor((0.1, 0.2, 0.3, 0.4))
        logits[1, 0] = -0.5

        target[2, movement] = True
        logits[2, movement] = torch.tensor((1.0, 2.0, 3.0, 4.0))
        logits[2, 0] = -0.25

        combined = _final_action_logit_metric_counts(logits, target, movement)
        first = _final_action_logit_metric_counts(
            logits[:1], target[:1], movement
        )
        remainder = _final_action_logit_metric_counts(
            logits[1:], target[1:], movement
        )
        self.assertEqual(
            set(combined),
            {
                "final_positive_key_logit_mean",
                "final_inactive_movement_key_logit_max",
                "final_inactive_movement_key_logit_mean",
                "final_non_movement_key_logit_max",
            },
        )
        self.assertTrue(
            all(
                torch.allclose(first[name] + remainder[name], combined[name])
                for name in combined
            )
        )
        self.assertAlmostEqual(float(combined["final_positive_key_logit_mean"]), 5.5)
        self.assertAlmostEqual(
            float(combined["final_inactive_movement_key_logit_max"]),
            -0.6,
            places=6,
        )
        self.assertAlmostEqual(
            float(combined["final_inactive_movement_key_logit_mean"]),
            -1.75,
        )
        self.assertAlmostEqual(
            float(combined["final_non_movement_key_logit_max"]),
            -0.25,
        )

    def test_per_exit_metrics_aggregate_by_supervised_sample(self) -> None:
        assert torch is not None
        system, source, _config = self.make_system()
        combined_batch = next(
            source.iter_batches(
                split="train",
                epoch=0,
                start_batch=0,
                batch_size=3,
                max_batches=1,
            )
        )
        split_batches = (
            TrajectoryBatch(
                split=combined_batch.split,
                burn_in_steps=combined_batch.burn_in_steps,
                sequences=combined_batch.sequences[:1],
            ),
            TrajectoryBatch(
                split=combined_batch.split,
                burn_in_steps=combined_batch.burn_in_steps,
                sequences=combined_batch.sequences[1:],
            ),
        )
        with torch.no_grad():
            combined = system.objective(combined_batch)
            separate = tuple(system.objective(batch) for batch in split_batches)

        self.assertEqual(combined.samples, sum(result.samples for result in separate))
        for exit_index in range(system.objective.model.config.cognitive_cycles + 1):
            for suffix in (
                "action_loss",
                "button_exact_match",
                "movement_exact_match",
            ):
                name = f"exit_{exit_index}_{suffix}"
                expected = sum(
                    result.metrics[name] * result.samples for result in separate
                ) / combined.samples
                self.assertTrue(
                    torch.allclose(combined.metrics[name], expected, atol=1e-6, rtol=1e-6),
                    name,
                )

    def test_real_objective_performs_one_bounded_cpu_optimizer_step(self) -> None:
        assert torch is not None
        system, source, _config = self.make_system()
        self.assertTrue(
            all(
                type(value) in {str, int, bool}
                for value in system.runtime_fingerprint.values()
            )
        )
        batch = next(
            source.iter_batches(
                split="train",
                epoch=0,
                start_batch=0,
                batch_size=1,
                max_batches=1,
            )
        )
        parameter = (
            system.objective.model.brain_cell.blocks[0]
            .thought_attention.feed_forward[0]
            .weight
        )
        before = parameter.detach().clone()
        result = system.train_optimizer_step((batch,))

        self.assertTrue(math.isfinite(result.loss))
        self.assertEqual(result.samples, 1)
        exit_metric_names = {
            name for name in result.metrics if name.startswith("exit_")
        }
        expected_exit_metric_names = {
            f"exit_{exit_index}_{suffix}"
            for exit_index in range(system.objective.model.config.cognitive_cycles + 1)
            for suffix in (
                "action_loss",
                "button_exact_match",
                "movement_exact_match",
            )
        }
        self.assertEqual(exit_metric_names, expected_exit_metric_names)
        per_exit_action_losses = tuple(
            result.metrics[f"exit_{exit_index}_action_loss"]
            for exit_index in range(system.objective.model.config.cognitive_cycles + 1)
        )
        self.assertAlmostEqual(
            result.metrics["action_loss"],
            sum(per_exit_action_losses) / len(per_exit_action_losses),
            places=6,
        )
        self.assertTrue(
            {
                "action_loss",
                "value_loss",
                "world_loss",
                "diversity_loss",
                "movement_exact_match",
                "movement_predicted_active_count",
                "movement_target_active_count",
                "movement_false_positive_count",
                "non_movement_key_false_positive_count",
                "movement_opposite_conflict_rate",
                "previous_control_movement_exact_match",
                "movement_changed_samples_per_sample",
                "movement_changed_exact_matches_per_sample",
                "all_off_movement_exact_match",
                "all_four_movement_exact_match",
                "movement_w_true_positive_rate",
                "movement_w_predicted_positive_rate",
                "movement_w_target_positive_rate",
                "final_positive_key_logit_mean",
                "final_inactive_movement_key_logit_max",
                "final_inactive_movement_key_logit_mean",
                "final_non_movement_key_logit_max",
                "exit_0_action_loss",
                "exit_0_button_exact_match",
                "exit_0_movement_exact_match",
                "exit_1_action_loss",
                "exit_1_button_exact_match",
                "exit_1_movement_exact_match",
                "exit_2_action_loss",
                "exit_2_button_exact_match",
                "exit_2_movement_exact_match",
            }.issubset(
                result.metrics
            )
        )
        self.assertTrue(all(math.isfinite(value) for value in result.metrics.values()))
        self.assertFalse(torch.equal(before, parameter.detach()))
        self.assertEqual(system.device.type, "cpu")

    def test_evaluation_is_repeatable_and_does_not_consume_rng(self) -> None:
        assert torch is not None
        system, source, _config = self.make_system()
        batch = next(
            source.iter_batches(
                split="validation",
                epoch=0,
                start_batch=0,
                batch_size=1,
                max_batches=1,
            )
        )
        fingerprint = system.runtime_fingerprint
        self.assertEqual(
            fingerprint["cublas_workspace_config"],
            os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
        )
        self.assertEqual(fingerprint["sdpa_policy"], "runtime_default")

        thought_summaries: list[Any] = []
        handle = system.objective.model.register_forward_hook(
            lambda _module, _inputs, output: thought_summaries.append(
                output.diagnostics.thought_summaries.detach().clone()
            )
        )
        rng_before = torch.get_rng_state().clone()
        try:
            first = system.evaluate_batch(batch)
            rng_between = torch.get_rng_state().clone()
            second = system.evaluate_batch(batch)
            rng_after = torch.get_rng_state().clone()
        finally:
            handle.remove()

        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng_before, rng_between))
        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertEqual(len(thought_summaries), batch.sequence_length * 2)
        first_initial = thought_summaries[0]
        second_initial = thought_summaries[batch.sequence_length]
        self.assertTrue(torch.equal(first_initial, second_initial))
        self.assertGreater(
            float((first_initial[:, 1:] - first_initial[:, :1]).abs().max()),
            1e-6,
        )

    def test_checkpoint_round_trip_restores_one_system_and_exact_rng(self) -> None:
        assert torch is not None
        system, source, config = self.make_system()
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
        cursor = TrainerCursor(epoch=0, next_batch=1, optimizer_step=1)
        rng_state = system.capture_rng_state()
        target = (
            system.objective.model.brain_cell.blocks[0]
            .thought_attention.feed_forward[0]
            .weight
        )
        expected_parameter = target.detach().clone()
        expected_runtime = dict(system.runtime_fingerprint)
        code_hash = "c" * 64

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            digest = save_checkpoint(
                path,
                cursor=cursor,
                system_state=system.checkpoint_state(),
                rng_state=rng_state,
                config_sha256=config.config_sha256,
                data_sha256=source.manifest_sha256,
                code_sha256=code_hash,
                runtime_fingerprint=expected_runtime,
                policy=config.resource_policy,
            )
            self.assertEqual(len(digest), 64)

            system.restore_rng_state(rng_state)
            expected_random = torch.rand(4)
            with torch.no_grad():
                target.add_(10.0)
            torch.manual_seed(999)

            loaded = load_checkpoint(
                path,
                expected_config_sha256=config.config_sha256,
                expected_data_sha256=source.manifest_sha256,
                expected_code_sha256=code_hash,
                expected_runtime_fingerprint=expected_runtime,
                expected_checkpoint_sha256=digest,
            )
            self.assertEqual(loaded.cursor, cursor)
            system.restore_checkpoint_state(loaded.system_state)
            system.restore_rng_state(loaded.rng_state)
            actual_random = torch.rand(4)

            self.assertTrue(torch.equal(target.detach(), expected_parameter))
            self.assertTrue(torch.equal(actual_random, expected_random))
            self.assertEqual(
                set(loaded.system_state),
                {"objective", "optimizer", "scheduler", "scaler"},
            )

            with self.assertRaises(ValueError):
                load_checkpoint(
                    path,
                    expected_config_sha256="d" * 64,
                    expected_data_sha256=source.manifest_sha256,
                    expected_code_sha256=code_hash,
                    expected_runtime_fingerprint=expected_runtime,
                )


if __name__ == "__main__":
    unittest.main()
