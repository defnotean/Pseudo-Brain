from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.training.batches import (
    MazeChaseBatchConfig,
    MazeChaseBatchSource,
    MixedWorldBatchConfig,
    MixedWorldBatchSource,
    MovingShapesBatchSource,
    TrajectoryBatch,
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
)

try:
    import torch
except ModuleNotFoundError:  # Phase 0's stdlib-only environment remains supported.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem


def _moving_config(**overrides: object) -> DatasetConfig:
    knobs: dict[str, object] = {
        "kind": "moving_shapes",
        "train_sequences": 4,
        "validation_sequences": 2,
        "test_sequences": 2,
        "sequence_length": 8,
        "burn_in_steps": 1,
        "seed_offset": 0,
        "hazard_count": 1,
        "tick_period_ns": 16_666_667,
        "discount": 0.99,
    }
    knobs.update(overrides)
    return DatasetConfig(**knobs)  # type: ignore[arg-type]


def _maze_config(**overrides: object) -> MazeChaseBatchConfig:
    knobs: dict[str, object] = {
        "train_sequences": 4,
        "validation_sequences": 2,
        "test_sequences": 2,
        "sequence_length": 8,
        "burn_in_steps": 1,
    }
    knobs.update(overrides)
    return MazeChaseBatchConfig(**knobs)  # type: ignore[arg-type]


def _mixed_config(
    *,
    moving: DatasetConfig | None = None,
    maze: MazeChaseBatchConfig | None = None,
) -> MixedWorldBatchConfig:
    return MixedWorldBatchConfig(
        moving_shapes=moving if moving is not None else _moving_config(),
        maze_chase=maze if maze is not None else _maze_config(),
    )


class MixedWorldBatchConfigTests(unittest.TestCase):
    def test_validation_fails_closed_on_mismatched_members(self) -> None:
        with self.assertRaises(ValueError):
            MixedWorldBatchConfig(
                moving_shapes=object(),  # type: ignore[arg-type]
                maze_chase=_maze_config(),
            )
        with self.assertRaises(ValueError):
            MixedWorldBatchConfig(
                moving_shapes=_moving_config(),
                maze_chase=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _mixed_config(maze=_maze_config(train_sequences=8))
        with self.assertRaises(ValueError):
            _mixed_config(maze=_maze_config(sequence_length=16))
        with self.assertRaises(ValueError):
            _mixed_config(maze=_maze_config(burn_in_steps=2))
        with self.assertRaises(ValueError):
            _mixed_config(maze=_maze_config(discount=0.5))

    def test_source_rejects_wrong_config_and_split(self) -> None:
        with self.assertRaises(ValueError):
            MixedWorldBatchSource(object())  # type: ignore[arg-type]
        source = MixedWorldBatchSource(_mixed_config())
        with self.assertRaises(ValueError):
            source.batches_per_epoch(split="heldout", batch_size=1)


class MixedWorldBatchSourceTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_knob_sensitive(self) -> None:
        baseline = MixedWorldBatchSource(_mixed_config()).manifest_sha256
        self.assertEqual(
            MixedWorldBatchSource(_mixed_config()).manifest_sha256, baseline
        )
        self.assertNotEqual(
            MixedWorldBatchSource(
                _mixed_config(maze=_maze_config(ghost_rule="ambush"))
            ).manifest_sha256,
            baseline,
        )
        self.assertNotEqual(
            MixedWorldBatchSource(
                _mixed_config(moving=_moving_config(hazard_count=3))
            ).manifest_sha256,
            baseline,
        )

    def test_manifest_differs_from_either_member_source(self) -> None:
        mixed = MixedWorldBatchSource(_mixed_config())
        moving = MovingShapesBatchSource(_moving_config())
        maze = MazeChaseBatchSource(_maze_config())
        self.assertNotEqual(mixed.manifest_sha256, moving.manifest_sha256)
        self.assertNotEqual(mixed.manifest_sha256, maze.manifest_sha256)
        self.assertNotEqual(moving.manifest_sha256, maze.manifest_sha256)

    def test_unshuffled_validation_round_robins_both_worlds(self) -> None:
        source = MixedWorldBatchSource(_mixed_config())
        manifests = source.split_world_manifests("validation")
        batches = list(
            source.iter_batches(
                split="validation", epoch=5, start_batch=0, batch_size=1
            )
        )
        self.assertEqual(len(batches), 4)
        worlds = [batch.sequences[0].manifest_sha256 for batch in batches]
        self.assertEqual(
            worlds,
            [
                manifests["moving_shapes"],
                manifests["maze_chase"],
                manifests["moving_shapes"],
                manifests["maze_chase"],
            ],
        )
        self.assertEqual(
            [batch.sequences[0].sequence_index for batch in batches],
            [0, 0, 1, 1],
        )

    def test_batches_repeat_exactly_and_resume_at_a_batch_boundary(self) -> None:
        source = MixedWorldBatchSource(_mixed_config())
        self.assertEqual(source.batches_per_epoch(split="train", batch_size=2), 4)
        first_epoch = list(
            source.iter_batches(split="train", epoch=0, start_batch=0, batch_size=2)
        )
        repeat = list(
            source.iter_batches(split="train", epoch=0, start_batch=0, batch_size=2)
        )
        self.assertEqual(len(first_epoch), 4)
        for batch, again in zip(first_epoch, repeat):
            self.assertIsInstance(batch, TrajectoryBatch)
            self.assertEqual(batch.sample_count, 2 * 7)
            self.assertEqual(
                [sequence.content_sha256 for sequence in batch.sequences],
                [sequence.content_sha256 for sequence in again.sequences],
            )
        tail = list(
            source.iter_batches(split="train", epoch=0, start_batch=3, batch_size=2)
        )
        self.assertEqual(
            [sequence.content_sha256 for sequence in tail[0].sequences],
            [sequence.content_sha256 for sequence in first_epoch[3].sequences],
        )

    def test_train_epoch_contains_both_worlds(self) -> None:
        source = MixedWorldBatchSource(_mixed_config())
        manifests = source.split_world_manifests("train")
        seen = {
            batch.sequences[0].manifest_sha256
            for batch in source.iter_batches(
                split="train", epoch=0, start_batch=0, batch_size=1
            )
        }
        self.assertEqual(seen, set(manifests.values()))


@unittest.skipIf(torch is None, "PyTorch is optional for the play-safe suite")
class TaskSpecialistSmokeTests(unittest.TestCase):
    """B3: unchanged generalist versus per-world fine-tune, one CPU loop."""

    def _training_config(self) -> TrainingConfig:
        return TrainingConfig(
            schema_version=2,
            run=RunConfig(
                name="task-specialist-smoke",
                seed=20260818,
                model_factory="irene_brain.model.torch_model:IreneBrainModel",
                max_optimizer_steps=2,
            ),
            dataset=_moving_config(sequence_length=2, train_sequences=2),
            optimization=OptimizationConfig(
                batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                max_gradient_norm=1.0,
                warmup_steps=0,
                scheduler_kind="constant_after_warmup",
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

    def _model(self) -> IreneBrainModel:
        assert torch is not None
        config = replace(
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
        return IreneBrainModel(config, input_resolution=(8, 8), plan_steps=2)

    def _system(self, model: IreneBrainModel) -> TorchTrainingSystem:
        return TorchTrainingSystem(ThoughtFieldObjective(model), self._training_config())

    @staticmethod
    def _snapshot(model: IreneBrainModel) -> dict[str, object]:
        return {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }

    def _evaluate(
        self, system: TorchTrainingSystem, source: object
    ) -> dict[str, float]:
        batch = next(
            source.iter_batches(
                split="validation",
                epoch=0,
                start_batch=0,
                batch_size=1,
                max_batches=1,
            )
        )
        result = system.evaluate_batch(batch)
        return dict(result.metrics)

    def test_real_objective_steps_on_mixed_batches(self) -> None:
        assert torch is not None
        model = self._model()
        system = self._system(model)
        source = MixedWorldBatchSource(_mixed_config())
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
        self.assertEqual(result.samples, batch.sample_count)
        self.assertGreater(result.metrics["action_loss"], 0.0)
        self.assertFalse(torch.equal(before, parameter.detach()))

    def test_specialists_fine_tune_from_an_unchanged_generalist_snapshot(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        torch.manual_seed(20260818)
        mixed = MixedWorldBatchSource(_mixed_config())
        shapes = MovingShapesBatchSource(_moving_config())
        maze = MazeChaseBatchSource(_maze_config())

        generalist_model = self._model()
        generalist = self._system(generalist_model)
        for step in range(2):
            batch = next(
                mixed.iter_batches(
                    split="train",
                    epoch=0,
                    start_batch=step,
                    batch_size=1,
                    max_batches=1,
                )
            )
            result = generalist.train_optimizer_step((batch,))
            self.assertTrue(math.isfinite(result.loss))
        snapshot = self._snapshot(generalist_model)

        shapes_model = self._model()
        shapes_model.load_state_dict(snapshot, strict=True)
        shapes_system = self._system(shapes_model)
        maze_model = self._model()
        maze_model.load_state_dict(snapshot, strict=True)
        maze_system = self._system(maze_model)

        shapes_batch = next(
            shapes.iter_batches(
                split="train", epoch=0, start_batch=0, batch_size=1, max_batches=1
            )
        )
        maze_batch = next(
            maze.iter_batches(
                split="train", epoch=0, start_batch=0, batch_size=1, max_batches=1
            )
        )
        shapes_result = shapes_system.train_optimizer_step((shapes_batch,))
        maze_result = maze_system.train_optimizer_step((maze_batch,))
        self.assertTrue(math.isfinite(shapes_result.loss))
        self.assertTrue(math.isfinite(maze_result.loss))

        generalist_reload = {
            name: tensor.detach()
            for name, tensor in generalist_model.state_dict().items()
        }
        for name, tensor in snapshot.items():
            self.assertTrue(torch.equal(generalist_reload[name], tensor))

        shapes_eval = self._evaluate(shapes_system, shapes)
        maze_eval = self._evaluate(maze_system, maze)
        generalist_shapes = self._evaluate(generalist, shapes)
        generalist_maze = self._evaluate(generalist, maze)
        for row in (shapes_eval, maze_eval, generalist_shapes, generalist_maze):
            self.assertTrue(math.isfinite(row["action_loss"]))
            self.assertGreater(row["action_loss"], 0.0)

        shapes_changed = any(
            not torch.equal(
                shapes_model.state_dict()[name], snapshot[name]  # type: ignore[arg-type]
            )
            for name in snapshot
        )
        maze_changed = any(
            not torch.equal(
                maze_model.state_dict()[name], snapshot[name]  # type: ignore[arg-type]
            )
            for name in snapshot
        )
        self.assertTrue(shapes_changed)
        self.assertTrue(maze_changed)


if __name__ == "__main__":
    unittest.main()
