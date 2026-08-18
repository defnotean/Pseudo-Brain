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


def _batch_config(**overrides: object) -> MazeChaseBatchConfig:
    knobs: dict[str, object] = {
        "train_sequences": 4,
        "validation_sequences": 2,
        "test_sequences": 2,
        "sequence_length": 8,
        "burn_in_steps": 1,
    }
    knobs.update(overrides)
    return MazeChaseBatchConfig(**knobs)  # type: ignore[arg-type]


class MazeChaseBatchConfigTests(unittest.TestCase):
    def test_validation_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            _batch_config(train_sequences=0)
        with self.assertRaises(ValueError):
            _batch_config(sequence_length=1)
        with self.assertRaises(ValueError):
            _batch_config(sequence_length=8, burn_in_steps=8)
        with self.assertRaises(ValueError):
            _batch_config(ghost_count=0)
        with self.assertRaises(ValueError):
            _batch_config(ghost_rule="random")
        with self.assertRaises(ValueError):
            _batch_config(ghost_elroy=1)
        with self.assertRaises(ValueError):
            _batch_config(sticky_direction=0)
        with self.assertRaises(ValueError):
            _batch_config(input_delay_ticks=17)
        with self.assertRaises(ValueError):
            _batch_config(discount=1.5)
        with self.assertRaises(ValueError):
            _batch_config(seed_offset=(1 << 62) - 1, train_sequences=4)

    def test_source_rejects_wrong_config_and_split(self) -> None:
        with self.assertRaises(ValueError):
            MazeChaseBatchSource(object())  # type: ignore[arg-type]
        source = MazeChaseBatchSource(_batch_config())
        with self.assertRaises(ValueError):
            source.batches_per_epoch(split="heldout", batch_size=1)


class MazeChaseBatchSourceTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_knob_sensitive(self) -> None:
        baseline = MazeChaseBatchSource(_batch_config()).manifest_sha256
        self.assertEqual(
            MazeChaseBatchSource(_batch_config()).manifest_sha256, baseline
        )
        for override in (
            {"ghost_rule": "ambush"},
            {"ghost_elroy": True},
            {"input_delay_ticks": 2},
            {"burn_in_steps": 2},
        ):
            self.assertNotEqual(
                MazeChaseBatchSource(_batch_config(**override)).manifest_sha256,
                baseline,
                msg=f"manifest must change with {override}",
            )

    def test_manifest_differs_from_the_moving_shapes_source(self) -> None:
        moving = MovingShapesBatchSource(
            DatasetConfig(
                kind="moving_shapes",
                train_sequences=4,
                validation_sequences=2,
                test_sequences=2,
                sequence_length=8,
                burn_in_steps=1,
                seed_offset=0,
                hazard_count=1,
                tick_period_ns=16_666_667,
                discount=0.99,
            )
        )
        maze = MazeChaseBatchSource(_batch_config())
        self.assertNotEqual(moving.manifest_sha256, maze.manifest_sha256)

    def test_batches_repeat_exactly_and_resume_at_a_batch_boundary(self) -> None:
        source = MazeChaseBatchSource(_batch_config())
        self.assertEqual(source.batches_per_epoch(split="train", batch_size=2), 2)
        first_epoch = list(
            source.iter_batches(split="train", epoch=0, start_batch=0, batch_size=2)
        )
        repeat = list(
            source.iter_batches(split="train", epoch=0, start_batch=0, batch_size=2)
        )
        self.assertEqual(len(first_epoch), 2)
        for batch, again in zip(first_epoch, repeat):
            self.assertIsInstance(batch, TrajectoryBatch)
            self.assertEqual(batch.sample_count, 2 * 7)
            self.assertEqual(
                [sequence.content_sha256 for sequence in batch.sequences],
                [sequence.content_sha256 for sequence in again.sequences],
            )
        tail = list(
            source.iter_batches(split="train", epoch=0, start_batch=1, batch_size=2)
        )
        self.assertEqual(
            [sequence.content_sha256 for sequence in tail[0].sequences],
            [sequence.content_sha256 for sequence in first_epoch[1].sequences],
        )

    def test_validation_split_is_not_shuffled(self) -> None:
        source = MazeChaseBatchSource(_batch_config())
        batches = list(
            source.iter_batches(
                split="validation", epoch=5, start_batch=0, batch_size=1
            )
        )
        self.assertEqual(
            [batch.sequences[0].sequence_index for batch in batches], [0, 1]
        )


@unittest.skipIf(torch is None, "PyTorch is optional for the play-safe suite")
class MazeChaseTrainerSmokeTests(unittest.TestCase):
    """One bounded CPU optimizer step on planner-teacher maze_chase batches."""

    def _training_config(self) -> TrainingConfig:
        return TrainingConfig(
            schema_version=1,
            run=RunConfig(
                name="maze-chase-smoke",
                seed=20260817,
                model_factory="irene_brain.model.torch_model:IreneBrainModel",
                max_optimizer_steps=2,
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

    def test_real_objective_steps_on_maze_chase_batches(self) -> None:
        assert torch is not None
        config = self._training_config()
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
        source = MazeChaseBatchSource(_batch_config())
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


if __name__ == "__main__":
    unittest.main()
