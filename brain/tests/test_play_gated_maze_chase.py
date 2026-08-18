from __future__ import annotations

from pathlib import Path
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.training.batches import (
    MazeChaseBatchSource,
    MovingShapesBatchSource,
    dataset_batch_source,
)
from irene_brain.training.config import DatasetConfig, load_training_config
from irene_brain.training.play_gate import (
    CAMPAIGN_ID,
    NOOP_COLLISION_FLOOR,
    NOOP_REWARD_FLOOR,
    PLAY_SEEDS,
    PLAY_TICKS,
)


def _moving() -> DatasetConfig:
    return DatasetConfig(
        kind="moving_shapes",
        train_sequences=4,
        validation_sequences=2,
        test_sequences=2,
        sequence_length=8,
        burn_in_steps=2,
        seed_offset=0,
        hazard_count=3,
        tick_period_ns=16_666_667,
        discount=0.99,
    )


def _maze() -> DatasetConfig:
    return DatasetConfig(
        kind="maze_chase",
        train_sequences=4,
        validation_sequences=2,
        test_sequences=2,
        sequence_length=8,
        burn_in_steps=2,
        seed_offset=0,
        hazard_count=3,
        tick_period_ns=16_666_667,
        discount=0.99,
    )


class PlayGatedMazeChaseConfigTests(unittest.TestCase):
    def test_dataset_kind_accepts_maze_chase_and_rejects_unknown(self) -> None:
        self.assertEqual(_maze().kind, "maze_chase")
        with self.assertRaises(ValueError):
            DatasetConfig(
                kind="pacman",
                train_sequences=4,
                validation_sequences=2,
                test_sequences=2,
                sequence_length=8,
                burn_in_steps=2,
                seed_offset=0,
                hazard_count=3,
                tick_period_ns=16_666_667,
                discount=0.99,
            )

    def test_dataset_batch_source_selects_the_named_world(self) -> None:
        moving = dataset_batch_source(_moving())
        maze = dataset_batch_source(_maze())
        self.assertIsInstance(moving, MovingShapesBatchSource)
        self.assertIsInstance(maze, MazeChaseBatchSource)
        self.assertNotEqual(moving.manifest_sha256, maze.manifest_sha256)

    def test_probe_recipe_parses_as_schema_two_maze_chase(self) -> None:
        config = load_training_config(
            ROOT / "configs" / "training" / "dgx-play-maze-chase-distill-probe.toml"
        )
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(config.dataset.kind, "maze_chase")
        self.assertEqual(config.run.max_optimizer_steps, 32)
        self.assertEqual(
            config.run.model_factory,
            "irene_brain.training.factory:build_thesis_model",
        )
        self.assertEqual(config.optimization.scheduler_kind, "constant_after_warmup")
        self.assertEqual(config.precision.device, "cuda")
        source = dataset_batch_source(config.dataset)
        self.assertIsInstance(source, MazeChaseBatchSource)

    def test_play_gate_floor_is_frozen(self) -> None:
        self.assertEqual(CAMPAIGN_ID, "play_gated_maze_chase_distill_v1")
        self.assertEqual(PLAY_SEEDS, (5, 9))
        self.assertEqual(PLAY_TICKS, 240)
        self.assertEqual(NOOP_REWARD_FLOOR, -161.0)
        self.assertEqual(NOOP_COLLISION_FLOOR, 17)

    def test_play_eval_does_not_hardcode_cpu_device(self) -> None:
        import inspect

        from irene_brain.evaluation.closed_loop_play import evaluate_closed_loop_play

        source = inspect.getsource(evaluate_closed_loop_play)
        self.assertNotIn('torch.device("cpu")', source)
        self.assertIn("parameters()", source)


if __name__ == "__main__":
    unittest.main()
