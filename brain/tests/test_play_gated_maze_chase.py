from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.training.batches import (
    MazeChaseBatchSource,
    MovingShapesBatchSource,
    dataset_batch_source,
)
from irene_brain.training.config import DatasetConfig, load_training_config
from irene_brain.training.play_gate import (
    CAMPAIGN_ID,
    CAMPAIGN_PELLET_FLOOR,
    NOOP_COLLISION_FLOOR,
    NOOP_REWARD_FLOOR,
    PLAY_PEAK_DROP_PELLETS,
    PLAY_SEEDS,
    PLAY_TICKS,
    STICKY_PELLET_BAND,
    is_one_key_wasd,
    play_peak_should_stop,
    play_score,
)
from irene_brain.types import GenericControl, HidKey


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

    def test_one_twenty_eight_step_probe_keeps_the_same_recipe(self) -> None:
        short = load_training_config(
            ROOT / "configs" / "training" / "dgx-play-maze-chase-distill-probe.toml"
        )
        longer = load_training_config(
            ROOT / "configs" / "training" / "dgx-play-maze-chase-distill-probe-128.toml"
        )
        self.assertEqual(longer.schema_version, 2)
        self.assertEqual(longer.dataset.kind, "maze_chase")
        self.assertEqual(longer.run.max_optimizer_steps, 128)
        self.assertEqual(longer.run.seed, short.run.seed)
        self.assertEqual(longer.run.model_factory, short.run.model_factory)
        self.assertEqual(longer.optimization.scheduler_kind, short.optimization.scheduler_kind)
        self.assertNotEqual(short.config_sha256, longer.config_sha256)

    def test_window32_probe_keeps_the_step_budget_and_lengthens_teacher(self) -> None:
        short = load_training_config(
            ROOT / "configs" / "training" / "dgx-play-maze-chase-distill-probe.toml"
        )
        windowed = load_training_config(
            ROOT / "configs" / "training" / "dgx-play-maze-chase-distill-window32.toml"
        )
        self.assertEqual(windowed.schema_version, 2)
        self.assertEqual(windowed.dataset.kind, "maze_chase")
        self.assertEqual(windowed.run.max_optimizer_steps, 32)
        self.assertEqual(windowed.dataset.sequence_length, 32)
        self.assertEqual(short.dataset.sequence_length, 8)
        self.assertEqual(windowed.run.seed, short.run.seed)
        self.assertEqual(windowed.run.model_factory, short.run.model_factory)
        self.assertEqual(
            windowed.optimization.scheduler_kind, short.optimization.scheduler_kind
        )
        self.assertNotEqual(short.config_sha256, windowed.config_sha256)
        source = dataset_batch_source(windowed.dataset)
        self.assertIsInstance(source, MazeChaseBatchSource)

    def test_episode_windows_probe_keeps_eight_tick_windows_and_later_horizon(self) -> None:
        short = load_training_config(
            ROOT / "configs" / "training" / "dgx-play-maze-chase-distill-probe.toml"
        )
        windows = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-episode-windows.toml"
        )
        self.assertEqual(windows.schema_version, 2)
        self.assertEqual(windows.dataset.kind, "maze_chase")
        self.assertEqual(windows.run.max_optimizer_steps, 32)
        self.assertEqual(windows.dataset.sequence_length, 8)
        self.assertEqual(windows.dataset.episode_horizon, 240)
        self.assertEqual(short.dataset.episode_horizon, 0)
        self.assertEqual(windows.run.seed, short.run.seed)
        self.assertEqual(windows.run.model_factory, short.run.model_factory)
        self.assertEqual(
            windows.optimization.scheduler_kind, short.optimization.scheduler_kind
        )
        self.assertNotEqual(short.config_sha256, windows.config_sha256)
        self.assertNotIn("episode_horizon", short.to_dict()["dataset"])
        self.assertEqual(windows.to_dict()["dataset"]["episode_horizon"], 240)
        source = dataset_batch_source(windows.dataset)
        self.assertIsInstance(source, MazeChaseBatchSource)
        self.assertEqual(
            windows.config_sha256,
            "8738d61a34216dc6749919171ad60eaec64c7e7cf6edc80932cde7a1ff296e7b",
        )
        self.assertNotIn("play_decode_kind", windows.to_dict()["objective"])

    def test_exclusive_argmax_probe_keeps_episode_windows_teacher(self) -> None:
        windows = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-episode-windows.toml"
        )
        exclusive = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-exclusive-argmax.toml"
        )
        self.assertEqual(exclusive.schema_version, 2)
        self.assertEqual(exclusive.dataset.kind, "maze_chase")
        self.assertEqual(exclusive.run.max_optimizer_steps, 32)
        self.assertEqual(exclusive.dataset.sequence_length, 8)
        self.assertEqual(exclusive.dataset.episode_horizon, 240)
        self.assertEqual(
            exclusive.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(
            windows.objective.play_decode_kind,
            "independent_logit_gt_zero_v1",
        )
        self.assertEqual(exclusive.run.seed, windows.run.seed)
        self.assertEqual(exclusive.run.model_factory, windows.run.model_factory)
        self.assertEqual(
            exclusive.optimization.scheduler_kind,
            windows.optimization.scheduler_kind,
        )
        self.assertNotEqual(windows.config_sha256, exclusive.config_sha256)
        self.assertEqual(
            exclusive.to_dict()["objective"]["play_decode_kind"],
            "exclusive_argmax_wasd_v1",
        )
        source = dataset_batch_source(exclusive.dataset)
        self.assertIsInstance(source, MazeChaseBatchSource)
        self.assertEqual(
            source.manifest_sha256,
            dataset_batch_source(windows.dataset).manifest_sha256,
        )

        smoke = load_training_config(
            ROOT / "configs" / "training" / "dgx-smoke.toml"
        )
        with self.assertRaisesRegex(ValueError, "only valid for maze_chase"):
            replace(
                smoke,
                objective=replace(
                    smoke.objective,
                    play_decode_kind="exclusive_argmax_wasd_v1",
                ),
            )

    def test_exclusive_softmax_probe_keeps_argmax_play_decode(self) -> None:
        exclusive_decode = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-exclusive-argmax.toml"
        )
        exclusive_ce = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-exclusive-ce.toml"
        )
        self.assertEqual(exclusive_ce.schema_version, 2)
        self.assertEqual(exclusive_ce.dataset.kind, "maze_chase")
        self.assertEqual(exclusive_ce.run.max_optimizer_steps, 32)
        self.assertEqual(exclusive_ce.dataset.sequence_length, 8)
        self.assertEqual(exclusive_ce.dataset.episode_horizon, 240)
        self.assertEqual(
            exclusive_ce.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(
            exclusive_ce.objective.action_loss_kind,
            "exclusive_wasd_softmax_v1",
        )
        self.assertEqual(
            exclusive_decode.objective.action_loss_kind,
            "support_aware_calibrated_v1",
        )
        self.assertEqual(exclusive_ce.run.seed, exclusive_decode.run.seed)
        self.assertEqual(
            exclusive_ce.run.model_factory,
            exclusive_decode.run.model_factory,
        )
        self.assertNotEqual(
            exclusive_decode.config_sha256,
            exclusive_ce.config_sha256,
        )
        self.assertEqual(
            exclusive_ce.config_sha256,
            "6419c66f4fff8c8a6105d198beea33cf7f2131aab51ff4e28693243a7148c5f3",
        )
        self.assertEqual(
            exclusive_ce.to_dict()["objective"]["action_loss_kind"],
            "exclusive_wasd_softmax_v1",
        )
        self.assertEqual(
            exclusive_ce.to_dict()["objective"]["play_decode_kind"],
            "exclusive_argmax_wasd_v1",
        )
        source = dataset_batch_source(exclusive_ce.dataset)
        self.assertIsInstance(source, MazeChaseBatchSource)
        self.assertEqual(
            source.manifest_sha256,
            dataset_batch_source(exclusive_decode.dataset).manifest_sha256,
        )
        action_only = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-exclusive-ce-action-only.toml"
        )
        self.assertEqual(action_only.schema_version, 2)
        self.assertEqual(action_only.dataset.kind, "maze_chase")
        self.assertEqual(action_only.run.max_optimizer_steps, 32)
        self.assertEqual(action_only.dataset.episode_horizon, 240)
        self.assertEqual(
            action_only.objective.action_loss_kind,
            "exclusive_wasd_softmax_v1",
        )
        self.assertEqual(
            action_only.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(action_only.objective.action_weight, 1.0)
        self.assertEqual(action_only.objective.value_weight, 0.0)
        self.assertEqual(action_only.objective.world_weight, 0.0)
        self.assertEqual(action_only.objective.diversity_weight, 0.0)
        self.assertEqual(action_only.objective.continuous_action_weight, 0.0)
        self.assertEqual(action_only.run.seed, exclusive_ce.run.seed)
        self.assertEqual(
            action_only.run.model_factory,
            exclusive_ce.run.model_factory,
        )
        self.assertNotEqual(action_only.config_sha256, exclusive_ce.config_sha256)
        self.assertEqual(
            action_only.config_sha256,
            "6af0d222e61175421a819360796422e4f0f9ed0f5ec7405c4c4f9666e21fed3a",
        )
        self.assertEqual(
            dataset_batch_source(action_only.dataset).manifest_sha256,
            source.manifest_sha256,
        )
        value_only = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-exclusive-ce-value-only.toml"
        )
        self.assertEqual(value_only.objective.action_loss_kind, "exclusive_wasd_softmax_v1")
        self.assertEqual(value_only.objective.play_decode_kind, "exclusive_argmax_wasd_v1")
        self.assertEqual(value_only.objective.action_weight, 1.0)
        self.assertEqual(value_only.objective.value_weight, 0.1)
        self.assertEqual(value_only.objective.world_weight, 0.0)
        self.assertEqual(value_only.objective.diversity_weight, 0.0)
        self.assertEqual(value_only.objective.continuous_action_weight, 0.0)
        self.assertNotEqual(value_only.config_sha256, action_only.config_sha256)
        self.assertEqual(
            value_only.config_sha256,
            "868e067ac061f49334077080331ee710d003bacef2325a355e2000766368158d",
        )
        tiled = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-tiled-windows.toml"
        )
        self.assertEqual(tiled.schema_version, 2)
        self.assertEqual(tiled.dataset.kind, "maze_chase")
        self.assertEqual(tiled.run.max_optimizer_steps, 32)
        self.assertEqual(tiled.dataset.sequence_length, 8)
        self.assertEqual(tiled.dataset.episode_horizon, 240)
        self.assertEqual(tiled.dataset.window_sampling, "tiled")
        self.assertEqual(tiled.dataset.train_sequences, 30)
        self.assertEqual(
            tiled.objective.action_loss_kind,
            "exclusive_wasd_softmax_v1",
        )
        self.assertEqual(
            tiled.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(tiled.objective.action_weight, 1.0)
        self.assertEqual(tiled.objective.value_weight, 0.1)
        self.assertEqual(tiled.objective.world_weight, 0.1)
        self.assertEqual(tiled.objective.diversity_weight, 0.05)
        self.assertEqual(tiled.objective.continuous_action_weight, 0.25)
        self.assertEqual(exclusive_ce.dataset.window_sampling, "uniform")
        self.assertNotEqual(tiled.config_sha256, exclusive_ce.config_sha256)
        self.assertNotEqual(tiled.config_sha256, value_only.config_sha256)
        self.assertEqual(
            tiled.config_sha256,
            "2c126296c820830d037c0fc14053f1cbdbd05ec40d218ea2e84975cb02002b83",
        )
        tiled_source = dataset_batch_source(tiled.dataset)
        self.assertIsInstance(tiled_source, MazeChaseBatchSource)
        self.assertNotEqual(tiled_source.manifest_sha256, source.manifest_sha256)
        episode_update = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-episode-update.toml"
        )
        self.assertEqual(episode_update.dataset.window_sampling, "tiled")
        self.assertEqual(episode_update.dataset.train_sequences, 30)
        self.assertEqual(episode_update.optimization.batch_size, 1)
        self.assertEqual(episode_update.optimization.gradient_accumulation_steps, 30)
        self.assertEqual(tiled.optimization.gradient_accumulation_steps, 1)
        self.assertEqual(
            episode_update.objective.action_loss_kind,
            "exclusive_wasd_softmax_v1",
        )
        self.assertEqual(
            episode_update.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(episode_update.objective.value_weight, 0.1)
        self.assertEqual(episode_update.objective.world_weight, 0.1)
        self.assertEqual(episode_update.objective.diversity_weight, 0.05)
        self.assertEqual(episode_update.objective.continuous_action_weight, 0.25)
        self.assertNotEqual(episode_update.config_sha256, tiled.config_sha256)
        self.assertEqual(
            episode_update.config_sha256,
            "b9b7888c194f72d91b6464b6e3e99dc2e52103db35c9a4d441ca69b60ee80c40",
        )
        self.assertEqual(
            dataset_batch_source(episode_update.dataset).manifest_sha256,
            tiled_source.manifest_sha256,
        )
        multi_episode = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-multi-episode.toml"
        )
        self.assertEqual(multi_episode.dataset.window_sampling, "tiled")
        self.assertEqual(multi_episode.dataset.train_sequences, 90)
        self.assertEqual(multi_episode.optimization.batch_size, 1)
        self.assertEqual(
            multi_episode.optimization.gradient_accumulation_steps,
            30,
        )
        self.assertEqual(
            multi_episode.objective.action_loss_kind,
            "exclusive_wasd_softmax_v1",
        )
        self.assertEqual(
            multi_episode.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(multi_episode.objective.value_weight, 0.1)
        self.assertEqual(multi_episode.objective.world_weight, 0.1)
        self.assertEqual(multi_episode.objective.diversity_weight, 0.05)
        self.assertEqual(multi_episode.objective.continuous_action_weight, 0.25)
        self.assertNotEqual(multi_episode.config_sha256, episode_update.config_sha256)
        self.assertEqual(
            multi_episode.config_sha256,
            "4231288135f8465008a106f1133a8f1a9321f7ee0a8cf7aa76494a0788c72c54",
        )
        self.assertNotEqual(
            dataset_batch_source(multi_episode.dataset).manifest_sha256,
            tiled_source.manifest_sha256,
        )
        self.assertEqual(
            dataset_batch_source(multi_episode.dataset).manifest_sha256,
            "e397017434729535104815a0c0429179fb7d246d5f3c99ccfe0241c4bd57f16c",
        )
        turn_weighted = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-turn-weighted.toml"
        )
        self.assertEqual(turn_weighted.dataset.window_sampling, "tiled")
        self.assertEqual(turn_weighted.dataset.train_sequences, 90)
        self.assertEqual(
            turn_weighted.optimization.gradient_accumulation_steps,
            30,
        )
        self.assertEqual(
            turn_weighted.objective.action_loss_kind,
            "exclusive_wasd_softmax_turn_weighted_v1",
        )
        self.assertEqual(
            turn_weighted.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(turn_weighted.objective.value_weight, 0.1)
        self.assertNotEqual(turn_weighted.config_sha256, multi_episode.config_sha256)
        self.assertEqual(
            turn_weighted.config_sha256,
            "3af3cd9974020d3b72f202552605fc6b1910a52f3937f06dc69b66286d759f7b",
        )
        self.assertEqual(
            dataset_batch_source(turn_weighted.dataset).manifest_sha256,
            dataset_batch_source(multi_episode.dataset).manifest_sha256,
        )
        turn_weighted_128 = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-turn-weighted-128.toml"
        )
        self.assertEqual(turn_weighted_128.run.max_optimizer_steps, 128)
        self.assertEqual(turn_weighted_128.dataset.window_sampling, "tiled")
        self.assertEqual(turn_weighted_128.dataset.train_sequences, 90)
        self.assertEqual(
            turn_weighted_128.optimization.gradient_accumulation_steps,
            30,
        )
        self.assertEqual(
            turn_weighted_128.objective.action_loss_kind,
            "exclusive_wasd_softmax_turn_weighted_v1",
        )
        self.assertEqual(
            turn_weighted_128.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(turn_weighted_128.run.seed, turn_weighted.run.seed)
        self.assertNotEqual(
            turn_weighted_128.config_sha256,
            turn_weighted.config_sha256,
        )
        self.assertEqual(
            turn_weighted_128.config_sha256,
            "7ad447d0e09f304751bcf2337419255b2355875c84b00dc1e682cb13c6fd4ad9",
        )
        self.assertEqual(
            dataset_batch_source(turn_weighted_128.dataset).manifest_sha256,
            dataset_batch_source(turn_weighted.dataset).manifest_sha256,
        )
        play_peak = load_training_config(
            ROOT
            / "configs"
            / "training"
            / "dgx-play-maze-chase-distill-play-peak.toml"
        )
        self.assertEqual(play_peak.run.max_optimizer_steps, 64)
        self.assertEqual(play_peak.logging.play_eval_every_steps, 8)
        self.assertEqual(play_peak.logging.play_early_stop_kind, "play_peak_v1")
        self.assertEqual(play_peak.logging.evaluate_every_steps, 32)
        self.assertEqual(play_peak.logging.checkpoint_every_steps, 8)
        self.assertEqual(play_peak.dataset.window_sampling, "tiled")
        self.assertEqual(play_peak.dataset.train_sequences, 90)
        self.assertEqual(play_peak.optimization.gradient_accumulation_steps, 30)
        self.assertEqual(
            play_peak.objective.action_loss_kind,
            "exclusive_wasd_softmax_turn_weighted_v1",
        )
        self.assertEqual(
            play_peak.objective.play_decode_kind,
            "exclusive_argmax_wasd_v1",
        )
        self.assertEqual(play_peak.run.seed, turn_weighted.run.seed)
        self.assertNotEqual(play_peak.config_sha256, turn_weighted.config_sha256)
        self.assertNotEqual(play_peak.config_sha256, turn_weighted_128.config_sha256)
        self.assertEqual(
            play_peak.config_sha256,
            "1d5e29963993766cd945ab344b29d149f47a873057923385251b7af6d6163059",
        )
        self.assertEqual(
            dataset_batch_source(play_peak.dataset).manifest_sha256,
            dataset_batch_source(turn_weighted.dataset).manifest_sha256,
        )

        smoke = load_training_config(
            ROOT / "configs" / "training" / "dgx-smoke.toml"
        )
        with self.assertRaisesRegex(ValueError, "only valid for maze_chase"):
            replace(
                smoke,
                objective=replace(
                    smoke.objective,
                    action_loss_kind="exclusive_wasd_softmax_v1",
                    play_decode_kind="exclusive_argmax_wasd_v1",
                ),
            )
        with self.assertRaisesRegex(ValueError, "requires exclusive_argmax_wasd_v1"):
            replace(
                exclusive_decode,
                objective=replace(
                    exclusive_decode.objective,
                    action_loss_kind="exclusive_wasd_softmax_v1",
                    play_decode_kind="independent_logit_gt_zero_v1",
                ),
            )
        with self.assertRaisesRegex(ValueError, "only valid for maze_chase"):
            replace(
                smoke,
                logging=replace(
                    smoke.logging,
                    play_eval_every_steps=8,
                ),
            )

    def test_play_gate_floor_is_frozen(self) -> None:
        self.assertEqual(CAMPAIGN_ID, "play_gated_maze_chase_distill_v1")
        self.assertEqual(PLAY_SEEDS, (5, 9))
        self.assertEqual(PLAY_TICKS, 240)
        self.assertEqual(NOOP_REWARD_FLOOR, -161.0)
        self.assertEqual(NOOP_COLLISION_FLOOR, 17)
        self.assertEqual(CAMPAIGN_PELLET_FLOOR, 32)
        self.assertEqual(STICKY_PELLET_BAND, 20)
        self.assertEqual(PLAY_PEAK_DROP_PELLETS, 8)

    def test_play_peak_v1_stops_on_pellet_drop_and_one_key_s(self) -> None:
        mixed = {
            "pellets_eaten": 38,
            "collisions": 43,
            "sticky_or_idle": False,
            "movement_mask_histogram": [[2, 377], [4, 103]],
        }
        sticky_s = {
            "pellets_eaten": 20,
            "collisions": 390,
            "sticky_or_idle": False,
            "movement_mask_histogram": [[0, 26], [4, 454]],
        }
        self.assertFalse(play_peak_should_stop(mixed, None))
        self.assertTrue(play_peak_should_stop(sticky_s, mixed))
        self.assertFalse(is_one_key_wasd({2: 377, 4: 103}))
        self.assertTrue(is_one_key_wasd({0: 26, 4: 454}))
        self.assertGreater(play_score(mixed), play_score(sticky_s))
        mild_drop = {
            "pellets_eaten": 30,
            "collisions": 50,
            "sticky_or_idle": False,
            "movement_mask_histogram": [[2, 300], [4, 180]],
        }
        self.assertTrue(play_peak_should_stop(mild_drop, mixed))
        tiny_drop = {
            "pellets_eaten": 36,
            "collisions": 40,
            "sticky_or_idle": False,
            "movement_mask_histogram": [[2, 360], [4, 120]],
        }
        self.assertFalse(play_peak_should_stop(tiny_drop, mixed))
        one_key_a = {
            "pellets_eaten": 36,
            "collisions": 40,
            "sticky_or_idle": False,
            "movement_mask_histogram": [[2, 480]],
        }
        self.assertTrue(play_peak_should_stop(one_key_a, mixed))

    def test_play_eval_does_not_hardcode_cpu_device(self) -> None:
        import inspect

        from irene_brain.evaluation.closed_loop_play import evaluate_closed_loop_play

        source = inspect.getsource(evaluate_closed_loop_play)
        self.assertNotIn('torch.device("cpu")', source)
        self.assertIn("parameters()", source)

    def test_play_gate_reads_pellet_eaten_not_target_collected(self) -> None:
        import inspect

        from irene_brain.training import play_gate as module

        source = inspect.getsource(module.evaluate_maze_chase_play)
        self.assertIn('totals["pellets_eaten"]', source)
        self.assertNotIn('totals["targets_collected"]', source)
        self.assertIn("movement_mask_histogram", source)
        self.assertIn("campaign_success", source)
        self.assertIn("sticky_or_idle", source)
        self.assertIn("play_decode_kind", source)
        self.assertIn("decode_kind", source)

    def test_maze_chase_play_counts_pellets_not_targets(self) -> None:
        class _Cycle:
            uses_privileged_state = False

            def __init__(self) -> None:
                self._index = 0
                self._keys = (
                    int(HidKey.D),
                    int(HidKey.S),
                    int(HidKey.A),
                    int(HidKey.W),
                )

            def reset(self, episode_seed: int) -> None:
                del episode_seed
                self._index = 0

            def act(self, observation: object) -> GenericControl:
                del observation
                key = self._keys[self._index % 4]
                self._index += 1
                return GenericControl(keys_down=(key,))

        config = ClosedLoopPlayConfig(episode_seeds=(5,), max_ticks=24)
        episode = run_policy_closed_loop_episode(
            _Cycle(),
            seed=5,
            config=config,
            environment_factory=lambda: MazeChaseEnv(
                ghost_count=3,
                ghost_period=2,
                extra_loops=16,
                max_ticks=24,
            ),
        )
        self.assertEqual(episode.targets_collected, 0)
        self.assertGreater(episode.pellets_eaten, 0)


if __name__ == "__main__":
    unittest.main()
