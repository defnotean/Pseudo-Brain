from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.data.maze_chase_dataset import (
    DatasetSplit,
    MazeChaseDatasetConfig,
    MazeChaseSequenceDataset,
    maze_chase_dataset_manifest_sha256,
)
from irene_brain.data.moving_shapes_dataset import (
    split_episode_seed,
    split_for_episode_seed,
)
from irene_brain.types import HidKey


def _config(**overrides: object) -> MazeChaseDatasetConfig:
    knobs: dict[str, object] = {"sequence_count": 4, "sequence_length": 48}
    knobs.update(overrides)
    return MazeChaseDatasetConfig(**knobs)  # type: ignore[arg-type]


class MazeChaseDatasetConfigTests(unittest.TestCase):
    def test_defaults_describe_the_canonical_slot(self) -> None:
        config = MazeChaseDatasetConfig()
        self.assertEqual(config.sequence_count, 8_192)
        self.assertEqual(config.sequence_length, 128)
        self.assertEqual(config.total_transitions, 1_048_576)
        self.assertEqual(config.ghost_count, 3)
        self.assertEqual(config.ghost_period, 2)
        self.assertEqual(config.extra_loops, 16)
        self.assertEqual(config.ghost_rule, "direct")

    def test_invalid_configuration_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(split="heldout")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(sequence_count=0)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(sequence_length=0)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(seed_offset=(1 << 62) - 1, sequence_count=2)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(ghost_count=9)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(ghost_period=0)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(player_period=65)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(extra_loops=65)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(ghost_rule="random")
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(ghost_elroy=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(input_delay_ticks=17)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(sticky_direction=0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(discount=1.5)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(sequence_length=8, episode_horizon=8)
        with self.assertRaises(ValueError):
            MazeChaseDatasetConfig(sequence_length=8, episode_horizon=7)

    def test_manifest_is_canonical_and_knob_sensitive(self) -> None:
        config = _config()
        self.assertEqual(
            maze_chase_dataset_manifest_sha256(config),
            MazeChaseSequenceDataset(config).manifest_sha256,
        )
        baseline = maze_chase_dataset_manifest_sha256(config)
        for override in (
            {"split": DatasetSplit.VALIDATION},
            {"sequence_count": 8},
            {"ghost_period": 3},
            {"ghost_rule": "ambush"},
            {"ghost_elroy": True},
            {"input_delay_ticks": 2},
            {"sticky_direction": True},
            {"player_period": 2},
            {"episode_horizon": 256},
        ):
            self.assertNotEqual(
                maze_chase_dataset_manifest_sha256(_config(**override)),
                baseline,
                msg=f"manifest must change with {override}",
            )

    def test_manifest_records_the_planner_teacher(self) -> None:
        manifest = _config().manifest_dict()
        self.assertEqual(manifest["generator_id"], "irene.maze_chase.planner_teacher.v1")
        self.assertEqual(
            manifest["teacher_identity"], "diagnostic.scripted_maze_chase_planner.v1"
        )
        self.assertEqual(manifest["teacher_inputs"], "visible_rgb_only")
        self.assertEqual(manifest["environment_family"], "maze_chase")
        self.assertEqual(manifest["license_record_id"], "original-project-content")
        self.assertNotIn("episode_horizon", manifest)
        self.assertNotIn("window_sampling", manifest)

    def test_spawn_only_omits_episode_horizon_from_the_manifest(self) -> None:
        spawn = maze_chase_dataset_manifest_sha256(_config())
        explicit_zero = maze_chase_dataset_manifest_sha256(_config(episode_horizon=0))
        self.assertEqual(spawn, explicit_zero)


class MazeChaseSequenceDatasetTests(unittest.TestCase):
    def test_materialized_sequence_is_well_formed(self) -> None:
        dataset = MazeChaseSequenceDataset(_config())
        sequence = dataset[0]
        self.assertEqual(sequence.split, DatasetSplit.TRAIN)
        self.assertEqual(
            sequence.episode_seed, split_episode_seed(DatasetSplit.TRAIN, 0)
        )
        self.assertLessEqual(len(sequence.transitions), 48)
        final = sequence.transitions[-1]
        self.assertTrue(final.terminated_target or final.truncated_target)
        # Terminal value targets are zero-bootstrapped: the final value is
        # exactly the final reward.
        self.assertEqual(
            final.value_target.hex(),
            final.reward_target.hex(),
        )

    def test_teacher_eats_pellets_from_pixels_only(self) -> None:
        dataset = MazeChaseSequenceDataset(_config())
        events = [
            event
            for transition in dataset[0].transitions
            for event in transition.event_targets
        ]
        self.assertIn("pellet_eaten", events)

    def test_materialization_is_deterministic_per_index(self) -> None:
        dataset = MazeChaseSequenceDataset(_config())
        first = dataset[1]
        second = dataset[1]
        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertNotEqual(first.content_sha256, dataset[0].content_sha256)

    def test_split_namespaces_stay_disjoint(self) -> None:
        for index in (0, 1, 2, 3):
            self.assertIs(
                split_for_episode_seed(split_episode_seed(DatasetSplit.TRAIN, index)),
                DatasetSplit.TRAIN,
            )
            self.assertIs(
                split_for_episode_seed(
                    split_episode_seed(DatasetSplit.VALIDATION, index)
                ),
                DatasetSplit.VALIDATION,
            )

    def test_epoch_indices_are_a_deterministic_bijection(self) -> None:
        dataset = MazeChaseSequenceDataset(_config())
        identity = dataset.epoch_indices(epoch=0, shuffle=False)
        self.assertEqual(identity, (0, 1, 2, 3))
        shuffled = dataset.epoch_indices(epoch=3)
        self.assertEqual(sorted(shuffled), [0, 1, 2, 3])
        self.assertEqual(shuffled, dataset.epoch_indices(epoch=3))

    def test_index_validation_fails_closed(self) -> None:
        dataset = MazeChaseSequenceDataset(_config())
        with self.assertRaises(TypeError):
            dataset[True]  # type: ignore[index]
        with self.assertRaises(IndexError):
            dataset[4]
        with self.assertRaises(ValueError):
            MazeChaseSequenceDataset(object())  # type: ignore[arg-type]

    def test_variant_configuration_materializes(self) -> None:
        dataset = MazeChaseSequenceDataset(
            _config(ghost_rule="mixed", ghost_elroy=True, input_delay_ticks=1)
        )
        sequence = dataset[0]
        self.assertGreater(len(sequence.transitions), 0)
        self.assertEqual(sequence.manifest_sha256, dataset.manifest_sha256)

    def test_episode_windows_sample_later_ticks_and_non_d_actions(self) -> None:
        spawn = MazeChaseSequenceDataset(
            _config(sequence_count=8, sequence_length=8)
        )
        windows = MazeChaseSequenceDataset(
            _config(sequence_count=8, sequence_length=8, episode_horizon=24)
        )
        self.assertEqual(
            spawn.config.manifest_dict()["generator_id"],
            "irene.maze_chase.planner_teacher.v1",
        )
        window_manifest = windows.config.manifest_dict()
        self.assertEqual(
            window_manifest["generator_id"],
            "irene.maze_chase.planner_teacher.episode_windows.v1",
        )
        self.assertEqual(window_manifest["episode_horizon"], 24)
        self.assertEqual(
            window_manifest["window_sampling"], "uniform_start_across_episode"
        )
        self.assertNotEqual(spawn.manifest_sha256, windows.manifest_sha256)

        spawn_starts = [sequence.transitions[0].observation.frame_id for sequence in spawn]
        window_starts = [
            sequence.transitions[0].observation.frame_id for sequence in windows
        ]
        self.assertEqual(spawn_starts, [0] * 8)
        self.assertEqual([len(sequence.transitions) for sequence in windows], [8] * 8)
        self.assertTrue(
            any(start >= 8 for start in window_starts),
            msg=f"episode windows stayed at spawn: {window_starts}",
        )
        teacher_keys = {
            key
            for sequence in windows
            for transition in sequence.transitions
            for key in transition.action_target.keys_down
        }
        self.assertTrue(
            teacher_keys - {int(HidKey.D)},
            msg=f"episode-window teacher was D-only: {sorted(teacher_keys)}",
        )


if __name__ == "__main__":
    unittest.main()
