from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.data.moving_shapes_dataset import (
    split_episode_seed,
    split_for_episode_seed,
)
from irene_brain.data.solver_dataset import (
    SOLVER_WORLD_NAMES,
    DatasetSplit,
    SolverDatasetConfig,
    SolverSequenceDataset,
    solver_dataset_manifest_sha256,
)


def _config(**overrides: object) -> SolverDatasetConfig:
    knobs: dict[str, object] = {"sequence_count": 4, "sequence_length": 48}
    knobs.update(overrides)
    return SolverDatasetConfig(**knobs)  # type: ignore[arg-type]


class SolverDatasetConfigTests(unittest.TestCase):
    def test_defaults_describe_the_canonical_keys_doors_slot(self) -> None:
        config = SolverDatasetConfig()
        self.assertEqual(config.world, "keys_doors")
        self.assertEqual(config.sequence_count, 8_192)
        self.assertEqual(config.sequence_length, 128)
        self.assertEqual(config.total_transitions, 1_048_576)

    def test_world_registry_covers_the_solver_worlds(self) -> None:
        self.assertEqual(
            sorted(SOLVER_WORLD_NAMES),
            ["junction", "keys_doors", "occlusion", "pursuit"],
        )

    def test_invalid_configuration_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            SolverDatasetConfig(world="moving_shapes")
        with self.assertRaises(ValueError):
            SolverDatasetConfig(world="maze_chase")
        with self.assertRaises(ValueError):
            SolverDatasetConfig(world=3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SolverDatasetConfig(split="heldout")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SolverDatasetConfig(sequence_count=0)
        with self.assertRaises(ValueError):
            SolverDatasetConfig(sequence_length=0)
        with self.assertRaises(ValueError):
            SolverDatasetConfig(seed_offset=(1 << 62) - 1, sequence_count=2)
        with self.assertRaises(ValueError):
            SolverDatasetConfig(discount=1.5)

    def test_manifest_is_canonical_and_knob_sensitive(self) -> None:
        config = _config()
        self.assertEqual(
            solver_dataset_manifest_sha256(config),
            SolverSequenceDataset(config).manifest_sha256,
        )
        baseline = solver_dataset_manifest_sha256(config)
        for override in (
            {"world": "junction"},
            {"world": "occlusion"},
            {"world": "pursuit"},
            {"split": DatasetSplit.VALIDATION},
            {"sequence_count": 8},
            {"discount": 0.5},
        ):
            self.assertNotEqual(
                solver_dataset_manifest_sha256(_config(**override)),
                baseline,
                msg=f"manifest must change with {override}",
            )

    def test_manifest_records_the_solver_teacher(self) -> None:
        manifest = _config(world="occlusion").manifest_dict()
        self.assertEqual(manifest["generator_id"], "irene.occlusion.solver_teacher.v1")
        self.assertEqual(
            manifest["teacher_identity"], "diagnostic.scripted_occlusion_memory.v1"
        )
        self.assertEqual(manifest["teacher_inputs"], "visible_rgb_only")
        self.assertEqual(manifest["environment_family"], "occlusion")
        self.assertEqual(manifest["license_record_id"], "original-project-content")


class SolverSequenceDatasetTests(unittest.TestCase):
    def test_materialized_sequence_is_well_formed_on_every_world(self) -> None:
        for world in SOLVER_WORLD_NAMES:
            dataset = SolverSequenceDataset(_config(world=world))
            sequence = dataset[0]
            self.assertEqual(sequence.split, DatasetSplit.TRAIN)
            self.assertEqual(
                sequence.episode_seed, split_episode_seed(DatasetSplit.TRAIN, 0)
            )
            self.assertEqual(len(sequence.transitions), 48)
            final = sequence.transitions[-1]
            self.assertTrue(final.truncated_target)
            self.assertFalse(final.terminated_target)
            # Terminal value targets are zero-bootstrapped: the final value
            # is exactly the final reward.
            self.assertEqual(final.value_target.hex(), final.reward_target.hex())
            # Boundary-frame continuity: each next-observation target is the
            # following transition's input frame.
            for first, second in zip(
                sequence.transitions, sequence.transitions[1:]
            ):
                self.assertEqual(
                    first.next_observation_target.rgb,
                    second.observation.rgb,
                )

    def test_teachers_score_from_pixels_only_on_every_world(self) -> None:
        # Every solver teacher collects at least one target within a
        # 128-tick demonstration of the canonical slot.
        for world in SOLVER_WORLD_NAMES:
            dataset = SolverSequenceDataset(
                _config(world=world, sequence_length=128)
            )
            rewards = sum(
                transition.reward_target for transition in dataset[0].transitions
            )
            self.assertGreater(rewards, 0.0, msg=f"{world} teacher must score")

    def test_materialization_is_deterministic_per_index(self) -> None:
        dataset = SolverSequenceDataset(_config(world="junction"))
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
        dataset = SolverSequenceDataset(_config())
        identity = dataset.epoch_indices(epoch=0, shuffle=False)
        self.assertEqual(identity, (0, 1, 2, 3))
        shuffled = dataset.epoch_indices(epoch=3)
        self.assertEqual(sorted(shuffled), [0, 1, 2, 3])
        self.assertEqual(shuffled, dataset.epoch_indices(epoch=3))

    def test_index_validation_fails_closed(self) -> None:
        dataset = SolverSequenceDataset(_config())
        with self.assertRaises(TypeError):
            dataset[True]  # type: ignore[index]
        with self.assertRaises(IndexError):
            dataset[4]
        with self.assertRaises(ValueError):
            SolverSequenceDataset(object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            solver_dataset_manifest_sha256(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
