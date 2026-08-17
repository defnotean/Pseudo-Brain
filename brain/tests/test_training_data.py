from __future__ import annotations

import unittest

from irene_brain.data import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequenceDataset,
    dataset_manifest_sha256,
    split_episode_seed,
    split_for_episode_seed,
)
from irene_brain.types import MODEL_OBSERVATION_FIELDS, ModelObservation


def small_config(*, split: DatasetSplit = DatasetSplit.TRAIN) -> MovingShapesDatasetConfig:
    return MovingShapesDatasetConfig(
        split=split,
        sequence_count=11,
        sequence_length=5,
        seed_offset=19,
        hazard_count=2,
        discount=0.99,
    )


class DeterministicTrainingDatasetTests(unittest.TestCase):
    def test_random_access_is_exactly_repeatable_and_content_addressed(self) -> None:
        config = small_config()
        first_dataset = MovingShapesSequenceDataset(config)
        second_dataset = MovingShapesSequenceDataset(config)

        first = first_dataset[3]
        repeated = first_dataset[3]
        independent = second_dataset[3]

        self.assertEqual(first, repeated)
        self.assertEqual(first, independent)
        self.assertEqual(first.content_sha256, repeated.content_sha256)
        self.assertEqual(first.content_sha256, independent.content_sha256)
        self.assertEqual(first.manifest_sha256, dataset_manifest_sha256(config))
        self.assertEqual(len(first.transitions), config.sequence_length)

    def test_split_seed_namespaces_are_provably_disjoint(self) -> None:
        local_index = 12345
        seeds = {
            split: split_episode_seed(split, local_index)
            for split in DatasetSplit
        }
        self.assertEqual(len(set(seeds.values())), len(DatasetSplit))
        for split, seed in seeds.items():
            self.assertIs(split_for_episode_seed(seed), split)

        hashes = {
            MovingShapesSequenceDataset(small_config(split=split))[0].content_sha256
            for split in DatasetSplit
        }
        self.assertEqual(len(hashes), len(DatasetSplit))

    def test_epoch_order_is_a_deterministic_bijection_not_data_mutation(self) -> None:
        dataset = MovingShapesSequenceDataset(small_config())
        epoch_zero = dataset.epoch_indices(epoch=0)
        epoch_zero_again = dataset.epoch_indices(epoch=0)
        epoch_one = dataset.epoch_indices(epoch=1)

        self.assertEqual(epoch_zero, epoch_zero_again)
        self.assertEqual(set(epoch_zero), set(range(len(dataset))))
        self.assertEqual(set(epoch_one), set(range(len(dataset))))
        self.assertNotEqual(epoch_zero, epoch_one)
        self.assertEqual(
            tuple(sequence.sequence_index for sequence in dataset.iter_epoch(epoch=0)),
            epoch_zero,
        )

        by_index = {index: dataset[index].content_sha256 for index in range(len(dataset))}
        shuffled = {
            sequence.sequence_index: sequence.content_sha256
            for sequence in dataset.iter_epoch(epoch=1)
        }
        self.assertEqual(shuffled, by_index)

    def test_sequence_is_causal_and_discounted_targets_are_exact(self) -> None:
        config = small_config()
        sequence = MovingShapesSequenceDataset(config)[3]

        for index, transition in enumerate(sequence.transitions):
            self.assertIsInstance(transition.observation, ModelObservation)
            self.assertIsInstance(transition.next_observation_target, ModelObservation)
            self.assertEqual(
                transition.next_observation_target.previous_control,
                transition.applied_control,
            )
            self.assertEqual(
                transition.next_observation_target.frame_id,
                transition.observation.frame_id + 1,
            )
            if index:
                self.assertEqual(
                    transition.observation,
                    sequence.transitions[index - 1].next_observation_target,
                )

        self.assertTrue(
            sequence.transitions[-1].terminated_target
            or sequence.transitions[-1].truncated_target
        )
        expected_return = 0.0
        for transition in reversed(sequence.transitions):
            done = transition.terminated_target or transition.truncated_target
            expected_return = transition.reward_target + (
                0.0 if done else config.discount * expected_return
            )
            self.assertEqual(transition.value_target.hex(), expected_return.hex())

    def test_model_input_view_contains_no_labels_or_generator_identity(self) -> None:
        sequence = MovingShapesSequenceDataset(small_config())[0]
        observations = sequence.model_observations

        self.assertEqual(len(observations), len(sequence.transitions))
        for observation in observations:
            visible_fields = set(observation.__dataclass_fields__)
            self.assertEqual(visible_fields, set(MODEL_OBSERVATION_FIELDS))
            for forbidden in (
                "action_target",
                "reward_target",
                "value_target",
                "event_targets",
                "episode_seed",
                "split",
                "generator_id",
                "privileged_state",
            ):
                self.assertNotIn(forbidden, visible_fields)


if __name__ == "__main__":
    unittest.main()
