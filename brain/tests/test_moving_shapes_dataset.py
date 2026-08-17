from __future__ import annotations

import dataclasses
import math
import unittest
from dataclasses import replace

from irene_brain.data import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequenceDataset,
    dataset_manifest_sha256,
    split_episode_seed,
    split_for_episode_seed,
)
from irene_brain.types import MODEL_OBSERVATION_FIELDS, ModelObservation


def small_config(
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
) -> MovingShapesDatasetConfig:
    return MovingShapesDatasetConfig(
        split=split,
        sequence_count=7,
        sequence_length=12,
        seed_offset=19,
        hazard_count=2,
        tick_period_ns=10,
        discount=0.875,
    )


class MovingShapesDatasetTests(unittest.TestCase):
    def test_default_manifest_describes_lazy_million_transition_pilot(self) -> None:
        config = MovingShapesDatasetConfig()
        dataset = MovingShapesSequenceDataset(config)

        self.assertEqual(config.total_transitions, 1_048_576)
        self.assertEqual(len(dataset), 8_192)
        self.assertFalse(hasattr(dataset, "__dict__"))
        self.assertEqual(
            config.manifest_dict()["origin"],
            "in-repository deterministic procedural environment",
        )
        self.assertEqual(
            config.manifest_dict()["license_record_id"],
            "original-project-content",
        )

    def test_split_seed_namespaces_are_exact_and_disjoint(self) -> None:
        seeds: dict[DatasetSplit, set[int]] = {}
        for split in DatasetSplit:
            seeds[split] = {split_episode_seed(split, index) for index in range(256)}
            self.assertTrue(all(split_for_episode_seed(seed) is split for seed in seeds[split]))

        self.assertTrue(seeds[DatasetSplit.TRAIN].isdisjoint(seeds[DatasetSplit.VALIDATION]))
        self.assertTrue(seeds[DatasetSplit.TRAIN].isdisjoint(seeds[DatasetSplit.TEST]))
        self.assertTrue(seeds[DatasetSplit.VALIDATION].isdisjoint(seeds[DatasetSplit.TEST]))
        with self.assertRaises(ValueError):
            split_for_episode_seed(3 << 62)

    def test_manifest_identity_is_canonical_and_configuration_sensitive(self) -> None:
        config = small_config()
        self.assertEqual(
            dataset_manifest_sha256(config),
            "2f50a6ec242ad0377eb49106a0381683b058e0cce2564e15fbab62e936ea2b12",
        )
        self.assertNotEqual(
            dataset_manifest_sha256(config),
            dataset_manifest_sha256(replace(config, split=DatasetSplit.TEST)),
        )
        self.assertNotEqual(
            dataset_manifest_sha256(config),
            dataset_manifest_sha256(replace(config, discount=0.5)),
        )

    def test_random_access_generation_is_exact_and_order_independent(self) -> None:
        first = MovingShapesSequenceDataset(small_config())
        second = MovingShapesSequenceDataset(small_config())

        expected = first[4]
        first[0]
        first[6]
        regenerated = first[4]
        independently_generated = second[4]

        self.assertEqual(regenerated, expected)
        self.assertEqual(independently_generated, expected)
        self.assertEqual(
            regenerated.content_sha256,
            "87bbb8df8e5222953ac191db9d1b31cba5ce3b11f70d7cadf60038fddcc0d8de",
        )
        self.assertEqual(first[-1], first[6])
        self.assertEqual(first[1:3], (first[1], first[2]))

    def test_transition_boundaries_separate_inputs_from_all_targets(self) -> None:
        sequence = MovingShapesSequenceDataset(small_config())[0]
        self.assertEqual(len(sequence.transitions), 12)
        self.assertEqual(len(sequence.model_observations), 12)
        self.assertEqual(len(sequence.action_targets), 12)
        self.assertEqual(len(sequence.world_prediction_targets), 12)

        forbidden = {
            "episode_seed",
            "split",
            "reward_target",
            "value_target",
            "event_targets",
            "action_target",
            "next_observation_target",
        }
        for index, transition in enumerate(sequence.transitions):
            self.assertIsInstance(transition.observation, ModelObservation)
            self.assertEqual(
                {field.name for field in dataclasses.fields(transition.observation)},
                MODEL_OBSERVATION_FIELDS,
            )
            self.assertTrue(forbidden.isdisjoint(transition.observation.__dataclass_fields__))
            self.assertEqual(
                transition.next_observation_target.previous_control,
                transition.applied_control,
            )
            self.assertEqual(
                transition.world_prediction_target,
                transition.next_observation_target,
            )
            self.assertEqual(transition.action_target, transition.applied_control)
            if index:
                self.assertEqual(
                    transition.observation,
                    sequence.transitions[index - 1].next_observation_target,
                )

        self.assertFalse(sequence.transitions[-2].truncated_target)
        self.assertTrue(sequence.transitions[-1].truncated_target)

    def test_value_targets_are_exact_discounted_returns(self) -> None:
        sequence = MovingShapesSequenceDataset(small_config())[3]
        running = 0.0
        expected: list[float] = []
        for transition in reversed(sequence.transitions):
            done = transition.terminated_target or transition.truncated_target
            running = transition.reward_target + (0.0 if done else sequence.discount * running)
            expected.append(running)
        expected.reverse()

        for transition, value in zip(sequence.transitions, expected):
            self.assertEqual(transition.value_target.hex(), value.hex())
            self.assertTrue(math.isfinite(transition.value_target))

    def test_deterministic_epoch_order_is_a_bijection(self) -> None:
        config = replace(small_config(), sequence_count=17)
        dataset = MovingShapesSequenceDataset(config)
        first = dataset.epoch_indices(epoch=5)
        repeated = dataset.epoch_indices(epoch=5)

        self.assertEqual(first, repeated)
        self.assertEqual(sorted(first), list(range(len(dataset))))
        self.assertEqual(
            dataset.epoch_indices(epoch=99, shuffle=False),
            tuple(range(len(dataset))),
        )
        self.assertEqual(
            tuple(sequence.sequence_index for sequence in dataset.iter_epoch(epoch=5)),
            first,
        )

    def test_invalid_configuration_and_index_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MovingShapesDatasetConfig(split="dev")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MovingShapesDatasetConfig(sequence_count=0)
        with self.assertRaises(ValueError):
            MovingShapesDatasetConfig(sequence_length=0)
        with self.assertRaises(ValueError):
            MovingShapesDatasetConfig(discount=float("nan"))
        with self.assertRaises(ValueError):
            MovingShapesDatasetConfig(discount=1.01)
        with self.assertRaises(ValueError):
            MovingShapesDatasetConfig(seed_offset=(1 << 62) - 1, sequence_count=2)

        dataset = MovingShapesSequenceDataset(small_config())
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]
        with self.assertRaises(IndexError):
            _ = dataset[-len(dataset) - 1]
        with self.assertRaises(TypeError):
            _ = dataset[True]  # type: ignore[index]

    def test_cross_split_sequences_cannot_share_world_or_content_identity(self) -> None:
        train = MovingShapesSequenceDataset(small_config(split=DatasetSplit.TRAIN))[2]
        validation = MovingShapesSequenceDataset(
            small_config(split=DatasetSplit.VALIDATION)
        )[2]
        test = MovingShapesSequenceDataset(small_config(split=DatasetSplit.TEST))[2]

        self.assertEqual(len({train.episode_seed, validation.episode_seed, test.episode_seed}), 3)
        self.assertEqual(len({train.content_sha256, validation.content_sha256, test.content_sha256}), 3)
        self.assertIs(split_for_episode_seed(train.episode_seed), DatasetSplit.TRAIN)
        self.assertIs(
            split_for_episode_seed(validation.episode_seed),
            DatasetSplit.VALIDATION,
        )
        self.assertIs(split_for_episode_seed(test.episode_seed), DatasetSplit.TEST)


if __name__ == "__main__":
    unittest.main()
