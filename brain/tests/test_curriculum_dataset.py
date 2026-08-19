from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.data.curriculum_dataset import (
    CURRICULUM_SCENARIOS,
    CurriculumDataset,
    CurriculumDatasetConfig,
    CurriculumScenario,
    CurriculumSequence,
    CurriculumSequenceDataset,
    CurriculumTransition,
    _scenario,
    curriculum_dataset_manifest_sha256,
)
from irene_brain.data.moving_shapes_dataset import (
    DatasetSplit,
    split_episode_seed,
    split_for_episode_seed,
)
from irene_brain.environments.junction import _DIRECTIONS, _bfs_distances
from irene_brain.environments.maze_chase import MazeChaseEnv, _carve_maze_with_loops
from irene_brain.types import HidKey, ModelObservation


def _config(**overrides: object) -> CurriculumDatasetConfig:
    knobs: dict[str, object] = {"sequence_count": 5, "sequence_length": 16}
    knobs.update(overrides)
    return CurriculumDatasetConfig(**knobs)  # type: ignore[arg-type]


class CurriculumDatasetConfigTests(unittest.TestCase):
    def test_defaults_describe_the_canonical_curriculum_slot(self) -> None:
        config = CurriculumDatasetConfig()
        self.assertEqual(config.split, DatasetSplit.TRAIN)
        self.assertEqual(config.scenario, CurriculumScenario.MIXED)
        self.assertEqual(config.sequence_count, 8_192)
        self.assertEqual(config.sequence_length, 32)
        self.assertEqual(config.total_transitions, 262_144)
        self.assertEqual(config.ghost_count, 3)
        self.assertEqual(config.ghost_period, 2)
        self.assertEqual(config.player_period, 1)
        self.assertEqual(config.extra_loops, 16)
        self.assertEqual(config.ghost_rule, "direct")

    def test_scenario_resolution_and_aliases(self) -> None:
        self.assertEqual(_scenario("wall_unsticking"), CurriculumScenario.WALL_UNSTICKING)
        self.assertEqual(_scenario("scenario_a"), CurriculumScenario.WALL_UNSTICKING)
        self.assertEqual(_scenario("unsticking"), CurriculumScenario.WALL_UNSTICKING)
        self.assertEqual(_scenario("junction_decisions"), CurriculumScenario.JUNCTION_DECISIONS)
        self.assertEqual(_scenario("scenario_b"), CurriculumScenario.JUNCTION_DECISIONS)
        self.assertEqual(_scenario("junction"), CurriculumScenario.JUNCTION_DECISIONS)
        self.assertEqual(_scenario("hazard_evasion"), CurriculumScenario.HAZARD_EVASION)
        self.assertEqual(_scenario("scenario_c"), CurriculumScenario.HAZARD_EVASION)
        self.assertEqual(_scenario("evasion"), CurriculumScenario.HAZARD_EVASION)
        self.assertEqual(_scenario("motor_babbling"), CurriculumScenario.MOTOR_BABBLING)
        self.assertEqual(_scenario("scenario_d"), CurriculumScenario.MOTOR_BABBLING)
        self.assertEqual(_scenario("babbling"), CurriculumScenario.MOTOR_BABBLING)
        self.assertEqual(_scenario("standard_navigation"), CurriculumScenario.STANDARD_NAVIGATION)
        self.assertEqual(_scenario("scenario_e"), CurriculumScenario.STANDARD_NAVIGATION)
        self.assertEqual(_scenario("standard"), CurriculumScenario.STANDARD_NAVIGATION)
        self.assertEqual(_scenario("mixed"), CurriculumScenario.MIXED)
        self.assertEqual(_scenario("all"), CurriculumScenario.MIXED)

        with self.assertRaises(ValueError):
            _scenario("unknown_scenario")
        with self.assertRaises(ValueError):
            _scenario(123)  # type: ignore[arg-type]

    def test_invalid_configuration_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(split="heldout")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(scenario="invalid_scenario")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(sequence_count=0)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(sequence_length=0)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(seed_offset=(1 << 62) - 1, sequence_count=2)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(ghost_count=0)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(ghost_count=9)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(ghost_period=0)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(player_period=0)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(player_period=65)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(extra_loops=65)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(ghost_rule="random")
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(ghost_elroy=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(input_delay_ticks=17)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(sticky_direction=0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(discount=1.5)
        with self.assertRaises(ValueError):
            CurriculumDatasetConfig(discount=-0.1)

    def test_manifest_is_canonical_and_knob_sensitive(self) -> None:
        config = _config()
        self.assertEqual(
            curriculum_dataset_manifest_sha256(config),
            CurriculumDataset(config).manifest_sha256,
        )
        baseline = curriculum_dataset_manifest_sha256(config)
        for override in (
            {"scenario": CurriculumScenario.WALL_UNSTICKING},
            {"scenario": CurriculumScenario.JUNCTION_DECISIONS},
            {"scenario": CurriculumScenario.HAZARD_EVASION},
            {"scenario": CurriculumScenario.MOTOR_BABBLING},
            {"scenario": CurriculumScenario.STANDARD_NAVIGATION},
            {"split": DatasetSplit.VALIDATION},
            {"sequence_count": 10},
            {"sequence_length": 32},
            {"ghost_count": 2},
            {"ghost_period": 3},
            {"ghost_rule": "ambush"},
            {"ghost_elroy": True},
            {"input_delay_ticks": 1},
            {"sticky_direction": True},
            {"player_period": 2},
            {"discount": 0.95},
        ):
            self.assertNotEqual(
                curriculum_dataset_manifest_sha256(_config(**override)),
                baseline,
                msg=f"manifest must change with {override}",
            )

    def test_manifest_records_expected_curriculum_metadata(self) -> None:
        manifest = _config(scenario=CurriculumScenario.WALL_UNSTICKING).manifest_dict()
        self.assertEqual(
            manifest["generator_id"], "irene.curriculum_recovery.wall_unsticking.v1"
        )
        self.assertEqual(manifest["scenario"], "wall_unsticking")
        self.assertEqual(manifest["environment_family"], "maze_chase")
        self.assertEqual(manifest["dataset_family"], "curriculum_recovery")
        self.assertEqual(
            manifest["teacher_identity"], "diagnostic.scripted_maze_chase_planner.v1"
        )
        self.assertEqual(manifest["license_record_id"], "original-project-content")


class CurriculumDatasetScenarioTests(unittest.TestCase):
    def test_scenario_a_wall_unsticking(self) -> None:
        config = _config(
            scenario=CurriculumScenario.WALL_UNSTICKING,
            sequence_count=5,
            sequence_length=16,
        )
        dataset = CurriculumDataset(config)
        for i in range(len(dataset)):
            sequence = dataset[i]
            self.assertEqual(sequence.split, DatasetSplit.TRAIN)
            self.assertGreater(len(sequence.transitions), 0)
            self.assertLessEqual(len(sequence.transitions), 16)

            # Check initial transition: player was placed against a wall facing the wall
            first_t = sequence.transitions[0]
            # Action target must be an unsticking action (valid keypress turning away from wall)
            self.assertTrue(len(first_t.action_target.keys_down) > 0)
            target_key = first_t.action_target.keys_down[0]
            self.assertIn(target_key, (int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)))

            # Verify that following transitions continue valid gameplay
            for t in sequence.transitions:
                self.assertIsInstance(t.observation, ModelObservation)
                self.assertTrue(t.observation.frame_id >= 0)

    def test_scenario_b_junction_decisions(self) -> None:
        config = _config(
            scenario=CurriculumScenario.JUNCTION_DECISIONS,
            sequence_count=5,
            sequence_length=16,
        )
        dataset = CurriculumDataset(config)
        for i in range(len(dataset)):
            sequence = dataset[i]
            self.assertGreater(len(sequence.transitions), 0)
            first_t = sequence.transitions[0]
            # Planner produces valid action target at the junction
            if first_t.action_target.keys_down:
                self.assertIn(
                    first_t.action_target.keys_down[0],
                    (int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)),
                )

    def test_scenario_c_hazard_evasion(self) -> None:
        config = _config(
            scenario=CurriculumScenario.HAZARD_EVASION,
            sequence_count=5,
            sequence_length=16,
        )
        dataset = CurriculumDataset(config)
        for i in range(len(dataset)):
            sequence = dataset[i]
            self.assertGreater(len(sequence.transitions), 0)
            # The sequence should execute steps without crash
            self.assertTrue(len(sequence.transitions) <= 16)
            final = sequence.transitions[-1]
            self.assertTrue(final.terminated_target or final.truncated_target)

    def test_scenario_d_motor_babbling(self) -> None:
        config = _config(
            scenario=CurriculumScenario.MOTOR_BABBLING,
            sequence_count=5,
            sequence_length=16,
        )
        dataset = CurriculumDataset(config)
        for i in range(len(dataset)):
            sequence = dataset[i]
            self.assertEqual(len(sequence.transitions), config.sequence_length)
            final = sequence.transitions[-1]
            self.assertTrue(final.terminated_target or final.truncated_target)

            # In motor babbling, applied_control and action_target are the exploratory actions
            for t in sequence.transitions:
                self.assertEqual(t.applied_control, t.action_target)
                for key in t.applied_control.keys_down:
                    self.assertIn(
                        key,
                        (int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)),
                    )

    def test_scenario_e_standard_navigation(self) -> None:
        config = _config(
            scenario=CurriculumScenario.STANDARD_NAVIGATION,
            sequence_count=5,
            sequence_length=16,
        )
        dataset = CurriculumDataset(config)
        for i in range(len(dataset)):
            sequence = dataset[i]
            self.assertGreater(len(sequence.transitions), 0)
            self.assertLessEqual(len(sequence.transitions), 16)
            # Standard maze starts frame_id at 0
            self.assertEqual(sequence.transitions[0].observation.frame_id, 0)
            final = sequence.transitions[-1]
            self.assertTrue(final.terminated_target or final.truncated_target)


class CurriculumDatasetBatchAndInvariantsTests(unittest.TestCase):
    def test_mixed_batch_distributes_all_scenarios_evenly(self) -> None:
        config = _config(
            scenario=CurriculumScenario.MIXED,
            sequence_count=15,
            sequence_length=8,
        )
        dataset = CurriculumDataset(config)
        expected_scenarios = [
            CurriculumScenario.WALL_UNSTICKING,
            CurriculumScenario.JUNCTION_DECISIONS,
            CurriculumScenario.HAZARD_EVASION,
            CurriculumScenario.MOTOR_BABBLING,
            CurriculumScenario.STANDARD_NAVIGATION,
        ] * 3

        for i in range(15):
            self.assertEqual(dataset.scenario_for_index(i), expected_scenarios[i])
            seq = dataset[i]
            self.assertEqual(seq.sequence_index, i)
            self.assertGreater(len(seq.transitions), 0)

    def test_seed_determinism_and_content_hash(self) -> None:
        config = _config(scenario=CurriculumScenario.MIXED, sequence_count=5)
        dataset1 = CurriculumDataset(config)
        dataset2 = CurriculumDataset(config)

        for i in range(5):
            seq1 = dataset1[i]
            seq2 = dataset2[i]
            self.assertEqual(seq1.content_sha256, seq2.content_sha256)
            self.assertEqual(seq1.episode_seed, seq2.episode_seed)

        # Different sequence indices have distinct content
        self.assertNotEqual(dataset1[0].content_sha256, dataset1[1].content_sha256)

    def test_split_isolation_and_disjoint_namespaces(self) -> None:
        for idx in range(5):
            train_seed = split_episode_seed(DatasetSplit.TRAIN, idx)
            val_seed = split_episode_seed(DatasetSplit.VALIDATION, idx)
            test_seed = split_episode_seed(DatasetSplit.TEST, idx)

            self.assertIs(split_for_episode_seed(train_seed), DatasetSplit.TRAIN)
            self.assertIs(split_for_episode_seed(val_seed), DatasetSplit.VALIDATION)
            self.assertIs(split_for_episode_seed(test_seed), DatasetSplit.TEST)

            self.assertNotEqual(train_seed, val_seed)
            self.assertNotEqual(train_seed, test_seed)
            self.assertNotEqual(val_seed, test_seed)

        train_ds = CurriculumDataset(_config(split=DatasetSplit.TRAIN, sequence_count=3))
        val_ds = CurriculumDataset(_config(split=DatasetSplit.VALIDATION, sequence_count=3))
        test_ds = CurriculumDataset(_config(split=DatasetSplit.TEST, sequence_count=3))

        self.assertEqual(train_ds[0].split, DatasetSplit.TRAIN)
        self.assertEqual(val_ds[0].split, DatasetSplit.VALIDATION)
        self.assertEqual(test_ds[0].split, DatasetSplit.TEST)

    def test_transition_invariants(self) -> None:
        dataset = CurriculumDataset(_config(sequence_count=5, sequence_length=16))
        for sequence in dataset:
            for i, t in enumerate(sequence.transitions):
                # Check frame_id continuity
                self.assertEqual(
                    t.next_observation_target.frame_id,
                    t.observation.frame_id + 1,
                )
                # Check elapsed_ns progression
                self.assertGreater(
                    t.next_observation_target.elapsed_ns,
                    t.observation.elapsed_ns,
                )
                # Check control pipeline reflection
                self.assertEqual(
                    t.next_observation_target.previous_control,
                    t.applied_control,
                )
                # Check adjacent frame boundary continuity
                if i > 0:
                    prior = sequence.transitions[i - 1]
                    self.assertEqual(t.observation, prior.next_observation_target)

            final = sequence.transitions[-1]
            self.assertTrue(final.terminated_target or final.truncated_target)
            # Terminal zero-bootstrap: final value is final reward
            self.assertEqual(final.value_target.hex(), final.reward_target.hex())

    def test_epoch_indices_and_iter_epoch(self) -> None:
        dataset = CurriculumDataset(_config(sequence_count=6))
        identity = dataset.epoch_indices(epoch=0, shuffle=False)
        self.assertEqual(identity, (0, 1, 2, 3, 4, 5))

        shuffled1 = dataset.epoch_indices(epoch=1, shuffle=True)
        shuffled2 = dataset.epoch_indices(epoch=1, shuffle=True)
        self.assertEqual(shuffled1, shuffled2)
        self.assertEqual(sorted(shuffled1), [0, 1, 2, 3, 4, 5])

        sequences = list(dataset.iter_epoch(epoch=1, shuffle=True))
        self.assertEqual(len(sequences), 6)
        self.assertEqual(
            [s.sequence_index for s in sequences],
            list(shuffled1),
        )

    def test_index_slicing_and_bounds_checking(self) -> None:
        dataset = CurriculumDataset(_config(sequence_count=5))
        sliced = dataset[1:4]
        self.assertIsInstance(sliced, tuple)
        self.assertEqual(len(sliced), 3)
        self.assertEqual(sliced[0].sequence_index, 1)
        self.assertEqual(sliced[2].sequence_index, 3)

        # Negative indexing
        self.assertEqual(dataset[-1].sequence_index, 4)

        with self.assertRaises(IndexError):
            _ = dataset[5]
        with self.assertRaises(IndexError):
            _ = dataset[-6]
        with self.assertRaises(TypeError):
            _ = dataset[True]  # type: ignore[index]
        with self.assertRaises(TypeError):
            _ = dataset["0"]  # type: ignore[index]

    def test_backward_compatibility_alias(self) -> None:
        config = _config(sequence_count=2)
        ds = CurriculumSequenceDataset(config)
        self.assertIsInstance(ds, CurriculumDataset)
        self.assertEqual(len(ds), 2)


if __name__ == "__main__":
    unittest.main()
