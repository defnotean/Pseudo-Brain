from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.data.moving_shapes_dataset import (
    DatasetSplit,
    MovingShapesSequence,
    MovingShapesTransition,
)
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import EXCLUSIVE_ARGMAX_WASD_V1
from irene_brain.evaluation.diagnostic_policies import (
    NoOpPolicy,
    ScriptedMazeChasePlannerPolicy,
    _movement_control,
)
from irene_brain.training.batches import TrajectoryBatch
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
from irene_brain.training.dagger_distill import (
    DAggerAggregationBuffer,
    DAggerConfig,
    DAggerDistiller,
    DAggerIterationResult,
    collect_interactive_rollout,
)
from irene_brain.types import GenericControl, Observation

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem


class FixedActionPolicy:
    """Diagnostic policy that always outputs a fixed movement mask."""

    __slots__ = ("mask",)

    def __init__(self, mask: int = 8) -> None:
        self.mask = mask

    def reset(self, episode_seed: int) -> None:
        pass

    def act(self, observation: Observation) -> GenericControl:
        return _movement_control(self.mask)


class DAggerConfigTests(unittest.TestCase):
    def test_default_config_valid(self) -> None:
        config = DAggerConfig()
        self.assertEqual(config.iterations, 10)
        self.assertEqual(config.episodes_per_iteration, 4)
        self.assertEqual(config.sequence_length, 8)
        self.assertEqual(config.burn_in_steps, 1)
        self.assertEqual(config.initial_beta, 1.0)
        self.assertEqual(config.beta_decay, 0.5)

    def test_validation_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            DAggerConfig(iterations=0)
        with self.assertRaises(ValueError):
            DAggerConfig(episodes_per_iteration=0)
        with self.assertRaises(ValueError):
            DAggerConfig(max_ticks_per_episode=0)
        with self.assertRaises(ValueError):
            DAggerConfig(initial_beta=-0.1)
        with self.assertRaises(ValueError):
            DAggerConfig(initial_beta=1.5)
        with self.assertRaises(ValueError):
            DAggerConfig(beta_decay=1.1)
        with self.assertRaises(ValueError):
            DAggerConfig(min_beta=-0.1)
        with self.assertRaises(ValueError):
            DAggerConfig(sequence_length=1)
        with self.assertRaises(ValueError):
            DAggerConfig(sequence_length=8, burn_in_steps=8)
        with self.assertRaises(ValueError):
            DAggerConfig(sequence_length=8, burn_in_steps=9)
        with self.assertRaises(ValueError):
            DAggerConfig(batch_size=0)
        with self.assertRaises(ValueError):
            DAggerConfig(updates_per_iteration=0)
        with self.assertRaises(ValueError):
            DAggerConfig(buffer_capacity=0)
        with self.assertRaises(ValueError):
            DAggerConfig(discount=1.5)
        with self.assertRaises(ValueError):
            DAggerConfig(decode_kind="invalid_decode_mode")

    def test_beta_decay_schedule(self) -> None:
        config = DAggerConfig(initial_beta=1.0, beta_decay=0.5, min_beta=0.1)
        self.assertAlmostEqual(config.beta_for_iteration(0), 1.0)
        self.assertAlmostEqual(config.beta_for_iteration(1), 0.5)
        self.assertAlmostEqual(config.beta_for_iteration(2), 0.25)
        self.assertAlmostEqual(config.beta_for_iteration(3), 0.125)
        self.assertAlmostEqual(config.beta_for_iteration(4), 0.1)
        self.assertAlmostEqual(config.beta_for_iteration(10), 0.1)


class DAggerAggregationBufferTests(unittest.TestCase):
    def _create_dummy_rollout(self, num_ticks: int = 16) -> list[MovingShapesTransition]:
        env = MazeChaseEnv(ghost_count=1, ghost_period=2, extra_loops=4, max_ticks=num_ticks)
        planner = ScriptedMazeChasePlannerPolicy(ghost_period=2)
        transitions = collect_interactive_rollout(
            student_model=None,
            env=env,
            expert_planner=planner,
            beta=1.0,
            seed=42,
            max_ticks=num_ticks,
        )
        return list(transitions)

    def test_buffer_add_rollout_and_growth(self) -> None:
        buffer = DAggerAggregationBuffer(sequence_length=4, capacity=100, stride=4)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.total_transitions, 0)
        self.assertEqual(buffer.total_episodes, 0)

        rollout = self._create_dummy_rollout(num_ticks=12)
        added_seqs = buffer.add_rollout(rollout)

        self.assertGreater(added_seqs, 0)
        self.assertEqual(len(buffer), added_seqs)
        self.assertEqual(buffer.total_episodes, 1)
        self.assertEqual(buffer.total_transitions, len(rollout))

    def test_buffer_capacity_eviction_fifo(self) -> None:
        buffer = DAggerAggregationBuffer(sequence_length=4, capacity=3, stride=4)
        rollout = self._create_dummy_rollout(num_ticks=16)  # produces >= 4 sequences of length 4
        buffer.add_rollout(rollout)

        self.assertEqual(len(buffer), 3)

    def test_sample_batch_shapes_and_invariants(self) -> None:
        buffer = DAggerAggregationBuffer(sequence_length=4, capacity=50, stride=2)
        rollout = self._create_dummy_rollout(num_ticks=16)
        buffer.add_rollout(rollout)

        rng = random.Random(12345)
        batch = buffer.sample_batch(batch_size=2, burn_in_steps=1, rng=rng)

        self.assertIsInstance(batch, TrajectoryBatch)
        self.assertEqual(batch.split, "train")
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.sequence_length, 4)
        self.assertEqual(batch.burn_in_steps, 1)
        self.assertEqual(batch.sample_count, 2 * (4 - 1))

        for seq in batch.sequences:
            self.assertIsInstance(seq, MovingShapesSequence)
            self.assertEqual(len(seq.transitions), 4)
            self.assertTrue(seq.content_sha256)
            for i, t in enumerate(seq.transitions):
                if i > 0:
                    self.assertEqual(t.observation, seq.transitions[i - 1].next_observation_target)
            last = seq.transitions[-1]
            self.assertTrue(last.terminated_target or last.truncated_target)


class InteractiveRolloutTests(unittest.TestCase):
    def test_pure_expert_rollout(self) -> None:
        env = MazeChaseEnv(ghost_count=2, ghost_period=2, extra_loops=8, max_ticks=20)
        planner = ScriptedMazeChasePlannerPolicy(ghost_period=2)
        transitions = collect_interactive_rollout(
            student_model=None,
            env=env,
            expert_planner=planner,
            beta=1.0,
            seed=101,
            max_ticks=20,
        )

        self.assertGreater(len(transitions), 0)
        self.assertLessEqual(len(transitions), 20)
        for t in transitions:
            self.assertIsInstance(t, MovingShapesTransition)
            self.assertEqual(t.next_observation_target.frame_id, t.observation.frame_id + 1)
            self.assertEqual(t.next_observation_target.previous_control, t.applied_control)
            self.assertEqual(t.applied_control, t.action_target)

    def test_student_policy_mixed_rollout(self) -> None:
        env = MazeChaseEnv(ghost_count=2, ghost_period=2, extra_loops=8, max_ticks=20)
        planner = ScriptedMazeChasePlannerPolicy(ghost_period=2)
        student = NoOpPolicy()

        transitions = collect_interactive_rollout(
            student_model=student,
            env=env,
            expert_planner=planner,
            beta=0.5,
            seed=202,
            max_ticks=20,
        )

        self.assertGreater(len(transitions), 0)
        for t in transitions:
            self.assertIsInstance(t, MovingShapesTransition)
            self.assertEqual(t.next_observation_target.frame_id, t.observation.frame_id + 1)

    def test_torch_student_model_rollout(self) -> None:
        if torch is None:
            self.skipTest("torch is required")
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
        env = MazeChaseEnv(ghost_count=1, ghost_period=2, extra_loops=4, max_ticks=10)
        planner = ScriptedMazeChasePlannerPolicy(ghost_period=2)

        transitions = collect_interactive_rollout(
            student_model=model,
            env=env,
            expert_planner=planner,
            beta=0.0,
            seed=404,
            max_ticks=10,
        )

        self.assertGreater(len(transitions), 0)
        self.assertLessEqual(len(transitions), 10)
        for t in transitions:
            self.assertIsInstance(t, MovingShapesTransition)
            self.assertEqual(t.next_observation_target.frame_id, t.observation.frame_id + 1)
            self.assertEqual(t.next_observation_target.previous_control, t.applied_control)



class ExpertLabelingOfStudentStatesTests(unittest.TestCase):
    def test_expert_corrects_student_suboptimal_actions(self) -> None:
        """When student policy takes suboptimal actions, applied_control reflects student

        actions while action_target carries the expert's recovery label.
        """
        env = MazeChaseEnv(ghost_count=2, ghost_period=2, extra_loops=8, max_ticks=15)
        planner = ScriptedMazeChasePlannerPolicy(ghost_period=2)
        # Student always presses No-Op (keys_down=())
        student = NoOpPolicy()

        transitions = collect_interactive_rollout(
            student_model=student,
            env=env,
            expert_planner=planner,
            beta=0.0,  # Pure student driving
            seed=303,
            max_ticks=15,
        )

        self.assertGreater(len(transitions), 0)
        expert_action_found = False
        for t in transitions:
            # Student drove: applied control was No-Op (keys_down=())
            self.assertEqual(t.applied_control, GenericControl())
            # But the expert supervisor computed a recovery action
            if t.action_target.keys_down:
                expert_action_found = True

        self.assertTrue(
            expert_action_found,
            "Expert planner should recommend corrective non-empty actions from student states",
        )


class DAggerDistillerEndToEndTests(unittest.TestCase):
    def _make_training_system(self) -> tuple[TorchTrainingSystem, IreneBrainModel]:
        assert torch is not None
        torch.set_num_threads(1)
        torch.manual_seed(20260818)

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

        config = TrainingConfig(
            schema_version=1,
            run=RunConfig(
                name="dagger-smoke-probe",
                seed=20260818,
                model_factory="irene_brain.model.torch_model:IreneBrainModel",
                max_optimizer_steps=100,
            ),
            dataset=DatasetConfig(
                kind="moving_shapes",
                train_sequences=4,
                validation_sequences=2,
                test_sequences=2,
                sequence_length=4,
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

        objective = ThoughtFieldObjective(model)
        system = TorchTrainingSystem(objective, config)
        return system, model

    def test_single_end_to_end_dagger_iteration(self) -> None:
        if torch is None:
            self.skipTest("torch is required for end-to-end distillation test")

        system, model = self._make_training_system()
        dagger_config = DAggerConfig(
            iterations=2,
            episodes_per_iteration=1,
            max_ticks_per_episode=16,
            initial_beta=1.0,
            beta_decay=0.5,
            sequence_length=4,
            burn_in_steps=1,
            batch_size=1,
            updates_per_iteration=2,
            ghost_count=1,
            ghost_period=2,
            extra_loops=4,
        )

        distiller = DAggerDistiller(
            config=dagger_config,
            student_model=model,
            training_system=system,
        )

        param = (
            model.brain_cell.blocks[0]
            .thought_attention.feed_forward[0]
            .weight
        )
        param_before = param.detach().clone()

        result = distiller.run_dagger_iteration(iteration=0)

        self.assertIsInstance(result, DAggerIterationResult)
        self.assertEqual(result.iteration, 0)
        self.assertEqual(result.beta, 1.0)
        self.assertEqual(result.episodes_collected, 1)
        self.assertGreater(result.transitions_collected, 0)
        self.assertGreater(result.sequences_in_buffer, 0)
        self.assertTrue(math.isfinite(result.mean_train_loss))
        self.assertGreater(result.mean_train_loss, 0.0)
        self.assertIn("action_loss", result.training_metrics)

        # Confirm optimizer actually updated the weights
        param_after = param.detach().clone()
        self.assertFalse(torch.equal(param_before, param_after))

    def test_dagger_multi_iteration_training_loop(self) -> None:
        if torch is None:
            self.skipTest("torch is required for end-to-end distillation test")

        system, model = self._make_training_system()
        dagger_config = DAggerConfig(
            iterations=2,
            episodes_per_iteration=1,
            max_ticks_per_episode=16,
            initial_beta=1.0,
            beta_decay=0.5,
            sequence_length=4,
            burn_in_steps=1,
            batch_size=1,
            updates_per_iteration=1,
            ghost_count=1,
            ghost_period=2,
            extra_loops=4,
        )

        distiller = DAggerDistiller(
            config=dagger_config,
            student_model=model,
            training_system=system,
        )

        results = distiller.train_dagger(num_iterations=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].beta, 1.0)
        self.assertEqual(results[1].beta, 0.5)
        self.assertGreater(results[1].sequences_in_buffer, results[0].sequences_in_buffer)


if __name__ == "__main__":
    unittest.main()
