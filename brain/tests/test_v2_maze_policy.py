"""CPU contracts for Core V2 closed-loop maze-chase control."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.v21_live_play_qualification import StrictRgbOnlyPolicy
from irene_brain.types import GenericControl, HidKey, TextInput, TextProvenance
from irene_brain.v2 import (
    CONFIG_A_DECISION_ONLY,
    CONFIG_B_PREDICTIVE,
    CoreV2Config,
    CoreV2Model,
)
from irene_brain.v2.aggregator import AggregatedDecision
from irene_brain.v2.core import StepOutput
from irene_brain.v2.maze_policy import (
    OUTCOME_AWARE_MODEL_SEED_V1,
    CoreV2MazePolicy,
    OutcomeAwareCoreV2MazePolicy,
)
from irene_brain.v2.outcome_model import ActionOutcomeTable


def _all_action_config() -> CoreV2Config:
    return CoreV2Config(
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
        hazard_parameterization="probability_sigmoid_v1",
        outcome_architecture="all_action_table_v1",
    )


def _outcome_table(
    *,
    action_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
    rewards: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
    hazards: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
) -> ActionOutcomeTable:
    ids = torch.tensor((action_ids,), dtype=torch.long)
    canonical_rewards = torch.tensor((rewards,), dtype=torch.float32)
    canonical_hazards = torch.tensor((hazards,), dtype=torch.float32)
    column_rewards = canonical_rewards.gather(1, ids).unsqueeze(-1)
    column_hazards = canonical_hazards.gather(1, ids).unsqueeze(-1)
    next_latent = ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 120)
    return ActionOutcomeTable(
        action_ids=ids,
        predicted_next_latent=next_latent,
        predicted_reward=column_rewards,
        predicted_reward_logits=None,
        predicted_hazard=column_hazards,
        # Deliberately unrelated raw values: the live policy must consume the
        # calibrated probability field above, never raw hazard logits.
        raw_hazard_logits=torch.flip(column_hazards, dims=(1,)) * 100.0,
        predicted_hazard_logits=torch.zeros_like(column_hazards),
    )


def _step_output(
    model: CoreV2Model,
    *,
    decision_values: tuple[float, ...],
    table: ActionOutcomeTable,
) -> StepOutput:
    batch = 1
    action_count = model.config.actions
    thought_count = model.config.thoughtlets
    width = model.config.width
    values = torch.tensor((decision_values,), dtype=torch.float32)
    decision = AggregatedDecision(
        action_dist=values.softmax(dim=-1),
        action_values=values,
        thought_action_probs=torch.zeros(batch, thought_count, action_count),
        thought_existence=torch.zeros(batch, thought_count, 1),
        thought_branch_prob=torch.zeros(batch, thought_count, 1),
        thought_utility=torch.zeros(batch, thought_count, 1),
        mass_per_action=torch.zeros(batch, action_count),
    )
    return StepOutput(
        latent=torch.zeros(batch, width),
        prediction_error=None,
        retrieved_memory=None,
        thoughts=torch.zeros(batch, thought_count, width),
        belief=torch.zeros(batch, width),
        hypotheses=None,
        decision=decision,
        pending_prediction=None,
        outcome_table=table,
    )


class CoreV2MazePolicyContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = CoreV2Config(
            braincell_dynamics="normalized_mixture_v1",
            belief_dynamics="convex_gated_v1",
            hazard_parameterization="probability_sigmoid_v1",
        )
        self.model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE).eval()
        self.policy = CoreV2MazePolicy(self.model)
        self.observation = MazeChaseEnv(ghost_count=1).reset(99)

    def test_requires_explicit_episode_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "before reset"):
            self.policy.act(self.observation)

    def test_action_is_exclusive_idle_or_wasd(self) -> None:
        self.policy.reset(99)
        control = self.policy.act(self.observation)

        self.assertLessEqual(len(control.keys_down), 1)
        self.assertEqual(control.mouse_buttons, ())
        self.assertEqual(control.gamepad_buttons, ())

    def test_reset_replays_first_decision_exactly(self) -> None:
        self.policy.reset(99)
        first = self.policy.act(self.observation)
        self.policy.reset(99)
        replay = self.policy.act(self.observation)

        self.assertEqual(first, replay)

    def test_policy_drives_the_continuous_loop_without_invalid_controls(self) -> None:
        config = ClosedLoopPlayConfig(
            episode_seeds=(99,),
            max_ticks=8,
            hazard_count=1,
        )
        report = run_policy_closed_loop_episode(
            self.policy,
            seed=99,
            config=config,
            environment_factory=lambda: MazeChaseEnv(ghost_count=1, max_ticks=8),
        )

        self.assertEqual(report.ticks_advanced, 8)
        self.assertEqual(report.decisions_submitted, 8)
        self.assertEqual(report.decisions_rejected, 0)
        self.assertEqual(report.opposite_conflicts, 0)
        self.assertEqual(report.non_movement_key_activations, 0)


class OutcomeAwareCoreV2MazePolicyContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = CoreV2Model(
            config=_all_action_config(),
            flags=CONFIG_B_PREDICTIVE,
        ).eval()
        self.policy = OutcomeAwareCoreV2MazePolicy(self.model)
        self.observation = MazeChaseEnv(ghost_count=1).reset(99)

    def test_is_opt_in_and_requires_exhaustive_outcome_model(self) -> None:
        legacy = CoreV2Model(
            config=CoreV2Config(
                hazard_parameterization="probability_sigmoid_v1",
            ),
            flags=CONFIG_B_PREDICTIVE,
        )
        with self.assertRaisesRegex(ValueError, "all_action_table_v1"):
            OutcomeAwareCoreV2MazePolicy(legacy)

        no_world_model = CoreV2Model(
            config=_all_action_config(),
            flags=CONFIG_A_DECISION_ONLY,
        )
        with self.assertRaisesRegex(ValueError, "world-model learning"):
            OutcomeAwareCoreV2MazePolicy(no_world_model)

        self.assertEqual(CoreV2MazePolicy.identity, "irene.core_v2.maze_policy.v1")

    def test_registered_coefficients_and_model_seed_are_immutable(self) -> None:
        self.assertEqual(self.policy.coefficients.decision, 1.0)
        self.assertEqual(self.policy.coefficients.reward, 1.0)
        self.assertEqual(self.policy.coefficients.hazard, 3.0)
        self.assertEqual(self.policy.model_seed, OUTCOME_AWARE_MODEL_SEED_V1)
        with self.assertRaises(FrozenInstanceError):
            self.policy.coefficients.hazard = 0.0
        with self.assertRaises(AttributeError):
            self.policy.coefficients = self.policy.coefficients

    def test_reward_and_calibrated_hazard_causally_change_selection(self) -> None:
        decision_values = torch.tensor(((0.0, 1.0, 0.0, 0.0, 0.9),))
        neutral = _outcome_table()
        reward_changed = _outcome_table(rewards=(0.0, 0.0, 0.0, 0.0, 1.0))
        hazard_changed = _outcome_table(hazards=(0.0, 0.1, 0.0, 0.0, 0.0))

        self.assertEqual(
            self.policy._select_semantic_action(
                decision_values=decision_values,
                outcome_table=neutral,
            ),
            1,
        )
        self.assertEqual(
            self.policy._select_semantic_action(
                decision_values=decision_values,
                outcome_table=reward_changed,
            ),
            4,
        )
        self.assertEqual(
            self.policy._select_semantic_action(
                decision_values=decision_values,
                outcome_table=hazard_changed,
            ),
            4,
        )

    def test_semantic_table_permutation_cannot_change_selection(self) -> None:
        values = torch.tensor(((0.0, 0.4, 0.3, 0.2, 0.1),))
        rewards = (0.0, 0.0, 0.0, 0.8, 0.0)
        hazards = (0.0, 0.0, 0.0, 0.1, 0.0)
        canonical = _outcome_table(rewards=rewards, hazards=hazards)
        permuted = _outcome_table(
            action_ids=(4, 3, 0, 2, 1),
            rewards=rewards,
            hazards=hazards,
        )

        canonical_action = self.policy._select_semantic_action(
            decision_values=values,
            outcome_table=canonical,
        )
        permuted_action = self.policy._select_semantic_action(
            decision_values=values,
            outcome_table=permuted,
        )
        self.assertEqual(canonical_action, 3)
        self.assertEqual(permuted_action, canonical_action)

    def test_nonfinite_scores_and_fractional_action_ids_fail_closed(self) -> None:
        table = _outcome_table()
        with self.assertRaisesRegex(RuntimeError, "decision values must be finite"):
            self.policy._select_semantic_action(
                decision_values=torch.tensor(((0.0, float("nan"), 0.0, 0.0, 0.0),)),
                outcome_table=table,
            )
        fractional = ActionOutcomeTable(
            action_ids=table.action_ids.to(torch.float32),
            predicted_next_latent=table.predicted_next_latent,
            predicted_reward=table.predicted_reward,
            predicted_reward_logits=table.predicted_reward_logits,
            predicted_hazard=table.predicted_hazard,
        )
        with self.assertRaisesRegex(RuntimeError, "integer dtype"):
            self.policy._select_semantic_action(
                decision_values=torch.zeros(1, 5),
                outcome_table=fractional,
            )

    def test_rgb_is_only_observation_input_and_prior_action_is_internal(self) -> None:
        changed_metadata = replace(
            self.observation,
            frame_id=123_456,
            capture_tick=987_654,
            elapsed_ns=555_555,
            previous_control=GenericControl(
                keys_down=(int(HidKey.W), int(HidKey.D))
            ),
            audio_pcm_s16le=b"not model input",
            text_inputs=(
                TextInput(TextProvenance.VISIBLE_UI, "not model input"),
            ),
        )
        pixels_seen: list[torch.Tensor] = []
        previous_actions_seen: list[int] = []
        table = _outcome_table(rewards=(0.0, 0.0, 0.0, 0.0, 2.0))

        def fake_forward(pixels, state, *, prev_action, **_kwargs):
            pixels_seen.append(pixels.detach().clone())
            previous_actions_seen.append(int(prev_action.item()))
            return (
                _step_output(
                    self.model,
                    decision_values=(0.0, 1.0, 0.0, 0.0, 0.0),
                    table=table,
                ),
                state,
            )

        with patch.object(self.model, "forward", side_effect=fake_forward):
            self.policy.reset(1)
            first = self.policy.act(self.observation)
            second = self.policy.act(changed_metadata)
            self.policy.reset(4_000_000_000)
            replay = self.policy.act(changed_metadata)

        self.assertEqual(first, GenericControl(keys_down=(int(HidKey.D),)))
        self.assertEqual(second, first)
        self.assertEqual(replay, first)
        self.assertEqual(previous_actions_seen, [0, 4, 0])
        self.assertTrue(torch.equal(pixels_seen[0], pixels_seen[1]))
        self.assertTrue(torch.equal(pixels_seen[0], pixels_seen[2]))
        self.assertNotIn("_episode_seed", vars(self.policy))

    def test_act_rgb_fails_closed_when_external_prior_disagrees(self) -> None:
        self.policy.reset(99)
        with patch.object(self.model, "forward") as forward:
            with self.assertRaisesRegex(RuntimeError, "internal prior"):
                self.policy.act_rgb(self.observation.rgb, 1)
        forward.assert_not_called()
        self.assertEqual(self.policy._previous_action, 0)

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.policy.act_rgb(self.observation.rgb, True)
        with self.assertRaisesRegex(ValueError, "idle/W/A/S/D"):
            self.policy.act_rgb(self.observation.rgb, 5)

    def test_directly_satisfies_strict_rgb_only_live_policy(self) -> None:
        table = _outcome_table(rewards=(0.0, 0.0, 0.0, 0.0, 2.0))
        previous_actions_seen: list[int] = []

        def fake_forward(pixels, state, *, prev_action, **_kwargs):
            del pixels
            previous_actions_seen.append(int(prev_action.item()))
            return (
                _step_output(
                    self.model,
                    decision_values=(0.0, 1.0, 0.0, 0.0, 0.0),
                    table=table,
                ),
                state,
            )

        strict_policy = StrictRgbOnlyPolicy(self.policy, model_reset_seed=555)
        with patch.object(self.model, "forward", side_effect=fake_forward):
            strict_policy.reset(999_001)
            first = strict_policy.act(self.observation)
            second = strict_policy.act(
                replace(
                    self.observation,
                    previous_control=GenericControl(keys_down=(int(HidKey.W),)),
                )
            )

        self.assertEqual(first, GenericControl(keys_down=(int(HidKey.D),)))
        self.assertEqual(second, first)
        self.assertEqual(previous_actions_seen, [0, 4])
        self.assertEqual(strict_policy.action_histogram, (0, 0, 0, 0, 2))

    def test_fixed_first_tick_rng_is_independent_and_does_not_escape(self) -> None:
        random_values: list[float] = []
        table = _outcome_table()

        def fake_forward(pixels, state, *, prev_action, **_kwargs):
            del pixels, prev_action
            random_values.append(float(torch.rand(()).item()))
            return (
                _step_output(
                    self.model,
                    decision_values=(1.0, 0.0, 0.0, 0.0, 0.0),
                    table=table,
                ),
                state,
            )

        torch.manual_seed(765)
        expected_external = float(torch.rand(()).item())
        torch.manual_seed(765)
        with patch.object(self.model, "forward", side_effect=fake_forward):
            self.policy.reset(10)
            self.policy.act(self.observation)
            actual_external = float(torch.rand(()).item())
            self.policy.reset(999_999)
            self.policy.act(self.observation)

        self.assertEqual(random_values[0], random_values[1])
        self.assertEqual(actual_external, expected_external)

    def test_selected_action_owns_next_pending_prediction(self) -> None:
        table = _outcome_table(
            action_ids=(4, 3, 2, 1, 0),
            rewards=(0.0, 0.0, 0.0, 0.0, 2.0),
        )
        output = _step_output(
            self.model,
            decision_values=(0.0, 1.0, 0.0, 0.0, 0.0),
            table=table,
        )
        with patch.object(
            self.model,
            "forward",
            side_effect=lambda pixels, state, **kwargs: (output, state),
        ):
            self.policy.reset(99)
            control = self.policy.act(self.observation)

        self.assertEqual(control, GenericControl(keys_down=(int(HidKey.D),)))
        pending = self.policy._state.fast.pending_prediction
        expected = table.gather(torch.tensor((4,), dtype=torch.long))
        self.assertTrue(
            torch.equal(
                pending.predicted_next_latent,
                expected.predicted_next_latent,
            )
        )
        self.assertTrue(
            torch.equal(pending.predicted_reward, expected.predicted_reward)
        )
        self.assertTrue(
            torch.equal(pending.predicted_hazard, expected.predicted_hazard)
        )

    def test_controls_are_exactly_exclusive_idle_w_a_s_d(self) -> None:
        expected = {
            0: GenericControl.neutral(),
            1: GenericControl(keys_down=(int(HidKey.W),)),
            2: GenericControl(keys_down=(int(HidKey.A),)),
            3: GenericControl(keys_down=(int(HidKey.S),)),
            4: GenericControl(keys_down=(int(HidKey.D),)),
        }
        for action, expected_control in expected.items():
            with self.subTest(action=action):
                values = [-1.0] * 5
                values[action] = 1.0
                output = _step_output(
                    self.model,
                    decision_values=tuple(values),
                    table=_outcome_table(),
                )
                with patch.object(
                    self.model,
                    "forward",
                    side_effect=lambda pixels, state, **kwargs: (output, state),
                ):
                    self.policy.reset(action)
                    control = self.policy.act(self.observation)
                self.assertEqual(control, expected_control)
                self.assertLessEqual(len(control.keys_down), 1)
                self.assertEqual(control.mouse_buttons, ())
                self.assertEqual(control.gamepad_buttons, ())


if __name__ == "__main__":
    unittest.main()
