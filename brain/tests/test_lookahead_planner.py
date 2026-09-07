from __future__ import annotations

from dataclasses import replace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
    from irene_brain.model import (
        DirectionalAction,
        LatentHazardHead,
        LatentLookaheadPlanner,
        LatentRewardHead,
        LatentSensoryTransition,
        LookaheadBranchResult,
        LookaheadPlanResult,
        LookaheadRolloutStep,
        ThoughtFieldConfig,
        directional_to_control_vector,
        generate_directional_candidate_sequences,
    )
    from irene_brain.model.torch_model import BrainState, IreneBrainModel
    from irene_brain.types import GenericControl, HidKey, Observation, RgbFrame


def tiny_config() -> ThoughtFieldConfig:
    """Return tiny topology-faithful configuration for CPU tests."""
    return replace(
        ThoughtFieldConfig.smoke(),
        core_width=16,
        sensor_tokens=4,
        belief_tokens=2,
        working_memory_tokens=1,
        thoughtlets=4,
        goal_context_tokens=2,
        cognitive_cycles=2,
        brain_cell_blocks=1,
        attention_heads=2,
        routed_neighbors=1,
        episodic_memory_entries=8,
        retrieved_entries_per_thoughtlet=1,
    )


@unittest.skipUnless(torch is not None, "PyTorch is required for lookahead planner tests")
class LookaheadPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

    def setUp(self) -> None:
        assert torch is not None
        torch.manual_seed(42)

    def test_candidate_sequence_generation_and_reversals(self) -> None:
        """Test candidate action sequence generation across horizons and reversal pruning."""
        # Horizon 1: 4 directional actions
        seqs_h1 = generate_directional_candidate_sequences(1, prune_reversals=True)
        self.assertEqual(len(seqs_h1), 4)

        # Horizon 2: 4*4 = 16 total, minus 4 opposite pairs (W-S, S-W, A-D, D-A) = 12
        seqs_h2 = generate_directional_candidate_sequences(2, prune_reversals=True)
        self.assertEqual(len(seqs_h2), 12)
        for seq in seqs_h2:
            self.assertNotIn(seq, [(DirectionalAction.W, DirectionalAction.S),
                                   (DirectionalAction.S, DirectionalAction.W),
                                   (DirectionalAction.A, DirectionalAction.D),
                                   (DirectionalAction.D, DirectionalAction.A)])

        # Horizon 3 with reversals allowed: 4^3 = 64
        seqs_h3_raw = generate_directional_candidate_sequences(3, prune_reversals=False)
        self.assertEqual(len(seqs_h3_raw), 64)

        # Horizon 6 with reversals pruned: 4 * 3^5 = 972
        seqs_h6 = generate_directional_candidate_sequences(6, prune_reversals=True)
        self.assertEqual(len(seqs_h6), 972)

        # Horizon bounds checking
        with self.assertRaises(ValueError):
            generate_directional_candidate_sequences(0)
        with self.assertRaises(ValueError):
            generate_directional_candidate_sequences(9)

    def test_directional_to_control_vector(self) -> None:
        """Test conversion of directional actions to 307-d HID control vector."""
        vec_w = directional_to_control_vector(DirectionalAction.W)
        self.assertEqual(vec_w.shape, (307,))
        self.assertEqual(vec_w[int(HidKey.W)], 1.0)
        self.assertEqual(vec_w[int(HidKey.S)], 0.0)
        self.assertEqual(vec_w.sum().item(), 1.0)

        vec_none = directional_to_control_vector(DirectionalAction.NONE)
        self.assertEqual(vec_none.sum().item(), 0.0)

    def test_multi_step_latent_unrolling_shapes(self) -> None:
        """Test unrolling H steps (H in [1, 5]) in latent space produces exact expected tensor shapes."""
        config = tiny_config()
        model = IreneBrainModel(config)
        state = model.initial_state(1)
        sensors = torch.randn(1, config.sensor_tokens, config.core_width)

        for h in [1, 2, 3, 4, 5]:
            planner = LatentLookaheadPlanner(model=model, config=config, horizon=h)
            plan_res = planner.plan(state=state, sensors=sensors, horizon=h)

            self.assertIsInstance(plan_res, LookaheadPlanResult)
            self.assertIsInstance(plan_res.best_action, DirectionalAction)
            self.assertEqual(plan_res.best_action_control.shape, (config.actuator.total_queries,))
            self.assertEqual(len(plan_res.best_branch.action_sequence), h)
            self.assertEqual(len(plan_res.best_branch.rollout_steps), h)

            for step in plan_res.best_branch.rollout_steps:
                self.assertIsInstance(step, LookaheadRolloutStep)
                self.assertIsInstance(step.predicted_reward, float)
                self.assertIsInstance(step.predicted_danger, float)
                self.assertIsInstance(step.predicted_value, float)
                self.assertIsInstance(step.is_hazard, bool)

    def test_gradient_isolation_and_state_immutability(self) -> None:
        """Verify that planner evaluation does not mutate or attach gradients to persistent BrainState."""
        config = tiny_config()
        model = IreneBrainModel(config)
        state = model.initial_state(1)
        sensors = torch.randn(1, config.sensor_tokens, config.core_width)

        # Snapshot tensors before planning
        b_before = state.belief.clone()
        m_before = state.working_memory.clone()
        t_before = state.thoughts.clone()
        g_before = state.goal_context.clone()
        a_before = state.thought_age_seconds.clone()

        planner = LatentLookaheadPlanner(model=model, config=config, horizon=3)
        plan_res = planner.plan(state=state, sensors=sensors)

        # Ensure state was not modified in-place
        self.assertTrue(torch.equal(state.belief, b_before))
        self.assertTrue(torch.equal(state.working_memory, m_before))
        self.assertTrue(torch.equal(state.thoughts, t_before))
        self.assertTrue(torch.equal(state.goal_context, g_before))
        self.assertTrue(torch.equal(state.thought_age_seconds, a_before))

        # Ensure no grad on outputs
        self.assertFalse(plan_res.best_action_control.requires_grad)

    def test_hazard_avoidance_and_branch_pruning(self) -> None:
        """Test that high predicted hazards cause branches to be pruned and safe paths chosen."""
        config = tiny_config()
        planner = LatentLookaheadPlanner(
            config=config,
            horizon=2,
            hazard_weight=10.0,
            hazard_prune_threshold=0.4,
        )
        state = IreneBrainModel(config).initial_state(1)
        sensors = torch.zeros(1, config.sensor_tokens, config.core_width)

        # Mock the hazard and reward heads to simulate a ghost trap on action W
        # and a reward on action D
        def mock_hazard_forward(thought_summary: torch.Tensor, action_feature: torch.Tensor | None = None) -> torch.Tensor:
            batch = thought_summary.shape[0]
            dangers = torch.zeros(batch)
            if action_feature is not None:
                # Check if W is active (action feature contains control token)
                pass
            return dangers

        # Create two explicit candidate branches: (W, W) which hits a hazard, and (D, D) which is safe and gets reward
        cand_seqs = [
            (DirectionalAction.W, DirectionalAction.W),
            (DirectionalAction.D, DirectionalAction.D),
        ]

        # Force hazard head to output 0.9 for branch 0 (W, W) and 0.01 for branch 1 (D, D)
        original_hazard_fwd = planner.hazard_head.forward
        original_reward_fwd = planner.reward_head.forward

        step_counter = [0]
        def custom_hazard_fwd(ts: torch.Tensor, af: torch.Tensor | None = None) -> torch.Tensor:
            batch = ts.shape[0]
            # When branch 0 (W) is unrolled, danger is high (0.85). For branch 1 (D), danger is 0.05
            danger = torch.zeros(batch)
            danger[0] = 0.85
            danger[1] = 0.05
            return danger

        def custom_reward_fwd(ts: torch.Tensor, af: torch.Tensor | None = None) -> torch.Tensor:
            batch = ts.shape[0]
            reward = torch.zeros(batch)
            reward[0] = 0.1
            reward[1] = 1.5
            return reward

        planner.hazard_head.forward = custom_hazard_fwd  # type: ignore[assignment]
        planner.reward_head.forward = custom_reward_fwd  # type: ignore[assignment]

        plan_res = planner.plan(
            state=state,
            sensors=sensors,
            candidate_sequences=cand_seqs,
            horizon=2,
        )

        # Branch 0 (W, W) should be pruned due to hazard >= 0.4
        self.assertTrue(plan_res.all_branches[0].is_pruned)
        self.assertIn("hazard_collision", plan_res.all_branches[0].prune_reason or "")
        self.assertEqual(plan_res.all_branches[0].cumulative_utility, -1e9)

        # Branch 1 (D, D) should NOT be pruned and should have positive utility
        self.assertFalse(plan_res.all_branches[1].is_pruned)
        self.assertGreater(plan_res.all_branches[1].cumulative_utility, 0.0)

        # Planner must pick DirectionalAction.D
        self.assertEqual(plan_res.best_action, DirectionalAction.D)
        self.assertEqual(plan_res.chosen_index, 1)
        self.assertEqual(plan_res.pruned_count, 1)

    def test_deterministic_execution(self) -> None:
        """Verify identical repeated executions produce bit-for-bit identical plan results."""
        config = tiny_config()
        model = IreneBrainModel(config)
        state = model.initial_state(1)
        sensors = torch.randn(1, config.sensor_tokens, config.core_width)

        planner = LatentLookaheadPlanner(model=model, config=config, horizon=2)
        res1 = planner.plan(state=state, sensors=sensors)
        res2 = planner.plan(state=state, sensors=sensors)

        self.assertEqual(res1.best_action, res2.best_action)
        self.assertTrue(torch.equal(res1.best_action_control, res2.best_action_control))
        self.assertEqual(res1.chosen_index, res2.chosen_index)
        self.assertEqual(res1.branch_utilities, res2.branch_utilities)

    def test_zero_cheating_compliance(self) -> None:
        """Verify policy contract: uses_privileged_state is False and operates purely in latent space."""
        config = tiny_config()
        model = IreneBrainModel(config)
        policy = LatentLookaheadPolicy(model=model, horizon=2)

        self.assertFalse(policy.uses_privileged_state)
        self.assertEqual(policy.identity, "model.latent_lookahead.v1")

    def test_latent_lookahead_policy_decide_and_act(self) -> None:
        """Test LatentLookaheadPolicy closed-loop decision making and state progression."""
        config = tiny_config()
        model = IreneBrainModel(config)
        policy = LatentLookaheadPolicy(model=model, horizon=2)

        policy.reset(12345)
        self.assertEqual(len(policy.plan_history), 0)
        self.assertIsNone(policy.last_plan)

        # Synthetic 32x32 RGB observation
        frame = RgbFrame(width=32, height=32, pixels=bytes(32 * 32 * 3))
        obs1 = Observation(
            frame_id=0,
            capture_tick=0,
            elapsed_ns=0,
            rgb=frame,
            previous_control=GenericControl(),
        )

        control1, stats1, val1 = policy.decide(obs1, elapsed_seconds=1.0 / 60.0)
        self.assertIsInstance(control1, GenericControl)
        self.assertIn("lookahead_utility", stats1)
        self.assertIn("lookahead_pruned_count", stats1)
        self.assertEqual(len(policy.plan_history), 1)
        self.assertIsNotNone(policy.last_plan)

        # Subsequent step
        obs2 = Observation(
            frame_id=1,
            capture_tick=16_666_667,
            elapsed_ns=16_666_667,
            rgb=frame,
            previous_control=control1,
        )
        control2 = policy.act(obs2)
        self.assertIsInstance(control2, GenericControl)
        self.assertEqual(len(policy.plan_history), 2)


if __name__ == "__main__":
    unittest.main()
