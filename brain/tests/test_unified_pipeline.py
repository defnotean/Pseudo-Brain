"""Unit tests for the unified Irene pipeline integration."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.environments.maze_chase import MazeChaseEnv
    from irene_brain.environments.moving_shapes import MovingShapesEnv
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayConfig,
        STRUCTURED_ACTION_GROUP_V1,
        evaluate_closed_loop_play,
        run_policy_closed_loop_episode,
    )
    from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
    from irene_brain.model.action_groups import (
        StructuredActionLoss,
        StructuredActionSpec,
        StructuredActuatorHead,
    )
    from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.types import GenericControl, HidKey


@unittest.skipUnless(torch is not None, "PyTorch is required for pipeline tests")
class UnifiedPipelineIntegrationTests(unittest.TestCase):
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

    def test_structured_actuator_integrated_with_brain_model(self) -> None:
        """Verify IreneBrainModel outputs can feed directly into StructuredActuatorHead and StructuredActionLoss."""
        assert torch is not None
        config = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config)
        model.eval()

        batch = 2
        pixels = torch.randn(batch, 3, 32, 32)
        prev_ctrl = torch.zeros(batch, 307)
        elapsed = torch.full((batch,), 1.0 / 60.0)

        with torch.no_grad():
            output = model(pixels, prev_ctrl, elapsed)

        # Connect model thought state to StructuredActuatorHead
        spec = StructuredActionSpec(width=config.core_width, movement_mode="5way")
        structured_head = StructuredActuatorHead(width=config.core_width, spec=spec)
        loss_fn = StructuredActionLoss(spec=spec)

        thought_features = output.diagnostics.thought_summaries.mean(dim=1)  # [batch, width]
        pred = structured_head(features=thought_features)

        self.assertEqual(pred.movement_probs.shape, (batch, 5))
        self.assertEqual(pred.control_vector.shape, (batch, 307))

        target = torch.zeros(batch, 307)
        target[0, int(HidKey.W)] = 1.0
        target[1, int(HidKey.D)] = 1.0

        loss_out = loss_fn(pred, target)
        self.assertGreater(float(loss_out.loss.detach()), 0.0)
        self.assertIn("movement_loss", loss_out.metrics)

    def test_latent_lookahead_policy_closed_loop_maze_chase(self) -> None:
        """Verify LatentLookaheadPolicy drives a closed-loop MazeChaseEnv episode."""
        assert torch is not None
        config = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config)
        model.eval()

        planner = LatentLookaheadPlanner(
            model=model,
            horizon=2,
            gamma=0.9,
            hazard_weight=3.0,
        )
        policy = LatentLookaheadPolicy(model=model, planner=planner)

        play_config = ClosedLoopPlayConfig(
            episode_seeds=(101,),
            max_ticks=20,
            hazard_count=2,
            decode_kind=STRUCTURED_ACTION_GROUP_V1,
        )

        def maze_factory() -> MazeChaseEnv:
            return MazeChaseEnv(
                max_ticks=20,
                ghost_count=2,
            )

        report = run_policy_closed_loop_episode(
            policy,
            seed=101,
            config=play_config,
            environment_factory=maze_factory,
        )

        self.assertEqual(report.episode_seed, 101)
        self.assertGreaterEqual(report.ticks_advanced, 1)
        self.assertEqual(report.opposite_conflicts, 0)
        self.assertEqual(report.continuous_outside_deadzone, 0)

    def test_evaluate_closed_loop_play_with_structured_action_decode(self) -> None:
        """Verify evaluate_closed_loop_play works with STRUCTURED_ACTION_GROUP_V1."""
        assert torch is not None
        from dataclasses import replace
        base_config = ThoughtFieldConfig.smoke()
        config = replace(
            base_config,
            actuator=replace(base_config.actuator, continuous_squash="deadzone_tanh"),
        )
        model = IreneBrainModel(config)
        model.eval()

        play_config = ClosedLoopPlayConfig(
            episode_seeds=(202,),
            max_ticks=15,
            hazard_count=1,
            decode_kind=STRUCTURED_ACTION_GROUP_V1,
        )

        report = evaluate_closed_loop_play(
            model,
            config=play_config,
            model_description="smoke_test_model",
        )

        self.assertEqual(len(report.episodes), 1)
        ep = report.episodes[0]
        self.assertEqual(ep.opposite_conflicts, 0)
        self.assertEqual(ep.continuous_outside_deadzone, 0)


if __name__ == "__main__":
    unittest.main()
