"""Unit tests for upgraded spatial_cognitive_diagnostics.py."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from irene_brain.environments.maze_chase import MazeChaseEnv
    from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
    from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
    from irene_brain.evaluation.spatial_cognitive_diagnostics import (
        SpatialCognitiveTelemetry,
        _classify_maze_topology,
        compute_effective_rank,
        compute_model_param_digest,
        run_cognitive_depth_ablation,
        run_instrumented_diagnostic_episode,
    )
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel


@unittest.skipUnless(torch is not None, "PyTorch required")
class SpatialCognitiveDiagnosticsTests(unittest.TestCase):
    def test_effective_rank_computation(self) -> None:
        # Identical collinear rows -> rank ~ 1
        collinear = torch.ones(8, 32)
        rank_low = compute_effective_rank(collinear)
        self.assertAlmostEqual(rank_low, 1.0, delta=0.1)

        # Full rank random orthogonal matrix -> rank > 4
        random_mat = torch.randn(8, 32)
        rank_high = compute_effective_rank(random_mat)
        self.assertGreater(rank_high, 3.0)

    def test_maze_topology_classification(self) -> None:
        maze = {(1, 1), (1, 2), (2, 1), (1, 0), (0, 1)}
        topology = _classify_maze_topology(maze)
        self.assertEqual(topology[(1, 1)], "junction")  # 4 neighbors
        self.assertEqual(topology[(1, 2)], "dead_end")  # 1 neighbor
        self.assertEqual(topology[(2, 1)], "dead_end")

    def test_run_instrumented_diagnostic_episode(self) -> None:
        config = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config)
        policy = LatentLookaheadPolicy(model=model)
        expert = ScriptedMazeChasePlannerPolicy()
        env = MazeChaseEnv(max_ticks=25, ghost_count=2)

        telemetry = run_instrumented_diagnostic_episode(
            policy=policy,
            env=env,
            seed=42,
            max_ticks=25,
            expert_policy=expert,
        )

        self.assertIsInstance(telemetry, SpatialCognitiveTelemetry)
        self.assertEqual(telemetry.seed, 42)
        self.assertGreater(telemetry.ticks_simulated, 0)
        self.assertGreaterEqual(telemetry.mean_thoughtlet_effective_rank, 1.0)
        self.assertIn("catches_by_topology", telemetry.to_dict())
        self.assertIn("outcome_disagreement", telemetry.to_dict())

    def test_cognitive_depth_ablation(self) -> None:
        config = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config)
        ablation = run_cognitive_depth_ablation(
            model=model,
            seeds=(101,),
            cycles_list=(1, 2),
            max_ticks=10,
        )
        self.assertIn(1, ablation)
        self.assertIn(2, ablation)
        self.assertIn("mean_effective_rank", ablation[1])
        self.assertIn("latency_ms_per_step", ablation[1])


if __name__ == "__main__":
    unittest.main()
