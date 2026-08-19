"""Unit tests for spatial_cognitive_diagnostics.py."""

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
        compute_model_param_digest,
        run_instrumented_diagnostic_episode,
    )
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel


@unittest.skipUnless(torch is not None, "PyTorch required")
class SpatialCognitiveDiagnosticsTests(unittest.TestCase):
    def test_maze_topology_classification(self) -> None:
        maze = {(1, 1), (1, 2), (2, 1), (1, 0), (0, 1)}
        topology = _classify_maze_topology(maze)
        self.assertEqual(topology[(1, 1)], "junction")  # 4 neighbors
        self.assertEqual(topology[(1, 2)], "dead_end")  # 1 neighbor
        self.assertEqual(topology[(2, 1)], "dead_end")

    def test_model_param_digest_stability(self) -> None:
        config = ThoughtFieldConfig.smoke()
        model1 = IreneBrainModel(config)
        model2 = IreneBrainModel(config)
        digest1 = compute_model_param_digest(model1)
        digest2 = compute_model_param_digest(model2)
        self.assertEqual(len(digest1), 16)
        self.assertNotEqual(digest1, digest2)  # Different random seeds produce different digests

    def test_run_instrumented_diagnostic_episode(self) -> None:
        config = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config)
        policy = LatentLookaheadPolicy(model=model)
        expert = ScriptedMazeChasePlannerPolicy()
        env = MazeChaseEnv(max_ticks=20, ghost_count=2)

        telemetry = run_instrumented_diagnostic_episode(
            policy=policy,
            env=env,
            seed=42,
            max_ticks=20,
            expert_policy=expert,
        )

        self.assertIsInstance(telemetry, SpatialCognitiveTelemetry)
        self.assertEqual(telemetry.seed, 42)
        self.assertGreater(telemetry.ticks_simulated, 0)
        self.assertGreaterEqual(telemetry.mean_nearest_ghost_dist, 0.0)
        self.assertIn("unsafe_intersection_entries", telemetry.to_dict())
        self.assertIn("expert_disagreement_pct", telemetry.to_dict())


if __name__ == "__main__":
    unittest.main()
