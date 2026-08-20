"""Unit tests verifying real-time 60 Hz arcade loop execution and latency compliance."""

from __future__ import annotations

from dataclasses import replace
import time
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.environments.maze_chase import MazeChaseEnv
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayConfig,
        STRUCTURED_ACTION_GROUP_V1,
        run_policy_closed_loop_episode,
    )
    from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
    from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.runtime.clock import ManualClock
    from irene_brain.runtime.continuous import ContinuousDriver
    from irene_brain.types import GenericControl


@unittest.skipUnless(torch is not None, "PyTorch is required for real-time arcade tests")
class RealtimeArcadePlayTests(unittest.TestCase):
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

    def test_60hz_arcade_tick_budget_and_latency(self) -> None:
        """Verify the model comfortably completes inference within the 60 Hz 16.6ms tick budget."""
        assert torch is not None
        base_config = ThoughtFieldConfig.smoke()
        model_config = replace(
            base_config,
            core_width=32,
            thoughtlets=4,
            cognitive_cycles=2,
            actuator=replace(base_config.actuator, continuous_squash="deadzone_tanh"),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IreneBrainModel(model_config).to(device)
        model.eval()

        planner = LatentLookaheadPlanner(
            model=model,
            horizon=1,
            gamma=0.95,
            hazard_weight=4.0,
        )
        policy = LatentLookaheadPolicy(model=model, planner=planner, selective_gating=True, device=device)
        policy.reset(1702)

        env = MazeChaseEnv(max_ticks=60, ghost_count=2)
        obs = env.reset(1702)

        # Warm-up inference
        _ = policy.decide(obs, 1.0 / 60.0)

        # Measure 60 consecutive frame decisions
        latencies_ms = []
        for _ in range(60):
            t0 = time.perf_counter()
            ctrl, stats, _ = policy.decide(obs, 1.0 / 60.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            outcome = env.step(ctrl)
            obs = outcome.observation
            if outcome.terminated or outcome.truncated:
                obs = env.reset(1702)

        p50_ms = sorted(latencies_ms)[len(latencies_ms) // 2]
        p99_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]

        # 60 Hz frame interval is 16.67 ms. Inference should easily pass well under the budget.
        self.assertLess(p50_ms, 16.67, f"p50 latency {p50_ms:.2f}ms exceeds 16.67ms frame budget")
        self.assertLess(p99_ms, 45.0, f"p99 latency {p99_ms:.2f}ms exceeds safety margin")

    def test_continuous_driver_simulated_realtime_maze_chase(self) -> None:
        """Verify ContinuousDriver runs a closed-loop episode under strict timing contracts."""
        assert torch is not None
        base_config = ThoughtFieldConfig.smoke()
        model_config = replace(
            base_config,
            core_width=32,
            thoughtlets=4,
            cognitive_cycles=2,
            actuator=replace(base_config.actuator, continuous_squash="deadzone_tanh"),
        )
        model = IreneBrainModel(model_config)
        model.eval()

        policy = LatentLookaheadPolicy(model=model)

        play_config = ClosedLoopPlayConfig(
            episode_seeds=(303,),
            max_ticks=30,
            hazard_count=2,
            tick_period_ns=16_666_667,  # 60 Hz
            decision_interval_ns=16_666_667,
            inference_latency_ns=2_000_000,  # 2 ms simulated latency
            decode_kind=STRUCTURED_ACTION_GROUP_V1,
        )

        def maze_factory() -> MazeChaseEnv:
            return MazeChaseEnv(max_ticks=30, ghost_count=2)

        report = run_policy_closed_loop_episode(
            policy,
            seed=303,
            config=play_config,
            environment_factory=maze_factory,
        )

        self.assertEqual(report.episode_seed, 303)
        self.assertGreaterEqual(report.ticks_advanced, 20)
        self.assertEqual(report.opposite_conflicts, 0)
        self.assertEqual(report.continuous_outside_deadzone, 0)
        self.assertEqual(report.decisions_rejected, 0)


if __name__ == "__main__":
    unittest.main()
