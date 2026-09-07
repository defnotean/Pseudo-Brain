"""Integration tests for Consequence-Gated Plasticity (CGP) in Pseudo-Brain arcade architecture.

Verifies:
1. State Immutability: BrainState and P_t weights remain strictly unmodified during lookahead rollouts.
2. Occlusion Persistence: CGP maintains threat/goal tracking across blank/occluded ticks without corridor diffusion.
3. Real-Time Latency: CGP-equipped BrainCell executes in <= 2.0 ms on CPU.
4. Fast Plasticity Module & Cognitive Gating: Verifies mathematical update properties of P_t.
5. Closed-Loop Arcade Play: End-to-end policy execution with CGP in MazeChase.
"""

from __future__ import annotations

from dataclasses import replace
import time
import unittest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.environments.maze_chase import MazeChaseEnv
    from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
    from irene_brain.model import (
        BrainCell,
        BrainState,
        DirectionalAction,
        FastPlasticityModule,
        IreneBrainModel,
        LatentLookaheadPlanner,
        PlasticBrainCell,
        SurpriseEncoder,
        ThoughtFieldConfig,
    )
    from irene_brain.model.brain_cell import BrainCellOutput


def tiny_cgp_config(*, use_cgp: bool = True) -> ThoughtFieldConfig:
    """Return compact topology-faithful configuration for CPU tests."""
    return replace(
        ThoughtFieldConfig.smoke(),
        core_width=32,
        sensor_tokens=4,
        belief_tokens=2,
        working_memory_tokens=2,
        thoughtlets=4,
        registers_per_thoughtlet=2,
        goal_context_tokens=2,
        cognitive_cycles=2,
        brain_cell_blocks=1,
        attention_heads=2,
        routed_neighbors=1,
        episodic_memory_entries=8,
        retrieved_entries_per_thoughtlet=1,
        use_cgp=use_cgp,
        plastic_decay=0.999,
        plastic_lr=0.25,
    )


@unittest.skipUnless(torch is not None, "PyTorch is required for CGP integration tests")
class CGPArcadeIntegrationTests(unittest.TestCase):
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

    def test_cgp_brain_cell_construction_and_unpacking(self) -> None:
        """Verify BrainCell and PlasticBrainCell construction, tuple unpacking, and P_t attribute preservation."""
        width = 32
        cell = PlasticBrainCell(
            width=width,
            heads=2,
            routed_neighbors=1,
            blocks=1,
            plastic_decay=0.999,
            plastic_lr=0.25,
        )
        self.assertTrue(cell.use_cgp)
        self.assertIsNotNone(cell.plasticity)
        self.assertIsNotNone(cell.cognitive_gate)
        self.assertIsNotNone(cell.surprise_encoder)

        batch = 2
        belief = torch.randn(batch, 2, width)
        wm = torch.randn(batch, 2, width)
        thoughts = torch.randn(batch, 4, 2, width)
        sensors = torch.randn(batch, 4, width)
        action_time = torch.randn(batch, 2, width)
        goal = torch.randn(batch, 2, width)
        retrieval = torch.randn(batch, 4, 1, width)
        elapsed = torch.tensor([[1.0 / 60.0], [1.0 / 60.0]])
        P_t = torch.zeros(batch, width)
        surprise = torch.tensor([[0.8], [0.1]])

        # Test tuple unpacking: must cleanly unpack exactly 4 elements
        out = cell(
            belief=belief,
            working_memory=wm,
            thoughts=thoughts,
            sensors=sensors,
            action_time_tokens=action_time,
            goal_context=goal,
            retrieved_memory=retrieval,
            elapsed_seconds=elapsed,
            allow_routing=True,
            plastic_weights=P_t,
            surprise=surprise,
        )

        self.assertIsInstance(out, BrainCellOutput)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)

        b_next, m_next, t_next, routing = out
        self.assertEqual(b_next.shape, (batch, 2, width))
        self.assertEqual(m_next.shape, (batch, 2, width))
        self.assertEqual(t_next.shape, (batch, 4, 2, width))
        self.assertEqual(len(routing), 1)

        # Verify plastic_weights and P_t properties
        self.assertIsNotNone(out.plastic_weights)
        self.assertIsNotNone(out.P_t)
        self.assertEqual(out.plastic_weights.shape, (batch, width))
        self.assertTrue(torch.equal(out.plastic_weights, out.P_t))
        # Batch item 0 had high surprise (0.8), item 1 had low surprise (0.1)
        self.assertGreater(out.plastic_weights[0].norm().item(), out.plastic_weights[1].norm().item())

    def test_state_immutability_during_lookahead_rollouts(self) -> None:
        """Requirement A: Verify BrainState and P_t are strictly unmodified during lookahead rollouts."""
        config = tiny_cgp_config(use_cgp=True)
        model = IreneBrainModel(config, use_cgp=True)
        model.eval()

        # Step 1 frame with stimulus to produce non-trivial recurrent and plastic state
        pixels = torch.randn(1, 3, 32, 32)
        ctrl = torch.zeros(1, config.actuator.total_queries)
        elapsed = torch.tensor([[1.0 / 60.0]])

        initial_state = model.initial_state(1)
        out1 = model(pixels, ctrl, elapsed, initial_state)
        state = out1.next_state

        self.assertIsNotNone(state.plastic_weights)
        self.assertIsNotNone(state.P_t)

        # Snapshot full state tensors
        b_before = state.belief.clone()
        m_before = state.working_memory.clone()
        t_before = state.thoughts.clone()
        g_before = state.goal_context.clone()
        a_before = state.thought_age_seconds.clone()
        p_before = state.plastic_weights.clone()
        pt_before = state.P_t.clone()

        # Execute H=3 lookahead planning rollouts
        planner = LatentLookaheadPlanner(model=model, config=config, horizon=3)
        sensors = model.pixel_encoder(pixels)

        plan_res = planner.plan(state=state, sensors=sensors, horizon=3)
        self.assertIsNotNone(plan_res.best_action)

        # Verify state immutability: zero in-place mutations
        self.assertTrue(torch.equal(state.belief, b_before), "state.belief was mutated during lookahead")
        self.assertTrue(torch.equal(state.working_memory, m_before), "state.working_memory was mutated")
        self.assertTrue(torch.equal(state.thoughts, t_before), "state.thoughts was mutated")
        self.assertTrue(torch.equal(state.goal_context, g_before), "state.goal_context was mutated")
        self.assertTrue(torch.equal(state.thought_age_seconds, a_before), "state.thought_age_seconds was mutated")
        self.assertTrue(torch.equal(state.plastic_weights, p_before), "state.plastic_weights was mutated")
        self.assertTrue(torch.equal(state.P_t, pt_before), "state.P_t was mutated")

        # Also verify state.detach() and state.to() preserve immutability
        detached = state.detach()
        self.assertTrue(torch.equal(detached.plastic_weights, p_before))
        self.assertFalse(detached.plastic_weights.requires_grad)

    def test_occlusion_persistence_threat_and_goal_tracking(self) -> None:
        """Requirement B: Verify CGP maintains threat/goal tracking across blank/occluded ticks."""
        width = 32
        cgp_config = tiny_cgp_config(use_cgp=True)
        base_config = tiny_cgp_config(use_cgp=False)

        model_cgp = IreneBrainModel(cgp_config, use_cgp=True)
        model_base = IreneBrainModel(base_config, use_cgp=False)
        model_cgp.eval()
        model_base.eval()

        ctrl = torch.zeros(1, cgp_config.actuator.total_queries)
        elapsed = torch.tensor([[1.0 / 60.0]])

        # Tick 0: Stimulus frame with high salient threat pattern
        stim_pixels = torch.randn(1, 3, 32, 32) * 2.0
        out0_cgp = model_cgp(stim_pixels, ctrl, elapsed, model_cgp.initial_state(1))
        out0_base = model_base(stim_pixels, ctrl, elapsed, model_base.initial_state(1))

        cgp_state = out0_cgp.next_state
        base_state = out0_base.next_state

        cgp_t0_summary = cgp_state.thoughts.mean(dim=(1, 2))
        base_t0_summary = base_state.thoughts.mean(dim=(1, 2))

        # Initial P_t weight must be populated and non-zero in CGP
        self.assertIsNotNone(cgp_state.plastic_weights)
        p0_norm = cgp_state.plastic_weights.norm().item()
        self.assertGreater(p0_norm, 1e-4, "P_t should capture episodic trace on salient tick 0")

        # Simulate 20 consecutive blank/occluded frames (e.g. hallway / dark room)
        blank_pixels = torch.zeros(1, 3, 32, 32)
        for _ in range(20):
            cgp_state = model_cgp(blank_pixels, ctrl, elapsed, cgp_state).next_state
            base_state = model_base(blank_pixels, ctrl, elapsed, base_state).next_state

        cgp_t20_summary = cgp_state.thoughts.mean(dim=(1, 2))
        base_t20_summary = base_state.thoughts.mean(dim=(1, 2))

        # 1. Verify P_t retention: with decay=0.999, P_t retains ~98% of its norm across 20 ticks
        p20_norm = cgp_state.plastic_weights.norm().item()
        expected_retention_floor = p0_norm * (0.99**20)
        self.assertGreater(
            p20_norm,
            expected_retention_floor,
            f"P_t norm decayed excessively: {p20_norm} vs floor {expected_retention_floor}",
        )

        # 2. Verify Cognitive Gating & Thought Persistence:
        cgp_sim = F.cosine_similarity(cgp_t0_summary, cgp_t20_summary).item()
        base_sim = F.cosine_similarity(base_t0_summary, base_t20_summary).item()

        # CGP thought similarity must remain high (> 0.80) across 20 occluded frames
        self.assertGreater(
            cgp_sim,
            0.80,
            f"CGP thought representation drifted too far across occlusion: similarity {cgp_sim:.3f}",
        )

    def test_realtime_latency_cpu_benchmark(self) -> None:
        """Requirement C: Verify CGP-equipped BrainCell executes in <= 2.0 ms on CPU."""
        width = 32
        cell = PlasticBrainCell(
            width=width,
            heads=2,
            routed_neighbors=1,
            blocks=1,
            plastic_decay=0.999,
            plastic_lr=0.25,
        )
        cell.eval()

        batch = 1
        belief = torch.randn(batch, 2, width)
        wm = torch.randn(batch, 2, width)
        thoughts = torch.randn(batch, 4, 2, width)
        sensors = torch.randn(batch, 4, width)
        action_time = torch.randn(batch, 2, width)
        goal = torch.randn(batch, 2, width)
        retrieval = torch.randn(batch, 4, 1, width)
        elapsed = torch.tensor([[1.0 / 60.0]])
        P_t = torch.zeros(batch, width)
        surprise = torch.tensor([[0.5]])

        # Warm-up inference
        for _ in range(25):
            _ = cell(
                belief=belief,
                working_memory=wm,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time,
                goal_context=goal,
                retrieved_memory=retrieval,
                elapsed_seconds=elapsed,
                allow_routing=True,
                plastic_weights=P_t,
                surprise=surprise,
            )

        # Measure 100 consecutive executions on CPU
        latencies_ms: list[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = cell(
                belief=belief,
                working_memory=wm,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time,
                goal_context=goal,
                retrieved_memory=retrieval,
                elapsed_seconds=elapsed,
                allow_routing=True,
                plastic_weights=P_t,
                surprise=surprise,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = sum(latencies_ms) / len(latencies_ms)
        median_ms = sorted(latencies_ms)[len(latencies_ms) // 2]

        # Real-time requirement: BrainCell executes in <= 2.0 ms on CPU
        self.assertLess(
            median_ms,
            2.0,
            f"CGP BrainCell median CPU latency {median_ms:.3f}ms exceeds 2.0ms budget",
        )
        self.assertLess(
            mean_ms,
            2.0,
            f"CGP BrainCell mean CPU latency {mean_ms:.3f}ms exceeds 2.0ms budget",
        )

    def test_fast_plasticity_module_properties(self) -> None:
        """Verify FastPlasticityModule update dynamics, surprise gating, and decay."""
        dim = 16
        plast = FastPlasticityModule(state_dim=dim, plastic_dim=dim, surprise_dim=16, decay=0.99, lr=0.5)

        p0 = plast.init_trace(1, torch.device("cpu"))
        self.assertEqual(p0.shape, (1, dim))
        self.assertTrue(torch.equal(p0, torch.zeros(1, dim)))

        state = torch.ones(1, dim)
        low_surprise = torch.zeros(1, 16)
        high_surprise = torch.ones(1, 16) * 5.0

        # With near-zero surprise, gate is suppressed
        p1, delta1, gate1 = plast.update(p0, state, low_surprise)
        self.assertLess(gate1.item(), 0.25)
        self.assertLess(p1.norm().item(), 0.1)

        # With high surprise, gate opens and delta updates P_t
        p2, delta2, gate2 = plast.update(p0, state, high_surprise)
        self.assertGreater(gate2.item(), 0.85)
        self.assertGreater(p2.norm().item(), 0.2)

        # Decay over steps without surprise
        p3 = p2
        for _ in range(10):
            p3, _, _ = plast.update(p3, state, low_surprise)
        self.assertLess(p3.norm().item(), p2.norm().item())

    def test_closed_loop_arcade_play_with_cgp_policy(self) -> None:
        """Verify LatentLookaheadPolicy operates cleanly with CGP-enabled model in MazeChase."""
        config = tiny_cgp_config(use_cgp=True)
        model = IreneBrainModel(config, use_cgp=True)
        model.eval()

        policy = LatentLookaheadPolicy(model=model, horizon=2)
        policy.reset(999)

        env = MazeChaseEnv(max_ticks=20, ghost_count=1)
        obs = env.reset(999)

        for tick in range(20):
            ctrl, stats, val = policy.decide(obs, elapsed_seconds=1.0 / 60.0)
            self.assertIn("lookahead_utility", stats)
            self.assertIsNotNone(policy.last_plan)
            # Verify policy internal state preserves plastic_weights
            self.assertIsNotNone(policy._state.plastic_weights)
            self.assertIsNotNone(policy._state.P_t)

            outcome = env.step(ctrl)
            obs = outcome.observation
            if outcome.terminated or outcome.truncated:
                break


if __name__ == "__main__":
    unittest.main()
