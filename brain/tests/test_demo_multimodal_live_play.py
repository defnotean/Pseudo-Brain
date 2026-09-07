"""Unit tests for the interactive multimodal live play demo.

Verifies:
1. Terminal ASCII grid rendering with agent, key, door, walls, corridor, and hazards.
2. Side-by-side dashboard view composition with slot energy bars and latency telemetry.
3. End-to-end episodic execution in smoke mode with zero token drift and milestone latch consolidation.
4. CLI subprocess execution with --smoke flag.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.semantic.multimodal_model import MultimodalPseudoBrainModel
from irene_brain.semantic.tokenizer import SemanticTokenizer
from memory_benchmark.difficulty_curve_ablation import CorridorDelayKeysDoorsEnv
from semantic_benchmark.demo_multimodal_live_play import (
    build_dashboard_view,
    render_ascii_grid,
    render_slot_energy_bar,
    run_live_play_episode,
)


class TestDemoMultimodalLivePlay(unittest.TestCase):
    """Test suite for the multimodal live play demo runner."""

    def setUp(self):
        torch.manual_seed(42)
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    def test_render_ascii_grid_components(self):
        """Verify ASCII grid correctly positions agent, key, door, walls, and hazards."""
        env = CorridorDelayKeysDoorsEnv(corridor_delay=4)
        env.reset(seed=1000)

        grid_lines = render_ascii_grid(env)
        grid_str = "\n".join(grid_lines)

        self.assertEqual(len(grid_lines), 18, "Grid must have 16 rows + 2 border lines.")
        self.assertIn("@", grid_str, "Player '@' must be visible.")
        self.assertIn("K", grid_str, "Key 'K' must be visible before pickup.")
        self.assertIn("D", grid_str, "Door 'D' must be visible before opening.")
        self.assertIn("#", grid_str, "Wall '#' cells must be present.")
        self.assertIn(".", grid_str, "Corridor '.' cells must be present.")

        # Trigger delay to test corridor hazard marker '^'
        env._delay_active = True
        env._delay_counter = 4
        hazard_grid = "\n".join(render_ascii_grid(env))
        self.assertIn("^", hazard_grid, "Corridor delay hazard '^' must appear during delay phase.")

    def test_dashboard_view_construction(self):
        """Verify dashboard composition generates required cognitive metrics and slot bars."""
        env = CorridorDelayKeysDoorsEnv(corridor_delay=4)
        env.reset(seed=1000)

        thoughts = torch.randn(4, 32)
        P_t = torch.zeros(4, 5)
        P_t[0, 1] = 4.0

        view = build_dashboard_view(
            env=env,
            tick=12,
            action=1,
            dt_ms=1.15,
            delta_consequence=2.0,
            milestone_latch_val=4.0,
            slot0_persistence=1.0000,
            thoughts=thoughts,
            P_t=P_t,
            directive="retrieve key, ignore hallway hazard, unlock blue door",
        )

        self.assertIn("PSEUDO-BRAIN COGNITIVE DASHBOARD", view)
        self.assertIn("Action Chosen:", view)
        self.assertIn("Consequence Surprise:", view)
        self.assertIn("Milestone Latch P_t:", view)
        self.assertIn("CONSOLIDATED", view)
        self.assertIn("Slot 0 Persistence:", view)
        self.assertIn("THOUGHT SLOT ENERGY MATRIX", view)
        self.assertIn("Slot 0 [Language Directive]", view)
        self.assertIn("Slot 1 [Visual Perception ]", view)

    def test_live_play_episode_execution_smoke(self):
        """Verify end-to-end live play episode in smoke mode satisfies 100% goals and 60Hz latency."""
        tokenizer = SemanticTokenizer(max_threads=4)
        model = MultimodalPseudoBrainModel(
            vocab_size=tokenizer.vocab_size,
            K=4,
            thought_size=32,
            embed_dim=32,
            proj_dim=64,
            visual_dim=64,
            num_visual_tokens=4,
            n_actions=5,
        )

        env = CorridorDelayKeysDoorsEnv(corridor_delay=4)
        res = run_live_play_episode(
            model=model,
            tokenizer=tokenizer,
            env=env,
            seed=1000,
            directive="retrieve key, ignore hallway hazard, unlock blue door",
            smoke=True,
            interactive=False,
            device=torch.device("cpu"),
        )

        self.assertTrue(res["success"], "Episode must complete successfully.")
        self.assertTrue(res["key_collected"], "Key must be collected.")
        self.assertTrue(res["door_unlocked"], "Door must be unlocked.")
        self.assertTrue(res["target_collected"], "Target must be reached.")
        self.assertGreaterEqual(
            res["prereq_latch_milestone"], 4.0, "Milestone latch P_t must consolidate (>= 4.0)."
        )
        self.assertGreaterEqual(
            res["slot0_persistence_final"],
            0.999,
            "Slot 0 thought must maintain 1.000 persistence without token replay.",
        )
        self.assertTrue(
            res["latency_under_16_67ms"], "Steady-state latency must satisfy 16.67ms (60Hz) SLA."
        )
        self.assertLess(
            res["mean_latency_ms"], 5.0, "Mean latency should be ~1-2ms on CPU."
        )

    def test_cli_smoke_execution(self):
        """Verify demo script CLI runs cleanly with --smoke flag."""
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "experiments" / "semantic_benchmark" / "demo_multimodal_live_play.py"),
            "--smoke",
            "--episodes",
            "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
        self.assertEqual(proc.returncode, 0, f"CLI exited with code {proc.returncode}: {proc.stderr}")
        self.assertIn("SUCCESS", proc.stdout)
        self.assertIn("Latch P_t: 4.", proc.stdout)
        self.assertIn("Slot0 Persist: 1.0000", proc.stdout)
        self.assertIn("DEMO RUN SUMMARY", proc.stdout)


if __name__ == "__main__":
    unittest.main()
