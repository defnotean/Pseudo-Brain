"""Unit tests for Phase 2 task suite and baseline models."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from irene_brain.environments.phase2_suite import (
    Phase2TaskEnvironment,
    Phase2WorldConfig,
    TaskFamily,
    make_family_suite,
)
from irene_brain.model.phase2_baselines import (
    BaselineVariant,
    build_phase2_model,
)
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.types import GenericControl, HidKey


class Phase2SuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_all_five_task_families_instantiate_and_step(self) -> None:
        """Verify each of the 5 task families generates valid configs and runs steps."""
        for family in TaskFamily:
            configs = make_family_suite(family)
            self.assertGreater(len(configs), 0)
            env = Phase2TaskEnvironment(configs[0])
            obs = env.reset(100)
            self.assertEqual(len(obs.rgb.pixels), 16 * 16 * 3)

            ctrl = GenericControl(keys_down=(int(HidKey.W),))
            outcome = env.step(ctrl)
            self.assertEqual(len(outcome.observation.rgb.pixels), 16 * 16 * 3)
            self.assertIsInstance(outcome.reward, float)

    def test_control_remapping(self) -> None:
        """Verify inverted and rotated control remappings."""
        cfg_inv = Phase2WorldConfig(family=TaskFamily.FAMILY_E_CHANGED_DYNAMICS, control_remapping="inverted")
        env_inv = Phase2TaskEnvironment(cfg_inv)
        ctrl_w = GenericControl(keys_down=(int(HidKey.W),))
        remapped_w = env_inv._remap_control(ctrl_w)
        self.assertEqual(remapped_w.keys_down, (int(HidKey.S),))

        cfg_rot = Phase2WorldConfig(family=TaskFamily.FAMILY_E_CHANGED_DYNAMICS, control_remapping="rotated_90")
        env_rot = Phase2TaskEnvironment(cfg_rot)
        remapped_rot = env_rot._remap_control(ctrl_w)
        self.assertEqual(remapped_rot.keys_down, (int(HidKey.D),))

    def test_occlusion_masking(self) -> None:
        """Verify partial observability fogs unobserved regions."""
        cfg_occ = Phase2WorldConfig(family=TaskFamily.FAMILY_D_PARTIAL_OBSERVABILITY, occlusion_radius=2)
        env_occ = Phase2TaskEnvironment(cfg_occ)
        obs = env_occ.reset(100)
        raw = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
        px = getattr(env_occ._underlying_env, "_player_x", 8)
        py = getattr(env_occ._underlying_env, "_player_y", 8)
        far_x = 0 if px > 8 else 15
        far_y = 0 if py > 8 else 15
        far_mean = float(raw[far_y, far_x].mean())
        self.assertLess(far_mean, 50.0)

    def test_all_eight_phase2_baselines_instantiate_and_forward(self) -> None:
        """Verify all 8 baseline variants plus Pseudo-Brain instantiate and run."""
        base_config = ThoughtFieldConfig(
            thoughtlets=4,
            cognitive_cycles=2,
            core_width=16,
            attention_heads=2,
            sensor_tokens=4,
            belief_tokens=2,
            working_memory_tokens=2,
            goal_context_tokens=4,
            registers_per_thoughtlet=2,
            brain_cell_blocks=1,
            routed_neighbors=1,
        )
        pixels = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
        ctrl = torch.zeros((1, base_config.actuator.total_queries), dtype=torch.float32)
        dt = torch.tensor([0.016], dtype=torch.float32)

        for variant in BaselineVariant:
            model = build_phase2_model(variant, base_config=base_config)
            model.eval()
            state = model.initial_state(1)
            with torch.no_grad():
                out = model(pixels, ctrl, dt, state)
            self.assertEqual(out.action.button_logits.shape, (1, 296), f"Failed for {variant}")
            self.assertTrue(torch.isfinite(out.action.button_logits).all(), f"NaNs in {variant}")
            self.assertTrue(torch.isfinite(out.next_state.thoughts).all(), f"NaNs in {variant} thoughts")


if __name__ == "__main__":
    unittest.main()
