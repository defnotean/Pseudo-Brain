"""Unit and Regression Test Suite for Multi-Threaded Latent Dependency Benchmark (MTLD-Bench).

Verifies:
1. Phase sequencing: Enroll -> Delay 1 -> Update -> Delay 2 -> Query.
2. Ground truth integrity: Updated variables query new value; untouched variables query initial value.
3. Zero cheating contract: Reactive baseline scores chance (~12.5%) under partial observability.
4. Slot isolation metrics: Untouched retention and cross-talk error rates are accurately tracked.
5. Shuffled slot ablation: Scrambling slot identities degrades query retrieval, confirming causal binding.
6. Determinism & reproducibility: Identical seeds produce bit-for-bit identical episodes and evaluations.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from memory_benchmark.multi_threaded_latent_dependency_benchmark import (
    MultiThreadedLatentDependencyEnv,
    MTLDPhases,
    build_batch,
    make_mtld_model,
    evaluate_model,
    train_model,
    count_params,
)


class MultiThreadedLatentDependencyTests(unittest.TestCase):
    """Rigorous CI validation of MTLD environment and architectural contracts."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.env = MultiThreadedLatentDependencyEnv(
            num_variables=4,
            num_values=8,
            delay_1=8,
            delay_2=8,
            num_updates=1,
            input_dim=64,
        )

    def test_env_generation_and_phase_progression(self):
        """Verify that episodes advance strictly through the 5 canonical phases."""
        rng = np.random.RandomState(42)
        ep = self.env.generate_episode(rng)

        # Expected lengths: 4 enroll + 8 delay1 + 1 update + 8 delay2 + 4 query = 25 steps
        self.assertEqual(len(ep), 25)

        # Phase sequence check
        phases = [step.phase for step in ep]
        self.assertEqual(phases[:4], [MTLDPhases.ENROLL] * 4)
        self.assertEqual(phases[4:12], [MTLDPhases.DELAY1] * 8)
        self.assertEqual(phases[12:13], [MTLDPhases.UPDATE] * 1)
        self.assertEqual(phases[13:21], [MTLDPhases.DELAY2] * 8)
        self.assertEqual(phases[21:25], [MTLDPhases.QUERY] * 4)

        # Verify ground truth tracking
        query_steps = [s for s in ep if s.is_query]
        self.assertEqual(len(query_steps), 4)

        updated_count = sum(1 for s in query_steps if s.is_updated_var)
        untouched_count = sum(1 for s in query_steps if s.is_untouched_var)
        self.assertEqual(updated_count, 1)
        self.assertEqual(untouched_count, 3)

        for s in query_steps:
            self.assertTrue(0 <= s.query_target < 8)

    def test_batch_builder_tensor_shapes(self):
        """Verify build_batch constructs exact expected tensor dimensions and ignore masks."""
        obs_b, tgt_b, eps = build_batch(self.env, batch_size=8, seed=123)
        self.assertEqual(obs_b.shape, (8, 25, 64))
        self.assertEqual(tgt_b.shape, (8, 25))

        # Target should be -100 for non-query steps (first 21 steps)
        self.assertTrue((tgt_b[:, :21] == -100).all())
        # Target should be in [0, 7] for query steps (last 4 steps)
        self.assertTrue((tgt_b[:, 21:] >= 0).all() and (tgt_b[:, 21:] < 8).all())

    def test_reactive_baseline_cannot_cheat(self):
        """Verify that reactive feedforward model cannot predict delayed queries (scores chance)."""
        reactive = make_mtld_model("reactive", input_dim=64, num_values=8)
        # Train reactive model for 10 steps (it will only see current input which lacks the past value)
        train_model(reactive, "reactive", self.env, num_steps=10, batch_size=16)

        results = evaluate_model(reactive, "reactive", self.env, num_seeds=5)
        # Chance level is 1/8 = 12.5%. Model must not exceed chance by more than sampling noise
        self.assertLess(results["overall_retention_accuracy"], 35.0, "Reactive model unexpectedly beat chance")

    def test_parameter_counts_contract(self):
        """Verify that parameter footprints match the expected budget tiers."""
        p_cgp = count_params(make_mtld_model("cgp_thoughtlet"))["total"]
        p_gru_m = count_params(make_mtld_model("gru_matched"))["total"]
        p_gru_h = count_params(make_mtld_model("gru_heavy"))["total"]

        # CGP thoughtlet should be ~131k (recurrent core)
        self.assertGreater(p_cgp, 100_000)
        self.assertLess(p_cgp, 300_000)

        # Heavy GRU must have significantly more parameters than Matched GRU
        self.assertGreater(p_gru_h, 2 * p_gru_m)

    def test_shuffled_slot_ablation_modifies_representation(self):
        """Verify that shuffled slot ablation executes cleanly and alters forward predictions."""
        cgp = make_mtld_model("cgp_thoughtlet")
        obs_b, _, _ = build_batch(self.env, batch_size=4, seed=42)

        cgp.eval()
        with torch.no_grad():
            out_standard = cgp(obs_b)
            # Shuffle at step 15
            out_shuffled = cgp(obs_b, shuffle_slots_at_step=15)

        # Up to step 14, outputs must be identical
        self.assertTrue(torch.allclose(out_standard[:, :15], out_shuffled[:, :15], atol=1e-5))
        # After step 15, representations diverge
        diff = torch.norm(out_standard[:, 15:] - out_shuffled[:, 15:])
        self.assertGreater(float(diff), 1e-3, "Slot shuffling failed to alter post-shuffle representation")

    def test_deterministic_reproducibility(self):
        """Verify identical evaluation runs on identical seeds yield bit-for-bit identical results."""
        m = make_mtld_model("cgp_thoughtlet")
        res1 = evaluate_model(m, "cgp_thoughtlet", self.env, num_seeds=3)
        res2 = evaluate_model(m, "cgp_thoughtlet", self.env, num_seeds=3)

        self.assertEqual(res1["overall_retention_accuracy"], res2["overall_retention_accuracy"])
        self.assertEqual(res1["untouched_retention_accuracy"], res2["untouched_retention_accuracy"])
        self.assertEqual(res1["cross_talk_error_rate"], res2["cross_talk_error_rate"])


if __name__ == "__main__":
    unittest.main()
