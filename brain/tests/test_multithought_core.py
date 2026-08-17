from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.training.config import (
    OBJECTIVE_OPTIONAL_DEFAULTS,
    ObjectiveConfig,
    load_training_config,
)


BRAIN_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError:  # The stdlib-only play-safe runtime remains valid.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.model.brain_cell import StructuredBrainBlock
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.training.objective import _thought_diagnostic_metric_counts


def tiny_multiblock_config() -> ThoughtFieldConfig:
    return replace(
        ThoughtFieldConfig.smoke(),
        core_width=16,
        sensor_tokens=4,
        belief_tokens=2,
        working_memory_tokens=2,
        thoughtlets=4,
        goal_context_tokens=1,
        cognitive_cycles=2,
        brain_cell_blocks=2,
        attention_heads=2,
        routed_neighbors=1,
        episodic_memory_entries=8,
        retrieved_entries_per_thoughtlet=1,
    )


class ObjectiveConfigurationTests(unittest.TestCase):
    def test_objective_table_is_strict_hashed_and_gate_is_bounded(self) -> None:
        config_dir = BRAIN_ROOT / "configs" / "training"
        configs = {
            path.stem: load_training_config(path)
            for path in sorted(config_dir.glob("*.toml"))
        }
        self.assertGreaterEqual(len(configs), 4)
        overfit = configs["dgx-stagea-action-overfit"]
        overfit_b = configs["dgx-stagea-action-overfit-b"]
        for name, config in configs.items():
            if name in {
                "dgx-stagea-action-overfit",
                "dgx-stagea-action-overfit-b",
                # The unregistered v3 review candidate intentionally carries
                # non-default recipe fields; test_rcq_v3_recipe pins them.
                "dgx-rcq-v3-reference-candidate",
            }:
                continue
            self.assertEqual(
                config.objective,
                ObjectiveConfig(
                    action_weight=1.0,
                    value_weight=0.1,
                    world_weight=0.1,
                    diversity_weight=0.05,
                ),
            )
            expected_objective = {
                key: value
                for key, value in asdict(config.objective).items()
                if OBJECTIVE_OPTIONAL_DEFAULTS.get(key, key) != value
            }
            self.assertEqual(
                config.to_dict()["objective"],
                {
                    **expected_objective,
                    "button_support_control_indices": [4, 7, 22, 26],
                },
            )

        self.assertEqual(
            overfit.objective,
            ObjectiveConfig(
                action_loss_kind="sparse_hard_negative_v1",
                action_weight=1.0,
                value_weight=0.0,
                world_weight=0.0,
                diversity_weight=0.0,
            ),
        )
        self.assertEqual(overfit.run.max_optimizer_steps, 300)
        self.assertEqual(overfit.dataset.train_sequences, 8)
        self.assertEqual(overfit.optimization.gradient_accumulation_steps, 8)
        self.assertEqual(overfit.optimization.warmup_steps, 10)
        self.assertEqual(overfit.optimization.weight_decay, 0.0)
        self.assertEqual(
            overfit_b.objective,
            ObjectiveConfig(
                action_loss_kind="support_aware_calibrated_v1",
                action_weight=1.0,
                value_weight=0.0,
                world_weight=0.0,
                diversity_weight=0.0,
            ),
        )
        self.assertEqual(overfit_b.run.max_optimizer_steps, 300)
        self.assertEqual(overfit_b.dataset, overfit.dataset)
        self.assertEqual(overfit_b.optimization, overfit.optimization)
        self.assertEqual(overfit_b.logging, overfit.logging)
        self.assertEqual(overfit_b.resources, overfit.resources)
        self.assertNotEqual(overfit_b.config_sha256, overfit.config_sha256)
        self.assertEqual(
            overfit.to_dict()["objective"],
            {
                **{
                    key: value
                    for key, value in asdict(overfit.objective).items()
                    if OBJECTIVE_OPTIONAL_DEFAULTS.get(key, key) != value
                },
                "button_support_control_indices": [4, 7, 22, 26],
            },
        )

        bootstrap = configs["phase1-bootstrap"]
        gate = configs["dgx-stagea-gate"]
        self.assertEqual(gate.run.max_optimizer_steps, 100)
        self.assertEqual(gate.dataset, bootstrap.dataset)
        self.assertEqual(gate.optimization, bootstrap.optimization)

        changed = replace(
            bootstrap,
            objective=replace(bootstrap.objective, diversity_weight=0.0),
        )
        self.assertNotEqual(changed.config_sha256, bootstrap.config_sha256)

        source = (config_dir / "dgx-smoke.toml").read_text(encoding="utf-8")
        objective_table = """[objective]
action_loss_kind = "support_aware_calibrated_v1"
button_support_control_indices = [4, 7, 22, 26]
button_support_weight = 0.8
button_background_weight = 0.2
button_background_tail_mix = 0.9
button_background_tail_temperature = 0.1
continuous_action_weight = 0.25
action_weight = 1.0
value_weight = 0.1
world_weight = 0.1
diversity_weight = 0.05

"""
        self.assertIn(objective_table, source)
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-objective.toml"
            missing.write_text(
                source.replace(objective_table, ""),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "objective"):
                load_training_config(missing)

            unknown = Path(temporary) / "unknown-objective.toml"
            unknown.write_text(
                source.replace(
                    "diversity_weight = 0.05",
                    "diversity_weight = 0.05\nfuture_divergence_weight = 1.0",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                load_training_config(unknown)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in play-safe runtime")
class MultiThoughtCoreTests(unittest.TestCase):
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
        torch.manual_seed(23063)
        self.config = tiny_multiblock_config()
        self.model = IreneBrainModel(
            self.config,
            input_resolution=(8, 8),
            plan_steps=2,
        ).cpu()
        self.pixels = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)
        self.previous_control = torch.zeros(1, self.config.actuator.total_queries)
        self.elapsed = torch.tensor([1.0 / 60.0])

    def test_default_eval_reuses_distinct_identities_without_rng_consumption(self) -> None:
        assert torch is not None
        self.model.eval()
        seen_codes: list[object] = []
        handle = self.model.noise_projection.register_forward_pre_hook(
            lambda _module, inputs: seen_codes.append(inputs[0].detach().clone())
        )
        rng_before = torch.get_rng_state().clone()
        try:
            with torch.no_grad():
                first = self.model(
                    self.pixels,
                    self.previous_control,
                    self.elapsed,
                )
                repeat = self.model(
                    self.pixels,
                    self.previous_control,
                    self.elapsed,
                )
                continued_a = self.model(
                    self.pixels,
                    self.previous_control,
                    self.elapsed,
                    first.next_state,
                )
                continued_b = self.model(
                    self.pixels,
                    self.previous_control,
                    self.elapsed,
                    first.next_state,
                )
        finally:
            handle.remove()
        rng_after = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertTrue(
            torch.equal(first.next_state.thoughts, repeat.next_state.thoughts)
        )
        self.assertTrue(
            torch.equal(continued_a.next_state.thoughts, continued_b.next_state.thoughts)
        )
        self.assertEqual(len(seen_codes), 6)
        self.assertTrue(all(torch.equal(seen_codes[0], code) for code in seen_codes[1:]))
        for output in (first, continued_a):
            summaries = output.diagnostics.thought_summaries
            self.assertGreater(
                float((summaries[:, 1:] - summaries[:, :1]).abs().max()),
                1e-6,
            )

        named_parameters = dict(self.model.named_parameters())
        self.assertNotIn("_thought_identity_codes", named_parameters)
        override = torch.zeros(
            1,
            self.config.thoughtlets,
            self.config.core_width,
        )
        with torch.no_grad():
            overridden = self.model(
                self.pixels,
                self.previous_control,
                self.elapsed,
                thought_noise=override,
            )
        self.assertFalse(
            torch.equal(overridden.next_state.thoughts, first.next_state.thoughts)
        )

    def test_selected_utility_values_receive_a_finite_nonzero_gradient(self) -> None:
        assert torch is not None
        block = StructuredBrainBlock(width=16, heads=2, routed_neighbors=1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(9182)

        def sample(*shape: int) -> object:
            return torch.randn(*shape, generator=generator)

        inputs = {
            "belief": sample(2, 2, 16),
            "working_memory": sample(2, 2, 16),
            "thoughts": sample(2, 4, 3, 16),
            "sensors": sample(2, 4, 16),
            "action_time_tokens": sample(2, 2, 16),
            "goal_context": sample(2, 1, 16),
            "retrieved_memory": sample(2, 4, 1, 16),
            "elapsed_seconds": torch.full((2, 1), 0.05),
            "allow_routing": True,
        }
        _belief, working_memory, _thoughts, _routing = block(**inputs)
        loss = working_memory.square().mean()
        loss.backward()
        gradient = block.utility.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 1e-6)

        with torch.no_grad():
            baseline_memory = block(**inputs)[1]
            block.utility.weight.mul_(-4.0)
            perturbed_memory = block(**inputs)[1]
        self.assertGreater(
            float((baseline_memory - perturbed_memory).abs().max()),
            1e-6,
        )

    def test_all_block_routing_and_applied_refresh_gate_are_exposed(self) -> None:
        assert torch is not None
        self.model.eval()
        initial_state = self.model.initial_state(1)
        with torch.no_grad():
            expected_expire = torch.softmax(
                self.model.thought_predictions.lifecycle(
                    initial_state.thoughts.mean(dim=2)
                ),
                dim=-1,
            )[..., 2]
            output = self.model(
                self.pixels,
                self.previous_control,
                self.elapsed,
                initial_state,
            )

        expected_diagnostics = (
            self.config.cognitive_cycles * self.config.brain_cell_blocks
        )
        self.assertEqual(len(output.diagnostics.routing_indices), expected_diagnostics)
        self.assertEqual(len(output.diagnostics.routing_weights), expected_diagnostics)
        for index, (indices, weights) in enumerate(
            zip(
                output.diagnostics.routing_indices,
                output.diagnostics.routing_weights,
                strict=True,
            )
        ):
            routed = 0 if index < self.config.brain_cell_blocks else 1
            self.assertEqual(indices.shape, (1, self.config.thoughtlets, routed))
            self.assertEqual(weights.shape, (1, self.config.thoughtlets, routed))
        self.assertEqual(
            output.diagnostics.applied_expire_probability.shape,
            (1, self.config.thoughtlets),
        )
        self.assertTrue(
            torch.allclose(
                output.diagnostics.applied_expire_probability,
                expected_expire,
            )
        )

    def test_diagnostic_metric_counts_are_finite_and_naturally_ranged(self) -> None:
        assert torch is not None
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                self.pixels,
                self.previous_control,
                self.elapsed,
            )
            prediction_error = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
            movement_indices = torch.tensor([26, 4, 22, 7], dtype=torch.long)
            metrics = _thought_diagnostic_metric_counts(
                output,
                prediction_error,
                movement_indices,
            )

        expected = {
            "thought_summary_rank_proxy",
            "thought_full_register_rank_proxy",
            "thought_slot_private_energy",
            "movement_query_effective_slot_count",
            "mean_applied_expire_probability",
            "world_best_second_error_gap",
        }
        self.assertEqual(set(metrics), expected)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        unbounded_nonnegative = {
            "movement_query_effective_slot_count",
            "world_best_second_error_gap",
        }
        for name in expected - unbounded_nonnegative:
            self.assertGreaterEqual(float(metrics[name]), 0.0)
            self.assertLessEqual(float(metrics[name]), 1.0)
        effective_slots = float(metrics["movement_query_effective_slot_count"])
        self.assertGreaterEqual(effective_slots, 1.0)
        self.assertLessEqual(effective_slots, float(self.config.thoughtlets))
        self.assertAlmostEqual(float(metrics["world_best_second_error_gap"]), 0.1)


if __name__ == "__main__":
    unittest.main()
