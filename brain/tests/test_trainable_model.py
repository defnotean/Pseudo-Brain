from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace
from typing import Any

from irene_brain.model.spec import ThoughtFieldConfig

try:
    import torch
except ModuleNotFoundError:  # Phase 0's stdlib-only environment remains supported.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.model.brain_cell import BrainCell
    from irene_brain.model.torch_model import IreneBrainModel


def tiny_config() -> ThoughtFieldConfig:
    """Return the smallest topology-faithful configuration used by CPU tests."""

    return replace(
        ThoughtFieldConfig.smoke(),
        core_width=16,
        sensor_tokens=4,
        belief_tokens=2,
        working_memory_tokens=1,
        thoughtlets=4,
        goal_context_tokens=1,
        cognitive_cycles=2,
        brain_cell_blocks=1,
        attention_heads=2,
        routed_neighbors=1,
        episodic_memory_entries=8,
        retrieved_entries_per_thoughtlet=1,
    )


@unittest.skipUnless(torch is not None, "PyTorch is not installed in the play-safe runtime")
class TrainableThoughtFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch permits this setting only before parallel work begins.
                pass

    def setUp(self) -> None:
        assert torch is not None
        torch.manual_seed(1729)
        self.config = tiny_config()
        self.model = IreneBrainModel(
            self.config,
            input_resolution=(8, 8),
            plan_steps=2,
        ).cpu()

    def inputs(self) -> tuple[Any, Any, Any, Any]:
        assert torch is not None
        pixels = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
        pixels = pixels / float(3 * 8 * 8)
        previous_control = torch.zeros(1, self.config.actuator.total_queries)
        elapsed_seconds = torch.tensor([1.0 / 30.0])
        thought_noise = torch.zeros(
            1,
            self.config.thoughtlets,
            self.config.core_width,
        )
        for slot in range(self.config.thoughtlets):
            thought_noise[0, slot, slot] = 1.0
        return pixels, previous_control, elapsed_seconds, thought_noise

    def forward(self, *, state: Any = None, max_cycles: int | None = None) -> Any:
        pixels, previous_control, elapsed_seconds, thought_noise = self.inputs()
        return self.model(
            pixels,
            previous_control,
            elapsed_seconds,
            state=state,
            max_cycles=max_cycles,
            thought_noise=thought_noise,
        )

    def test_forward_shapes_preserve_every_thought_and_actuator_query(self) -> None:
        assert torch is not None
        self.model.eval()
        with torch.no_grad():
            output = self.forward()

        batch = 1
        thoughts = self.config.thoughtlets
        registers = self.config.registers_per_thoughtlet
        width = self.config.core_width
        queries = self.config.actuator.total_queries
        complete_state_tokens = (
            self.config.sensor_tokens
            + self.config.belief_tokens
            + thoughts * registers
            + self.config.working_memory_tokens
            + thoughts * self.config.retrieved_entries_per_thoughtlet
            + self.config.goal_context_tokens
        )

        self.assertEqual(output.next_state.belief.shape, (batch, 2, width))
        self.assertEqual(output.next_state.working_memory.shape, (batch, 1, width))
        self.assertEqual(
            output.next_state.thoughts.shape,
            (batch, thoughts, registers, width),
        )
        self.assertEqual(output.next_state.goal_context.shape, (batch, 1, width))
        self.assertEqual(output.next_state.thought_age_seconds.shape, (batch, thoughts))

        self.assertEqual(output.action.control.shape, (batch, queries))
        self.assertEqual(output.action.button_logits.shape, (batch, 296))
        self.assertEqual(output.action.mouse_zero_logit.shape, (batch, 1))
        self.assertEqual(output.action.mouse_mean.shape, (batch, 2))
        self.assertEqual(output.action.scroll_mean.shape, (batch, 1))
        self.assertEqual(output.action.gamepad_axis_mean.shape, (batch, 8))
        self.assertEqual(output.action.control_plan.shape, (batch, 2, queries))
        self.assertEqual(output.action.query_features.shape, (batch, queries, width))
        self.assertEqual(
            output.action.state_attention.shape,
            (batch, queries, complete_state_tokens),
        )
        self.assertEqual(output.action.thought_attention.shape, (batch, queries, thoughts))

        self.assertEqual(len(output.anytime_actions), self.config.cognitive_cycles + 1)
        self.assertEqual(output.value.shape, (batch,))
        self.assertEqual(
            output.world.focus_logits.shape,
            (batch, thoughts, self.config.sensor_tokens + self.config.belief_tokens),
        )
        self.assertEqual(output.world.horizon_logits.shape, (batch, thoughts, 6))
        self.assertEqual(
            output.world.candidate_action_logits.shape,
            (batch, thoughts, queries),
        )
        self.assertEqual(output.world.future_embedding.shape, (batch, thoughts, width))
        self.assertEqual(output.world.lifecycle_logits.shape, (batch, thoughts, 3))

        diagnostics = output.diagnostics
        self.assertFalse(diagnostics.uses_pooled_integration_token)
        self.assertEqual(diagnostics.cycles_completed, self.config.cognitive_cycles)
        self.assertEqual(diagnostics.thought_summaries.shape, (batch, thoughts, width))
        self.assertEqual(
            diagnostics.thought_cosine_similarity.shape,
            (batch, thoughts, thoughts),
        )
        self.assertEqual(
            diagnostics.actuator_thought_attention.shape,
            (batch, queries, thoughts),
        )
        self.assertEqual(len(diagnostics.routing_indices), self.config.cognitive_cycles)
        self.assertEqual(diagnostics.routing_indices[0].shape, (batch, thoughts, 0))
        self.assertEqual(
            diagnostics.routing_indices[1].shape,
            (batch, thoughts, self.config.routed_neighbors),
        )

    def test_one_cell_and_parameter_set_are_shared_across_slots_and_cycles(self) -> None:
        assert torch is not None
        cells = [module for module in self.model.modules() if isinstance(module, BrainCell)]
        self.assertEqual(len(cells), 1)
        self.assertIs(cells[0], self.model.brain_cell)

        base_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        more_cycles = IreneBrainModel(
            replace(self.config, cognitive_cycles=3),
            input_resolution=(8, 8),
            plan_steps=2,
        )
        more_slots = IreneBrainModel(
            replace(self.config, thoughtlets=6),
            input_resolution=(8, 8),
            plan_steps=2,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in more_cycles.parameters()),
            base_parameters,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in more_slots.parameters()),
            base_parameters,
        )

        calls: list[int] = []
        handle = self.model.brain_cell.register_forward_hook(
            lambda module, _inputs, _output: calls.append(id(module))
        )
        try:
            self.model.eval()
            with torch.no_grad():
                self.forward()
        finally:
            handle.remove()
        self.assertEqual(calls, [id(self.model.brain_cell)] * self.config.cognitive_cycles)

    def test_state_persists_and_slot_diversity_is_observable(self) -> None:
        assert torch is not None
        self.model.eval()
        with torch.no_grad():
            first = self.forward()
            continued = self.forward(state=first.next_state)
            reset = self.forward()

        self.assertFalse(
            torch.allclose(continued.next_state.thoughts, reset.next_state.thoughts)
        )
        self.assertTrue(
            torch.allclose(
                first.next_state.thought_age_seconds,
                torch.full_like(first.next_state.thought_age_seconds, 1.0 / 30.0),
            )
        )

        summaries = first.diagnostics.thought_summaries
        pairwise_delta = summaries[:, 1:] - summaries[:, :1]
        self.assertGreater(float(pairwise_delta.abs().max()), 1e-6)
        similarity = first.diagnostics.thought_cosine_similarity
        self.assertTrue(
            torch.allclose(
                similarity.diagonal(dim1=-2, dim2=-1),
                torch.ones(1, self.config.thoughtlets),
                atol=1e-5,
            )
        )
        self.assertFalse(torch.allclose(similarity, torch.ones_like(similarity)))
        self.assertGreater(float(first.diagnostics.actuator_thought_attention.sum()), 0.0)

    def test_backward_and_one_cpu_optimizer_step_reach_the_shared_core(self) -> None:
        assert torch is not None
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        core_parameter = self.model.brain_cell.blocks[0].thought_attention.feed_forward[0].weight
        before = core_parameter.detach().clone()

        output = self.forward()
        losses = [output.value.square().mean()]
        losses.extend(
            tensor.square().mean()
            for field in dataclasses.fields(output.action)
            if (tensor := getattr(output.action, field.name)).is_floating_point()
        )
        losses.extend(
            tensor.square().mean()
            for field in dataclasses.fields(output.world)
            if (tensor := getattr(output.world, field.name)).is_floating_point()
        )
        loss = torch.stack(losses).sum()
        self.assertTrue(torch.isfinite(loss))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(core_parameter.grad)
        assert core_parameter.grad is not None
        self.assertTrue(torch.isfinite(core_parameter.grad).all())
        self.assertGreater(float(core_parameter.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in self.model.parameters())
        )
        optimizer.step()

        self.assertFalse(torch.equal(before, core_parameter.detach()))


if __name__ == "__main__":
    unittest.main()
