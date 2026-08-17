from __future__ import annotations

import unittest

from irene_brain.model import ActuatorQuerySpec, AttentionContract, ThoughtFieldConfig


class ThoughtFieldSpecificationTests(unittest.TestCase):
    def test_thesis_mvp_shapes_are_locked(self) -> None:
        config = ThoughtFieldConfig.thesis_mvp()
        layout = config.state_layout(batch_size=2)

        self.assertEqual(layout.belief.dimensions, (2, 48, 384))
        self.assertEqual(layout.working_memory.dimensions, (2, 16, 384))
        self.assertEqual(layout.thought_field.dimensions, (2, 32, 3, 384))
        self.assertEqual(layout.goal_context.dimensions, (2, 8, 384))
        self.assertEqual(layout.retrieved_memory.dimensions, (2, 32, 2, 384))
        self.assertEqual(layout.actuator_queries.dimensions, (2, 307, 384))

    def test_persistent_state_matches_the_documented_small_memory_envelope(self) -> None:
        config = ThoughtFieldConfig.thesis_mvp()
        self.assertEqual(config.persistent_state_bytes(batch_size=1), 129_024)
        self.assertEqual(config.tied_core_parameter_estimate, 7_077_888)

    def test_actuators_read_every_rich_state_without_pooling(self) -> None:
        contract = ThoughtFieldConfig.thesis_mvp().attention_contract
        actuator_reads = set(contract.reads["actuator_query"])

        self.assertFalse(contract.uses_pooled_integration_token)
        self.assertTrue(contract.weights_shared_across_thoughtlets)
        self.assertTrue(contract.weights_shared_across_cycles)
        self.assertEqual(contract.cycle_one_cross_thought_neighbors, 0)
        self.assertEqual(contract.later_cross_thought_neighbors, 2)
        self.assertTrue(
            {
                "sensor",
                "belief",
                "all_thoughtlet_registers",
                "working_memory",
                "retrieved_memory",
                "goal_context",
            }.issubset(actuator_reads)
        )

    def test_invalid_nonshared_or_pooled_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AttentionContract(
                reads={"actuator_query": ("thoughts",)},
                cycle_one_cross_thought_neighbors=0,
                later_cross_thought_neighbors=2,
                uses_pooled_integration_token=True,
                weights_shared_across_thoughtlets=True,
                weights_shared_across_cycles=True,
            )

    def test_width_and_routing_invariants_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            ThoughtFieldConfig(core_width=385)
        with self.assertRaises(ValueError):
            ThoughtFieldConfig(thoughtlets=2, routed_neighbors=2)

    def test_actuator_dimensions_are_fixed_to_the_wire_contract(self) -> None:
        actuator = ActuatorQuerySpec()
        self.assertEqual(
            (
                actuator.keyboard_keys,
                actuator.mouse_buttons,
                actuator.gamepad_buttons,
                actuator.gamepad_axes,
            ),
            (256, 8, 32, 8),
        )
        with self.assertRaises(ValueError):
            ActuatorQuerySpec(gamepad_axes=7)
        with self.assertRaises(ValueError):
            ActuatorQuerySpec(mouse_buttons=True)  # type: ignore[arg-type]

    def test_boolean_and_integer_fields_do_not_accept_python_coercions(self) -> None:
        with self.assertRaises(ValueError):
            ThoughtFieldConfig(schema_version=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AttentionContract(
                reads={"actuator_query": ("thoughts",)},
                cycle_one_cross_thought_neighbors=False,  # type: ignore[arg-type]
                later_cross_thought_neighbors=2,
                uses_pooled_integration_token=False,
                weights_shared_across_thoughtlets=True,
                weights_shared_across_cycles=True,
            )
        with self.assertRaises(ValueError):
            AttentionContract(
                reads={"actuator_query": ("thoughts",)},
                cycle_one_cross_thought_neighbors=0,
                later_cross_thought_neighbors=2,
                uses_pooled_integration_token=0,  # type: ignore[arg-type]
                weights_shared_across_thoughtlets=True,
                weights_shared_across_cycles=True,
            )


if __name__ == "__main__":
    unittest.main()
