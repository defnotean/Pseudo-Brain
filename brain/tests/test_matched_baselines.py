from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - play-safe no-ML environment
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if torch is not None:
    from irene_brain.model.baselines import (
        DENSE_COMMUNICATION_IDENTITY,
        MONOLITHIC_IDENTITY,
        NO_COMMUNICATION_IDENTITY,
        PARAMETER_MATCHED_MONOLITHIC_IDENTITY,
        REACTIVE_IDENTITY,
        REFERENCE_IDENTITY,
        RESET_STATE_IDENTITY,
        SERIAL_DEPTH_IDENTITY,
        DenseCommunicationSlotBaseline,
        MonolithicRecurrentBaseline,
        NoCommunicationSlotBaseline,
        ParameterMatchedMonolithicBaseline,
        ReactiveSlotBaseline,
        ResetStateSlotBaseline,
        SerialDepthSlotBaseline,
        allocated_parameter_counts,
        build_architecture_manifest,
    )
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.training.batches import MovingShapesBatchSource
    from irene_brain.training.config import load_training_config
    from irene_brain.training.objective import ThoughtFieldObjective


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class MatchedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    @staticmethod
    def slot_config() -> ThoughtFieldConfig:
        return replace(
            ThoughtFieldConfig.smoke(),
            core_width=16,
            sensor_tokens=4,
            belief_tokens=2,
            working_memory_tokens=1,
            thoughtlets=4,
            registers_per_thoughtlet=3,
            goal_context_tokens=1,
            cognitive_cycles=2,
            brain_cell_blocks=1,
            attention_heads=2,
            routed_neighbors=1,
            episodic_memory_entries=8,
            retrieved_entries_per_thoughtlet=1,
        )

    @classmethod
    def monolithic_config(cls, *, width: int = 16) -> ThoughtFieldConfig:
        return replace(
            cls.slot_config(),
            core_width=width,
            thoughtlets=1,
            registers_per_thoughtlet=1,
            routed_neighbors=0,
        )

    def inputs(self, *, width: int = 16) -> tuple[object, object, object]:
        del width
        assert torch is not None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(773)
        return (
            torch.rand(1, 3, 8, 8, generator=generator),
            torch.zeros(1, 307),
            torch.tensor([0.05]),
        )

    def test_variant_identities_are_canonical_and_unique(self) -> None:
        identities = (
            REFERENCE_IDENTITY,
            NO_COMMUNICATION_IDENTITY,
            MONOLITHIC_IDENTITY,
            PARAMETER_MATCHED_MONOLITHIC_IDENTITY,
            RESET_STATE_IDENTITY,
            DENSE_COMMUNICATION_IDENTITY,
            REACTIVE_IDENTITY,
            SERIAL_DEPTH_IDENTITY,
        )
        self.assertEqual(len({item.variant_id for item in identities}), len(identities))
        self.assertEqual(len({item.sha256 for item in identities}), len(identities))
        for identity in identities:
            self.assertEqual(json.loads(identity.canonical_json)["variant_id"], identity.variant_id)
            self.assertEqual(len(identity.sha256), 64)
        self.assertEqual(self.monolithic_config().routed_neighbors, 0)
        with self.assertRaisesRegex(ValueError, "one-latent controls"):
            replace(self.slot_config(), routed_neighbors=0)

    def test_isolated_slots_match_allocated_parameters_and_do_not_cross_talk(self) -> None:
        assert torch is not None
        torch.manual_seed(11)
        reference = IreneBrainModel(self.slot_config(), input_resolution=(8, 8))
        torch.manual_seed(11)
        isolated = NoCommunicationSlotBaseline(
            self.slot_config(),
            input_resolution=(8, 8),
        )
        reference_counts = allocated_parameter_counts(reference)
        isolated_counts = allocated_parameter_counts(isolated)
        self.assertEqual(reference_counts["total"], isolated_counts["total"])
        self.assertEqual(reference_counts["trainable"], isolated_counts["trainable"])
        self.assertEqual(reference_counts["architecturally_disconnected_trainable"], 0)
        self.assertGreater(
            isolated_counts["architecturally_disconnected_trainable"],
            0,
        )
        self.assertEqual(set(reference.state_dict()), set(isolated.state_dict()))
        self.assertTrue(
            all(
                torch.equal(reference.state_dict()[name], isolated.state_dict()[name])
                for name in reference.state_dict()
            )
        )
        self.assertEqual(
            reference.config.persistent_state_bytes(batch_size=1),
            isolated.config.persistent_state_bytes(batch_size=1),
        )

        pixels, control, elapsed = self.inputs()
        state = isolated.initial_state(1)
        changed_thoughts = state.thoughts.clone()
        changed_thoughts[:, 0].add_(3.0)
        changed_state = replace(state, thoughts=changed_thoughts)
        isolated.eval()
        with torch.no_grad():
            first = isolated(pixels, control, elapsed, state=state, max_cycles=2)
            changed = isolated(
                pixels,
                control,
                elapsed,
                state=changed_state,
                max_cycles=2,
            )
        self.assertTrue(
            torch.equal(
                first.next_state.thoughts[:, 1:],
                changed.next_state.thoughts[:, 1:],
            )
        )
        self.assertTrue(
            torch.equal(first.next_state.working_memory, changed.next_state.working_memory)
        )
        self.assertTrue(
            all(indices.shape[-1] == 0 for indices in first.diagnostics.routing_indices)
        )

    def test_reset_slots_ignore_incoming_thought_state(self) -> None:
        assert torch is not None
        torch.manual_seed(11)
        reference = IreneBrainModel(self.slot_config(), input_resolution=(8, 8))
        torch.manual_seed(11)
        reset = ResetStateSlotBaseline(self.slot_config(), input_resolution=(8, 8))
        reference_counts = allocated_parameter_counts(reference)
        reset_counts = allocated_parameter_counts(reset)
        self.assertEqual(reference_counts["total"], reset_counts["total"])
        self.assertEqual(reference_counts["trainable"], reset_counts["trainable"])
        self.assertEqual(reset_counts["architecturally_disconnected_trainable"], 16 * 3 + 3)
        self.assertEqual(set(reference.state_dict()), set(reset.state_dict()))
        self.assertTrue(
            all(
                torch.equal(reference.state_dict()[name], reset.state_dict()[name])
                for name in reference.state_dict()
            )
        )

        pixels, control, elapsed = self.inputs()
        state = reset.initial_state(1)
        changed_thoughts = state.thoughts.clone()
        changed_thoughts[:, 0].add_(3.0)
        changed_ages = state.thought_age_seconds.clone()
        changed_ages[:, 1].add_(12.0)
        changed_state = replace(
            state,
            thoughts=changed_thoughts,
            thought_age_seconds=changed_ages,
        )
        reset.eval()
        with torch.no_grad():
            first = reset(pixels, control, elapsed, state=state, max_cycles=2)
            changed = reset(pixels, control, elapsed, state=changed_state, max_cycles=2)
        # Persistence is gone: the incoming thought state cannot influence
        # anything the model computes this step.
        self.assertTrue(torch.equal(first.next_state.thoughts, changed.next_state.thoughts))
        self.assertTrue(
            torch.equal(first.next_state.working_memory, changed.next_state.working_memory)
        )
        self.assertTrue(torch.equal(first.action.control, changed.action.control))
        self.assertTrue(
            torch.equal(
                first.next_state.thought_age_seconds,
                changed.next_state.thought_age_seconds,
            )
        )
        self.assertTrue(
            torch.equal(
                first.diagnostics.applied_expire_probability,
                torch.ones_like(first.diagnostics.applied_expire_probability),
            )
        )

    def test_dense_routing_reaches_every_other_slot(self) -> None:
        assert torch is not None
        torch.manual_seed(11)
        reference = IreneBrainModel(self.slot_config(), input_resolution=(8, 8))
        torch.manual_seed(11)
        dense = DenseCommunicationSlotBaseline(
            self.slot_config(),
            input_resolution=(8, 8),
        )
        reference_counts = allocated_parameter_counts(reference)
        dense_counts = allocated_parameter_counts(dense)
        self.assertEqual(reference_counts["total"], dense_counts["total"])
        self.assertEqual(reference_counts["trainable"], dense_counts["trainable"])
        self.assertEqual(dense_counts["architecturally_disconnected_trainable"], 0)
        self.assertEqual(set(reference.state_dict()), set(dense.state_dict()))
        self.assertTrue(
            all(
                torch.equal(reference.state_dict()[name], dense.state_dict()[name])
                for name in reference.state_dict()
            )
        )

        pixels, control, elapsed = self.inputs()
        dense.eval()
        with torch.no_grad():
            output = dense(pixels, control, elapsed, max_cycles=2)
        thoughtlets = self.slot_config().thoughtlets
        # Cycle 0 is private; later cycles route densely to all K-1 peers.
        cycle_one_indices = output.diagnostics.routing_indices[-1]
        cycle_one_weights = output.diagnostics.routing_weights[-1]
        self.assertEqual(cycle_one_indices.shape, (1, thoughtlets, thoughtlets - 1))
        self.assertEqual(cycle_one_weights.shape, (1, thoughtlets, thoughtlets - 1))
        for slot in range(thoughtlets):
            self.assertEqual(
                set(cycle_one_indices[0, slot].tolist()),
                {peer for peer in range(thoughtlets) if peer != slot},
            )
        weight_sums = cycle_one_weights.sum(dim=-1)
        self.assertTrue(
            torch.allclose(weight_sums, torch.ones_like(weight_sums))
        )
        self.assertTrue(torch.isfinite(output.action.control).all())

    def test_reactive_control_ignores_every_incoming_state_field(self) -> None:
        assert torch is not None
        torch.manual_seed(11)
        reference = IreneBrainModel(self.slot_config(), input_resolution=(8, 8))
        torch.manual_seed(11)
        reactive = ReactiveSlotBaseline(self.slot_config(), input_resolution=(8, 8))
        reference_counts = allocated_parameter_counts(reference)
        reactive_counts = allocated_parameter_counts(reactive)
        self.assertEqual(reference_counts["total"], reactive_counts["total"])
        self.assertEqual(reference_counts["trainable"], reactive_counts["trainable"])
        self.assertEqual(reactive_counts["architecturally_disconnected_trainable"], 0)
        self.assertEqual(set(reference.state_dict()), set(reactive.state_dict()))

        pixels, control, elapsed = self.inputs()
        state = reactive.initial_state(1)
        changed_state = replace(
            state,
            belief=state.belief + 5.0,
            working_memory=state.working_memory - 2.0,
            thoughts=state.thoughts + 3.0,
            thought_age_seconds=state.thought_age_seconds + 7.0,
        )
        reactive.eval()
        with torch.no_grad():
            fresh = reactive(pixels, control, elapsed, state=None, max_cycles=2)
            poisoned = reactive(pixels, control, elapsed, state=changed_state, max_cycles=2)
        self.assertTrue(torch.equal(fresh.action.control, poisoned.action.control))
        self.assertTrue(
            torch.equal(fresh.next_state.thoughts, poisoned.next_state.thoughts)
        )
        self.assertTrue(
            torch.equal(fresh.next_state.belief, poisoned.next_state.belief)
        )
        self.assertTrue(
            torch.equal(
                fresh.next_state.working_memory, poisoned.next_state.working_memory
            )
        )
        self.assertTrue(torch.isfinite(fresh.action.control).all())

    def test_serial_depth_matches_block_applications_without_tying(self) -> None:
        assert torch is not None
        reference_config = self.slot_config()
        serial_config = replace(
            reference_config,
            cognitive_cycles=1,
            brain_cell_blocks=(
                reference_config.cognitive_cycles * reference_config.brain_cell_blocks
            ),
        )
        torch.manual_seed(11)
        reference = IreneBrainModel(reference_config, input_resolution=(8, 8))
        torch.manual_seed(11)
        serial = SerialDepthSlotBaseline(serial_config, input_resolution=(8, 8))
        reference_counts = allocated_parameter_counts(reference)
        serial_counts = allocated_parameter_counts(serial)
        # Untied depth: one extra block per removed cycle, everything else equal.
        reference_block_parameters = sum(
            parameter.numel()
            for name, parameter in reference.named_parameters()
            if name.startswith("brain_cell.blocks.")
        )
        self.assertEqual(
            serial_counts["trainable"] - reference_counts["trainable"],
            reference_block_parameters * (reference_config.cognitive_cycles - 1),
        )
        self.assertEqual(serial_counts["architecturally_disconnected_trainable"], 0)

        pixels, control, elapsed = self.inputs()
        serial.eval()
        with torch.no_grad():
            output = serial(pixels, control, elapsed)
        self.assertEqual(len(output.anytime_actions), 2)
        self.assertEqual(output.diagnostics.cycles_completed, 1)
        # Serial depth replaces recurrence, so even the first (and only) cycle
        # routes sparsely to the configured number of neighbors.
        thoughtlets = reference_config.thoughtlets
        for indices in output.diagnostics.routing_indices:
            self.assertEqual(
                indices.shape,
                (1, thoughtlets, reference_config.routed_neighbors),
            )
        self.assertTrue(torch.isfinite(output.action.control).all())

    def test_monolithic_control_runs_with_one_finite_recurrent_latent(self) -> None:
        assert torch is not None
        model = MonolithicRecurrentBaseline(
            self.monolithic_config(),
            input_resolution=(8, 8),
        )
        pixels, control, elapsed = self.inputs()
        output = model(pixels, control, elapsed)
        self.assertEqual(output.next_state.thoughts.shape, (1, 1, 1, 16))
        self.assertEqual(output.diagnostics.thought_cosine_similarity.shape, (1, 1, 1))
        self.assertEqual(output.action.control.shape, (1, 307))
        self.assertTrue(torch.isfinite(output.action.control).all())
        self.assertTrue(
            all(indices.shape == (1, 1, 0) for indices in output.diagnostics.routing_indices)
        )

    def test_monolithic_objective_treats_empty_slot_pairs_as_zero_diversity(self) -> None:
        assert torch is not None
        training = load_training_config(ROOT / "configs" / "training" / "dgx-smoke.toml")
        source = MovingShapesBatchSource(training.dataset)
        batch = next(
            source.iter_batches(
                split="train",
                epoch=0,
                start_batch=0,
                batch_size=1,
                max_batches=1,
            )
        )
        model = MonolithicRecurrentBaseline(
            self.monolithic_config(),
            input_resolution=(8, 8),
        )
        result = ThoughtFieldObjective(model)(batch)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(float(result.metrics["diversity_loss"]), 0.0)

    def test_manifest_separates_parameter_and_uncalibrated_latency_regimes(self) -> None:
        reference = IreneBrainModel(self.slot_config(), input_resolution=(8, 8))
        isolated = NoCommunicationSlotBaseline(
            self.slot_config(), input_resolution=(8, 8)
        )
        monolithic = MonolithicRecurrentBaseline(
            self.monolithic_config(), input_resolution=(8, 8)
        )
        parameter_control = ParameterMatchedMonolithicBaseline(
            self.monolithic_config(width=18), input_resolution=(8, 8)
        )
        models = {
            REFERENCE_IDENTITY.variant_id: (reference, "tests:reference"),
            NO_COMMUNICATION_IDENTITY.variant_id: (isolated, "tests:isolated"),
            MONOLITHIC_IDENTITY.variant_id: (monolithic, "tests:monolithic"),
            PARAMETER_MATCHED_MONOLITHIC_IDENTITY.variant_id: (
                parameter_control,
                "tests:parameter_control",
            ),
        }
        manifest = build_architecture_manifest(
            models,
            parameter_tolerance_fraction=0.5,
        )
        regimes = manifest["fairness_regimes"]
        self.assertTrue(regimes["parameter_matched"]["verified"])
        self.assertFalse(regimes["measured_latency_or_flop_matched"]["verified"])
        self.assertEqual(
            regimes["measured_latency_or_flop_matched"]["status"],
            "hardware_calibration_required_before_training",
        )
        self.assertIn("manifest_sha256", manifest)
        self.assertIn("does not establish independent thoughts", manifest["claim_boundary"])

    def test_checked_in_manifest_and_recipes_are_strictly_identified(self) -> None:
        manifest_path = ROOT / "configs" / "baseline-architecture-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_digest = manifest.pop("manifest_sha256")
        canonical = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(sha256(canonical.encode("utf-8")).hexdigest(), recorded_digest)

        implementation = manifest["implementation"]
        source_files = implementation["source_files"]
        for relative_path, expected_digest in source_files.items():
            self.assertEqual(
                sha256((ROOT / relative_path).read_bytes()).hexdigest(),
                expected_digest,
            )
        canonical_sources = json.dumps(
            source_files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            sha256(canonical_sources.encode("utf-8")).hexdigest(),
            implementation["source_bundle_sha256"],
        )

        identities = {
            identity.variant_id: identity
            for identity in (
                REFERENCE_IDENTITY,
                NO_COMMUNICATION_IDENTITY,
                MONOLITHIC_IDENTITY,
                PARAMETER_MATCHED_MONOLITHIC_IDENTITY,
                RESET_STATE_IDENTITY,
                DENSE_COMMUNICATION_IDENTITY,
                REACTIVE_IDENTITY,
                SERIAL_DEPTH_IDENTITY,
            )
        }
        entries = {entry["variant_id"]: entry for entry in manifest["variants"]}
        self.assertEqual(set(entries), set(identities))
        reference_config = load_training_config(
            ROOT / "configs" / "training" / "dgx-stagea-continuation-gate-b.toml"
        )

        def normalized(config: object) -> dict[str, object]:
            payload = config.to_dict()
            payload["run"]["name"] = "<variant>"
            payload["run"]["model_factory"] = "<variant>"
            return payload

        for variant_id, identity in identities.items():
            entry = entries[variant_id]
            self.assertEqual(entry["variant_identity_sha256"], identity.sha256)
            recipe = entry["training_recipe"]
            recipe_path = ROOT / recipe["path"]
            raw = recipe_path.read_bytes()
            config = load_training_config(recipe_path)
            self.assertEqual(sha256(raw).hexdigest(), recipe["raw_sha256"])
            self.assertEqual(config.config_sha256, recipe["canonical_config_sha256"])
            self.assertEqual(config.run.model_factory, entry["model_factory"])
            self.assertEqual(normalized(config), normalized(reference_config))

        self.assertEqual(entries[REFERENCE_IDENTITY.variant_id]["allocated_parameters"]["trainable"], 29_674_318)
        self.assertEqual(entries[NO_COMMUNICATION_IDENTITY.variant_id]["allocated_parameters"]["trainable"], 29_674_318)
        self.assertEqual(entries[PARAMETER_MATCHED_MONOLITHIC_IDENTITY.variant_id]["allocated_parameters"]["trainable"], 29_643_900)
        self.assertEqual(entries[RESET_STATE_IDENTITY.variant_id]["allocated_parameters"]["trainable"], 29_674_318)
        self.assertEqual(
            entries[RESET_STATE_IDENTITY.variant_id]["allocated_parameters"]["architecturally_disconnected_trainable"],
            384 * 3 + 3,
        )
        self.assertEqual(entries[DENSE_COMMUNICATION_IDENTITY.variant_id]["allocated_parameters"]["trainable"], 29_674_318)
        self.assertEqual(
            entries[DENSE_COMMUNICATION_IDENTITY.variant_id]["allocated_parameters"]["architecturally_disconnected_trainable"],
            0,
        )
        self.assertEqual(entries[REACTIVE_IDENTITY.variant_id]["allocated_parameters"]["trainable"], 29_674_318)
        self.assertEqual(entries[SERIAL_DEPTH_IDENTITY.variant_id]["allocated_parameters"]["trainable"], 75_849_558)


if __name__ == "__main__":
    unittest.main()
