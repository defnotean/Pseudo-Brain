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
        PARAMETER_MATCHED_MONOLITHIC_IDENTITY,
        REFERENCE_IDENTITY,
        allocated_parameter_counts,
    )
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.world_model_actor import (
        ACTION_EMBED_WIDTH,
        CONTROL_VECTOR_WIDTH,
        WORLD_MODEL_ACTOR_IDENTITY,
        LatentWorldModelActor,
    )
    from irene_brain.training.batches import MovingShapesBatchSource
    from irene_brain.training.config import DatasetConfig, load_training_config
    from irene_brain.training.factory import (
        build_thesis_model,
        build_thesis_parameter_matched_monolithic_model,
        build_thesis_world_model_actor_model,
    )
    from irene_brain.training.world_model_objective import LatentRolloutObjective


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class WorldModelActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    @staticmethod
    def _small_actor() -> "LatentWorldModelActor":
        assert torch is not None
        config = replace(
            ThoughtFieldConfig.smoke(),
            core_width=18,
            sensor_tokens=4,
            belief_tokens=2,
            working_memory_tokens=1,
            thoughtlets=1,
            registers_per_thoughtlet=1,
            goal_context_tokens=1,
            cognitive_cycles=2,
            brain_cell_blocks=1,
            attention_heads=2,
            routed_neighbors=0,
            episodic_memory_entries=8,
            retrieved_entries_per_thoughtlet=1,
        )
        return LatentWorldModelActor(config, input_resolution=(8, 8), plan_steps=2)

    @staticmethod
    def _length_eight_batch() -> object:
        assert torch is not None
        source = MovingShapesBatchSource(
            DatasetConfig(
                kind="moving_shapes",
                train_sequences=2,
                validation_sequences=1,
                test_sequences=1,
                sequence_length=8,
                burn_in_steps=2,
                seed_offset=0,
                hazard_count=3,
                tick_period_ns=16_666_667,
                discount=0.99,
            )
        )
        return next(
            source.iter_batches(
                split="train", epoch=0, start_batch=0, batch_size=1, max_batches=1
            )
        )

    def test_identity_is_strictly_identified(self) -> None:
        assert torch is not None
        model = self._small_actor()
        self.assertEqual(
            model.architecture_variant_id, "irene.world_model_actor.gru_latent.v1"
        )
        self.assertEqual(model.architecture_identity, WORLD_MODEL_ACTOR_IDENTITY)
        self.assertEqual(
            model.architecture_identity.matching_role,
            "own_recipe_family_world_model_actor_control",
        )
        self.assertEqual(
            model.training_objective_class_path,
            "irene_brain.training.world_model_objective:LatentRolloutObjective",
        )

    def test_thesis_factories_land_inside_the_parameter_band(self) -> None:
        assert torch is not None
        recipe_dir = ROOT / "configs" / "training"
        reference_config = load_training_config(
            recipe_dir / "dgx-stagea-continuation-gate-b.toml"
        )
        reference = build_thesis_model(reference_config)
        self.assertEqual(
            reference.architecture_variant_id, REFERENCE_IDENTITY.variant_id
        )
        reference_count = allocated_parameter_counts(reference)["trainable"]

        actor_config = load_training_config(
            recipe_dir / "baseline-stagea-world-model-actor.toml"
        )
        self.assertEqual(
            actor_config.run.model_factory,
            "irene_brain.training.factory:build_thesis_world_model_actor_model",
        )
        actor = build_thesis_world_model_actor_model(actor_config)
        self.assertEqual(
            actor.architecture_variant_id, WORLD_MODEL_ACTOR_IDENTITY.variant_id
        )
        actor_count = allocated_parameter_counts(actor)["trainable"]
        delta_fraction = abs(actor_count - reference_count) / reference_count
        self.assertLessEqual(delta_fraction, 0.01)

        # The actor is the parameter-matched monolithic trunk plus its
        # transition, embedding, and decoder heads — nothing else may differ.
        matched_config = load_training_config(
            recipe_dir / "baseline-stagea-monolithic-parameter-matched.toml"
        )
        matched = build_thesis_parameter_matched_monolithic_model(matched_config)
        self.assertEqual(
            matched.architecture_variant_id,
            PARAMETER_MATCHED_MONOLITHIC_IDENTITY.variant_id,
        )
        matched_count = allocated_parameter_counts(matched)["trainable"]
        actor_head_parameters = sum(
            parameter.numel()
            for name, parameter in actor.named_parameters()
            if name.split(".")[0]
            in {"action_embed", "latent_transition", "latent_decoder"}
        )
        self.assertGreater(actor_head_parameters, 0)
        self.assertEqual(actor_count - actor_head_parameters, matched_count)
        trunk_state = {
            name: value
            for name, value in actor.state_dict().items()
            if name.split(".")[0]
            not in {"action_embed", "latent_transition", "latent_decoder"}
        }
        self.assertEqual(set(trunk_state), set(matched.state_dict()))

    def test_roll_sensor_embedding_shapes_determinism_and_fail_closed(self) -> None:
        assert torch is not None
        torch.manual_seed(13)
        model = self._small_actor()
        model.eval()
        width = model.config.core_width
        latent = torch.randn(3, width)
        actions = tuple(torch.randn(3, CONTROL_VECTOR_WIDTH) for _ in range(3))
        with torch.no_grad():
            first = model.roll_sensor_embedding(latent, actions)
            second = model.roll_sensor_embedding(latent, actions)
        self.assertEqual(tuple(first.shape), (3, width))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

        # The transition is residual: an all-zero action sequence still moves
        # the latent only through the learned correction, and a zero latent
        # with zero actions decodes deterministically.
        zero_latent = torch.zeros(2, width)
        zero_actions = (torch.zeros(2, CONTROL_VECTOR_WIDTH),)
        with torch.no_grad():
            rolled = model.roll_sensor_embedding(zero_latent, zero_actions)
        self.assertEqual(tuple(rolled.shape), (2, width))
        self.assertTrue(torch.isfinite(rolled).all())

        with self.assertRaises(ValueError):
            model.roll_sensor_embedding(latent, ())
        with self.assertRaises(ValueError):
            model.roll_sensor_embedding(latent, [torch.randn(3, width)])
        with self.assertRaises(ValueError):
            model.roll_sensor_embedding(latent, [torch.randn(4, CONTROL_VECTOR_WIDTH)])

    def test_rollout_objective_reports_finite_multi_horizon_losses(self) -> None:
        assert torch is not None
        torch.manual_seed(17)
        model = self._small_actor()
        objective = LatentRolloutObjective(model)
        result = objective(self._length_eight_batch())
        self.assertTrue(torch.isfinite(result.loss))
        for horizon in (1, 2, 4):
            metric = result.metrics[f"world_loss_h{horizon}"]
            self.assertTrue(torch.isfinite(metric))
        self.assertNotIn("world_loss_h8", result.metrics)
        combined = (
            result.metrics["world_loss_h1"]
            + result.metrics["world_loss_h2"]
            + result.metrics["world_loss_h4"]
        ) / 3
        self.assertTrue(torch.allclose(result.metrics["world_loss"], combined))
        # The inherited action/value/diversity terms stay metric-comparable.
        for name in (
            "action_loss",
            "value_loss",
            "diversity_loss",
            "key_accuracy",
            "thought_pairwise_cosine_mean",
            "movement_query_slot_entropy",
        ):
            self.assertIn(name, result.metrics)
            self.assertTrue(torch.isfinite(result.metrics[name]))

        # Backpropagation reaches the transition and decoder heads.
        result.loss.backward()
        head_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.split(".")[0] == "latent_transition"
        ]
        self.assertTrue(head_grads)
        self.assertTrue(any(grad is not None for grad in head_grads))

    def test_rollout_objective_rejects_models_without_rollout(self) -> None:
        assert torch is not None
        from irene_brain.model.baselines import ParameterMatchedMonolithicBaseline

        torch.manual_seed(19)
        plain = ParameterMatchedMonolithicBaseline(
            self._small_actor().config, input_resolution=(8, 8), plan_steps=2
        )
        objective = LatentRolloutObjective(plain)
        with self.assertRaises(ValueError):
            objective(self._length_eight_batch())

    def test_checked_in_manifest_is_strictly_identified(self) -> None:
        assert torch is not None
        manifest_path = ROOT / "configs" / "world-model-actor-manifest.json"
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

        self.assertEqual(manifest["manifest_kind"], "world_model_actor_recipe_family")
        self.assertEqual(manifest["reference_variant_id"], REFERENCE_IDENTITY.variant_id)
        self.assertEqual(
            manifest["training_objective_class_path"],
            "irene_brain.training.world_model_objective:LatentRolloutObjective",
        )
        self.assertEqual(len(manifest["variants"]), 1)
        entry = manifest["variants"][0]
        self.assertEqual(entry["variant_id"], WORLD_MODEL_ACTOR_IDENTITY.variant_id)
        self.assertEqual(
            entry["variant_identity_sha256"], WORLD_MODEL_ACTOR_IDENTITY.sha256
        )
        self.assertEqual(
            entry["training_recipe"]["path"].replace("\\", "/"),
            "configs/training/baseline-stagea-world-model-actor.toml",
        )
        recipe_raw = (ROOT / entry["training_recipe"]["path"]).read_bytes()
        self.assertEqual(
            sha256(recipe_raw).hexdigest(),
            entry["training_recipe"]["raw_sha256"],
        )
        self.assertLessEqual(
            entry["absolute_trainable_parameter_delta_fraction"],
            manifest["parameter_tolerance_fraction"],
        )
        self.assertEqual(
            entry["allocated_parameters"]["trainable"],
            manifest["reference_trainable_parameters"]
            + entry["trainable_parameter_delta_from_reference"],
        )

    def test_rollout_objective_import_surface(self) -> None:
        assert torch is not None
        # train.py resolves the fail-closed hook exactly like a model factory.
        from irene_brain.training.train import _model_factory

        resolved = _model_factory(LatentWorldModelActor.training_objective_class_path)
        self.assertIs(resolved, LatentRolloutObjective)


if __name__ == "__main__":
    unittest.main()
