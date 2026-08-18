from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ModuleNotFoundError:  # Phase 0's stdlib-only environment remains supported.
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.evaluation.checkpoint_compare import (
        load_analysis_payload,
        metric_delta_table,
        objective_states,
        parameter_differences,
        require_same_run_identity,
    )
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.training.batches import MovingShapesBatchSource
    from irene_brain.training.checkpoint import TrainerCursor, save_checkpoint
    from irene_brain.training.config import (
        DatasetConfig,
        DeterminismConfig,
        LoggingConfig,
        OptimizationConfig,
        PrecisionConfig,
        ResourceConfig,
        RunConfig,
        TrainingConfig,
    )
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem


@unittest.skipIf(torch is None, "PyTorch is optional for the play-safe suite")
class CheckpointCompareTests(unittest.TestCase):
    @staticmethod
    def _config() -> TrainingConfig:
        return TrainingConfig(
            schema_version=1,
            run=RunConfig(
                name="checkpoint-compare-test",
                seed=20260818,
                model_factory="irene_brain.model.torch_model:IreneBrainModel",
                max_optimizer_steps=4,
            ),
            dataset=DatasetConfig(
                kind="moving_shapes",
                train_sequences=4,
                validation_sequences=2,
                test_sequences=2,
                sequence_length=2,
                burn_in_steps=1,
                seed_offset=0,
                hazard_count=1,
                tick_period_ns=33_333_333,
                discount=0.99,
            ),
            optimization=OptimizationConfig(
                batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                max_gradient_norm=1.0,
                warmup_steps=0,
            ),
            precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
            determinism=DeterminismConfig(
                enabled=True, num_workers=0, compile_model=False
            ),
            logging=LoggingConfig(
                log_every_steps=1,
                evaluate_every_steps=1,
                validation_batches=1,
                checkpoint_every_steps=1,
                keep_last_checkpoints=1,
            ),
            resources=ResourceConfig(
                allow_gpu=False,
                allow_capture=False,
                allow_hid_output=False,
                allow_background_threads=False,
                allow_network=False,
                allow_subprocess=False,
                write_artifacts=True,
                cpu_threads=1,
            ),
        )

    @staticmethod
    def _system(config: TrainingConfig, *, seed: int) -> TorchTrainingSystem:
        torch.manual_seed(seed)
        model_config = replace(
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
        model = IreneBrainModel(model_config, input_resolution=(8, 8), plan_steps=2)
        return TorchTrainingSystem(ThoughtFieldObjective(model), config)

    def _save(self, system: TorchTrainingSystem, path: Path, step: int) -> str:
        config = system.config
        return save_checkpoint(
            path,
            cursor=TrainerCursor(optimizer_step=step),
            system_state=system.checkpoint_state(),
            rng_state=system.capture_rng_state(),
            config_sha256=config.config_sha256,
            data_sha256="d" * 64,
            code_sha256="c" * 64,
            runtime_fingerprint={"runtime": "test-cpu", "world_size": 1},
            policy=config.resource_policy,
        )

    def test_analysis_roundtrip_and_run_identity_guard(self) -> None:
        assert torch is not None
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.pt"
            second = Path(directory) / "b.pt"
            self._save(self._system(config, seed=7), first, step=1)
            self._save(self._system(config, seed=7), second, step=2)
            payload_a = load_analysis_payload(first)
            payload_b = load_analysis_payload(second)
            self.assertEqual(payload_a["cursor"]["optimizer_step"], 1)
            self.assertEqual(payload_b["cursor"]["optimizer_step"], 2)
            require_same_run_identity(payload_a, payload_b)
            mutated = dict(payload_b)
            mutated["data_sha256"] = "e" * 64
            with self.assertRaises(ValueError):
                require_same_run_identity(payload_a, mutated)
            with self.assertRaises(ValueError):
                load_analysis_payload(first.parent / "missing.pt")

    def test_parameter_differences_rank_a_perturbed_tensor(self) -> None:
        assert torch is not None
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            system = self._system(config, seed=7)
            path = Path(directory) / "a.pt"
            self._save(system, path, step=1)
            baseline = objective_states(load_analysis_payload(path))
            target_name = next(
                name
                for name, value in baseline.items()
                if name.endswith("thought_attention.feed_forward.0.weight")
            )
            with torch.no_grad():
                parameter = dict(system.objective.named_parameters())[target_name]
                parameter.add_(0.5)
            perturbed = objective_states(
                {
                    "system_state": {"objective": system.objective.state_dict()},
                }
            )
            rows = parameter_differences(baseline, perturbed)
            self.assertEqual(rows[0]["tensor"], target_name)
            self.assertFalse(rows[0]["identical"])
            self.assertGreater(rows[0]["relative_l2"], 0.0)
            unchanged = [row for row in rows if row["tensor"] != target_name]
            self.assertTrue(all(row["identical"] for row in unchanged))
            with self.assertRaises(ValueError):
                parameter_differences(baseline, {"extra": torch.zeros(1)})

    def test_metric_delta_table_requires_shared_keys(self) -> None:
        first = [{"action_loss": 1.0}, {"action_loss": 3.0}]
        second = [{"action_loss": 2.0}, {"action_loss": 2.0}]
        table = metric_delta_table(first, second)
        self.assertEqual(table["action_loss"], {"first": 2.0, "second": 2.0, "delta": 0.0})
        with self.assertRaises(ValueError):
            metric_delta_table(first, [{"action_loss": 2.0, "extra": 1.0}, {"action_loss": 2.0, "extra": 1.0}])
        with self.assertRaises(ValueError):
            metric_delta_table([], [])


if __name__ == "__main__":
    unittest.main()
