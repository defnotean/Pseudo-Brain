from __future__ import annotations

from dataclasses import replace
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

from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    decode_closed_loop_control,
)
from irene_brain.training.batches import BUTTON_TARGET_INDICES


def _packed_logits(*channels: int) -> list[float]:
    logits = [0.0] * len(BUTTON_TARGET_INDICES)
    for channel in channels:
        logits[BUTTON_TARGET_INDICES.index(channel)] = 1.0
    return logits

if torch is not None:
    from irene_brain.evaluation.closed_loop_play import evaluate_closed_loop_play
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel


class DecodeClosedLoopControlTests(unittest.TestCase):
    def test_threshold_is_strictly_above_zero(self) -> None:
        logits = _packed_logits(4)
        logits[BUTTON_TARGET_INDICES.index(22)] = 0.0
        control, stats = decode_closed_loop_control(logits, [0.0] * 11)
        self.assertEqual(control.keys_down, (4,))
        self.assertEqual(stats["active_button_count"], 1)

    def test_movement_mask_and_opposite_conflict(self) -> None:
        # W (26) and S (22) are opposites; A (4) without D (7) is not a conflict.
        _, stats = decode_closed_loop_control(
            _packed_logits(26, 22, 4), [0.0] * 11
        )
        self.assertEqual(stats["opposite_conflict"], 1)
        self.assertEqual(stats["movement_mask"], 0b0111)

    def test_deadzone_counting(self) -> None:
        continuous = [0.05, -0.05, 0.0500001, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _, stats = decode_closed_loop_control(
            _packed_logits(), continuous
        )
        self.assertEqual(stats["continuous_outside_deadzone"], 2)
        self.assertAlmostEqual(stats["continuous_max_abs"], 0.2)

    def test_non_movement_and_mouse_buttons(self) -> None:
        control, stats = decode_closed_loop_control(
            _packed_logits(40, 256), [0.0] * 11
        )
        self.assertEqual(control.keys_down, (40,))
        self.assertEqual(control.mouse_buttons, (0,))
        self.assertEqual(stats["non_movement_key_count"], 1)
        self.assertEqual(stats["mouse_button_count"], 1)

    def test_decode_rejects_wrong_lengths(self) -> None:
        with self.assertRaises(ValueError):
            decode_closed_loop_control([0.0] * 295, [0.0] * 11)
        with self.assertRaises(ValueError):
            decode_closed_loop_control(_packed_logits(), [0.0] * 10)


class ClosedLoopPlayConfigTests(unittest.TestCase):
    def test_requires_non_empty_unique_seeds(self) -> None:
        with self.assertRaises(ValueError):
            ClosedLoopPlayConfig(episode_seeds=())
        with self.assertRaises(ValueError):
            ClosedLoopPlayConfig(episode_seeds=(7, 7))

    def test_rejects_out_of_range_knobs(self) -> None:
        with self.assertRaises(ValueError):
            ClosedLoopPlayConfig(episode_seeds=(1,), max_ticks=0)
        with self.assertRaises(ValueError):
            ClosedLoopPlayConfig(episode_seeds=(1,), inference_latency_ns=-1)
        with self.assertRaises(ValueError):
            ClosedLoopPlayConfig(episode_seeds=(1,), expiry_slack_ns=0)

    def test_config_round_trips_through_dict(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(11, 22), max_ticks=30)
        encoded = config.to_dict()
        self.assertEqual(encoded["episode_seeds"], [11, 22])
        self.assertEqual(encoded["max_ticks"], 30)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class ClosedLoopPlayEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        config = replace(
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
        torch.manual_seed(0)
        cls.model = IreneBrainModel(config)

    def _play_config(self, **overrides: object) -> ClosedLoopPlayConfig:
        knobs: dict[str, object] = {
            "episode_seeds": (5, 9),
            "max_ticks": 24,
            "hazard_count": 2,
        }
        knobs.update(overrides)
        return ClosedLoopPlayConfig(**knobs)  # type: ignore[arg-type]

    def test_smoke_model_finishes_episodes_with_finite_outputs(self) -> None:
        report = evaluate_closed_loop_play(
            self.model,
            config=self._play_config(),
            model_description="untrained smoke model",
        )
        self.assertEqual(len(report.episodes), 2)
        for episode in report.episodes:
            self.assertEqual(episode.ticks_advanced, 24)
            self.assertGreater(episode.decisions_submitted, 0)
            self.assertTrue(episode.continuous_max_abs == episode.continuous_max_abs)

    def test_zero_latency_run_has_no_rejections(self) -> None:
        report = evaluate_closed_loop_play(
            self.model,
            config=self._play_config(inference_latency_ns=0),
            model_description="zero latency",
        )
        self.assertEqual(report.to_dict()["totals"]["decisions_rejected"], 0)
        for episode in report.episodes:
            self.assertEqual(episode.rejections_by_reason, ())

    def test_high_latency_causes_stale_frame_rejections(self) -> None:
        config = self._play_config(
            inference_latency_ns=5 * 16_666_667,
            submit_deadline_slack_ns=10 * 16_666_667,
            expiry_slack_ns=100 * 16_666_667,
        )
        report = evaluate_closed_loop_play(
            self.model,
            config=config,
            model_description="high latency",
        )
        totals = report.to_dict()["totals"]
        self.assertGreater(totals["decisions_rejected"], 0)
        reasons = {
            reason
            for episode in report.episodes
            for reason, _ in episode.rejections_by_reason
        }
        self.assertIn("action does not target the newest observation", reasons)
        dropped = sum(episode.observations_dropped for episode in report.episodes)
        self.assertGreater(dropped, 0)

    def test_evaluation_is_deterministic(self) -> None:
        config = self._play_config()
        first = evaluate_closed_loop_play(
            self.model, config=config, model_description="determinism"
        )
        second = evaluate_closed_loop_play(
            self.model, config=config, model_description="determinism"
        )
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.sha256, second.sha256)

    def test_report_round_trips_through_canonical_json(self) -> None:
        report = evaluate_closed_loop_play(
            self.model,
            config=self._play_config(),
            model_description="canonical",
        )
        decoded = __import__("json").loads(report.canonical_json)
        self.assertEqual(decoded["schema_version"], 1)
        self.assertEqual(decoded["totals"]["episodes"], 2)
        self.assertEqual(decoded["config"]["episode_seeds"], [5, 9])

    def test_training_mode_is_restored(self) -> None:
        self.model.train()
        try:
            evaluate_closed_loop_play(
                self.model,
                config=self._play_config(episode_seeds=(3,), max_ticks=4),
                model_description="mode restore",
            )
            self.assertTrue(self.model.training)
        finally:
            self.model.eval()


if __name__ == "__main__":
    unittest.main()
