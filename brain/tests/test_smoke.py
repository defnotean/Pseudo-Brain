from __future__ import annotations

import unittest
from pathlib import Path

from irene_brain.smoke import run_smoke


class Phase0SmokeTests(unittest.TestCase):
    def test_smoke_is_deterministic_and_uses_no_expansive_resource(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = root / "configs" / "experiment" / "phase0-smoke.toml"
        first = run_smoke(config)
        second = run_smoke(config)

        self.assertEqual(first, second)
        self.assertEqual(first.replay_steps, 1000)
        self.assertFalse(first.gpu_used)
        self.assertFalse(first.capture_used)
        self.assertFalse(first.hid_output_used)
        self.assertFalse(first.background_threads_used)
        self.assertFalse(first.artifacts_written)


if __name__ == "__main__":
    unittest.main()
