from __future__ import annotations

import tomllib
import importlib
import sys
import threading
import unittest
from pathlib import Path


class PlaySafeConfigurationTests(unittest.TestCase):
    def test_phase0_profile_cannot_use_performance_resources(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "experiment" / "phase0-smoke.toml").open("rb") as stream:
            config = tomllib.load(stream)

        experiment = config["experiment"]
        runtime = config["runtime"]
        self.assertEqual(experiment["device"], "cpu")
        self.assertFalse(experiment["allow_gpu"])
        self.assertFalse(experiment["allow_capture"])
        self.assertFalse(experiment["allow_background_threads"])
        self.assertFalse(runtime["record_video"])
        self.assertFalse(runtime["write_artifacts"])

    def test_importing_phase0_modules_starts_nothing_and_loads_no_accelerator(self) -> None:
        modules = (
            "irene_brain",
            "irene_brain.data",
            "irene_brain.environments",
            "irene_brain.evaluation.compliance",
            "irene_brain.experiment",
            "irene_brain.runtime.clock",
            "irene_brain.runtime.continuous",
            "irene_brain.runtime.policy",
        )
        before = tuple(thread.ident for thread in threading.enumerate())
        for name in modules:
            importlib.import_module(name)
        after = tuple(thread.ident for thread in threading.enumerate())

        self.assertEqual(after, before)
        forbidden = {
            "torch",
            "tensorflow",
            "jax",
            "cv2",
            "mss",
            "PIL",
            "pyautogui",
            "pynput",
        }
        self.assertTrue(forbidden.isdisjoint(sys.modules))


if __name__ == "__main__":
    unittest.main()
