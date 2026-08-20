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
        code = (
            "import importlib, sys, threading\n"
            "modules = ('irene_brain', 'irene_brain.data', 'irene_brain.environments', "
            "'irene_brain.evaluation.compliance', 'irene_brain.experiment', "
            "'irene_brain.runtime.clock', 'irene_brain.runtime.continuous', 'irene_brain.runtime.policy')\n"
            "before = tuple(thread.ident for thread in threading.enumerate())\n"
            "for name in modules:\n"
            "    importlib.import_module(name)\n"
            "after = tuple(thread.ident for thread in threading.enumerate())\n"
            "assert after == before, f'{after} != {before}'\n"
            "forbidden = {'torch', 'tensorflow', 'jax', 'cv2', 'mss', 'PIL', 'pyautogui', 'pynput'}\n"
            "assert forbidden.isdisjoint(sys.modules), f'{forbidden.intersection(sys.modules)}'\n"
        )
        import subprocess
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
