from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from irene_brain.project_paths import resolve_workspace_path


BRAIN_ROOT = Path(__file__).resolve().parents[1]


class ProjectPathTests(unittest.TestCase):
    def test_relative_paths_anchor_under_brain_root_independent_of_cwd(self) -> None:
        elsewhere = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with patch("os.getcwd", return_value=str(elsewhere)):
            resolved = resolve_workspace_path(Path("runs") / "local-test")

        self.assertEqual(resolved, BRAIN_ROOT / "runs" / "local-test")
        self.assertTrue(resolved.is_absolute())

    def test_absolute_dgx_style_paths_remain_explicit(self) -> None:
        absolute = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.assertEqual(resolve_workspace_path(absolute), absolute)

    def test_parent_and_drive_relative_traversal_are_rejected(self) -> None:
        rejected = (Path("..") / "Irene" / "run", Path("runs/../../Irene/run"))
        if os.name == "nt":
            rejected += (Path("C:relative-run"),)
        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    resolve_workspace_path(path)

    def test_existing_reparse_component_is_rejected(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        component = root / "linked"
        component.mkdir()

        with patch(
            "irene_brain.project_paths._is_reparse",
            side_effect=lambda path: path == component,
        ):
            with self.assertRaisesRegex(ValueError, "symlink or reparse"):
                resolve_workspace_path(component / "result.json")

    def test_all_three_cli_writers_resolve_their_paths(self) -> None:
        expected_calls = {
            BRAIN_ROOT / "src" / "irene_brain" / "training" / "train.py": 3,
            BRAIN_ROOT / "scripts" / "build_baseline_architecture_manifest.py": 1,
            BRAIN_ROOT / "scripts" / "build_first_matched_multiseed_campaign.py": 1,
        }
        for path, minimum_calls in expected_calls.items():
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "resolve_workspace_path"
                ]
                self.assertGreaterEqual(len(calls), minimum_calls)


if __name__ == "__main__":
    unittest.main()
