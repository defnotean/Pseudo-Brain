from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest


BRAIN_ROOT = Path(__file__).resolve().parents[1]


class ProjectIdentityTests(unittest.TestCase):
    def test_distribution_is_pseudo_brain_with_historical_import_namespace(self) -> None:
        with (BRAIN_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]

        self.assertEqual(project["name"], "pseudo-brain")
        self.assertEqual(project["authors"], [{"name": "Pseudo-Brain project"}])
        self.assertTrue((BRAIN_ROOT / "src" / "irene_brain" / "__init__.py").is_file())

    def test_human_facing_titles_are_rebranded_without_rewriting_schema_ids(self) -> None:
        self.assertEqual(
            (BRAIN_ROOT / "PLAN.md").read_text(encoding="utf-8").splitlines()[0],
            "# Pseudo-Brain: Streaming Thought-Field Model Build Plan",
        )
        schema_titles = {
            "experiment.schema.json": "Pseudo-Brain experiment manifest",
            "lifetime.schema.json": "Pseudo-Brain lifetime manifest",
            "step.schema.json": "Pseudo-Brain lifetime step record",
            "branch.schema.json": "Pseudo-Brain counterfactual branch record",
        }
        for filename, expected_title in schema_titles.items():
            with self.subTest(filename=filename):
                payload = json.loads(
                    (BRAIN_ROOT / "schemas" / filename).read_text(encoding="utf-8")
                )
                self.assertEqual(payload["title"], expected_title)
                self.assertEqual(
                    payload["$id"],
                    f"https://irene.local/schemas/{filename}",
                )


if __name__ == "__main__":
    unittest.main()
