from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.evaluation.closed_loop_play import ClosedLoopPlayConfig
from irene_brain.evaluation.cross_world_matrix import (
    CrossWorldMatrixReport,
    default_world_slots,
    evaluate_cross_world_matrix,
    matrix_policies,
)
from irene_brain.evaluation.diagnostic_policies import OraclePolicy


def _config(**overrides: object) -> ClosedLoopPlayConfig:
    knobs: dict[str, object] = {
        "episode_seeds": (5, 9),
        "max_ticks": 120,
        "hazard_count": 3,
    }
    knobs.update(overrides)
    return ClosedLoopPlayConfig(**knobs)  # type: ignore[arg-type]


class CrossWorldMatrixTests(unittest.TestCase):
    def test_canonical_slots_and_policies(self) -> None:
        self.assertEqual(
            [slot.identity for slot in default_world_slots()],
            [
                "world.moving_shapes.v1",
                "world.pursuit.v1",
                "world.junction.v1",
                "world.occlusion.v1",
                "world.keys_doors.v1",
                "world.maze_chase.v1",
            ],
        )
        self.assertEqual(
            [policy.identity for policy in matrix_policies()],
            [
                "diagnostic.noop.v1",
                "diagnostic.random_movement.v1",
                "diagnostic.scripted_chase.v1",
                "diagnostic.scripted_pellet_teacher.v1",
            ],
        )

    def test_matrix_covers_every_policy_world_pair(self) -> None:
        report = evaluate_cross_world_matrix(config=_config())
        self.assertEqual(len(report.cells), 24)
        for slot in default_world_slots():
            for policy in matrix_policies():
                cell = report.cell(slot.identity, policy.identity)
                self.assertEqual(len(cell.episodes), 2)

    def test_matrix_is_deterministic(self) -> None:
        first = evaluate_cross_world_matrix(config=_config())
        second = evaluate_cross_world_matrix(config=_config())
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.sha256, second.sha256)

    def test_noop_floor_is_flat_on_every_world(self) -> None:
        report = evaluate_cross_world_matrix(config=_config())
        for slot in default_world_slots():
            totals = report.cell(slot.identity, "diagnostic.noop.v1").to_dict()[
                "totals"
            ]
            self.assertEqual(totals["targets_collected"], 0)
            self.assertEqual(totals["decisions_rejected"], 0)

    def test_scripted_chaser_collects_on_moving_shapes(self) -> None:
        report = evaluate_cross_world_matrix(config=_config())
        totals = report.cell(
            "world.moving_shapes.v1", "diagnostic.scripted_chase.v1"
        ).to_dict()["totals"]
        self.assertGreaterEqual(totals["targets_collected"], 1)

    def test_matrix_rejects_privileged_policies(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_cross_world_matrix(
                config=_config(), policies=(OraclePolicy(),)
            )

    def test_matrix_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_cross_world_matrix(config=object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            evaluate_cross_world_matrix(config=_config(), worlds=())
        with self.assertRaises(ValueError):
            evaluate_cross_world_matrix(config=_config(), policies=())
        slots = default_world_slots()
        with self.assertRaises(ValueError):
            evaluate_cross_world_matrix(config=_config(), worlds=(slots[0], slots[0]))

    def test_report_round_trips_through_canonical_json(self) -> None:
        import json

        report = evaluate_cross_world_matrix(
            config=_config(episode_seeds=(5,), max_ticks=30)
        )
        decoded = json.loads(report.canonical_json)
        self.assertEqual(decoded["schema_version"], 1)
        self.assertEqual(len(decoded["cells"]), 24)
        self.assertEqual(decoded["config"]["episode_seeds"], [5])

    def test_cell_reports_must_share_the_matrix_config(self) -> None:
        report = evaluate_cross_world_matrix(
            config=_config(episode_seeds=(5,), max_ticks=30)
        )
        other = evaluate_cross_world_matrix(
            config=_config(episode_seeds=(6,), max_ticks=30)
        )
        cells = (report.cells[0],) + tuple(
            (world, policy, other_cell)
            for world, policy, other_cell in other.cells[1:2]
        )
        with self.assertRaises(ValueError):
            CrossWorldMatrixReport(
                schema_version=1, config=report.config, cells=cells
            )


if __name__ == "__main__":
    unittest.main()
