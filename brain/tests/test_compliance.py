from __future__ import annotations

import dataclasses
import unittest

from irene_brain.data.records import PRIVILEGED_STEP_FIELDS
from irene_brain.evaluation.compliance import (
    ComplianceCode,
    DeploymentManifest,
    audit_deployment,
    require_compliant,
)
from irene_brain.types import ModelObservation


class ComplianceTests(unittest.TestCase):
    def test_one_checkpoint_and_observation_allowlist_is_compliant(self) -> None:
        fields = tuple(field.name for field in dataclasses.fields(ModelObservation))
        manifest = DeploymentManifest(
            checkpoint_hashes=("a" * 64, "a" * 64),
            model_input_fields=fields,
        )

        self.assertTrue(audit_deployment(manifest).compliant)
        self.assertIs(require_compliant(manifest), manifest)

    def test_every_stored_or_current_control_alias_is_rejected(self) -> None:
        canonical = tuple(field.name for field in dataclasses.fields(ModelObservation))
        aliases = PRIVILEGED_STEP_FIELDS | {
            "action",
            "applied_action",
            "current_action",
            "current_control",
            "requested_action",
        }
        for alias in aliases:
            with self.subTest(alias=alias):
                report = audit_deployment(
                    DeploymentManifest(
                        checkpoint_hashes=("a" * 64,),
                        model_input_fields=(*canonical, f"record.{alias}"),
                    )
                )
                self.assertIn(
                    ComplianceCode.PRIVILEGED_INPUT,
                    {violation.code for violation in report.violations},
                )

    def test_unknown_noncanonical_model_field_is_rejected(self) -> None:
        report = audit_deployment(
            DeploymentManifest(
                checkpoint_hashes=("a" * 64,),
                model_input_fields=("observation.rgb", "observation.debug_hint"),
            )
        )
        self.assertIn(
            ComplianceCode.PRIVILEGED_INPUT,
            {violation.code for violation in report.violations},
        )

    def test_all_one_brain_violations_are_reported_together(self) -> None:
        manifest = DeploymentManifest(
            checkpoint_hashes=("a" * 64, "b" * 64),
            model_input_fields=(
                "observation.rgb",
                "manifest.world_lineage_id",
                "target.future_events",
            ),
            per_game_adapters=("minecraft-head",),
            uses_external_planner=True,
            uses_remote_inference_in_deadline_loop=True,
        )

        codes = {violation.code for violation in audit_deployment(manifest).violations}
        self.assertEqual(
            codes,
            {
                ComplianceCode.MULTIPLE_CHECKPOINTS,
                ComplianceCode.GAME_ID_INPUT,
                ComplianceCode.PER_GAME_ADAPTER,
                ComplianceCode.EXTERNAL_PLANNER,
                ComplianceCode.PRIVILEGED_INPUT,
                ComplianceCode.REMOTE_DEADLINE_INFERENCE,
            },
        )

    def test_manifest_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValueError):
            DeploymentManifest.from_dict(
                {
                    "checkpoint_hashes": ["a" * 64],
                    "hidden_game_router": True,
                }
            )


if __name__ == "__main__":
    unittest.main()
