"""RCQ-v3 evaluator lineage identity and namespace-safety guards.

The historical RCQ-v2 evaluator bundle is pinned by digest inside the immutable
v2 registrations and must never be edited.  The v3 lineage therefore lives in
new modules derived by ``brain/scripts/build_rcq_v3_lineage.py`` with asserted
exact-count substitutions.  These tests pin:

1. The v3 identity constants (fresh sealed TEST namespace, qualification id,
   evaluator id, run id, paths) and their independence from the frozen v2
   identity, which must stay byte-identical.
2. Substitution completeness: no v2-identity string may survive in the v3
   modules outside the deliberate residuals (shared gate IDs, shared
   check-module imports, unchanged function names, and the v2 history entries
   in the v3 namespace ledger).
3. The static sealed-range guard for the v3 TEST family, using only the
   target-blind overlap predicate (no claimed dataset is ever constructed).
4. Cross-lineage rejection: the v2 registration payload must fail v3
   validation and the frozen v2 registration file must keep its external pin.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

from irene_brain.data import DatasetSplit, MovingShapesDatasetConfig
from irene_brain.data.moving_shapes_dataset import (
    MovingShapesSequenceDataset,
    _claimed_rcq_v3_test_dataset,
    _overlaps_rcq_v2_sealed_test_range,
    _overlaps_rcq_v3_sealed_test_range,
)
from irene_brain.evaluation import rcq_v2_final, rcq_v3_final
from irene_brain.evaluation.rcq_v2 import RCQInputError
from irene_brain.training.config import load_training_config


BRAIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BRAIN_ROOT.parent
EVALUATION_ROOT = BRAIN_ROOT / "src" / "irene_brain" / "evaluation"

RCQ_V2_REGISTRATION_SHA256 = (
    "6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690"
)
RCQ_V2_CLAIM_ID = (
    "33e89580b6fd43ddf0e6765a683c9a9aad60dc1faba93d76c03a470bd1520f69"
)
RCQ_V3_CONFIG_SHA256 = (
    "17f2c1c2e95e30e5bd50dbf4ed409d9c25904c40b89b3b5eff52676f7d658105"
)

# Deliberate residual tokens in the v3 modules: shared frozen gate IDs and
# check-module imports, unchanged function names, and the v2 history entries
# in the v3 namespace ledger.  Everything else v2-flavored is forbidden.
FORBIDDEN_RESIDUALS = (
    "rcq_v2_reference_v2",
    "rcq-v2-reference-v2",
    "dgx-rcq-v2",
    "rcq_v2_final_v1",
    "rcq_v2_pin_pretraining_v1",
    "rcq_v2_authorize_final_v1",
    "rcq_v2_preclaim_v1",
    "rcq_v2_verify_receipt_v1",
    "rcq_v2_final_once_v1",
    "rcq-v2-final.",
    "dgx-rcq-v2-reference.toml",
    "excluded_from_rcq_v2",
    "RCQ-v2",
)


def _test_config(*, split: DatasetSplit, start: int, count: int = 1) -> object:
    return MovingShapesDatasetConfig(
        split=split,
        sequence_count=count,
        sequence_length=8,
        seed_offset=start,
        hazard_count=3,
        tick_period_ns=16_666_667,
        discount=0.99,
    )


class RCQV3IdentityTests(unittest.TestCase):
    def test_v3_constants_and_independence_from_v2(self) -> None:
        self.assertEqual(rcq_v3_final.FINAL_EVALUATOR_ID, "rcq_v3_final_v1")
        self.assertEqual(rcq_v3_final.RECIPIENT_OFFSET, 4_194_304)
        self.assertEqual(rcq_v3_final.DONOR_OFFSET, 4_194_816)
        self.assertEqual(rcq_v3_final.RETIRED_END, 4_195_328)
        self.assertEqual(rcq_v3_final.GUARD_END, 4_195_840)
        self.assertEqual(rcq_v3_final.RECIPIENT_SEQUENCES, 512)
        self.assertEqual(rcq_v3_final.DONOR_SEQUENCES, 512)
        self.assertEqual(rcq_v3_final.FINAL_STEP, 2_048)
        self.assertEqual(rcq_v3_final.FUTURE_CAMPAIGN_OFFSET, 2_097_152)
        self.assertEqual(
            rcq_v3_final.FINAL_RANGE_CLAIM_PROTOCOL,
            "irene_moving_shapes_test_range_retirement_v1",
        )
        self.assertEqual(
            rcq_v3_final.EVALUATOR_BUNDLE_FILES,
            (
                "evaluation/rcq_v2.py",
                "evaluation/rcq_v3_final.py",
                "evaluation/rcq_v3_torch.py",
            ),
        )
        self.assertNotEqual(
            rcq_v3_final._final_range_claim_id(), RCQ_V2_CLAIM_ID
        )
        # The frozen v2 identity must not have moved.
        self.assertEqual(rcq_v2_final.FINAL_EVALUATOR_ID, "rcq_v2_final_v1")
        self.assertEqual(rcq_v2_final.RECIPIENT_OFFSET, 3_145_728)
        self.assertEqual(rcq_v2_final.GUARD_END, 3_147_264)
        self.assertEqual(rcq_v2_final._final_range_claim_id(), RCQ_V2_CLAIM_ID)
        self.assertEqual(
            rcq_v2_final.EVALUATOR_BUNDLE_FILES,
            (
                "evaluation/rcq_v2.py",
                "evaluation/rcq_v2_final.py",
                "evaluation/rcq_v2_torch.py",
            ),
        )

    def test_v3_namespace_is_disjoint_from_every_historical_range(self) -> None:
        v3 = (4_194_304, 4_195_840)
        historical = (
            ("v0_opened_retired", 1_048_576, 1_050_112),
            ("first_matched_reservation", 2_097_152, 2_097_920),
            ("v2_sealed_family", 3_145_728, 3_147_264),
        )
        for name, start, end in historical:
            with self.subTest(name=name):
                self.assertTrue(v3[1] <= start or v3[0] >= end)
        # The v3 family is contiguous: recipient, donor, then guard.
        self.assertEqual(rcq_v3_final.DONOR_OFFSET, 4_194_304 + 512)
        self.assertEqual(rcq_v3_final.RETIRED_END, 4_194_816 + 512)
        self.assertEqual(rcq_v3_final.GUARD_END, 4_195_328 + 512)

    def test_v3_modules_carry_no_v2_identity_residuals(self) -> None:
        for name in (
            "rcq_v3_final.py",
            "rcq_v3_torch.py",
            "rcq_v3_registration.py",
        ):
            text = (EVALUATION_ROOT / name).read_text(encoding="utf-8")
            for token in FORBIDDEN_RESIDUALS:
                with self.subTest(file=name, token=token):
                    self.assertNotIn(token, text)
            with self.subTest(file=name, token="qualification"):
                self.assertIn("rcq_v3_reference_v1", text)

    def test_v3_registered_config_identity(self) -> None:
        config = load_training_config(
            BRAIN_ROOT / "configs" / "training" / "dgx-rcq-v3-reference.toml"
        )
        self.assertEqual(config.config_sha256, RCQ_V3_CONFIG_SHA256)
        self.assertEqual(config.run.name, "dgx-rcq-v3-reference")
        self.assertEqual(config.run.seed, 1702)
        self.assertEqual(config.run.max_optimizer_steps, 2_048)
        self.assertEqual(config.stages[0].end_optimizer_step, 1_536)
        self.assertEqual(config.objective.continuous_deadzone_hinge_weight, 0.5)
        self.assertEqual(config.objective.continuous_deadzone_hinge_margin, 0.04)
        self.assertEqual(config.objective.opposite_key_pair_weight, 0.25)
        self.assertEqual(config.objective.continuous_output_squash, "deadzone_tanh")

    def test_frozen_v2_registration_keeps_its_external_pin(self) -> None:
        encoded = (REPO_ROOT / "registrations" / "rcq-v2-reference-v2.json").read_bytes()
        self.assertEqual(sha256(encoded).hexdigest(), RCQ_V2_REGISTRATION_SHA256)

    def test_v2_registration_payload_fails_v3_validation(self) -> None:
        raw = json.loads(
            (REPO_ROOT / "registrations" / "rcq-v2-reference-v2.json").read_text(
                encoding="utf-8"
            )
        )
        # The payload remains valid under the frozen v2 validator ...
        rcq_v2_final._validate_registration(raw)
        # ... and is rejected by the v3 lineage before anything else happens.
        with self.assertRaises(RCQInputError):
            rcq_v3_final._validate_registration(raw)


class RCQV3SealedRangeGuardTests(unittest.TestCase):
    def test_static_guard_identifies_sealed_rcq_v3_test_range_overlap(self) -> None:
        for name, start, count, expected in (
            ("recipient", 4_194_304, 1, True),
            ("donor", 4_194_816, 1, True),
            ("guard", 4_195_328, 1, True),
            ("overlap", 4_194_303, 2, True),
            ("adjacent", 4_195_840, 1, False),
        ):
            with self.subTest(name=name):
                self.assertIs(
                    _overlaps_rcq_v3_sealed_test_range(
                        _test_config(split=DatasetSplit.TEST, start=start, count=count)
                    ),
                    expected,
                )
        self.assertFalse(
            _overlaps_rcq_v3_sealed_test_range(
                _test_config(split=DatasetSplit.TRAIN, start=4_194_304)
            )
        )

    def test_v3_guard_is_independent_from_the_v2_guard(self) -> None:
        self.assertFalse(
            _overlaps_rcq_v3_sealed_test_range(
                _test_config(split=DatasetSplit.TEST, start=3_145_728)
            )
        )
        self.assertFalse(
            _overlaps_rcq_v2_sealed_test_range(
                _test_config(split=DatasetSplit.TEST, start=4_194_304)
            )
        )

    def test_plain_construction_cannot_open_the_v3_family(self) -> None:
        for start in (4_194_304, 4_194_816, 4_195_328):
            with self.subTest(start=start):
                with self.assertRaises(PermissionError):
                    MovingShapesSequenceDataset(
                        _test_config(split=DatasetSplit.TEST, start=start, count=8)
                    )

    def test_claimed_v3_factory_rejects_any_non_registered_slice(self) -> None:
        # Only rejection paths are exercised: no claimed dataset is ever
        # constructed outside the trusted post-claim evaluator.
        for start, count in (
            (4_194_304, 1),
            (4_194_304, 511),
            (4_194_303, 512),
            (4_194_817, 512),
            (4_195_328, 512),
        ):
            with self.subTest(start=start, count=count):
                with self.assertRaises(PermissionError):
                    _claimed_rcq_v3_test_dataset(
                        _test_config(split=DatasetSplit.TEST, start=start, count=count)
                    )
        with self.assertRaises(PermissionError):
            _claimed_rcq_v3_test_dataset(
                _test_config(split=DatasetSplit.VALIDATION, start=4_194_304, count=512)
            )


if __name__ == "__main__":
    unittest.main()
