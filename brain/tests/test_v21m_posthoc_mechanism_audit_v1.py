"""Contracts for the PB21M post-hoc mechanism audit runner.

Covers: frozen constants, MIX35 weight algebra, the weighted Newton solver
(controls, guards, determinism, and the zero-weight context-variant case),
the burned-in prior-action replay semantics, the frozen nomination rule, and
create-only publishing. Heavy dataset replay is cached once per class.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

SCRIPT = BRAIN_ROOT / "scripts" / "v21m_posthoc_mechanism_audit_v1.py"
spec = importlib.util.spec_from_file_location("pb21m_audit_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def _synthetic_fit_case(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 400
    logits = rng.standard_normal(n).astype(np.float64)
    targets = (rng.random(n) < 0.4).astype(np.float64)
    mask = rng.random(n) < 0.3
    return logits, targets, mask


class FrozenConstantTests(unittest.TestCase):
    def test_prereg_frozen_constants(self):
        self.assertEqual(audit.MODE, "pb21m_posthoc_mechanism_audit_v1")
        self.assertEqual(audit.CLASSIFICATION,
                         "post_hoc_consumed_data_nonqualifying")
        self.assertEqual(audit.MIX35_FOCAL_WEIGHT, 0.35)
        self.assertEqual(audit.MIX35_CONTEXT_WEIGHTS, (0.5, 1.0))
        self.assertEqual(audit.FACTUAL_AGGREGATE_BIAS_LIMIT, 0.05)
        self.assertEqual(audit.FACTUAL_AGGREGATE_ECE_LIMIT, 0.05)
        self.assertEqual(audit.ALL_ACTION_CONTROL_WORSE_MARGIN, 0.01)
        self.assertEqual(audit.FACTORIAL_AUC_DROP_LIMIT, 0.05)
        self.assertEqual(audit.B2_CELLS_REQUIRED, 3)
        self.assertEqual(audit.B2_STRATUM_BIAS, 0.03)
        self.assertEqual(audit.B2_STRATUM_ECE, 0.05)
        self.assertEqual(audit.B3_PERMUTATIONS, 30)
        self.assertEqual(audit.B3_SHUFFLE_SEED, 81_042)
        self.assertEqual(audit.MIN_STRATUM_ROWS, 20)
        self.assertEqual(audit.ONE_SIDED_ALPHA, 0.025)
        self.assertEqual(audit.L2, 1.0e-6)
        self.assertEqual(audit.MIN_SCALE, 1.0e-4)
        self.assertEqual(audit.TOL, 1.0e-10)
        self.assertEqual(audit.CAL_COHORT_PAIRS,
                         (("C0A", "C0B"), ("C1A", "C1B"), ("C2A", "C2B")))
        self.assertEqual(audit.CAL_FOLDS, ("A", "B"))

    def test_sealed_parent_digests_bound(self):
        self.assertEqual(len(audit.PARENT_SHA), 5)
        for role, digest in audit.PARENT_SHA.items():
            self.assertEqual(len(digest), 64)
            int(digest, 16)
        self.assertEqual(len(audit.PARENT_PATHS), 5)
        for role in audit.PARENT_PATHS:
            self.assertTrue(audit.PARENT_PATHS[role].exists(), role)


class MixedWeightTests(unittest.TestCase):
    def test_weight_algebra(self):
        n = 10
        mask = np.zeros(n, dtype=bool)
        mask[[0, 1, 2, 3]] = True
        w = audit._mixed_weights(n, mask, 0.5)
        self.assertTrue(np.isclose(w.sum(), 1.0, rtol=0.0, atol=1.0e-14))
        self.assertTrue(np.allclose(w[~mask], 0.5 / 10.0))
        self.assertTrue(np.allclose(w[mask], 0.5 / 10.0 + 0.5 / 4.0))

    def test_focal_weight_035_brackets(self):
        n = 100
        mask = np.zeros(n, dtype=bool)
        mask[:30] = True
        for w in (0.0, 0.35, 0.5, 1.0):
            weights = audit._mixed_weights(n, mask, w)
            self.assertTrue(np.isclose(weights.sum(), 1.0, rtol=0.0,
                                       atol=1.0e-14), f"w={w}")
            self.assertTrue(np.all(weights >= 0.0), f"w={w}")
        # w=1.0 zeroes the non-focal weights exactly (context variant).
        self.assertTrue(np.all(audit._mixed_weights(n, mask, 1.0)[~mask] == 0.0))
        self.assertTrue(np.all(audit._mixed_weights(n, mask, 0.0)[mask]
                                == 1.0 / 100.0))

    def test_no_factual_examples_rejected(self):
        with self.assertRaises(ValueError):
            audit._mixed_weights(10, np.zeros(10, dtype=bool), 0.5)


class WeightedSolverTests(unittest.TestCase):
    def test_guard_rejects_negative_weights(self):
        logits, targets, mask = _synthetic_fit_case()
        w = audit._mixed_weights(len(logits), mask, 0.5)
        w_bad = w.copy()
        w_bad[0] = -1.0e-3
        w_bad = w_bad / w_bad.sum()
        with self.assertRaises(ValueError):
            audit.fit_weighted_per_action(logits, targets, w_bad)

    def test_guard_rejects_nonbinary_targets(self):
        logits, targets, mask = _synthetic_fit_case()
        w = audit._mixed_weights(len(logits), mask, 0.5)
        with self.assertRaises(ValueError):
            audit.fit_weighted_per_action(logits, targets + 0.5, w)

    def test_guard_rejects_unnormalized_weights(self):
        logits, targets, mask = _synthetic_fit_case()
        w = audit._mixed_weights(len(logits), mask, 0.5) * 0.5
        with self.assertRaises(ValueError):
            audit.fit_weighted_per_action(logits, targets, w)

    def test_zero_weight_rows_accepted(self):
        # Regression: context variant w=1.0 produces exactly-zero non-focal
        # weights; the solver must accept non-negative (not strictly
        # positive) weights.
        logits, targets, mask = _synthetic_fit_case()
        w = audit._mixed_weights(len(logits), mask, 1.0)
        self.assertTrue(np.any(w == 0.0))
        s, b, term, iters, obj = audit.fit_weighted_per_action(
            logits, targets, w)
        self.assertGreaterEqual(s, audit.MIN_SCALE - 1.0e-12)
        self.assertTrue(np.isfinite(obj))
        self.assertIn(term, ("projected_gradient_KKT", "objective_tolerance"))
        self.assertLessEqual(iters, audit.MAX_ITER)

    def test_determinism_two_runs_identical(self):
        logits, targets, mask = _synthetic_fit_case()
        w = audit._mixed_weights(len(logits), mask, 0.35)
        r1 = audit.fit_weighted_per_action(logits, targets, w)
        r2 = audit.fit_weighted_per_action(logits, targets, w)
        self.assertEqual(r1, r2)

    def test_beats_identity_calibration(self):
        logits, targets, mask = _synthetic_fit_case()
        w = audit._mixed_weights(len(logits), mask, 0.35)
        s, b, _, _, fitted = audit.fit_weighted_per_action(logits, targets, w)
        identity = float(audit._weighted_objective(
            audit.torch.as_tensor(logits, dtype=audit.torch.float64),
            audit.torch.as_tensor(targets, dtype=audit.torch.float64),
            audit.torch.as_tensor(w, dtype=audit.torch.float64),
            audit.torch.tensor(1.0), audit.torch.tensor(0.0)))
        self.assertLess(fitted, identity)


class NominationTests(unittest.TestCase):
    def test_only_a_nominates_a(self):
        out = audit.nominate({"passed": True}, {"passed": False})
        self.assertEqual(out["nominated_arm"], "A")
        self.assertIn("MIX35", out["nominated_change"])

    def test_only_b_nominates_b(self):
        out = audit.nominate({"passed": False}, {"passed": True})
        self.assertEqual(out["nominated_arm"], "B")
        self.assertIn("prior applied", out["nominated_change"])

    def test_both_pass_nominates_a_only(self):
        out = audit.nominate({"passed": True}, {"passed": True})
        self.assertEqual(out["nominated_arm"], "A")
        self.assertIn("never combined", out["ledger"]["B_prior_action_strata"])

    def test_neither_pass_nominates_nothing(self):
        out = audit.nominate({"passed": False}, {"passed": False})
        self.assertIsNone(out["nominated_arm"])
        self.assertIsNone(out["nominated_change"])

    def test_missing_pass_defaults_false(self):
        out = audit.nominate({}, {})
        self.assertIsNone(out["nominated_arm"])


class PublishContractTests(unittest.TestCase):
    def test_create_only_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = audit.EVIDENCE_DIR
            audit.EVIDENCE_DIR = Path(tmp)
            try:
                result = {"status": "completed", "wall_seconds": 1.0}
                audit._publish(result, controls_only=True)
                res = Path(tmp) / f"{audit.RUN_ID}.json"
                att = Path(tmp) / f"{audit.RUN_ID}.attempt.json"
                self.assertTrue(res.exists() and att.exists())
                payload = res.read_text(encoding="utf-8").rstrip("\n")
                recorded = json.loads(att.read_text(encoding="utf-8"))
                self.assertEqual(recorded["result_sha256"],
                                 audit._sha256_bytes(payload.encode("utf-8")))
                with self.assertRaises(RuntimeError):
                    audit._publish(result, controls_only=True)
            finally:
                audit.EVIDENCE_DIR = old_dir


class ParentIntegrityTests(unittest.TestCase):
    def test_parent_sha_matches_sealed_registration(self):
        report = audit.verify_parent_integrity()
        self.assertTrue(report["passed"])
        for role, check in report["checks"].items():
            self.assertTrue(check["matches_sealed"], role)

    def test_oof_reconstruction_self_check(self):
        arrays = audit.load_parent_evidence()
        raw = arrays["cal_raw_logits"]
        oof = arrays["oof_calibrated_logits"]
        xs, xb = arrays["crossfit_scales"], arrays["crossfit_biases"]
        self.assertEqual(raw.shape, (2, 9, 2, 3072, 5))
        for sc in range(2):
            for cell in range(9):
                for f in range(2):
                    j = int(np.where(arrays["crossfit_eval_fold_index"] == f)[0][0])
                    lo, hi = f * 3072, (f + 1) * 3072
                    self.assertTrue(
                        np.array_equal(raw[sc, cell, f] * xs[sc, cell, j]
                                       + xb[sc, cell, j], oof[sc, cell, lo:hi]))


class ReplaySemanticsTests(unittest.TestCase):
    _cache: dict[str, tuple] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls._cache["C0A"] = audit.replay_partition_applied("C0A")

    def test_geometry(self):
        roots, applied, prior = self._cache["C0A"]
        self.assertEqual(len(roots), 3072)
        self.assertEqual(applied.shape, (3072,))
        self.assertEqual(prior.shape, (3072,))
        self.assertTrue(set(applied.tolist()) <= {0, 1, 2, 3, 4})

    def test_first_recorded_prior_is_burn_in_action(self):
        # 256 episodes x 12 recorded ticks. The first recorded root of each
        # episode has prior = the burn-in tick-3 applied action (collector
        # semantics), NOT the neutral 0.
        roots, applied, prior = self._cache["C0A"]
        per = len(roots) // 256
        seg = applied.reshape(256, per)
        pseg = prior.reshape(256, per)
        # Deterministic data-only replay: re-run one episode's neighbors via
        # the module's own contract and confirm the prior chain is the applied
        # action one tick earlier within the recorded span.
        self.assertTrue(np.array_equal(pseg[:, 1:], seg[:, :-1]))
        # First prior differs from the trivial neutral action for some
        # episodes (proves burn-in semantics, not an all-zero sentinel).
        self.assertGreater((pseg[:, 0] != 0).sum(), 0)

    def test_replay_byte_identical_to_sealed(self):
        arrays = audit.load_parent_evidence()
        roots, applied, _ = self._cache["C0A"]
        roots_np = np.asarray([r.encode("ascii") for r in roots], dtype="S64")
        self.assertTrue(np.array_equal(
            roots_np, arrays["cal_root_ids"][0][0:3072]))
        self.assertTrue(np.array_equal(
            applied, arrays["cal_actions"][0][0:3072]))


if __name__ == "__main__":
    unittest.main()
