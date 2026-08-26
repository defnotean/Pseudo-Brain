"""Focused contracts for the PB21O isotonic hazard-calibration trial.

Covers: frozen trial constants, PAV determinism/monotonicity + sklearn
cross-check, the complementary-fold PAV convention, partition disjointness
against the occupied-range registry, the conditioned-path re-derivation
determinism, and create-only publishing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

SCRIPT = BRAIN_ROOT / "scripts" / "v21o_isotonic_hazard_calibration_v1.py"
spec = importlib.util.spec_from_file_location("pb21o_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
pb21o = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21o
spec.loader.exec_module(pb21o)


class FrozenTrialConstantTests(unittest.TestCase):
    def test_prereg_frozen_constants(self):
        self.assertEqual(pb21o.MODE, "pb21o_isotonic_hazard_calibration_v1")
        self.assertEqual(pb21o.CLASSIFICATION,
                         "fresh_isotonic_hazard_calibration_single_mechanism")
        self.assertEqual(pb21o.CAL_SEED_OFFSET, 167_774_720)
        self.assertEqual(pb21o.CAL_EPISODES, 256)
        self.assertEqual(pb21o.CAL_BURN_IN, 4)
        self.assertEqual(pb21o.CAL_SEQ_LEN, 16)
        self.assertEqual(pb21o.ROOTS_PER_EPISODE, 12)
        self.assertEqual(pb21o.TOTAL_ROOTS, 3072)
        self.assertEqual(pb21o.FOLD_ROWS, 1536)
        self.assertEqual(pb21o.TRIAL_SEED, 43)  # identical conditioning seed
        self.assertEqual(pb21o.FACTUAL_AGGREGATE_BIAS_LIMIT, 0.05)
        self.assertEqual(pb21o.FACTUAL_AGGREGATE_ECE_LIMIT, 0.05)
        self.assertEqual(pb21o.ALL_ACTION_CONTROL_WORSE_MARGIN, 0.01)
        self.assertEqual(pb21o.FACTORIAL_AUC_DROP_LIMIT, 0.05)
        self.assertEqual(pb21o.ONE_SIDED_ALPHA, 0.025)
        self.assertEqual(pb21o.QUANTILE_METHOD, "linear")
        self.assertEqual(pb21o.BOOTSTRAP_RESAMPLES, 10_000)
        self.assertEqual(pb21o.MIN_STRATUM_ROWS, 20)

    def test_schedule_bound_to_pb21n_refinement_contract(self):
        import v21n_prior_action_hazard_conditioning_v1 as pb21n
        self.assertEqual(pb21o.REFINEMENT_PASSES, pb21n.REFINEMENT_PASSES)
        self.assertEqual(pb21o.REFINEMENT_ROOT_BATCH_SIZE,
                         pb21n.REFINEMENT_ROOT_BATCH_SIZE)
        self.assertEqual(pb21o.REFINEMENT_LEARNING_RATE,
                         pb21n.REFINEMENT_LEARNING_RATE)

    def test_parent_seals_match_pb21n(self):
        import v21n_prior_action_hazard_conditioning_v1 as pb21n
        self.assertEqual(pb21o.PARENT_STATE_SHA256,
                         pb21n.PARENT_STATE_SHA256)
        self.assertEqual(pb21o.V3_RESULT_SHA256, pb21n.V3_RESULT_SHA256)


class PavContractTests(unittest.TestCase):
    def test_monotone_and_deterministic(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(500)
        t = (rng.random(500) < 0.5).astype(float)
        nv, np_, steps = pb21o.pav_fit(x, t)
        for _ in range(3):
            nv2, np2, steps2 = pb21o.pav_fit(x, t)
            self.assertTrue(np.all(np.diff(np_) >= -1e-12))
            self.assertTrue(np.array_equal(nv, nv2))
            self.assertTrue(np.array_equal(np_, np2))
            self.assertEqual(steps, steps2)

    def test_matches_sklearn(self):
        from sklearn.isotonic import IsotonicRegression
        rng = np.random.default_rng(11)
        worst = 0.0
        for _ in range(3):
            n = 400
            x = rng.standard_normal(n)
            t = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.2 * x))).astype(float)
            nv, np_, _ = pb21o.pav_fit(x, t)
            mine = pb21o.pav_evaluate(x, nv, np_)
            sk = IsotonicRegression(out_of_bounds="clip",
                                    y_min=0.0, y_max=1.0).fit(x, t)
            worst = max(worst, float(np.max(np.abs(mine - sk.predict(x)))))
        self.assertLess(worst, 1e-9)

    def test_complementary_fold_convention(self):
        # Build a tiny fake ev: fold f must be calibrated with the map fit
        # on fold 1-f. Verify fold0 uses fold1's (logit,target) pairs.
        fold_index = np.array([0] * 4 + [1] * 4, dtype=np.int64)
        beliefs = np.zeros(8)
        targets = np.zeros((8, 5))
        ev = pb21o.pb21n.CALEvidence(
            beliefs=np.zeros((8, pb21o.WIDTH)),
            targets=targets,
            applied=np.zeros(8, dtype=np.int64),
            prior_applied=np.zeros(8, dtype=np.int64),
            episode_ordinal=np.array([0, 0, 1, 1, 2, 2, 3, 3]),
            root_ids=np.zeros(8, dtype="S64"),
            base_raw_logits=np.zeros((8, 5)),
            fold_index=fold_index,
            episode_permutation_sha256="0" * 64,
            partition_manifest_sha256="1" * 64,
            rows=8,
        )
        raw = np.zeros((2, 8, 5))
        # only fold 0 slot rows 0..3 valid; fold 1 slot rows 4..7 valid
        raw[0][:4, 0] = [0.0, 1.0, 2.0, 3.0]
        raw[1][4:, 0] = [10.0, 11.0, 12.0, 13.0]
        targets[:4, 0] = [0.0, 0.0, 1.0, 1.0]
        targets[4:, 0] = [0.0, 1.0, 1.0, 1.0]
        calibrated, prov = pb21o.pav_crossfit_calibration(raw, ev)
        # fold 0's rows 0..3 must be calibrated with the map fit on fold 1
        # (values 10..13). Since the fit x-range is far above eval x-range
        # (0..3), right-continuous eval clamps to the first node prob.
        self.assertEqual(prov[0]["cal_fold"], 0)
        self.assertEqual(prov[0]["fit_fold"], 1)
        self.assertTrue(np.isfinite(calibrated).all())
        # complement rows must be the defined 0.0
        self.assertTrue(np.allclose(calibrated[0][4:], 0.0))
        self.assertTrue(np.allclose(calibrated[1][:4], 0.0))

    def test_empty_and_degenerate(self):
        nv, np_, steps = pb21o.pav_fit(np.array([]), np.array([]))
        self.assertEqual(steps, 0)
        # constant targets: no violation -> no pooling; each point is its
        # own block, all probability 0.0 (monotone, flat).
        x = np.array([1.0, 2.0, 3.0])
        t = np.array([0.0, 0.0, 0.0])
        nv, np_, steps = pb21o.pav_fit(x, t)
        self.assertTrue(np.allclose(np_, 0.0))
        self.assertTrue(np.all(np.diff(np_) >= -1e-12))
        # empty map is the only case that raises
        empty_nv, empty_np, _ = pb21o.pav_fit(np.array([]), np.array([]))
        with self.assertRaises(ValueError):
            pb21o.pav_evaluate(np.array([1.0]), empty_nv, empty_np)
        # a genuine violation must pool: [1,0] targets -> single mean block
        nv2, np2, steps2 = pb21o.pav_fit(
            np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        self.assertEqual(steps2, 1)
        self.assertAlmostEqual(float(np2[0]), 0.5)


class PartitionDisjointnessTests(unittest.TestCase):
    def test_pb21o_cal_range_free_in_occupied_registry(self):
        import v21m_fresh_bal_cal_first_qualification_v1 as m
        recs = m.occupied_range_registry()
        lo = pb21o.CAL_SEED_OFFSET
        hi = lo + pb21o.CAL_EPISODES
        collisions = [
            r for r in recs
            if r.get("split") == "train"
            and not (hi <= r["effective_start"] or lo >= r["effective_stop_exclusive"])
        ]
        self.assertEqual(collisions, [])

    def test_immediately_after_pb21n_cal(self):
        self.assertEqual(pb21o.CAL_SEED_OFFSET, 167_774_720)
        self.assertEqual(167_774_464 + 256, 167_774_720)


class ConditionedPathDeterminismTests(unittest.TestCase):
    def test_condition_prior_path_rederives_identical(self):
        import v21n_prior_action_hazard_conditioning_v1 as pb21n
        fold_index = np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64)
        fold_index[: pb21n.FOLD_ROWS] = 0
        fold_index[pb21n.FOLD_ROWS:] = 1
        rng = np.random.default_rng(5)
        beliefs = rng.standard_normal(
            (pb21n.TOTAL_ROOTS, pb21n.WIDTH), dtype=np.float64)
        targets = (rng.random((pb21n.TOTAL_ROOTS, 5)) < 0.3).astype(np.float64)
        prior = rng.integers(0, 5, size=pb21n.TOTAL_ROOTS)
        ev = pb21n.CALEvidence(
            beliefs=beliefs, targets=targets,
            applied=np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64),
            prior_applied=prior,
            episode_ordinal=np.repeat(np.arange(256, dtype=np.int64), 12),
            root_ids=np.zeros(pb21n.TOTAL_ROOTS, dtype="S64"),
            base_raw_logits=np.zeros((pb21n.TOTAL_ROOTS, 5)),
            fold_index=fold_index,
            episode_permutation_sha256="0" * 64,
            partition_manifest_sha256="1" * 64,
            rows=pb21n.TOTAL_ROOTS,
        )
        saved = (pb21n.REFINEMENT_PASSES, pb21n.REFINEMENT_ROOT_BATCH_SIZE)
        pb21n.REFINEMENT_PASSES = 1
        pb21n.REFINEMENT_ROOT_BATCH_SIZE = 24
        try:
            parent = pb21n.load_parent()[0].world_model.outcome_model
            a = pb21o.condition_prior_path(parent, ev)
            b = pb21o.condition_prior_path(parent, ev)
        finally:
            pb21n.REFINEMENT_PASSES, pb21n.REFINEMENT_ROOT_BATCH_SIZE = saved
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.array_equal(a, b))
        # complement half zero-defined
        for f in range(2):
            self.assertTrue(np.allclose(a[f][fold_index != f], 0.0))


class PublishContractTests(unittest.TestCase):
    def test_create_only_publish(self):
        ev_cls = pb21o.pb21n.CALEvidence
        fake_ev = ev_cls(
            beliefs=np.zeros((pb21o.TOTAL_ROOTS, pb21o.WIDTH)),
            targets=np.zeros((pb21o.TOTAL_ROOTS, 5)),
            applied=np.zeros(pb21o.TOTAL_ROOTS, dtype=np.int64),
            prior_applied=np.zeros(pb21o.TOTAL_ROOTS, dtype=np.int64),
            episode_ordinal=np.repeat(np.arange(256, dtype=np.int64), 12),
            root_ids=np.zeros(pb21o.TOTAL_ROOTS, dtype="S64"),
            base_raw_logits=np.zeros((pb21o.TOTAL_ROOTS, 5)),
            fold_index=np.repeat(np.arange(2), pb21o.FOLD_ROWS),
            episode_permutation_sha256="0" * 64,
            partition_manifest_sha256="1" * 64,
            rows=pb21o.TOTAL_ROOTS,
        )
        table = lambda: np.zeros((2, pb21o.TOTAL_ROOTS, 5))
        with tempfile.TemporaryDirectory() as tmp:
            old = pb21o.EVIDENCE_DIR
            pb21o.EVIDENCE_DIR = Path(tmp)
            try:
                result_doc = {"status": "completed", "wall_seconds": 1.0}
                pb21o._publish(result_doc, fake_ev, table(), table(),
                              table(), table())
                res = Path(tmp) / f"{pb21o.RUN_ID}.json"
                att = Path(tmp) / f"{pb21o.RUN_ID}.attempt.json"
                reg = Path(tmp) / f"{pb21o.RUN_ID}.registration.json"
                evf = Path(tmp) / f"{pb21o.RUN_ID}.evidence.npz"
                self.assertTrue(res.exists() and att.exists())
                self.assertTrue(reg.exists() and evf.exists())
                payload = res.read_text(encoding="utf-8").rstrip("\n")
                recorded = json.loads(att.read_text(encoding="utf-8"))
                self.assertEqual(recorded["result_sha256"],
                                 pb21o._sha256_bytes(payload.encode()))
                self.assertFalse(recorded["retry_allowed"])
                with self.assertRaises(RuntimeError):
                    pb21o._publish(result_doc, fake_ev, table(), table(),
                                   table(), table())
            finally:
                pb21o.EVIDENCE_DIR = old


if __name__ == "__main__":
    unittest.main()
