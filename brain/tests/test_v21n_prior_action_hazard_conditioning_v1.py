"""Focused contracts for the PB21N prior-action hazard trial.

Covers: frozen trial constants, the fresh partition's disjointness against
the occupied-range registry, the grafted hazard-path init-identity contract
(the prior channel must add ~nothing before training), the hazard-refinement
step-count + determinism contract, the episode-clustered bootstrap geometry,
and create-only publishing. Full evidence collection / gates are exercised by
the trial runner itself; these tests keep the machinery honest.
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

SCRIPT = BRAIN_ROOT / "scripts" / "v21n_prior_action_hazard_conditioning_v1.py"
spec = importlib.util.spec_from_file_location("pb21n_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
pb21n = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21n
spec.loader.exec_module(pb21n)

_SHARED_PARENT: object = None


def _shared_parent():
    global _SHARED_PARENT
    if _SHARED_PARENT is None:
        model, _, _ = pb21n.load_parent()
        _SHARED_PARENT = model.world_model.outcome_model
    return _SHARED_PARENT


class FrozenTrialConstantTests(unittest.TestCase):
    def test_prereg_frozen_constants(self):
        self.assertEqual(pb21n.MODE, "pb21n_prior_action_hazard_conditioning_v1")
        self.assertEqual(pb21n.CLASSIFICATION,
                         "fresh_hazard_prior_action_single_change")
        self.assertEqual(pb21n.CAL_SEED_OFFSET, 167_774_464)
        self.assertEqual(pb21n.CAL_EPISODES, 256)
        self.assertEqual(pb21n.CAL_BURN_IN, 4)
        self.assertEqual(pb21n.CAL_SEQ_LEN, 16)
        self.assertEqual(pb21n.ROOTS_PER_EPISODE, 12)
        self.assertEqual(pb21n.TOTAL_ROOTS, 3072)
        self.assertEqual(pb21n.FOLD_ROWS, 1536)
        self.assertEqual(pb21n.FACTUAL_AGGREGATE_BIAS_LIMIT, 0.05)
        self.assertEqual(pb21n.FACTUAL_AGGREGATE_ECE_LIMIT, 0.05)
        self.assertEqual(pb21n.ALL_ACTION_CONTROL_WORSE_MARGIN, 0.01)
        self.assertEqual(pb21n.FACTORIAL_AUC_DROP_LIMIT, 0.05)
        self.assertEqual(pb21n.ONE_SIDED_ALPHA, 0.025)
        self.assertEqual(pb21n.QUANTILE_METHOD, "linear")
        self.assertEqual(pb21n.BOOTSTRAP_RESAMPLES, 10_000)
        self.assertEqual(pb21n.G5_SHUFFLE_PERMUTATIONS, 30)
        self.assertEqual(pb21n.G5_SHUFFLE_SEED, 81_043)
        self.assertEqual(pb21n.MIN_STRATUM_ROWS, 20)

    def test_schedule_bound_to_v21i_refinement_contract(self):
        import v21i_development_runner as development
        self.assertEqual(pb21n.REFINEMENT_PASSES,
                         development.HAZARD_REFINEMENT_PASSES)
        self.assertEqual(pb21n.REFINEMENT_ROOT_BATCH_SIZE,
                         development.HAZARD_REFINEMENT_ROOT_BATCH_SIZE)
        self.assertEqual(pb21n.REFINEMENT_LEARNING_RATE,
                         development.HAZARD_REFINEMENT_LEARNING_RATE)
        self.assertEqual(pb21n.REFINEMENT_WEIGHT_DECAY,
                         development.HAZARD_REFINEMENT_WEIGHT_DECAY)
        self.assertEqual(pb21n.REFINEMENT_CLIP_NORM,
                         development.HAZARD_REFINEMENT_CLIP_NORM)

    def test_parent_seals_bound(self):
        self.assertEqual(len(pb21n.PARENT_STATE_SHA256), 64)
        self.assertEqual(len(pb21n.V3_RESULT_SHA256), 64)
        import v21i_strict_live_representation_probe_v1 as strict
        self.assertEqual(pb21n.PARENT_SHA256,
                         strict.EXACT_V3_PARENT_SHA256)
        self.assertEqual(pb21n.PARENT_STATE_SHA256,
                         strict.EXACT_V3_PARENT_STATE_SHA256)


class PartitionDisjointnessTests(unittest.TestCase):
    def test_pb21n_cal_range_free_in_occupied_registry(self):
        import v21m_fresh_bal_cal_first_qualification_v1 as m
        recs = m.occupied_range_registry()
        lo, hi = pb21n.CAL_SEED_OFFSET, pb21n.CAL_SEED_OFFSET + pb21n.CAL_EPISODES
        collisions = [
            r for r in recs
            if r.get("split") == "train"
            and not (hi <= r["effective_start"] or lo >= r["effective_stop_exclusive"])
        ]
        self.assertEqual(collisions, [])

    def test_immediately_after_pb21m_c2b(self):
        # C2B ends at 167_774_464; PB21N-CAL starts there.
        self.assertEqual(pb21n.CAL_SEED_OFFSET, 167_774_464)
        self.assertEqual(pb21n.CAL_SEED_OFFSET - 1, 167_774_463)


class GraftContractTests(unittest.TestCase):
    def test_init_identity_within_float32_tolerance(self):
        parent = _shared_parent()
        base = pb21n.graft_base(parent)
        prior = pb21n.graft_prior(parent)
        report = pb21n.assert_prior_init_identity(base, prior)
        self.assertTrue(report["passed"])
        self.assertLess(report["max_abs_delta_at_init"], 1.0e-6)

    def test_base_graft_matches_parent_hazard_path(self):
        parent = _shared_parent()
        base = pb21n.graft_base(parent)
        for dst, src in pb21n._PARENT_HAZARD_TENSOR_MAP:
            self.assertTrue(torch.equal(
                pb21n._param(base, dst),
                pb21n._param(parent, src),
            ), dst)
        self.assertTrue(torch.equal(
            base.outcome_trunk[0].weight,
            parent.hazard_outcome_trunk[0].weight,
        ))

    def test_prior_graft_zero_extension(self):
        parent = _shared_parent()
        prior = pb21n.graft_prior(parent)
        parent_trunk = parent.hazard_outcome_trunk[0]
        w = prior.outcome_trunk[0].weight
        self.assertEqual(w.shape, (pb21n.HIDDEN, 3 * pb21n.HIDDEN))
        self.assertTrue(torch.equal(
            w[:, : 2 * pb21n.HIDDEN], parent_trunk.weight
        ))
        self.assertTrue(torch.equal(
            w[:, 2 * pb21n.HIDDEN:],
            torch.zeros_like(w[:, 2 * pb21n.HIDDEN:]),
        ))

    def test_grafting_does_not_mutate_parent(self):
        parent = _shared_parent()
        before = pb21n._state_dict_sha256(parent.state_dict())
        pb21n.graft_base(parent)
        pb21n.graft_prior(parent)
        after = pb21n._state_dict_sha256(parent.state_dict())
        self.assertEqual(before, after)


class HazardRefinementContractTests(unittest.TestCase):
    def test_optimizer_step_count_exact(self):
        module = pb21n.HazardPath(prior=False)
        fold_index = np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64)
        fold_index[: pb21n.FOLD_ROWS] = 0
        fold_index[pb21n.FOLD_ROWS:] = 1
        rng = np.random.default_rng(0)
        beliefs = rng.standard_normal(
            (pb21n.TOTAL_ROOTS, pb21n.WIDTH), dtype=np.float64
        )
        targets = (rng.random((pb21n.TOTAL_ROOTS, 5)) < 0.3).astype(np.float64)
        prior = rng.integers(0, 5, size=pb21n.TOTAL_ROOTS)
        # One fold only (fold=0 -> fit set = fold_index != 0 = 1536 rows).
        # Short schedule override via monkeypatch to keep the test fast.
        saved = (pb21n.REFINEMENT_PASSES, pb21n.REFINEMENT_ROOT_BATCH_SIZE)
        pb21n.REFINEMENT_PASSES = 1
        pb21n.REFINEMENT_ROOT_BATCH_SIZE = 24
        try:
            record = pb21n.train_hazard_fold(
                module, beliefs, targets, None,
                fold_index=fold_index, fold=0,
            )
        finally:
            pb21n.REFINEMENT_PASSES, pb21n.REFINEMENT_ROOT_BATCH_SIZE = saved
        # fit set = complement fold = FOLD_ROWS rows
        n = pb21n.FOLD_ROWS
        expected = int(np.ceil(n / 24)) * 1
        self.assertEqual(record["optimizer_steps"], expected)
        self.assertEqual(record["root_count"], n)
        self.assertEqual(len(record["pass_records"]), 1)

    def test_training_deterministic_two_runs_identical(self):
        fold_index = np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64)
        fold_index[: pb21n.FOLD_ROWS] = 0
        fold_index[pb21n.FOLD_ROWS:] = 1
        rng = np.random.default_rng(1)
        beliefs = rng.standard_normal(
            (pb21n.TOTAL_ROOTS, pb21n.WIDTH), dtype=np.float64
        )
        targets = (rng.random((pb21n.TOTAL_ROOTS, 5)) < 0.3).astype(np.float64)
        saved = (pb21n.REFINEMENT_PASSES, pb21n.REFINEMENT_ROOT_BATCH_SIZE)
        pb21n.REFINEMENT_PASSES = 1
        pb21n.REFINEMENT_ROOT_BATCH_SIZE = 24
        try:
            m1 = pb21n.graft_base(_shared_parent())
            m2 = pb21n.graft_base(_shared_parent())
            pb21n.train_hazard_fold(
                m1, beliefs, targets, None, fold_index=fold_index, fold=0)
            pb21n.train_hazard_fold(
                m2, beliefs, targets, None, fold_index=fold_index, fold=0)
        finally:
            pb21n.REFINEMENT_PASSES, pb21n.REFINEMENT_ROOT_BATCH_SIZE = saved
        for (n1, t1), (n2, t2) in zip(
            m1.named_parameters(), m2.named_parameters()
        ):
            self.assertTrue(torch.equal(t1, t2), n1)

    def test_rejects_misaligned_arrays(self):
        module = pb21n.HazardPath(prior=False)
        fold_index = np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64)
        fold_index[: pb21n.FOLD_ROWS] = 0
        fold_index[pb21n.FOLD_ROWS:] = 1
        # beliefs has the wrong row count -> must raise before any training.
        with self.assertRaises(ValueError):
            pb21n.train_hazard_fold(
                module,
                np.zeros((pb21n.FOLD_ROWS, pb21n.WIDTH)),
                np.zeros((pb21n.FOLD_ROWS, 5)),
                None, fold_index=fold_index, fold=0,
            )

    def test_crossfit_oof_slots_zero_defined_complement_and_deterministic(self):
        """Regression (PB21N trial #1 false-negative determinism): every OOF
        slot must be fully defined (complement half zero-initialized), so a
        whole-array byte comparison between two re-derivations is meaningful
        instead of reading uninitialized memory."""
        fold_index = np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64)
        fold_index[: pb21n.FOLD_ROWS] = 0
        fold_index[pb21n.FOLD_ROWS:] = 1
        rng = np.random.default_rng(3)
        beliefs = rng.standard_normal(
            (pb21n.TOTAL_ROOTS, pb21n.WIDTH), dtype=np.float64
        )
        targets = (rng.random((pb21n.TOTAL_ROOTS, 5)) < 0.3).astype(np.float64)
        prior = rng.integers(0, 5, size=pb21n.TOTAL_ROOTS)
        ev = pb21n.CALEvidence(
            beliefs=beliefs, targets=targets, applied=np.zeros(
                pb21n.TOTAL_ROOTS, dtype=np.int64),
            prior_applied=prior,
            episode_ordinal=np.repeat(
                np.arange(256, dtype=np.int64), 12),
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
            r1, p1, _ = pb21n.crossfit_arms(_shared_parent(), ev)
            r2, p2, _ = pb21n.crossfit_arms(_shared_parent(), ev)
        finally:
            pb21n.REFINEMENT_PASSES, pb21n.REFINEMENT_ROOT_BATCH_SIZE = saved
        # 1. every slot fully finite (no uninitialized/NaN complement)
        for tab in (r1, p1):
            self.assertTrue(np.isfinite(tab).all())
        # 2. complement half is the defined 0.0
        for tab in (r1, p1):
            for f in range(2):
                self.assertTrue(
                    np.allclose(tab[f][fold_index != f], 0.0)
                )
        # 3. whole-array byte-exact reproducibility (the determinism check
        #    is now meaningful end-to-end)
        self.assertTrue(np.array_equal(r1, r2))
        self.assertTrue(np.array_equal(p1, p2))


class BootstrapGeometryTests(unittest.TestCase):
    def test_fold_slice_ordinal_offset_safe(self):
        # Fold-1 slice has ordinals 128..255 (offset, not 0-based); the
        # bootstrap must not NaN on empty groups.
        n = pb21n.FOLD_ROWS
        per_root = np.random.default_rng(2).standard_normal((n, 2))
        n_eps = n // pb21n.ROOTS_PER_EPISODE
        ordinal = np.repeat(
            np.arange(128, 128 + n_eps, dtype=np.int64),
            pb21n.ROOTS_PER_EPISODE,
        )
        self.assertEqual(ordinal.shape, (n,))
        self.assertEqual(ordinal.min(), 128)
        self.assertEqual(ordinal.max(), 128 + n_eps - 1)
        draws = pb21n._episode_cluster_bootstrap(
            per_root, ordinal, draws=64, seed=99
        )
        self.assertEqual(draws.shape, (64, 2))
        self.assertTrue(np.isfinite(draws).all())

    def test_geometry_drift_rejected(self):
        ordinal = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)  # uneven
        per_root = np.zeros((7, 2))
        with self.assertRaises(ValueError):
            pb21n._episode_cluster_bootstrap(per_root, ordinal, draws=4,
                                             seed=1)


class GateMetricTests(unittest.TestCase):
    def test_factual_mask_selects_applied_action_only(self):
        applied = np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)
        mask = pb21n._factual_mask(applied)
        self.assertEqual(mask.sum(), len(applied))
        for i, a in enumerate(applied):
            self.assertEqual(int(mask[i, a]), 1)
            for o in range(5):
                if o != a:
                    self.assertEqual(int(mask[i, o]), 0)

    def test_factual_losses_per_root(self):
        targets = np.zeros((4, 5), dtype=np.float64)
        applied = np.array([0, 1, 2, 3], dtype=np.int64)
        probs = np.full((4, 5), 0.5)
        losses = pb21n._factual_losses(targets, probs, applied)
        self.assertEqual(losses.shape, (4, 2))
        # single factual action: BCE at p=0.5, t=0 = log(2); Brier = 0.25
        self.assertTrue(np.allclose(losses, [np.log(2.0), 0.25],
                                    atol=1.0e-9))

    def test_allclose_helper(self):
        x = np.array([-2.0, 0.0, 2.0])
        self.assertTrue(np.allclose(pb21n._sigmoid(x),
                                    np.array([0.119203, 0.5, 0.880797]),
                                    atol=1.0e-5))


class PublishContractTests(unittest.TestCase):
    def test_create_only_publish(self):
        ev_cls = pb21n.CALEvidence
        fake_ev = ev_cls(
            beliefs=np.zeros((pb21n.TOTAL_ROOTS, pb21n.WIDTH)),
            targets=np.zeros((pb21n.TOTAL_ROOTS, 5)),
            applied=np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64),
            prior_applied=np.zeros(pb21n.TOTAL_ROOTS, dtype=np.int64),
            episode_ordinal=np.repeat(np.arange(256, dtype=np.int64), 12),
            root_ids=np.zeros(pb21n.TOTAL_ROOTS, dtype="S64"),
            base_raw_logits=np.zeros((pb21n.TOTAL_ROOTS, 5)),
            fold_index=np.repeat(np.arange(2), pb21n.FOLD_ROWS),
            episode_permutation_sha256="0" * 64,
            partition_manifest_sha256="1" * 64,
            rows=pb21n.TOTAL_ROOTS,
        )
        table = lambda: np.zeros((2, pb21n.TOTAL_ROOTS, 5))
        with tempfile.TemporaryDirectory() as tmp:
            old = pb21n.EVIDENCE_DIR
            pb21n.EVIDENCE_DIR = Path(tmp)
            try:
                result_doc = {"status": "completed", "wall_seconds": 1.0}
                pb21n._publish(
                    result_doc, fake_ev, table(), table(), table(),
                    table(), table(),
                )
                res = Path(tmp) / f"{pb21n.RUN_ID}.json"
                att = Path(tmp) / f"{pb21n.RUN_ID}.attempt.json"
                reg = Path(tmp) / f"{pb21n.RUN_ID}.registration.json"
                ev = Path(tmp) / f"{pb21n.RUN_ID}.evidence.npz"
                self.assertTrue(res.exists() and att.exists())
                self.assertTrue(reg.exists() and ev.exists())
                payload = res.read_text(encoding="utf-8").rstrip("\n")
                recorded = json.loads(att.read_text(encoding="utf-8"))
                self.assertEqual(recorded["result_sha256"],
                                 pb21n._sha256_bytes(payload.encode()))
                self.assertFalse(recorded["retry_allowed"])
                with self.assertRaises(RuntimeError):
                    pb21n._publish(
                        result_doc, fake_ev, table(), table(), table(),
                        table(), table(),
                    )
            finally:
                pb21n.EVIDENCE_DIR = old


if __name__ == "__main__":
    unittest.main()
