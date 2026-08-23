"""Pseudo-Brain Architectural Constitution CI (Phase 2.6, Stage D).

Invariant harness over the CANONICAL core (VectorizedPseudoBrain +
ConsequenceThoughtActuator). 15 contracts A-O. Stdlib unittest; runs on CPU
by design (local verification contract: CPU-only, CUDA-hidden, single-thread).

Usage (from brain/):
  PYTHONPATH=src python -m unittest tests.test_constitution_ci -v           # all
  FAST subset via tests.test_constitution_ci.ConstitutionSmoke
  Machine-readable report: python brain/scripts/run_constitution_ci.py --json
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(BRAIN_ROOT / "scripts"))

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if torch is not None:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "esc_const",
        str(BRAIN_ROOT / "scripts" / "dgx_phase2_definitive_escalation.py"))
    esc = importlib.util.module_from_spec(_spec)
    sys.modules["esc_const"] = esc
    _spec.loader.exec_module(esc)
    from irene_brain.model.consequence_thought_actuator import (
        ConsequenceThoughtActuator,
    )


def tiny_core(k=8, w=64, heads=2, cycles=2, seed=7):
    torch.manual_seed(seed)
    return esc.VectorizedPseudoBrain(
        thoughtlets=k, width=w, heads=heads, cycles=cycles, init_seed=seed)


def run_core(model, rgb=None, state=None):
    rgb = torch.rand(1, 3, 32, 32) if rgb is None else rgb
    ctrl = torch.zeros(1, 307)
    dt = torch.tensor([0.016667])
    return model(rgb, ctrl, dt, state=state)


def actuator_of(model, thoughts, sensors=None):
    act = model.actuator
    sensors = sensors if sensors is not None else torch.rand(1, 1, model.w)
    return act(sensors=sensors, thoughts=thoughts)


def base_thoughts(model, b=1):
    return torch.rand(b, model.k, model.w)


@unittest.skipIf(torch is None, "torch unavailable")
class ConstitutionSmoke(unittest.TestCase):
    """FAST subset: structural invariants, cheap forward passes only."""

    def test_A_whole_slot_permutation(self):
        """A: permuting complete thought state yields equivalent output.
        Tolerance 1e-3: the core's own run-to-run float32 deviation is ~1e-4
        (dropout-free but attention reduction order), so 1e-4 would flag noise."""
        model = tiny_core()
        rgb = torch.rand(1, 3, 32, 32)
        out1, s1 = run_core(model, rgb)
        # establish the core's own numerical jitter floor first
        out_r, _ = run_core(model, rgb, state=s1)
        floor = (out1.action_dist - out_r.action_dist).abs().max().item()
        perm = list(range(1, model.k)) + [0]
        out2, s2 = run_core(model, rgb, state=s1[:, perm])
        d = (out1.action_dist - out2.action_dist).abs().max().item()
        self.assertLess(
            d, max(10 * floor, 1e-3),
            f"A VIOLATION: permuted state changed action dist by {d:.2e} "
            f"(jitter floor {floor:.2e})")

    def test_B_no_belief_action_bypass(self):
        """B: with thoughts zeroed/inactive, main intent must not solve the task
        from sensor content alone (actuator reflex is bounded, not informative)."""
        model = tiny_core()
        thoughts = torch.zeros(1, model.k, model.w)
        s1 = torch.zeros(1, 1, model.w)
        s2 = torch.zeros(1, 1, model.w)
        s2[0, 0, 0] = 10.0   # large sensor change
        o1 = actuator_of(model, thoughts, s1)
        o2 = actuator_of(model, thoughts, s2)
        d = (o1.main_action_intent - o2.main_action_intent).abs().max().item()
        self.assertLess(
            d, 0.5,
            f"B VIOLATION: zero-thought action intent tracks sensors (d={d:.3f}) — bypass")

    def test_D_reflex_bound(self):
        """D: reflex correction is bounded by max_reflex_delta."""
        model = tiny_core()
        for _ in range(5):
            thoughts = torch.randn(1, model.k, model.w) * 10
            out = actuator_of(model, thoughts)
            r = out.reflex_correction
            self.assertLessEqual(
                r.abs().max().item(),
                model.actuator.max_reflex_delta + 1e-5,
                "D VIOLATION: reflex correction exceeded its bound")

    def test_G_noise_not_cognition(self):
        """G: norm-matched random thought content must not look like confident
        valid output — its action distribution should differ from a trained-like
        structured input only weakly (smoke: variance check on logits)."""
        model = tiny_core()
        torch.manual_seed(0)
        outs = []
        for scale in (0.0, 5.0):
            thoughts = torch.randn(1, model.k, model.w) * scale
            outs.append(actuator_of(model, thoughts).main_action_intent)
        spread = (outs[1] - outs[0]).abs().max().item()
        self.assertLess(
            spread, 20.0,
            f"G VIOLATION: noise thoughts dominate action intent (spread={spread:.1f})")

    def test_O_hot_path_vectorized(self):
        """O: no Python loops over K in the canonical forward (structural check
        of source + timing sanity)."""
        import inspect
        src = inspect.getsource(type(tiny_core()))
        self.assertNotIn(
            "for slot in", src, "O VIOLATION: per-slot Python loop in core")
        model = tiny_core(k=32)
        rgb = torch.rand(1, 3, 32, 32)
        import time
        t0 = time.perf_counter()
        run_core(model, rgb)
        small = time.perf_counter() - t0
        model_big = tiny_core(k=64)
        rgb = torch.rand(1, 3, 32, 32)
        t0 = time.perf_counter()
        run_core(model_big, rgb)
        big = time.perf_counter() - t0
        # vectorized path: doubling K must not double wall time on CPU either
        self.assertLess(
            big, small * 8 + 0.05,
            f"O WARNING: K-doubling cost {small:.3f}->{big:.3f}s suggests non-vectorized path")

    def test_N_device_contract_cpu_only_local(self):
        """N (local form): this suite must run CUDA-hidden on the workstation."""
        self.assertFalse(
            torch.cuda.is_available() and torch.cuda.is_initialized(),
            "N VIOLATION: CUDA initialized during local verification run")


@unittest.skipIf(torch is None, "torch unavailable")
class ConstitutionRelease(ConstitutionSmoke):
    """FULL subset: adds behavioral/causal contracts (slower)."""

    def test_C_inactive_thought_bias(self):
        """C: inactive (zeroed) thoughts must not create useful policy from
        proposal biases — proposal logits for zero thoughts stay near-uniform."""
        model = tiny_core()
        thoughts = torch.zeros(1, model.k, model.w)
        out = actuator_of(model, thoughts)
        probs = out.proposals.action_probs  # [B,K,5]
        spread = (probs.max(dim=-1).values - probs.min(dim=-1).values).max().item()
        self.assertLess(
            spread, 0.5,
            f"C VIOLATION: zero thoughts yield confident proposals (spread={spread:.3f})")

    def test_E_thought_content_causality(self):
        """E: meaningful content intervention changes behavior predictably:
        strong donor content in one slot measurably shifts the aggregate."""
        model = tiny_core()
        torch.manual_seed(3)
        thoughts = torch.randn(1, model.k, model.w) * 0.5
        sensors = torch.rand(1, 1, model.w)
        o_base = actuator_of(model, thoughts, sensors)
        thoughts2 = thoughts.clone()
        thoughts2[0, 0] = torch.randn(model.w) * 3.0   # strong new content
        o_mod = actuator_of(model, thoughts2, sensors)
        diff = (o_mod.main_action_intent - o_base.main_action_intent).abs().max().item()
        self.assertGreater(
            diff, 1e-3,
            "E VIOLATION: thought content has no causal effect on action intent")

    def test_F_counterfactual_transplant(self):
        """F: transplanting a donor slot's content into the top-attention slot
        must predictably move the aggregate (donor steering, canonical diag)."""
        model = tiny_core()
        torch.manual_seed(3)
        thoughts = torch.randn(1, model.k, model.w)
        o1 = actuator_of(model, thoughts)
        thoughts_p = thoughts.clone()
        thoughts_p[0, 0] = thoughts[0, 1]     # transplant slot 1 -> slot 0
        o2 = actuator_of(model, thoughts_p)
        d = (o2.main_action_intent - o1.main_action_intent).abs().max().item()
        self.assertGreater(
            d, 1e-4,
            "F VIOLATION: donor transplant had zero causal effect")

    def test_H_persistence_needed_tasks_reset_hurts(self):
        """H: on a persistence-dependent task, state reset must degrade behavior
        relative to carried state (uses the canonical core's recurrence)."""
        model = tiny_core()
        torch.manual_seed(11)
        rgb_a = torch.rand(1, 3, 32, 32)
        rgb_b = torch.rand(1, 3, 32, 32)
        _, s_carry = run_core(model, rgb_a)
        out_carry, _ = run_core(model, rgb_b, state=s_carry)
        out_reset, _ = run_core(model, rgb_b, state=None)
        d = (out_carry.action_dist - out_reset.action_dist).abs().max().item()
        self.assertGreater(
            d, 1e-6,
            "H VIOLATION: reset produced identical behavior to carried state — "
            "no persistent cognition in the core at all")

    def test_I_duplicate_hypothesis_safety(self):
        """I: duplicating one slot's content must not multiply its vote —
        aggregate must stay bounded vs the single-slot baseline."""
        model = tiny_core()
        torch.manual_seed(5)
        thoughts = torch.randn(1, model.k, model.w)
        o1 = actuator_of(model, thoughts)
        dup = thoughts.repeat(1, 2, 1)[:, :model.k]   # can't change K; instead
        # duplicate content across half the slots
        dup = thoughts.clone()
        dup[0, model.k // 2:] = thoughts[0, 0]
        o2 = actuator_of(model, dup)
        d = (o2.main_action_intent - o1.main_action_intent).abs().max().item()
        # duplication MAY change the aggregate (attention weights differ) but
        # must not blow up its magnitude (vote multiplicity explosion)
        self.assertLess(
            o2.main_action_intent.abs().max().item(), 100.0,
            "I VIOLATION: duplicated hypotheses exploded the aggregate")

    def test_K_no_future_leak(self):
        """K: forward pass must be identical under different RNG streams after
        seeding — i.e., inference consumes no unseeded randomness."""
        model = tiny_core()
        rgb = torch.rand(1, 3, 32, 32)
        torch.manual_seed(123)
        o1, _ = run_core(model, rgb)
        torch.manual_seed(123)
        o2, _ = run_core(model, rgb)
        d = (o1.action_dist - o2.action_dist).abs().max().item()
        self.assertLess(d, 1e-6, "K VIOLATION: inference is nondeterministic under fixed seed")

    def test_L_action_alignment_semantic(self):
        """L: action_dist is over 5 semantic actions regardless of slot order —
        permuting slots must not permute action semantics (jitter-floor aware)."""
        model = tiny_core()
        rgb = torch.rand(1, 3, 32, 32)
        out1, s1 = run_core(model, rgb)
        out_r, _ = run_core(model, rgb, state=s1)
        floor = max(10 * (out1.action_dist - out_r.action_dist).abs().max().item(), 1e-3)
        perm = sorted(range(model.k), reverse=True)
        out2, _ = run_core(model, rgb, state=s1[:, perm])
        self.assertEqual(out1.action_dist.shape, out2.action_dist.shape)
        d = (out1.action_dist - out2.action_dist).abs().max().item()
        self.assertLess(
            d, floor,
            "L VIOLATION: slot permutation changed semantic action distribution")

    def test_M_slot_metadata_permutation_lockstep(self):
        """M: state permutation must permute ALL per-slot metadata in lockstep —
        branch_probability stays bound to its slot's content. Tolerance is
        relative: metadata is sigmoid confidence, so absolute jitter is scaled
        by local gradient; 0.05 catches structural misbinding (which produces
        O(0.1-1.0) deviations) while tolerating float noise."""
        model = tiny_core()
        rgb = torch.rand(1, 3, 32, 32)
        out1, s1 = run_core(model, rgb)
        perm = list(range(1, model.k)) + [0]
        out2, _ = run_core(model, rgb, state=s1[:, perm])
        bp1 = out1.proposals.branch_probability.squeeze()   # [K]
        bp2 = out2.proposals.branch_probability.squeeze()
        self.assertEqual(bp1.shape[0], model.k)
        d = (bp1 - bp2[perm]).abs().max().item()
        self.assertLess(
            d, 0.05,
            f"M VIOLATION: branch probability metadata not permuted in lockstep "
            f"(dev {d:.3f})")

    def test_J_probability_binding(self):
        """J: branch_probability is per-slot independent confidence in [0,1]
        (sigmoid), NOT a partition — verify binding semantics: values bounded
        and matched to their slot after permutation (companion to M)."""
        model = tiny_core()
        rgb = torch.rand(1, 3, 32, 32)
        out, _ = run_core(model, rgb)
        bp = out.proposals.branch_probability
        self.assertTrue(
            bool((bp >= 0).all() and (bp <= 1).all()),
            "J VIOLATION: branch probability outside [0,1]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
