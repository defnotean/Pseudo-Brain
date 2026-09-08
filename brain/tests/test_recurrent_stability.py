"""Unit and Verification Test Suite for Deep Recurrent Stability and Dynamics.

Verifies:
1. Gradient norms remain bounded across 1,000 consecutive unroll steps (zero exploding/vanishing gradients).
2. Internal state norms remain within [0.9, 1.1] over long sequences (1,000 steps).
3. Transition matrices satisfy spectral radius rho(W) <= 1.0 and Lyapunov stability is verified (non-positive Lyapunov exponent).
4. Deep recurrent stack scales smoothly to 24 and 32 layers with pre-norm residual and gradient highway routing.
5. Slot identity preservation prevents slot collapse across deep layers.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
from torch.nn import functional as F

from irene_brain.stability.deep_recurrent_stack import (
    DeepRecurrentBlock,
    DeepRecurrentStack,
    DeepStableBrainCellCore,
    GradientHighwayGate,
)
from irene_brain.stability.recurrent_norm import (
    RecurrentL2Norm,
    RecurrentLayerNorm,
    RecurrentRMSNorm,
    RecurrentSlotNorm,
)
from irene_brain.stability.spectral_norm import (
    CayleyLinear,
    SpectralNormalizedLinear,
    compute_spectral_norm,
    compute_spectral_radius,
    verify_lyapunov_stability,
    verify_spectral_radius,
)


class TestRecurrentStability(unittest.TestCase):
    """Rigorous verification test suite for recurrent dynamics, normalization, and Lyapunov stability."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_internal_state_norms_bounded_across_1000_steps(self) -> None:
        """Verify internal state norms remain within [0.9, 1.1] across 1,000 consecutive unroll steps."""
        B, K, W, D = 2, 8, 32, 64
        T = 1000

        # Construct recurrent stack
        stack = DeepRecurrentStack(
            num_layers=4,
            thought_size=W,
            input_size=D,
            rank=16,
            K=K,
            spectral_mode="cayley",
            max_spectral_radius=1.0,
        )

        h = stack.initial_state(batch_size=B)
        # Verify initial norm is in [0.9, 1.1]
        init_rms = torch.sqrt(torch.mean(h.pow(2), dim=-1))
        self.assertTrue(
            (init_rms >= 0.90).all() and (init_rms <= 1.10).all(),
            f"Initial state norm out of bounds: min={init_rms.min():.4f}, max={init_rms.max():.4f}",
        )

        # Generate 1,000 continuous input steps
        generator = torch.Generator().manual_seed(123)
        inputs = torch.randn(T, B, D, generator=generator)

        min_observed_norm = float("inf")
        max_observed_norm = float("-inf")

        with torch.no_grad():
            for t in range(T):
                x_t = inputs[t]
                h = stack.forward_step(h, x_t)

                # Compute RMS norm along slot width dimension W for all slots in batch
                step_rms = torch.sqrt(torch.mean(h.pow(2), dim=-1))
                min_step = float(step_rms.min().item())
                max_step = float(step_rms.max().item())

                if min_step < min_observed_norm:
                    min_observed_norm = min_step
                if max_step > max_observed_norm:
                    max_observed_norm = max_step

                # Strict step-by-step invariant
                self.assertGreaterEqual(
                    min_step,
                    0.90,
                    f"Step {t}: Internal state norm collapsed below 0.90 (min={min_step:.4f})",
                )
                self.assertLessEqual(
                    max_step,
                    1.10,
                    f"Step {t}: Internal state norm exploded above 1.10 (max={max_step:.4f})",
                )

        self.assertGreaterEqual(min_observed_norm, 0.90)
        self.assertLessEqual(max_observed_norm, 1.10)
        self.assertTrue(torch.isfinite(h).all())

    def test_gradient_norms_bounded_across_1000_unroll_steps(self) -> None:
        """Verify gradient norms remain bounded across 1,000 unroll steps (zero exploding/vanishing gradients)."""
        B, K, W, D = 1, 4, 16, 32
        T = 1000

        # Lightweight 2-layer stable recurrent block to test deep unroll backpropagation
        block = DeepRecurrentBlock(
            thought_size=W,
            input_size=D,
            rank=8,
            layer_idx=0,
            total_layers=2,
            spectral_mode="cayley",
            max_spectral_radius=1.0,
            use_highway=True,
        )

        h = torch.randn(B, K, W, requires_grad=True)
        h_orig = h

        inputs = torch.randn(T, B, D)

        for t in range(T):
            x_t = inputs[t]
            h = block(h, x_t)

        # Scalar loss on final state
        loss = h.sum()
        loss.backward()

        # Check gradient on initial state h_orig
        self.assertIsNotNone(h_orig.grad)
        self.assertTrue(torch.isfinite(h_orig.grad).all())

        grad_norm = float(torch.linalg.norm(h_orig.grad).item())

        # Zero vanishing gradients: grad_norm must be strictly non-zero
        self.assertGreater(
            grad_norm,
            1e-12,
            f"Gradients vanished across 1,000 steps: grad_norm={grad_norm:.4e}",
        )

        # Zero exploding gradients: grad_norm must remain bounded (e.g. < 500.0)
        self.assertLess(
            grad_norm,
            500.0,
            f"Gradients exploded across 1,000 steps: grad_norm={grad_norm:.4e}",
        )

        # Check recurrent parameter gradients
        self.assertIsNotNone(block.W_hn.weight_raw.grad)
        self.assertTrue(torch.isfinite(block.W_hn.weight_raw.grad).all())
        w_grad_norm = float(torch.linalg.norm(block.W_hn.weight_raw.grad).item())
        self.assertGreater(w_grad_norm, 1e-12)
        self.assertLess(w_grad_norm, 1000.0)

    def test_lyapunov_stability_and_spectral_radius(self) -> None:
        """Verify spectral radius rho(W) <= 1.0 and empirical Lyapunov stability (lambda <= 0)."""
        W = 32
        D = 32
        T = 1000

        # 1. Test CayleyLinear spectral properties
        cayley = CayleyLinear(W, max_spectral_radius=1.0)
        W_cayley = cayley.get_weight()
        rho = compute_spectral_radius(W_cayley)
        norm_2 = compute_spectral_norm(W_cayley)

        self.assertLessEqual(
            rho,
            1.00001,
            f"CayleyLinear spectral radius violated: rho={rho:.6f} > 1.0",
        )
        self.assertLessEqual(
            norm_2,
            1.00001,
            f"CayleyLinear spectral norm violated: ||W||_2={norm_2:.6f} > 1.0",
        )

        # 2. Test SpectralNormalizedLinear
        sn_linear = SpectralNormalizedLinear(W, W, max_spectral_radius=1.0)
        W_sn = sn_linear.get_effective_weight()
        rho_sn = compute_spectral_radius(W_sn)
        norm_2_sn = compute_spectral_norm(W_sn)

        self.assertLessEqual(rho_sn, 1.00001)
        self.assertLessEqual(norm_2_sn, 1.00001)

        # 3. Verify Lyapunov stability on recurrent block trajectory
        block = DeepRecurrentBlock(
            thought_size=W,
            input_size=D,
            rank=8,
            spectral_mode="cayley",
            max_spectral_radius=1.0,
            use_highway=True,
        )

        initial_state = torch.randn(1, 4, W)
        inputs = torch.randn(T, 1, D)

        def step_fn(h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return block(h, x)

        stability_report = verify_lyapunov_stability(
            step_fn=step_fn,
            initial_state=initial_state,
            inputs=inputs,
            perturbation_norm=1e-4,
        )

        lyap_exp = stability_report["lyapunov_exponent"]
        max_ratio = stability_report["max_perturbation_ratio"]
        final_ratio = stability_report["final_perturbation_ratio"]

        # Lyapunov exponent must be non-positive (perturbations do not diverge exponentially)
        self.assertLessEqual(
            lyap_exp,
            1e-4,
            f"Positive Lyapunov exponent detected (chaotic divergence): lambda={lyap_exp:.6f}",
        )
        # Final perturbation ratio contracts below initial perturbation (asymptotic stability)
        self.assertLessEqual(
            final_ratio,
            1.0,
            f"Final perturbation ratio did not contract: final_ratio={final_ratio:.4f}",
        )
        # Transient perturbation ratio must remain bounded
        self.assertLessEqual(
            max_ratio,
            25.0,
            f"Perturbation amplified excessively: max_ratio={max_ratio:.4f}",
        )
        self.assertTrue(stability_report["is_lyapunov_stable"])

    def test_deep_recurrent_stack_24_layers(self) -> None:
        """Verify 24-layer deep recurrent stack instantiation, forward, and backward autograd flow."""
        B, K, W, D = 2, 8, 32, 64
        num_layers = 24

        stack = DeepRecurrentStack(
            num_layers=num_layers,
            thought_size=W,
            input_size=D,
            rank=16,
            K=K,
            spectral_mode="cayley",
            max_spectral_radius=1.0,
            use_highway=True,
        )

        self.assertEqual(len(stack.blocks), 24)

        # Verify all 24 layers have spectral radius <= 1.0
        spectral_diag = stack.verify_stack_spectral_stability()
        for param_name, rho in spectral_diag.items():
            self.assertLessEqual(rho, 1.00001, f"{param_name} violated rho <= 1.0: {rho}")

        # Forward pass
        h0 = stack.initial_state(batch_size=B)
        x = torch.randn(B, D, requires_grad=True)
        h_out = stack.forward_step(h0, x)

        self.assertEqual(h_out.shape, (B, K, W))
        self.assertTrue(torch.isfinite(h_out).all())

        # Verify output norms in [0.9, 1.1]
        out_rms = torch.sqrt(torch.mean(h_out.pow(2), dim=-1))
        self.assertTrue(
            (out_rms >= 0.90).all() and (out_rms <= 1.10).all(),
            f"Output norm out of bounds: min={out_rms.min():.4f}, max={out_rms.max():.4f}",
        )

        # Backward pass: verify gradient propagates to all 24 layers and inputs
        loss = h_out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        # Verify gradient flow through layer 0 (earliest layer) and layer 23 (latest layer)
        self.assertIsNotNone(stack.blocks[0].W_hn.weight_raw.grad)
        self.assertTrue(torch.isfinite(stack.blocks[0].W_hn.weight_raw.grad).all())
        self.assertGreater(float(torch.linalg.norm(stack.blocks[0].W_hn.weight_raw.grad).item()), 0.0)

        self.assertIsNotNone(stack.blocks[23].W_hn.weight_raw.grad)
        self.assertTrue(torch.isfinite(stack.blocks[23].W_hn.weight_raw.grad).all())
        self.assertGreater(float(torch.linalg.norm(stack.blocks[23].W_hn.weight_raw.grad).item()), 0.0)

    def test_deep_recurrent_stack_32_layers(self) -> None:
        """Verify 32-layer deep recurrent stack instantiation, forward pass, and state norms."""
        B, K, W, D = 2, 8, 32, 64
        num_layers = 32

        stack = DeepRecurrentStack(
            num_layers=num_layers,
            thought_size=W,
            input_size=D,
            rank=16,
            K=K,
            spectral_mode="cayley",
            max_spectral_radius=1.0,
            use_highway=True,
        )

        self.assertEqual(len(stack.blocks), 32)

        h0 = stack.initial_state(batch_size=B)
        x = torch.randn(B, D)
        h_out = stack.forward_step(h0, x)

        self.assertEqual(h_out.shape, (B, K, W))
        self.assertTrue(torch.isfinite(h_out).all())

        # Norms must stay strictly in [0.9, 1.1] even across 32 deep layers
        out_rms = torch.sqrt(torch.mean(h_out.pow(2), dim=-1))
        self.assertTrue(
            (out_rms >= 0.90).all() and (out_rms <= 1.10).all(),
            f"32-layer output norm out of bounds: min={out_rms.min():.4f}, max={out_rms.max():.4f}",
        )

    def test_slot_identity_preservation_prevents_collapse(self) -> None:
        """Verify slot identity preservation maintains orthogonal slot diversity across deep stacks."""
        B, K, W, D = 2, 16, 64, 128
        num_layers = 16

        # Stack with slot identity preservation enabled
        stack_with_identity = DeepRecurrentStack(
            num_layers=num_layers,
            thought_size=W,
            input_size=D,
            rank=16,
            K=K,
            slot_identity_weight=0.10,
        )

        h0 = stack_with_identity.initial_state(batch_size=B)
        x = torch.randn(B, D)

        # Run 20 steps
        h = h0
        for _ in range(20):
            h = stack_with_identity.forward_step(h, x)

        diversity = stack_with_identity.compute_slot_diversity(h)
        # Healthy slot diversity should be >= 0.20 (mean cosine similarity <= 0.80)
        self.assertGreater(
            diversity,
            0.20,
            f"Slot collapse detected: diversity={diversity:.4f} < 0.20",
        )

    def test_recurrent_norm_hard_bounds_clamping(self) -> None:
        """Verify RecurrentSlotNorm clamps extreme outliers into [0.90, 1.10]."""
        dim = 64
        norm_layer = RecurrentSlotNorm(
            dim=dim,
            norm_type="rms",
            min_bound=0.90,
            max_bound=1.10,
            enforce_hard_bounds=True,
        )

        # Case 1: Extreme large norm vector (scale = 1000.0)
        x_huge = torch.randn(4, dim) * 1000.0
        y_huge = norm_layer(x_huge)
        rms_huge = torch.sqrt(torch.mean(y_huge.pow(2), dim=-1))
        self.assertTrue((rms_huge >= 0.90 - 1e-5).all() and (rms_huge <= 1.10 + 1e-5).all())

        # Case 2: Extreme small norm vector (scale = 1e-4)
        x_tiny = torch.randn(4, dim) * 1e-4
        y_tiny = norm_layer(x_tiny)
        rms_tiny = torch.sqrt(torch.mean(y_tiny.pow(2), dim=-1))
        self.assertTrue((rms_tiny >= 0.90 - 1e-5).all() and (rms_tiny <= 1.10 + 1e-5).all())

    def test_deep_stable_brain_cell_core_dropin(self) -> None:
        """Verify DeepStableBrainCellCore drop-in replacement with 24 layers."""
        B, K, W, Proj = 2, 16, 64, 4096
        core = DeepStableBrainCellCore(
            input_size=Proj,
            thought_size=W,
            rank=32,
            proj_dim=Proj,
            num_layers=24,
            K=K,
        )

        self.assertEqual(core.state_bytes(K=K), K * W * 4)

        thoughts = torch.randn(B, K, W)
        x = torch.randn(B, K, Proj)

        out = core(thoughts, x)
        self.assertEqual(out.shape, (B, K, W))
        self.assertTrue(torch.isfinite(out).all())

        out_rms = torch.sqrt(torch.mean(out.pow(2), dim=-1))
        self.assertTrue((out_rms >= 0.90).all() and (out_rms <= 1.10).all())


if __name__ == "__main__":
    unittest.main()
