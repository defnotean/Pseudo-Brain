"""Comprehensive Unit Tests for Parallel Associative Prefix Scan."""

import pytest
import torch

from irene_brain.parallel.parallel_scan import (
    associative_binary_op,
    associative_scan_hillis_steele,
    associative_scan_work_efficient,
    parallel_scan,
    parallel_scan_with_cumprod,
    sequential_scan,
)


def test_associative_binary_operator_associativity():
    """Verify that (x • y) • z == x • (y • z) numerically."""
    torch.manual_seed(42)
    shape = (4, 8)
    a1, b1 = torch.rand(shape), torch.randn(shape)
    a2, b2 = torch.rand(shape), torch.randn(shape)
    a3, b3 = torch.rand(shape), torch.randn(shape)

    # (x • y) • z
    a_xy, b_xy = associative_binary_op(a1, b1, a2, b2)
    a_xyz_left, b_xyz_left = associative_binary_op(a_xy, b_xy, a3, b3)

    # x • (y • z)
    a_yz, b_yz = associative_binary_op(a2, b2, a3, b3)
    a_xyz_right, b_xyz_right = associative_binary_op(a1, b1, a_yz, b_yz)

    assert torch.allclose(a_xyz_left, a_xyz_right, atol=1e-6)
    assert torch.allclose(b_xyz_left, b_xyz_right, atol=1e-6)


@pytest.mark.parametrize("T", [1, 2, 3, 7, 8, 15, 16, 31, 32, 64, 127, 128])
@pytest.mark.parametrize("method", ["work_efficient", "hillis_steele"])
def test_parallel_scan_matches_sequential(T: int, method: str):
    """Verify parallel scan matches sequential loop across odd/even sequence lengths."""
    torch.manual_seed(T)
    B, D = 4, 16
    a = torch.rand(B, T, D) * 0.9 + 0.05
    b = torch.randn(B, T, D)
    h0 = torch.randn(B, D)

    h_seq = sequential_scan(a, b, h0=h0, dim=1)
    h_par = parallel_scan(a, b, h0=h0, dim=1, method=method)

    diff = (h_seq - h_par).abs().max().item()
    assert diff < 1e-5, f"Mismatch at T={T}, method={method}: max diff={diff}"


def test_parallel_scan_without_h0():
    """Verify prefix scan when h0 is None (equivalent to h0=0)."""
    torch.manual_seed(42)
    B, T, D = 2, 20, 8
    a = torch.rand(B, T, D) * 0.8 + 0.1
    b = torch.randn(B, T, D)

    h_seq = sequential_scan(a, b, h0=None)
    h_par = parallel_scan(a, b, h0=None, method="work_efficient")

    assert torch.allclose(h_seq, h_par, atol=1e-5)


def test_parallel_scan_multi_dimensional():
    """Verify scan on 4D tensors [B, T, K, W] (multi-slot Pseudo-Brain shape)."""
    torch.manual_seed(42)
    B, T, K, W = 2, 16, 4, 8
    a = torch.rand(B, T, K, W) * 0.85 + 0.1
    b = torch.randn(B, T, K, W)
    h0 = torch.randn(B, K, W)

    h_seq = sequential_scan(a, b, h0=h0, dim=1)
    h_par = parallel_scan(a, b, h0=h0, dim=1, method="work_efficient")

    assert h_par.shape == (B, T, K, W)
    assert torch.allclose(h_seq, h_par, atol=1e-5)


def test_parallel_scan_autograd_gradients():
    """Verify backward gradient flow for a, b, and h0 matches sequential autograd."""
    torch.manual_seed(42)
    B, T, D = 4, 32, 16

    a_seq = (torch.rand(B, T, D) * 0.8 + 0.1).requires_grad_(True)
    b_seq = torch.randn(B, T, D, requires_grad=True)
    h0_seq = torch.randn(B, D, requires_grad=True)

    a_par = a_seq.detach().clone().requires_grad_(True)
    b_par = b_seq.detach().clone().requires_grad_(True)
    h0_par = h0_seq.detach().clone().requires_grad_(True)

    out_seq = sequential_scan(a_seq, b_seq, h0_seq)
    loss_seq = (out_seq ** 2).sum()
    loss_seq.backward()

    out_par = parallel_scan(a_par, b_par, h0_par, method="work_efficient")
    loss_par = (out_par ** 2).sum()
    loss_par.backward()

    assert torch.allclose(loss_seq, loss_par, atol=1e-5)
    assert torch.allclose(a_seq.grad, a_par.grad, atol=1e-4)
    assert torch.allclose(b_seq.grad, b_par.grad, atol=1e-4)
    assert torch.allclose(h0_seq.grad, h0_par.grad, atol=1e-4)


def test_parallel_scan_dim_transposition():
    """Verify scanning along non-default dimensions."""
    torch.manual_seed(42)
    # Scan along dim=0: shape [T, B, D]
    T, B, D = 12, 3, 6
    a = torch.rand(T, B, D) * 0.9
    b = torch.randn(T, B, D)
    h0 = torch.randn(B, D)

    h_seq = sequential_scan(a, b, h0=h0, dim=0)
    h_par = parallel_scan(a, b, h0=h0, dim=0, method="work_efficient")

    assert h_par.shape == (T, B, D)
    assert torch.allclose(h_seq, h_par, atol=1e-5)


def test_parallel_scan_edge_cases():
    """Verify edge cases: T=0, T=1, broadcastable gates."""
    # T=0
    a0 = torch.empty(2, 0, 4)
    b0 = torch.empty(2, 0, 4)
    out0 = parallel_scan(a0, b0)
    assert out0.shape == (2, 0, 4)

    # T=1
    a1 = torch.rand(2, 1, 4)
    b1 = torch.randn(2, 1, 4)
    h0 = torch.randn(2, 4)
    out1 = parallel_scan(a1, b1, h0=h0)
    expected1 = a1 * h0.unsqueeze(1) + b1
    assert torch.allclose(out1, expected1, atol=1e-6)

    # Scalar gate broadcasting
    a_scalar = torch.full((2, 8, 1), 0.5)
    b_vec = torch.randn(2, 8, 4)
    out_bcast = parallel_scan(a_scalar, b_vec)
    assert out_bcast.shape == (2, 8, 4)


def test_parallel_scan_with_cumprod():
    """Verify parallel_scan_with_cumprod returns matching cumprod and states."""
    torch.manual_seed(42)
    B, T, D = 2, 8, 4
    a = torch.rand(B, T, D) * 0.8 + 0.1
    b = torch.randn(B, T, D)

    cum_a, states = parallel_scan_with_cumprod(a, b, method="work_efficient")
    expected_cum_a = torch.cumprod(a, dim=1)

    assert torch.allclose(cum_a, expected_cum_a, atol=1e-5)
    assert torch.allclose(states, parallel_scan(a, b, method="work_efficient"), atol=1e-5)
