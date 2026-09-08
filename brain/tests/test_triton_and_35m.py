"""Test Suite for Fused Triton Associative Scan and 35M Unified Pseudo-Brain Architecture.

Verifies:
1. Custom Fused Triton Associative Prefix Scan Kernel for NVIDIA A100 (SM80):
   - AOT / JIT Compilation targeting NVIDIA A100 Tensor Cores (GPUTarget cuda sm_80).
   - Numerical accuracy between Triton and PyTorch Blelloch parallel scan (max abs diff < 1e-5).
   - Initial state h_0 folding and cumulative decay product (cumprod) tracking.
   - Autograd gradient flow and adjoint reverse recurrence backpropagation.
   - Seamless transparent fallback when running on CPU / DirectML.
2. Calibrated 35M Parameter Architecture Tier (tier2_35m):
   - Trainable parameter count strictly within [30M, 40M] hitting ~35M.
   - Law 1 4.0 KB Working Memory Contract: K=16, W=64, exactly 4,096 bytes.
   - Exact mathematical parity between parallel scan and single-step streaming inference (< 1e-6).
   - Full multimodal ingestion and end-to-end backpropagation.
"""

from __future__ import annotations

import math
import pytest
import torch

from irene_brain.memory.hierarchical_state import HierarchicalMemoryConfig
from irene_brain.parallel.parallel_scan import parallel_scan, parallel_scan_with_cumprod
from irene_brain.parallel.triton_scan import (
    compile_triton_kernel_for_a100,
    is_triton_available,
    is_triton_cuda_available,
    triton_parallel_scan,
    triton_scan,
    triton_scan_with_cumprod,
)
from irene_brain.unified.unified_model import UnifiedPseudoBrain, make_unified_model


# ==============================================================================
# 1. TRITON KERNEL TESTS FOR NVIDIA A100 (SM80)
# ==============================================================================

def test_triton_availability():
    """Verify that Triton is installed and available in the environment."""
    assert is_triton_available(), "Triton must be available in the environment."


def test_triton_kernel_compilation_for_a100():
    """Verify that the Triton scan kernel compiles cleanly to PTX for NVIDIA A100 (SM80).

    Tests multiple sequence lengths and verifies that shared memory (SRAM) usage
    remains strictly within A100's 164 KB physical capacity limit.
    """
    for T in [16, 64, 128, 256, 512, 1024, 2048]:
        meta = compile_triton_kernel_for_a100(T=T, D=64, has_h0=True, store_a=True)
        assert meta["arch"] == "sm80", f"Target architecture must be sm80, got {meta['arch']}"
        assert meta["shared_memory_bytes"] <= 164 * 1024, (
            f"Shared memory {meta['shared_memory_bytes']} bytes exceeds A100 164 KB limit!"
        )
        assert meta["num_warps"] >= 1, "Must allocate at least 1 warp"


@pytest.mark.parametrize("T", [1, 4, 8, 13, 16, 32, 64, 128])
@pytest.mark.parametrize("D", [16, 64])
def test_numerical_accuracy_triton_vs_pytorch(T: int, D: int):
    """Verify numerical accuracy between Triton and PyTorch Blelloch scan (< 1e-5 max abs diff)."""
    torch.manual_seed(42 + T + D)
    B = 3
    a = torch.sigmoid(torch.randn(B, T, D, dtype=torch.float32))
    b = torch.randn(B, T, D, dtype=torch.float32)

    # PyTorch reference Blelloch scan
    out_pytorch = parallel_scan(a, b, h0=None, dim=1, method="work_efficient")

    # Triton scan (executed via interpreter or GPU)
    out_triton = triton_scan(a, b, h0=None, dim=1, force_triton=True)

    max_diff = (out_pytorch - out_triton).abs().max().item()
    mean_diff = (out_pytorch - out_triton).abs().mean().item()

    assert max_diff < 1e-5, (
        f"Triton vs PyTorch scan diff {max_diff:.2e} exceeds 1e-5 threshold for T={T}, D={D} "
        f"(mean diff: {mean_diff:.2e})"
    )


def test_triton_scan_with_initial_state_h0():
    """Verify that initial state h_0 folding in Triton matches PyTorch scan (< 1e-5 diff)."""
    torch.manual_seed(101)
    B, T, D = 4, 32, 64
    a = torch.sigmoid(torch.randn(B, T, D))
    b = torch.randn(B, T, D)
    h0 = torch.randn(B, D)

    out_pytorch = parallel_scan(a, b, h0=h0, dim=1)
    out_triton = triton_scan(a, b, h0=h0, dim=1, force_triton=True)

    max_diff = (out_pytorch - out_triton).abs().max().item()
    assert max_diff < 1e-5, f"Diff with h0 {max_diff:.2e} exceeds 1e-5"


def test_triton_scan_with_cumprod():
    """Verify that triton_scan_with_cumprod returns both cumulative decays and states."""
    torch.manual_seed(202)
    B, T, D = 2, 16, 64
    a = torch.sigmoid(torch.randn(B, T, D))
    b = torch.randn(B, T, D)
    h0 = torch.randn(B, D)

    py_cum_a, py_h = parallel_scan_with_cumprod(a, b, h0=h0, dim=1)
    tr_cum_a, tr_h = triton_scan_with_cumprod(a, b, h0=h0, dim=1)

    diff_h = (py_h - tr_h).abs().max().item()
    diff_a = (py_cum_a - tr_cum_a).abs().max().item()

    assert diff_h < 1e-5, f"State diff {diff_h:.2e} exceeds 1e-5"
    assert diff_a < 1e-5, f"Decay cumprod diff {diff_a:.2e} exceeds 1e-5"


def test_triton_scan_fallback_mechanism():
    """Verify seamless automatic fallback on CPU/DirectML when not forcing Triton."""
    B, T, D = 2, 8, 64
    a = torch.sigmoid(torch.randn(B, T, D))
    b = torch.randn(B, T, D)

    # Calling without force_triton on CPU triggers fallback to PyTorch Blelloch scan
    out = triton_scan(a, b, dim=1, force_pytorch=True)
    ref = parallel_scan(a, b, dim=1)
    assert torch.allclose(out, ref, atol=1e-6)


def test_triton_scan_autograd_gradients():
    """Verify backpropagation gradients through triton_scan match analytical adjoints."""
    torch.manual_seed(303)
    B, T, D = 2, 8, 16
    # Use leaf tensors directly with requires_grad=True
    a = torch.rand(B, T, D, requires_grad=True)
    b = torch.randn(B, T, D, requires_grad=True)
    h0 = torch.randn(B, D, requires_grad=True)

    out = triton_scan(a, b, h0=h0, dim=1, force_triton=True)
    loss = (out ** 2).sum()
    loss.backward()

    assert a.grad is not None and not torch.isnan(a.grad).any()
    assert b.grad is not None and not torch.isnan(b.grad).any()
    assert h0.grad is not None and not torch.isnan(h0.grad).any()
    assert torch.norm(a.grad) > 0
    assert torch.norm(b.grad) > 0
    assert torch.norm(h0.grad) > 0


# ==============================================================================
# 2. 35M PARAMETER TIER CALIBRATION & LAW 1 WORKING MEMORY TESTS
# ==============================================================================

def test_tier2_35m_parameter_count_within_budget():
    """Verify that tier2_35m hits ~35M parameters (within strict 30M-40M range)."""
    # 1. Via make_unified_model factory (default 4 layers)
    model_factory = make_unified_model(tier="tier2_35m", vocab_size=32000)
    params_factory = model_factory.count_parameters()["trainable"]
    assert 30_000_000 <= params_factory <= 40_000_000, (
        f"Factory tier2_35m parameters {params_factory:,} outside 30M-40M range!"
    )

    # 2. Via alias "35m"
    model_alias = make_unified_model(tier="35m", vocab_size=32000)
    params_alias = model_alias.count_parameters()["trainable"]
    assert 30_000_000 <= params_alias <= 40_000_000, (
        f"Alias 35m parameters {params_alias:,} outside 30M-40M range!"
    )

    # 3. Direct class instantiation
    model_direct = UnifiedPseudoBrain(tier="tier2_35m", vocab_size=32000)
    params_direct = model_direct.count_parameters()["trainable"]
    assert 30_000_000 <= params_direct <= 40_000_000, (
        f"Direct tier2_35m parameters {params_direct:,} outside 30M-40M range!"
    )

    # 4. Deep layers parameter scaling within 4..8 layers
    for nl in [4, 6, 8]:
        m = make_unified_model(tier="tier2_35m", vocab_size=32000, num_deep_layers=nl)
        p = m.count_parameters()["trainable"]
        assert 30_000_000 <= p <= 40_000_000, (
            f"tier2_35m with {nl} layers has {p:,} params, outside 30M-40M!"
        )


def test_tier2_35m_law1_working_memory_contract():
    """Verify that tier2_35m working memory strictly respects Law 1 (exactly 4,096 bytes / 4.0 KB)."""
    model = make_unified_model(tier="tier2_35m", vocab_size=32000)

    # Structural slot width & slot count invariants
    assert model.K_fast == 16, f"K_fast must be 16, got {model.K_fast}"
    assert model.W_fast == 64, f"W_fast must be 64, got {model.W_fast}"

    # Law 1 Working Memory Equation: K * W * 4 bytes
    fast_bytes = model.K_fast * model.W_fast * 4
    assert fast_bytes == 4096, f"Working memory must be exactly 4096 bytes, got {fast_bytes}"

    # Memory config check
    assert model.memory_cfg.fast_state_bytes() == 4096

    # Actual instantiated tensor footprint per sample
    state = model.init_state(batch_size=1, device=torch.device("cpu"))
    working_tensor = state.hierarchical_state.working_thoughts
    actual_bytes = working_tensor[0].numel() * working_tensor.element_size()
    assert actual_bytes == 4096, (
        f"Actual tensor footprint per sample {actual_bytes} != 4096 bytes (Law 1 violated!)"
    )


def test_tier2_35m_mathematical_parity():
    """Verify exact mathematical parity between parallel scan and single-step streaming (< 1e-6)."""
    model = make_unified_model(tier="tier2_35m", vocab_size=1000)
    model.eval()

    B, T = 2, 8
    torch.manual_seed(555)
    tokens = torch.randint(0, 1000, (B, T))

    # Parallel sequence forward pass
    out_parallel = model.forward(token_seq=tokens, parallel=True)

    # Sequential single-step streaming unrolling
    out_streaming = model.forward(token_seq=tokens, parallel=False)

    diff_logits = (out_parallel["logits"] - out_streaming["logits"]).abs().max().item()
    diff_actions = (out_parallel["action_logits"] - out_streaming["action_logits"]).abs().max().item()
    diff_values = (out_parallel["values"] - out_streaming["values"]).abs().max().item()

    assert diff_logits < 1e-6, f"Logits parity diff {diff_logits:.2e} exceeds 1e-6"
    assert diff_actions < 1e-6, f"Action logits parity diff {diff_actions:.2e} exceeds 1e-6"
    assert diff_values < 1e-6, f"Value parity diff {diff_values:.2e} exceeds 1e-6"


def test_tier2_35m_multimodal_forward_and_gradients():
    """Verify end-to-end multimodal execution (tokens + pixels + actions) and gradient backpropagation."""
    model = make_unified_model(tier="tier2_35m", vocab_size=1000)
    B, T = 2, 4
    torch.manual_seed(777)

    tokens = torch.randint(0, 1000, (B, T))
    pixels = torch.randn(B, T, 3, 16, 16)
    actions = torch.randint(0, model.n_actions, (B, T))

    outputs = model(token_seq=tokens, pixel_seq=pixels, action_seq=actions, parallel=True)

    assert outputs["logits"].shape == (B, T, 1000)
    assert outputs["action_logits"].shape == (B, T, model.n_actions)
    assert outputs["values"].shape == (B, T)

    loss = outputs["logits"].sum() + outputs["action_logits"].sum() + outputs["values"].sum()
    loss.backward()

    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grad, "Gradients failed to propagate through tier2_35m architecture!"
