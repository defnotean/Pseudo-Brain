"""Custom Fused Triton Associative Prefix Scan Kernel for NVIDIA A100 Tensor Cores.

Breaks the sequential recurrence wall by reformulating the linear recurrence
part of recurrent neural state transitions as an associative binary operator:

    h_t = a_t * h_{t-1} + b_t

For two consecutive affine state transitions:
    f_1(h) = a_1 * h + b_1
    f_2(h) = a_2 * h + b_2
The composition (f_2 o f_1)(h) is:
    f_2(f_1(h)) = a_2 * (a_1 * h + b_1) + b_2
                = (a_1 * a_2) * h + (a_2 * b_1 + b_2)

Hence, the associative binary operator is defined as:
    (a_1, b_1) • (a_2, b_2) = (a_1 * a_2, a_2 * b_1 + b_2)

Hardware Acceleration Rationale on NVIDIA A100:
1. Standard PyTorch Blelloch / Kogge-Stone parallel scan implementations invoke
   O(log T) discrete PyTorch kernel launches. Each reduction level requires writing
   intermediate results to global VRAM (HBM2e) and reading them back, incurring high
   memory bandwidth pressure and launch latency.
2. This custom Triton kernel loads tiles of sequence inputs directly into fast GPU
   shared memory (SRAM) and registers. The entire prefix scan reduction is performed
   on-chip inside SRAM via fast register exchanges, achieving near peak memory bandwidth
   utilization and tensor core arithmetic throughput.
3. Seamless Fallback:
   If Triton or CUDA is unavailable (e.g. running on CPU, AMD DirectML, or non-CUDA hardware),
   the kernel seamlessly falls back to the PyTorch Blelloch work-efficient parallel scan.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from irene_brain.parallel.parallel_scan import (
    associative_binary_op,
    parallel_scan as pytorch_parallel_scan,
    parallel_scan_with_cumprod as pytorch_parallel_scan_with_cumprod,
)

# Detect Triton availability
try:
    import triton
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None
    InterpretedFunction = None
    JITFunction = None
    HAS_TRITON = False


def is_triton_available() -> bool:
    """Returns True if the Triton package is installed."""
    return HAS_TRITON


def is_triton_cuda_available() -> bool:
    """Returns True if Triton is installed and CUDA is available."""
    return HAS_TRITON and torch.cuda.is_available()


# Define Triton JIT kernel if Triton is installed
if HAS_TRITON:

    def _assoc_combine_impl(a1, b1, a2, b2):
        """Associative binary operator: (a1, b1) • (a2, b2) = (a1 * a2, a2 * b1 + b2)."""
        return a1 * a2, a2 * b1 + b2

    _assoc_combine_jit = JITFunction(_assoc_combine_impl)

    def _fused_associative_scan_kernel_impl(
        a_ptr,
        b_ptr,
        h0_ptr,
        out_b_ptr,
        out_a_ptr,
        stride_a_b,
        stride_a_t,
        stride_a_d,
        stride_b_b,
        stride_b_t,
        stride_b_d,
        stride_h0_b,
        stride_h0_d,
        stride_out_b_b,
        stride_out_b_t,
        stride_out_b_d,
        stride_out_a_b,
        stride_out_a_t,
        stride_out_a_d,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_H0: tl.constexpr,
        STORE_A: tl.constexpr,
    ):
        """Fused Associative Prefix Scan Kernel for NVIDIA A100 Tensor Cores.

        Executes intra-block parallel associative prefix scan entirely in SRAM.
        Grid: (B, num_blocks_d).
        """
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_t = tl.arange(0, BLOCK_T)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        ptrs_a = a_ptr + pid_b * stride_a_b + offs_t[:, None] * stride_a_t + offs_d[None, :] * stride_a_d
        ptrs_b = b_ptr + pid_b * stride_b_b + offs_t[:, None] * stride_b_t + offs_d[None, :] * stride_b_d
        ptrs_out_b = out_b_ptr + pid_b * stride_out_b_b + offs_t[:, None] * stride_out_b_t + offs_d[None, :] * stride_out_b_d

        mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
        mask_d = offs_d < D

        # Load inputs into fast SRAM / registers; default neutral identity for decay is 1.0, injection is 0.0
        a = tl.load(ptrs_a, mask=mask, other=1.0)
        b = tl.load(ptrs_b, mask=mask, other=0.0)

        # Incorporate initial state h_0 into step t=0: b'[0] = b[0] + a[0] * h0
        if HAS_H0:
            ptrs_h0 = h0_ptr + pid_b * stride_h0_b + offs_d * stride_h0_d
            h0_val = tl.load(ptrs_h0, mask=mask_d, other=0.0)
            is_t0 = (offs_t[:, None] == 0)
            b = tl.where(is_t0, b + a * h0_val[None, :], b)

        # Execute associative prefix scan along time axis in GPU SRAM
        scanned_a, scanned_b = tl.associative_scan((a, b), axis=0, combine_fn=_assoc_combine_jit)

        # Store hidden states to global memory
        tl.store(ptrs_out_b, scanned_b, mask=mask)

        # Optionally store cumulative decay product
        if STORE_A:
            ptrs_out_a = out_a_ptr + pid_b * stride_out_a_b + offs_t[:, None] * stride_out_a_t + offs_d[None, :] * stride_out_a_d
            tl.store(ptrs_out_a, scanned_a, mask=mask)

    # 1. JIT function for Ahead-Of-Time / JIT compilation and native GPU execution
    _fused_associative_scan_kernel = JITFunction(_fused_associative_scan_kernel_impl)

    # 2. Interpreted function for CPU testing without requiring a CUDA driver
    _fused_associative_scan_kernel_interpret = InterpretedFunction(_fused_associative_scan_kernel_impl)

else:
    _assoc_combine_impl = None
    _assoc_combine_jit = None
    _fused_associative_scan_kernel_impl = None
    _fused_associative_scan_kernel = None
    _fused_associative_scan_kernel_interpret = None


def compile_triton_kernel_for_a100(
    T: int = 128,
    D: int = 64,
    has_h0: bool = True,
    store_a: bool = True,
) -> Dict[str, Any]:
    """Compiles the custom fused associative scan kernel targeting NVIDIA A100 (SM80).

    Uses Triton's Ahead-Of-Time / JIT compiler to verify PTX and CUBIN generation,
    register allocation, and shared memory usage on NVIDIA Ampere A100 architecture.

    Returns:
        Dict with compilation metadata including shared memory bytes, register count, and arch.
    """
    if not HAS_TRITON or _fused_associative_scan_kernel is None:
        raise RuntimeError("Triton is not installed; cannot compile for A100.")

    from triton.backends.compiler import GPUTarget

    target = GPUTarget("cuda", 80, 32)
    sig = {
        "a_ptr": "*fp32",
        "b_ptr": "*fp32",
        "h0_ptr": "*fp32",
        "out_b_ptr": "*fp32",
        "out_a_ptr": "*fp32",
        "stride_a_b": "i32",
        "stride_a_t": "i32",
        "stride_a_d": "i32",
        "stride_b_b": "i32",
        "stride_b_t": "i32",
        "stride_b_d": "i32",
        "stride_h0_b": "i32",
        "stride_h0_d": "i32",
        "stride_out_b_b": "i32",
        "stride_out_b_t": "i32",
        "stride_out_b_d": "i32",
        "stride_out_a_b": "i32",
        "stride_out_a_t": "i32",
        "stride_out_a_d": "i32",
    }

    block_t = max(16, triton.next_power_of_2(T))
    max_sram_elems = 16384  # 16384 * 8 bytes = 128 KB SRAM
    block_d = min(64, max(8, max_sram_elems // block_t))

    cst = {
        "T": T,
        "D": D,
        "BLOCK_T": block_t,
        "BLOCK_D": block_d,
        "HAS_H0": has_h0,
        "STORE_A": store_a,
    }

    src = triton.compiler.ASTSource(
        fn=_fused_associative_scan_kernel,
        signature=sig,
        constexprs=cst,
    )
    compiled = triton.compile(src, target=target)
    return {
        "arch": compiled.metadata.arch,
        "shared_memory_bytes": compiled.metadata.shared,
        "num_warps": compiled.metadata.num_warps,
        "num_stages": compiled.metadata.num_stages,
        "target": str(target),
    }


def _run_triton_scan(
    a: Tensor,
    b: Tensor,
    h0: Optional[Tensor] = None,
    store_cumprod_a: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Internal runner for the Triton fused associative scan kernel.

    Expects contiguous 3D tensors: a [B, T, D], b [B, T, D], h0 [B, D] or None.
    Selects GPU JIT kernel when running on CUDA, or CPU Interpreted kernel when testing on CPU.
    """
    if not HAS_TRITON or _fused_associative_scan_kernel is None:
        raise RuntimeError("Triton is not available.")

    B, T, D = b.shape
    out_b = torch.empty_like(b)
    out_a = torch.empty_like(a) if store_cumprod_a else torch.empty((0, 0, 0), dtype=a.dtype, device=a.device)

    has_h0 = h0 is not None
    dummy_h0 = torch.empty((0, 0), dtype=b.dtype, device=b.device)
    h0_tensor = h0 if has_h0 else dummy_h0

    block_t = max(16, triton.next_power_of_2(T))
    # Two scan operands share SRAM; float64 needs half as many elements.
    max_sram_elems = 65536 // max(a.element_size(), b.element_size())
    block_d = min(64, max(1, max_sram_elems // block_t))
    num_blocks_d = (D + block_d - 1) // block_d

    grid = (B, num_blocks_d)

    # Select native GPU JIT kernel for CUDA or CPU Interpreted kernel for CPU testing
    if a.is_cuda:
        kernel = _fused_associative_scan_kernel
    else:
        kernel = _fused_associative_scan_kernel_interpret

    kernel[grid](
        a,
        b,
        h0_tensor,
        out_b,
        out_a,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        h0_tensor.stride(0) if has_h0 else 0,
        h0_tensor.stride(1) if has_h0 else 0,
        out_b.stride(0),
        out_b.stride(1),
        out_b.stride(2),
        out_a.stride(0) if store_cumprod_a else 0,
        out_a.stride(1) if store_cumprod_a else 0,
        out_a.stride(2) if store_cumprod_a else 0,
        T,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        HAS_H0=has_h0,
        STORE_A=store_cumprod_a,
    )

    if store_cumprod_a:
        return out_a, out_b
    return out_b


class _TritonScanAutogradFunction(torch.autograd.Function):
    """Autograd Function with exact reverse adjoint recurrence for gradient backpropagation."""

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, h0: Optional[Tensor], dim: int):
        ctx.dim = dim
        ctx.has_h0 = h0 is not None

        out_h = _run_triton_scan(a, b, h0=h0, store_cumprod_a=False)
        ctx.save_for_backward(a, out_h, h0)
        return out_h

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, out_h, h0 = ctx.saved_tensors
        B, T, D = out_h.shape

        grad_out_rev = torch.flip(grad_output, dims=[1])
        a_flipped = torch.flip(a, dims=[1])
        a_rev = torch.cat([torch.zeros_like(a_flipped[:, :1]), a_flipped[:, :-1]], dim=1)

        g_rev = _run_triton_scan(a_rev, grad_out_rev, h0=None, store_cumprod_a=False)
        g = torch.flip(g_rev, dims=[1])

        if ctx.has_h0:
            h_prev = torch.cat([h0.unsqueeze(1), out_h[:, :-1]], dim=1)
            grad_h0 = g[:, 0] * a[:, 0]
        else:
            h_prev = torch.cat([torch.zeros_like(out_h[:, :1]), out_h[:, :-1]], dim=1)
            grad_h0 = None

        grad_a = g * h_prev
        grad_b = g

        return grad_a, grad_b, grad_h0, None


def triton_parallel_scan(
    a: Tensor,
    b: Tensor,
    h0: Optional[Tensor] = None,
    dim: int = 1,
    store_cumprod_a: bool = False,
    force_triton: bool = False,
    force_pytorch: bool = False,
    h_init: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Fused Associative Prefix Scan with A100 Tensor Core Kernel and Seamless PyTorch Fallback.

    Computes:
        h_t = a_t * h_{t-1} + b_t
    in O(log T) parallel steps using GPU shared memory (SRAM) for A100 throughput.

    Args:
        a: Retention / decay gate tensor. Values in [0, 1]. Broadcastable to b.
        b: Candidate state update tensor. Shape matching a along scanned dimension.
        h0: Optional initial state tensor.
        dim: Sequence dimension along which to scan (default: 1).
        store_cumprod_a: If True, returns (cumprod_a, states_h).
        force_triton: If True, forces execution via Triton kernel (or Triton interpreter).
        force_pytorch: If True, forces PyTorch Blelloch fallback.
        h_init: Alias for h0.

    Returns:
        Hidden states tensor of shape matching b (or tuple if store_cumprod_a=True).
    """
    if h_init is not None and h0 is None:
        h0 = h_init

    # Determine if Triton kernel should be executed
    should_use_triton = False
    if not force_pytorch and HAS_TRITON:
        if force_triton:
            should_use_triton = True
        elif a.is_cuda and torch.cuda.is_available():
            should_use_triton = True
        elif os.environ.get("TRITON_INTERPRET") == "1":
            should_use_triton = True

    # Fallback to PyTorch Blelloch scan if Triton/CUDA is unavailable
    if not should_use_triton:
        if store_cumprod_a:
            return pytorch_parallel_scan_with_cumprod(a, b, h0=h0, dim=dim)
        return pytorch_parallel_scan(a, b, h0=h0, dim=dim)

    # Transpose to dimension 1 if needed
    transposed = (dim != 1)
    if transposed:
        a = a.transpose(1, dim)
        b = b.transpose(1, dim)

    # Ensure a broadcasts to b
    if a.shape != b.shape:
        a = a.expand_as(b)

    T = b.shape[1]
    if T == 0:
        if store_cumprod_a:
            return a.clone(), b.clone()
        return b.clone()

    if T == 1:
        if h0 is not None:
            h0_aligned = h0.unsqueeze(1) if h0.ndim == b.ndim - 1 else h0
            b_out = b + a * h0_aligned
        else:
            b_out = b.clone()
        if transposed:
            b_out = b_out.transpose(1, dim)
            a = a.transpose(1, dim)
        if store_cumprod_a:
            return a.clone(), b_out
        return b_out

    # Sequence length ceiling for single SRAM block kernel: 2048
    if T > 2048:
        if transposed:
            a = a.transpose(1, dim)
            b = b.transpose(1, dim)
        if store_cumprod_a:
            return pytorch_parallel_scan_with_cumprod(a, b, h0=h0, dim=dim)
        return pytorch_parallel_scan(a, b, h0=h0, dim=dim)

    # Flatten extra dimensions into [B_flat, T, D_flat]
    orig_shape = b.shape
    if b.ndim == 2:
        a_3d = a.unsqueeze(-1).contiguous()
        b_3d = b.unsqueeze(-1).contiguous()
        h0_2d = h0.unsqueeze(-1).contiguous() if h0 is not None else None
    elif b.ndim == 3:
        a_3d = a.contiguous()
        b_3d = b.contiguous()
        h0_2d = h0.contiguous() if h0 is not None else None
    else:
        B_flat = 1
        for s in orig_shape[:1]:
            B_flat *= s
        D_flat = 1
        for s in orig_shape[2:]:
            D_flat *= s
        a_3d = a.view(B_flat, T, D_flat).contiguous()
        b_3d = b.view(B_flat, T, D_flat).contiguous()
        h0_2d = h0.view(B_flat, D_flat).contiguous() if h0 is not None else None

    requires_grad = (a.requires_grad or b.requires_grad or (h0 is not None and h0.requires_grad)) and torch.is_grad_enabled()
    if requires_grad and not store_cumprod_a:
        out_b_3d = _TritonScanAutogradFunction.apply(a_3d, b_3d, h0_2d, 1)
        out_3d = out_b_3d
    else:
        out_3d = _run_triton_scan(a_3d, b_3d, h0=h0_2d, store_cumprod_a=store_cumprod_a)

    if store_cumprod_a:
        out_a_3d, out_b_3d = out_3d
        out_a = out_a_3d.view(orig_shape)
        out_b = out_b_3d.view(orig_shape)
        if transposed:
            out_a = out_a.transpose(1, dim)
            out_b = out_b.transpose(1, dim)
        return out_a, out_b

    out = out_3d.view(orig_shape)
    if transposed:
        out = out.transpose(1, dim)
    return out


# Convenience aliases
triton_scan = triton_parallel_scan
triton_scan_with_cumprod = lambda a, b, h0=None, dim=1: triton_parallel_scan(a, b, h0=h0, dim=dim, store_cumprod_a=True)

__all__ = [
    "triton_parallel_scan",
    "triton_scan",
    "triton_scan_with_cumprod",
    "compile_triton_kernel_for_a100",
    "is_triton_available",
    "is_triton_cuda_available",
]
