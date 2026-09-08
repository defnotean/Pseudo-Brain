"""Parallel Recurrence and Associative Scan Subsystem for Pseudo-Brain."""

from irene_brain.parallel.parallel_scan import (
    associative_binary_op,
    associative_scan_hillis_steele,
    associative_scan_work_efficient,
    parallel_scan,
    parallel_scan_with_cumprod,
    sequential_scan,
)
from irene_brain.parallel.triton_scan import (
    compile_triton_kernel_for_a100,
    is_triton_available,
    is_triton_cuda_available,
    triton_parallel_scan,
    triton_scan,
    triton_scan_with_cumprod,
)
from irene_brain.parallel.fused_cell import (
    FusedBrainCellCore,
    FusedCognitiveState,
    FusedSlotRecurrentBlock,
    ParallelNativeSemanticPseudoBrain,
)

__all__ = [
    "associative_binary_op",
    "associative_scan_hillis_steele",
    "associative_scan_work_efficient",
    "parallel_scan",
    "parallel_scan_with_cumprod",
    "sequential_scan",
    "compile_triton_kernel_for_a100",
    "is_triton_available",
    "is_triton_cuda_available",
    "triton_parallel_scan",
    "triton_scan",
    "triton_scan_with_cumprod",
    "FusedBrainCellCore",
    "FusedCognitiveState",
    "FusedSlotRecurrentBlock",
    "ParallelNativeSemanticPseudoBrain",
]
