"""Hardware telemetry and device resolution utility for Pseudo-Brain.

Supports seamless operation across:
- Local workstations (CPU, optional DirectML)
- DGX Spark (NVIDIA A100/H100)
- Google Colab (NVIDIA T4, V100, A100, L4)
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict, Optional

import torch


def resolve_device(
    device_str: str = "auto",
    require_cuda: bool = False,
) -> torch.device:
    """Resolve compute device with strict validation and clear error reporting.

    Args:
        device_str: 'auto', 'cuda', 'cuda:0', 'cpu', or DirectML 'dml:0'.
        require_cuda: If True and CUDA is unavailable, raises RuntimeError with guidance.

    Returns:
        torch.device instance.
    """
    device_str = device_str.strip().lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if require_cuda:
            raise RuntimeError(
                "CUDA requested or required, but torch.cuda.is_available() is False.\n"
                "If running in Google Colab, ensure GPU runtime is enabled:\n"
                "  Runtime -> Change runtime type -> Hardware accelerator -> T4 / A100 / L4 GPU."
            )
        return torch.device("cpu")

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device_str}' requested, but CUDA is not available on this system.\n"
                f"Available devices: CPU (CUDA available: False)."
            )
        return torch.device(device_str)

    if device_str.startswith("dml:"):
        try:
            import torch_directml  # type: ignore
            idx = int(device_str.split(":")[1])
            return torch_directml.device(idx)
        except Exception as e:
            raise RuntimeError(f"DirectML device requested but unavailable: {e}")

    return torch.device(device_str)


def get_hardware_summary() -> Dict[str, Any]:
    """Inspects available compute hardware and returns structured telemetry."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = None
    cuda_version = torch.version.cuda if hasattr(torch.version, "cuda") else None
    compute_cap = None

    if cuda_avail:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = round(props.total_memory / (1024**3), 2)
            compute_cap = f"{props.major}.{props.minor}"
        except Exception:
            gpu_name = "CUDA Device (query error)"

    return {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "compute_capability": compute_cap,
    }


def format_hardware_summary() -> str:
    """Returns standardized hardware summary string conforming to user specification."""
    info = get_hardware_summary()
    gpu_mem_str = f"{info['gpu_memory_gb']} GB" if info["gpu_memory_gb"] is not None else "N/A"
    gpu_name_str = info["gpu_name"] if info["gpu_name"] is not None else "None"
    cuda_ver_str = info["cuda_version"] if info["cuda_version"] is not None else "N/A"

    return (
        f"GPU: {gpu_name_str}\n"
        f"CUDA available: {info['cuda_available']}\n"
        f"GPU name: {gpu_name_str}\n"
        f"GPU memory: {gpu_mem_str}\n"
        f"PyTorch version: {info['pytorch_version']}\n"
        f"CUDA version: {cuda_ver_str}"
    )


def print_hardware_summary() -> None:
    print("=" * 60)
    print("PSEUDO-BRAIN HARDWARE TELEMETRY")
    print("=" * 60)
    print(format_hardware_summary())
    print("=" * 60)


if __name__ == "__main__":
    print_hardware_summary()
