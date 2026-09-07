"""Hardware Topology & Device Auto-Resolution Subsystem.

Provides cross-platform hardware acceleration discovery for Irene Brain:
1. NVIDIA CUDA (via torch.cuda)
2. AMD / Intel DirectML on Windows (via torch_directml or ONNX DML)
3. Multi-threaded CPU fallback (MKL / OpenMP) with thread pool tuning
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("irene_brain.device")


def probe_system_gpus() -> Dict[str, Any]:
    """Probe system video controllers via OS-level hardware discovery."""
    gpu_info = {
        "os": platform.system(),
        "release": platform.release(),
        "controllers": [],
        "has_amd_radeon": False,
        "has_nvidia": False,
    }

    if platform.system() == "Windows":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -Property Name, AdapterRAM, DriverVersion | ConvertTo-Json"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if proc.returncode == 0 and proc.stdout.strip():
                import json
                try:
                    data = json.loads(proc.stdout)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        name = str(item.get("Name", ""))
                        gpu_info["controllers"].append({
                            "name": name,
                            "ram_bytes": item.get("AdapterRAM", 0),
                            "driver": item.get("DriverVersion", ""),
                        })
                        if "Radeon" in name or "AMD" in name:
                            gpu_info["has_amd_radeon"] = True
                        if "NVIDIA" in name or "GeForce" in name or "RTX" in name:
                            gpu_info["has_nvidia"] = True
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Failed to query Win32_VideoController: %s", e)

    return gpu_info


def is_directml_available() -> bool:
    """Check if torch_directml or native DirectML tensor backend is accessible."""
    try:
        import torch_directml  # type: ignore
        return True
    except ImportError:
        return False


def get_directml_device(device_index: int = 0) -> Optional[torch.device]:
    """Return DirectML torch device if available."""
    try:
        import torch_directml  # type: ignore
        return torch_directml.device(device_index)
    except ImportError:
        return None


def resolve_optimal_device(preference: Optional[str] = None) -> Tuple[torch.device, str]:
    """Resolve optimal compute device for Pseudo-Brain models.
    
    Priority order:
    1. Explicit preference ('cuda', 'dml', 'directml', 'cpu')
    2. NVIDIA CUDA (if cuda.is_available())
    3. AMD / Intel DirectML (if torch_directml is installed)
    4. Multi-threaded CPU (default)
    """
    if preference:
        pref = preference.lower().strip()
        if pref == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda:0"), "cuda"
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu"), "cpu_fallback"
        elif pref in ("dml", "directml"):
            dml_dev = get_directml_device()
            if dml_dev is not None:
                return dml_dev, "directml"
            logger.warning("DirectML requested but torch_directml is not installed; falling back to CPU.")
            return torch.device("cpu"), "cpu_fallback"
        elif pref == "cpu":
            return torch.device("cpu"), "cpu"

    # Automatic selection
    if torch.cuda.is_available():
        return torch.device("cuda:0"), "cuda"

    dml_dev = get_directml_device()
    if dml_dev is not None:
        return dml_dev, "directml"

    return torch.device("cpu"), "cpu"


def configure_cpu_threading(num_threads: Optional[int] = None) -> int:
    """Configure CPU thread pool for deterministic real-time latency."""
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def get_device_telemetry(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Report hardware topology, active device, and memory characteristics."""
    gpu_hw = probe_system_gpus()
    active_dev, backend = resolve_optimal_device(str(device) if device else None)

    telemetry = {
        "active_device": str(active_dev),
        "backend": backend,
        "cuda_available": torch.cuda.is_available(),
        "directml_available": is_directml_available(),
        "cpu_threads": torch.get_num_threads(),
        "system_gpus": gpu_hw.get("controllers", []),
        "has_amd_radeon": gpu_hw.get("has_amd_radeon", False),
        "has_nvidia": gpu_hw.get("has_nvidia", False),
    }

    if backend == "cuda" and torch.cuda.is_available():
        telemetry["cuda_device_name"] = torch.cuda.get_device_name(0)
        telemetry["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
    elif gpu_hw.get("has_amd_radeon") and not is_directml_available():
        telemetry["recommendation"] = (
            "Detected AMD Radeon GPU without torch-directml. "
            "To enable local DirectML hardware acceleration, install: pip install torch-directml"
        )

    return telemetry
