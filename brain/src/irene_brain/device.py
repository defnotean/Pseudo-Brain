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
    if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
        return False
    try:
        import torch_directml  # type: ignore
        if hasattr(torch_directml, "is_available"):
            return bool(torch_directml.is_available())
        return bool(torch_directml.device_count() > 0)
    except (ImportError, Exception):
        return False


def get_directml_device_count() -> int:
    """Return the number of accessible DirectML devices."""
    if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
        return 0
    try:
        import torch_directml  # type: ignore
        return int(torch_directml.device_count())
    except (ImportError, Exception):
        return 0


def get_directml_device_name(device_index: int = 0) -> str:
    """Return friendly name of DirectML device at index."""
    if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
        return "DirectML disabled by CPU-only policy"
    try:
        import torch_directml  # type: ignore
        return str(torch_directml.device_name(device_index)).strip().replace("\x00", "")
    except (ImportError, Exception):
        return f"DirectML Device {device_index}"


def get_best_directml_device_index() -> int:
    """Select the optimal DirectML device index, prioritizing high-performance discrete GPUs.
    
    In multi-GPU environments (e.g. AMD Ryzen APU with integrated Radeon Graphics + dedicated
    AMD Radeon RX 9070 XT), enumerates all DirectML devices and selects the dedicated accelerator.
    """
    if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
        return 0
    try:
        import torch_directml  # type: ignore
        count = torch_directml.device_count()
        if count <= 1:
            return 0

        best_idx = 0
        best_score = -1000

        for idx in range(count):
            name = str(torch_directml.device_name(idx)).strip().replace("\x00", "").lower()
            score = 0

            # Dedicated / Flagship discrete GPU scoring
            if "9070" in name:
                score += 1000
            elif "rx" in name:
                score += 600
            elif "rtx" in name or "geforce" in name or "nvidia" in name:
                score += 500
            elif "radeon" in name:
                score += 200
            elif "arc" in name:
                score += 300

            # Penalize integrated GPUs or virtual display adapters if discrete accelerator exists
            if "graphics" in name and "rx" not in name:
                score -= 100
            if "basic render" in name or "virtual" in name:
                score -= 500

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx
    except Exception as e:
        logger.debug("Failed during DirectML device scoring: %s", e)
        return 0


def get_directml_device(device_index: Optional[int] = None) -> Optional[torch.device]:
    """Return DirectML torch device if available.
    
    Args:
        device_index: Specific DirectML device index (0, 1, ...). If None,
                      automatically discovers and selects the optimal accelerator
                      (e.g., dedicated AMD Radeon RX 9070 XT).
    """
    if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
        return None
    try:
        import torch_directml  # type: ignore
        if device_index is None:
            device_index = get_best_directml_device_index()
        return torch_directml.device(device_index)
    except (ImportError, Exception) as e:
        logger.debug("Failed to acquire DirectML device index %s: %s", device_index, e)
        return None


def resolve_optimal_device(preference: Optional[str] = None) -> Tuple[torch.device, str]:
    """Resolve optimal compute device for Pseudo-Brain models.
    
    Priority order:
    1. Explicit preference ('cuda', 'dml', 'directml', 'dml:0', 'dml:1', 'cpu')
    2. NVIDIA CUDA (if cuda.is_available())
    3. AMD / Intel DirectML (if torch_directml is installed, selecting optimal discrete GPU)
    4. Multi-threaded CPU (default fallback)
    """
    if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
        backend = "cpu" if preference is None or preference.lower().strip() == "cpu" else "cpu_fallback"
        return torch.device("cpu"), backend
    if preference:
        pref = preference.lower().strip()
        if pref.startswith("cuda"):
            if torch.cuda.is_available():
                idx = 0
                if ":" in pref:
                    try:
                        idx = int(pref.split(":")[1])
                    except ValueError:
                        pass
                return torch.device(f"cuda:{idx}"), "cuda"
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu"), "cpu_fallback"
        elif pref in ("dml", "directml", "gpu") or pref.startswith("dml:") or pref.startswith("directml:"):
            device_idx = None
            if ":" in pref:
                try:
                    device_idx = int(pref.split(":")[1])
                except ValueError:
                    device_idx = None
            dml_dev = get_directml_device(device_idx)
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
    if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
        num_threads = 1
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def get_device_telemetry(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Report hardware topology, active device, and memory characteristics."""
    gpu_hw = probe_system_gpus()
    active_dev, backend = resolve_optimal_device(str(device) if device else None)

    telemetry: Dict[str, Any] = {
        "os": gpu_hw.get("os", platform.system()),
        "release": gpu_hw.get("release", platform.release()),
        "active_device": str(active_dev),
        "backend": backend,
        "cuda_available": torch.cuda.is_available(),
        "directml_available": is_directml_available(),
        "cpu_threads": torch.get_num_threads(),
        "system_gpus": gpu_hw.get("controllers", []),
        "has_amd_radeon": gpu_hw.get("has_amd_radeon", False),
        "has_nvidia": gpu_hw.get("has_nvidia", False),
    }

    if is_directml_available():
        try:
            import torch_directml  # type: ignore
            dml_count = torch_directml.device_count()
            telemetry["directml_device_count"] = dml_count
            telemetry["directml_devices"] = [
                {"index": i, "name": str(torch_directml.device_name(i)).strip().replace("\x00", "")}
                for i in range(dml_count)
            ]
            if backend == "directml":
                # Determine device index
                active_idx = getattr(active_dev, "index", None)
                if active_idx is None:
                    dev_str = str(active_dev)
                    if ":" in dev_str:
                        try:
                            active_idx = int(dev_str.split(":")[-1])
                        except ValueError:
                            active_idx = 0
                    else:
                        active_idx = 0
                telemetry["directml_device_index"] = active_idx
                telemetry["directml_device_name"] = str(torch_directml.device_name(active_idx)).strip().replace("\x00", "")
                if hasattr(torch_directml, "has_float64_support"):
                    try:
                        telemetry["directml_float64_support"] = torch_directml.has_float64_support(active_idx)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Failed to query DirectML telemetry details: %s", e)

    if backend == "cuda" and torch.cuda.is_available():
        telemetry["cuda_device_name"] = torch.cuda.get_device_name(0)
        telemetry["cuda_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
    elif gpu_hw.get("has_amd_radeon") and not is_directml_available():
        telemetry["recommendation"] = (
            "Detected AMD Radeon GPU without torch-directml. "
            "To enable local DirectML hardware acceleration, install: pip install torch-directml"
        )

    return telemetry
