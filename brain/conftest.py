import os
import sys
from pathlib import Path

# Local verification must never fall through to DirectML on Windows.
os.environ["PSEUDO_BRAIN_CPU_ONLY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

_root = Path(__file__).resolve().parent
for _p in [
    _root / "src",
    _root / "experiments",
    _root / "experiments" / "memory_benchmark",
    _root / "experiments" / "semantic_benchmark",
    _root / "experiments" / "agent_benchmarks",
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
