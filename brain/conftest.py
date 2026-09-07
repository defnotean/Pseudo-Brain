import sys
from pathlib import Path

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
