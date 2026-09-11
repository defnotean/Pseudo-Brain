from __future__ import annotations
import os
import pathlib
import subprocess
import sys

brain_root = pathlib.Path(__file__).resolve().parents[1]
tests_root = brain_root / "tests"

env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["CUDA_VISIBLE_DEVICES"] = "-1"
env["PSEUDO_BRAIN_CPU_ONLY"] = "1"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["NUMEXPR_NUM_THREADS"] = "1"
env["VECLIB_MAXIMUM_THREADS"] = "1"
env["BLIS_NUM_THREADS"] = "1"
env["RAYON_NUM_THREADS"] = "1"
env["TOKENIZERS_PARALLELISM"] = "false"
env["HIP_VISIBLE_DEVICES"] = "-1"
env["ROCR_VISIBLE_DEVICES"] = "-1"
env["PYTHONPATH"] = str(brain_root / "src")

if __name__ == "__main__":
    # pytest also collects unittest.TestCase, while unittest silently misses
    # the repository's free-standing pytest functions and parametrized cases.
    # Preserve pytest's nonzero status for collection failures and zero tests.
    command = [sys.executable, "-m", "pytest", str(tests_root), *sys.argv[1:]]
    raise SystemExit(subprocess.run(command, cwd=brain_root, env=env).returncode)
