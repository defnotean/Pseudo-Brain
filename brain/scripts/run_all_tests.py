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

test_files = sorted(tests_root.glob("test_*.py"))
total = len(test_files)
print(f"Running {total} play-safe test modules...")

failures = []
for i, tf in enumerate(test_files, 1):
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(tests_root),
        "-t",
        str(brain_root),
        "-p",
        tf.name,
    ]
    res = subprocess.run(cmd, cwd=str(brain_root), env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[{i}/{total}] FAIL: {tf.name}")
        print(res.stdout)
        print(res.stderr)
        failures.append(tf.name)
    else:
        print(f"[{i}/{total}] PASS: {tf.name}")

if failures:
    print(f"\n{len(failures)} failed modules: {failures}")
    sys.exit(1)
else:
    print(f"\nAll {total} test modules passed successfully!")
    sys.exit(0)
