"""Constitution CI runner — machine-readable results + human summary.

Usage:
  python brain/scripts/run_constitution_ci.py            # smoke (fast)
  python brain/scripts/run_constitution_ci.py --full     # release subset
  python brain/scripts/run_constitution_ci.py --json out.json
Exit code 0 iff all selected contracts pass.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[1]


def run_subset(full: bool) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BRAIN_ROOT / "src")
    target = "tests.test_constitution_ci.ConstitutionRelease" if full else \
        "tests.test_constitution_ci.ConstitutionSmoke"
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", target, "-v"],
        capture_output=True, text=True, cwd=str(BRAIN_ROOT), env=env)
    return proc.returncode, proc.stderr + proc.stdout


def parse_results(output: str) -> list[dict]:
    """Parse unittest -v output: 'test_x (module.Class.test_x)\\ndoc ... ok'"""
    import re
    results = []
    pattern = re.compile(
        r"^(test_\w+) \(([\w.]+)\)\s*\n(.*?)\.\.\. (ok|FAIL|ERROR|skipped)",
        re.M | re.S)
    for match in pattern.finditer(output):
        test, cls_path, _doc, status = match.groups()
        cls = cls_path.split(".")[-2] if "." in cls_path else cls_path
        status_map = {"ok": "pass", "FAIL": "fail", "ERROR": "error",
                      "skipped": "skipped"}
        results.append({"test": test, "class": cls,
                        "status": status_map.get(status, status)})
    # dedupe smoke/release overlap keeping worst status per test+class
    seen: dict[tuple, dict] = {}
    for r in results:
        key = (r["class"], r["test"])
        if key not in seen or r["status"] != "pass":
            seen[key] = r
    return list(seen.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="release subset")
    ap.add_argument("--json", type=str, default=None, help="write JSON report")
    args = ap.parse_args()

    t0 = time.time()
    rc, output = run_subset(args.full)
    duration = time.time() - t0
    results = parse_results(output)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = [r for r in results if r["status"] != "pass"]

    report = {
        "suite": "pseudo-brain-architectural-constitution",
        "subset": "release" if args.full else "smoke",
        "passed": passed,
        "failed": len(failed),
        "duration_s": round(duration, 2),
        "contracts": results,
        "verdict": "CONSTITUTION_HOLD" if rc == 0 else "CONSTITUTION_VIOLATED",
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)

    print("=" * 60)
    print(f"PSEUDO-BRAIN CONSTITUTION CI ({report['subset']})")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["test"]):
        mark = "PASS" if r["status"] == "pass" else r["status"].upper()
        print(f"  [{mark:>4}] {r['test']}")
    print("-" * 60)
    print(f"{passed} passed, {len(failed)} failed in {duration:.1f}s")
    print(f"VERDICT: {report['verdict']}")
    for r in failed:
        contract = r["test"].split("_")[1] if "_" in r["test"] else "?"
        print(f"\nBROKEN CONTRACT {contract}: {r['test']}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
