"""Isolated entry for the create-once RCQ-v3 registration builder.

Windows PowerShell strips double quotes from ``python -c`` payloads, so the
operator wrapper invokes this file with ``python -I`` instead of an inline
bootstrap. The helper only rewrites ``sys.path`` / ``sys.argv`` and then
executes the target-blind builder; it does not open TEST data.
"""

from __future__ import annotations

import runpy
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit(
            "usage: run_rcq_v3_registration.py <source_root> <repo_root> <config>"
        )
    source, root, config = argv[1], argv[2], argv[3]
    sys.path.insert(0, source)
    sys.argv = [
        "irene_brain.evaluation.rcq_v3_registration",
        "--training-release-root",
        root,
        "--config",
        config,
    ]
    runpy.run_module(
        "irene_brain.evaluation.rcq_v3_registration",
        run_name="__main__",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
