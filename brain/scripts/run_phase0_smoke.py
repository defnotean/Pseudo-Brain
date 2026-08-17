from __future__ import annotations

import json
from pathlib import Path

from irene_brain.smoke import run_smoke


def main() -> None:
    brain_root = Path(__file__).resolve().parents[1]
    summary = run_smoke(
        brain_root / "configs" / "experiment" / "phase0-smoke.toml"
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
