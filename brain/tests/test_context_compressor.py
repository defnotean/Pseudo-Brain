"""Proof that context_compressor GUARANTEES fit even when Spark is offline and a
single atomic artifact overflows the entire budget.

Run: py -3.11 brain/tests/test_context_compressor.py
Constraints (AGENTS.md): CPU-only, CUDA-hidden, no network services.
"""

from __future__ import annotations

import sys
import os

# Make the brain package importable without install.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "brain", "src"))

from irene_brain.runtime.context_compressor import (  # noqa: E402
    ContextStore,
    compress,
    estimate_tokens,
    SparkConfig,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# Spark forced offline via an unreachable URL (proves graceful degradation).
OFFLINE_CFG = SparkConfig(base_url="http://127.0.0.1:1/v1", timeout_s=1)


def test_passthrough_small():
    c = compress("hello world", max_tokens=100)
    check("T0 passthrough fits", c.token_estimate <= 100)
    check("T0 no info loss", c.text == "hello world", c.text)


def test_single_oversized_artifact_offline():
    """The pathological case: one tool output bigger than the whole window,
    with Spark DOWN. Guarantee must still hold via T3/T4."""
    big = ("x" * 4000 + "\n") * 200  # ~800KB of text ~ 200k tokens
    MAX = 4000
    c = compress(big, max_tokens=MAX, spark_cfg=OFFLINE_CFG, allow_images=True)
    check("single artifact fits offline (<=MAX)", c.token_estimate <= MAX,
          f"got {c.token_estimate}")
    check("guaranteed_fit flag set", c.guaranteed_fit)
    check("degraded gracefully (not T2)", c.tier_used != "T2-spark", c.tier_used)
    # Either text truncation or image rasterization must have bounded it.
    check("used T3 or T4 fallback", c.tier_used in ("T3-truncation", "T4-image", "T4-image+vision"),
          c.tier_used)
    if c.images:
        check("image ingestion <= MAX", c.ingestion_tokens <= MAX, f"ingest={c.ingestion_tokens}")


def test_store_never_exceeds_budget():
    store = ContextStore(max_tokens=2000, spark_cfg=OFFLINE_CFG)
    for i in range(50):
        store.add("user", "This is a fairly long message number " + str(i) * 50
                  + " with repeated content to grow the transcript quickly.")
    c = store.ensure_fit(tmp_dir=None)
    check("store fits after 50 turns offline", c.token_estimate <= 2000,
          f"got {c.token_estimate}")
    check("store guaranteed flag", c.guaranteed_fit)


def test_zero_budget():
    c = compress("anything at all", max_tokens=1)
    check("zero/one budget never errors", c.token_estimate <= 1, str(c.token_estimate))


def test_images_always_bounded():
    """Even if text is 10x the budget, image ingestion tokens are capped <= budget."""
    huge = "decision: keep the model frozen.\n" * 5000
    MAX = 3000
    c = compress(huge, max_tokens=MAX, spark_cfg=OFFLINE_CFG, allow_images=True)
    if c.images:
        check("image ingestion <= MAX", c.ingestion_tokens <= MAX,
              f"ingest={c.ingestion_tokens}")
    else:
        check("no images but still fits", c.token_estimate <= MAX, str(c.token_estimate))


if __name__ == "__main__":
    print("== context_compressor guarantee tests (Spark offline, CPU-only) ==")
    test_passthrough_small()
    test_single_oversized_artifact_offline()
    test_store_never_exceeds_budget()
    test_zero_budget()
    test_images_always_bounded()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
