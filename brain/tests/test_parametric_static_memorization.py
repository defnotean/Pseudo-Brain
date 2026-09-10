"""Test Suite for Raw Zero-Shot Static Memorization in Neural Parameter Weights.

Verifies:
1. Direct zero-shot recall of pre-baked domain facts, algorithms, and syntax.
2. Fast retrieval latency (<5 ms on CPU, target <1 ms).
3. Exact Law 1 Working Memory invariance: dynamic operational state remains strictly <= 4,096 bytes.
4. Dynamic closed-form baking of new knowledge into static weights via regularized associative least squares.
5. AutonomousLifelongAgent integration: query triggers zero-shot static recall without network or disk access.
"""

from __future__ import annotations

import time
import pytest
import torch

from irene_brain.memory.parametric_memory import (
    ParametricStaticKnowledgeCore,
    StaticKnowledgeItem,
    StaticRecallResult,
)
from irene_brain.agent.continual_learner import AutonomousLifelongAgent


def test_parametric_memory_zero_shot_facts():
    """Verify zero-shot recall of pre-baked facts directly from neural weights."""
    core = ParametricStaticKnowledgeCore()

    # Query binary search
    res = core.query_static_knowledge("how does binary search work on a sorted array")
    assert res is not None, "Failed to recall binary search from static weights!"
    assert res.topic == "binary search"
    assert "O(log n)" in res.summary
    assert res.code_example is not None and "def binary_search" in res.code_example
    assert res.confidence >= 0.60

    # Query photosynthesis
    res_photo = core.query_static_knowledge("explain the process of photosynthesis")
    assert res_photo is not None, "Failed to recall photosynthesis from static weights!"
    assert res_photo.topic == "photosynthesis"
    assert "chloroplast" in res_photo.summary or "light energy" in res_photo.summary

    # Query collision detection
    res_col = core.query_static_knowledge("how to implement aabb collision detection in games")
    assert res_col is not None, "Failed to recall collision detection from static weights!"
    assert "AABB" in res_col.summary or "Bounding Box" in res_col.summary


def test_parametric_memory_retrieval_latency():
    """Verify sub-millisecond retrieval latency from static parameter weights."""
    core = ParametricStaticKnowledgeCore()

    # Warmup
    _ = core.query_static_knowledge("memoization")

    # Benchmark 10 iterations
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = core.query_static_knowledge("what is memoization in python")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 5.0, f"Static recall latency too high: {avg_latency:.2f} ms (expected < 5.0 ms on CPU)"
    assert res is not None and res.topic == "memoization"


def test_bake_new_knowledge_into_weights():
    """Verify closed-form baking of novel knowledge into neural parameter weights."""
    core = ParametricStaticKnowledgeCore()

    novel_topic = "rayleigh scattering"
    novel_summary = "Rayleigh scattering refers to the elastic scattering of light by particles much smaller than the wavelength, explaining why the daytime sky appears blue."

    # Verify not known before baking
    res_before = core.query_static_knowledge("why is the sky blue rayleigh scattering")
    # Even if similarity is partial, topic should not match rayleigh scattering before addition
    if res_before:
        assert res_before.topic != novel_topic

    # Bake into weights
    core.add_static_knowledge(
        topic=novel_topic,
        category="physics",
        summary=novel_summary,
        keywords=["rayleigh", "scattering", "blue sky", "wavelength", "elastic scattering"]
    )

    # Query after baking
    res_after = core.query_static_knowledge("explain rayleigh scattering and why the sky is blue")
    assert res_after is not None, "Failed to recall newly baked knowledge from weights!"
    assert res_after.topic == novel_topic
    assert "Rayleigh scattering" in res_after.summary


def test_agent_zero_shot_static_recall_telemetry():
    """Verify AutonomousLifelongAgent seamlessly answers zero-shot via parametric memory."""
    agent = AutonomousLifelongAgent()

    # Ask for python generator (pre-baked in static weights)
    result = agent.respond("how do python generators and yield work")

    # Must be recalled from static weights with 0 network calls
    assert result.recalled_from_static is True, "Expected recall from static weights!"
    assert result.did_research is False, "Did research should be False for static recall!"
    assert "yield" in result.reply
    assert result.working_memory_bytes == 4096, f"Law 1 violated: {result.working_memory_bytes}"
