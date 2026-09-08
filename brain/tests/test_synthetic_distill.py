"""Tests for High-Density Synthetic Reasoning Distillation Generator."""

import pytest
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from irene_brain.data.synthetic_distill import SyntheticDistillationEngine


def test_distillation_math_formatting():
    engine = SyntheticDistillationEngine(seed=101)
    for _ in range(50):
        item = engine.sample_math()
        assert "[RESP]" in item
        assert "[EOS]" in item
        resp_idx = item.index("[RESP]")
        assert "<thought>" in item
        assert "</thought>" in item
        assert "<solution>" in item
        assert "</solution>" in item
        # Ensure <thought> is strictly after [RESP]
        assert item.index("<thought>") > resp_idx
        assert item.index("<solution>") > item.index("</thought>")


def test_distillation_code_assertions():
    engine = SyntheticDistillationEngine(seed=202)
    for _ in range(30):
        item = engine.sample_code()
        assert "[RESP]" in item
        assert "def " in item
        assert "[EOS]" in item


def test_distillation_geography_coverage():
    engine = SyntheticDistillationEngine(seed=303)
    capitals_seen = set()
    for _ in range(100):
        item = engine.sample_geography()
        assert "capital" in item.lower()
        for country, capital in engine.geography:
            if capital in item:
                capitals_seen.add(capital)
    assert len(capitals_seen) > 10


def test_distillation_science_coverage():
    engine = SyntheticDistillationEngine(seed=404)
    for _ in range(20):
        item = engine.sample_science()
        assert "[RESP]" in item
        assert len(item) > 80
