"""Unit tests for NeuralSemanticRouter and Acronym Auto-Indexing."""

import tempfile
from pathlib import Path
import pytest
import torch

from irene_brain.unified.unified_model import UnifiedPseudoBrain
from irene_brain.agent.neural_router import NeuralSemanticRouter
from irene_brain.agent.continual_learner import AutonomousLifelongAgent


@pytest.fixture
def test_model():
    return UnifiedPseudoBrain(vocab_size=2048, proj_dim=4096, rank=32, num_deep_layers=2)


def test_neural_router_saliency(test_model):
    """Test that NeuralSemanticRouter extracts meaningful semantic topic without regex."""
    router = NeuralSemanticRouter(test_model, vocab_size=2048)

    prompt = "then, how about what is the fortnite save the world"
    topic = router.extract_semantic_topic(prompt)

    assert "fortnite" in topic.lower()
    assert "save" in topic.lower()
    assert "world" in topic.lower()

    tokens = [w.lower() for w in topic.split()]
    assert "then" not in tokens
    assert "how" not in tokens
    assert "what" not in tokens
    assert "is" not in tokens


def test_neural_router_clean_topic(test_model):
    """Test extraction on space telescope query."""
    router = NeuralSemanticRouter(test_model, vocab_size=2048)
    prompt = "so what can you tell me about the james webb space telescope"
    topic = router.extract_semantic_topic(prompt)

    assert "james" in topic.lower()
    assert "webb" in topic.lower()
    assert "telescope" in topic.lower()
    assert "tell" not in [w.lower() for w in topic.split()]


def test_acronym_auto_indexing_and_recall(test_model):
    """Test that consolidating a topic auto-indexes acronyms and recalls instantly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "test_state.pt"
        agent = AutonomousLifelongAgent(model=test_model, state_save_path=str(state_path))

        agent._consolidate_to_episodic(
            topic="James Webb Space Telescope",
            summary=(
                "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. "
                "It was launched on 25 December 2021 by NASA, ESA, and CSA."
            ),
            code_example=None,
            language="general",
            canonical_topic="james_webb_space_telescope",
        )

        lesson = agent.episodic_lessons["james_webb_space_telescope"]
        assert "jwst" in lesson["aliases"], f"JWST acronym should be auto-indexed in aliases: {lesson['aliases']}"

        res = agent.respond("Can you explain JWST?")
        assert res.recalled_from_episodic is True
        assert res.did_research is False
        assert "infrared astronomy" in res.reply.lower()
        assert res.elapsed_ms < 50.0
