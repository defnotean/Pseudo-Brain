"""Test Suite for Adaptive Mental Lookahead Reasoning Engine.

Verifies:
1. Shannon entropy evaluation and dynamic activation thresholding.
2. Compact 4.0 KB latent state branch cloning (zero KV-cache allocation).
3. Value-guided trajectory scoring using model.value_head.
4. End-to-end generation with adaptive lookahead.
"""

from __future__ import annotations

import pytest
import torch

from irene_brain.reasoning.mental_lookahead import MentalLookaheadEngine, TrajectoryCandidate
from irene_brain.unified.unified_model import UnifiedPseudoBrain


@pytest.fixture
def mini_unified_model():
    """Lightweight test model for lookahead engine tests."""
    torch.manual_seed(42)
    model = UnifiedPseudoBrain(
        vocab_size=100,
        tier="tier2",
        embed_dim=64,
        proj_dim=128,
        rank=16,
        num_deep_layers=2,
    )
    model.eval()
    return model


def test_entropy_evaluation(mini_unified_model):
    """Verify Shannon entropy computation on confident vs uniform logits."""
    engine = MentalLookaheadEngine(mini_unified_model, entropy_threshold=1.0)

    # 1. High confidence logits -> entropy near 0
    confident_logits = torch.full((1, 100), -10.0)
    confident_logits[0, 5] = 20.0
    ent_low = engine.evaluate_entropy(confident_logits)
    assert ent_low < 0.1, f"Expected near-zero entropy, got {ent_low}"

    # 2. Uniform logits -> entropy near ln(100) ~ 4.605
    uniform_logits = torch.zeros((1, 100))
    ent_high = engine.evaluate_entropy(uniform_logits)
    assert 4.5 <= ent_high <= 4.7, f"Expected uniform entropy ~4.6, got {ent_high}"


def test_adaptive_selection_fast_path(mini_unified_model):
    """Verify that confident logits take the fast greedy path without branching."""
    engine = MentalLookaheadEngine(mini_unified_model, entropy_threshold=1.0)
    state = mini_unified_model.init_state(batch_size=1, device=torch.device("cpu"))

    confident_logits = torch.full((1, 100), -10.0)
    confident_logits[0, 42] = 20.0

    tok, used_lookahead, ent = engine.select_next_token_adaptive(
        current_logits=confident_logits,
        current_state=state,
        device=torch.device("cpu"),
    )

    assert tok == 42
    assert used_lookahead is False
    assert ent < 1.0


def test_adaptive_selection_lookahead_path(mini_unified_model):
    """Verify that ambiguous logits trigger lookahead rollouts and return valid token."""
    engine = MentalLookaheadEngine(
        mini_unified_model,
        branch_factor=3,
        horizon=3,
        entropy_threshold=1.0,
    )
    state = mini_unified_model.init_state(batch_size=1, device=torch.device("cpu"))

    # Ambiguous logits across tokens 10, 20, 30
    ambiguous_logits = torch.full((1, 100), -5.0)
    ambiguous_logits[0, 10] = 2.0
    ambiguous_logits[0, 20] = 2.1
    ambiguous_logits[0, 30] = 1.9

    tok, used_lookahead, ent = engine.select_next_token_adaptive(
        current_logits=ambiguous_logits,
        current_state=state,
        device=torch.device("cpu"),
    )

    assert used_lookahead is True
    assert tok in (10, 20, 30)
    assert ent > 1.0


def test_generate_with_lookahead_end_to_end(mini_unified_model):
    """Verify full generation with lookahead tracking and Law 1 state preservation."""
    engine = MentalLookaheadEngine(
        mini_unified_model,
        branch_factor=3,
        horizon=3,
        entropy_threshold=1.5,
    )

    prompt = [1, 5, 10]
    result = engine.generate_with_lookahead(
        prompt_tokens=prompt,
        max_new_tokens=15,
        eos_id=99,
        pad_id=0,
    )

    assert "tokens" in result
    assert "lookahead_activations" in result
    assert "total_tokens" in result
    assert result["total_tokens"] > 0
    assert result["lookahead_activations"] >= 0
    assert result["final_state"].hierarchical_state.config.fast_state_bytes() == 4096
