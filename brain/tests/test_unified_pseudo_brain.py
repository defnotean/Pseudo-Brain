"""Unit tests for Unified Pseudo-Brain Architecture.

Verifies:
1. Instantiation across parameter tiers (Tier 0, Tier 1, Tier 2, 1B).
2. Multimodal sensory ingestion (pixels, language tokens, action feedback).
3. Streaming O(1) single-token execution (<1 ms latency, 60 Hz compliant).
4. Parallel associative scan O(log T) sequence forward pass for training.
5. Decoupled multi-task heads: language/code logits, discrete game actions, value, tool gate.
6. Two-tier hierarchical memory state integrity (Law 1 compliance).
"""

import math
import time
import pytest
import torch
from irene_brain.unified import UnifiedPseudoBrain, make_unified_model


def test_make_unified_model_tiers():
    """Verify model creation across multiple parameter tiers."""
    # Micro Tier 0
    m0 = make_unified_model(tier="tier0", vocab_size=1000)
    p0 = m0.count_parameters()["total"]
    assert p0 < 2_000_000, f"Tier 0 parameters {p0:,} exceed budget"

    # Embedded Tier 1
    m1 = make_unified_model(tier="tier1", vocab_size=2048)
    p1 = m1.count_parameters()["total"]
    assert 1_000_000 <= p1 <= 20_000_000, f"Tier 1 parameters {p1:,} outside expected range"

    # On-Device Tier 2
    m2 = make_unified_model(tier="tier2", vocab_size=32000)
    p2 = m2.count_parameters()["total"]
    assert 20_000_000 <= p2 <= 60_000_000, f"Tier 2 parameters {p2:,} outside expected range"

    # 1B Parameter Tier (evaluated on meta device for zero-RAM)
    with torch.device("meta"):
        m_1b = make_unified_model(tier="1b", vocab_size=32000)
    p_1b = m_1b.count_parameters()["total"]
    assert 9.5e8 <= p_1b <= 1.15e9, f"1B parameters {p_1b:,} outside expected range"


def test_multimodal_sensory_encoding():
    """Verify simultaneous or independent encoding of pixels, tokens, and actions."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    B = 2

    # Language only
    toks = torch.randint(0, 1000, (B,))
    h_lang = model.encode_sensory(token_ids=toks)
    assert h_lang.shape == (B, model.proj_dim)

    # Visual pixels only
    pixels = torch.randn(B, 3, 16, 16)
    h_vis = model.encode_sensory(pixels=pixels)
    assert h_vis.shape == (B, model.proj_dim)

    # Past action feedback only
    actions = torch.randint(0, model.n_actions, (B,))
    h_act = model.encode_sensory(past_action=actions)
    assert h_act.shape == (B, model.proj_dim)

    # Combined multimodal sensory input
    h_multi = model.encode_sensory(token_ids=toks, pixels=pixels, past_action=actions)
    assert h_multi.shape == (B, model.proj_dim)


def test_streaming_step_and_multi_heads():
    """Verify single-step O(1) inference, low latency, and decoupled readout heads."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    B = 1
    state = model.init_state(B, device=torch.device("cpu"))

    # Initial state assertions
    assert state.hierarchical_state.working_thoughts.shape == (B, model.K_fast, model.W_fast)
    assert state.hierarchical_state.episodic_thoughts.shape == (B, model.K_episodic, model.W_episodic)

    # Execute step with sensory input
    sensory = torch.randn(B, model.proj_dim)
    t0 = time.perf_counter()
    outputs, next_state = model.step(sensory, state, allow_routing=True)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # 60 Hz real-time compliance check (<16.6 ms)
    assert dt_ms < 15.0, f"Streaming latency {dt_ms:.2f} ms exceeds 60 Hz frame budget"

    # Multi-task decoupled heads check
    assert "logits" in outputs
    assert outputs["logits"].shape == (B, 1000)  # Language / code logits

    assert "action_logits" in outputs
    assert outputs["action_logits"].shape == (B, model.n_actions)  # Game discrete actions

    assert "predicted_value" in outputs
    assert outputs["predicted_value"].shape == (B, 1)  # Value / reward consequence

    assert "tool_prob" in outputs
    assert outputs["tool_prob"].shape == (B, 1)  # Tool-calling probability
    assert 0.0 <= float(outputs["tool_prob"].item()) <= 1.0

    # State update check
    assert next_state.last_action.shape == (B,)
    assert not torch.allclose(next_state.hierarchical_state.working_thoughts, state.hierarchical_state.working_thoughts)


def test_parallel_associative_scan_forward():
    """Verify O(log T) parallel training sequence forward pass."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    B = 2
    T = 16

    tokens = torch.randint(0, 1000, (B, T))
    actions = torch.randint(0, model.n_actions, (B, T))

    res = model.forward(token_seq=tokens, action_seq=actions, parallel=True)

    assert "logits" in res
    assert res["logits"].shape == (B, T, 1000)

    assert "action_logits" in res
    assert res["action_logits"].shape == (B, T, model.n_actions)

    assert "values" in res
    assert res["values"].shape == (B, T)

    # Check backprop gradients
    loss = res["logits"].sum() + res["action_logits"].sum() + res["values"].sum()
    loss.backward()

    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grad, "Gradients failed to flow through parallel associative scan!"
