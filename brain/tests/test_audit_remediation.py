"""Comprehensive unit and contract verification for audit remediations.

Covers:
1. Recurrent state persistence (step_count, P_t, working_salience) across 100+ steps.
2. Truthful trace generation: rejection of truncated generations, zero fabricated EOS,
   and exact emitted token preservation.
3. Observation token limit enforcement before state ingestion.
4. Fail-fast checkpoint loading with FileNotFoundError on missing files.
5. Strict metric boundedness in [0.0, 1.0] for target token accuracy.
6. Mathematical parity between forward_sequence_sequential and sequential step() unrolling.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from irene_brain.agent.procedural_evaluator import compute_span_similarities
from irene_brain.agent.recurrent_software_agent import GenerationResult, RecurrentSoftwareAgent
from irene_brain.agent.software_environment import EnvironmentObservation, NeuralSoftwareEnvironment
from irene_brain.unified.unified_model import UnifiedPseudoBrain, make_unified_model


def test_state_persistence_across_steps():
    """Verify that step_count, P_t, active_thread, and working_salience persist across 100 steps."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    model.eval()

    state = model.init_state(batch_size=1, device=torch.device("cpu"))
    assert int(state.hierarchical_state.step_count.item()) == 0
    assert state.hierarchical_state.working_salience is not None

    p_t_dummy = torch.randn(1, 16, 64)
    state.P_t = p_t_dummy
    state.hierarchical_state.P_t = p_t_dummy

    for step_idx in range(100):
        token = torch.tensor([42], dtype=torch.long)
        sensory = model.encode_sensory(token_ids=token)
        outputs, state = model.step(sensory, state, token_id=token, allow_routing=False)
        assert int(state.hierarchical_state.step_count.item()) == step_idx + 1
        assert state.P_t is not None
        assert state.hierarchical_state.P_t is not None
        assert state.hierarchical_state.working_salience is not None


def test_fail_fast_checkpoint_loading():
    """Verify that RecurrentSoftwareAgent raises FileNotFoundError on missing checkpoint."""
    with pytest.raises(FileNotFoundError, match="Checkpoint file does not exist"):
        RecurrentSoftwareAgent(checkpoint_path="nonexistent_champion_weights.pt")


def test_metric_boundedness_and_exact_match():
    """Verify that compute_span_similarities never exceeds 1.0 on pathological inputs."""
    # Pathological repetition test from Audit Finding #35
    tok_acc, char_sim = compute_span_similarities("a_a_a_a_a_a_a.py", "a_b.py")
    assert 0.0 <= tok_acc <= 1.0
    assert 0.0 <= char_sim <= 1.0

    # Exact match test
    tok_acc_exact, char_sim_exact = compute_span_similarities("data_loader.py", "data_loader.py")
    assert tok_acc_exact == 1.0
    assert char_sim_exact == 1.0

    # Completely disjoint test
    tok_acc_zero, _ = compute_span_similarities("foo", "bar")
    assert tok_acc_zero == 0.0


def test_generation_result_string_compatibility():
    """Verify GenerationResult behaves as str while preserving metadata."""
    res = GenerationResult("ACTION: WRITE_FILE mod.py", [10, 20, 30], "eos")
    assert isinstance(res, str)
    assert res == "ACTION: WRITE_FILE mod.py"
    assert res.startswith("ACTION: ")
    assert res.stop_reason == "eos"
    assert res.token_ids == [10, 20, 30]


def test_rejection_of_truncated_actions_without_fabricated_eos():
    """Verify that truncated actions are NOT executed and no EOS is fabricated."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    executed_actions = []

    class MockEnv:
        def execute_action(self, action):
            executed_actions.append(action)
            return EnvironmentObservation("WRITE_FILE", True, "ok")

    # Monkeypatch generate_action_autoregressive to simulate token-limit cutoff
    def mock_truncated_generation(*args, **kwargs):
        return GenerationResult("ACTION: WRITE_FILE incomplete_", [1, 2, 3], "token_limit")

    agent.generate_action_autoregressive = mock_truncated_generation

    res = agent.execute_episode("Task: Implement foo\nTarget: foo.py", MockEnv(), max_cycles=3)

    assert res.success is False
    assert res.stop_reason == "action_token_limit"
    assert len(executed_actions) == 0, "Truncated action must NOT be executed in environment!"
    assert len(res.trace) == 1
    assert res.trace[0]["executed"] is False
    assert res.trace[0]["generation_stop"] == "token_limit"
    assert res.trace[0]["token_ids"] == [1, 2, 3]


def test_observation_token_limit_enforcement():
    """Verify that observations exceeding max_observation_tokens trigger rejection before ingestion."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    class GiantObsEnv:
        def execute_action(self, action):
            # Return an observation with 500 words which exceeds a max limit of 10
            return EnvironmentObservation("READ_FILE", True, "word " * 500)

    # Valid initial action
    def mock_valid_generation(*args, **kwargs):
        return GenerationResult("ACTION: READ_FILE foo.py", [1, 2], "eos")

    agent.generate_action_autoregressive = mock_valid_generation

    res = agent.execute_episode(
        "Task: Read foo\nTarget: foo.py",
        GiantObsEnv(),
        max_cycles=3,
        max_observation_tokens=10,
    )

    assert res.success is False
    assert res.stop_reason == "observation_token_limit"


def test_forward_sequence_sequential_parity():
    """Verify mathematical parity between forward_sequence_sequential and step() unrolling."""
    torch.manual_seed(42)
    model = make_unified_model(tier="tier1_3m", vocab_size=500, use_routing=True)
    model.eval()

    tokens = torch.randint(1, 500, (1, 16))

    with torch.no_grad():
        # 1. Unroll via forward_sequence_sequential with allow_routing=True
        seq_out = model.forward_sequence_sequential(token_seq=tokens, allow_routing=True)

        # 2. Unroll manually step-by-step with allow_routing=True
        state = model.init_state(1, torch.device("cpu"))
        step_logits = []
        for t in range(tokens.shape[1]):
            tok_t = tokens[:, t]
            sensory = model.encode_sensory(token_ids=tok_t)
            outputs, state = model.step(sensory, state, allow_routing=True, token_id=tok_t)
            step_logits.append(outputs["logits"])
        manual_logits = torch.stack(step_logits, dim=1)

        diff = (seq_out["logits"] - manual_logits).abs().max().item()
        assert diff < 1e-6, f"Sequential unrolling discrepancy: {diff}"
