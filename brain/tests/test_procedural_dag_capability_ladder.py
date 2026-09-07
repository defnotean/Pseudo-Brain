"""Regression and CI Test Suite for Procedural Task DAG Capability Ladder (WS3 / WS6).

Verifies:
1. All 8 capability levels (L1 - L8) construct valid toolsets and verification targets.
2. Blind anti-perseveration heuristic permanently bans failed actions, causing failure on L4.
3. PseudoBrainAgent with contextual IOR adapts to state changes, lifts inhibition, and solves L4.
4. PseudoBrainAgent recovers from transient stochastic timeouts (L7).
5. Shuffle invariance: Agent operates without relying on fixed tool indexing or layout bias.
"""

from __future__ import annotations

import random
import pytest
import torch

from irene_brain.agent.tools import ToolRegistry
from irene_brain.agent.loop import PseudoBrainAgent
from irene_brain.agent.goal import GoalSpecification
from agent_benchmarks.procedural_task_dag_benchmark import (
    SimulatedSystemState,
    build_ladder_level,
    evaluate_blind_suppression_agent,
    evaluate_pseudobrain_agent,
    evaluate_uninhibited_agent,
)


def test_capability_ladder_construction():
    """Verify all 8 ladder levels construct cleanly with consistent interfaces."""
    sys_state = SimulatedSystemState(seed=42)
    for lvl in range(1, 9):
        sys_state.reset(seed=42 + lvl)
        tools, check_fn, desc = build_ladder_level(lvl, sys_state)
        assert len(tools) >= 2, f"Level {lvl} has fewer than 2 tools"
        assert callable(check_fn), f"Level {lvl} check_fn is not callable"
        assert isinstance(desc, str) and len(desc) > 0, f"Level {lvl} description missing"


def test_blind_suppression_fails_on_state_contingent_retry():
    """Verify blind anti-perseveration permanently bans execute_job on L4, achieving 0%."""
    sys_state = SimulatedSystemState(seed=42)
    tools, _, _ = build_ladder_level(4, sys_state)
    # Ensure execute_job is attempted first to trigger initial failure and ban
    tool_names = [t.name for t in tools]
    exec_idx = tool_names.index("execute_job")
    tools[0], tools[exec_idx] = tools[exec_idx], tools[0]
    success, steps, _ = evaluate_blind_suppression_agent(tools, max_steps=12)
    assert not success, "Blind suppression should fail on L4 because execute_job is banned upon initial failure"


def test_pseudobrain_succeeds_on_state_contingent_retry():
    """Verify PseudoBrainAgent uses contextual IOR to solve L4 without manual tool bias."""
    sys_state = SimulatedSystemState(seed=42)
    tools, check_fn, desc = build_ladder_level(4, sys_state)
    rng = random.Random(42)
    shuffled_tools = list(tools)
    rng.shuffle(shuffled_tools)
    success, steps, _ = evaluate_pseudobrain_agent(
        shuffled_tools, check_fn, sys_state, desc, level=4, max_steps=15
    )
    assert success, "PseudoBrainAgent should succeed on L4 via state-contingent IOR reset"
    assert "build" in sys_state.state_vars.get("jobs_done", set())
    assert sys_state.state_vars.get("subsystem") == "operational"


def test_uninhibited_agent_perseveration_on_l4():
    """Verify uninhibited agent (scale=0) fails or struggles on L4 due to perseveration."""
    failures = 0
    seeds = [42, 43, 44, 45, 46]
    for s in seeds:
        sys_state = SimulatedSystemState(seed=s)
        tools, check_fn, desc = build_ladder_level(4, sys_state)
        rng = random.Random(s)
        shuffled_tools = list(tools)
        rng.shuffle(shuffled_tools)
        success, _, _ = evaluate_uninhibited_agent(
            shuffled_tools, check_fn, sys_state, desc, level=4, max_steps=15
        )
        if not success:
            failures += 1
    assert failures > 0, "Uninhibited agent should exhibit perseverative failure on L4"


def test_pseudobrain_transient_fault_recovery():
    """Verify PseudoBrainAgent recovers from transient timeout on L7."""
    sys_state = SimulatedSystemState(seed=42)
    tools, check_fn, desc = build_ladder_level(7, sys_state)
    rng = random.Random(42)
    shuffled_tools = list(tools)
    rng.shuffle(shuffled_tools)
    success, steps, _ = evaluate_pseudobrain_agent(
        shuffled_tools, check_fn, sys_state, desc, level=7, max_steps=15
    )
    assert success, "PseudoBrainAgent should recover and succeed on L7"
    assert "stochastic_ok" in sys_state.state_vars.get("jobs_done", set())


def test_zero_tool_bias_invariant():
    """Verify PseudoBrainAgent has exactly zero manual logit bias in core policy head."""
    sys_state = SimulatedSystemState(seed=42)
    tools, _, _ = build_ladder_level(1, sys_state)
    reg = ToolRegistry(tools)
    agent = PseudoBrainAgent(registry=reg, device_str="cpu")
    # All biases in policy_head must be exactly 0 (initial unprimed state)
    assert torch.all(agent.core.policy_head.bias == 0.0), "Policy head must not have manual tool biases"
