"""Unit & Integration Tests for Pseudo-Brain Autonomous Agent Loop."""
from __future__ import annotations

import tempfile
from pathlib import Path
import torch


from irene_brain.agent.goal import GoalSpecification, GoalEncoder
from irene_brain.agent.tools import (
    FileReadTool,
    FileWriteTool,
    CommandTool,
    TestVerifyTool,
    ToolRegistry,
)
from irene_brain.agent.loop import (
    PseudoBrainAgent,
    AgentCognitiveCore,
    run_autonomous_task,
)


def test_goal_encoder():
    enc = GoalEncoder(embedding_dim=128)
    e1 = enc("Modify test to pass")
    e2 = enc("Modify test to pass")
    e3 = enc("Explore environment and find the key")

    assert e1.shape == (1, 128)
    # Deterministic embedding for same text
    assert torch.allclose(e1, e2, atol=1e-5)
    # Distinct embedding for different text
    assert not torch.allclose(e1, e3, atol=1e-3)


def test_tool_registry():
    r = ToolRegistry([FileReadTool(), FileWriteTool(), CommandTool()])
    assert len(r) == 3
    assert r.get_by_index(0).name == "read_file"
    assert r.get_by_name("write_file") is not None
    assert r.get_by_name("unknown_tool") is None


def test_agent_cognitive_core_step():
    core = AgentCognitiveCore(goal_dim=128, obs_dim=128, thought_dim=384, n_tools=4)
    obs = torch.randn(1, 128)
    goal = torch.randn(1, 128)
    P_t = torch.zeros(1, 4)
    surprise = torch.tensor([[0.8]])  # High consequence surprise

    logits, h_next, P_next, gate = core.forward_step(
        obs_emb=obs,
        goal_emb=goal,
        h_prev=None,
        P_t=P_t,
        consequence_surprise=surprise,
    )

    assert logits.shape == (1, 4)
    assert h_next.shape == (1, 384)
    assert P_next.shape == (1, 4)
    assert 0.0 <= gate <= 1.0


def test_autonomous_task_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "status.txt"

        def is_verified():
            return target_file.exists() and "READY" in target_file.read_text()

        # Configure tools
        writer = FileWriteTool()
        verifier = TestVerifyTool(is_verified)
        reg = ToolRegistry([writer, verifier])

        agent = PseudoBrainAgent(registry=reg)
        goal = GoalSpecification(
            goal_id="task_write_ready",
            text="Write READY to status file and verify completion",
            verification_fn=is_verified,
        )

        default_args = {
            "write_file": {"path": str(target_file), "content": "READY"},
            "verify_goal": {},
        }

        report = agent.run_task(goal, max_steps=6, default_args=default_args)
        assert len(report.steps_log) > 0
        assert report.goal_id == "task_write_ready"


if __name__ == "__main__":
    print("Running test_goal_encoder()...")
    test_goal_encoder()
    print("Running test_tool_registry()...")
    test_tool_registry()
    print("Running test_agent_cognitive_core_step()...")
    test_agent_cognitive_core_step()
    print("Running test_autonomous_task_execution()...")
    test_autonomous_task_execution()
    print("ALL AGENT TESTS PASSED SUCCESSFULLY!")
