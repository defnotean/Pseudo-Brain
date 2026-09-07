"""Priority 4 (P4): Five-Case Inhibition of Return (IOR) Causal Validation Suite.

Verifies:
- Case A: Dead-end action failure -> action must be suppressed.
- Case B: Action failure followed by state repair -> action must be retried and succeed.
- Case C: Same tool called with different/corrected arguments -> must not be suppressed.
- Case D: Same tool + same arguments called after environmental change -> suppression must decay.
- Case E: Repeated persistent failure -> exploratory strategy shift via entropy expansion.
"""

from __future__ import annotations

import math
import unittest
import torch
import numpy as np

from irene_brain.agent.loop import AgentCognitiveCore, PseudoBrainAgent
from irene_brain.agent.tools import Tool, ToolRegistry, ToolResult
from irene_brain.agent.goal import GoalSpecification


class MockDummyTool(Tool):
    def __init__(self, name: str, success: bool = True, reward: float = 1.0, output: str = "ok"):
        self.name = name
        self.description = f"Mock tool {name}"
        self.should_succeed = success
        self.reward_val = reward
        self.output_val = output
        self.call_count = 0
        self.last_args: dict = {}

    def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        self.last_args = kwargs
        if self.should_succeed:
            return ToolResult(success=True, output=self.output_val, reward=self.reward_val)
        return ToolResult(success=False, output="", error=self.output_val, reward=-1.0)


class IORCausalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.registry = ToolRegistry()
        self.tool0 = MockDummyTool("tool_primary", success=False, output="Error: missing precondition")
        self.tool1 = MockDummyTool("tool_repair", success=True, output="Repaired precondition")
        self.tool2 = MockDummyTool("tool_alternative", success=True, output="Alternative execution")
        self.registry.register(self.tool0)
        self.registry.register(self.tool1)
        self.registry.register(self.tool2)

        self.core = AgentCognitiveCore(
            goal_dim=32,
            obs_dim=32,
            thought_dim=64,
            n_tools=3,
        )

    def test_case_a_dead_end_action_failure_suppressed(self) -> None:
        """Case A: Action failure without state repair suppresses the failing action."""
        obs = torch.zeros(1, 32)
        goal = torch.zeros(1, 32)
        p_t = torch.zeros(1, 3)
        surprise = torch.tensor([[1.0]])

        # Baseline logits before failure
        base_logits, h0, _, _ = self.core.forward_step(obs, goal, None, p_t, torch.zeros(1, 1))

        # Step with failure on action 0
        failed_logits, h1, p_next, _ = self.core.forward_step(
            obs, goal, h0, p_t, surprise,
            last_action=0, last_success=False, state_novelty=0.0
        )

        # Action 0 logit must drop significantly relative to baseline
        delta_logit_0 = failed_logits[0, 0].item() - base_logits[0, 0].item()
        self.assertLess(delta_logit_0, -2.0, f"Action 0 was not strongly suppressed: {delta_logit_0}")

        # Action 0 must not be chosen over alternative actions
        self.assertNotEqual(int(failed_logits.argmax().item()), 0)

    def test_case_b_action_failure_followed_by_state_repair_retried_and_succeeds(self) -> None:
        """Case B: Action failure followed by successful state repair allows retrying the action."""
        agent = PseudoBrainAgent(self.registry, goal_dim=32, obs_dim=32, thought_dim=64)
        agent.core.set_tool_bias(0, 1.5)  # Prioritize tool_primary initially

        goal = GoalSpecification(
            goal_id="test_repair",
            text="Execute tool_primary after repair",
            verification_fn=lambda: self.tool0.call_count >= 2 and self.tool0.should_succeed,
        )

        # Simulation:
        # Step 1: tool0 fails (missing precondition)
        # Step 2: tool1 runs (repair action) and succeeds, fixing the precondition
        # Step 3: tool0 is now repaired and will succeed!
        step_tracker = {"step": 0}

        def custom_arg_provider(step, tool_name, logs, g):
            step_tracker["step"] = step
            if step == 2:
                # The repair tool executes, and makes tool0 succeed on next try
                self.tool0.should_succeed = True
                self.tool0.output_val = "repaired"
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=5,
            arg_provider=custom_arg_provider,
        )

        executed_tools = [log.tool_name for log in report.steps_log]
        self.assertGreaterEqual(len(executed_tools), 3)

        # Step 1 must have run tool_primary and failed
        self.assertEqual(executed_tools[0], "tool_primary")
        self.assertFalse(report.steps_log[0].success)

        # Step 2 must NOT repeat tool_primary (IOR active)
        self.assertNotEqual(executed_tools[1], "tool_primary")

        # Step 3: after repair, tool_primary can be retried and succeeds!
        self.assertIn("tool_primary", executed_tools[2:], "tool_primary must be retried after repair")
        final_tool0_log = [l for l in report.steps_log if l.tool_name == "tool_primary"][-1]
        self.assertTrue(final_tool0_log.success, "Retried tool_primary must succeed")

    def test_case_c_different_arguments_not_suppressed(self) -> None:
        """Case C: Specific suppression targeting signature does not suppress tool with different arguments."""
        obs = torch.zeros(1, 32)
        goal = torch.zeros(1, 32)
        p_t = torch.zeros(1, 3)
        surprise = torch.tensor([[1.0]])

        # Suppress specifically action 0 with signature A
        specific_supp = {0: -8.0}
        logits_suppressed, _, _, _ = self.core.forward_step(
            obs, goal, None, p_t, surprise,
            specific_suppression=specific_supp
        )
        self.assertLess(logits_suppressed[0, 0].item(), -5.0)

        # Calling with corrected arguments (no specific suppression on action 0)
        logits_unsuppressed, _, _, _ = self.core.forward_step(
            obs, goal, None, p_t, surprise,
            specific_suppression={}  # signature differs
        )
        self.assertGreater(logits_unsuppressed[0, 0].item(), logits_suppressed[0, 0].item() + 5.0)

    def test_case_d_suppression_decays_with_state_novelty(self) -> None:
        """Case D: Suppression decays exponentially when environment state novelty is observed."""
        obs = torch.zeros(1, 32)
        goal = torch.zeros(1, 32)
        p_t = torch.zeros(1, 3)
        surprise = torch.tensor([[1.0]])

        # 1. State novelty = 0.0 (static, no environmental change)
        logits_static, _, _, _ = self.core.forward_step(
            obs, goal, None, p_t, surprise,
            last_action=0, last_success=False, state_novelty=0.0
        )

        # 2. State novelty = 2.0 (significant environmental change)
        logits_novel, _, _, _ = self.core.forward_step(
            obs, goal, None, p_t, surprise,
            last_action=0, last_success=False, state_novelty=2.0
        )

        # The suppression on action 0 must be much weaker under state novelty
        supp_static = -logits_static[0, 0].item()
        supp_novel = -logits_novel[0, 0].item()
        self.assertLess(supp_novel, supp_static, f"Suppression under novelty ({supp_novel}) should be less than static ({supp_static})")

    def test_case_e_repeated_failures_exploratory_entropy_expansion(self) -> None:
        """Case E: Consecutive persistent failures increase policy entropy, encouraging exploration."""
        obs = torch.zeros(1, 32)
        goal = torch.zeros(1, 32)
        p_t = torch.zeros(1, 3)
        surprise = torch.tensor([[1.0]])

        # 0 failures
        logits_0, _, _, _ = self.core.forward_step(
            obs, goal, None, p_t, surprise, consecutive_failures=0
        )
        probs_0 = torch.softmax(logits_0, dim=-1)
        entropy_0 = -(probs_0 * torch.log(probs_0 + 1e-8)).sum().item()

        # 4 consecutive failures
        logits_4, _, _, _ = self.core.forward_step(
            obs, goal, None, p_t, surprise, consecutive_failures=4
        )
        probs_4 = torch.softmax(logits_4, dim=-1)
        entropy_4 = -(probs_4 * torch.log(probs_4 + 1e-8)).sum().item()

        self.assertGreater(
            entropy_4,
            entropy_0,
            f"Entropy under repeated failures ({entropy_4:.4f}) must be higher than baseline ({entropy_0:.4f})",
        )


if __name__ == "__main__":
    unittest.main()
